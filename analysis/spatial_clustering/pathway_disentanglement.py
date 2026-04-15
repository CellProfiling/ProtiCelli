#!/usr/bin/env python3
"""
Reactome pathway dotplot.

Fetches sub-pathways of one or more Reactome pathway IDs up to a given depth,
extracts their gene members via the Reactome ContentService API, and produces a
dotplot/heatmap from pathway-level fold-change values per cluster.

Dot colour  = pathway-level log2 fold-change summary (default: RdBu_r)
Dot size    = -log10(BH-adjusted p-value)

Author: Konstantin Kahnert
"""
import argparse
from collections import deque
import json
import logging
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy.stats import mannwhitneyu, wilcoxon

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — colour palettes, organelle labels, cluster display mappings
# ---------------------------------------------------------------------------

REACTOME_BASE = "https://reactome.org/ContentService"

CLUSTER_COLORS: dict[int, tuple[float, float, float]] = {
    1:  (255/255,   0/255,   0/255),   # Red
    2:  (  0/255,   0/255, 255/255),   # Blue
    3:  (  0/255, 128/255,   0/255),   # Green
    4:  (255/255, 255/255,   0/255),   # Yellow
    5:  (255/255,   0/255, 255/255),   # Magenta
    6:  (  0/255, 255/255, 255/255),   # Cyan
    7:  (255/255, 165/255,   0/255),   # Orange
    8:  (128/255,   0/255, 128/255),   # Purple
    9:  (  0/255, 128/255, 128/255),   # Teal
    10: (135/255, 206/255, 235/255),   # Sky Blue
    11: (128/255,   0/255,   0/255),   # Maroon
}

# ColorBrewer Set1 palette (Paul's color scheme for pathway_marker_clustering)
CLUSTER_COLORS_PAUL: dict[int, tuple[float, float, float]] = {
    1:  (228/255,  26/255,  28/255),   # red
    2:  ( 55/255, 126/255, 184/255),   # blue
    3:  ( 77/255, 175/255,  74/255),   # green
    4:  (152/255,  78/255, 163/255),   # purple
    5:  (255/255, 127/255,   0/255),   # orange
    6:  (255/255, 255/255,  51/255),   # yellow
    7:  (247/255, 129/255, 191/255),   # pink
}

COLOR_SCHEMES: dict[str, dict[int, tuple[float, float, float]]] = {
    "default": CLUSTER_COLORS,
    "paul":    CLUSTER_COLORS_PAUL,
}

# Organelle name labels for organelle_marker_clustering (keyed by cluster number).
# Colors match CLUSTER_COLORS 1:1.  Clusters 5 and 7 have no organelle assignment.
ORGANELLE_LABELS: dict[int, str] = {
    1:  "Extracellular Matrix",
    2:  "Cell Membrane",
    3:  "Cytosol",
    4:  "ER",
    5:  "n/a",
    6:  "Nucleoplasm",
    7:  "n/a",
    8:  "Mitochondria",
    9:  "Nuclear Speckles",
    10: "Nuclear Membrane",
    11: "Nucleoli",
}

PATHWAY_SCHEMA_CLASSES = {"Pathway", "TopLevelPathway"}

_COLOR_NAMES = ["red", "blue", "green", "yellow", "magenta", "cyan",
                "orange", "purple", "teal", "sky blue", "maroon"]
COLOR_NAME_TO_RGB = {name: CLUSTER_COLORS[i + 1] for i, name in enumerate(_COLOR_NAMES)}
COLOR_NAME_TO_RGB["black"] = (0.0, 0.0, 0.0)

# Maps display position index → original (pre-rename) cluster number, for box coloring.
# Mirrors run_gsea_pipeline.py / replot_annotated_clusters.py.
CLUSTER_DISPLAY_COLORS: dict[str, list[int]] = {
    "MCF-7_transcription":          [1, 2, 6, 4, 7, 5, 3],
    "U2OS_rtk_signaling":           [1, 2, 3, 5, 7, 4, 6],
}

# ---------------------------------------------------------------------------
# Annotation CSV loader
# ---------------------------------------------------------------------------

def load_annotation_csv(path: Path) -> dict[int, dict]:
    """Load cluster_annotations.csv and return a per-cluster annotation map.

    Returns {cluster_num: {"color": (r,g,b), "label": str}}.
    Cluster 0 (Background) is skipped.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    result: dict[int, dict] = {}
    for _, row in df.iterrows():
        cl = int(row["Cluster"])
        if cl == 0:
            continue
        color_name = str(row.get("Color", "")).strip().lower()
        color = COLOR_NAME_TO_RGB.get(color_name, (0.5, 0.5, 0.5))
        annotation = str(row.get("Annotation", "")).strip()
        secondary = str(row.get("Secondary_Annotation", "")).strip()
        if secondary and secondary.lower() not in ("", "nan"):
            label = f"{annotation}, {secondary}"
        else:
            label = annotation
        result[cl] = {"color": color, "label": label}
    return result


# ---------------------------------------------------------------------------
# API helpers with caching
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with cache_path.open() as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    with cache_path.open("w") as f:
        json.dump(cache, f)


def _api_get(url: str, cache: dict, cache_path: Path, retries: int = 3) -> list | dict | None:
    if url in cache:
        return cache[url]
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                cache[url] = None
                _save_cache(cache, cache_path)
                return None
            resp.raise_for_status()
            data = resp.json()
            cache[url] = data
            _save_cache(cache, cache_path)
            return data
        except Exception as exc:
            if attempt == retries - 1:
                logger.warning("Failed to fetch %s: %s", url, exc)
                return None
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Reactome traversal
# ---------------------------------------------------------------------------

def get_direct_children(stid: str, cache: dict, cache_path: Path) -> list[dict]:
    """Return direct children of *stid* (both Pathway and Reaction types) via hasEvent."""
    url = f"{REACTOME_BASE}/data/query/{stid}"
    data = _api_get(url, cache, cache_path)
    if not data or not isinstance(data, dict):
        return []
    has_event = data.get("hasEvent", [])
    # hasEvent may contain dicts or raw integer dbIds; keep only full objects
    return [e for e in has_event if isinstance(e, dict)]


def get_direct_child_pathways(stid: str, cache: dict, cache_path: Path) -> list[dict]:
    """Return direct sub-pathways (Pathway/TopLevelPathway) of *stid*."""
    children = get_direct_children(stid, cache, cache_path)
    return [e for e in children if e.get("schemaClass") in PATHWAY_SCHEMA_CLASSES]


def resolve_pathway_id(stid: str, cache: dict, cache_path: Path) -> list[str]:
    """
    Resolve *stid* to a list of Pathway-class stIds.

    If *stid* is already a Pathway/TopLevelPathway, return [stid].
    If it is a Reaction or other event, find enclosing pathways via
    /data/pathways/low/entity/{dbId} and return those.
    """
    url = f"{REACTOME_BASE}/data/query/{stid}"
    data = _api_get(url, cache, cache_path)
    if not data:
        logger.warning("Could not resolve %s", stid)
        return [stid]

    schema = data.get("schemaClass", "")
    if schema in PATHWAY_SCHEMA_CLASSES:
        return [stid]

    # Non-pathway: find enclosing pathways
    db_id = data.get("dbId")
    display = data.get("displayName", stid)
    logger.info("%s is a %s (%s), finding enclosing pathways...", stid, schema, display)
    enc_url = f"{REACTOME_BASE}/data/pathways/low/entity/{db_id}?species=9606"
    enc_data = _api_get(enc_url, cache, cache_path)
    if not enc_data:
        logger.warning("No enclosing pathways found for %s", stid)
        return [stid]
    result = [e["stId"] for e in enc_data if e.get("schemaClass") in PATHWAY_SCHEMA_CLASSES]
    logger.info("Using enclosing pathway(s): %s", result)
    return result if result else [stid]


def get_pathway_genes(stid: str, cache: dict, cache_path: Path) -> list[str]:
    """Return unique gene symbols for all participants in *stid*."""
    url = f"{REACTOME_BASE}/data/participants/{stid}"
    data = _api_get(url, cache, cache_path)
    if not data:
        return []
    genes: set[str] = set()
    for item in data:
        for ref in item.get("refEntities", []):
            if ref.get("schemaClass") == "ReferenceGeneProduct":
                # displayName format: "UniProt:P07204 THBD"
                display = ref.get("displayName", "")
                parts = display.strip().split()
                if parts:
                    genes.add(parts[-1].upper())
    return sorted(genes)


def traverse_pathways(
    root_ids: list[str],
    depth: int,
    cache: dict,
    cache_path: Path,
) -> dict[str, dict]:
    """
    BFS traversal up to *depth* levels below each root.

    Returns a dict keyed by stId:
        {stId: {"name": str, "depth": int, "parent": str | None, "genes": list[str]}}

    Depth 1 = direct children of root; depth 2 = grandchildren, etc.
    The root itself is NOT included.
    """
    result: dict[str, dict] = {}

    # queue items: (stid, current_depth, parent_stid)
    queue: deque[tuple[str, int, str | None]] = deque((rid, 0, None) for rid in root_ids)
    visited: set[str] = set(root_ids)

    while queue:
        current_id, current_depth, _ = queue.popleft()
        if current_depth >= depth:
            continue

        children = get_direct_child_pathways(current_id, cache, cache_path)
        logger.info("%s%s: %d sub-pathways at depth %d",
                    "  " * current_depth, current_id, len(children), current_depth + 1)

        for child in children:
            child_stid = child.get("stId", "")
            if not child_stid or child_stid in visited:
                continue
            visited.add(child_stid)
            child_depth = current_depth + 1
            result[child_stid] = {
                "name": child.get("displayName", child_stid),
                "depth": child_depth,
                "parent": current_id if current_id not in root_ids else None,
                "genes": [],  # filled below
            }
            queue.append((child_stid, child_depth, current_id))

    # Fetch genes for every collected pathway
    logger.info("Fetching genes for %d pathways...", len(result))
    for stid, info in result.items():
        genes = get_pathway_genes(stid, cache, cache_path)
        info["genes"] = genes
        logger.info("  %s (%s): %d genes", stid, info["name"][:50], len(genes))

    return result


# ---------------------------------------------------------------------------
# Expression statistics
# ---------------------------------------------------------------------------

def compute_dotplot_data(
    pathways: dict[str, dict],
    expr_df: pd.DataFrame,
    cluster_cols: list[str],
    min_overlap: int,
    metric: str = "mean",
    is_foldchange: bool = False,
) -> pd.DataFrame:
    """
    For each (pathway, cluster) compute an expression summary, overlap size, and p-value.

    Parameters
    ----------
    metric : "mean" | "sum"
        Aggregation applied to gene values within each pathway/cluster.
    is_foldchange : bool
        If True, computes a one-sided one-sample Wilcoxon signed-rank test on log2(FC)
        (H0: median log2FC = 0, H1: median log2FC > 0).  If False, computes a one-sided
        Mann-Whitney U test comparing pathway gene intensities in this cluster vs. all
        other clusters pooled (1-vs-rest, H1: cluster > rest).

    Returns DataFrame with columns:
        stId, name, depth, parent, cluster, value, overlap_size, neg_log10_pval, pval
    """
    gene_upper = expr_df["Gene"].str.upper()
    expr_genes = set(gene_upper)
    aggregate = "sum" if metric == "sum" else "mean"
    rows = []
    for stid, info in pathways.items():
        pw_genes = [g for g in info["genes"] if g in expr_genes]
        if len(pw_genes) < min_overlap:
            continue
        sub = expr_df[gene_upper.isin(pw_genes)]
        for col in cluster_cols:
            if col not in sub.columns:
                continue
            val = float(sub[col].agg(aggregate))

            # --- statistical test ---
            vals = sub[col].dropna().values
            if is_foldchange:
                # One-sample one-sided Wilcoxon on log2FC values (already transformed by
                # main()): H0: median log2FC = 0, H1: median log2FC > 0 (upregulated)
                if len(vals) >= 4 and not np.all(vals == vals[0]):
                    _, pval = wilcoxon(vals, alternative="greater")
                else:
                    pval = 1.0
            else:
                # One-sided Mann-Whitney U, 1-vs-rest: H1 expression in this cluster
                # is greater than in all other clusters combined
                other_cols = [c for c in cluster_cols if c != col and c in sub.columns]
                group2 = sub[other_cols].values.ravel()
                group2 = group2[~np.isnan(group2)]
                if len(vals) >= 2 and len(group2) >= 2:
                    _, pval = mannwhitneyu(vals, group2, alternative="greater")
                else:
                    pval = 1.0

            neg_log10_pval = -np.log10(max(pval, 1e-300))

            rows.append({
                "stId": stid,
                "name": info["name"],
                "depth": info["depth"],
                "parent": info["parent"],
                "cluster": col,
                "value": val,
                "overlap_size": len(pw_genes),
                "pval": pval,
                "neg_log10_pval": neg_log10_pval,
            })
    return pd.DataFrame(rows)


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (same order as input)."""
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    adjusted = pvals[order] * n / np.arange(1, n + 1, dtype=float)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def _get_cluster_cols(expr_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return expression frame without background and sorted cluster columns."""
    cols = sorted(
        [c for c in expr_df.columns if c.lower().startswith("cluster")],
        key=_cluster_num,
    )
    expr_df = expr_df.drop(columns=["Cluster_0"], errors="ignore")
    return expr_df, [c for c in cols if c != "Cluster_0"]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _cluster_num(col: str) -> int:
    try:
        return int(col.split("_")[-1])
    except ValueError:
        return 0


def _build_row_order(dotplot_df: pd.DataFrame) -> tuple[list[str], list[str], list[int]]:
    """Return (stid_list, name_list, depth_list) in display order."""
    depth1 = dotplot_df[dotplot_df["depth"] == 1][["stId", "name", "parent"]].drop_duplicates()
    depth2 = dotplot_df[dotplot_df["depth"] == 2][["stId", "name", "parent"]].drop_duplicates()
    depth_other = dotplot_df[~dotplot_df["depth"].isin([1, 2])][["stId", "name", "depth", "parent"]].drop_duplicates()

    ordered_rows: list[dict] = []
    if not depth2.empty:
        parents_with_children = set(depth2["parent"].dropna())
        for _, p in depth1.sort_values("name").iterrows():
            ordered_rows.append({"stId": p["stId"], "name": p["name"], "depth": 1})
            if p["stId"] in parents_with_children:
                for _, c in depth2[depth2["parent"] == p["stId"]].sort_values("name").iterrows():
                    ordered_rows.append({"stId": c["stId"], "name": c["name"], "depth": 2})
        for _, c in depth2[~depth2["parent"].isin(depth1["stId"])].sort_values("name").iterrows():
            ordered_rows.append({"stId": c["stId"], "name": c["name"], "depth": 2})
    else:
        for _, p in depth1.sort_values("name").iterrows():
            ordered_rows.append({"stId": p["stId"], "name": p["name"], "depth": 1})
    for _, r in depth_other.sort_values("name").iterrows():
        ordered_rows.append({"stId": r["stId"], "name": r["name"], "depth": int(r["depth"])})

    stids  = [r["stId"]  for r in ordered_rows]
    names  = [r["name"]  for r in ordered_rows]
    depths = [r["depth"] for r in ordered_rows]
    return stids, names, depths


def _draw_cluster_axis(
    ax: plt.Axes,
    cluster_order: list[str],
    cluster_colors: dict[int, tuple[float, float, float]],
    cluster_color_nums: dict[str, int] | None = None,
    annotation_map: dict[int, dict] | None = None,
    organelle_labels: bool = False,
    tick_length: float = 3,
    label_fontsize: float = 10,
    footer_fontsize: float = 10,
) -> None:
    n_clusters = len(cluster_order)
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([""] * n_clusters)
    ax.tick_params(axis="x", length=tick_length, direction="out")
    ax.set_xlim(-0.5, n_clusters - 0.5)

    trans_x = ax.get_xaxis_transform()
    for i, col in enumerate(cluster_order):
        cl_num = _cluster_num(col)
        color_num = cluster_color_nums.get(col, cl_num) if cluster_color_nums else cl_num
        if annotation_map:
            color = annotation_map.get(cl_num, {}).get("color", (0.5, 0.5, 0.5))
            ax.plot(
                i, -0.025,
                marker="s", markersize=9,
                color=color, markeredgecolor="black", markeredgewidth=0.5,
                transform=trans_x, clip_on=False,
            )
            continue

        if organelle_labels:
            color = cluster_colors.get(color_num, (0.5, 0.5, 0.5))
            label = ORGANELLE_LABELS.get(cl_num, "n/a")
            ax.text(
                i, -0.01, label,
                transform=trans_x,
                ha="right", va="top", fontsize=9,
                rotation=35,
                color=color,
                fontweight="bold",
                clip_on=False,
            )
            continue

        color = cluster_colors.get(color_num, (0.5, 0.5, 0.5))
        ax.text(
            i, -0.01, str(cl_num),
            transform=trans_x,
            ha="center", va="top", fontsize=label_fontsize,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="none",
                edgecolor=color,
                linewidth=1.2,
            ),
            clip_on=False,
        )

    if not annotation_map:
        x_footer = "Organelle" if organelle_labels else "Cluster"
        ax.text(
            0.5,
            -0.35 if organelle_labels else -0.07,
            x_footer,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=footer_fontsize,
        )


def _draw_pathway_axis(
    ax: plt.Axes,
    row_names: list[str],
    row_depths: list[int],
    parent_width: int,
    child_width: int,
    parent_fontsize: float,
    child_fontsize: float,
) -> None:
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels([""] * len(row_names))
    ax.tick_params(axis="y", length=0)

    trans_y = ax.get_yaxis_transform()
    for i, (name, dep) in enumerate(zip(row_names, row_depths)):
        if dep == 1:
            label = textwrap.fill(name, width=parent_width)
            ax.text(
                -0.01, i, label,
                transform=trans_y,
                ha="right", va="center", fontsize=parent_fontsize,
                fontweight="bold", color=(0.1, 0.1, 0.1),
                clip_on=False,
            )
        else:
            label = textwrap.fill("· " + name, width=child_width)
            ax.text(
                -0.01, i, label,
                transform=trans_y,
                ha="right", va="center", fontsize=child_fontsize,
                color=(0.3, 0.3, 0.3),
                clip_on=False,
            )


def _cmap_and_norm(
    values: pd.Series,
    colormap: str,
) -> tuple:
    """Return (cmap, norm, is_diverging) given values and colormap name."""
    vmin, vmax = float(values.min()), float(values.max())
    cmap = cm.get_cmap(colormap)
    diverging_cmaps = {"RdBu", "RdBu_r", "bwr", "seismic", "coolwarm", "PiYG", "PRGn"}
    is_diverging = colormap in diverging_cmaps
    if is_diverging:
        # Always centre at 0 (values are expected to be log2 FC)
        bound = max(abs(vmax), abs(vmin))
        norm = plt.Normalize(vmin=-bound, vmax=bound)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm, is_diverging


IMG_W_IN = 4.0   # fixed width of the image panel when present


def _add_image_panel(
    fig: plt.Figure,
    img_path: Path,
    title: str,
    fig_w: float,
    fig_h: float,
) -> tuple[float, float, float]:
    """Draw a single cell image as the left panel (used by make_heatmap).

    Returns (img_frac_w, ax_bottom, ax_h_frac).
    """
    img = Image.open(img_path)
    img_w, img_h = img.size
    img_frac_w = IMG_W_IN / fig_w

    margin_lr   = 0.01
    content_top = 0.89
    margin_tb   = 0.05

    ax_w_frac = img_frac_w - margin_lr
    ax_h_in   = ax_w_frac * fig_w * (img_h / img_w)
    ax_h_frac = min(ax_h_in / fig_h, content_top - margin_tb)
    ax_bottom  = content_top - ax_h_frac

    fig.text(margin_lr + ax_w_frac * 0.5, 0.95, title,
             fontsize=11, fontweight="bold", va="top", ha="center")

    ax_img = fig.add_axes([margin_lr, ax_bottom, ax_w_frac, ax_h_frac])
    ax_img.imshow(np.array(img), interpolation="lanczos", aspect="auto")
    ax_img.axis("off")

    return img_frac_w, ax_bottom, ax_h_frac


def _add_image_panels(
    fig: plt.Figure,
    img_paths: list[Path],
    fig_w: float,
    fig_h: float,
    ax_bottom: float,
    ax_height: float,
) -> None:
    """Stack cell images vertically as squares in the left panel."""
    n = len(img_paths)
    if n == 0:
        return
    margin_l = 0.1 / fig_w
    col_w = IMG_W_IN - 0.1
    slot_h = ax_height * fig_h / n
    sq = min(col_w, slot_h)
    sq_w, sq_h = sq / fig_w, sq / fig_h
    slot_frac = ax_height / n
    left = margin_l + (IMG_W_IN / fig_w - margin_l - sq_w) / 2

    for idx, path in enumerate(img_paths):
        slot_bot = ax_bottom + (n - 1 - idx) * slot_frac
        if idx == 0:
            bot = slot_bot + slot_frac - sq_h
        elif idx == n - 1:
            bot = slot_bot
        else:
            bot = slot_bot + (slot_frac - sq_h) / 2
        ax = fig.add_axes([left, bot, sq_w, sq_h])
        ax.imshow(np.array(Image.open(path)), interpolation="lanczos")
        ax.axis("off")


def _compute_figure_layout(
    n_rows: int,
    n_clusters: int,
    *,
    label_w_in: float,
    col_w_in: float,
    row_h_in: float,
    extra_w_in: float,
    top_in: float,
    bot_in: float,
    has_images: bool,
) -> dict[str, float]:
    """Compute figure and axes dimensions shared by dotplot and heatmap."""
    plot_w_in = n_clusters * col_w_in
    data_w = label_w_in + plot_w_in + extra_w_in
    fig_w = (IMG_W_IN + data_w) if has_images else data_w
    img_frac_w = (IMG_W_IN / fig_w) if has_images else 0.0
    plot_h_in = max(2.0, n_rows * row_h_in)
    fig_h = plot_h_in + top_in + bot_in
    return {
        "fig_w": fig_w, "fig_h": fig_h,
        "plot_w_in": plot_w_in, "plot_h_in": plot_h_in,
        "img_frac_w": img_frac_w,
        "ax_left": img_frac_w + label_w_in / fig_w,
        "ax_width": plot_w_in / fig_w,
        "ax_bottom": bot_in / fig_h,
        "ax_height": plot_h_in / fig_h,
    }


def make_dotplot(
    dotplot_df: pd.DataFrame,
    cluster_cols: list[str],
    output_path: Path,
    title: str = "Reactome Sub-pathway Expression",
    colormap: str = "Reds",
    value_label: str = "Mean expression",
    cluster_color_nums: dict[str, int] | None = None,
    img_paths: list[Path] | None = None,
    is_foldchange: bool = False,
    cluster_colors: dict[int, tuple[float, float, float]] | None = None,
    organelle_labels: bool = False,
    annotation_map: dict[int, dict] | None = None,
) -> None:
    """Render a dotplot (pathways × clusters) to *output_path*.

    cluster_color_nums maps Cluster_N column name → original cluster number
    for the colored tick boxes (handles post-rename cell lines).
    img_paths: list of cell-image PNGs stacked top-to-bottom in the left panel.
    """
    if cluster_colors is None:
        cluster_colors = CLUSTER_COLORS

    if dotplot_df.empty:
        logger.warning("No data to plot.")
        return

    row_stids, row_names, row_depths = _build_row_order(dotplot_df)
    n_rows = len(row_stids)

    if n_rows == 0:
        logger.warning("No pathways with sufficient gene overlap.")
        return

    cluster_order = sorted(cluster_cols, key=_cluster_num)
    n_clusters = len(cluster_order)

    # Map to indices
    stid_to_yi = {s: i for i, s in enumerate(row_stids)}
    cluster_to_xi = {c: i for i, c in enumerate(cluster_order)}

    df = dotplot_df.copy()
    df["_xi"] = df["cluster"].map(cluster_to_xi)
    df["_yi"] = df["stId"].map(stid_to_yi)
    df = df.dropna(subset=["_xi", "_yi"])
    df["_xi"] = df["_xi"].astype(int)
    df["_yi"] = df["_yi"].astype(int)

    # Dot sizing — scaled by -log10(p-value)
    MIN_S, MAX_S = 10, 200
    max_nlp = df["neg_log10_pval"].max()
    if max_nlp > 0:
        df["dot_size"] = ((df["neg_log10_pval"] / max_nlp) * (MAX_S - MIN_S) + MIN_S).clip(upper=MAX_S)
    else:
        df["dot_size"] = MIN_S

    # Color normalization
    cmap, norm, _ = _cmap_and_norm(df["value"], colormap)

    # Figure layout
    _img_paths: list[Path] = [p for p in (img_paths or []) if p and p.exists()]
    has_images = len(_img_paths) > 0
    GAP_IN = 0.25
    LEGEND_W_IN = 3.0 if annotation_map else 0.0

    lay = _compute_figure_layout(
        n_rows, n_clusters,
        label_w_in=3.8, col_w_in=0.35, row_h_in=0.264,
        extra_w_in=GAP_IN + 0.30 + LEGEND_W_IN,
        top_in=0.60, bot_in=1.4 if organelle_labels else 0.80,
        has_images=has_images,
    )
    fig_w, fig_h = lay["fig_w"], lay["fig_h"]
    ax_bottom, ax_height = lay["ax_bottom"], lay["ax_height"]

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.text(0.5, 0.97, title, fontsize=11, fontweight="bold", va="top", ha="center")

    if has_images:
        _add_image_panels(fig, _img_paths, fig_w, fig_h, ax_bottom, ax_height)

    ax_left, ax_width = lay["ax_left"], lay["ax_width"]
    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

    sc = ax.scatter(
        df["_xi"], df["_yi"],
        c=df["value"], s=df["dot_size"],
        cmap=cmap, norm=norm,
        edgecolors="black", linewidths=0.4, zorder=3,
    )

    _draw_cluster_axis(
        ax,
        cluster_order,
        cluster_colors,
        cluster_color_nums=cluster_color_nums,
        annotation_map=annotation_map,
        organelle_labels=organelle_labels,
        tick_length=3,
        label_fontsize=10,
        footer_fontsize=10,
    )
    _draw_pathway_axis(
        ax,
        row_names,
        row_depths,
        parent_width=57,
        child_width=60,
        parent_fontsize=9,
        child_fontsize=8.5,
    )

    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Colorbar is placed in the right panel (below legends) — drawn after legends below

    # Build size-legend handles (used below, either in right panel or standalone)
    size_handles = []
    legend_title = ""
    if max_nlp > 0:
        nlp_min = -np.log10(0.05)          # smallest significant dot
        nlp_mid = (nlp_min + max_nlp) / 2  # midpoint in -log10 space
        legend_nlp = [nlp_min, nlp_mid, max_nlp]
        legend_labels = [f"p={10 ** -n:.2g}" for n in legend_nlp]
        legend_sizes = [
            float(np.clip((nlp / max_nlp) * (MAX_S - MIN_S) + MIN_S, MIN_S, MAX_S))
            for nlp in legend_nlp
        ]
        size_handles = [
            plt.scatter([], [], s=s, c="gray", edgecolors="black", linewidths=0.4, label=l)
            for s, l in zip(legend_sizes, legend_labels)
        ]
        legend_title = (
            "-log10(p-BH)\n(Wilcoxon, log2FC)"
            if is_foldchange
            else "-log10(p-BH)\n(MW-U, 1-vs-rest)"
        )

    # Annotation legend — right panel, 1 column, square handles
    # Size legend placed below it in the same right panel
    if annotation_map:
        from matplotlib.legend_handler import HandlerPatch

        class _HandlerSquare(HandlerPatch):
            def create_artists(self, legend, orig_handle, xdescent, ydescent,
                               width, height, fontsize, trans):
                size = min(width, height)
                p = mpatches.Rectangle(
                    (xdescent + (width - size) / 2, ydescent + (height - size) / 2),
                    size, size,
                    facecolor=orig_handle.get_facecolor(),
                    edgecolor=orig_handle.get_edgecolor(),
                    linewidth=orig_handle.get_linewidth(),
                    transform=trans,
                )
                return [p]

        ann_handles = [
            mpatches.Patch(
                facecolor=info["color"], edgecolor="black", linewidth=0.5,
                label=f"Cluster {cl_num}: {info['label']}",
            )
            for cl_num, info in sorted(annotation_map.items())
        ]
        leg_x = ax_left + ax_width + (GAP_IN + 0.15) / fig_w
        leg_y = ax_bottom + ax_height  # align top with dotplot top
        fig.legend(
            handles=ann_handles,
            title="Cluster annotations",
            title_fontsize=9, fontsize=9,
            loc="upper left",
            bbox_to_anchor=(leg_x, leg_y),
            bbox_transform=fig.transFigure,
            ncol=1, frameon=True,
            handler_map={mpatches.Patch: _HandlerSquare()},
        )
        # Size legend below annotation legend (estimate annotation legend height)
        ann_legend_h_in = len(annotation_map) * 0.25 + 0.65
        if size_handles:
            leg_y_size = leg_y - ann_legend_h_in / fig_h
            fig.legend(
                handles=size_handles,
                title=legend_title,
                title_fontsize=9, fontsize=9,
                loc="upper left",
                bbox_to_anchor=(leg_x, leg_y_size),
                bbox_transform=fig.transFigure,
                ncol=1, frameon=True,
            )
        # Horizontal colorbar below size legend (or annotation legend if no size handles)
        size_legend_h_in = (3 * 0.25 + 0.65) if size_handles else 0.0
        cbar_gap_in   = 0.45
        cbar_h_in     = 0.18
        cbar_w_in     = min(LEGEND_W_IN - 0.4, 2.0)
        cbar_bot_in   = (leg_y * fig_h) - ann_legend_h_in - size_legend_h_in - cbar_gap_in - cbar_h_in
        cbar_ax = fig.add_axes([
            leg_x,
            cbar_bot_in / fig_h,
            cbar_w_in / fig_w,
            cbar_h_in / fig_h,
        ])
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(value_label, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
    else:
        # No annotation panel — vertical colorbar and size legend next to dotplot
        cbar_ax = fig.add_axes([
            ax_left + ax_width + 0.01,
            ax_bottom + ax_height * 0.3,
            0.015,
            ax_height * 0.4,
        ])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label(value_label, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        if size_handles:
            ax.legend(
                handles=size_handles,
                title=legend_title,
                title_fontsize=9, fontsize=9,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
                ncol=len(size_handles), frameon=True,
            )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def make_heatmap(
    dotplot_df: pd.DataFrame,
    cluster_cols: list[str],
    output_path: Path,
    title: str = "Reactome Sub-pathway Expression",
    colormap: str = "Reds",
    value_label: str = "Mean expression",
    cluster_color_nums: dict[str, int] | None = None,
    img_path: Path | None = None,
    cluster_colors: dict[int, tuple[float, float, float]] | None = None,
) -> None:
    """Render a heatmap (pathways × clusters) to *output_path*.

    Each cell is filled with the pathway colour; overlap_size is shown as text.
    """
    if cluster_colors is None:
        cluster_colors = CLUSTER_COLORS

    if dotplot_df.empty:
        logger.warning("No data to plot.")
        return

    row_stids, row_names, row_depths = _build_row_order(dotplot_df)
    n_rows = len(row_stids)
    if n_rows == 0:
        logger.warning("No pathways with sufficient gene overlap.")
        return

    cluster_order = sorted(cluster_cols, key=_cluster_num)
    n_clusters = len(cluster_order)

    stid_to_yi   = {s: i for i, s in enumerate(row_stids)}
    cluster_to_xi = {c: i for i, c in enumerate(cluster_order)}

    df = dotplot_df.copy()
    df["_xi"] = df["cluster"].map(cluster_to_xi)
    df["_yi"] = df["stId"].map(stid_to_yi)
    df = df.dropna(subset=["_xi", "_yi"])
    df["_xi"] = df["_xi"].astype(int)
    df["_yi"] = df["_yi"].astype(int)

    # Build 2-D grids
    val_grid     = np.full((n_rows, n_clusters), np.nan)
    overlap_grid = np.zeros((n_rows, n_clusters), dtype=int)
    for _, row in df.iterrows():
        val_grid    [row["_yi"], row["_xi"]] = row["value"]
        overlap_grid[row["_yi"], row["_xi"]] = row["overlap_size"]

    cmap_obj, norm, _ = _cmap_and_norm(df["value"], colormap)

    # Figure layout
    has_img = img_path is not None and img_path.exists()
    lay = _compute_figure_layout(
        n_rows, n_clusters,
        label_w_in=2.8, col_w_in=0.38, row_h_in=0.30,
        extra_w_in=0.55 + 0.30,
        top_in=0.60, bot_in=0.80,
        has_images=bool(img_path),
    )
    fig_w = lay["fig_w"]
    ax_left, ax_width = lay["ax_left"], lay["ax_width"]

    if has_img:
        iw, ih = Image.open(img_path).size
        fig_h = max(7.0, (IMG_W_IN * ih / iw) / 0.84)
    else:
        fig_h = lay["fig_h"]

    fig = plt.figure(figsize=(fig_w, fig_h))

    if has_img:
        _, ax_bottom, ax_height = _add_image_panel(fig, img_path, title, fig_w, fig_h)
    else:
        ax_bottom, ax_height = lay["ax_bottom"], lay["ax_height"]
        fig.text(0.5, 0.97, title, fontsize=11, fontweight="bold",
                 va="top", ha="center")

    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

    # Draw heatmap
    im = ax.imshow(
        val_grid,
        aspect="auto",
        cmap=cmap_obj, norm=norm,
        origin="upper",
        interpolation="nearest",
    )

    # Annotate cells with overlap size
    for yi in range(n_rows):
        for xi in range(n_clusters):
            n = overlap_grid[yi, xi]
            if n > 0:
                val = val_grid[yi, xi]
                # Pick text colour for contrast
                mapped = norm(val)
                cell_rgba = cmap_obj(mapped)
                lum = 0.299 * cell_rgba[0] + 0.587 * cell_rgba[1] + 0.114 * cell_rgba[2]
                txt_color = "black" if lum > 0.5 else "white"
                ax.text(xi, yi, str(n), ha="center", va="center",
                        fontsize=6, color=txt_color)

    _draw_cluster_axis(
        ax,
        cluster_order,
        cluster_colors,
        cluster_color_nums=cluster_color_nums,
        tick_length=0,
        label_fontsize=8,
        footer_fontsize=8,
    )
    _draw_pathway_axis(
        ax,
        row_names,
        row_depths,
        parent_width=38,
        child_width=40,
        parent_fontsize=7,
        child_fontsize=6.5,
    )

    ax.set_xlim(-0.5, n_clusters - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)  # keep origin="upper" inversion: row 0 at top

    # Title (only when no image panel)
    if not (img_path and img_path.exists()):
        fig.text(
            ax_left + ax_width / 2,
            ax_bottom + ax_height + 0.02,
            title, fontsize=11, fontweight="bold", va="bottom", ha="center",
        )

    # Colorbar
    cbar_ax = fig.add_axes([
        ax_left + ax_width + 0.01,
        ax_bottom + ax_height * 0.3,
        0.015,
        ax_height * 0.4,
    ])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(value_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Legend: cell numbers = gene overlap
    ax.text(
        0.5, -0.16,
        "Cell numbers = genes with expression data",
        transform=ax.transAxes, ha="center", va="top", fontsize=7,
        color=(0.4, 0.4, 0.4),
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reactome sub-pathway expression dotplot/heatmap")
    p.add_argument("--pathway-ids", nargs="+", required=True,
                   help="Reactome stable IDs, e.g. R-HSA-75848")
    p.add_argument("--expression-csv", required=True, type=Path,
                   help="CSV with columns: Gene, Cluster_1, ..., Cluster_N")
    p.add_argument("--depth", type=int, default=1,
                   help="Traversal depth (1=direct children, 2=grandchildren, etc.)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output PNG path")
    p.add_argument("--cache", type=Path, default=None,
                   help="JSON cache file for API responses (default: reactome_cache.json next to script)")
    p.add_argument("--min-overlap", type=int, default=1,
                   help="Minimum genes overlapping expression data to include a pathway (default: 1)")
    p.add_argument("--pval-threshold", type=float, default=1.0,
                   help="Only show dots with p-value below this threshold (default: 1.0 = show all)")
    p.add_argument("--title", type=str, default=None,
                   help="Figure title (default: auto-generated from pathway IDs)")
    p.add_argument("--plot-type", choices=["dotplot", "heatmap"], default="dotplot",
                   help="Visualization style: dotplot (default) or heatmap")
    p.add_argument("--metric", choices=["mean", "sum"], default="mean",
                   help="Aggregation over genes per pathway/cluster: mean (default) or sum")
    p.add_argument("--input-mode", choices=["expression", "foldchange"], default="expression",
                   help="Interpret expression values as raw expression or fold-change values.")
    p.add_argument("--colormap", type=str, default=None,
                   help="Matplotlib colormap (default: Reds for expression, RdBu_r for fold-change). "
                        "Diverging cmaps (RdBu_r etc.) are centred at 0.")
    p.add_argument("--cell-line", type=str, default=None,
                   help="Cell line name (e.g. Hep-G2_fatty_acid_metabolism) to apply correct "
                        "cluster box colours after cluster renaming.")
    p.add_argument("--color-scheme", choices=list(COLOR_SCHEMES.keys()), default="default",
                   help="Cluster box colour palette: 'default' or 'paul' (ColorBrewer Set1).")
    p.add_argument("--image", type=Path, nargs="+", default=None,
                   help="One or more clustered-cell PNGs stacked top-to-bottom in the left panel.")
    p.add_argument("--organelle-labels", action="store_true",
                   help="Label x-axis with organelle names (colored) instead of cluster number boxes.")
    p.add_argument("--annotation-csv", type=Path, default=None,
                   help="CSV with columns Cluster,Color,Annotation,Secondary_Annotation. "
                        "When provided, draws a filled colored square per cluster on the x-axis "
                        "and adds a cluster annotation legend below the plot.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).parent
    cache_path = args.cache if args.cache else script_dir / "reactome_cache.json"
    cache = _load_cache(cache_path)

    # Load expression data
    expr_df = pd.read_csv(args.expression_csv)
    expr_df.columns = [c.strip() for c in expr_df.columns]
    expr_df, cluster_cols = _get_cluster_cols(expr_df)
    logger.info("Expression data: %d genes, clusters: %s", len(expr_df), cluster_cols)

    # Resolve IDs (handle reactions / non-pathway stIds)
    logger.info("Resolving pathway IDs: %s", args.pathway_ids)
    resolved_ids: list[str] = []
    for pid in args.pathway_ids:
        resolved_ids.extend(resolve_pathway_id(pid, cache, cache_path))
    resolved_ids = list(dict.fromkeys(resolved_ids))  # deduplicate, preserve order
    logger.info("Using pathway roots: %s", resolved_ids)

    # Traverse Reactome hierarchy
    logger.info("Traversing Reactome hierarchy (depth=%d)...", args.depth)
    pathways = traverse_pathways(resolved_ids, args.depth, cache, cache_path)

    if not pathways:
        logger.error("No sub-pathways found. Check pathway IDs and depth.")
        return

    # Determine input interpretation explicitly
    is_foldchange = args.input_mode == "foldchange"
    logger.info("Input mode: %s", args.input_mode)

    if is_foldchange:
        # log2-transform fold-change values (Gene column stays as-is)
        for col in cluster_cols:
            expr_df[col] = np.log2(expr_df[col].clip(lower=1e-9)).clip(lower=-2)
        logger.info("Applied log2 transform to fold-change values (negative values capped at -2).")

    # Compute per-(pathway, cluster) stats
    logger.info("Computing expression statistics (metric=%s)...", args.metric)
    dotplot_df = compute_dotplot_data(
        pathways, expr_df, cluster_cols, args.min_overlap, metric=args.metric,
        is_foldchange=is_foldchange,
    )
    logger.info("Rows: %d pathways × %d clusters",
                dotplot_df["stId"].nunique(), dotplot_df["cluster"].nunique())

    # Benjamini-Hochberg multiple testing correction across all (pathway × cluster) tests
    if not dotplot_df.empty:
        pvals = dotplot_df["pval"].values.copy()
        bh_pvals = _bh_adjust(pvals)
        dotplot_df = dotplot_df.copy()
        dotplot_df["pval"] = bh_pvals
        dotplot_df["neg_log10_pval"] = -np.log10(np.maximum(bh_pvals, 1e-300))
        logger.info("Applied Benjamini-Hochberg correction over %d tests.", len(pvals))

    if args.pval_threshold < 1.0:
        dotplot_df = dotplot_df[dotplot_df["pval"] < args.pval_threshold]
        logger.info("After BH-corrected p-value filter (<%s): %d rows",
                     args.pval_threshold, len(dotplot_df))

    if dotplot_df.empty:
        logger.error("No pathways with sufficient gene overlap.")
        return

    # Auto-select colormap and value label
    metric_label = {"mean": "Mean", "sum": "Summed"}[args.metric]
    if is_foldchange:
        colormap    = args.colormap or "RdBu_r"
        value_label = f"{metric_label} log2 fold change"
    else:
        colormap    = args.colormap or "Reds"
        value_label = f"{metric_label} expression"

    # Cluster box colours: look up pre-rename mapping for known cell lines
    cluster_color_nums: dict[str, int] | None = None
    if args.cell_line and args.cell_line in CLUSTER_DISPLAY_COLORS:
        color_list = CLUSTER_DISPLAY_COLORS[args.cell_line]
        cluster_color_nums = {
            f"Cluster_{i + 1}": color_list[i]
            for i in range(len(color_list))
        }
        logger.info("Using cluster colour mapping for %s: %s", args.cell_line, cluster_color_nums)

    # Figure title
    title = args.title or f"Sub-pathways of {', '.join(args.pathway_ids)}"

    # Cluster colour palette
    cluster_colors = COLOR_SCHEMES[args.color_scheme]

    # Annotation map (from --annotation-csv)
    annotation_map = load_annotation_csv(args.annotation_csv) if args.annotation_csv else None
    if annotation_map:
        logger.info("Loaded annotation map from %s: %d clusters", args.annotation_csv, len(annotation_map))

    # Render
    args.output.parent.mkdir(parents=True, exist_ok=True)
    img_paths = args.image or []   # list[Path] (nargs='+') or empty
    if args.plot_type == "heatmap":
        make_heatmap(dotplot_df, cluster_cols, args.output, title=title,
                     colormap=colormap, value_label=value_label,
                     cluster_color_nums=cluster_color_nums,
                     img_path=img_paths[0] if img_paths else None,
                     cluster_colors=cluster_colors)
    else:
        make_dotplot(dotplot_df, cluster_cols, args.output, title=title,
                     colormap=colormap, value_label=value_label,
                     cluster_color_nums=cluster_color_nums,
                     img_paths=img_paths, is_foldchange=is_foldchange,
                     cluster_colors=cluster_colors,
                     organelle_labels=args.organelle_labels,
                     annotation_map=annotation_map)


if __name__ == "__main__":
    main()
