"""Reactome sub-pathway dotplot (Fig 5G): hierarchy traversal through the Reactome ContentService,
per-(sub-pathway, cluster) statistics, and the dotplot renderer. Companion module of plot_dotplot.py.

Every ContentService response is cached in data/reactome_cache.json (keyed by request URL) on
first use, so a rerun is offline and reproducible; delete the cache to refresh the hierarchy.
"""
import json
import textwrap
from collections import deque
from pathlib import Path

import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon

from paperlib import save_figure

REACTOME_BASE = "https://reactome.org/ContentService"
PATHWAY_SCHEMA_CLASSES = {"Pathway", "TopLevelPathway"}
# Cluster colours by colour name, as used by the pixel-level clustering's own annotation table.
COLOR_RGB = {
    "red": (1.0, 0.0, 0.0), "blue": (0.0, 0.0, 1.0), "green": (0.0, 128 / 255, 0.0), "yellow": (1.0, 1.0, 0.0),
    "magenta": (1.0, 0.0, 1.0), "cyan": (0.0, 1.0, 1.0), "orange": (1.0, 165 / 255, 0.0), "purple": (128 / 255, 0.0, 128 / 255),
    "teal": (0.0, 128 / 255, 128 / 255), "sky blue": (135 / 255, 206 / 255, 235 / 255), "maroon": (128 / 255, 0.0, 0.0),
}


# ------------------------------------------------------------------ Reactome ContentService, cached
class Cache(dict):
    """ContentService responses keyed by URL; a miss is fetched once and written back to `path`."""

    def __init__(self, path: Path):
        self.path = Path(path)
        super().__init__(json.loads(self.path.read_text()) if self.path.exists() else {})

    def get_url(self, url: str):
        if url not in self:
            import requests
            r = requests.get(url, timeout=30)
            r.raise_for_status()                    # never cache an error body
            self[url] = r.json()
            self.path.write_text(json.dumps(self, indent=1))
        return self[url]


def child_pathways(stid: str, cache: Cache) -> list[dict]:
    """Direct sub-pathways (Pathway/TopLevelPathway) of `stid`, via hasEvent."""
    data = cache.get_url(f"{REACTOME_BASE}/data/query/{stid}")
    return [e for e in data.get("hasEvent", []) if isinstance(e, dict) and e.get("schemaClass") in PATHWAY_SCHEMA_CLASSES]


def pathway_genes(stid: str, cache: Cache) -> list[str]:
    """Unique gene symbols of all ReferenceGeneProduct participants ('UniProt:P07204 THBD' -> THBD)."""
    genes: set[str] = set()
    for item in cache.get_url(f"{REACTOME_BASE}/data/participants/{stid}") or []:
        for ref in item.get("refEntities", []):
            if ref.get("schemaClass") == "ReferenceGeneProduct":
                parts = ref.get("displayName", "").strip().split()
                if parts:
                    genes.add(parts[-1].upper())
    return sorted(genes)


def traverse_pathways(root_id: str, depth: int, cache: Cache) -> dict[str, dict]:
    """BFS to `depth` levels below the root pathway -> {stId: {name, depth, parent, genes}}."""
    assert cache.get_url(f"{REACTOME_BASE}/data/query/{root_id}").get("schemaClass") in PATHWAY_SCHEMA_CLASSES
    result: dict[str, dict] = {}
    queue: deque = deque([(root_id, 0)])
    visited: set[str] = {root_id}
    while queue:
        cur, d = queue.popleft()
        if d >= depth:
            continue
        for child in child_pathways(cur, cache):
            cid = child.get("stId", "")
            if not cid or cid in visited:
                continue
            visited.add(cid)
            result[cid] = {"name": child.get("displayName", cid), "depth": d + 1,
                           "parent": cur if cur != root_id else None, "genes": []}
            queue.append((cid, d + 1))
    for stid, info in result.items():
        info["genes"] = pathway_genes(stid, cache)
    return result


# ------------------------------------------------------------------ statistics
def _cluster_num(col: str) -> int:
    return int(col.split("_")[-1])


def pathway_cluster_stats(pathways: dict, expr_df: pd.DataFrame, cluster_cols: list[str]) -> pd.DataFrame:
    """Per (pathway, cluster): mean log2 spatial enrichment over the pathway's genes, and a
    ONE-SIDED one-sample Wilcoxon signed-rank p (H0 median = 0, H1 > 0); p = 1 with fewer than
    four detected member proteins (or all-equal values)."""
    gene_upper = expr_df["Gene"].str.upper()
    expr_genes = set(gene_upper)
    rows = []
    for stid, info in pathways.items():
        pw_genes = [g for g in info["genes"] if g in expr_genes]
        if not pw_genes:
            continue
        sub = expr_df[gene_upper.isin(pw_genes)]
        for col in cluster_cols:
            vals = sub[col].dropna().values
            pval = wilcoxon(vals, alternative="greater")[1] if len(vals) >= 4 and not np.all(vals == vals[0]) else 1.0
            rows.append(dict(stId=stid, name=info["name"], depth=info["depth"], parent=info["parent"],
                             cluster=col, value=float(sub[col].mean()), overlap_size=len(pw_genes), pval=pval))
    return pd.DataFrame(rows)


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, same order as the input."""
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = pvals[order] * n / np.arange(1, n + 1, dtype=float)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def dotplot_table(root_id: str, depth: int, enrichment_csv: Path, cache_path: Path) -> pd.DataFrame:
    """The full BH-corrected (sub-pathway x cluster) table the dotplot is filtered from.

    The enrichment CSV holds the protein-level spatial enrichment s_{k,g} (mean binarised
    occupancy in cluster k over the unweighted mean across the other clusters; Methods), one
    Cluster_<k> column per cluster. It is log2-transformed here, floored at -2, before testing
    and plotting."""
    cache = Cache(cache_path)
    expr = pd.read_csv(enrichment_csv)
    expr.columns = [c.strip() for c in expr.columns]
    cols = sorted([c for c in expr.columns if c.startswith("Cluster_") and _cluster_num(c) > 0], key=_cluster_num)
    for c in cols:
        expr[c] = np.log2(expr[c].clip(lower=1e-9)).clip(lower=-2)
    df = pathway_cluster_stats(traverse_pathways(root_id, depth, cache), expr, cols)
    df["pval"] = bh_adjust(df["pval"].to_numpy())
    df["neg_log10_pval"] = -np.log10(np.maximum(df["pval"].to_numpy(), 1e-300))
    df.attrs["cluster_cols"] = cols
    return df


# ------------------------------------------------------------------ figure
def _row_order(df: pd.DataFrame) -> tuple[list[str], list[str], list[int]]:
    """Display order: depth-1 parents alphabetically, each followed by its depth-2 children."""
    depth1 = df[df["depth"] == 1][["stId", "name"]].drop_duplicates().sort_values("name")
    depth2 = df[df["depth"] == 2][["stId", "name", "parent"]].drop_duplicates().sort_values("name")
    rows = []
    for p in depth1.itertuples():
        rows.append((p.stId, p.name, 1))
        rows += [(c.stId, c.name, 2) for c in depth2[depth2["parent"] == p.stId].itertuples()]
    rows += [(c.stId, c.name, 2) for c in depth2[~depth2["parent"].isin(depth1["stId"])].itertuples()]
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def _draw_cluster_axis(ax, cluster_order, annotation) -> None:
    """x axis: a coloured square per cluster."""
    n = len(cluster_order)
    ax.set_xticks(range(n)); ax.set_xticklabels([""] * n)
    ax.tick_params(axis="x", length=3, direction="out")
    ax.set_xlim(-0.5, n - 0.5)
    for i, col in enumerate(cluster_order):
        ax.plot(i, -0.025, marker="s", markersize=9, color=annotation[_cluster_num(col)]["color"],
                markeredgecolor="black", markeredgewidth=0.5, transform=ax.get_xaxis_transform(), clip_on=False)


def _draw_pathway_axis(ax, names, depths) -> None:
    """y axis: parents bold, children indented with a bullet, both wrapped."""
    ax.set_yticks(range(len(names))); ax.set_yticklabels([""] * len(names))
    ax.tick_params(axis="y", length=0)
    for i, (name, dep) in enumerate(zip(names, depths)):
        if dep == 1:
            ax.text(-0.01, i, textwrap.fill(name, width=57), transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=9, fontweight="bold", color=(0.1, 0.1, 0.1), clip_on=False)
        else:
            ax.text(-0.01, i, textwrap.fill("· " + name, width=60), transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=8.5, color=(0.3, 0.3, 0.3), clip_on=False)


def _add_image_panels(fig, img_paths, fig_w, fig_h, ax_bottom, ax_height, img_w_in) -> None:
    """Stack the cell images vertically as squares in the left panel."""
    n = len(img_paths)
    margin_l = 0.1 / fig_w
    slot_h = ax_height * fig_h / n
    sq = min(img_w_in - 0.1, slot_h)
    sq_w, sq_h = sq / fig_w, sq / fig_h
    slot_frac = ax_height / n
    left = margin_l + (img_w_in / fig_w - margin_l - sq_w) / 2
    for idx, path in enumerate(img_paths):
        slot_bot = ax_bottom + (n - 1 - idx) * slot_frac
        bot = (slot_bot + slot_frac - sq_h if idx == 0 else slot_bot if idx == n - 1
               else slot_bot + (slot_frac - sq_h) / 2)
        ax = fig.add_axes([left, bot, sq_w, sq_h])
        ax.imshow(np.array(Image.open(path)), interpolation="lanczos")
        ax.axis("off")


class _HandlerSquare(HandlerPatch):
    """Square legend handles for the cluster annotation legend."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        size = min(width, height)
        return [mpatches.Rectangle((xdescent + (width - size) / 2, ydescent + (height - size) / 2),
                                   size, size, facecolor=orig_handle.get_facecolor(),
                                   edgecolor=orig_handle.get_edgecolor(),
                                   linewidth=orig_handle.get_linewidth(), transform=trans)]


def make_dotplot(df: pd.DataFrame, cluster_cols: list[str], output_path: Path, title: str,
                 annotation: dict[int, dict], img_paths: list[Path], bot_in: float = 1.4) -> None:
    """Dot colour = mean log2 spatial enrichment (diverging, centred at 0); dot size = -log10(BH p).
    `annotation` maps cluster number -> {"color": rgb, "label": str}. `bot_in` is the bottom
    margin in inches. Saved through paperlib.save_figure (SVG + PNG)."""
    img_w_in = 4.0
    row_stids, row_names, row_depths = _row_order(df)
    n_rows = len(row_stids)
    cluster_order = sorted(cluster_cols, key=_cluster_num)
    n_clusters = len(cluster_order)
    df = df.copy()
    df["_xi"] = df["cluster"].map({c: i for i, c in enumerate(cluster_order)})
    df["_yi"] = df["stId"].map({s: i for i, s in enumerate(row_stids)})

    MIN_S, MAX_S = 10, 200
    max_nlp = df["neg_log10_pval"].max()
    df["dot_size"] = ((df["neg_log10_pval"] / max_nlp) * (MAX_S - MIN_S) + MIN_S).clip(upper=MAX_S)
    bound = float(df["value"].abs().max())
    cmap, norm = plt.get_cmap("RdBu_r"), plt.Normalize(vmin=-bound, vmax=bound)

    GAP_IN, LEGEND_W_IN = 0.25, 3.0
    label_w_in, col_w_in, row_h_in, top_in = 3.8, 0.35, 0.264, 0.60
    plot_w_in = n_clusters * col_w_in
    fig_w = img_w_in + label_w_in + plot_w_in + GAP_IN + 0.30 + LEGEND_W_IN
    img_frac_w = img_w_in / fig_w
    plot_h_in = max(2.0, n_rows * row_h_in)
    fig_h = plot_h_in + top_in + bot_in
    ax_left, ax_width = img_frac_w + label_w_in / fig_w, plot_w_in / fig_w
    ax_bottom, ax_height = bot_in / fig_h, plot_h_in / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.text(0.5, 0.97, title, fontsize=11, fontweight="bold", va="top", ha="center")
    _add_image_panels(fig, img_paths, fig_w, fig_h, ax_bottom, ax_height, img_w_in)
    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
    sc = ax.scatter(df["_xi"], df["_yi"], c=df["value"], s=df["dot_size"], cmap=cmap, norm=norm,
                    edgecolors="black", linewidths=0.4, zorder=3)
    _draw_cluster_axis(ax, cluster_order, annotation)
    _draw_pathway_axis(ax, row_names, row_depths)
    ax.set_ylim(-0.5, n_rows - 0.5); ax.invert_yaxis()
    ax.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)

    nlp_min = -np.log10(0.05)
    legend_nlp = [nlp_min, (nlp_min + max_nlp) / 2, max_nlp]
    sizes = [float(np.clip((v / max_nlp) * (MAX_S - MIN_S) + MIN_S, MIN_S, MAX_S)) for v in legend_nlp]
    size_handles = [plt.scatter([], [], s=s, c="gray", edgecolors="black", linewidths=0.4, label=f"p={10 ** -v:.2g}")
                    for s, v in zip(sizes, legend_nlp)]
    ann_handles = [mpatches.Patch(facecolor=info["color"], edgecolor="black", linewidth=0.5, label=f"Cluster {cl}: {info['label']}")
                   for cl, info in sorted(annotation.items())]
    leg_x = ax_left + ax_width + (GAP_IN + 0.15) / fig_w
    leg_y = ax_bottom + ax_height
    fig.legend(handles=ann_handles, title="Cluster annotations", title_fontsize=9, fontsize=9,
               loc="upper left", bbox_to_anchor=(leg_x, leg_y), bbox_transform=fig.transFigure,
               ncol=1, frameon=True, handler_map={mpatches.Patch: _HandlerSquare()})
    ann_h_in = len(annotation) * 0.25 + 0.65
    fig.legend(handles=size_handles, title="-log10(p-BH)\n(one-sided Wilcoxon)", title_fontsize=9, fontsize=9,
               loc="upper left", bbox_to_anchor=(leg_x, leg_y - ann_h_in / fig_h), bbox_transform=fig.transFigure,
               ncol=1, frameon=True)
    size_h_in = 3 * 0.25 + 0.65
    cbar_h_in, cbar_w_in = 0.18, min(LEGEND_W_IN - 0.4, 2.0)
    cbar_bot_in = (leg_y * fig_h) - ann_h_in - size_h_in - 0.45 - cbar_h_in
    cbar_ax = fig.add_axes([leg_x, cbar_bot_in / fig_h, cbar_w_in / fig_w, cbar_h_in / fig_h])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("mean log2 spatial enrichment", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    save_figure(fig, output_path, dpi=300)
