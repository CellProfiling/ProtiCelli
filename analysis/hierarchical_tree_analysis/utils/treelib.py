"""Hierarchical-clustering trees: label IO, ONE hierarchy builder, ONE graphviz layout, the
left-to-right renumbering fixed point, two thin renderers (circle nodes for Fig 3C, heatmap nodes
for the Fig 6B consensus tree) and the Fig 3D 2x2 subcluster plot. Shared by plot_panels_C_D.py,
plot_tree_heatmaps.py and the compute scripts next to them.

Node keys are (level, cluster_id), level 1 being the single root cluster; the tuned resolution is
not part of the key because two levels can tune to the same value. Node INSERTION ORDER is
load-bearing: graphviz `dot` derives within-rank x positions partly from it, so every builder here
inserts levels in order and clusters in np.unique order, exactly as the shipped figures were laid out.
"""
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from paperlib import save_figure

RES_RE = re.compile(r"leiden_lvl(\d+)_res([\d.]+)")


# ------------------------------------------------------------------ label IO
def load_cluster_labels(path, expected_levels: int = 7) -> tuple[pd.DataFrame, list[str]]:
    """Per-protein cluster labels TSV (gene_name + leiden_lvl<N>_res<R> columns) ->
    (DataFrame, the first `expected_levels` leiden columns sorted by level)."""
    df = pd.read_csv(path, sep="\t")
    if "gene_name" not in df.columns:
        raise ValueError(f"Missing 'gene_name' column in {path}")
    parsed = []
    for c in [c for c in df.columns if c.startswith("leiden_")]:
        m = RES_RE.match(c)
        if not m:
            raise ValueError(f"column {c!r} is not leiden_lvl<N>_res<R>")
        parsed.append((int(m.group(1)), c))
    if len(parsed) < expected_levels:
        raise ValueError(f"Expected >= {expected_levels} leiden columns in {path}, got {len(parsed)}")
    return df, [c for _, c in sorted(parsed)][:expected_levels]


def resolutions_of(cols) -> list[float]:
    return [float(RES_RE.match(c).group(2)) for c in cols]


def align_datasets(df_real, df_gen, real_cols, gen_cols):
    """Both label frames on the common genes, sorted by gene name (deterministic)."""
    common = sorted(set(df_real["gene_name"]) & set(df_gen["gene_name"]))
    if not common:
        raise ValueError("No common genes found between datasets")
    df_r = df_real[df_real["gene_name"].isin(common)].sort_values("gene_name").reset_index(drop=True)
    df_g = df_gen[df_gen["gene_name"].isin(common)].sort_values("gene_name").reset_index(drop=True)
    assert all(df_r["gene_name"] == df_g["gene_name"]), "Gene alignment failed"
    return df_r[["gene_name"] + real_cols], df_g[["gene_name"] + gen_cols]


# ------------------------------------------------------------------ the hierarchy graph
def build_hierarchy(genes, levels, threshold: float = 0.1, cohesion=None) -> nx.DiGraph:
    """Directed tree over cluster nodes (level, cluster_id) from per-level label arrays.

    genes     : gene name per row
    levels    : [(level, labels)] in level order; level 1 is the single all-genes cluster
    threshold : parent -> child edge when |child & parent| / |child| > threshold
    cohesion  : optional {level: {cluster_id: score}}; when given, nodes also carry
                size and mean_consensus attributes (the consensus tree colours by it)
    """
    genes = np.asarray(genes)
    G = nx.DiGraph()
    clusters: list[tuple] = []
    for level, labels in levels:
        labels = np.asarray(labels)
        here = {}
        for lab in np.unique(labels):
            lab = int(lab)
            gs = set(genes[labels == lab])
            attrs = {"genes": gs}
            if cohesion is not None:
                attrs.update(size=len(gs), mean_consensus=cohesion.get(level, {}).get(lab, 0.0))
            G.add_node((level, lab), **attrs)
            here[lab] = gs
        clusters.append((level, here))
    for (pl, parents), (cl, children) in zip(clusters, clusters[1:]):
        for pid, pg in parents.items():
            for cid, cg in children.items():
                if cg and len(cg & pg) / len(cg) > threshold:
                    G.add_edge((pl, pid), (cl, cid))
    return G


def hierarchy_from_labels(df, cols, max_levels: int = 7) -> tuple[nx.DiGraph, list[int]]:
    """The tree of one label frame (levels 1..max_levels) and its level list."""
    used = cols[:max_levels]
    levels = list(range(1, len(used) + 1))
    G = build_hierarchy(df["gene_name"].to_numpy(), [(lvl, df[c].to_numpy()) for lvl, c in zip(levels, used)])
    return G, levels


# ------------------------------------------------------------------ graphviz layout
def dot_layout(G, nodesep=1, ranksep=0.7) -> dict:
    """Positions from graphviz `dot`, top to bottom. Every node is laid out at one fixed size with
    no label, so the geometry depends on the tree alone (dot would otherwise size nodes by their
    key string). Raises if dot is unusable: spring_layout is on a different scale and once
    produced a 7,460,527-px canvas that read as a data bug."""
    env_bin = os.path.dirname(sys.executable)           # the conda build ships `dot` next to the interpreter
    if env_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")
    if not shutil.which("dot"):
        raise RuntimeError("graphviz `dot` not on PATH")
    try:
        A = nx.nx_agraph.to_agraph(G)
        A.node_attr.update(label="", fixedsize="true", width="1", height="0.5")
        A.layout(prog="dot", args=f"-Grankdir=TB -Gnodesep={nodesep} -Granksep={ranksep}")
        return {n: tuple(float(v) for v in A.get_node(n).attr["pos"].split(",")) for n in G.nodes}
    except Exception as exc:
        raise RuntimeError(f"graphviz layout failed: {exc}") from exc


def level_order(G, pos, levels) -> dict:
    """Left-to-right cluster ids per level, from the layout (x, then id breaks exact ties)."""
    out = {}
    for lvl in levels:
        at = [(pos[n][0], n[1]) for n in G.nodes if n[0] == lvl]
        out[lvl] = [c for _, c in sorted(at, key=lambda t: (t[0], str(t[1])))]
    return out


def pin_positions(G, pos) -> dict:
    """{(level, frozenset(genes)): (x, y)} for every node of G: the identity that survives
    a relabelling, so a layout made on raw ids can be reused for the renumbered tree."""
    out = {}
    for n in G.nodes:
        key = (n[0], frozenset(G.nodes[n]["genes"]))
        assert key not in out, "two clusters at one level share a gene set"
        out[key] = pos[n]
    return out


def pinned_layout(G, pinned: dict) -> dict:
    """Positions of G's nodes looked up in a `pin_positions` map."""
    return {n: pinned[(n[0], frozenset(G.nodes[n]["genes"]))] for n in G.nodes}


# ------------------------------------------------------------------ left-to-right renumbering
def renumber_left_to_right(df, cols, max_levels: int = 7, iter_max: int = 8):
    """Relabel cluster ids so each level reads 0,1,2,... left to right in the dot layout.

    A pure relabelling (partition unchanged; asserted), iterated to a fixed point because node
    identity is (level, id) and renaming can itself nudge the layout. Returns
    (relabelled frame, summary dict, old->new rows)."""
    used = cols[:max_levels]
    res = resolutions_of(used)
    levels = list(range(1, len(used) + 1))
    work, converged, it = df.copy(), False, 0
    for it in range(1, iter_max + 1):
        G, _ = hierarchy_from_labels(work, cols, max_levels)
        order = level_order(G, dot_layout(G), levels)
        if it > 1 and all(order[lvl] == sorted(order[lvl]) for lvl in levels):
            converged = True
            break
        base = work.copy()
        for lvl, c in zip(levels, used):
            mapping = {old: new for new, old in enumerate(order[lvl])}
            work[c] = base[c].map(mapping)
            assert work[c].notna().all(), f"{c}: unmapped cluster id"
    G, _ = hierarchy_from_labels(work, cols, max_levels)
    final = level_order(G, dot_layout(G), levels)
    summary = dict(iterations=it, converged=bool(converged),
                   levels_ascending={str(lvl): bool(final[lvl] == sorted(final[lvl])) for lvl in levels})
    rows = []
    for lvl, r, c in zip(levels, res, used):
        old_sets = {frozenset(g): k for k, g in df.groupby(c).gene_name.apply(frozenset).items()}
        for new_id, g in work.groupby(c).gene_name.apply(frozenset).items():
            rows.append(dict(level=lvl, resolution=r, column=c, old_cluster=str(old_sets[frozenset(g)]),
                             new_cluster=int(new_id), n_genes=len(g)))
    for c in used:
        assert set(df.groupby(c).gene_name.apply(frozenset)) == set(work.groupby(c).gene_name.apply(frozenset)), \
            f"{c}: partition changed, not a relabelling"
        assert sorted(work[c].unique()) == list(range(work[c].nunique())), f"{c}: ids are not 0..k-1"
    return work, summary, rows


# ------------------------------------------------------------------ GO enrichment (network)
def go_enrichment(G, finest_level) -> dict:
    """g:Profiler per node: GO:CC, GO:MF+GO:BP at the finest level; None for the root (level 1)
    and for nodes with < 5 genes. Multi-gene names keep their first part. The GO release behind
    the web service moves, so a rerun can label a node with a different top term than an earlier run."""
    from gprofiler import GProfiler
    gp = GProfiler(return_dataframe=True)
    out = {}
    for node in G.nodes:
        genes = list({g.split("+")[0] for g in G.nodes[node].get("genes", set())})
        if len(genes) < 5 or node[0] == 1:
            out[node] = None
            continue
        sources = ["GO:CC"] if node[0] != finest_level else ["GO:MF", "GO:BP"]
        try:
            out[node] = gp.profile(organism="hsapiens", query=genes, sources=sources,
                                   no_evidences=False, user_threshold=0.05)
        except Exception as exc:
            print(f"  [warn] g:Profiler failed for node {node} ({len(genes)} genes): {exc}; left unlabelled", flush=True)
            out[node] = None
    return out


# ------------------------------------------------------------------ renderer 1: circle nodes (Fig 3C)
def render_circle_tree(G, enrich, out, pos) -> None:
    """Node size 400 + n genes, colour -log10 p of the top GO term, label '<level>.<id>: <term>'."""
    import math
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("Reds")
    truncated = mcolors.LinearSegmentedColormap.from_list("truncated_reds", cmap(np.linspace(0, 0.85, 256)))
    sizes, colors, label_dict = [], [], {}
    for node in G.nodes:
        sizes.append(400 + len(G.nodes[node].get("genes", [])))
        level = node[0]
        df = enrich.get(node)
        if df is not None and hasattr(df, "empty") and not df.empty:
            term = df.iloc[0]
            label_dict[node] = f"{level}.{node[1]}: {term['name']}"
            colors.append(max(0.0, min(50.0, -math.log10(max(float(term["p_value"]), 1e-300)))))
        else:
            label_dict[node] = f"{level}.{node[1]}: -"
            colors.append(0.0)
    plt.figure(figsize=(58, 22))
    nodes = nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, cmap=truncated)
    nx.draw_networkx_edges(G, pos, edge_color="#D3D3D3", width=2, arrows=False)
    last = max(n[0] for n in G.nodes)
    for key, (x, y) in pos.items():
        plt.text(x, y - 7.5, label_dict[key], fontsize=16, fontweight="bold", ha="left", va="top",
                 rotation=-90 if key[0] == last else -30)
    plt.colorbar(nodes, label="-log10(p-value)", fraction=0.01, location="right", aspect=30)
    plt.title("Hierarchical Leiden clustering tree with most enriched GO terms")
    plt.axis("off")
    save_figure(plt.gcf(), out, dpi=300)


# ------------------------------------------------------------------ renderer 2: heatmap nodes (consensus tree)
TILE_SIZE = 512                                  # heatmap tile, px (one plane of the 512 x 512 stack)
MIN_NODE_SIZE = 3                                # nodes with fewer genes are not drawn
IMAGE_ZOOM, DPI = 0.12, 150                      # tile zoom in the figure; output dpi
HEATMAP_CMAP, COHESION_CMAP = "hot", "YlOrRd"    # tile colours (0-1); frame colours (0-1)
NODESEP, RANKSEP = 3.0, 4.0                      # graphviz spacing before the rescale below
BORDER_WIDTH, LABEL_FONTSIZE = 60, 20            # cohesion frame, px; node label, pt


def _rrect_mask(H, W, r, yy, xx):
    r = max(r, 1)
    mask = np.ones((H, W), dtype=bool)
    for cy, cx, row_cond, col_cond in [(r, r, yy <= r, xx <= r), (r, W - r - 1, yy <= r, xx >= W - r - 1),
                                       (H - r - 1, r, yy >= H - r - 1, xx <= r),
                                       (H - r - 1, W - r - 1, yy >= H - r - 1, xx >= W - r - 1)]:
        mask[(row_cond & col_cond) & ~((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2)] = False
    return mask


def rounded_rect_mask(rgba, radius_frac=0.15):
    """Alpha-mask an RGBA tile to a rounded rectangle."""
    h, w = rgba.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    out = rgba.copy()
    out[~_rrect_mask(h, w, int(min(h, w) * radius_frac), yy, xx), 3] = 0.0
    return out


def cohesion_border(rgba, border_color, border_width, radius_frac=0.15):
    """A thick coloured rounded frame (with a thin white inner gap) around an RGBA tile."""
    h, w = rgba.shape[:2]
    white_bw = max(2, int(border_width * 0.3))
    total_bw = border_width + white_bw
    H2, W2 = h + 2 * total_bw, w + 2 * total_bw
    out = np.zeros((H2, W2, 4), dtype=np.float32)
    out[total_bw:total_bw + h, total_bw:total_bw + w] = rgba
    inner_r = int(min(h, w) * radius_frac)
    mid_r, outer_r = inner_r + white_bw, inner_r + white_bw + border_width
    yy, xx = np.ogrid[:H2, :W2]
    outer_mask = _rrect_mask(H2, W2, outer_r, yy, xx)
    mid_mask = np.zeros((H2, W2), dtype=bool)
    yy_m, xx_m = np.ogrid[:h + 2 * white_bw, :w + 2 * white_bw]
    mid_mask[border_width:border_width + h + 2 * white_bw, border_width:border_width + w + 2 * white_bw] = \
        _rrect_mask(h + 2 * white_bw, w + 2 * white_bw, mid_r, yy_m, xx_m)
    inner_mask = np.zeros((H2, W2), dtype=bool)
    yy_h, xx_w = np.ogrid[:h, :w]
    inner_mask[total_bw:total_bw + h, total_bw:total_bw + w] = _rrect_mask(h, w, inner_r, yy_h, xx_w)
    out[mid_mask & ~inner_mask] = [1.0, 1.0, 1.0, 1.0]
    out[outer_mask & ~mid_mask] = border_color
    return out


def render_heatmap_tree(G, heatmaps, out_png) -> dict:
    """Tree with a per-node heatmap tile framed in its co-clustering (cohesion) colour.

    Level 1 is the root (1, 0). Node labels are '<level>.<display id>' with display ids
    counted left to right in the layout; returns {node: display_id}."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    last = max(n[0] for n in G.nodes)
    H = G.subgraph([n for n in G.nodes if G.nodes[n].get("size", 0) >= MIN_NODE_SIZE]).copy()
    pos = dot_layout(H, NODESEP, RANKSEP)

    nodes_by_level = defaultdict(list)
    for node in H.nodes:
        nodes_by_level[node[0]].append(node)
    display_id = {}
    for level_nodes in nodes_by_level.values():
        for rank, n in enumerate(sorted(level_nodes, key=lambda n: pos[n][0]), start=1):
            display_id[n] = rank
    H.nodes[(1, 0)]["mean_consensus"] = 1.0                     # the root is fully cohesive by definition
    labels = {node: f"{node[0]}.{display_id[node]}" for node in H.nodes}

    # scale the dot layout so the tightest same-level gap equals one node step, levels one level step
    n_levels = len(nodes_by_level)
    tile_px = TILE_SIZE + 2 * (BORDER_WIDTH + max(2, int(BORDER_WIDTH * 0.3)))
    thumb_in = tile_px * IMAGE_ZOOM / DPI
    node_step_in, level_step_in = thumb_in * 2.9, thumb_in * 9.0 + 0.5
    min_x_gap = float("inf")
    for level_nodes in nodes_by_level.values():
        xs = sorted(pos[n][0] for n in level_nodes)
        for a, b in zip(xs, xs[1:]):
            min_x_gap = min(min_x_gap, b - a)
    x_scale = 1.0 if min_x_gap < 1e-6 else node_step_in / min_x_gap
    ys_all = [pos[n][1] for n in H.nodes]
    y_scale = (n_levels - 1) * level_step_in / max(max(ys_all) - min(ys_all), 1e-6)
    x_min, y_min = min(pos[n][0] for n in H.nodes), min(ys_all)
    pad_in = max(thumb_in, 0.5)
    pos = {n: ((px - x_min) * x_scale + pad_in, (py - y_min) * y_scale + pad_in) for n, (px, py) in pos.items()}
    extra = level_step_in * 0.10                                # +10% gap below levels 4 and 5
    for n, (px, py) in list(pos.items()):
        lv = n[0]
        pos[n] = (px, py + (2 * extra if lv <= 4 else extra if lv == 5 else 0.0))
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    fig_w, fig_h = max(xs) + pad_in, max(ys) + pad_in

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    half = thumb_in * 0.55
    ax.set_xlim(min(xs) - half, max(xs) + half); ax.set_ylim(min(ys) - half, max(ys) + half)
    child_sizes = [min(H.nodes[u].get("size", 1), H.nodes[v].get("size", 1)) for u, v in H.edges()]
    max_csize = max(child_sizes) if child_sizes else 1
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color="#D3D3D3", arrows=False, style="solid",
                           width=[3.0 + 35.0 * (cs / max_csize) for cs in child_sizes])
    cmap_obj, norm = plt.get_cmap(HEATMAP_CMAP), mcolors.Normalize(vmin=0, vmax=1)
    coh_cmap, coh_norm = plt.get_cmap(COHESION_CMAP), mcolors.Normalize(vmin=0, vmax=1)
    for node in H.nodes:
        hm = heatmaps.get(node)
        if hm is None:
            hm = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
        rgba = rounded_rect_mask(cmap_obj(norm(hm)))
        rgba = cohesion_border(rgba, np.array(coh_cmap(coh_norm(H.nodes[node].get("mean_consensus", 0.0)))), BORDER_WIDTH)
        box = OffsetImage(rgba, zoom=IMAGE_ZOOM, interpolation="antialiased")
        box.image.axes = ax
        ax.add_artist(AnnotationBbox(box, pos[node], frameon=False, pad=0.0))
    for node, (x, y) in pos.items():
        ax.text(x, y - thumb_in * 1.5, labels[node], fontsize=LABEL_FONTSIZE, ha="left", va="top",
                rotation=-90 if node[0] == last else -30)
    for cm, nrm, y, label in ((cmap_obj, norm, 0.85, "Co-occupancy fraction (images)"),
                              (coh_cmap, coh_norm, 0.90, "Co-clustering score (frames)")):
        cax = fig.add_axes([0.80, y, 0.15, 0.015])
        sm = plt.cm.ScalarMappable(cmap=cm, norm=nrm); sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=LABEL_FONTSIZE); cb.ax.tick_params(labelsize=LABEL_FONTSIZE - 2)
    ax.set_title("Hierarchy tree with per-cluster co-occupancy heatmaps", fontsize=14)
    ax.axis("off")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_png, dpi=DPI)
    return display_id


# ------------------------------------------------------------------ Fig 3D: 2x2 subcluster UMAPs
def subcluster_2x2(genes, umap_real, umap_gen, real_clusters, gen_clusters, level_num, real_col, gen_col,
                   out, xlim, ylim) -> None:
    """Rows: the embedding of the real / the simulated images; columns: coloured by the real / the simulated clustering.
    Both arms are drawn with scanpy's categorical palette in a shared frame (xlim, ylim)."""
    import anndata as ad
    import matplotlib.pyplot as plt
    import scanpy as sc
    n_real, n_gen = len(set(real_clusters)), len(set(gen_clusters))
    arms = []
    for xy in (umap_real, umap_gen):
        a = ad.AnnData(obs=pd.DataFrame({"gene_name": genes}, index=genes))
        a.obsm["X_umap"] = np.asarray(xy, np.float32)
        a.obs["real_clusters"] = pd.Categorical(real_clusters)
        a.obs["gen_clusters"] = pd.Categorical(gen_clusters)
        arms.append(a)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    sc.set_figure_params(vector_friendly=True, dpi=300)
    for r, c, a, color, legend in [(0, 0, arms[0], "real_clusters", "none"),
                                   (0, 1, arms[0], "gen_clusters", "right margin" if n_gen <= 15 else "none"),
                                   (1, 0, arms[1], "real_clusters", "none"),
                                   (1, 1, arms[1], "gen_clusters", "right margin" if n_gen <= 15 else "none")]:
        sc.pl.umap(a, color=color, ax=axes[r, c], show=False, title="", legend_loc=legend)
    for ax in axes.ravel():
        ax.set_xlabel(""); ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    axes[0, 0].set_title(f"Clustering based on real\n({n_real} clusters)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 1].set_title(f"Clustering based on generated\n({n_gen} clusters)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 0].set_ylabel("Real", fontsize=13, fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("Generated", fontsize=13, fontweight="bold", labelpad=10)
    fig.suptitle(f"Level {level_num}: {real_col} vs {gen_col}\n({len(real_clusters)} genes)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, out, dpi=300)
