#!/usr/bin/env python3
"""
Hierarchy tree with per-cluster co-occupancy heatmap images as nodes.

Combines the hierarchy tree from the single-cell consensus pipeline with
per-cluster co-occupancy heatmaps (adapted from plot_leiden_maps.py).
Instead of colored circles, each tree node shows a 512x512 heatmap thumbnail
of binarized protein co-occupancy for that cluster's gene set.

Data flow per cell:
    cell_{ID}/cluster_labels_generated.tsv    protein_order.pkl (12800 genes)
      (7 Leiden levels)                        gene -> stack_index mapping
               |                                        |
    U2OS_superplexed_{ID}_stacked_images.npy (512x512x12800 uint8)
               |
    For each tree node:
      1. Get gene set from node attrs
      2. Map genes -> stack indices via protein_order
      3. Extract planes, binarize with Otsu x 1.25, average -> co-occupancy heatmap
      4. Apply 'hot' colormap -> RGBA thumbnail
      5. Place at node position using AnnotationBbox + OffsetImage

Author: Konstantin Kahnert

Usage:
    # Single cell:
    python plot_tree_with_heatmaps.py \\
        --cell-ids 000 \\
        --single-cell-dir .../single_cell_trees/U2OS \\
        --consensus-dir .../consensus_leiden/all_cells/uncentered/U2OS \\
        --base-dir /path/to/multi_cell \\
        --skip-enrichment

    # Multiple cells:
    python plot_tree_with_heatmaps.py \\
        --n-cells 5 \\
        --single-cell-dir .../single_cell_trees/U2OS \\
        --consensus-dir .../consensus_leiden/all_cells/uncentered/U2OS \\
        --base-dir /path/to/multi_cell \\
        --skip-enrichment
"""

import sys
import json
import pickle
import logging
import argparse
import gc
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.filters import threshold_otsu
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (from plot_leiden_maps.py)
# ---------------------------------------------------------------------------

TILE_SIZE = 512
THRESHOLD_MULT = 1.25


# ---------------------------------------------------------------------------
# Import shared functions
# ---------------------------------------------------------------------------

def _import_shared():
    """Import shared functions from utils/ sibling package."""
    utils_dir = str(Path(__file__).resolve().parent.parent / "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)

    from tree_helpers import (
        load_eval_cluster_labels,
        compute_cohesion_from_consensus,
        build_eval_hierarchy,
        run_tree_enrichment,
        discover_cell_ids,
        resolve_cell_ids,
    )
    return (
        load_eval_cluster_labels,
        compute_cohesion_from_consensus,
        build_eval_hierarchy,
        run_tree_enrichment,
        discover_cell_ids,
        resolve_cell_ids,
    )


# ---------------------------------------------------------------------------
# Protein order / stack helpers
# ---------------------------------------------------------------------------

def load_protein_order(pkl_path: Path) -> Dict[str, int]:
    """Load protein_order.pkl and return gene_name -> stack index mapping."""
    with open(pkl_path, "rb") as f:
        gene_order = pickle.load(f)
    return {g: i for i, g in enumerate(gene_order)}


def get_stack_path(stack_dir: Path, cell_line: str, cell_id: str) -> Path:
    """Build path to stacked_images.npy for a cell."""
    return stack_dir / f"{cell_line}_superplexed_{cell_id}_stacked_images.npy"


# ---------------------------------------------------------------------------
# Heatmap computation (adapted from plot_leiden_maps.py)
# ---------------------------------------------------------------------------

def compute_heatmap(stack: np.ndarray, gene_indices: List[int]) -> np.ndarray:
    """Compute co-occupancy heatmap for a set of genes.

    For each gene plane, binarize using Otsu threshold x THRESHOLD_MULT,
    then average all binary masks to get co-occupancy fraction (0-1).

    Args:
        stack: Memory-mapped array (H, W, N_genes), uint8.
        gene_indices: Indices into stack axis 2 for this cluster's genes.

    Returns:
        (TILE_SIZE, TILE_SIZE) float32 array, values in [0, 1].
    """
    valid = [i for i in gene_indices if i < stack.shape[2]]
    if not valid:
        return np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
    planes = stack[:TILE_SIZE, :TILE_SIZE, valid].astype(np.float32)
    binary = np.stack([
        (planes[:, :, k] > THRESHOLD_MULT * threshold_otsu(planes[:, :, k]))
        for k in range(planes.shape[2])
    ], axis=2)
    return binary.mean(axis=2).astype(np.float32)


def compute_all_heatmaps(
    G: nx.DiGraph,
    stack: np.ndarray,
    gene_to_idx: Dict[str, int],
) -> Dict[tuple, np.ndarray]:
    """Compute co-occupancy heatmaps for all nodes in the hierarchy graph.

    Args:
        G: Hierarchy DiGraph with 'genes' set in node attrs.
        stack: Memory-mapped stacked images array.
        gene_to_idx: Gene name -> stack index mapping.

    Returns:
        {node_key: heatmap_array} for every node in G.
    """
    heatmaps = {}
    total = G.number_of_nodes()
    for i, node in enumerate(G.nodes):
        genes = G.nodes[node].get("genes", set())
        indices = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        heatmaps[node] = compute_heatmap(stack, indices)
        if (i + 1) % 20 == 0 or (i + 1) == total:
            logger.info(f"  Heatmaps: {i + 1}/{total}")
    return heatmaps


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def apply_rounded_rect_mask(rgba: np.ndarray, radius_frac: float = 0.15) -> np.ndarray:
    """Apply a rounded rectangle alpha mask to an RGBA image.

    Args:
        rgba: (H, W, 4) float array in [0, 1].
        radius_frac: Corner radius as a fraction of the shorter side (default 0.15).

    Returns:
        Copy of rgba with alpha=0 outside the rounded rectangle.
    """
    h, w = rgba.shape[:2]
    r = int(min(h, w) * radius_frac)
    out = rgba.copy()

    # Build mask: True = inside rounded rect
    yy, xx = np.ogrid[:h, :w]
    corners = [
        (r,     r,     yy <= r,     xx <= r),      # top-left
        (r,     w-r-1, yy <= r,     xx >= w-r-1),  # top-right
        (h-r-1, r,     yy >= h-r-1, xx <= r),      # bottom-left
        (h-r-1, w-r-1, yy >= h-r-1, xx >= w-r-1),  # bottom-right
    ]
    mask = np.ones((h, w), dtype=np.float32)
    for cy, cx, row_cond, col_cond in corners:
        corner_pixels = row_cond & col_cond
        inside_circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        mask[corner_pixels & ~inside_circle] = 0.0

    out[~mask.astype(bool), 3] = 0.0
    return out


def add_cohesion_border(
    rgba: np.ndarray,
    border_color: np.ndarray,
    border_width: int,
    radius_frac: float = 0.15,
) -> np.ndarray:
    """Add a thick rounded-rectangle colored border around an RGBA image.

    Args:
        rgba: (H, W, 4) float RGBA, already rounded-rect masked.
        border_color: (4,) float RGBA color for the border.
        border_width: Border thickness in pixels.
        radius_frac: Corner radius fraction (same as apply_rounded_rect_mask).

    Returns:
        (H+2*total_bw, W+2*total_bw, 4) RGBA image with colored border ring.
    """
    h, w = rgba.shape[:2]
    white_bw = max(2, int(border_width * 0.3))
    total_bw = border_width + white_bw
    H2, W2 = h + 2 * total_bw, w + 2 * total_bw

    out = np.zeros((H2, W2, 4), dtype=np.float32)
    out[total_bw:total_bw + h, total_bw:total_bw + w] = rgba  # place original image in center

    inner_r = int(min(h, w) * radius_frac)
    mid_r = inner_r + white_bw
    outer_r = mid_r + border_width
    yy, xx = np.ogrid[:H2, :W2]

    def _rrect_mask(H, W, r, yy, xx):
        r = max(r, 1)
        corners = [
            (r,     r,     yy <= r,     xx <= r),
            (r,     W-r-1, yy <= r,     xx >= W-r-1),
            (H-r-1, r,     yy >= H-r-1, xx <= r),
            (H-r-1, W-r-1, yy >= H-r-1, xx >= W-r-1),
        ]
        mask = np.ones((H, W), dtype=bool)
        for cy, cx, row_cond, col_cond in corners:
            corner_pixels = row_cond & col_cond
            inside_circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
            mask[corner_pixels & ~inside_circle] = False
        return mask

    outer_mask = _rrect_mask(H2, W2, outer_r, yy, xx)

    mid_mask = np.zeros((H2, W2), dtype=bool)
    yy_m, xx_m = np.ogrid[:h + 2 * white_bw, :w + 2 * white_bw]
    mid_mask[border_width:border_width + h + 2 * white_bw, border_width:border_width + w + 2 * white_bw] = _rrect_mask(h + 2 * white_bw, w + 2 * white_bw, mid_r, yy_m, xx_m)

    # Build inner mask on full canvas
    inner_mask = np.zeros((H2, W2), dtype=bool)
    yy_h, xx_w = np.ogrid[:h, :w]
    inner_mask[total_bw:total_bw + h, total_bw:total_bw + w] = _rrect_mask(h, w, inner_r, yy_h, xx_w)

    # White border
    white_mask = mid_mask & ~inner_mask
    out[white_mask] = [1.0, 1.0, 1.0, 1.0]

    border_mask = outer_mask & ~mid_mask
    out[border_mask] = border_color

    return out


def _draw_colorbars(fig, cmap_obj, norm, cohesion_cmap_obj, cohesion_norm,
                    label_fontsize, images_above_nodes, node_color_only,
                    cohesion_border):
    """Draw compact horizontal colorbars using absolute figure coordinates."""
    cb_width = 0.15   # 15% of total figure width
    cb_height = 0.015  # Height of the bar
    cb_left = 0.80    # Anchor at 80% across the X axis

    def _add_bar(cmap, nrm, y, label):
        cax = fig.add_axes([cb_left, y, cb_width, cb_height])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=nrm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=label_fontsize)
        cb.ax.tick_params(labelsize=label_fontsize - 2)

    if images_above_nodes:
        _add_bar(cmap_obj, norm, 0.85, "Co-occupancy fraction (images)")
        _add_bar(cohesion_cmap_obj, cohesion_norm, 0.90, "Co-clustering score (nodes)")
    elif node_color_only:
        _add_bar(cohesion_cmap_obj, cohesion_norm, 0.88, "Co-clustering score (nodes)")
    else:
        _add_bar(cmap_obj, norm, 0.85, "Co-occupancy fraction (images)")
        if cohesion_border:
            _add_bar(cohesion_cmap_obj, cohesion_norm, 0.90, "Co-clustering score (frames)")


def visualize_tree_with_heatmaps(
    G: nx.DiGraph,
    heatmaps: Dict[tuple, np.ndarray],
    enrich_results: Dict,
    tuned_resolutions: List[float],
    output_path: Path,
    save_svg: bool = False,
    min_node_size: int = 3,
    image_zoom: float = 0.12,
    dpi: int = 150,
    heatmap_cmap: str = "hot",
    nodesep: float = 3.0,
    ranksep: float = 4.0,
    cluster_order_path: Optional[Path] = None,
    cohesion_cmap: str = "YlOrRd",
    border_width: int = 60,
    cohesion_border: bool = False,
    label_fontsize: int = 20,
    node_color_only: bool = False,
    images_above_nodes: bool = False,
) -> None:
    """Render hierarchy tree with heatmap images at each node position.

    Uses graphviz 'dot' layout with wide spacing, then places each node's
    heatmap as an AnnotationBbox with OffsetImage.

    Args:
        G: Hierarchy DiGraph.
        heatmaps: {node_key: (512,512) float32 array}.
        enrich_results: {node_key: DataFrame or None}.
        tuned_resolutions: 7 floats from cluster label columns.
        output_path: Where to save the PNG.
        save_svg: Also save SVG version.
        min_node_size: Skip nodes smaller than this.
        image_zoom: Zoom factor for OffsetImage (default 0.12 -> ~61px).
        dpi: Output DPI.
        heatmap_cmap: Colormap name for heatmaps.
        nodesep: Graphviz horizontal node separation.
        ranksep: Graphviz vertical rank separation.
        cluster_order_path: Optional path for cluster order TSV.
    """
    # Build level dict for labeling
    level_dict: Dict = {"root": 1}
    for i, res in enumerate(tuned_resolutions[1:]):
        if res not in level_dict:
            level_dict[res] = i + 2
    last_tuned_res = tuned_resolutions[-1] if len(tuned_resolutions) > 1 else None

    # Filter small nodes
    nodes_to_draw = [
        n for n in G.nodes if G.nodes[n].get("size", 0) >= min_node_size
    ]
    if not nodes_to_draw:
        logger.warning("No nodes above min_node_size threshold; drawing all nodes")
        nodes_to_draw = list(G.nodes)

    H = G.subgraph(nodes_to_draw).copy()
    if H.number_of_nodes() == 0:
        logger.warning("Empty graph, nothing to visualize")
        return

    # Compute layout
    try:
        pos = nx.nx_agraph.graphviz_layout(
            H, prog="dot",
            args=f"-Grankdir=TB -Gnodesep={nodesep} -Granksep={ranksep}",
        )
    except Exception:
        logger.warning("Graphviz not available, falling back to spring layout")
        pos = nx.spring_layout(H, seed=42)

    # Build per-level display IDs (left-to-right by x position)
    nodes_by_level: Dict = defaultdict(list)
    for node in H.nodes:
        nodes_by_level[node[0]].append(node)

    display_id: Dict = {}
    for res_key, level_nodes in nodes_by_level.items():
        sorted_nodes = sorted(level_nodes, key=lambda n: pos[n][0])
        for rank, n in enumerate(sorted_nodes, start=1):
            display_id[n] = rank
            
            # Force root node (1.1) to have a cohesion score of 1.0
            if level_dict.get(n[0], "?") == 1 and rank == 1:
                H.nodes[n]["mean_consensus"] = 1.0

    # Build labels
    label_dict = {}
    for node in H.nodes:
        res = node[0]
        level = level_dict.get(res, "?")
        did = display_id[node]
        df = enrich_results.get(node)
        if df is not None and hasattr(df, "empty") and not df.empty:
            name = df.iloc[0]['name'].replace("microtubule cytoskeleton", "MT cytoskeleton")
            label_dict[node] = f"{level}.{did}: {name}"
        else:
            label_dict[node] = f"{level}.{did}: -"

    # Save cluster order TSV
    if cluster_order_path is not None:
        rows = []
        for res_key, level_nodes in nodes_by_level.items():
            sorted_nodes = sorted(level_nodes, key=lambda n: pos[n][0])
            lv = level_dict.get(res_key, "?")
            for rank, n in enumerate(sorted_nodes, start=1):
                genes_sorted = sorted(H.nodes[n].get("genes", set()))
                rows.append({
                    "level": lv,
                    "resolution": res_key,
                    "display_cluster_id": rank,
                    "original_cluster_id": n[1],
                    "size": H.nodes[n].get("size", 0),
                    "genes": ",".join(genes_sorted),
                })
        pd.DataFrame(rows).sort_values(["level", "display_cluster_id"]).to_csv(
            cluster_order_path, sep="\t", index=False
        )
        logger.info(f"Cluster order saved to: {cluster_order_path}")

    # Keep graphviz layout intact — just scale x until the minimum
    # center-to-center distance between adjacent nodes at any level
    # is at least node_step_in (thumbnail width + gap).
    from collections import Counter
    level_sizes = Counter(n[0] for n in H.nodes)
    n_levels = len(level_sizes)

    if cohesion_border:
        total_bw = border_width + max(2, int(border_width * 0.3))
        tile_px = TILE_SIZE + 2 * total_bw
    else:
        tile_px = TILE_SIZE
    thumb_in = tile_px * image_zoom / dpi          # thumbnail side in inches
    node_step_multiplier = 2.9 if cohesion_border else 2.7
    node_step_in = thumb_in * node_step_multiplier  # minimum center-to-center (inches)
    level_step_in = thumb_in * 9.24 + 0.385 if images_above_nodes else thumb_in * 9.0 + 0.5  # vertical spacing

    # Find the minimum x gap between adjacent same-level nodes in graphviz coords
    min_x_gap = float("inf")
    for level_nodes in nodes_by_level.values():
        xs_level = sorted(pos[n][0] for n in level_nodes)
        for a, b in zip(xs_level, xs_level[1:]):
            min_x_gap = min(min_x_gap, b - a)

    # Scale factor: stretch x so the tightest gap equals node_step_in
    if min_x_gap < 1e-6:
        x_scale = 1.0
    else:
        x_scale = node_step_in / min_x_gap

    # Also scale y so level spacing matches level_step_in
    ys_all = [pos[n][1] for n in H.nodes]
    y_range_raw = max(ys_all) - min(ys_all)
    y_scale = (n_levels - 1) * level_step_in / max(y_range_raw, 1e-6)

    xs_all = [pos[n][0] for n in H.nodes]
    x_min_raw = min(xs_all)
    y_min_raw = min(ys_all)
    pad_in = max(thumb_in, 0.5)

    pos = {
        n: (
            (px - x_min_raw) * x_scale + pad_in,
            (py - y_min_raw) * y_scale + pad_in,
        )
        for n, (px, py) in pos.items()
    }

    # Extra vertical gap between levels 4-5 and 5-6 (+10% each)
    _extra_gap = level_step_in * 0.10
    for n in list(pos.keys()):
        lv = level_dict.get(n[0], 1)
        px, py = pos[n]
        if lv <= 4:
            pos[n] = (px, py + 2 * _extra_gap)
        elif lv == 5:
            pos[n] = (px, py + _extra_gap)

    xs_scaled = [p[0] for p in pos.values()]
    ys_scaled = [p[1] for p in pos.values()]

    # Pre-compute per-level image offsets (needed for fig_h and ylim before drawing).
    # Each level's offset is set by its largest node so all images in a row are aligned.
    if images_above_nodes:
        node_scatter = {n: 400 + H.nodes[n].get("size", 1) for n in H.nodes}
        level_img_offset = {}
        for level_key, level_nodes in nodes_by_level.items():
            max_S = max(node_scatter[n] for n in level_nodes)
            level_img_offset[level_key] = (max_S / np.pi) ** 0.5 / 72.0 + thumb_in * 1.6
        max_img_offset = max(level_img_offset.values())
    else:
        node_scatter = {}
        level_img_offset = {}
        max_img_offset = 0.0

    fig_w = max(xs_scaled) + pad_in
    fig_h = max(ys_scaled) + pad_in
    if images_above_nodes:
        fig_h += max_img_offset + thumb_in * 0.6 + thumb_in * 1.5  # image top + bottom label room

    logger.info(
        f"Figure size: {fig_w:.1f} x {fig_h:.1f} inches at {dpi} DPI "
        f"= {int(fig_w*dpi)} x {int(fig_h*dpi)} px "
        f"({H.number_of_nodes()} nodes, {H.number_of_edges()} edges)"
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    half = thumb_in * 0.55
    y_top_extra = (max_img_offset + thumb_in * 0.6) if images_above_nodes else 0.0
    y_bottom_extra = thumb_in * 1.5 if images_above_nodes else 0.0
    ax.set_xlim(min(xs_scaled) - half, max(xs_scaled) + half)
    ax.set_ylim(min(ys_scaled) - half - y_bottom_extra, max(ys_scaled) + half + y_top_extra)

    # 1. Draw edges (behind images)
    if images_above_nodes:
        # Uniform edge width for images_above mode
        edge_widths = [5.0] * len(H.edges())
    else:
        # Size-mapped edge width for border and other modes
        child_sizes = [min(H.nodes[u].get("size", 1), H.nodes[v].get("size", 1)) for u, v in H.edges()]
        max_csize = max(child_sizes) if child_sizes else 1
        edge_widths = [3.0 + 35.0 * (cs / max_csize) for cs in child_sizes]
    nx.draw_networkx_edges(
        H, pos, ax=ax,
        edge_color="#D3D3D3", width=edge_widths, arrows=False,
        style="solid",
    )

    # 2. Place heatmap images at node positions
    cmap_obj = plt.get_cmap(heatmap_cmap)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cohesion_cmap_obj = plt.get_cmap(cohesion_cmap)
    cohesion_norm = mcolors.Normalize(vmin=0, vmax=1)

    if images_above_nodes:
        # Same sizing as the original tree (400 + n_genes, no cap).
        # node_scatter and level_img_offset pre-computed above for layout.
        node_colors = [
            cohesion_cmap_obj(cohesion_norm(H.nodes[n].get("mean_consensus", 0.0)))
            for n in H.nodes
        ]
        node_sizes = [node_scatter[n] for n in H.nodes]
        nx.draw_networkx_nodes(
            H, pos, ax=ax,
            node_color=node_colors, node_size=node_sizes, alpha=1.0,
        )

        for node in H.nodes:
            hm = heatmaps.get(node)
            if hm is None:
                hm = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
            rgba = apply_rounded_rect_mask(cmap_obj(norm(hm)))
            imagebox = OffsetImage(rgba, zoom=image_zoom, interpolation="antialiased")
            imagebox.image.axes = ax
            ab = AnnotationBbox(
                imagebox,
                (pos[node][0], pos[node][1] + level_img_offset[node[0]]),
                frameon=False, pad=0.0,
            )
            ax.add_artist(ab)
    elif node_color_only:
        # Draw colored circles mapped to cohesion score
        node_colors = [
            cohesion_cmap_obj(cohesion_norm(H.nodes[n].get("mean_consensus", 0.0)))
            for n in H.nodes
        ]
        node_sizes = [400 + H.nodes[n].get("size", 1) for n in H.nodes]
        nx.draw_networkx_nodes(
            H, pos, ax=ax,
            node_color=node_colors, node_size=node_sizes, alpha=0.9,
        )
    else:
        for node in H.nodes:
            hm = heatmaps.get(node)
            if hm is None:
                hm = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)

            rgba = apply_rounded_rect_mask(cmap_obj(norm(hm)))

            if cohesion_border:
                cohesion_score = H.nodes[node].get("mean_consensus", 0.0)
                border_color = np.array(cohesion_cmap_obj(cohesion_norm(cohesion_score)))
                rgba = add_cohesion_border(rgba, border_color, border_width)

            imagebox = OffsetImage(rgba, zoom=image_zoom, interpolation="antialiased")
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, pos[node], frameon=False, pad=0.0)
            ax.add_artist(ab)

    # 3. Add text labels below each node
    for node, (x, y) in pos.items():
        label = label_dict.get(node, "")
        if images_above_nodes:
            # Per-level: label just below the largest node circle on this level
            y_offset = (node_scatter[node] / np.pi) ** 0.5 / 72.0 + thumb_in * 0.25
        else:
            y_offset = thumb_in * 1.5
        rotation = -90 if node[0] == last_tuned_res else -30
        ax.text(
            x, y - y_offset, label,
            fontsize=label_fontsize,
            ha="left", va="top", rotation=rotation,
        )

    # 4. Add compact horizontal colorbars using absolute figure coordinates
    _draw_colorbars(fig, cmap_obj, norm, cohesion_cmap_obj, cohesion_norm,
                    label_fontsize, images_above_nodes, node_color_only,
                    cohesion_border)

    ax.set_title(
        "Hierarchy tree with per-cluster co-occupancy heatmaps",
        fontsize=14,
    )
    ax.axis("off")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(str(output_path), format="png", dpi=dpi, bbox_inches="tight")
    logger.info(f"Tree visualization saved to: {output_path}")

    if save_svg:
        svg_path = output_path.with_suffix(".svg")
        plt.rcParams["svg.fonttype"] = "none"
        plt.savefig(str(svg_path), format="svg", bbox_inches="tight", dpi=300)
        logger.info(f"SVG saved to: {svg_path}")

    plt.close()


# ---------------------------------------------------------------------------
# Per-cell processing
# ---------------------------------------------------------------------------

def process_cell(
    cell_id: str,
    single_cell_dir: Path,
    consensus_dir: Path,
    all_gene_names: List[str],
    consensus_resolutions: List[float],
    gene_to_idx: Dict[str, int],
    stack_dir: Path,
    cell_line: str,
    output_dir_override: Optional[Path],
    viz_config,
    # Shared functions passed in:
    load_eval_cluster_labels,
    compute_cohesion_from_consensus,
    build_eval_hierarchy,
    run_tree_enrichment,
) -> bool:
    """Process one cell: build hierarchy, compute heatmaps, render tree.

    Returns True on success, False on error.
    """
    cell_dir = single_cell_dir / f"cell_{cell_id}"
    tsv_path = cell_dir / "cluster_labels_generated.tsv"

    if not tsv_path.exists():
        logger.error(f"Missing cluster labels for cell {cell_id}: {tsv_path}")
        return False

    # Output directory
    if output_dir_override:
        out_dir = output_dir_override / f"cell_{cell_id}"
    else:
        out_dir = cell_dir / "tree"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Processing cell {cell_id} ===")

    try:
        # Load cluster labels
        gene_names, cluster_assignments, level_cols, tuned_resolutions = \
            load_eval_cluster_labels(tsv_path)

        # Optionally limit the number of hierarchy levels to plot
        if viz_config.n_levels is not None:
            tuned_resolutions = tuned_resolutions[:viz_config.n_levels]
            level_cols = level_cols[:viz_config.n_levels]
            cluster_assignments = {
                k: v for k, v in cluster_assignments.items() if k in set(level_cols)
            }

        if viz_config.replot:
            # Fast path: load cached cohesion + enrichment + heatmaps
            cohesion_cache_path = out_dir / "cohesion_by_level_cache.json"
            enrich_cache_path = out_dir / "enrich_results_cache.pkl"
            if not cohesion_cache_path.exists() or not enrich_cache_path.exists():
                logger.error(
                    f"Replot caches missing for cell {cell_id}: "
                    f"run without --replot first."
                )
                return False
            logger.info(f"  --replot: loading cached cohesion")
            with open(cohesion_cache_path) as f:
                raw = json.load(f)
            cohesion_by_level = {
                col: {int(k): v for k, v in vals.items()}
                for col, vals in raw.items()
            }
            logger.info(f"  --replot: loading cached enrichment")
            with open(enrich_cache_path, "rb") as f:
                enrich_results = pickle.load(f)
        else:
            # Full path: compute cohesion from consensus matrices
            logger.info(f"  Computing cluster cohesion from consensus matrices ...")
            cohesion_by_level = compute_cohesion_from_consensus(
                cluster_assignments=cluster_assignments,
                level_cols=level_cols,
                gene_names=gene_names,
                consensus_dir=consensus_dir,
                consensus_resolutions=consensus_resolutions,
                all_gene_names=all_gene_names,
            )

            # Save cohesion cache
            cohesion_cache = {
                level_col: {str(k): v for k, v in cohesions.items()}
                for level_col, cohesions in cohesion_by_level.items()
            }
            cohesion_cache_path = out_dir / "cohesion_by_level_cache.json"
            with open(cohesion_cache_path, "w") as f:
                json.dump(cohesion_cache, f)
            logger.info(f"  Cohesion cache saved to: {cohesion_cache_path}")

        # Build hierarchy
        logger.info(f"  Building hierarchy graph ...")
        G = build_eval_hierarchy(
            gene_names=gene_names,
            cluster_assignments=cluster_assignments,
            level_cols=level_cols,
            tuned_resolutions=tuned_resolutions,
            cohesion_by_level=cohesion_by_level,
            overlap_threshold=viz_config.overlap_threshold,
        )

        if not viz_config.replot:
            # GO enrichment
            enrich_results = run_tree_enrichment(
                G, tuned_resolutions, skip_enrichment=viz_config.skip_enrichment,
            )

            # Save enrichment cache
            enrich_cache_path = out_dir / "enrich_results_cache.pkl"
            with open(enrich_cache_path, "wb") as f:
                pickle.dump(enrich_results, f)
            logger.info(f"  Enrichment cache saved to: {enrich_cache_path}")

        # Load or compute heatmaps
        heatmap_cache_path = out_dir / "heatmaps_cache.pkl"
        if viz_config.replot and heatmap_cache_path.exists():
            logger.info(f"  --replot: loading cached heatmaps from {heatmap_cache_path}")
            with open(heatmap_cache_path, "rb") as f:
                heatmaps = pickle.load(f)
        else:
            stack_path = get_stack_path(stack_dir, cell_line, cell_id)
            if not stack_path.exists():
                logger.error(f"Stack not found: {stack_path}")
                return False
            logger.info(f"  Loading stack: {stack_path}")
            stack = np.load(str(stack_path), mmap_mode="r")
            logger.info(f"  Stack shape: {stack.shape}")

            logger.info(f"  Computing heatmaps for {G.number_of_nodes()} nodes ...")
            heatmaps = compute_all_heatmaps(G, stack, gene_to_idx)
            del stack
            gc.collect()

            with open(heatmap_cache_path, "wb") as f:
                pickle.dump(heatmaps, f)
            logger.info(f"  Heatmap cache saved to: {heatmap_cache_path}")

        # Render tree with heatmap images
        logger.info(f"  Rendering tree with heatmap images ...")
        level_suffix = f"_{viz_config.n_levels}levels" if viz_config.n_levels is not None else ""
        if viz_config.images_above_nodes:
            base_name = "hierarchy_tree_nodes_images_above"
        elif viz_config.node_color_only:
            base_name = "hierarchy_tree_consensus"
        elif viz_config.cohesion_border:
            base_name = "hierarchy_tree_heatmaps_border"
        else:
            base_name = "hierarchy_tree_heatmaps"
        tree_output_path = out_dir / f"{base_name}{level_suffix}_{viz_config.cohesion_cmap}.png"
        visualize_tree_with_heatmaps(
            G=G,
            heatmaps=heatmaps,
            enrich_results=enrich_results,
            tuned_resolutions=tuned_resolutions,
            output_path=tree_output_path,
            save_svg=viz_config.save_svg,
            min_node_size=viz_config.min_node_size,
            image_zoom=viz_config.image_zoom,
            dpi=viz_config.dpi,
            heatmap_cmap=viz_config.heatmap_cmap,
            nodesep=viz_config.nodesep,
            ranksep=viz_config.ranksep,
            cluster_order_path=out_dir / "cluster_order.tsv",
            cohesion_cmap=viz_config.cohesion_cmap,
            border_width=viz_config.border_width,
            cohesion_border=viz_config.cohesion_border,
            label_fontsize=viz_config.label_fontsize,
            node_color_only=viz_config.node_color_only,
            images_above_nodes=viz_config.images_above_nodes,
        )

        # Save cohesion stats
        stats = {}
        for level_col, cohesions in cohesion_by_level.items():
            vals = list(cohesions.values())
            stats[level_col] = {
                "n_clusters": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        stats_path = out_dir / "cohesion_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"  Cohesion stats saved to: {stats_path}")

        # Clean up
        del heatmaps
        gc.collect()

        logger.info(f"  Done. Outputs in: {out_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to process cell {cell_id}: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hierarchy tree with per-cluster co-occupancy heatmap images as nodes"
        )
    )

    # Cell selection
    cell_group = parser.add_mutually_exclusive_group(required=True)
    cell_group.add_argument(
        "--cell-ids", type=str, nargs="+", metavar="ID",
        help="Explicit cell IDs to process (e.g. 000 006 062)",
    )
    cell_group.add_argument(
        "--n-cells", type=int, metavar="N",
        help="Sample N cells randomly from available cell directories",
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for --n-cells sampling (default: 42)",
    )
    parser.add_argument(
        "--single-cell-dir", type=str, required=True,
        help="Path to single_cell_trees/{CELL_LINE}/ directory",
    )
    parser.add_argument(
        "--consensus-dir", type=str, required=True,
        help="Path to consensus_leiden/all_cells/uncentered/{CELL_LINE}/ directory",
    )
    parser.add_argument(
        "--base-dir", type=str, required=True,
        help="Multi-cell base directory (for loading consensus gene names)",
    )
    parser.add_argument(
        "--cell-line", type=str, default="U2OS",
        help="Cell line name (default: U2OS)",
    )
    parser.add_argument(
        "--consensus-resolutions", type=float, nargs="+",
        default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5],
        help="6 consensus resolutions mapping to Leiden levels 2-7",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory root (each cell -> <output-dir>/cell_{ID}/)",
    )
    parser.add_argument(
        "--overlap-threshold", type=float, default=0.1,
        help="Min fractional overlap for parent-child edges (default: 0.1)",
    )
    parser.add_argument(
        "--min-node-size", type=int, default=3,
        help="Minimum cluster size to display in tree (default: 3)",
    )
    parser.add_argument(
        "--skip-enrichment", action="store_true",
        help="Skip GO enrichment (faster; labels show cluster size only)",
    )
    parser.add_argument("--save-svg", action="store_true", help="Also save SVG")
    parser.add_argument(
        "--replot", action="store_true",
        help="Replot from cached cohesion + enrichment (skip consensus matrix loading)",
    )

    # Heatmap / stack args
    parser.add_argument(
        "--stack-dir", type=str, required=True,
        help="Directory containing *_stacked_images.npy files",
    )
    parser.add_argument(
        "--protein-order", type=str, required=True,
        help="Path to protein_order.pkl (gene -> stack index mapping)",
    )
    parser.add_argument(
        "--image-zoom", type=float, default=0.12,
        help="Zoom factor for heatmap thumbnails (default: 0.12 -> ~61px)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output DPI (default: 150)",
    )
    parser.add_argument(
        "--heatmap-cmap", type=str, default="hot",
        help="Colormap for heatmaps (default: hot)",
    )
    parser.add_argument(
        "--nodesep", type=float, default=3.0,
        help="Graphviz horizontal node separation (default: 3.0)",
    )
    parser.add_argument(
        "--ranksep", type=float, default=4.0,
        help="Graphviz vertical rank separation (default: 4.0)",
    )
    parser.add_argument(
        "--n-levels", type=int, default=None,
        help="Limit number of hierarchy levels to plot (default: all). "
             "Levels are counted from root; e.g. --n-levels 6 plots levels 1-6.",
    )
    parser.add_argument(
        "--label-fontsize", type=int, default=20,
        help="Font size for node labels (default: 20)",
    )
    parser.add_argument(
        "--node-color-only", action="store_true",
        help="Draw tree with nodes colored by cohesion score instead of heatmap images",
    )
    parser.add_argument(
        "--images-above-nodes", action="store_true",
        help="Draw colored circle nodes (cohesion) with heatmap images placed above each node",
    )
    parser.add_argument(
        "--cohesion-border", action="store_true",
        help="Show cohesion score as a colored rounded-rect border around each node image",
    )
    parser.add_argument(
        "--cohesion-cmap", type=str, default="YlOrRd",
        help="Colormap for cohesion border (default: YlOrRd)",
    )
    parser.add_argument(
        "--border-width", type=int, default=60,
        help="Border width in pixels in source image (default: 60; at zoom=0.12 -> ~7px)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if len(args.consensus_resolutions) != 6:
        parser.error(
            f"--consensus-resolutions must have exactly 6 values, "
            f"got {len(args.consensus_resolutions)}"
        )

    single_cell_dir = Path(args.single_cell_dir)
    consensus_dir = Path(args.consensus_dir)
    base_dir = Path(args.base_dir)
    cell_line = args.cell_line
    stack_dir = Path(args.stack_dir)
    protein_order_path = Path(args.protein_order)
    output_dir_override = Path(args.output_dir) if args.output_dir else None

    from types import SimpleNamespace
    viz_config = SimpleNamespace(
        skip_enrichment=args.skip_enrichment, save_svg=args.save_svg,
        overlap_threshold=args.overlap_threshold, min_node_size=args.min_node_size,
        replot=args.replot, image_zoom=args.image_zoom, dpi=args.dpi,
        heatmap_cmap=args.heatmap_cmap, nodesep=args.nodesep, ranksep=args.ranksep,
        n_levels=args.n_levels, cohesion_cmap=args.cohesion_cmap,
        border_width=args.border_width, cohesion_border=args.cohesion_border,
        label_fontsize=args.label_fontsize, node_color_only=args.node_color_only,
        images_above_nodes=args.images_above_nodes,
    )

    if not single_cell_dir.exists():
        parser.error(f"--single-cell-dir does not exist: {single_cell_dir}")
    if not args.replot and not consensus_dir.exists():
        parser.error(f"--consensus-dir does not exist: {consensus_dir}")
    if not protein_order_path.exists():
        parser.error(f"--protein-order does not exist: {protein_order_path}")

    # ---- Load shared functions ----
    (
        load_eval_cluster_labels,
        compute_cohesion_from_consensus,
        build_eval_hierarchy,
        run_tree_enrichment,
        discover_cell_ids,
        resolve_cell_ids,
    ) = _import_shared()

    # ---- Resolve cell IDs ----
    try:
        cell_ids = resolve_cell_ids(
            single_cell_dir=single_cell_dir,
            cell_ids=args.cell_ids,
            n_cells=args.n_cells,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))

    logger.info(f"Processing {len(cell_ids)} cell(s): {cell_ids}")

    # ---- Load consensus gene names (not needed in replot mode) ----
    sys.path.insert(0, str(Path(__file__).parent))
    cell_line_dir = base_dir / cell_line
    if not args.replot:
        logger.info("Loading consensus gene names from first cell h5ad ...")
        from analyze_consensus_leiden import load_gene_names_from_first_cell
        all_gene_names = load_gene_names_from_first_cell(cell_line_dir)
        logger.info(f"Loaded {len(all_gene_names)} gene names")
    else:
        all_gene_names = []

    # ---- Load protein order ----
    logger.info(f"Loading protein order from {protein_order_path} ...")
    gene_to_idx = load_protein_order(protein_order_path)
    logger.info(f"Protein order: {len(gene_to_idx)} genes")

    # ---- Process each cell ----
    n_success = 0
    n_failed = 0

    for cell_id in cell_ids:
        ok = process_cell(
            cell_id=cell_id,
            single_cell_dir=single_cell_dir,
            consensus_dir=consensus_dir,
            all_gene_names=all_gene_names,
            consensus_resolutions=args.consensus_resolutions,
            gene_to_idx=gene_to_idx,
            stack_dir=stack_dir,
            cell_line=cell_line,
            output_dir_override=output_dir_override,
            viz_config=viz_config,
            load_eval_cluster_labels=load_eval_cluster_labels,
            compute_cohesion_from_consensus=compute_cohesion_from_consensus,
            build_eval_hierarchy=build_eval_hierarchy,
            run_tree_enrichment=run_tree_enrichment,
        )
        if ok:
            n_success += 1
        else:
            n_failed += 1

    logger.info(
        f"Finished: {n_success} succeeded, {n_failed} failed "
        f"(total {len(cell_ids)})"
    )

    if n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
