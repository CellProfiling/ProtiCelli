"""
plot_cluster_heatmaps_v3.py
─────────────────────────────────────────────────────────────────────────────
Generates two figures per Leiden resolution level.

Layout change from v2:
  Rows  = clusters
  Cols  = cells  (30 cells total) 

Main figure
  • Only clusters C5, C6, C20, C21, C36, C37 (display_cluster_id)
  • No landmark row

Supplement figure
  • All clusters at the given resolution
  • Landmark row at the top (false-colour RGB: ch0→red, ch2→blue, ch3→yellow)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from skimage.filters import threshold_otsu
from tifffile import imread

# ─────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────       

# 30 cells: cycle the original 12 to fill the list
# _BASE_CELLS    = [2, 13, 14, 21, 24, 49,63,65, 168, 185, 177, 105, 106, 134, 126, 118,  50, 189,        
# 166, 120, 150, 114, 116,  17, 140, 70, 71, 3, 103, 128]
SELECTED_CELLS = [2, 13, 14, 21, 24, 49,63,65, 168, 185, 177, 105, 106, 134, 126, 118,  50, 189, 
166, 120, 150, 114, 116,  17, 140, 70, 71, 3, 103, 128]

# display_cluster_id values shown in the main figure
MAIN_CLUSTER_IDS = {5, 6, 20, 21, 36, 37}

# Resolution levels present in the TSV
RES_LEVELS = {
    '0.341':  'lvl5_res0.341',
    # '0.3426': 'lvl6_res0.3426',
}


TILE_SIZE      = 512
THRESHOLD_MULT = 1.25
 
# Tile sizes and DPI for each figure type
MAIN_DISPLAY_PX = 500    # px per tile (rendered)
MAIN_DPI        = 900
 
SUPP_DISPLAY_PX = 120    # smaller to keep supplement manageable
SUPP_DPI        = 150
 
# Typography
FONT_CORNER   = 5
FONT_COL_HDR  = 5    # cell-id column headers (rotated)
FONT_ROW_MAIN = 7    # cluster row labels — main figure
FONT_ROW_SUPP = 6    # cluster row labels — supplement
FONT_MAXVAL   = 4
 
# Paths
STACK_DIR   = '/scratch/users/sunh/cytoVL'
REF_DIR     = '/scratch/groups/emmalu/multimodal_phenotyping/dataset/U2OS'
PROTEIN_PKL = 'protein_order.pkl'
CLUSTER_TSV = 'cluster_order.tsv'
OUT_DIR     = 'cluster_heatmap_plots_v3'
 
# Landmark channel mapping: (ref_channel_index, rgb_colour)
LANDMARK_CHANNELS = [
    (0, 'red'),
    (2, 'blue'),
    (3, 'yellow'),
]
LANDMARK_LEGEND = [
    ('Microtub.', '#ff4444'),
    ('Nucleus',   '#4488ff'),
    ('ER',        '#ffee00'),
]
 
os.makedirs(OUT_DIR, exist_ok=True)
 
# ─────────────────────────────────────────────────────────────────
#  Load shared data
# ─────────────────────────────────────────────────────────────────
print("Loading protein order …")
gene_order       = pickle.load(open(PROTEIN_PKL, 'rb'))
gene_order_lower = [g.lower() for g in gene_order]
 
print("Loading cluster TSV …")
cluster_df = pd.read_csv(CLUSTER_TSV, sep='\t', dtype=str)
cluster_df.columns = cluster_df.columns.str.strip()
cluster_df['display_cluster_id']  = cluster_df['display_cluster_id'].astype(int)
cluster_df['original_cluster_id'] = cluster_df['original_cluster_id'].astype(int)
cluster_df['resolution']          = cluster_df['resolution'].str.strip()
cluster_df = cluster_df[cluster_df['resolution'] != 'root'].reset_index(drop=True)
 
# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────
def find_gene_idx(gene_name: str):
    g = gene_name.lower().strip()
    for i, ref in enumerate(gene_order_lower):
        if g == ref or g in ref:
            return i
    return None
 
 
def autobright(channel: np.ndarray,
               sat_lo: float = 0.1,
               sat_hi: float = 99.9,
               gamma:  float = 1.5) -> np.ndarray:
    lo = np.percentile(channel, sat_lo)
    hi = np.percentile(channel, sat_hi)
    if hi <= lo:
        hi = lo + 1.0
    stretched = np.clip((channel.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return stretched ** gamma
 
 
def make_landmark_rgb(dapi_raw: np.ndarray) -> np.ndarray:
    """Build false-colour RGB from (h, w, 3) ref-channel crop."""
    h, w = dapi_raw.shape[:2]
    r = np.zeros((h, w), dtype=np.float32)
    g = np.zeros((h, w), dtype=np.float32)
    b = np.zeros((h, w), dtype=np.float32)
    for local_idx, (_, colour) in enumerate(LANDMARK_CHANNELS):
        if local_idx >= dapi_raw.shape[2]:
            continue
        ch = autobright(dapi_raw[:, :, local_idx])
        if colour == 'red':
            r = np.clip(r + ch, 0.0, 1.0)
        elif colour == 'blue':
            b = np.clip(b + ch, 0.0, 1.0)
        elif colour == 'yellow':
            r = np.clip(r + ch, 0.0, 1.0)
            g = np.clip(g + ch, 0.0, 1.0)
    return (np.clip(np.stack([r, g, b], axis=2), 0.0, 1.0) * 255).astype(np.uint8)
 
 
def compute_heatmap(stack: np.ndarray, indices: list) -> np.ndarray:
    valid = [i for i in indices if i < stack.shape[2]]
    if not valid:
        return np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
    planes = stack[:TILE_SIZE, :TILE_SIZE, valid].astype(np.float32)
    binary = np.stack([
        (planes[:, :, k] > THRESHOLD_MULT * threshold_otsu(planes[:, :, k]))
        for k in range(planes.shape[2])
    ], axis=2)
    return binary.mean(axis=2).astype(np.float32)
 
 
def genes_to_indices(genes_str: str) -> list:
    genes = [g.strip() for g in genes_str.split(',') if g.strip()]
    return [i for i in (find_gene_idx(g) for g in genes) if i is not None]
 
 
def _spine_style(ax, color='#444444', lw=0.3):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(lw)
 
# ─────────────────────────────────────────────────────────────────
#  Pre-cache stacks  (load each unique cell once)
# ─────────────────────────────────────────────────────────────────
print("\nPre-loading cell stacks and reference images …")
unique_cells = list(dict.fromkeys(SELECTED_CELLS))
stacks: dict = {}
refs:   dict = {}
for cell_id in unique_cells:
    sp = os.path.join(STACK_DIR,
                      f'U2OS_superplexed_{cell_id:03d}_stacked_images.npy')
    if os.path.exists(sp):
        stacks[cell_id] = np.load(sp, mmap_mode='r')
        print(f"  ✓  Stack  Cell {cell_id}")
    else:
        stacks[cell_id] = None
        print(f"  ✗  Stack  Cell {cell_id}  (not found)")
 
    ref_path = os.path.join(REF_DIR, f'U2OS_image_{cell_id}.tiff')
    if os.path.exists(ref_path):
        ref_img = imread(ref_path)
        refs[cell_id] = ref_img[:TILE_SIZE, :TILE_SIZE, [0, 2, 3]].astype(np.float32)
        print(f"  ✓  Ref    Cell {cell_id}")
    else:
        refs[cell_id] = None
        print(f"  ✗  Ref    Cell {cell_id}  (not found)")
 
norm_cb = mpl.colors.Normalize(vmin=0, vmax=1)
cmap_hot = mpl.cm.hot
 
# ─────────────────────────────────────────────────────────────────
#  Figure builder  (clusters as rows, cells as columns)
# ─────────────────────────────────────────────────────────────────
def build_figure(cluster_info: list,
                 selected_cells: list,
                 stacks: dict,
                 refs:   dict,
                 show_landmarks: bool,
                 level_label:    str,
                 dpi:        int,
                 display_px: int,
                 font_row:   int = 6) -> plt.Figure:
    """
    Parameters
    ----------
    cluster_info  : [(display_id, orig_id, gene_indices), …]  ordered by display_id
    selected_cells: ordered list of cell IDs (length = N_CELLS)
    show_landmarks: if True, add a landmark-image row above the cluster rows
    """
    N_CLUSTERS = len(cluster_info)
    N_CELLS    = len(selected_cells)
    TILE_IN    = display_px / dpi        # tile size in inches
 
    ROW_LABEL_W  = 0.75   # inches — cluster-label column
    COL_HEADER_H = 0.40   # inches — cell-id header row
    LM_ROW_SCALE = 1.0    # relative height of the landmark row vs a tile row
 
    N_LM_ROWS   = 1 if show_landmarks else 0
    N_DATA_ROWS = N_LM_ROWS + N_CLUSTERS
 
    FIG_W = ROW_LABEL_W + N_CELLS  * TILE_IN
    FIG_H = COL_HEADER_H + N_DATA_ROWS * TILE_IN
 
    # GridSpec dimensions
    n_gs_rows = 1 + N_DATA_ROWS          # header + data
    n_gs_cols = 1 + N_CELLS              # label  + cells
 
    height_ratios = (
        [COL_HEADER_H / TILE_IN]
        + ([LM_ROW_SCALE] if show_landmarks else [])
        + [1.0] * N_CLUSTERS
    )
    width_ratios = [ROW_LABEL_W / TILE_IN] + [1.0] * N_CELLS
 
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')
    gs  = gridspec.GridSpec(
        n_gs_rows, n_gs_cols,
        figure       = fig,
        left=0.0, right=1.0, top=1.0, bottom=0.0,
        hspace=0.02, wspace=0.02,
        width_ratios = width_ratios,
        height_ratios= height_ratios,
    )
 
    # ── Top-left corner (level label) ────────────────────────────
    ax_c = fig.add_subplot(gs[0, 0])
    ax_c.set_axis_off()
    ax_c.text(0.5, 0.5, level_label.replace('_', '\n'),
              transform=ax_c.transAxes, fontsize=FONT_CORNER,
              color='#777777', ha='center', va='center',
              fontfamily='DejaVu Sans', style='italic')
 
    # ── Cell column headers (row 0, cols 1..N_CELLS) ─────────────
    for col_idx, cell_id in enumerate(selected_cells):
        ax = fig.add_subplot(gs[0, col_idx + 1])
        ax.set_axis_off()
        ax.text(0.5, 0.1, f"Cell\n{cell_id}",
                transform=ax.transAxes,
                fontsize=FONT_COL_HDR, fontweight='bold',
                color='#222222', ha='center', va='bottom',
                fontfamily='DejaVu Sans', rotation=90)
 
    # ── Landmark row (gs row 1, all cell columns) ─────────────────
    cluster_gs_start = 1   # gs row index at which cluster rows begin
    if show_landmarks:
        cluster_gs_start = 2
 
        # Row label with colour-coded channel key
        ax_lm_lbl = fig.add_subplot(gs[1, 0])
        ax_lm_lbl.set_axis_off()
        for k, (label, colour) in enumerate(LANDMARK_LEGEND):
            ax_lm_lbl.text(
                0.5, 0.82 - k * 0.30, label,
                transform=ax_lm_lbl.transAxes,
                fontsize=font_row - 1, fontweight='bold',
                color=colour, ha='center', va='center',
                fontfamily='DejaVu Sans',
            )
 
        # Landmark image for each cell
        for col_idx, cell_id in enumerate(selected_cells):
            ax_lm = fig.add_subplot(gs[1, col_idx + 1])
            _spine_style(ax_lm, color='#666666', lw=0.4)
            dapi_raw = refs.get(cell_id)
            if dapi_raw is None:
                ax_lm.set_facecolor('#111111')
                ax_lm.text(0.5, 0.5, 'N/A', transform=ax_lm.transAxes,
                           fontsize=3, color='#888888', ha='center', va='center')
            else:
                ax_lm.imshow(make_landmark_rgb(dapi_raw),
                             aspect='auto', interpolation='bilinear')
 
    # ── Cluster rows ──────────────────────────────────────────────
    for row_idx, (disp_id, orig_id, indices) in enumerate(cluster_info):
        gs_row = row_idx + cluster_gs_start
 
        # Row label
        ax_lbl = fig.add_subplot(gs[gs_row, 0])
        ax_lbl.set_axis_off()
        n_prot = len(indices)
        ax_lbl.text(0.5, 0.5,
                    f"C{disp_id}\n({n_prot}p)",
                    transform=ax_lbl.transAxes,
                    fontsize=font_row, fontweight='bold',
                    color='#111111', ha='center', va='center',
                    fontfamily='DejaVu Sans')
 
        # One panel per cell
        for col_idx, cell_id in enumerate(selected_cells):
            ax = fig.add_subplot(gs[gs_row, col_idx + 1])
            _spine_style(ax)
 
            stack = stacks.get(cell_id)
            if stack is None:
                ax.set_facecolor('#111111')
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        fontsize=3, color='#888888', ha='center', va='center')
            else:
                heatmap = compute_heatmap(stack, indices)
                ax.set_facecolor('black')
                ax.imshow(heatmap, cmap='hot', vmin=0, vmax=1,
                          aspect='auto', interpolation='bilinear')
                if display_px >= 64:
                    ax.text(0.95, 0.04, f"{heatmap.max():.2f}",
                            transform=ax.transAxes, fontsize=FONT_MAXVAL,
                            color='white', ha='right', va='bottom',
                            fontfamily='DejaVu Sans',
                            bbox=dict(boxstyle='round,pad=0.06',
                                      fc='black', alpha=0.45, lw=0))
 
    # ── Colorbar ──────────────────────────────────────────────────
    cb_ax = fig.add_axes([
        1.004,
        COL_HEADER_H / FIG_H,
        0.010,
        (N_DATA_ROWS * TILE_IN) / FIG_H,
    ])
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_cb, cmap=cmap_hot),
        cax=cb_ax, orientation='vertical',
    )
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['0', '0.5', '1'], fontsize=5, color='black')
    cb.set_label('co-occupancy', fontsize=5, color='#555555',
                 fontfamily='DejaVu Sans', style='italic', labelpad=3)
    cb.outline.set_edgecolor('#aaaaaa')
    cb.outline.set_linewidth(0.4)
 
    return fig
 
# ─────────────────────────────────────────────────────────────────
#  Per-resolution-level loop
# ─────────────────────────────────────────────────────────────────
processed_resolutions = cluster_df['resolution'].unique()
 
for res_str, level_label in RES_LEVELS.items():
    if res_str not in processed_resolutions:
        print(f"\n⚠  Resolution {res_str} not found in TSV — skipping.")
        continue
 
    level_rows = (
        cluster_df[cluster_df['resolution'] == res_str]
        .sort_values('display_cluster_id')
        .reset_index(drop=True)
    )
 
    # Build full ordered cluster list for this resolution
    cluster_info_all = []
    for _, row in level_rows.iterrows():
        disp_id = int(row['display_cluster_id'])
        orig_id = int(row['original_cluster_id'])
        indices = genes_to_indices(row['genes'])
        cluster_info_all.append((disp_id, orig_id, indices))
 
    # Subset for main figure
    cluster_info_main = [
        (d, o, idx) for d, o, idx in cluster_info_all
        if d in MAIN_CLUSTER_IDS
    ]
 
    res_tag  = res_str.replace('.', 'p')
    N_ALL    = len(cluster_info_all)
    N_MAIN   = len(cluster_info_main)
 
    print(f"\n{'═'*65}")
    print(f"  Level : {level_label}  |  {N_ALL} total clusters")
 
    # ────────────────────────────────────────────────────────────
    #  Main figure
    #  Rows: C5, C6, C20, C21, C36, C37 only | no landmark row
    # ────────────────────────────────────────────────────────────
    if not cluster_info_main:
        print(f"  ⚠  None of {MAIN_CLUSTER_IDS} found — skipping main figure.")
    else:
        print(f"  Main : {N_MAIN} clusters × {len(SELECTED_CELLS)} cells  "
              f"(px={MAIN_DISPLAY_PX}, dpi={MAIN_DPI})")
        fig_main = build_figure(
            cluster_info   = cluster_info_main,
            selected_cells = SELECTED_CELLS,
            stacks         = stacks,
            refs           = refs,
            show_landmarks = False,
            level_label    = level_label,
            dpi            = MAIN_DPI,
            display_px     = MAIN_DISPLAY_PX,
            font_row       = FONT_ROW_MAIN,
        )
        out_main = os.path.join(
            OUT_DIR,
            f"MAIN_{level_label}_res{res_tag}_{N_MAIN}cl_30cells.pdf",
        )
        fig_main.savefig(out_main, dpi=MAIN_DPI, facecolor='white',
                         bbox_inches='tight', pad_inches=0.04)
        plt.close(fig_main)
        print(f"  ✓  Main  → {out_main}")
 
    # ────────────────────────────────────────────────────────────
    #  Supplement figure
    #  Rows: all clusters | landmark row at top
    # ────────────────────────────────────────────────────────────
    print(f"  Supp : {N_ALL} clusters × {len(SELECTED_CELLS)} cells  "
          f"(px={SUPP_DISPLAY_PX}, dpi={SUPP_DPI})")
    fig_supp = build_figure(
        cluster_info   = cluster_info_all,
        selected_cells = SELECTED_CELLS,
        stacks         = stacks,
        refs           = refs,
        show_landmarks = True,
        level_label    = level_label,
        dpi            = SUPP_DPI,
        display_px     = SUPP_DISPLAY_PX,
        font_row       = FONT_ROW_SUPP,
    )
    out_supp = os.path.join(
        OUT_DIR,
        f"SUPP_{level_label}_res{res_tag}_{N_ALL}cl_30cells.pdf",
    )
    fig_supp.savefig(out_supp, dpi=SUPP_DPI, facecolor='white',
                     bbox_inches='tight', pad_inches=0.04)
    plt.close(fig_supp)
    print(f"  ✓  Supp  → {out_supp}")
 
print("\n" + "═" * 65)
print("All levels done.")
print("═" * 65)
 