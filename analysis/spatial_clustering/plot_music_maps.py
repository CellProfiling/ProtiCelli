import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib as mpl
import pandas as pd 
import pickle
import os 
from skimage.filters import threshold_otsu
from tifffile import imread

# ─────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────
CELL_ID          = 153 # 122, 166, 186, 174, 193, 71, 49, 98, 73, 153
TILE_SIZE        = 512           # native image crop
DISPLAY_PX       = 256           # rendered tile size in pixels → compact
N_REPRESENTATIVE = 5
THRESHOLD_MULT   = 1.5
DPI              = 300

# ── Scale bar ──────────────────────────────────────────────────────
SHOW_SCALE_BAR   = False          # set False to hide scale bar on leftmost panel
SCALE_BAR_UM     = 10            # physical length in µm
SCALE_BAR_PX     = 93.721        # corresponding pixels in the native 512×512 image
                                  # (pixel size = 0.1067 µm/px → 10 µm / 0.1067 ≈ 93.7 px)
SCALE_BAR_COLOR  = 'white'
SCALE_BAR_HEIGHT = 15             # bar thickness in native pixels (dense, like reference)
SCALE_BAR_MARGIN = 18            # distance from bottom-left corner in native pixels

FONT_PROTEIN  = 10
FONT_TITLE    = 12
FONT_CBAR     = 9
FONT_BADGE    = 8
       
REF_PATH   = f'/scratch/groups/emmalu/multimodal_phenotyping/dataset/U2OS/U2OS_image_{CELL_ID}.tiff'
STACK_PATH = f'/scratch/users/sunh/cytoVL/U2OS_superplexed_{CELL_ID:03d}_stacked_images.npy'
OUT_DIR    = f'music_map_plots/U2OS_{CELL_ID:03d}'
os.makedirs(OUT_DIR, exist_ok=True)          

# ─────────────────────────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────────────────────────
gene_order       = pickle.load(open('protein_order.pkl', 'rb'))
gene_order_lower = [g.lower() for g in gene_order]

img_stack_original = np.load(STACK_PATH)
ref      = imread(REF_PATH)
dapi_raw = ref[:TILE_SIZE, :TILE_SIZE, 2].astype(np.float32)
systems_data = pd.read_csv('U2OS Cell Map Assemblies.csv')

# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────
def autobright(ch, sat_lo=0.1, sat_hi=99.9, gamma=1.5):
    lo, hi = np.percentile(ch, sat_lo), np.percentile(ch, sat_hi)
    if hi <= lo: hi = lo + 1
    return (np.clip((ch.astype(np.float32) - lo) / (hi - lo), 0, 1) ** gamma * 255).astype(np.uint8)

def snr_cv(ch):
    ch = ch.astype(np.float32)
    m = ch.mean()
    return ch.std() / m if m > 1e-6 else 0.0

def make_tile(blue_u8, green_u8):
    t = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    t[:, :, 2] = blue_u8
    t[:, :, 1] = green_u8
    return t

def find_idx(protein):
    p = protein.lower()
    for i, g in enumerate(gene_order_lower):
        if p in g:
            return i
    return None

def sanitise(name):
    return "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()

def draw_scale_bar(ax, tile_size=TILE_SIZE):
    """Draw a solid scale bar on ax in the bottom-left corner.

    The bar is drawn in data coordinates (0..tile_size), so it scales
    correctly regardless of the rendered display size.
    """
    if not SHOW_SCALE_BAR:
        return

    m   = SCALE_BAR_MARGIN            # margin from corner (native px)
    bar_len = SCALE_BAR_PX            # bar length in native px
    bar_h   = SCALE_BAR_HEIGHT        # bar height in native px

    # bottom-left anchor
    x0 = m
    y0 = tile_size - m - bar_h        # image origin is top-left in imshow

    rect = mpatches.Rectangle(
        (x0, y0), bar_len, bar_h,
        linewidth=0,
        edgecolor='none',
        facecolor=SCALE_BAR_COLOR,
        zorder=10,
    )
    ax.add_patch(rect)


blue_ch  = autobright(dapi_raw)
tile_in  = DISPLAY_PX / DPI
title_h  = 0.32
xlabel_h = 0.24
n_cols   = N_REPRESENTATIVE + 2      # 5 protein tiles + "+N" panel + heatmap
fig_w    = n_cols * tile_in + 0.05
fig_h    = tile_in + title_h + xlabel_h

# ─────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────
for _, row in systems_data.iterrows():
    if row.get('GPT4 score category') != 'High Score':
        continue
    assembly_name = row.get('GPT4 default name', 'None')
    if assembly_name == 'None':
        continue

    try:
        proteins = row['Proteins'].split(' ')
    except AttributeError:
        continue

    protein_indices = {}
    for protein in proteins:
        idx = find_idx(protein)
        if idx is not None and idx < img_stack_original.shape[2]:
            protein_indices[protein] = idx

    if not protein_indices:
        print(f"Skipping {assembly_name}: no proteins found")
        continue

    n_total      = len(protein_indices)
    n_remaining  = max(0, n_total - N_REPRESENTATIVE)

    # Top-N by CV
    scored       = {n: snr_cv(img_stack_original[:TILE_SIZE, :TILE_SIZE, i])
                    for n, i in protein_indices.items()}
    top_proteins = sorted(scored, key=scored.get, reverse=True)[:N_REPRESENTATIVE]
    top_indices  = [protein_indices[p] for p in top_proteins]

    # Heatmap from all proteins
    all_ch = img_stack_original[:TILE_SIZE, :TILE_SIZE,
                                 list(protein_indices.values())].astype(np.float32)
    binary = np.stack([
        (all_ch[:, :, c] > THRESHOLD_MULT * threshold_otsu(all_ch[:, :, c]))
        for c in range(all_ch.shape[2])
    ], axis=2)
    heatmap = binary.sum(axis=2) / all_ch.shape[2]

    # ── Figure — white background ─────────────────────────────────
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    gs = gridspec.GridSpec(
        1, n_cols,
        figure=fig,
        left=0.0, right=1.0,
        top=1.0 - title_h / fig_h,
        bottom=xlabel_h / fig_h,
        wspace=0.03,
    )

    # ── Protein tiles ─────────────────────────────────────────────
    for col, (protein, idx) in enumerate(zip(top_proteins, top_indices)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor('black')
        green_ch = autobright(
            img_stack_original[:TILE_SIZE, :TILE_SIZE, idx].astype(np.float32))
        ax.imshow(make_tile(blue_ch, green_ch), aspect='auto', interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor('#cccccc')
            sp.set_linewidth(0.6)
        ax.set_xlabel(protein, fontsize=FONT_PROTEIN, color='black',
                      fontfamily='DejaVu Sans', fontweight='bold', labelpad=3)

        # Scale bar on leftmost panel only
        if col == 0:
            draw_scale_bar(ax)

    # ── "+N proteins" panel ───────────────────────────────────────
    ax_more = fig.add_subplot(gs[0, N_REPRESENTATIVE])
    ax_more.set_facecolor('white')
    ax_more.set_xticks([]); ax_more.set_yticks([])
    for sp in ax_more.spines.values():
        sp.set_visible(False)
        sp.set_edgecolor('#cccccc')
        sp.set_linewidth(0.6)
    if n_remaining > 0:
        ax_more.text(0.5, 0.55, f'+{n_remaining} more \nproteins...',
                     transform=ax_more.transAxes,
                     fontsize=8, fontweight='bold', color='#333333',
                     fontfamily='DejaVu Sans', ha='center', va='center')
    ax_more.set_xlabel('', fontsize=FONT_PROTEIN, labelpad=3)   # keeps vertical alignment

    # ── Heatmap panel ─────────────────────────────────────────────
    ax_hm = fig.add_subplot(gs[0, N_REPRESENTATIVE + 1])
    ax_hm.set_facecolor('black')
    im = ax_hm.imshow(heatmap, cmap='hot', vmin=0, vmax=1,
                      aspect='auto', interpolation='bilinear')
    ax_hm.set_xticks([]); ax_hm.set_yticks([])
    for sp in ax_hm.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor('#cccccc')
        sp.set_linewidth(0.6)
    ax_hm.set_xlabel('Assembly', fontsize=FONT_PROTEIN, color='#555555',
                     fontfamily='DejaVu Sans', labelpad=3, style='italic')

    # ── Assembly title ─────────────────────────────────────────────
    fig.suptitle(f'{assembly_name}',
                 fontsize=FONT_TITLE, color='black', fontweight='bold',
                 fontfamily='DejaVu Sans',
                 x=0.5, y=1.0 - (title_h * 0.18) / fig_h, va='top')

    fig.savefig(os.path.join(OUT_DIR, f"{sanitise(assembly_name)}.svg"),
                facecolor='white', bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f"  ✓  {assembly_name}  (N={n_total}, +{n_remaining} not shown)")

# ─────────────────────────────────────────────────────────────────
#  Standalone vertical colorbar — white background
# ─────────────────────────────────────────────────────────────────
fig_cb = plt.figure(figsize=(0.35, tile_in + xlabel_h), facecolor='white')
ax_cb  = fig_cb.add_axes([0.05, xlabel_h / (tile_in + xlabel_h),
                           0.40, tile_in / (tile_in + xlabel_h)])
cb = fig_cb.colorbar(
    mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap='hot'),
    cax=ax_cb, orientation='vertical',
)
cb.set_ticks([0, 0.5, 1])
cb.set_ticklabels(['0', '0.5', '1'], fontsize=FONT_CBAR, color='black')
cb.set_label('co-occupancy', fontsize=FONT_CBAR, color='#555555',
             fontfamily='DejaVu Sans', style='italic', labelpad=6)
cb.outline.set_edgecolor('#aaaaaa')
cb.outline.set_linewidth(0.5)
ax_cb.yaxis.set_tick_params(color='black', width=0.5, length=2)

fig_cb.savefig(os.path.join(OUT_DIR, '_colorbar.png'),
               dpi=DPI, facecolor='white', bbox_inches='tight', pad_inches=0.05)
plt.close(fig_cb)
print("  ✓  Colorbar saved → _colorbar.png")
print("\nDone.")