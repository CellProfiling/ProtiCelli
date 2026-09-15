"""Panel A (Fig 3A, Fig S5A): joint UMAP grids, real vs simulated images, twelve cell lines.

One joint UMAP per cell line over the stacked per-protein SubCell embeddings of real and of
ProtiCelli-simulated images (the arms; see ../utils/build_arms.py). The fits and the 15-NN
statistics come from fit_umaps.py (one part per line in data/panelA_parts/); this script joins
the parts into the statistics table
and draws the two grids.

| section | output |
|---|---|
| 1 | `data/panelA_stats.csv`: arm separation and location purity per line, 1,536-D |
| 2 | `figures/F_panelA_main6.{svg,png}` (six largest lines, Fig 3A) and `F_panelA_supp6` (Fig S5A) |
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "utils")]
from paperlib import setup
setup(threads=4)       # SVG hash salt, SOURCE_DATE_EPOCH, BLAS threads -- before matplotlib is imported

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
import paperlib as pl
from fit_umaps import NONE, PARTS

RES, FIG = HERE / "data", HERE / "figures"
LINES = sorted(pl.SHORT.values())

# ------------------------------------------------------------------------------
# 1. The statistics table
# ------------------------------------------------------------------------------
# One row per line from the parts, sorted by protein count. The text quotes the arm separation
# range and the median ratio of generated to real purity on the dominant (35-location) labels;
# the panels print the grouped (13-class) purities.

parts = {p.name[len("panelA_"):-len(".csv")]: p for p in PARTS.glob("panelA_*.csv")}
missing = [l for l in LINES if l not in parts]
assert not missing, f"INCOMPLETE panel A part set: missing {missing}"
stats_tab = (pd.concat([pd.read_csv(parts[l], **pl.RT) for l in LINES], ignore_index=True)
             .sort_values("n_genes", ascending=False).reset_index(drop=True))
stats_tab.to_csv(RES / "panelA_stats.csv", index=False)
print(stats_tab[["line", "n_genes", "arm_separation", "purity_real_grouped", "purity_gen_grouped"]])
print(f"arm separation {stats_tab.arm_separation.min():.3f}-{stats_tab.arm_separation.max():.3f}; "
      f"median gen/real purity (dominant) {(stats_tab.purity_gen_dominant / stats_tab.purity_real_dominant).median():.3f}")
stats = {r.line: r for r in stats_tab.itertuples()}

# ------------------------------------------------------------------------------
# 2. The grids
# ------------------------------------------------------------------------------
# Columns are ordered by protein count; that order decides which six lines go in each figure and
# which column is the alignment reference. Each column is an independent joint fit with no
# canonical orientation, so every column is rigidly mapped onto the widest one (U2OS) by
# orthogonal Procrustes on the proteins the two share, fitted on the real arm and applied to both;
# no scaling, so within-column statistics are unchanged.

ARM_COLORS, OTHER = {"real": "#1F77B4", "gen": "#D62728"}, "#D9D9D9"
LEGEND_NAME = {"Endoplasmic reticulum": "ER"}
MIN_ALIGN_SHARED = 50

data = {}
for f in PARTS.glob("coords_*.npz"):
    z = np.load(f, allow_pickle=True)
    data[f.name[len("coords_"):-len(".npz")]] = {"real": z["real"].astype(np.float32), "gen": z["gen"].astype(np.float32),
                                                 "genes": z["genes"].astype(str), "label": z["location_grouped"].astype(str)}
order_all = sorted(data, key=lambda l: (-len(data[l]["genes"]), l))

ref = data[order_all[0]]
ri = {g: i for i, g in enumerate(ref["genes"])}
for ln in order_all[1:]:
    d = data[ln]
    shared = np.intersect1d(d["genes"], ref["genes"])
    if len(shared) < MIN_ALIGN_SHARED:
        continue
    si = {g: i for i, g in enumerate(d["genes"])}
    A = d["real"][[si[g] for g in shared]].astype(np.float64)
    B = ref["real"][[ri[g] for g in shared]].astype(np.float64)
    ca, cb = A.mean(0), B.mean(0)
    Rot, _ = orthogonal_procrustes(A - ca, B - cb)
    for k in ("real", "gen"):
        d[k] = (((d[k].astype(np.float64) - ca) @ Rot) + cb).astype(np.float32)

# Palette: the 12 most common groups over all twelve lines, so a colour means the same
# compartment in both figures; tab20 indices chosen for maximal separation.
counts = pd.Series(np.concatenate([data[l]["label"] for l in order_all])).value_counts()
keys = [k for k in counts.index if k != NONE][:12]
palette = dict(zip(keys, [plt.get_cmap("tab20").colors[i] for i in (0, 2, 18, 6, 4, 12, 16, 8, 5, 7, 10, 3)]))


def point_size(n):
    """Marker area for a panel of n points, so ink coverage stays roughly constant (30x range in n)."""
    return float(np.clip(150.0 / np.sqrt(max(n, 1)), 1.1, 9.0))


def grid(order, out):
    """3 rows x N columns: both arms by source; the real and the generated arm alone, coloured by
    location, as subsets of the SAME embedding (not refits), sharing row 1's limits."""
    ncol, LEGEND_IN = len(order), 1.55
    fig_w = 1.85 * ncol + LEGEND_IN
    fig, axes = plt.subplots(3, ncol, figsize=(fig_w, 5.9), gridspec_kw={"hspace": 0.07, "wspace": 0.06}, squeeze=False)
    axes_right = 1.0 - LEGEND_IN / fig_w
    for j, ln in enumerate(order):
        d = data[ln]; rxy, gxy, lab = d["real"], d["gen"], d["label"]; n = len(d["genes"]); st = stats[ln]
        s1, s2 = point_size(2 * n), point_size(n)
        lo = [min(rxy[:, i].min(), gxy[:, i].min()) for i in (0, 1)]
        hi = [max(rxy[:, i].max(), gxy[:, i].max()) for i in (0, 1)]
        pad = [(hi[i] - lo[i]) * 0.04 for i in (0, 1)]
        ax = axes[0][j]
        ax.scatter(rxy[:, 0], rxy[:, 1], s=s1, c=ARM_COLORS["real"], alpha=0.45, linewidths=0, rasterized=True)
        ax.scatter(gxy[:, 0], gxy[:, 1], s=s1, c=ARM_COLORS["gen"], alpha=0.45, linewidths=0, rasterized=True)
        ax.set_title(f"{ln}\nn={n:,}\narm sep. {st.arm_separation:.3f}", fontsize=7.5, pad=3, linespacing=1.25, loc="center")
        for row, xy, pur in ((1, rxy, st.purity_real_grouped), (2, gxy, st.purity_gen_grouped)):
            b = axes[row][j]
            b.text(0.5, 1.012, f"purity {pur:.3f}", transform=b.transAxes, ha="center", va="bottom", fontsize=6.0, color="#333333")
            oth = ~np.isin(lab, keys)
            if oth.any():
                b.scatter(xy[oth, 0], xy[oth, 1], s=s2, c=OTHER, alpha=0.5, linewidths=0, rasterized=True)
            for kk in keys:
                m = lab == kk
                if m.any():
                    b.scatter(xy[m, 0], xy[m, 1], s=s2, c=[palette[kk]], alpha=0.85, linewidths=0, rasterized=True)
        for row in range(3):
            b = axes[row][j]
            b.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); b.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            b.set_xticks([]); b.set_yticks([])
            for sp in b.spines.values():
                sp.set_visible(False)
    for row, lab in enumerate(("real + generated", "real", "generated")):
        axes[row][0].set_ylabel(lab, fontsize=7.5, va="center", ha="center", labelpad=8)
    hs = [plt.Line2D([], [], marker="o", ls="", ms=3.6, color=ARM_COLORS[k], label=v) for k, v in (("real", "real"), ("gen", "generated"))]
    fig.legend(handles=hs, fontsize=5.8, frameon=False, ncol=1, loc="lower left", bbox_to_anchor=(axes_right + 0.008, 0.70),
               handlelength=0.8, labelspacing=0.55, borderaxespad=0.0, title="arm", title_fontsize=6.2)
    hl = [plt.Line2D([], [], marker="o", ls="", ms=3.3, color=palette[k], label=LEGEND_NAME.get(k, k)) for k in keys]
    hl.append(plt.Line2D([], [], marker="o", ls="", ms=3.3, color=OTHER, label="other / unknown"))
    fig.legend(handles=hl, fontsize=5.8, frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(axes_right + 0.008, 0.635),
               handlelength=0.8, labelspacing=0.55, borderaxespad=0.0, title="dominant location", title_fontsize=6.2)
    fig.subplots_adjust(right=axes_right)
    pl.save_figure(fig, out, dpi=300)


FIG.mkdir(parents=True, exist_ok=True)
pl.apply_figure_style(frame="boxed", sizes=(8, 7, 6), dpi=150, bbox="standard")
grid(order_all[:6], FIG / "F_panelA_main6")
grid(order_all[6:], FIG / "F_panelA_supp6")
