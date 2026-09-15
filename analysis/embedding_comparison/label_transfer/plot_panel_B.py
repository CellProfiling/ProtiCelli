"""Panel B (Fig 3B): k-NN label transfer between the SubCell embeddings of real and of simulated images.

Each protein is assigned the location shared by its five most similar proteins in a reference set
of REAL images, for three kinds of query: real protein images, ProtiCelli-generated protein images,
and images with the protein channel blanked (the morphology-only floor). The 36 scoring cells
(12 lines x 3 sampling seeds) were run by score_transfer.py; this script aggregates the parts and
draws the figure.

| section | output |
|---|---|
| 1 | `data/lt_allseeds.csv` (every seed-level row), `data/lt_by_arm.csv` (mean over lines of the per-line seed means) |
| 2 | `figures/F_label_transfer.{svg,png}`, `data/lt_per_line.csv` (the plotted per-line seed means and min/max) |

Each dot is one cell line (its mean over the 3 sampling seeds; whisker = min to max across seeds,
zero-width for the real and morphology arms, which the sampling seed cannot touch). Boxes give
median and IQR across lines. Dashed lines: the majority-class baseline (accuracy) and 1 / mean
class count (macro-F1).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "utils")]
from paperlib import setup
setup(threads=4)       # SVG hash salt, SOURCE_DATE_EPOCH, BLAS threads -- before matplotlib is imported

import json
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import paperlib as pl
from score_transfer import ARMS, K, METRIC, part_path

RES, FIG = HERE / "data", HERE / "figures"
LINES = sorted(pl.SHORT.values())
SEEDS = list(pl.SAMP_SEEDS)

# ------------------------------------------------------------------------------
# 1. The tables
# ------------------------------------------------------------------------------
rows, ncls, maj = [], [], []
for samp in SEEDS:
    for line in LINES:
        p = part_path(samp, line)
        assert p.exists(), f"INCOMPLETE part set: missing {p.name}"
        rec = json.load(open(p))
        rows += rec["rows"]
        if samp == SEEDS[0]:
            cc = rec["class_counts"]; ncls.append(rec["n_classes"]); maj.append(max(cc.values()) / sum(cc.values()))
one = pd.DataFrame(rows)
one.to_csv(RES / "lt_allseeds.csv", index=False)
ncls, maj = float(np.mean(ncls)), float(np.mean(maj))       # chance levels over the lines

g = one.groupby(["line", "arm"])
d = g[["accuracy", "macro_f1"]].agg(["mean", "min", "max"])
d.columns = [f"{m}__{s}" for m, s in d.columns]
d = d.reset_index()
d["n_genes"] = g["n_genes"].first().to_numpy()
d.to_csv(RES / "lt_per_line.csv", index=False)
by_arm = d.groupby("arm")[["accuracy__mean", "macro_f1__mean"]].agg(["mean", "std"])
by_arm.columns = [f"{m.replace('__mean', '')}__{s}_over_lines" for m, s in by_arm.columns]
by_arm.reindex(ARMS).to_csv(RES / "lt_by_arm.csv")
print(by_arm.reindex(ARMS).round(3))

# ------------------------------------------------------------------------------
# 2. The figure
# ------------------------------------------------------------------------------
ARM_LABEL = {"real": "real\n4-channel", "gen": "generated\nprotein", "morph": "morphology only\n(protein blanked)"}
ARM_COLOR = {"real": "#1B6CA8", "gen": "#D1495B", "morph": "#9E9E9E"}


def panel(ax, metric, title, chance, label):
    """Box per arm over the lines, one dot per line at a jittered x with its min-max seed whisker."""
    data = [d.loc[d.arm == a, f"{metric}__mean"].to_numpy() for a in ARMS]
    pos = np.arange(len(ARMS))
    bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#222222", lw=1.2), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                    boxprops=dict(lw=0.8))
    for b, a in zip(bp["boxes"], ARMS):
        b.set_facecolor(ARM_COLOR[a]); b.set_alpha(0.45); b.set_edgecolor("#333333")
    rng = np.random.default_rng(0)
    for i, a in enumerate(ARMS):
        sub = d.loc[d.arm == a]
        x = i + rng.uniform(-0.16, 0.16, len(sub))
        ax.vlines(x, sub[f"{metric}__min"], sub[f"{metric}__max"], lw=0.7, color="#333333", alpha=0.75, zorder=2.5)
        ax.scatter(x, sub[f"{metric}__mean"], s=13, color=ARM_COLOR[a], edgecolor="#333333", linewidth=0.4, zorder=3, alpha=0.95)
    ax.axhline(chance, ls="--", lw=0.9, color="#666666", zorder=1)
    ax.annotate(label, xy=(0.015, chance), xycoords=ax.get_yaxis_transform(), xytext=(0, -8), textcoords="offset points",
                fontsize=6.2, color="#555555", ha="left", va="top")
    ax.set_xticks(pos)
    ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=6.8, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel(title, fontsize=8); ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", ls=":", lw=0.5, color="#CCCCCC"); ax.set_axisbelow(True)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, chance - 0.06 * (hi - lo)), hi)          # room for the chance annotation


FIG.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
panel(axes[0], "accuracy", "top-1 accuracy", maj, f"majority-class baseline ({maj:.2f})")
panel(axes[1], "macro_f1", "macro F1", 1 / ncls, f"chance, {ncls:.0f} classes ({1 / ncls:.2f})")
fig.suptitle(f"k={K} label transfer from real images, per-arm centred, {METRIC}; sampling seeds {','.join(map(str, SEEDS))}", fontsize=9)
fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.94))
pl.save_figure(fig, FIG / "F_label_transfer", dpi=300)
