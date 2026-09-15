"""Cluster-stability validation, Fig S10

Does the **generated** co-clustering score of a Fig 6B tree node (how consistently a cluster's
proteins co-cluster across single cells) predict anything about **real** images? Here it is tested
against the spread of the cluster's proteins in the SubCell embedding space of real U2OS images,
with the protein count held fixed by bootstrap and generated embeddings never used on the
real-spread axis. `data/node_master.parquet` (one row per tree node and level, produced by
`cluster_validation.py` from 295,893 real single cells) is the only input; the script draws
`figures/F_fig_S10.{svg,png}`.

Columns used: `cohesion` (the generated co-clustering score = Fig 6B node colour), `cos_dist_psub`
with `_lo/_hi` (real-cloud mean pairwise cosine distance at a fixed protein budget, with the 95%
bootstrap CI), `coloc_conc` (the co-occupancy score = Fig 6B heatmap axis), `level`.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "utils")]
from paperlib import setup
setup(threads=1)       # SVG hash salt, SOURCE_DATE_EPOCH, BLAS threads -- before matplotlib is imported

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import paperlib as pl

DATA, FIG = HERE / "data", HERE / "figures"
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 300,
})
df = pd.read_parquet(DATA / "node_master.parquet")
print(df.groupby("level").size().rename("nodes per level").to_dict())

# ------------------------------------------------------------------------------
# 1. The three panels
# ------------------------------------------------------------------------------
# Panel A: cohesion vs real spread at one level with bootstrap whiskers and a least-squares line;
# the box reports Spearman rho (primary), Pearson r, R^2 and n. Panel B: Spearman rho at every
# level with >= 14 clusters (levels 4-7; levels 2 and 3 have 2 and 6 clusters, too few to
# correlate), with significance stars. Panel C: the model's two Fig 6B scores, co-clustering
# (across cells) and co-occupancy (within the reference cell), against each other.

LEVEL, LEVEL_LABEL = 6, "leaf partition, 38 clusters"
COL, AGREE_COL, FIT = "#B23A48", "#7B6BA8", "#333333"
BAR_LEVELS = [4, 5, 6, 7]         # levels with >= 14 clusters


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def n_at(df, lvl):
    return int((df.level == lvl).sum())

def scatter(ax, df, level, letter, heading):
    """cohesion vs real-cloud spread (protein-count matched) at `level`, with bootstrap CIs."""
    s = df[df.level == level].dropna(subset=["cohesion", "cos_dist_psub"])
    x = s.cohesion.to_numpy(); y = s.cos_dist_psub.to_numpy()
    lo = s.cos_dist_psub_lo.to_numpy(); hi = s.cos_dist_psub_hi.to_numpy()
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="none", ecolor=COL,
                elinewidth=0.8, alpha=0.35, capsize=1.5, zorder=1)
    ax.scatter(x, y, s=36, c=COL, edgecolor="k", linewidth=0.4, alpha=0.9, zorder=2)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b * xs + a, "--", color=FIT, lw=1.2, zorder=3)
    rho, p = spearmanr(x, y); r, _ = pearsonr(x, y)
    ax.set_xlabel("Co-clustering score  (Fig 6B node colour)")
    ax.set_ylabel("Real single-cell spread\n(mean pairwise cosine distance)")
    ax.set_title(f"{letter}   {heading}", loc="left", fontweight="bold")
    ax.text(0.96, 0.95, f"Spearman $\\rho$ = {rho:+.2f}  (p = {p:.1g})\n"
            f"Pearson r = {r:+.2f},  $R^2$ = {r**2:.2f}\nn = {len(x)} clusters",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))


def by_level(ax, df, shown_level, letter):
    """Spearman rho of cohesion vs real spread at every level with >= 14 clusters."""
    xpos = np.arange(len(BAR_LEVELS))
    idx = BAR_LEVELS.index(shown_level)
    ax.axvspan(idx - 0.5, idx + 0.5, color="#eceff3", zorder=0)
    rhos, ps = [], []
    for lvl in BAR_LEVELS:
        s = df[df.level == lvl].dropna(subset=["cohesion", "cos_dist_psub"])
        rho, p = spearmanr(s.cohesion, s.cos_dist_psub)
        rhos.append(rho); ps.append(p)
    ax.bar(xpos, rhos, 0.8, color=COL, edgecolor="k", linewidth=0.5, zorder=2)
    for xp, rho, p in zip(xpos, rhos, ps):
        ax.text(xp, rho - 0.02, stars(p), ha="center", va="top", fontsize=9, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels([f"L{l}\n({n_at(df, l)})" for l in BAR_LEVELS])
    ax.get_xticklabels()[idx].set_fontweight("bold")
    ax.set_ylim(min(0.0, float(np.nanmin(rhos))) - 0.16, 0.12)
    ax.set_ylabel("Spearman $\\rho$\n(co-clustering vs real spread)")
    ax.set_xlabel("Hierarchy level  (n clusters); shaded = level shown at left/above")
    ax.set_title(f"{letter}   Holds across the hierarchy", loc="left", fontweight="bold")
    ax.text(0.015, 0.965, "* p<.05   ** p<.01   *** p<.001", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.8, color="#666")


def agreement(ax, df, level, letter):
    """The two generated Fig 6B axes (co-clustering, co-occupancy) agree -- model internal consistency."""
    s = df[df.level == level].dropna(subset=["cohesion", "coloc_conc"])
    x, y = s.cohesion.to_numpy(), s.coloc_conc.to_numpy()
    ax.scatter(x, y, s=36, c=AGREE_COL, edgecolor="k", linewidth=0.4, alpha=0.9)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b * xs + a, "--", color=FIT, lw=1.2)
    rho, p = spearmanr(x, y); r, _ = pearsonr(x, y)
    ax.set_xlabel("Co-clustering score  (node colour)")
    ax.set_ylabel("Co-occupancy score  (heatmap)")
    ax.set_title(f"{letter}   The two Fig 6B axes agree", loc="left", fontweight="bold")
    ax.text(0.96, 0.06, f"Spearman $\\rho$ = {rho:+.2f}  (p = {p:.1g})\nPearson r = {r:+.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

# ------------------------------------------------------------------------------
# 2. The figure
# ------------------------------------------------------------------------------
# The 1 x 3 panel at the leaf partition (level 6, 38 clusters). The shipped run gives Spearman
# rho = -0.55 (p = 3.8e-4), Pearson r = -0.74, n = 38 in panel A; every bar in panel B negative
# and significant; rho = +0.72 in panel C.

fig, axs = plt.subplots(1, 3, figsize=(14.4, 4.5), layout="constrained")
scatter(axs[0], df, LEVEL, "A", heading=f"Model score predicts real spread\n({LEVEL_LABEL})")
by_level(axs[1], df, LEVEL, "B")
agreement(axs[2], df, LEVEL, "C")
FIG.mkdir(exist_ok=True)
pl.save_figure(fig, FIG / "F_fig_S10")
