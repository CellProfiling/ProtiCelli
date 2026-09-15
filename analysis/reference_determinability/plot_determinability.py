"""Reference-channel determinability: Fig 2I (determinability side), Fig S4B, Fig S4C.

The per-(protein, cell line) tables in `data/` come from `determinability.py`; this script only
re-plots them. Every score is X / CEILING, where CEILING is the mean determinability of the three
landmark channels themselves (`data/ceiling_joint_k96.csv`); a landmark stain scores 1.0 by
construction.

| section | figure | output |
|---|---|---|
| 1 | Fig 2I, determinability by compartment (box) | `figures/determinability_by_compartment.*`, `data/determinability_by_compartment_{pairs,summary}.csv` |
| 2 | Fig S4B, single landmarks vs the joint score | `figures/joint_vs_single_channels.*` |
| 3 | Fig S4C, reliability rank vs determinability rank | `figures/reliability_vs_determinability.*`, `data/reliability_vs_determinability_ranking.csv` |

Section 3 needs the per-compartment reliability ranking of the ProtiCelli reliability analysis
(Fig 2I, blue): a text file with one compartment per line, most reliable first, at
`RELIABILITY_RANKING` (default `reliability_ranking.txt`, shipped next to this script).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "utils")]
from paperlib import setup
setup(threads=1)       # SVG hash salt, SOURCE_DATE_EPOCH, BLAS threads -- before matplotlib is imported

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
import paperlib as pl

DATA, FIG = HERE / "data", HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
ceil = pd.read_csv(DATA / "ceiling_joint_k96.csv")
CEILING = float(ceil.loc[ceil.channel == "CEILING", "mean_Xcv"].iloc[0])
print(f"landmark ceiling (joint K=96) = {CEILING:.10f}")

# ------------------------------------------------------------------------------
# 1. Fig 2I: determinability by compartment group
# ------------------------------------------------------------------------------
# Single-location (protein, cell line) pairs, mapped to their compartment group
# (location_groups.csv), joint score on the ceiling scale and capped at 1; groups with at
# least MIN_N pairs. The per-pair export is what Fig 2I is drawn from.

MIN_N = 8
group = pl.load_location_grouping(HERE / "location_groups.csv")
d = pd.read_csv(DATA / "matched_k96_per_protein_sc.csv")
d["locs"] = d["locations"].apply(pl.parse_locations)
mono = d[d["locs"].apply(len) == 1].copy()
mono["location"] = mono["locs"].str[0].map(group)
mono = mono.dropna(subset=["location"]).copy()

agg = mono.groupby("location").agg(n=("ensembl_id", "size"), Xmed=("joint", "median")).reset_index()
loc = agg[agg["n"] >= MIN_N].sort_values("Xmed", ascending=False).reset_index(drop=True)
data = [mono.loc[mono["location"] == lo, "joint"].to_numpy() for lo in loc["location"]]
print(f"compartments: {len(loc)}   pairs: {len(mono):,}")

summary = []
for lo, v in zip(loc["location"], data):
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    summary.append(dict(compartment_group=lo, n=len(v), median=med, q1=q1, q3=q3,
                        whisker_lo=v[v >= q1 - 1.5 * iqr].min(), whisker_hi=v[v <= q3 + 1.5 * iqr].max()))
pd.DataFrame(summary).round(4).to_csv(DATA / "determinability_by_compartment_summary.csv", index=False)
keep = mono[mono["location"].isin(set(loc["location"]))]
pd.DataFrame({"compartment_group": keep["location"], "ensembl_id": keep["ensembl_id"], "cell_line": keep["cell_line"],
              "gene": keep["gene"], "location": keep["locs"].str[0], "X_raw": keep["joint_raw"], "X": keep["joint"],
              "n_crop": keep["n_crop"], "MT": keep["MT"], "NUC": keep["NUC"], "ER": keep["ER"]}
             ).sort_values(["compartment_group", "ensembl_id", "cell_line"]
             ).to_csv(DATA / "determinability_by_compartment_pairs.csv", index=False)

xx = np.arange(len(loc))
fig, ax = plt.subplots(figsize=(max(9, 0.62 * len(loc)), 6))
ax.axhline(1.0, color="#2e7d32", lw=1.2, ls="--", zorder=1)
ax.axhline(0, color="0.6", lw=1.0, zorder=1)
ax.boxplot(data, positions=xx, widths=0.62, showfliers=False, patch_artist=True,
           medianprops=dict(color="#4b3fae", lw=2.2),
           boxprops=dict(facecolor="#8d80e0", edgecolor="#4b3fae", alpha=0.55, lw=1.1),
           whiskerprops=dict(color="#4b3fae", lw=1.1), capprops=dict(color="#4b3fae", lw=1.1))
ax.set_xticks(xx)
ax.set_xticklabels([f"{lo}  (n={int(n):,})" for lo, n in zip(loc["location"], loc["n"])], rotation=45, ha="right", fontsize=9)
ax.set_xlim(-0.7, len(loc) - 0.3); ax.set_ylim(-0.03, 1.05)
ax.set_ylabel("reference determinability X (normalised, landmark channel = 1)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
pl.save_figure(fig, FIG / "determinability_by_compartment")
print(f"capped at 1.0: {int((mono['joint'] >= 1.0).sum()):,}/{len(mono):,} pairs")

# ------------------------------------------------------------------------------
# 2. Fig S4B: three single landmarks vs the joint score
# ------------------------------------------------------------------------------
# The same compartment groups and pairs as Fig 2I: the median determinability from each landmark
# alone (K=32 each) and from all three jointly (K=96). The joint diamond is that compartment's
# Fig 2I median by construction.

CH = [("MT", "#d62728", "microtubules"), ("NUC", "#1f77b4", "nucleus"), ("ER", "#e0a30c", "endoplasmic reticulum")]
CHAN = [c[0] for c in CH]; JCOL = "#3a3a3a"
t = (keep.groupby("location").agg(n=("joint", "size"), **{c: (c, "median") for c in CHAN}, joint=("joint", "median"))
     .reset_index().rename(columns={"location": "comp"}).sort_values("joint").reset_index(drop=True))
yy = np.arange(len(t))
fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(t) + 1.5))
for i, row in t.iterrows():
    vals = [row[c] for c in CHAN] + [row["joint"]]
    ax.plot([min(vals), max(vals)], [i, i], color="0.88", lw=1.4, zorder=1, solid_capstyle="round")
for nm, col, _ in CH:
    ax.plot(t[nm], yy, "o", ms=6.4, color=col, zorder=3)
ax.plot(t["joint"], yy, "D", ms=7.6, mfc="none", mec=JCOL, mew=1.8, zorder=4)
ax.set_yticks(yy); ax.set_yticklabels([f"{r.comp}  ({r.n:,})" for r in t.itertuples()], fontsize=8)
ax.set_ylim(-0.9, len(t) - 0.2)
xmax = max(t[CHAN].to_numpy().max(), t["joint"].max()); ax.set_xlim(-0.012, xmax * 1.10)
ax.set_xlabel("reference determinability X (normalised, landmark channel = 1)", fontsize=8)
ax.tick_params(axis="x", labelsize=7); ax.spines[["top", "right"]].set_visible(False)
handles = [Line2D([], [], marker="o", ls="none", ms=6.4, color=c, label=f"{n} alone ({d})") for n, c, d in CH]
handles += [Line2D([], [], marker="D", ls="none", ms=7.6, mfc="none", mec=JCOL, mew=1.8, label="all three jointly")]
ax.legend(handles=handles, fontsize=7.2, frameon=False, loc="lower right")
fig.tight_layout()
pl.save_figure(fig, FIG / "joint_vs_single_channels")

# ------------------------------------------------------------------------------
# 3. Fig S4C: reliability rank vs determinability rank
# ------------------------------------------------------------------------------
# Rank the Fig 2I compartment groups by median normalised determinability and compare with the
# reliability ranking of the same compartments by Spearman correlation. The reliability ranking
# is the output of the ProtiCelli reliability analysis (Fig 2I, blue), read from
# RELIABILITY_RANKING: one compartment per line, most reliable first.

reliability = [ln.strip() for ln in open(os.environ.get("RELIABILITY_RANKING", HERE / "reliability_ranking.txt")) if ln.strip()]
med = keep.groupby("location")["joint"].median().sort_values(ascending=False)      # rank 1 = highest
our_rank = {c: i + 1 for i, c in enumerate(med.index)}
df = pd.DataFrame([dict(compartment=c, reliability_rank=i + 1, determinability_rank=our_rank.get(c, np.nan),
                        determinability_median=med.get(c, np.nan)) for i, c in enumerate(reliability)])
common = df.dropna(subset=["determinability_rank"]).reset_index(drop=True)
common["reliability_rank"] = np.arange(1, len(common) + 1)          # dense ranks over the common set
rho, p = spearmanr(common["determinability_rank"], common["reliability_rank"])
df["spearman_rho"], df["spearman_p"], df["n_common"] = rho, p, len(common)
df.to_csv(DATA / "reliability_vs_determinability_ranking.csv", index=False)
print(f"Spearman rho = {rho:+.4f}   p = {p:.3e}   n = {len(common)}")

n = len(common)
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.plot([0.5, n + 0.5], [0.5, n + 0.5], ls=":", color="0.6", lw=1, zorder=1, label="perfect agreement (y = x)")
ax.scatter(common["reliability_rank"], common["determinability_rank"], s=34, color="#4b3fae", zorder=3)
for r in common.itertuples():
    ax.annotate(r.compartment, (r.reliability_rank, r.determinability_rank), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.set_xlim(n + 0.5, 0.5); ax.set_ylim(n + 0.5, 0.5)
ax.set_xticks(range(1, n + 1, 2)); ax.set_yticks(range(1, n + 1, 2))
ax.set_xlabel("reliability rank  (1 = most reliable)")
ax.set_ylabel("reference-channel determinability rank  (1 = most determinable)")
ax.set_title(f"Spearman ρ = {rho:+.3f}  (p = {p:.1e}, n = {n})", fontsize=11)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
pl.save_figure(fig, FIG / "reliability_vs_determinability")
