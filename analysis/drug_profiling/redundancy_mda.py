"""How many independent things does the model actually predict? MDA.

The feature list is redundant by construction: rad0+rad1+rad2 = 1, so only two
are free; rad_com summarises the same radial profile; nuc_frac ~ manders_dapi.
Reporting them as separate rows implies breadth that is not there. This clusters
features by how their MEASURED effects covary and reports per-cluster.

On MDA the radial cluster tops recovery; this quantifies that it is one finding,
not four. Reads featan_mda_effects_wide.csv, and featan_mda_recovery.csv if
present.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform   

EFFECTS   = "featan_mda_effects_wide.csv"
RECOVERY  = "featan_mda_recovery.csv"
COPIED    = ["area_cell", "nuc_area_frac"]
CUT       = 0.30      # merge features whose |rho| >= 0.70
OUT       = "featan_mda_clusters.csv"

E = pd.read_csv(EFFECTS)
feats = sorted({c.split("|")[0] for c in E.columns if c.endswith("|real|d")})
feats = [f for f in feats if f not in COPIED]

print("algebraic dependencies among the reported features:")
print("  rad0 + rad1 + rad2 = 1 by construction (shells partition the cell)")
print("  -> at most 2 of the 3 are free; rad_com summarises the same profile")
print("  nuc_frac ~ manders_dapi;  manders_atub, manders_er ~ cytoplasmic fraction\n")

M = E[[f"{f}|real|d" for f in feats]].to_numpy(float)
ok = ~np.isnan(M).any(1)
M = M[ok]
print(f"{M.shape[0]} complete contrasts, {len(feats)} features")

C = spearmanr(M).statistic
C = np.atleast_2d(C)
np.fill_diagonal(C, 1.0)
A = np.abs(C)

ev = np.linalg.eigvalsh(np.corrcoef(M.T))
ev = ev[ev > 0]
pr = ev.sum() ** 2 / (ev ** 2).sum()
var90 = int(np.searchsorted(np.cumsum(np.sort(ev)[::-1]) / ev.sum(), 0.90) + 1)
print(f"\neffective dimensionality (participation ratio): {pr:.1f} of {len(feats)}")
print(f"components needed for 90% of variance          : {var90}")

D = 1.0 - A
np.fill_diagonal(D, 0.0)
D = (D + D.T) / 2
Z = linkage(squareform(D, checks=False), method="average")
lab = fcluster(Z, t=CUT, criterion="distance")

rec = {}
if os.path.exists(RECOVERY):
    R = pd.read_csv(RECOVERY)
    R = R[(R["defn"] == "topq") & (R["score"] == "cv")]
    rec = dict(zip(R["feature"], R["auprc"]))

rows = []
for c in sorted(set(lab)):
    mem = [feats[i] for i in range(len(feats)) if lab[i] == c]
    scores = {m: rec.get(m, np.nan) for m in mem}
    best = max(mem, key=lambda m: (scores[m] if not np.isnan(scores[m]) else -1))
    within = A[np.ix_([feats.index(m) for m in mem], [feats.index(m) for m in mem])]
    rows.append(dict(cluster=c, n=len(mem), representative=best,
                     auprc_best=scores[best],
                     mean_within_abs_rho=float(within[np.triu_indices(len(mem), 1)].mean())
                     if len(mem) > 1 else np.nan,
                     members=", ".join(sorted(mem))))

O = pd.DataFrame(rows).sort_values("auprc_best", ascending=False)
pd.set_option("display.width", 250, "display.max_colwidth", 110)
print(f"\n{'='*130}")
print(f"FEATURE CLUSTERS at |rho| >= {1-CUT:.2f} on the MEASURED effects")
print("  report one representative per cluster, not every member")
print("=" * 130)
print(O[["cluster", "n", "representative", "auprc_best",
         "mean_within_abs_rho", "members"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print(f"\n{'='*130}")
print("PAIRS THAT MEASURE THE SAME BIOLOGY BUT DIFFER IN GEOMETRY WEIGHTING")
print("=" * 130)
for a, b, note in [("nuc_frac", "nc_ratio", "signal fraction in nucleus vs nuclear:cytoplasmic mean"),
                   ("total_cell", "mean_cell", "integrated vs concentration"),
                   ("manders_dapi", "corr_dapi", "compartment fraction vs pixel-level covariation")]:
    ca, cb = f"{a}|real|d", f"{b}|real|d"
    if ca in E.columns and cb in E.columns:
        s = E[[ca, cb, f"{a}|pred|d", f"{b}|pred|d"]].dropna()
        ra = spearmanr(s[ca], s[f"{a}|pred|d"]).statistic
        rb = spearmanr(s[cb], s[f"{b}|pred|d"]).statistic
        print(f"  {a:<13} rho {ra:.3f}   vs   {b:<10} rho {rb:.3f}   ({note})")

O.to_csv(OUT, index=False)
print(f"\nsaved {OUT}")
