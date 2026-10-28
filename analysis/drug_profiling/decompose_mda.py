"""Is the concordance protein-specific? MDA, k=2.

d[gene, treat] = grand mean + treat main + gene main + interaction. Only the
interaction is a protein-specific perturbation response.

DEGENERACY AT k=2, stated plainly. With two non-control conditions the treatment
factor has ONE degree of freedom, so the two-way interaction residual is just
the half-difference between the Paclitaxel and Vorinostat effect vectors, per
gene. The -both column is therefore mathematically thin: it tests "does the
model get the DIFFERENCE between the two drugs' effects on a protein right",
which is a real question but not the rich interaction the macrophage 7-condition
version measured. Read -gene (gene main removed, i.e. discrimination within a
drug) as the primary specificity check here; -both is secondary.

Same well-vs-gene confound as macrophage: one well per gene x treat, so a high
interaction rho shows gene-specificity, not that the protein label was used.
Only a label-swapped regeneration (shuffle_control.py) separates those.

Reads featan_mda_effects_wide.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EFFECTS = "featan_mda_effects_wide.csv"
COPIED  = ["area_cell", "nuc_area_frac"]
ORDER   = ["PACLITAXEL", "VORINOSTAT"]
N_PERM  = 1000
SEED    = 0
MIN_OBS = 100          # lower than macrophage: k=2 means fewer cells per matrix

rng = np.random.default_rng(SEED)
E = pd.read_csv(EFFECTS)
feats = sorted({c.split("|")[0] for c in E.columns if c.endswith("|real|d")})
print(f"{len(E)} contrasts, {len(feats)} features")
print(f"NOTE k={len(ORDER)} non-control conditions: -both is a half-difference, "
      f"read -gene as primary\n")


def two_way(M):
    r = np.nanmean(M, axis=1, keepdims=True)
    c = np.nanmean(M, axis=0, keepdims=True)
    g = np.nanmean(M)
    return M - r - c + g


def center(M, how):
    if how == "raw":
        return M.copy()
    if how == "-treat":
        return M - np.nanmean(M, axis=0, keepdims=True)
    if how == "-gene":
        return M - np.nanmean(M, axis=1, keepdims=True)
    return two_way(M)


def rho_flat(A, B):
    a, b = A.ravel(), B.ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    return spearmanr(a[ok], b[ok]).statistic if ok.sum() > 20 else np.nan


rows = []
for f in feats:
    if f in COPIED:
        continue
    cr, cp = f"{f}|real|d", f"{f}|pred|d"
    if cr not in E.columns or cp not in E.columns:
        continue
    R = E.pivot_table(index="gene", columns="treat", values=cr, aggfunc="first")
    P = E.pivot_table(index="gene", columns="treat", values=cp, aggfunc="first")
    cols = [t for t in ORDER if t in R.columns and t in P.columns]
    R, P = R[cols].to_numpy(float), P[cols].to_numpy(float)
    if (~np.isnan(R)).sum() < MIN_OBS:
        continue

    rec = dict(feature=f, n=int((~np.isnan(R) & ~np.isnan(P)).sum()))
    for how in ("raw", "-treat", "-gene", "-both"):
        rec[how] = rho_flat(center(R, how), center(P, how))
    rec["kept_interaction"] = rec["-both"] / rec["raw"] if rec["raw"] else np.nan

    Rc = R - np.nanmean(R)
    tot = np.nanvar(Rc)
    tm = np.nanvar(np.nanmean(Rc, axis=0))
    gm = np.nanvar(np.nanmean(Rc, axis=1))
    inter = np.nanvar(two_way(R))
    rec["var_treat"] = tm / tot if tot else np.nan
    rec["var_gene"] = gm / tot if tot else np.nan
    rec["var_inter"] = inter / tot if tot else np.nan

    Rr, Pr = two_way(R), two_way(P)
    obs = rho_flat(Rr, Pr)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        Pp = Pr.copy()
        for j in range(Pp.shape[1]):
            col = Pp[:, j]
            ok = ~np.isnan(col)
            col[ok] = rng.permutation(col[ok])
            Pp[:, j] = col
        null[i] = rho_flat(Rr, Pp)
    rec["perm_mean"] = float(np.nanmean(null))
    rec["perm_p"] = float((np.nansum(null >= obs) + 1) / (N_PERM + 1))
    rec["z"] = (obs - np.nanmean(null)) / (np.nanstd(null) + 1e-9)
    rows.append(rec)

O = pd.DataFrame(rows).sort_values("-gene", ascending=False)
pd.set_option("display.width", 220, "display.max_rows", 100)
fmt = lambda v: f"{v:.3f}"

print("\n" + "=" * 118)
print("CONCORDANCE BY CENTRING LEVEL")
print("  raw = everything;  -treat = within stimulus;  -gene = within drug;  -both = interaction")
print("  at k=2, -gene is the primary specificity read; -both is a half-difference")
print("=" * 118)
print(O[["feature", "n", "raw", "-treat", "-gene", "-both", "kept_interaction"]]
      .to_string(index=False, float_format=fmt))

print("\n" + "=" * 118)
print("VARIANCE SHARES OF THE MEASURED EFFECT, and the gene-permutation null")
print("=" * 118)
print(O[["feature", "var_treat", "var_gene", "var_inter", "-both",
         "perm_mean", "z", "perm_p"]]
      .to_string(index=False, float_format=fmt))

O.to_csv("featan_mda_decompose.csv", index=False)
print("\nsaved featan_mda_decompose.csv")
print("\nReminder: gene and well are confounded (one well per gene x treat), so a")
print("high interaction rho shows gene-specificity, not that the protein label")
print("was used. Only shuffle_control.py separates those.")
