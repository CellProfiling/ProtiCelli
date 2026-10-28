"""Can generated images RECOVER the perturbed proteins, not just correlate on average?

Reads featan_mda_effects_wide.csv. MDA: untreated / Paclitaxel / Vorinostat.

Average rho is the wrong metric: it is dominated by the bulk of weakly affected
contrasts, and a high rho is compatible with failing completely in the tail.
Screening value lives in whether the strongest real responders are ranked highly
by the generated images.

Positives, two definitions:
  strict : q_real < ALPHA AND |d_real| >= D_MIN
  topq   : top TOP_FRAC of |d_real|, regardless of q

On MDA the `strict` base rate runs 0.44-0.75 (Paclitaxel effects are huge), so
strict AUPRC is saturated and NOT comparable across features. Use `topq`, which
pins the base rate at TOP_FRAC. The decile and per-treatment blocks below are
keyed off topq for the same reason.

Metrics that behave under class imbalance: AUPRC (baseline = base rate),
EF@x%, precision/recall@k, direction accuracy among recovered hits, plus
recovery stratified by true-effect decile.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

EFFECTS  = "featan_mda_effects_wide.csv"
COPIED   = ["area_cell", "nuc_area_frac"]
ALPHA    = 0.05
D_MIN    = 0.50        # strict def only; strict is saturated on MDA, kept for reference
TOP_FRAC = 0.10
EF_AT    = (0.01, 0.05, 0.10, 0.20)
K_AT     = (25, 50, 100, 200)
N_FOLDS  = 5
ORDER    = ["PACLITAXEL", "VORINOSTAT"]

E = pd.read_csv(EFFECTS)
feats = sorted({c.split("|")[0] for c in E.columns if c.endswith("|real|d")})
feats = [f for f in feats if f not in COPIED]
Xall = E[[f"{f}|pred|d" for f in feats]].to_numpy(float)
groups = E["gene"].to_numpy()
print(f"{len(E)} contrasts, {len(feats)} features")


def ef(y, score, frac):
    k = max(1, int(frac * len(y)))
    top = np.argsort(-score)[:k]
    base = y.mean()
    return (y[top].mean() / base) if base > 0 else np.nan


rows = []
for f in feats:
    yc, pc, qc = f"{f}|real|d", f"{f}|pred|d", f"{f}|real|q"
    if yc not in E.columns or qc not in E.columns:
        continue
    ok = E[[yc, pc, qc]].notna().all(1) & ~np.isnan(Xall).any(1)
    if ok.sum() < 200:
        continue
    dr = E.loc[ok, yc].to_numpy(float)
    dp = E.loc[ok, pc].to_numpy(float)
    q = E.loc[ok, qc].to_numpy(float)
    Xo, go = Xall[ok.to_numpy()], groups[ok.to_numpy()]

    yhat = np.full(len(dr), np.nan)
    gkf = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(go))))
    for tr, te in gkf.split(Xo, dr, go):
        sc = StandardScaler().fit(Xo[tr])
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(Xo[tr]), dr[tr])
        yhat[te] = m.predict(sc.transform(Xo[te]))

    for defn, pos in (("strict", (q < ALPHA) & (np.abs(dr) >= D_MIN)),
                      ("topq", np.abs(dr) >= np.quantile(np.abs(dr), 1 - TOP_FRAC))):
        y = pos.astype(int)
        if y.sum() < 20:
            continue
        for scorename, score in (("uni", np.abs(dp)), ("cv", np.abs(yhat))):
            r = dict(feature=f, defn=defn, score=scorename, n=len(y),
                     base=float(y.mean()),
                     auprc=float(average_precision_score(y, score)),
                     rho_all=spearmanr(dr, dp if scorename == "uni" else yhat).statistic)
            if y.sum() > 10:
                r["rho_pos"] = spearmanr(dr[y == 1],
                                         (dp if scorename == "uni" else yhat)[y == 1]).statistic
            for x in EF_AT:
                r[f"EF@{int(x*100)}%"] = ef(y, score, x)
            for k in K_AT:
                kk = min(k, len(y))
                top = np.argsort(-score)[:kk]
                r[f"prec@{k}"] = float(y[top].mean())
                r[f"rec@{k}"] = float(y[top].sum() / y.sum())
                sg = dp if scorename == "uni" else yhat
                r[f"dir@{k}"] = float((np.sign(sg[top]) == np.sign(dr[top])).mean())
            rows.append(r)

R = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_columns", 40)
fmt = lambda v: f"{v:.3f}"

for defn in ("strict", "topq"):
    for sc in ("uni", "cv"):
        S = R[(R["defn"] == defn) & (R["score"] == sc)].sort_values("auprc", ascending=False)
        if S.empty:
            continue
        note = "  [SATURATED on MDA, not comparable across features]" if defn == "strict" else ""
        print("\n" + "=" * 120)
        print(f"RECOVERY OF PERTURBED CONTRASTS   definition={defn}   score={sc}{note}")
        print(f"  AUPRC baseline = base rate. EF = fold enrichment over base rate.")
        print("=" * 120)
        cols = ["feature", "base", "auprc", "rho_all", "rho_pos"] + \
               [f"EF@{int(x*100)}%" for x in EF_AT] + \
               ["prec@50", "rec@50", "dir@50", "prec@200", "rec@200"]
        print(S[cols].head(20).to_string(index=False, float_format=fmt))

# decile + per-treatment keyed to TOPQ+cv (strict is saturated)
best = R[(R["defn"] == "topq") & (R["score"] == "cv")].sort_values("auprc", ascending=False)
if len(best):
    f = best.iloc[0]["feature"]
    yc, pc = f"{f}|real|d", f"{f}|pred|d"
    s = E[[yc, pc]].dropna()
    s["dec"] = pd.qcut(s[yc].abs(), 10, labels=False, duplicates="drop")
    g = s.groupby("dec").apply(
        lambda d: pd.Series({
            "n": len(d),
            "med_|d_real|": d[yc].abs().median(),
            "rho": spearmanr(d[yc], d[pc]).statistic,
            "dir_acc": float((np.sign(d[yc]) == np.sign(d[pc])).mean())}),
        include_groups=False)
    print("\n" + "=" * 120)
    print(f"PERFORMANCE BY TRUE EFFECT DECILE, feature = {f}  (topq best)")
    print("  the question is whether the last rows beat the first ones")
    print("=" * 120)
    print(g.to_string(float_format=fmt))

    # per treatment, strong responders defined per-treatment by the top TOP_FRAC
    # of |d_real| within that treatment, so Vorinostat (weaker) is not empty.
    print("\n" + "=" * 120)
    print(f"PER TREATMENT, top {int(TOP_FRAC*100)}% |d_real| within each treatment, feature = {f}")
    print("=" * 120)
    out = []
    for t in ORDER:
        m = (E["treat"] == t) & E[yc].notna() & E[pc].notna()
        d = E.loc[m, [yc, pc]]
        if len(d) < 20:
            continue
        thr = np.quantile(d[yc].abs(), 1 - TOP_FRAC)
        hit = d[d[yc].abs() >= thr]
        out.append(dict(treat=t, n=len(d), n_strong=len(hit),
                        thr=float(thr),
                        rho_strong=spearmanr(hit[yc], hit[pc]).statistic,
                        dir_strong=float((np.sign(hit[yc]) == np.sign(hit[pc])).mean()),
                        rho_all=spearmanr(d[yc], d[pc]).statistic))
    print(pd.DataFrame(out).to_string(index=False, float_format=fmt))

R.to_csv("featan_mda_recovery.csv", index=False)
print("\nsaved featan_mda_recovery.csv")
print("Read the topq tables; strict is saturated on this drug set.")
