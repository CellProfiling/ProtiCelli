"""Do generated images reproduce changes in CELL-TO-CELL heterogeneity? MDA.

Cliff's delta measures a shift in the population median. A treatment that
widens the distribution without moving the median gives delta ~ 0. This uses
the spread itself: log2 ratio of (IQR, MAD) treated vs control, per
(gene, treat, arm, feature), then asks whether the generated arm reproduces the
real arm's spread change, against a bootstrap split-half ceiling.

Reads mda_features_fixed.csv (per-cell). meta includes gene_full.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm

FEATURES_CSV = "mda_features.csv"
CONTROL  = "UNTREATED"
ORDER    = ["PACLITAXEL", "VORINOSTAT"]
MIN_N    = 50
MIN_HALF = 25
COPIED   = {"area_cell", "nuc_area_frac"}
SEED     = 0

rng = np.random.default_rng(SEED)
df = pd.read_csv(FEATURES_CSV)
meta = ["path", "kind", "gene", "gene_full", "treat", "cell_id"]
feats = [c for c in df.columns if c not in meta and c not in COPIED]
df = df[df["treat"].isin([CONTROL] + ORDER)]
print(f"{len(df)} rows, {len(feats)} features")


def spread(v, min_n=None):
    v = v[~np.isnan(v)]
    if len(v) < (MIN_N if min_n is None else min_n):
        return np.nan, np.nan
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return q75 - q25, np.median(np.abs(v - q50))


rows = []
for (gene, kind), g in tqdm(df.groupby(["gene", "kind"]), desc="gene x arm"):
    ctrl = g[g["treat"] == CONTROL]
    if len(ctrl) < MIN_N:
        continue
    base = {f: spread(ctrl[f].to_numpy(float)) for f in feats}
    h = rng.permutation(len(ctrl))
    b0 = {f: spread(ctrl[f].to_numpy(float)[h[:len(h)//2]], MIN_HALF) for f in feats}
    b1 = {f: spread(ctrl[f].to_numpy(float)[h[len(h)//2:]], MIN_HALF) for f in feats}
    for t in ORDER:
        tr = g[g["treat"] == t]
        if len(tr) < MIN_N:
            continue
        ht = rng.permutation(len(tr))
        rec = dict(gene=gene, kind=kind, treat=t, n=len(tr))
        for f in feats:
            a = tr[f].to_numpy(float)
            for j, nm in enumerate(("iqr", "mad")):
                num, den = spread(a)[j], base[f][j]
                rec[f"{f}|{nm}"] = (np.log2(num / den)
                                    if num and den and num > 0 and den > 0 else np.nan)
            n0 = spread(a[ht[:len(ht)//2]], MIN_HALF)[0]
            n1 = spread(a[ht[len(ht)//2:]], MIN_HALF)[0]
            rec[f"{f}|iqr0"] = (np.log2(n0 / b0[f][0])
                                if n0 and b0[f][0] and n0 > 0 and b0[f][0] > 0 else np.nan)
            rec[f"{f}|iqr1"] = (np.log2(n1 / b1[f][0])
                                if n1 and b1[f][0] and n1 > 0 and b1[f][0] > 0 else np.nan)
        rows.append(rec)

H = pd.DataFrame(rows)
if H.empty:
    raise SystemExit(f"no groups with >= {MIN_N} cells")
R = H[H["kind"] == "real"].set_index(["gene", "treat"])
P = H[H["kind"] == "pred"].set_index(["gene", "treat"])
idx = R.index.intersection(P.index)
R, P = R.loc[idx], P.loc[idx]
print(f"\n{len(idx)} gene x treatment contrasts with both arms")

out = []
for f in feats:
    for nm in ("iqr", "mad"):
        c = f"{f}|{nm}"
        s = pd.concat([R[c].rename("r"), P[c].rename("p")], axis=1).dropna()
        if len(s) < 50:
            continue
        rho = spearmanr(s["r"], s["p"]).statistic
        hh = R[[f"{f}|iqr0", f"{f}|iqr1"]].dropna()
        rel = spearmanr(hh.iloc[:, 0], hh.iloc[:, 1]).statistic if len(hh) > 50 else np.nan
        ceil = 2 * rel / (1 + rel) if rel and rel > -1 else np.nan
        out.append(dict(feature=f, spread=nm, n=len(s),
                        med_abs_change=float(s["r"].abs().median()),
                        rho=rho, ceiling=ceil,
                        frac=rho / ceil if ceil and ceil > 0 else np.nan))

O = pd.DataFrame(out).sort_values("frac", ascending=False)
pd.set_option("display.width", 200, "display.max_rows", 200)
print("\n" + "=" * 92)
print("CELL-TO-CELL HETEROGENEITY CHANGE, real vs generated")
print("  med_abs_change = typical |log2 spread ratio| in the real arm;")
print("                   near 0 means treatments barely change spread at all")
print("=" * 92)
print(O.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
O.to_csv("featan_mda_heterogeneity.csv", index=False)
print("\nsaved featan_mda_heterogeneity.csv")
