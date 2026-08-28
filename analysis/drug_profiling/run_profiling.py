"""ProtiCelli drug-perturbation profiling, two modes.

  validation : real + generated images. Reproduces the manuscript results:
               per-drug concordance vs a split-half ceiling (Fig. D),
               strong-responder recovery AP (Fig. E),
               accuracy by true-effect decile (Fig. F),
               and the feature ranking (Table S4).
  profiling  : generated images only. Ranks proteins by simulated effect within
               each drug, the intended antibody-free use.

    python run_profiling.py --mode validation --image_dir DIR \
        --control UNTREATED --drugs PACLITAXEL VORINOSTAT --out results
    python run_profiling.py --mode profiling  --image_dir DIR \
        --control DMSO --drugs DRUGA DRUGB --out results
"""
import os, re, glob, argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from profiling_core import (extract_features, compute_effects,
                            spearman_brown, COVAR, ABS)

TAIL = re.compile(r"__(?P<line>[A-Za-z0-9\-]+)_(?P<gene>[A-Za-z0-9,\-]+)_(?P<kind>real|pred)\.tif$")
POS  = re.compile(r"_(?P<well>[A-Za-z0-9]+)_R\d+_z\d+_crop_\d+__")


def parse_name(path, treats):
    fn = os.path.basename(path)
    t, p = TAIL.search(fn), POS.search(fn)
    if not (t and p):
        return None
    hits = {tok.upper() for tok in fn[:p.start()].split("_")} & treats
    if len(hits) != 1:
        return None
    kind = t.group("kind")
    return dict(kind=kind, gene=t.group("gene").split(",")[0],
                treat=hits.pop(), cell_id=fn[: -len(f"_{kind}.tif")], path=path)


def build_features(image_dir, treats, want_real):
    globs = ["*_pred.tif"] + (["*_real.tif"] if want_real else [])
    paths = sorted(sum((glob.glob(os.path.join(image_dir, g)) for g in globs), []))
    rows = []
    for pth in tqdm(paths, desc="features"):
        m = parse_name(pth, treats)
        if not m:
            continue
        fe = extract_features(pth)
        if fe is None:
            continue
        rows.append({**m, **fe})
    return pd.DataFrame(rows)


def _flag(f):
    return "COVAR" if f in COVAR else "ABS" if f in ABS else ""


def _halves(arm_long, f):
    g = arm_long.dropna(subset=[f"{f}|d0", f"{f}|d1"])
    return g[f"{f}|d0"], g[f"{f}|d1"]


def run_validation(long, feats, out):
    # feature ranking by concordance vs split-half ceiling (Table S4)
    rows = []
    for f in feats:
        R = long[long.arm == "real"].set_index(["gene", "treat"])[f"{f}|d"]
        P = long[long.arm == "pred"].set_index(["gene", "treat"])[f"{f}|d"]
        s = pd.concat([R.rename("r"), P.rename("p")], axis=1).dropna()
        if len(s) < 15:
            continue
        rho = spearmanr(s.r, s.p).statistic
        rr = spearman_brown(spearmanr(*_halves(long[long.arm == "real"], f)).statistic)
        rp = spearman_brown(spearmanr(*_halves(long[long.arm == "pred"], f)).statistic)
        ceil = np.sqrt(max(rr, 0) * max(rp, 0)) if rr > 0 and rp > 0 else np.nan
        rows.append(dict(feature=f, flag=_flag(f), n=len(s), rho=round(rho, 3),
                         ceiling=round(ceil, 3) if ceil == ceil else np.nan,
                         frac_of_ceiling=round(rho / ceil, 3) if ceil and ceil > 0 else np.nan))
    ranking = pd.DataFrame(rows).sort_values("frac_of_ceiling", ascending=False)
    ranking.to_csv(os.path.join(out, "feature_ranking.csv"), index=False)
    top = ranking[ranking.flag == ""].iloc[0].feature   # best non-covariate

    # per-drug recovery AP (Fig. E) + concordance (Fig. D)
    rec = []
    for t in long.treat.unique():
        R = long[(long.arm == "real") & (long.treat == t)].set_index("gene")[f"{top}|d"]
        P = long[(long.arm == "pred") & (long.treat == t)].set_index("gene")[f"{top}|d"]
        s = pd.concat([R.rename("r"), P.rename("p")], axis=1).dropna()
        if len(s) < 20:
            continue
        y = (s.r.abs() >= np.quantile(s.r.abs(), 0.90)).astype(int).to_numpy()
        rec.append(dict(drug=t, feature=top, n=len(s),
                        concordance_rho=round(spearmanr(s.r, s.p).statistic, 3),
                        recovery_AP=round(average_precision_score(y, s.p.abs().to_numpy()), 3),
                        chance=0.10))
    pd.DataFrame(rec).to_csv(os.path.join(out, "recovery.csv"), index=False)

    # accuracy by true-effect decile (Fig. F)
    R = long[long.arm == "real"].set_index(["gene", "treat"])[f"{top}|d"]
    P = long[long.arm == "pred"].set_index(["gene", "treat"])[f"{top}|d"]
    s = pd.concat([R.rename("r"), P.rename("p")], axis=1).dropna()
    s["dec"] = pd.qcut(s.r.abs(), 10, labels=False, duplicates="drop")
    dec = s.groupby("dec").apply(lambda d: pd.Series({
        "med_abs_d": round(d.r.abs().median(), 3),
        "rho": round(spearmanr(d.r, d.p).statistic, 3),
        "dir_acc": round(float((np.sign(d.r) == np.sign(d.p)).mean()), 3)}),
        include_groups=False)
    dec.to_csv(os.path.join(out, "decile.csv"))

    print(f"[validation] top feature {top}")
    print(pd.DataFrame(rec).to_string(index=False))
    print(f"decile rho {dec.rho.iloc[0]} -> {dec.rho.iloc[-1]}, "
          f"dir {dec.dir_acc.iloc[0]} -> {dec.dir_acc.iloc[-1]}")


def run_profiling(long, feats, out, top_feature):
    f = top_feature or "rad0"
    # Shell 0 is the OUTERMOST band: distance_transform_edt is ~0 at the cell
    # boundary and ~1 at the centre, and rad0 covers dt in [0, 1/N_SHELLS).
    # So on rad0 a positive Cliff's delta means signal moved toward the
    # PERIPHERY. rad2 (innermost) is the opposite. Only radial shells carry an
    # interior/periphery meaning; other features get a neutral up/down label.
    if f == "rad0":
        pos_label, neg_label = "periphery", "interior"
    elif f in ("rad2",):
        pos_label, neg_label = "interior", "periphery"
    else:
        pos_label, neg_label = "increase", "decrease"
    for t in long.treat.unique():
        d = long[(long.arm == "pred") & (long.treat == t)][["gene", f"{f}|d", f"{f}|q"]]
        d = d.dropna(subset=[f"{f}|d"]).rename(columns={f"{f}|d": "sim_delta", f"{f}|q": "sim_q"})
        d["direction"] = np.where(d.sim_delta > 0, pos_label, neg_label)
        d = d.reindex(d.sim_delta.abs().sort_values(ascending=False).index).reset_index(drop=True)
        d.to_csv(os.path.join(out, f"profiling_{t}.csv"), index=False)
        print(f"[profiling] {t}: {len(d)} proteins on {f}; top 5 {', '.join(d.gene.head(5))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validation", "profiling"], required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--drugs", nargs="+", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--min_n", type=int, default=20)
    ap.add_argument("--top_feature", default="rad0")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    treats = {a.control.upper(), *(d.upper() for d in a.drugs)}
    want_real = a.mode == "validation"

    fcsv = os.path.join(a.out, "features.csv")
    feat_df = None
    if os.path.exists(fcsv):
        cached = pd.read_csv(fcsv)
        cached_treats = set(cached["treat"].unique())
        cached_arms = set(cached["kind"].unique())
        need_arms = {"real", "pred"} if want_real else {"pred"}
        # reuse only if the cache covers exactly the requested design
        if treats <= cached_treats and need_arms <= cached_arms:
            feat_df = cached
            print(f"reusing cached {fcsv}")
        else:
            missing_t = treats - cached_treats
            raise SystemExit(
                f"cached {fcsv} does not match this run. "
                f"cached treatments {sorted(cached_treats)}, arms {sorted(cached_arms)}; "
                f"requested treatments {sorted(treats)}, arms {sorted(need_arms)}"
                + (f"; missing {sorted(missing_t)}" if missing_t else "")
                + f". Delete {fcsv} (or use a fresh --out) and rerun.")
    if feat_df is None:
        feat_df = build_features(a.image_dir, treats, want_real)
        feat_df.to_csv(fcsv, index=False)
    print(f"{len(feat_df)} cells, {feat_df.gene.nunique()} genes, arms {sorted(feat_df.kind.unique())}")

    arms = ("real", "pred") if want_real else ("pred",)
    long, feats = compute_effects(feat_df, a.control.upper(),
                                  [d.upper() for d in a.drugs], min_n=a.min_n, arms=arms)
    if long.empty:
        raise SystemExit("no (gene, drug) group cleared min_n")
    long.to_csv(os.path.join(a.out, "effects_wide.csv"), index=False)

    (run_validation if want_real else lambda l, f, o: run_profiling(l, f, o, a.top_feature))(long, feats, a.out)
    print(f"done -> {a.out}/")


if __name__ == "__main__":
    main()