"""Analyse mda_features.csv: which features carry signal, does the model
reproduce them. CM4AI MDA set: untreated / Paclitaxel / Vorinostat, ~447 genes.

    python analyze_features_mda.py

k=2 non-control conditions, so each gene yields at most 2 contrasts. Thresholds
and the profile block are tuned for that. The effects table is written BEFORE
the summary guard, so a thin/empty summary never costs you the expensive
per-cell recomputation. If summary is empty the script prints why and still
saves featan_mda_effects_wide.csv for inspection.

meta_cols matches the CSV on disk (path,kind,gene,treat,cell_id; no gene_full).
NEGATIVE_GENES empty: under an antimitotic the macrophage structural negatives
(nuclear-envelope, Golgi) all respond, so they cannot serve as a null.
""" 
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr, hypergeom
from tqdm.auto import tqdm

# -- Config -------------------------------------------------------------------
FEATURES_CSV  = "mda_features.csv"
OUT_PREFIX    = "featan_mda"
CONTROL       = "UNTREATED"
ORDER         = ["PACLITAXEL", "VORINOSTAT"]
MIN_N         = 20        # min paired cells per (gene, treat)
MIN_CONTRASTS = 8         # per-feature contrasts needed to summarise; k=2 so this is low
ALPHA         = 0.05
SEED          = 0
MAX_GENES     = 0         # 0 = all

ABS_FEATURES   = {"mean_cell", "total_cell", "p99_cell", "mean_all",
                  "mean_otsu", "total", "otsu_thr"}
COVAR_FEATURES = {"area_cell", "nuc_area_frac"}

NEGATIVE_GENES = []
NEG_TREAT_ONLY = None

rng = np.random.default_rng(SEED)


# -- Helpers ------------------------------------------------------------------
def cliff_fast(a, b):
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    r = rankdata(np.concatenate([a, b]))
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2
    return 2 * u1 / (n1 * n2) - 1


def cliff_p(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return np.nan, np.nan
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 0.0, 1.0
    res = mannwhitneyu(a, b, alternative="two-sided")
    return 2 * res.statistic / (n1 * n2) - 1, res.pvalue


def bh(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pp = p[ok]
    n = pp.size
    if n == 0:
        return q
    order = np.argsort(pp)
    r = pp[order] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(r, 0, 1)
    q[ok] = out
    return q


def spearman_brown(r):
    return 2 * r / (1 + r) if r > -1 else np.nan


# -- Load and pair ------------------------------------------------------------
print(f"loading {FEATURES_CSV}")
df = pd.read_csv(FEATURES_CSV)
meta_cols = ["path", "kind", "gene", "gene_full", "treat", "cell_id"]
feats = [c for c in df.columns if c not in meta_cols]
df = df[df["treat"].isin([CONTROL] + ORDER)]
print(f"{len(df)} rows, {len(feats)} features, {df['gene'].nunique()} genes, "
      f"treats {sorted(df['treat'].unique())}")

if MAX_GENES:
    keep = sorted(df["gene"].unique())[:MAX_GENES]
    df = df[df["gene"].isin(keep)]
    print(f"restricted to {len(keep)} genes")

wide = df.pivot_table(index=["gene", "treat", "cell_id"], columns="kind",
                      values=feats, aggfunc="first")
kinds = sorted({k for _, k in wide.columns})
print(f"pivot kinds present: {kinds}")
if not {"real", "pred"} <= set(kinds):
    raise SystemExit(
        f"pivot has kinds {kinds}, need both 'real' and 'pred'. The kind column "
        f"in {FEATURES_CSV} is not what the pivot expects.")

wide = wide.dropna(axis=0, how="all")
paired = wide.dropna(subset=[(f, k) for f in feats for k in ("real", "pred")
                             if (f, k) in wide.columns], how="any")
print(f"{len(paired)} fully-paired cells "
      f"(of {len(wide)} cell rows after pivot)")
if len(paired) == 0:
    # pinpoint: does ANY single feature pair, i.e. is it a NaN problem not a
    # pairing problem?
    ac = wide[[("area_cell", "real"), ("area_cell", "pred")]].dropna() \
        if ("area_cell", "real") in wide.columns else None
    hint = (f"area_cell alone pairs on {len(ac)} cells; the how='any' across all "
            f"{len(feats)} features is dropping every row because at least one "
            f"feature is NaN per cell. Switch to per-feature pairing (below)."
            if ac is not None and len(ac) else
            "even area_cell does not pair; cell_id is not identical across arms.")
    raise SystemExit(f"0 fully-paired cells. {hint}")
paired = paired.reset_index()
paired["half"] = rng.integers(0, 2, len(paired))


# -- Per-feature, per-treatment effects ---------------------------------------
# Per-feature NaN handling: a cell missing ONE feature should not drop it from
# every other feature's contrast. So effects are computed on the fully-paired
# frame, but each feature drops its own NaNs at delta time (a, b already do).
rows = []
genes = sorted(paired["gene"].unique())
for gene in tqdm(genes, desc="genes"):
    g = paired[paired["gene"] == gene]
    ctrl = g[g["treat"] == CONTROL]
    if len(ctrl) < MIN_N:
        continue
    for t in ORDER:
        tr = g[g["treat"] == t]
        if len(tr) < MIN_N:
            continue
        rec = dict(gene=gene, treat=t, n=len(tr), n_ctrl=len(ctrl))
        for f in feats:
            for k in ("real", "pred"):
                col = (f, k)
                if col not in g.columns:
                    continue
                a = tr[col].to_numpy(float); a = a[~np.isnan(a)]
                b = ctrl[col].to_numpy(float); b = b[~np.isnan(b)]
                d, p = cliff_p(a, b)
                rec[f"{f}|{k}|d"] = d
                rec[f"{f}|{k}|p"] = p
                for h in (0, 1):
                    ah = tr.loc[tr["half"] == h, col].to_numpy(float); ah = ah[~np.isnan(ah)]
                    bh_ = ctrl.loc[ctrl["half"] == h, col].to_numpy(float); bh_ = bh_[~np.isnan(bh_)]
                    rec[f"{f}|{k}|d{h}"] = cliff_fast(ah, bh_)
        rows.append(rec)

E = pd.DataFrame(rows)
if E.empty:
    raise SystemExit(
        f"no (gene,treat) group cleared MIN_N={MIN_N} in both control and "
        f"treatment. paired has {len(paired)} cells across {len(genes)} genes.")
print(f"\n{len(E)} gene x treatment contrasts")

# BH within gene, over the (k) treatments
for f in feats:
    for k in ("real", "pred"):
        c = f"{f}|{k}|p"
        if c not in E.columns:
            continue
        q = np.full(len(E), np.nan)
        for _, idx in E.groupby("gene").groups.items():
            q[E.index.get_indexer(idx)] = bh(E.loc[idx, c].values)
        E[f"{f}|{k}|q"] = q

# SAVE E NOW, before any guard, so a thin summary never discards it.
E.to_csv(f"{OUT_PREFIX}_effects_wide.csv", index=False)
print(f"saved {OUT_PREFIX}_effects_wide.csv ({len(E)} rows)")


# -- Per-feature summary ------------------------------------------------------
neg_present = [g for g in NEGATIVE_GENES if g in set(E["gene"])]
summary, contrast_counts = [], {}
for f in feats:
    dr, dp = f"{f}|real|d", f"{f}|pred|d"
    if dr not in E.columns or dp not in E.columns:
        continue
    s = E[[dr, dp, f"{f}|real|q", f"{f}|pred|q",
           f"{f}|real|d0", f"{f}|real|d1", f"{f}|pred|d0", f"{f}|pred|d1",
           "gene", "treat"]].dropna(subset=[dr, dp])
    contrast_counts[f] = len(s)
    if len(s) < MIN_CONTRASTS:
        continue
    sig_r = s[f"{f}|real|q"] < ALPHA
    sig_p = s[f"{f}|pred|q"] < ALPHA

    rho = spearmanr(s[dr], s[dp]).statistic
    rel_r = spearman_brown(spearmanr(s[f"{f}|real|d0"], s[f"{f}|real|d1"]).statistic)
    rel_p = spearman_brown(spearmanr(s[f"{f}|pred|d0"], s[f"{f}|pred|d1"]).statistic)
    ceil = np.sqrt(max(rel_r, 0) * max(rel_p, 0)) if rel_r > 0 and rel_p > 0 else np.nan
    frac = rho / ceil if ceil and ceil > 0 else np.nan

    both = s[sig_r & sig_p]
    nb = len(both)
    if nb >= 5:
        ru = int((both[dr] > 0).sum())
        gu = int((both[dp] > 0).sum())
        corr = int(((both[dr] > 0) == (both[dp] > 0)).sum())
        e_corr = 2 * gu * ru / nb + nb - gu - ru
        x_need = int(np.ceil((corr - (nb - gu - ru)) / 2))
        p_dir = float(hypergeom.sf(x_need - 1, nb, ru, gu))
        dir_acc, dir_null = corr / nb, e_corr / nb
        dir_fold = dir_acc / dir_null if dir_null > 0 else np.nan
    else:
        dir_acc = dir_null = dir_fold = p_dir = np.nan

    if neg_present:
        ng = s[s["gene"].isin(neg_present)]
        if NEG_TREAT_ONLY:
            ng = ng[ng["treat"] == NEG_TREAT_ONLY]
        neg_r = (ng[f"{f}|real|q"] < ALPHA).mean() if len(ng) else np.nan
        neg_p = (ng[f"{f}|pred|q"] < ALPHA).mean() if len(ng) else np.nan
    else:
        neg_r = neg_p = np.nan

    summary.append(dict(
        feature=f,
        flag=("ABS" if f in ABS_FEATURES else "COVAR" if f in COVAR_FEATURES else ""),
        n=len(s),
        sig_real=float(sig_r.mean()), sig_pred=float(sig_p.mean()),
        max_abs_d=float(s[dr].abs().max()),
        rho=rho, rel_real=rel_r, rel_pred=rel_p, ceiling=ceil, frac_of_ceiling=frac,
        dir_acc=dir_acc, dir_null=dir_null, dir_fold=dir_fold, p_dir=p_dir,
        neg_ctrl_real=neg_r, neg_ctrl_pred=neg_p))

if not summary:
    cc = pd.Series(contrast_counts).sort_values()
    print("\nsummary empty. Per-feature usable-contrast counts (need "
          f">= {MIN_CONTRASTS}):")
    print(cc.to_string())
    print(f"\nMax possible is {len(E)}. If counts are near {len(E)} but the "
          f"script still emptied, the threshold is wrong. If counts are tiny, "
          f"d_real or d_pred is NaN for most contrasts of that feature "
          f"(e.g. nc_ratio on rounded mitotic cells with no cytoplasm). "
          f"E is saved either way; inspect featan_mda_effects_wide.csv.")
    raise SystemExit("no feature summarised; see counts above")

S = pd.DataFrame(summary).sort_values("frac_of_ceiling", ascending=False)
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 200)
fmt = lambda v: f"{v:.3f}"

print("\n" + "=" * 110)
print("FEATURE RANKING, by attenuation-corrected concordance")
print("  ABS   = not scale-invariant, biased by the decoder clamp")
print("  COVAR = reference-derived, identical within a pair, NOT an evaluation target")
print("=" * 110)
print(S[["feature", "flag", "n", "sig_real", "sig_pred", "max_abs_d",
         "rho", "ceiling", "frac_of_ceiling"]].to_string(index=False, float_format=fmt))

print("\n" + "=" * 110)
print("DIRECTION AGREEMENT vs marginal-preserving null (exact hypergeometric)")
print("=" * 110)
print(S[["feature", "flag", "dir_acc", "dir_null", "dir_fold", "p_dir"]]
      .dropna(subset=["dir_fold"]).to_string(index=False, float_format=fmt))

if neg_present:
    print("\n" + "=" * 110)
    print(f"NEGATIVE CONTROLS ({', '.join(neg_present)}), expect ~{ALPHA:.2f}"
          + (f", treatment={NEG_TREAT_ONLY}" if NEG_TREAT_ONLY else ""))
    print("=" * 110)
    print(S[["feature", "flag", "neg_ctrl_real", "neg_ctrl_pred", "sig_real"]]
          .sort_values("neg_ctrl_real").to_string(index=False, float_format=fmt))
else:
    print("\nNO negative-control genes set. Under Paclitaxel there is no clean")
    print("structural null (nuclear-envelope and Golgi disassemble in mitosis).")
    print("Use the shuffled-label control as the negative instead.")

for f in COVAR_FEATURES & set(feats):
    s = E[[f"{f}|real|d", f"{f}|pred|d"]].dropna()
    if len(s):
        md = float((s.iloc[:, 0] - s.iloc[:, 1]).abs().max())
        ok = "OK" if md < 1e-6 else "UNEXPECTED"
        print(f"\ncovariate check  {f}: max |d_real - d_pred| = {md:.2e}  [{ok}]")


# -- Profile-level concordance ------------------------------------------------
prof_feats = [f for f in feats
              if f not in ABS_FEATURES and f not in COVAR_FEATURES
              and f"{f}|real|d" in E.columns]
if len(prof_feats) >= 5:
    R = E[[f"{f}|real|d" for f in prof_feats]].to_numpy(float)
    P = E[[f"{f}|pred|d" for f in prof_feats]].to_numpy(float)
    keep = ~np.isnan(R).any(1) & ~np.isnan(P).any(1)
    R, P = R[keep], P[keep]
    lab = E.loc[keep, ["gene", "treat"]].reset_index(drop=True)
    print("\n" + "=" * 110)
    print(f"PROFILE-LEVEL: {len(prof_feats)} scale-invariant features, "
          f"{len(R)} gene x treatment profiles")
    print("=" * 110)
    if len(R) >= 5:
        pc = np.array([spearmanr(R[i], P[i]).statistic for i in range(len(R))])
        print(f"per-profile real vs pred Spearman: median {np.nanmedian(pc):.3f}, "
              f"{100*np.nanmean(pc > 0):.0f}% positive")
        Rz = (R - R.mean(0)) / (R.std(0) + 1e-8)
        Pz = (P - P.mean(0)) / (P.std(0) + 1e-8)
        Rn = Rz / (np.linalg.norm(Rz, axis=1, keepdims=True) + 1e-8)
        Pn = Pz / (np.linalg.norm(Pz, axis=1, keepdims=True) + 1e-8)
        sim = Pn @ Rn.T
        rank = (sim > sim[np.arange(len(sim)), np.arange(len(sim))][:, None]).sum(1)
        print(f"top-1 retrieval {100*np.mean(rank == 0):.1f}%  "
              f"top-5 {100*np.mean(rank < 5):.1f}%  "
              f"chance {100/len(R):.2f}%  median rank {np.median(rank):.0f}/{len(R)}")
        tv = np.asarray(lab["treat"], dtype=object)
        same_t = tv[:, None] == tv[None, :]
        off = ~np.eye(len(sim), dtype=bool)
        print(f"mean similarity, same treatment {sim[same_t & off].mean():+.3f} vs "
              f"different {sim[(~same_t) & off].mean():+.3f}")
    else:
        print(f"only {len(R)} complete profiles; too few for retrieval.")

# -- Save ranking -------------------------------------------------------------
S.to_csv(f"{OUT_PREFIX}_feature_ranking.csv", index=False)
print(f"\nsaved {OUT_PREFIX}_feature_ranking.csv")