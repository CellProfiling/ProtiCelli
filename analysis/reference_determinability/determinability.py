#!/usr/bin/env python
"""Reference-channel determinability X on HPA v25.1 single cells (Fig 2I, Fig S4B-C; methods).

How much of a protein's within-cell spatial organisation do the three landmark stains (MT, NUC,
ER) determine? Per cell, the protein channel defines a spatial mass q over in-mask pixels and
L = KL(q || uniform) measures how structured it is. The landmarks give a 21-feature per-pixel
texture descriptor (per channel: raw + Gaussian blurs at sigma 2/4/8/16 + two DoGs), rank-
normalised in the mask and quantised with a protein-blind MiniBatch k-means codebook. Enrichment
learned from a protein's other cells (leave-one-cell-out, shrunk) predicts p for the held-out
cell; G = max(0, L - KL(q || p)) and X = sum G / sum L over the protein's cells. The joint
descriptor is quantised at K=96 and each single channel at K=32 (K linear in dimension); all
scores are divided by the landmark-from-landmark ceiling of the joint codebook.

Steps, in order (see run_reference_determinability.sh):
  index      crops_index.parquet -> data/index_sc_full.parquet, one row per segmented cell
  codebooks  fits the codebooks on 1,200 random cells x 1,500 in-mask pixels -> data/codebooks.joblib,
             and draws the K-sweep group sample -> data/ksweep_groups.csv
  worker     --shard i --nshards n: X (joint K=96, MT/NUC/ER K=32) for every cell of every
             (protein, cell line) group in the shard -> data/parts_matched_k/ (resumable, DONE_ markers)
  ceiling    each landmark supplied as the protein, explained by the joint codebook -> data/ceiling_joint_k96.csv
  aggregate  parts -> data/matched_k96_per_protein_sc.csv (ceiling-normalised; the table every figure reads)
  ksweep     the joint-K robustness sweep quoted in the methods (K = 32..4096, singles fixed at 32;
             compartment medians and the rank stability against K=32) -> data/convention_test_{raw,summary}.csv
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-threaded BLAS: reduction order is part of the result
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "utils"))
from paperlib import load_location_grouping, parse_locations

DATA = HERE / "data"
CROPS = os.environ.get("HPA_CROPS", "/scratch/users/ankitgpt/HPA_v25_1/crops")   # HPA v25.1 single-cell crops
LINES = ["A-431", "BJ", "CACO-2", "HEK293", "HeLa", "Hep-G2", "MCF-7", "PC-3",
         "Rh30", "SH-SY5Y", "U-251MG", "U2OS"]
ATLAS_MAP = {**{ln: ln for ln in LINES}, "BJ [Human fibroblast]": "BJ"}   # atlas_name -> cell_line

REF_CH = [0, 2, 3]                       # image channels [MT, protein, NUC, ER] -> the landmarks
CHAN = ["MT", "NUC", "ER"]
COLS = {"MT": slice(0, 7), "NUC": slice(7, 14), "ER": slice(14, 21)}   # each channel's 7 features
SIG_LEVELS = [0.0, 2.0, 4.0, 8.0, 16.0]
DOG_PAIRS = [(2.0, 8.0), (4.0, 16.0)]
MIN_MASK_PX = 3000                       # cells with smaller masks are discarded
MIN_PROT_PX = 300                        # ... or with fewer than this many bright protein pixels
SHRINK = 0.02                            # enrichment shrinkage toward "no enrichment"
EPS = 1e-12

SINGLE_K, JOINT_K = 32, 96
JOINT_KS = [32, 96, 256, 1024, 4096]     # the K sweep (1x, 3x, 8x, 32x, 128x the single K)
# MiniBatchKMeans (n_init, batch_size) per codebook. Two settings, because the reference run fitted
# the singles and joint K=32/256 in one pass and joint K=96/1024/4096 in a later one; kept as is so
# the shipped codebooks and the quoted sweep are reproducible.
FIT_SINGLE = (5, 4096)
FIT_JOINT = {32: (5, 4096), 96: (3, 8192), 256: (5, 4096), 1024: (3, 8192), 4096: (3, 8192)}
N_FIT_CELLS, PX_PER_CELL = 1200, 1500
CEILING_GROUPS, CEILING_MIN_CELLS, CEILING_MAX_CELLS = 70, 3, 15
SWEEP_GROUPS_PER_COMP, SWEEP_MAX_CELLS = 30, 15


# ------------------------------------------------------------------ cells -> (features, q)
def r01(v):
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    return (v - lo) / (hi - lo + 1e-8)


def _read(p, order):
    a = np.asarray(Image.open(p)).astype(np.float32)     # 16-bit raw (never convert("L"))
    return ndimage.zoom(a, 0.5, order=order)             # 1024 -> 512


def crop_stem(plate, position, sample):
    return f"{plate}_{position}_{sample}"


def load_cell(plate, stem, cell):
    """(image [MT, protein, NUC, ER] at 512x512, target-cell mask) or None if a PNG is missing."""
    base = os.path.join(CROPS, str(plate), f"{stem}_cell{cell}")
    try:
        red = _read(base + "_red.png", 1); grn = _read(base + "_green.png", 1)
        blu = _read(base + "_blue.png", 1); yel = _read(base + "_yellow.png", 1)
        cm = _read(base + "_cell_mask.png", 0) > 0.5
    except Exception:
        return None
    return np.stack([red, grn, blu, yel], -1), cm


def feat_maps(refs):
    """21 feature maps: per landmark the 5 blur levels + the 2 signed DoGs."""
    maps = []
    for P in refs:
        g = {s: (P if s == 0 else ndimage.gaussian_filter(P, s)) for s in SIG_LEVELS}
        maps.extend(g[s] for s in SIG_LEVELS)
        maps.extend(g[hi] - g[lo] for lo, hi in DOG_PAIRS)
    return maps


def in_mask_features(img, incell):
    """Rank-normalised (within the mask) 21-feature matrix for the in-mask pixels."""
    refs = [r01(img[..., c]) for c in REF_CH]
    Ffull = np.stack([m.ravel() for m in feat_maps(refs)], axis=1)
    Fin = np.empty((incell.size, Ffull.shape[1]), np.float32)
    for j in range(Ffull.shape[1]):
        col = Ffull[incell, j]
        ref_sorted = np.sort(col)
        Fin[:, j] = np.searchsorted(ref_sorted, col, side="right") / (ref_sorted.size + 1.0)
    return Fin


def spatial_mass(ch):
    """Normalised in-mask mass of a channel after subtracting its median, or None if empty."""
    q = np.clip(ch - np.median(ch), 0, None)
    return None if q.sum() < EPS else q / q.sum()


def prep(img, mask):
    """dict(Fin, q) for one cell, or None if the cell fails the mask / protein-signal filters."""
    prot = r01(img[..., 1])
    if int((prot > (prot.mean() + 3 * prot.std())).sum()) < MIN_PROT_PX:
        return None
    if int(mask.sum()) < MIN_MASK_PX:
        return None
    incell = np.flatnonzero(mask.ravel())
    q = spatial_mass(prot.ravel()[incell])
    return None if q is None else dict(Fin=in_mask_features(img, incell), q=q)


def load_group(recs):
    """Prepared cells of one (protein, cell line) group; recs = (plate, position, sample, cell)."""
    crops = []
    for plate, position, sample, cell in recs:
        res = load_cell(plate, crop_stem(plate, position, sample), cell)
        c = prep(*res) if res is not None else None
        if c is not None:
            crops.append(c)
    return crops


def group_recs(sub, max_cells=None):
    recs = list(zip(sub["if_plate_id"], sub["position"], sub["sample"], sub["cell"]))
    return recs if max_cells is None else recs[:max_cells]


# ------------------------------------------------------------------ the estimator
def cv_K(labs, qs, K):
    """Leave-one-cell-out (sum L, sum G) over a group's cells under one K-state labelling."""
    n = len(labs)
    Ms = np.empty((n, K)); As = np.empty((n, K))
    for j in range(n):
        N = qs[j].size
        Ms[j] = np.bincount(labs[j], weights=qs[j], minlength=K)
        As[j] = np.bincount(labs[j], minlength=K).astype(float) / N
    Mtot = Ms.sum(0); Atot = As.sum(0); ntr = n - 1
    sumL = sumG = 0.0
    for i in range(n):
        Mbar = (Mtot - Ms[i]) / ntr; Abar = (Atot - As[i]) / ntr
        e = (Mbar + SHRINK * Abar) / (Abar + SHRINK * Abar + EPS)
        e = np.where(Abar > 0, e, 1.0)
        qi = qs[i]; Nh = qi.size
        pv = e[labs[i]]; p = pv / (pv.sum() + EPS); p = 0.999 * p + 0.001 / Nh
        pos = qi > 0
        kl = float(np.sum(qi[pos] * np.log(qi[pos] / p[pos])))
        Hq = -np.sum(qi[pos] * np.log(qi[pos])); L = np.log(Nh) - Hq
        sumL += L; sumG += max(0.0, L - kl)
    return sumL, sumG


def X(labs, qs, K):
    sL, sG = cv_K(labs, qs, K)
    return sG / (sL + EPS)


def score_group(crops, books, joint_ks):
    """X of one group: joint at every K in joint_ks, and each single channel at SINGLE_K."""
    qs = [c["q"] for c in crops]
    out = {}
    for K in joint_ks:
        out[K] = X([books[("JOINT", K)].predict(c["Fin"]) for c in crops], qs, K)
    for nm in CHAN:
        out[nm] = X([books[(nm, SINGLE_K)].predict(c["Fin"][:, COLS[nm]]) for c in crops], qs, SINGLE_K)
    return out


def mono_compartment(df):
    """Rows annotated to exactly one HPA location, with their compartment group."""
    grouping = load_location_grouping(HERE / "location_groups.csv")
    locs = df["locations"].apply(parse_locations)
    m = df[locs.apply(len) == 1].copy()
    m["comp"] = locs[m.index].str[0].map(grouping)
    return m.dropna(subset=["comp"])


# ------------------------------------------------------------------ steps
def cmd_index(a):
    """One row per segmented cell of our 12 lines: ids, first-token gene, locations, crop keys."""
    def first_token(s):
        return np.nan if pd.isna(s) else str(s).split(",")[0].strip()
    ci = pd.read_parquet(os.path.join(CROPS, "crops_index.parquet"))
    print(f"crops_index: {len(ci):,} rows")
    ci["cell_line"] = ci["atlas_name"].map(ATLAS_MAP)
    ci = ci.dropna(subset=["cell_line"]).copy()
    print(f"in our 12 lines: {len(ci):,} rows")
    ci["ensembl_id"] = ci["ensembl_ids"].apply(first_token)
    ci["gene"] = ci["gene_names"].apply(first_token)
    ci = ci.dropna(subset=["ensembl_id"]).copy()
    out = ci[["ensembl_id", "cell_line", "gene", "locations", "if_plate_id", "position", "sample", "cell"]].copy()
    for c in ["if_plate_id", "position", "sample", "cell"]:     # stems/paths must build deterministically
        out[c] = out[c].astype(str)
    out = out.sort_values(["ensembl_id", "cell_line", "if_plate_id", "position", "sample", "cell"]).reset_index(drop=True)
    DATA.mkdir(exist_ok=True)
    out.to_parquet(DATA / "index_sc_full.parquet")
    g = out.groupby(["ensembl_id", "cell_line"]).size()
    print(f"wrote index: {len(out):,} cells, {len(g):,} (protein x line) groups, "
          f"{g.mean():.1f} cells/group, {out['gene'].nunique():,} genes")


def cmd_codebooks(a):
    """Fit every codebook on one pixel sample, then draw the K-sweep groups from the same RNG stream."""
    idx = pd.read_parquet(DATA / "index_sc_full.parquet")
    rng = np.random.default_rng(0)
    pick = idx.iloc[rng.choice(len(idx), N_FIT_CELLS, replace=False)]
    feats = []
    for r in pick.itertuples(index=False):
        res = load_cell(r.if_plate_id, crop_stem(r.if_plate_id, r.position, r.sample), r.cell)
        c = prep(*res) if res is not None else None
        if c is None:
            continue                     # a skipped cell consumes no draw, exactly as the reference run
        F = c["Fin"]
        feats.append(F[rng.choice(F.shape[0], min(PX_PER_CELL, F.shape[0]), replace=False)])
    Fall = np.concatenate(feats, 0).astype(np.float32)
    print(f"codebook sample: {Fall.shape[0]:,} px from {len(feats)} cells", flush=True)

    books = {}
    n_init, batch = FIT_SINGLE
    for nm in CHAN:
        books[(nm, SINGLE_K)] = MiniBatchKMeans(SINGLE_K, random_state=0, n_init=n_init,
                                                batch_size=batch).fit(Fall[:, COLS[nm]])
    for K in JOINT_KS:
        n_init, batch = FIT_JOINT[K]
        books[("JOINT", K)] = MiniBatchKMeans(K, random_state=0, n_init=n_init, batch_size=batch).fit(Fall)
        print(f"joint K={K} fit", flush=True)
    joblib.dump(books, DATA / "codebooks.joblib")

    # K-sweep sample: per compartment 30 single-location groups, <=15 cells each, >=2 cells
    m1 = mono_compartment(idx)
    rows = []
    for comp, sub in m1.groupby("comp"):
        keys = sub[["ensembl_id", "cell_line"]].drop_duplicates()
        keys = keys.iloc[rng.permutation(len(keys))[:SWEEP_GROUPS_PER_COMP]]
        for k in keys.itertuples(index=False):
            g = sub[(sub.ensembl_id == k.ensembl_id) & (sub.cell_line == k.cell_line)].head(SWEEP_MAX_CELLS)
            if len(g) >= 2:
                rows.append(g.assign(comp=comp))
    groups = pd.concat(rows, ignore_index=True)
    groups.to_csv(DATA / "ksweep_groups.csv", index=False)
    print(f"wrote codebooks ({len(books)} entries) and {groups.groupby(['ensembl_id', 'cell_line']).ngroups} sweep groups")


def cmd_worker(a):
    """X for every (protein, cell line) group of shard a.shard: joint at K=96, singles at K=32."""
    outdir = DATA / "parts_matched_k"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"s{a.shard:04d}of{a.nshards}"
    part, donef = outdir / f"matched_{tag}.parquet", outdir / f"DONE_{tag}"
    if donef.exists():
        print(f"[{tag}] already done"); return

    books = joblib.load(DATA / "codebooks.joblib")
    idx = pd.read_parquet(DATA / "index_sc_full.parquet")
    keys = idx[["ensembl_id", "cell_line"]].drop_duplicates().sort_values(
        ["ensembl_id", "cell_line"]).reset_index(drop=True)
    mine = keys.iloc[a.shard::a.nshards]
    allc = idx.set_index(["ensembl_id", "cell_line"]).sort_index()

    rows, done, failed = [], set(), []
    if part.exists():
        prev = pd.read_parquet(part)
        rows = prev.to_dict("records")
        done = set(map(tuple, prev[["ensembl_id", "cell_line"]].values))
        print(f"[{tag}] resuming with {len(rows)} rows", flush=True)

    def flush():
        if rows:
            pd.DataFrame(rows).to_parquet(str(part) + ".tmp", index=False)
            os.replace(str(part) + ".tmp", part)

    n_new = 0
    for g in mine.itertuples(index=False):
        key = (g.ensembl_id, g.cell_line)
        if key in done:
            continue
        grp = allc.loc[[key]]
        try:
            crops = load_group(group_recs(grp))
            if len(crops) >= 2:
                s = score_group(crops, books, [JOINT_K])
                rows.append(dict(ensembl_id=g.ensembl_id, cell_line=g.cell_line, gene=grp["gene"].iloc[0],
                                 locations=grp["locations"].iloc[0], n_crop=len(crops),
                                 joint=s[JOINT_K], **{nm: s[nm] for nm in CHAN}))
        except Exception as e:
            print(f"  [err] {key}: {e}", flush=True)
            failed.append(f"{g.ensembl_id}\t{g.cell_line}\t{e}")
        done.add(key); n_new += 1
        if n_new % 25 == 0:
            flush()
    flush()
    if failed:                                   # a failed group is recorded, never silently dropped
        (outdir / f"FAILED_{tag}").write_text("\n".join(failed) + "\n")
    donef.write_text(f"{len(rows)} groups, {len(failed)} failed\n")
    print(f"[{tag}] DONE {len(rows)} groups, {len(failed)} failed", flush=True)


_BOOK = None      # joint codebook, inherited by the ceiling Pool workers via fork


def _ceiling_group(recs):
    """Each landmark supplied AS the protein and explained by the joint context: X per channel."""
    loaded = []
    for plate, position, sample, cell in recs:
        res = load_cell(plate, crop_stem(plate, position, sample), cell)
        if res is None:
            continue
        img, mask = res
        incell = np.flatnonzero(mask.ravel())
        if incell.size < MIN_MASK_PX:
            continue
        loaded.append((in_mask_features(img, incell), {c: r01(img[..., c]).ravel()[incell] for c in REF_CH}))
    if len(loaded) < 2:
        return {}
    out = {}
    for c, nm in zip(REF_CH, CHAN):
        labs, qs = [], []
        for Fin, chans in loaded:
            q = spatial_mass(chans[c])
            if q is not None:
                labs.append(_BOOK.predict(Fin)); qs.append(q)
        if len(qs) >= 2:
            out[nm] = X(labs, qs, JOINT_K)
    return out


def _init_ceiling(book):
    global _BOOK
    _BOOK = book


def cmd_ceiling(a):
    """The landmark-from-landmark ceiling of the joint K=96 codebook (mean over MT, NUC, ER)."""
    book = joblib.load(DATA / "codebooks.joblib")[("JOINT", JOINT_K)]
    g = pd.read_parquet(DATA / "index_sc_full.parquet").groupby(["ensembl_id", "cell_line"])
    keys = [k for k, s in g.size().items() if s >= CEILING_MIN_CELLS]
    rng = np.random.default_rng(0)
    pick = [keys[i] for i in rng.choice(len(keys), min(CEILING_GROUPS, len(keys)), replace=False)]
    tasks = [group_recs(g.get_group(k), CEILING_MAX_CELLS) for k in pick]
    with Pool(a.n_jobs, initializer=_init_ceiling, initargs=(book,)) as pool:
        res = pool.map(_ceiling_group, tasks)
    rows, means = [], []
    for nm in CHAN:
        v = np.array([d[nm] for d in res if nm in d])
        assert v.size, f"{nm}: no ceiling group could be scored (are the crops readable?)"
        print(f"{nm:>4} n={v.size:<3} mean X = {v.mean():.4f} (sd {v.std():.4f})")
        rows.append(dict(K=JOINT_K, channel=nm, n_groups=int(v.size), mean_Xcv=float(v.mean())))
        means.append(v.mean())
    ceiling = float(np.mean(means))
    rows.append(dict(K=JOINT_K, channel="CEILING", n_groups=len(tasks), mean_Xcv=ceiling))
    pd.DataFrame(rows).to_csv(DATA / "ceiling_joint_k96.csv", index=False)
    print(f"JOINT CEILING (K={JOINT_K}) = {ceiling:.10f}")


def cmd_aggregate(a):
    """Per-(protein, line) table on the ceiling scale: the input of every determinability figure."""
    partdir = DATA / "parts_matched_k"
    parts = sorted(glob.glob(str(partdir / "matched_s*.parquet")))
    ndone = len(glob.glob(str(partdir / "DONE_s*")))
    assert ndone == a.nshards, f"incomplete: {ndone}/{a.nshards} shards done"
    failed = sum(len(open(f).read().splitlines()) for f in glob.glob(str(partdir / "FAILED_s*")))
    print(f"{failed} (protein, line) groups failed in the worker (see {partdir}/FAILED_*)" if failed else "no failed groups")
    d = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    d = d.drop_duplicates(["ensembl_id", "cell_line"]).reset_index(drop=True)
    cl = pd.read_csv(DATA / "ceiling_joint_k96.csv")
    ceiling = float(cl.loc[cl.channel == "CEILING", "mean_Xcv"].iloc[0])
    for c in ["joint"] + CHAN:
        d[c + "_raw"] = d[c]
        d[c] = np.clip(d[c] / ceiling, 0, 1)
    d["max_single"] = d[CHAN].max(axis=1)
    d.to_csv(DATA / "matched_k96_per_protein_sc.csv", index=False)
    print(f"ceiling {ceiling:.10f}; {len(d):,} groups, {int(d.n_crop.sum()):,} cells")


_BOOKS = None     # all codebooks, inherited by the sweep Pool workers via fork


def _sweep_group(task):
    key, recs = task
    crops = load_group(recs)
    if len(crops) < 2:
        return None
    s = score_group(crops, _BOOKS, JOINT_KS)
    return {"ensembl_id": key[0], "cell_line": key[1], "n_crop": len(crops),
            **{nm: s[nm] for nm in CHAN}, **{f"joint_K{K}": s[K] for K in JOINT_KS}}


def _init_sweep(books):
    global _BOOKS
    _BOOKS = books


def cmd_ksweep(a):
    """Joint K from 1x to 128x the single K on the compartment-spanning sample drawn by `codebooks`."""
    books = joblib.load(DATA / "codebooks.joblib")
    groups = pd.read_csv(DATA / "ksweep_groups.csv", dtype=str)
    tasks = [((e, ln), group_recs(sub)) for (e, ln), sub in groups.groupby(["ensembl_id", "cell_line"], sort=False)]
    print(f"scoring {len(tasks)} groups", flush=True)
    with Pool(a.n_jobs, initializer=_init_sweep, initargs=(books,)) as pool:
        d = pd.DataFrame([r for r in pool.map(_sweep_group, tasks) if r])
    d["max_single"] = d[CHAN].max(axis=1)
    d.to_csv(DATA / "convention_test_raw.csv", index=False)

    comp = groups.drop_duplicates(["ensembl_id", "cell_line"])[["ensembl_id", "cell_line", "comp"]]
    mm = d.merge(comp, on=["ensembl_id", "cell_line"], how="left")
    summ = []
    for K in JOINT_KS:
        j = d[f"joint_K{K}"]; dif = j - d["max_single"]
        gap = mm.groupby("comp").apply(lambda s: s[f"joint_K{K}"].median() - s["max_single"].median())
        summ.append(dict(joint_K=K, ratio_to_single=K / SINGLE_K, single_K=SINGLE_K,
                         joint_median=j.median(), max_single_median=d.max_single.median(),
                         diff_median=dif.median(), frac_joint_gt_max=(dif > 0).mean(),
                         n_comp_joint_below_max=int((gap < 0).sum()), n_comp=len(gap), n_groups=len(d),
                         rank_spearman_vs_K32=spearmanr(d[f"joint_K{SINGLE_K}"], j)[0]))
    t = pd.DataFrame(summ)
    t.to_csv(DATA / "convention_test_summary.csv", index=False)
    print(t.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("index")
    sub.add_parser("codebooks")
    w = sub.add_parser("worker")
    w.add_argument("--shard", type=int, required=True)
    w.add_argument("--nshards", type=int, required=True)
    for name in ("ceiling", "ksweep"):
        sub.add_parser(name).add_argument("--n-jobs", type=int, default=16)
    sub.add_parser("aggregate").add_argument("--nshards", type=int, default=1000)
    a = ap.parse_args()
    if a.step != "aggregate":
        assert os.path.isdir(CROPS), f"HPA_CROPS is not a directory: {CROPS}"
    globals()[f"cmd_{a.step}"](a)


if __name__ == "__main__":
    main()
