"""Core for ProtiCelli drug-perturbation profiling.

Shared by both modes:
  validation  (real + pred present)  -> concordance vs a repeatability ceiling
  profiling   (pred only)            -> rank proteins by simulated effect

One feature extractor, one effect estimator (Cliff's delta), used by both.
The only difference between modes is whether a `real` arm exists.

Feature and mask definitions match the manuscript (Methods; Table S4).
"""
import os, re, glob
import numpy as np
import pandas as pd
from tifffile import imread
from scipy.stats import rankdata, mannwhitneyu, spearmanr
from scipy.ndimage import (binary_fill_holes, binary_closing,
                           distance_transform_edt, center_of_mass, gaussian_filter)
from skimage.filters import threshold_otsu
from skimage.measure import label

# Channel order: DAPI, protein-of-interest, microtubule, ER.
CH_DAPI, CH_POI, CH_MT, CH_ER = 2, 1, 0, 3
N_SHELLS   = 3
CLOSE_SZ   = 5
SUB        = 4
MORAN_LAGS = (1, 2, 4, 8)

# Features identical between a real/pred pair (reference-derived). Covariates,
# never evaluation targets. Kept out of all rankings.
COVAR = {"area_cell", "nuc_area_frac"}
# Not scale-invariant; biased by the decoder intensity clamp. Reported flagged.
ABS   = {"mean_cell", "total_cell", "p99_cell", "mean_all",
         "mean_otsu", "total", "otsu_thr"}


# ── features ────────────────────────────────────────────────────────────────
def extract_features(path):
    """40 morphological features for one image stack. None if the mask is empty."""
    img = imread(path).astype(np.float32)
    if img.ndim == 3 and img.shape[0] <= 4 and img.shape[-1] > 4:
        img = np.moveaxis(img, 0, -1)
    poi = img[..., CH_POI]

    ref = np.zeros(poi.shape, np.float32)
    for c in (CH_DAPI, CH_MT, CH_ER):
        ch = img[..., c]
        lo, hi = np.percentile(ch[::SUB, ::SUB], [1, 99.5])
        ref += np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1)
    r8 = (ref * (255 / 3)).astype(np.uint8)
    raw = r8 > threshold_otsu(r8)
    if not raw.any():
        return None
    ys, xs = np.where(raw)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    cell = binary_fill_holes(binary_closing(raw[sl], np.ones((CLOSE_SZ, CLOSE_SZ))))
    if not cell.any():
        return None

    p = poi[sl]; v = p[cell]
    tot = float(v.sum()) + 1e-8; m = float(v.mean()); s = float(v.std()) + 1e-8
    o = {}
    o["mean_cell"], o["total_cell"] = m, tot
    o["p99_cell"] = float(np.percentile(v, 99))
    o["area_cell"] = float(cell.sum()); o["mean_all"] = float(poi.mean())

    sv = np.sort(v)
    o["cv_cell"] = s / (m + 1e-8)
    for frac, nm in ((0.01, "top1_frac"), (0.05, "top5_frac")):
        k = max(1, int(frac * sv.size)); o[nm] = float(sv[-k:].sum() / tot)
    o["gini"] = float(1 - 2 * (np.cumsum(sv) / tot).mean())
    h, _ = np.histogram(v, bins=64); h = h[h > 0] / v.size
    o["entropy"] = float(-(h * np.log2(h)).sum())
    q10, q50, q90 = np.percentile(v, [10, 50, 90])
    o["p90_p50"] = float(q90 / (q50 + 1e-8)); o["p90_p10"] = float(q90 / (q10 + 1e-8))
    o["mad_ratio"] = float(np.median(np.abs(v - q50)) / (q50 + 1e-8))
    o["skew"] = float(((v - m) ** 3).mean() / s ** 3)
    o["kurt"] = float(((v - m) ** 4).mean() / s ** 4)

    dap = img[..., CH_DAPI][sl]
    d8 = (dap * 255 / max(float(dap.max()), 1e-6)).astype(np.uint8)
    nuc = binary_fill_holes(d8 > threshold_otsu(d8)) & cell
    cyto = cell & ~nuc
    if nuc.any() and cyto.any():
        o["nc_ratio"] = float(p[nuc].mean() / (p[cyto].mean() + 1e-8))
        o["nuc_frac"] = float(p[nuc].sum() / tot)
    o["nuc_area_frac"] = float(nuc.sum() / cell.sum())

    # distance_transform_edt: ~0 at the cell boundary, ~1 at the centre.
    # rad0 = outermost shell (periphery), rad{N-1} = innermost (core).
    dt = distance_transform_edt(cell); dt /= (dt.max() + 1e-8)
    edges = np.linspace(0, 1, N_SHELLS + 1); edges[-1] = 1.01
    for i in range(N_SHELLS):
        sh = cell & (dt >= edges[i]) & (dt < edges[i + 1])
        o[f"rad{i}"] = float(p[sh].sum() / tot) if sh.any() else 0.0
    o["rad_com"] = float((v * dt[cell]).sum() / tot)

    x = np.where(cell, p - m, 0.0); den = float((x * x).sum()) + 1e-8
    for lag in MORAN_LAGS:
        nb = (np.roll(x, lag, 0) + np.roll(x, -lag, 0) +
              np.roll(x, lag, 1) + np.roll(x, -lag, 1))
        o[f"moran_lag{lag}"] = float((x * nb).sum() / (4 * den))

    for nm, c in (("dapi", CH_DAPI), ("atub", CH_MT), ("er", CH_ER)):
        b = img[..., c][sl][cell]
        o[f"corr_{nm}"] = float(np.corrcoef(v, b)[0, 1])
        o[f"manders_{nm}"] = float(v[b > b.mean()].sum() / tot)

    cy, cx = center_of_mass(cell); py, px = center_of_mass(np.where(cell, p, 0.0))
    o["mass_disp"] = float(np.hypot(py - cy, px - cx) /
                           (np.sqrt(cell.sum() / np.pi) + 1e-8))

    sm = gaussian_filter(np.where(cell, p, 0.0), 1.0)
    hi = sm > np.percentile(sm[cell], 90)
    lab, nobj = label(hi, return_num=True)
    o["n_obj"] = int(nobj)
    if nobj:
        areas = np.bincount(lab.ravel())[1:]
        o["obj_area"] = float(areas.mean())
        o["obj_area_cv"] = float(areas.std() / (areas.mean() + 1e-8))

    thr = float(threshold_otsu(p)); mk = p > thr
    if mk.any():
        vo = p[mk]
        o["mean_otsu"] = float(vo.mean()); o["total"] = float(vo.sum())
        o["area_frac"] = float(mk.sum() / cell.sum()); o["otsu_thr"] = thr
    return o


# ── effect size ─────────────────────────────────────────────────────────────
def cliffs_delta(a, b):
    """P(a>b) - P(a<b) via rank sums."""
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    r = rankdata(np.concatenate([a, b]))
    return 2 * (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2) - 1


def cliffs_delta_p(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return np.nan, np.nan
    res = mannwhitneyu(a, b, alternative="two-sided")
    return 2 * res.statistic / (n1 * n2) - 1, res.pvalue


def bh(p):
    p = np.asarray(p, float); q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p); pp = p[ok]; n = pp.size
    if n == 0:
        return q
    order = np.argsort(pp)
    r = np.minimum.accumulate((pp[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(r, 0, 1); q[ok] = out
    return q


# ── effects table, per (gene, treat), one or both arms ──────────────────────
def compute_effects(feat_df, control, order, min_n=20, seed=0,
                    arms=("real", "pred")):
    """feat_df columns: gene, treat, kind, cell_id, <features>.
    Returns a wide effects frame with <feat>|<arm>|d,p,q and split-half d0,d1
    for whichever arms are present. `arms` restricts which arms are computed;
    profiling mode passes ('pred',)."""
    rng = np.random.default_rng(seed)
    feats = [c for c in feat_df.columns
             if c not in ("gene", "treat", "kind", "cell_id", "gene_full", "path")]
    feat_df = feat_df.copy()
    feat_df["half"] = rng.integers(0, 2, len(feat_df))
    rows = []
    for gene, g in feat_df.groupby("gene"):
        for arm in arms:
            ga = g[g["kind"] == arm]
            ctrl = ga[ga["treat"] == control]
            if len(ctrl) < min_n:
                continue
            for t in order:
                tr = ga[ga["treat"] == t]
                if len(tr) < min_n:
                    continue
                rec = dict(gene=gene, treat=t, arm=arm, n=len(tr), n_ctrl=len(ctrl))
                for f in feats:
                    a = tr[f].to_numpy(float); b = ctrl[f].to_numpy(float)
                    d, pv = cliffs_delta_p(a, b)
                    rec[f"{f}|d"] = d; rec[f"{f}|p"] = pv
                    for hh in (0, 1):
                        ah = tr.loc[tr["half"] == hh, f].to_numpy(float)
                        bh_ = ctrl.loc[ctrl["half"] == hh, f].to_numpy(float)
                        rec[f"{f}|d{hh}"] = cliffs_delta(ah, bh_)
                rows.append(rec)
    long = pd.DataFrame(rows)
    if long.empty:
        return long, feats
    # BH within (gene, arm) across treatments
    for f in feats:
        q = np.full(len(long), np.nan)
        for _, idx in long.groupby(["gene", "arm"]).groups.items():
            q[long.index.get_indexer(idx)] = bh(long.loc[idx, f"{f}|p"].values)
        long[f"{f}|q"] = q
    return long, feats


def spearman_brown(r):
    return 2 * r / (1 + r) if r is not None and r > -1 else np.nan