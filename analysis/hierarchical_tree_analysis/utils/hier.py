"""Hierarchical Leiden subclustering for Fig 3C and 3D (the SubCell paper's 7-level ladder), with
two changes to the resolution tuner.

Seven levels with target cluster counts TARGETS and per-level neighbourhoods NEIGHBORS, euclidean,
each level subclustering every parent cluster. The resolution per level is TUNED:

1. max_modularity_fb -- among the target-attaining resolutions on a fixed 0.005 lattice around the
   first over-shoot, take the one of maximum Newman modularity (a shave and a genuine bipartition
   both satisfy k == 2; modularity separates them). When no lattice point attains the target the
   shipped count-only search (_tune) runs verbatim as the fallback.
2. n_iterations=-1 (converged Leiden) instead of the shipped 2.

Reproducing a shipped cell exactly needs the same CPU microarchitecture (Zen 3) and 4 threads:
the modularity argmax sits on near-ties and a different reduction order flips it (measured).
"""
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sc.settings.verbosity = 0

TARGETS = [1, 2, 6, 14, 22, 38, 86]
NEIGHBORS = [125, 100, 90, 55, 40, 25, 10]
METRIC = "euclidean"
RULE = "max_modularity_fb"
COARSE, FINE, LOOKBEHIND, LOOKAHEAD, MAX_RES = 0.02, 0.005, 0.04, 0.10, 3.0
N_ITERATIONS = -1


# ------------------------------------------------------------------ shipped core
def _leiden_labels(adata, res, seed):
    if res <= 0.0:
        return np.zeros(adata.n_obs, dtype=int)
    sc.tl.leiden(adata, flavor="igraph", resolution=res, n_iterations=N_ITERATIONS,
                 random_state=seed, key_added="_tmp", directed=False)
    return adata.obs["_tmp"].cat.codes.to_numpy()


class _Level:
    """Cached neighbour graphs for every parent subset at one hierarchy level."""

    def __init__(self, X, parent_labels, n_neighbors, seed):
        self.parts = []
        for lab in pd.unique(parent_labels):
            idx = np.where(parent_labels == lab)[0]
            if len(idx) < n_neighbors:
                self.parts.append((lab, idx, None))
                continue
            sub = ad.AnnData(X=np.ascontiguousarray(X[idx]))
            sc.pp.neighbors(sub, n_neighbors=n_neighbors, n_pcs=None, use_rep="X", metric=METRIC, random_state=seed)
            self.parts.append((lab, idx, sub))

    def labels_at(self, res, seed, n_obs):
        out = np.empty(n_obs, dtype=object)
        for lab, idx, sub in self.parts:
            if sub is None:
                out[idx] = f"{lab}.0"
                continue
            codes = _leiden_labels(sub, res, seed)
            for c in np.unique(codes):
                out[idx[codes == c]] = f"{lab}.{c}"
        return out

    def count_at(self, res, seed, n_obs):
        return len(set(self.labels_at(res, seed, n_obs)))


def _renumber(labels):
    """Consecutive 0..K-1 in order of first appearance, as the published code does."""
    uniq = pd.unique(labels)
    mapping = {old: str(new) for new, old in enumerate(uniq)}
    return np.array([mapping[v] for v in labels], dtype=object)


def _tune(level, target, seed, n_obs, step=0.02, max_res=3.0, tol=1e-3, max_evals=90):
    """The shipped count-only tuner, verbatim (the fallback)."""
    cache = {}

    def count(r):
        r = round(max(r, 0.0), 6)
        if r not in cache:
            cache[r] = level.count_at(r, seed, n_obs)
        return cache[r]

    prev_r, r = 0.0, 0.0
    n0 = count(0.0)
    if n0 == target:
        return 0.0, n0, "exact", len(cache), [k for k, v in sorted(cache.items()) if v == target]
    if n0 > target:
        return 0.0, n0, "inexact", len(cache), []
    found = None
    while len(cache) < max_evals and r < max_res:
        prev_r = r
        r = round(r + step, 6)
        n = count(r)
        if n == target:
            found = r
            break
        if n > target:
            fine = round(step / 4.0, 6)
            rr = prev_r
            while rr < r and len(cache) < max_evals:
                rr = round(rr + fine, 6)
                if count(rr) == target:
                    found = rr
                    break
            if found is not None:
                break
            lo, hi = prev_r, r
            while len(cache) < max_evals and (hi - lo) > tol:
                mid = round(0.5 * (lo + hi), 6)
                nm = count(mid)
                if nm == target:
                    found = mid
                    break
                if nm < target:
                    lo = mid
                else:
                    hi = mid
            break
    if found is None:
        best_r, best_n = min(cache.items(), key=lambda kv: (abs(kv[1] - target), kv[0]))
        return round(best_r, 4), best_n, "inexact", len(cache), []
    up = found
    while len(cache) < max_evals + 12:
        nxt = round(up + step / 4.0, 6)
        if nxt > max_res or count(nxt) != target:
            break
        up = nxt
    return round(found, 4), target, "exact", len(cache), sorted(k for k, v in cache.items() if v == target)


# ------------------------------------------------------------------ the adopted rule
def _sizes(labels):
    _, c = np.unique(np.asarray(labels, dtype=object).astype(str), return_counts=True)
    return np.sort(c)[::-1]


def _modularity(level, labels):
    """Newman modularity of `labels` on the level's block-diagonal kNN graph."""
    lab = np.asarray(labels, dtype=object).astype(str)
    uniq = {v: i for i, v in enumerate(np.unique(lab))}
    code = np.array([uniq[v] for v in lab], dtype=np.int64)
    K = len(uniq)
    e, d, two_m = np.zeros(K), np.zeros(K), 0.0
    for _l, idx, sub in level.parts:
        if sub is None:
            continue
        A = sp.triu(sub.obsp["connectivities"].tocsr(), k=1).tocoo()
        if A.nnz == 0:
            continue
        ci, cj = code[idx[A.row]], code[idx[A.col]]
        w = A.data.astype(np.float64)
        two_m += 2.0 * w.sum()
        np.add.at(d, ci, w); np.add.at(d, cj, w)
        same = ci == cj
        np.add.at(e, ci[same], w[same])
    if two_m <= 0:
        return float("nan")
    return float(np.sum(e / (two_m / 2.0) - (d / two_m) ** 2))


class _Grid:
    """The fixed candidate scan of one level: coarse steps to the first over-shoot, then a FINE
    lattice from LOOKBEHIND below it to LOOKAHEAD above. Caches labels per resolution."""

    def __init__(self, level, target, seed, n_obs):
        self.level, self.target, self.seed, self.n = level, target, seed, n_obs
        self.lab, self.r_over, self.base = {}, None, []
        if self.k(0.0) > self.target:
            self.r_over = 0.0
        else:
            r = 0.0
            while r < MAX_RES:
                r = round(r + COARSE, 6)
                if self.k(r) > self.target:
                    self.r_over = r
                    break
            if self.r_over is None:
                self.r_over = MAX_RES
        lo = max(0.0, round(self.r_over - LOOKBEHIND, 6))
        hi = min(MAX_RES, round(self.r_over + LOOKAHEAD, 6))
        n_steps = int(round((hi - lo) / FINE))
        self.base = [round(lo + i * FINE, 6) for i in range(n_steps + 1)]
        for r in self.base:
            self.labels(r)

    def labels(self, r):
        r = round(max(r, 0.0), 6)
        if r not in self.lab:
            self.lab[r] = _renumber(self.level.labels_at(r, self.seed, self.n))
        return self.lab[r]

    def k(self, r):
        return len(_sizes(self.labels(r)))

    def attaining(self):
        return [r for r in self.base if self.k(r) == self.target]


def _select_max_modularity(grid, hit):
    qs = {r: _modularity(grid.level, grid.labels(r)) for r in hit}
    pick = max(hit, key=lambda r: (qs[r] if qs[r] == qs[r] else -1e18, -r))
    return dict(rule=RULE, res=round(pick, 4), achieved=grid.target, status="exact", n_attaining=len(hit),
                modularity=qs[pick], n_evals=len(grid.lab),
                selector="criterion" if len(hit) > 1 else "single_attaining")


def tune(level, target, seed, n_obs):
    """Choose a resolution for `target` clusters under max_modularity_fb."""
    grid = _Grid(level, target, seed, n_obs)
    hit = grid.attaining()
    if not hit:
        res, got, status, nev, att = _tune(level, target, seed, n_obs)
        return dict(rule=RULE, res=res, achieved=got, status=status, n_attaining=len(att),
                    modularity=None, n_evals=len(grid.lab) + nev, selector="shipped_fallback")
    return _select_max_modularity(grid, hit)


def run_arm(X, seed, targets=TARGETS, neighbors=NEIGHBORS):
    """Full 7-level ladder on one arm. Returns (labels DataFrame level1..7, meta dict)."""
    n = X.shape[0]
    cols, records = [], []
    t0 = time.time()
    lvl = _Level(X, np.zeros(n, dtype=int), neighbors[0], seed)
    lab0 = _renumber(lvl.labels_at(0.0, seed, n))
    s0 = _sizes(lab0)
    cols.append(lab0)
    records.append(dict(level=1, target=targets[0], rule=RULE, res=0.0, k=int(len(s0)), status="skipped",
                        n_evals=0, sizes=[int(v) for v in s0], minority=int(s0[-1]),
                        minority_frac=float(s0[-1] / n), seconds=round(time.time() - t0, 1)))
    print("    lvl1 res=0.0 -> %d clusters (%ss)" % (len(s0), records[-1]["seconds"]), flush=True)
    parent = lab0
    for i in range(1, len(targets)):
        t0 = time.time()
        lvl = _Level(X, parent, neighbors[i], seed)
        r = tune(lvl, targets[i], seed, n)
        assert r["res"] is not None, r
        lab = _renumber(lvl.labels_at(r["res"], seed, n))
        s = _sizes(lab)
        k = int(len(s))
        out = dict(r)
        out.update(level=i + 1, target=targets[i], k=k, status=(r["status"] if k == targets[i] else "inexact"),
                   sizes=[int(v) for v in s], minority=int(s[-1]), minority_frac=float(s[-1] / n),
                   seconds=round(time.time() - t0, 1))
        records.append(out)
        print("    lvl%d res=%s -> %s/%d [%s via %s] minority=%s %s evals (%ss)"
              % (i + 1, out["res"], k, targets[i], out["status"], out["selector"], out["minority"],
                 out["n_evals"], out["seconds"]), flush=True)
        cols.append(lab)
        parent = lab
    df = pd.DataFrame({("level%d" % (i + 1)): cols[i] for i in range(len(cols))})
    meta = dict(seed=seed, rule=RULE, targets=targets, neighbors=neighbors, metric=METRIC, n_genes=int(n),
                records=records,
                leiden=dict(flavor="igraph", n_iterations=N_ITERATIONS, directed=False),
                grid=dict(coarse=COARSE, fine=FINE, lookbehind=LOOKBEHIND, lookahead=LOOKAHEAD, max_res=MAX_RES))
    return df, meta
