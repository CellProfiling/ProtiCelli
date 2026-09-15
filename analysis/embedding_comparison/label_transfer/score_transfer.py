"""Panel B, compute side: k-NN label transfer from the real arm to every arm, 12 lines x 3
sampling seeds. Writes one part JSON per (line, seed) into label_transfer/data/panelB_parts/,
which plot_panel_B.py aggregates and draws.

    python score_transfer.py "<atlas_name>" <samp>     # one cell (ARMS_DIR from the environment)

Design: k=5, cosine, one index (the real arm's training folds) queried by every arm; five folds
grouped over genes AND shared '+'-name components; scaler fit on the index rows only; the label is
the first-listed HPA location per (gene, line). The morphology-only floor (protein channel blanked
before SubCell encoding, then aggregated like the real arm) is joined on gene symbol from
MORPH_DIR/<line>.npz (genes, morph). Each arm is mean-centred on its own columns before scoring.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "utils")]
from paperlib import SHORT, canon_gene                                             # noqa: E402
from build_arms import NORM_V2, dominant_labels, load_arms, load_location_map  # noqa: E402

PARTS = Path(__file__).resolve().parent / "data" / "panelB_parts"
MORPH_DIR = Path(os.environ.get("MORPH_DIR", NORM_V2 / "paper_figs_caches" / "crossdata" / "v2_harmony"))
ARMS = ("real", "gen", "morph")
K, METRIC = 5, "cosine"
N_SPLITS, FOLD_SEED, MIN_CLASS, MIN_ROWS = 5, 42, 2, 50


# ------------------------------------------------------------------ the per-line cache
def build_cache(atlas: str, samp: int) -> dict:
    """Real/gen arms plus the morphology floor joined ON GENE SYMBOL, every arm centred."""
    line = SHORT[atlas]
    genes, real, gen, _ = load_arms(atlas, samp)
    genes = np.array([canon_gene(g) for g in genes])
    out = {"genes": genes, "real": real, "gen": gen}
    c = np.load(MORPH_DIR / f"{line}.npz", allow_pickle=True)
    pos = {g: i for i, g in enumerate(np.asarray(c["genes"]).astype(str))}
    hit = np.array([pos.get(g, -1) for g in genes])
    morph = np.full((len(genes), c["morph"].shape[1]), np.nan, np.float32)
    morph[hit >= 0] = np.asarray(c["morph"])[hit[hit >= 0]]
    out["morph"] = morph
    for arm in ARMS:                                       # per-arm centring; NaN rows stay NaN
        M = np.asarray(out[arm], dtype=np.float64).copy()
        ok = ~np.isnan(M).any(axis=1)
        M[ok] -= M[ok].mean(0)
        out[arm] = M.astype(np.float32)
    out["labels"] = dominant_labels(genes, line, load_location_map())
    return out


# ------------------------------------------------------------------ the transfer kernel
def gene_components(genes) -> np.ndarray:
    """Group id per gene: connected components over shared '+'-separated name parts, so
    multi-gene antibody targets never straddle a fold boundary with their namesakes."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in genes:
        anchor = f"gene::{g}"
        find(anchor)
        for p in [p for p in str(g).split("+") if p]:
            ra, rb = find(anchor), find(f"part::{p}")
            if ra != rb:
                parent[ra] = rb
    return np.array([find(f"gene::{g}") for g in genes])


def assign_folds(genes) -> np.ndarray:
    """Fold id per row, whole components kept together, sizes balanced greedily."""
    comp = gene_components(genes)
    uniq, counts = np.unique(comp, return_counts=True)
    rng = np.random.default_rng(FOLD_SEED)
    order = rng.permutation(len(uniq))
    uniq, counts = uniq[order], counts[order]
    order = np.argsort(-counts, kind="stable")
    loads = np.zeros(N_SPLITS, int)
    assign: dict[str, int] = {}
    for i in order:
        f = int(np.argmin(loads))
        assign[uniq[i]] = f
        loads[f] += counts[i]
    return np.array([assign[c] for c in comp])


def run_line(cache: dict, arms) -> dict:
    """k-NN transfer on the rows of `cache` for `arms`: per-arm accuracy and macro-F1."""
    from sklearn.metrics import f1_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    labels, genes = cache["labels"], cache["genes"]
    keep = labels != "unlabelled"
    vc = pd.Series(labels[keep]).value_counts()
    good = set(vc[vc >= MIN_CLASS].index)
    keep &= np.array([l in good for l in labels])
    if keep.sum() < MIN_ROWS or len(good) < 2:
        raise SystemExit(f"insufficient: {int(keep.sum())} scorable rows, {len(good)} classes")
    y = labels[keep]
    folds = assign_folds(genes[keep])
    X = {a: cache[a][keep] for a in arms}
    preds = {a: np.empty(len(y), dtype=object) for a in arms}
    for f in range(N_SPLITS):
        te = folds == f
        tr = ~te
        if not te.any():
            continue
        if len(np.unique(y[tr])) < 2:
            raise SystemExit(f"fold {f}: fewer than two classes among the index rows")
        scaler = StandardScaler().fit(X["real"][tr])            # index rows only
        clf = KNeighborsClassifier(n_neighbors=min(K, int(tr.sum())), metric=METRIC, weights="distance")
        clf.fit(scaler.transform(X["real"][tr]), y[tr])
        for a in arms:
            preds[a][te] = clf.predict(scaler.transform(X[a][te]))
    assert all(all(p is not None for p in preds[a]) for a in arms), "a row was never predicted"
    out = {"n_scorable": int(keep.sum()), "n_classes": int(len(good)),
           "class_counts": {str(c): int(n) for c, n in vc.items() if c in good}, "arms": {}}
    for a in arms:
        p = preds[a].astype(str)
        out["arms"][a] = {"accuracy": float((p == y).mean()),
                          "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)), "n_genes": int(len(y))}
    return out


def part_path(samp: int, line: str) -> Path:
    return PARTS / f"panelB_samp{samp}_{line}.json"


def score(atlas: str, samp: int) -> None:
    """Real/gen on the full cohort; the morphology floor on the rows it covers, same index design."""
    t0 = time.time()
    PARTS.mkdir(parents=True, exist_ok=True)
    line = SHORT[atlas]
    cache = build_cache(atlas, samp)
    res = run_line(cache, ("real", "gen"))
    ok = ~np.isnan(cache["morph"]).any(axis=1)
    sub = {k: v[ok] for k, v in cache.items()}
    res["arms"]["morph"] = run_line(sub, ("real", "morph"))["arms"]["morph"]
    rec = {"line": line, "atlas_name": atlas, "samp_seed": samp, "k": K, "metric": METRIC, "fold_seed": FOLD_SEED,
           "n_scorable": res["n_scorable"], "n_classes": res["n_classes"], "class_counts": res["class_counts"],
           "rows": [{"line": line, "arm": arm, "samp_seed": samp, "n_classes": res["n_classes"], **met}
                    for arm, met in res["arms"].items()],
           "seconds": round(time.time() - t0, 2)}
    out = part_path(samp, line)
    tmp = out.with_suffix(".json.tmp")
    json.dump(rec, open(tmp, "w"), indent=1)
    tmp.replace(out)
    print(f"[ok] {line:9s} samp{samp} " + "  ".join(f"{a}={res['arms'][a]['macro_f1']:.4f}" for a in ARMS), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    score(sys.argv[1], int(sys.argv[2]))
