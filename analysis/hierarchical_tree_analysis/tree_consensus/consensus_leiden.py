"""Multi-resolution consensus Leiden across single cells.

Heavy: reads the SubCell embeddings of every generated cell of one line (one h5ad, obs cell_id and
gene_name, 200 cells x 12,800 proteins x 1,536; 128 GB, 20 cores).

    python consensus_leiden.py --embeddings E.h5ad --output-dir O [--resolutions ...]

For each resolution gamma, flat Leiden on every cell's embedding as encoded, then the consensus
co-occurrence matrix M_gamma[i, j] = fraction of cells in which proteins i and j share a cluster;
written to <output>/res_<gamma>/consensus_matrix.npz (+ per_cell_labels.npz). Node cohesion in
the consensus tree is the mean off-diagonal M over a cluster's members (build_tree_caches.py).

No per-cell centring: all proteins of one cell share its landmark channels, so the embeddings
share a large common component, and the paper's sweep and the per-cell tree it scores were both
run on the embeddings as encoded.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ inputs
def load_cells(h5ad: Path):
    """Every cell's (n_proteins x 1536) block of the embeddings file, split on obs cell_id;
    the genes must be identical and in the same order in every cell."""
    import anndata as ad
    a = ad.read_h5ad(h5ad)
    cell = a.obs["cell_id"].astype(str).to_numpy()
    gene = a.obs["gene_name"].astype(str).to_numpy()
    X = a.X.toarray() if hasattr(a.X, "toarray") else a.X
    ids = sorted(set(cell))
    embeddings, genes = [], None
    for cid in ids:
        rows = np.flatnonzero(cell == cid)
        g = gene[rows].tolist()
        if genes is None:
            genes = g
        elif g != genes:
            raise ValueError(f"cell {cid}: gene list differs from the first cell")
        embeddings.append(np.asarray(X[rows], dtype=np.float32))
    return embeddings, ids, genes


# ------------------------------------------------------------------ per-cell Leiden + consensus
def cluster_cell(X, resolution, n_neighbors, metric, seed):
    """Flat Leiden on one cell's proteins (cosine kNN graph, fixed seed)."""
    import scanpy as sc
    a = sc.AnnData(X=X)
    sc.pp.neighbors(a, n_neighbors=n_neighbors, use_rep="X", metric=metric)
    sc.tl.leiden(a, resolution=resolution, flavor="leidenalg", n_iterations=2, random_state=seed)
    return a.obs["leiden"].astype(int).values


def consensus_matrix(all_labels, n_proteins) -> np.ndarray:
    M = np.zeros((n_proteins, n_proteins), dtype=np.float32)
    for labels in all_labels:
        for c in np.unique(labels):
            members = np.where(labels == c)[0]
            M[np.ix_(members, members)] += 1
    M /= len(all_labels)
    np.fill_diagonal(M, 1.0)
    return M


def sweep(a) -> None:
    from joblib import Parallel, delayed
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    embeddings, ids, genes = load_cells(Path(a.embeddings))
    n_proteins = len(genes)
    print(f"loaded {len(embeddings)} cells x {n_proteins} proteins", flush=True)
    diagnostics = {}
    for res in a.resolutions:
        res_dir = out_dir / f"res_{res:.2f}"
        res_dir.mkdir(exist_ok=True)
        t0 = time.time()
        labels = Parallel(n_jobs=a.n_cores, prefer="threads")(
            delayed(cluster_cell)(e, res, a.n_neighbors, a.metric, a.seed) for e in embeddings)
        t_leiden = time.time() - t0
        np.savez_compressed(res_dir / "per_cell_labels.npz", labels=np.array(labels))
        t0 = time.time()
        M = consensus_matrix(labels, n_proteins)
        off = M[np.triu_indices_from(M, k=1)]
        diagnostics[f"{res:.2f}"] = dict(
            mean=float(off.mean()), median=float(np.median(off)), std=float(off.std()), max=float(off.max()),
            **{f"pct_above_{t}": float(np.mean(off >= t)) for t in (0.3, 0.5, 0.7, 0.9)},
            mean_clusters_per_cell=float(np.mean([len(np.unique(l)) for l in labels])),
            time_leiden_s=round(t_leiden, 1), time_consensus_s=round(time.time() - t0, 1))
        np.savez_compressed(res_dir / "consensus_matrix.npz", M=M)
        print(f"gamma={res}: off-diagonal mean {diagnostics[f'{res:.2f}']['mean']:.4f}", flush=True)
        del M, off, labels
        gc.collect()
    json.dump(dict(parameters=vars(a), data=dict(n_cells=len(embeddings), n_proteins=n_proteins, cell_ids=ids),
                   sweep_diagnostics=diagnostics, total_seconds=round(time.time() - t_start, 1)),
              open(out_dir / "sweep_summary.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", required=True); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resolutions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5])   # levels 2-7 of the tree
    ap.add_argument("--n-neighbors", type=int, default=25); ap.add_argument("--metric", default="cosine")
    ap.add_argument("--n-cores", type=int, default=20); ap.add_argument("--seed", type=int, default=42)
    sweep(ap.parse_args())
