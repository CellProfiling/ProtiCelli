#!/usr/bin/env python
"""Cluster-stability validation (Fig S10), compute side: the real single-cell spread of every node
of the cell-134 consensus tree, against the node's generated co-clustering score.

Generated embeddings define the clusters (the tree, its cohesion, its co-occupancy heatmaps);
real HPA U2OS single-cell SubCell embeddings measure how tightly each cluster's proteins sit.
Spread is the mean pairwise cosine distance at a FIXED protein budget per level (bootstrap over
member proteins, <= 20 cells each), so cluster size cannot drive the correlation.

Steps, in order (see run_hierarchical_tree_analysis.sh, step validation):
  prep    real HPA per-cell features (.pth) -> the U2OS store
  nodes   the tree nodes (levels 2-7) with their cohesion (tree_consensus/data/cell_<id>, from build_tree_caches.py)
  spread  per node: real spread at the fixed protein budget, with the 95 % bootstrap CI
  coloc   per node: co-occupancy concentration of its Fig 6B heatmap
  merge   -> data/node_master.parquet (level, node_id, cohesion, n_proteins, n_cells, p_sub,
             cos_dist_psub{,_lo,_hi}, coloc_conc), the only input of plot_fig_S10.py

Inputs (env, defaults are the lab's paths): HPA_REAL_FEATURES (the SubCell per-cell features of
the real HPA images, .pth), CELL_ID (the reference cell, 134), VALIDATION_CACHE (intermediates).
"""
import argparse
import gzip
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "utils")]
import treelib as tl  # noqa: E402

TREE_DATA = HERE.parent / "tree_consensus" / "data" / f"cell_{os.environ.get('CELL_ID', '134')}"
CACHE = Path(os.environ.get("VALIDATION_CACHE", "/scratch/groups/emmalu/konstantin/proticelli/paper_v2/validation_cache"))
REAL_PTH = os.environ.get("HPA_REAL_FEATURES", "/scratch/groups/emmalu/subcell_ankit/features/"
                                               "MAE-CellS-ProtS-Pool/hpa_features/harmonized_features_if_plate_id.pth")
MIN_NODE_CELLS = 50          # nodes pooling fewer real cells are dropped
MIN_CELLS_PER_GENE = 10      # a member protein counts only with this many real cells
P_SUB_CAP = 200              # protein budget cap (only lowers level 2, never correlated: n=2)
CELLS_PER_PROTEIN, N_BOOT = 20, 100


# ------------------------------------------------------------------ spread
def mean_cos_dist(X):
    """Exact mean pairwise cosine distance via the normalised-vector sum, O(n d) in float64."""
    Xn = np.asarray(X, dtype=np.float64) / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    n = len(Xn)
    S = Xn.sum(0)
    return float(1.0 - (float(S @ S) - n) / (n * (n - 1)))


def gene_rows(meta):
    """gene -> row indices in the real store, for genes with enough cells."""
    return {g: sub["row"].to_numpy() for g, sub in meta.groupby("gene") if len(sub) >= MIN_CELLS_PER_GENE}


def protein_budget_spread(genes, grows, emb, p_sub, seed=0):
    """Bootstrap spread over p_sub member proteins x <= CELLS_PER_PROTEIN cells: mean and 95 % CI;
    NaN when the node has fewer scorable proteins than the budget."""
    valid = [g for g in genes if g in grows]
    if len(valid) < p_sub:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(N_BOOT):
        rows = []
        for g in rng.choice(valid, p_sub, replace=False):
            gr = grows[g]
            rows.append(gr if len(gr) <= CELLS_PER_PROTEIN else rng.choice(gr, CELLS_PER_PROTEIN, replace=False))
        acc.append(mean_cos_dist(np.asarray(emb[np.sort(np.concatenate(rows))])))
    lo, hi = np.percentile(acc, [2.5, 97.5])
    return float(np.mean(acc)), float(lo), float(hi)


# ------------------------------------------------------------------ steps
def cmd_prep():
    """Real HPA per-cell features, U2OS only: gene per row + the embedding matrix."""
    import torch
    obj = torch.load(REAL_PTH, map_location="cpu", weights_only=False)      # (metadata DataFrame, tensor)
    meta = next(v for v in obj if isinstance(v, pd.DataFrame)).reset_index(drop=True)
    emb = np.asarray(next(v.numpy() for v in obj if torch.is_tensor(v)), dtype=np.float32)
    sel = np.flatnonzero(meta["atlas_name"].astype(str).eq("U2OS").to_numpy())
    m = pd.DataFrame({"gene": meta["gene_names"].astype(str).iloc[sel].values})
    m["row"] = np.arange(len(m))
    CACHE.mkdir(parents=True, exist_ok=True)
    m.to_parquet(CACHE / "real_u2os_meta.parquet")
    np.save(CACHE / "real_u2os_emb.npy", emb[sel])
    print(f"{len(m)} U2OS cells, {m['gene'].nunique()} genes")


def cmd_nodes():
    """One row per tree node at levels 2-7: members and the co-clustering score."""
    labels, cols = tl.load_cluster_labels(TREE_DATA / "cluster_labels_generated.tsv")
    cohesion = json.load(open(TREE_DATA / "cohesion_by_level_cache.json"))
    rows = []
    for level, col in enumerate(cols[1:], start=2):
        for cid, sub in labels.groupby(col):
            genes = sub["gene_name"].astype(str).tolist()
            rows.append(dict(level=level, node_id=int(cid), n_proteins=len(genes), genes=",".join(genes),
                             cohesion=float(cohesion[col].get(str(int(cid)), np.nan))))
    df = pd.DataFrame(rows).sort_values(["level", "node_id"]).reset_index(drop=True)
    df.to_parquet(CACHE / "tree_nodes.parquet")
    print(f"{len(df)} nodes; per level:", df.groupby("level").size().to_dict())


def cmd_spread():
    """Real spread per node at a per-level protein budget: the smallest kept node's protein count."""
    nodes = pd.read_parquet(CACHE / "tree_nodes.parquet")
    grows = gene_rows(pd.read_parquet(CACHE / "real_u2os_meta.parquet"))
    emb = np.load(CACHE / "real_u2os_emb.npy", mmap_mode="r")
    members = {i: [g for g in r.genes.split(",") if g in grows] for i, r in nodes.iterrows()}
    n_cells = {i: sum(len(grows[g]) for g in gs) for i, gs in members.items()}
    kept = [i for i in nodes.index if n_cells[i] >= MIN_NODE_CELLS]
    budget = {lvl: max(3, min(min(len(members[i]) for i in kept if nodes.level[i] == lvl), P_SUB_CAP))
              for lvl in sorted({nodes.level[i] for i in kept})}
    print("protein budget per level:", budget, flush=True)
    out = []
    for i in kept:
        r = nodes.loc[i]
        p_sub = budget[r.level]
        mean, lo, hi = protein_budget_spread(members[i], grows, emb, p_sub)
        out.append(dict(level=int(r.level), node_id=int(r.node_id), cohesion=float(r.cohesion),
                        n_proteins=int(r.n_proteins), n_cells=n_cells[i], p_sub=int(p_sub),
                        cos_dist_psub=mean, cos_dist_psub_lo=lo, cos_dist_psub_hi=hi))
    df = pd.DataFrame(out)
    df.to_parquet(CACHE / "real_node_spread.parquet")
    print(f"{len(df)} nodes kept; per level:", df.groupby("level").size().to_dict())


def cmd_coloc():
    """Co-occupancy concentration sum(H^2)/sum(H) of each node's Fig 6B heatmap (keys (level, id))."""
    with gzip.open(TREE_DATA / "heatmaps_cache.pkl.gz", "rb") as fh:
        H = pickle.load(fh)
    rows = [dict(level=lvl, node_id=int(cid), coloc_conc=float((hm * hm).sum() / hm.sum()))
            for (lvl, cid), hm in H.items() if lvl > 1 and hm.sum() > 0]
    df = pd.DataFrame(rows).sort_values(["level", "node_id"]).reset_index(drop=True)
    df.to_parquet(CACHE / "node_coloc.parquet")
    print(f"{len(df)} nodes; coloc_conc {df.coloc_conc.min():.3f}..{df.coloc_conc.max():.3f}")


def cmd_merge():
    rs = pd.read_parquet(CACHE / "real_node_spread.parquet")
    cl = pd.read_parquet(CACHE / "node_coloc.parquet")
    df = rs.merge(cl, on=["level", "node_id"], how="left")
    (HERE / "data").mkdir(exist_ok=True)
    df.to_parquet(HERE / "data" / "node_master.parquet")
    print(f"node_master: {len(df)} nodes x {df.shape[1]} columns")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("prep", "nodes", "spread", "coloc", "merge"))
    globals()[f"cmd_{ap.parse_args().step}"]()
