"""The consensus tree, compute side: node cohesion from the consensus matrices and per-node
co-occupancy heatmaps from the stacked image arrays.

Heavy: the consensus matrices are 12,800 x 12,800; the stacks are per-cell image arrays.
Reads the cell's labels from <out-dir>/cell_<id>/cluster_labels_generated.tsv (cell_tree.py) and
writes, next to them, cohesion_by_level_cache.json and heatmaps_cache.pkl.gz (keyed by
(level, cluster id)), which plot_tree_heatmaps.py and cluster_validation.py read. The stack is
the Proteome2Cell image array of the cell (512 x 512 x 12,800 uint8) with its protein order.

    python build_tree_caches.py --cell-id 134 --consensus-dir .../consensus_leiden/U2OS \\
        --stack .../U2OS_superplexed_134_stacked_images.npy --protein-order .../protein_order.pkl --out-dir OUT
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "utils")]
import treelib as tl  # noqa: E402

CONSENSUS_RESOLUTIONS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]     # levels 2-7 of the tree, by position
THRESHOLD_MULT = 1.25


# ------------------------------------------------------------------ cohesion
def cluster_cohesion(M, members) -> float:
    """Mean off-diagonal consensus among a cluster's members; a singleton is 1.0."""
    n = len(members)
    if n <= 1:
        return 1.0
    sub = M[np.ix_(members, members)]
    return float((sub.sum() - np.trace(sub)) / (n * (n - 1)))


def cohesion_from_consensus(genes, cols, labels_by_col, consensus_dir: Path) -> dict:
    """{level_col: {cluster_id: cohesion}} for levels 2-7 from res_<gamma>/consensus_matrix.npz;
    the matrices are in the label file's gene order."""
    out = {}
    for col, res in zip(cols[1:], CONSENSUS_RESOLUTIONS):
        p = consensus_dir / f"res_{res:.2f}" / "consensus_matrix.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        M = np.load(p)["M"]
        assert M.shape == (len(genes), len(genes)), f"{p}: consensus matrix is not in the label file's gene order"
        labels = labels_by_col[col]
        coh = {int(c): cluster_cohesion(M, np.flatnonzero(labels == c)) for c in np.unique(labels)}
        out[col] = coh
        print(f"  {col}: {len(coh)} clusters, mean cohesion {np.mean(list(coh.values())):.3f}", flush=True)
        del M
    return out


# ------------------------------------------------------------------ heatmaps
def heatmap(stack, gene_indices) -> np.ndarray:
    """Co-occupancy: per gene, Otsu-threshold its plane (x1.25); mean of the binary masks."""
    from skimage.filters import threshold_otsu
    valid = [i for i in gene_indices if i < stack.shape[2]]
    if not valid:
        return np.zeros((tl.TILE_SIZE, tl.TILE_SIZE), dtype=np.float32)
    planes = stack[:tl.TILE_SIZE, :tl.TILE_SIZE, valid].astype(np.float32)
    binary = np.stack([planes[:, :, k] > THRESHOLD_MULT * threshold_otsu(planes[:, :, k])
                       for k in range(planes.shape[2])], axis=2)
    return binary.mean(axis=2).astype(np.float32)


def all_heatmaps(G, stack, gene_to_idx) -> dict:
    return {node: heatmap(stack, [gene_to_idx[g] for g in G.nodes[node]["genes"] if g in gene_to_idx])
            for node in G.nodes}


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--consensus-dir", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--protein-order", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    for p in (a.stack, a.protein_order):
        assert p.exists(), f"missing input: {p}"
    out = a.out_dir / f"cell_{a.cell_id}"
    df, cols = tl.load_cluster_labels(out / "cluster_labels_generated.tsv")
    genes = df["gene_name"].astype(str).tolist()
    labels_by_col = {c: df[c].values.astype(int) for c in cols}
    cohesion = cohesion_from_consensus(genes, cols, labels_by_col, a.consensus_dir)
    json.dump({c: {str(k): v for k, v in d.items()} for c, d in cohesion.items()},
              open(out / "cohesion_by_level_cache.json", "w"))

    G = tl.build_hierarchy(genes, [(lvl, labels_by_col[c]) for lvl, c in enumerate(cols, start=1)], threshold=0.1,
                           cohesion={lvl: cohesion[c] for lvl, c in enumerate(cols[1:], start=2)})
    gene_to_idx = {g: i for i, g in enumerate(pickle.load(open(a.protein_order, "rb")))}
    stack = np.load(str(a.stack), mmap_mode="r")
    heatmaps = all_heatmaps(G, stack, gene_to_idx)
    with gzip.open(out / "heatmaps_cache.pkl.gz", "wb") as fh:
        pickle.dump(heatmaps, fh)
    print(f"{G.number_of_nodes()} nodes, {len(heatmaps)} heatmaps ->", a.out_dir, flush=True)


if __name__ == "__main__":
    main()
