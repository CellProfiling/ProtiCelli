"""The single-cell hierarchy tree (Fig 6B), compute side: the 7-level Leiden ladder of one
simulated cell's per-protein SubCell embeddings, with the same tuner as the U2OS real-vs-simulated ladder
(utils/hier.py: targets 1, 2, 6, 14, 22, 38, 86; neighbourhoods 125 .. 10; euclidean).

    python cell_tree.py --embeddings E.h5ad --out-dir OUT [--cell-id 134] [--seed 42]

E.h5ad holds every generated cell of the line (obs cell_id, gene_name; the consensus_leiden.py
input); the cell is the rows whose numeric cell_id equals --cell-id. Writes
OUT/cell_<id>/cluster_labels_generated.tsv (gene_name + leiden_lvl<N>_res<R> per level, the
format every tree script reads) and clustering_metadata.json (the tuner record per level).
"""
import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "utils")]
import hier as H  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cell-id", default="134")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    adata = ad.read_h5ad(a.embeddings)
    adata = adata[adata.obs["cell_id"].astype(str).map(int).to_numpy() == int(a.cell_id)]
    assert adata.n_obs, f"no rows with cell_id {a.cell_id} in {a.embeddings}"
    genes = adata.obs["gene_name"].astype(str).to_numpy()
    X = np.asarray(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, np.float32)
    labels, meta = H.run_arm(X, a.seed)
    out = a.out_dir / f"cell_{a.cell_id}"
    out.mkdir(parents=True, exist_ok=True)
    labels = labels.rename(columns={"level%d" % (i + 1): "leiden_lvl%d_res%s" % (i, repr(float(meta["records"][i]["res"])))
                                    for i in range(len(H.TARGETS))})
    labels.insert(0, "gene_name", genes)
    labels.to_csv(out / "cluster_labels_generated.tsv", sep="\t", index=False)
    json.dump(dict(cell_id=a.cell_id, embeddings=str(a.embeddings), **meta), open(out / "clustering_metadata.json", "w"), indent=1)
    print(f"cell {a.cell_id}: {len(genes)} proteins ->", out, flush=True)


if __name__ == "__main__":
    main()
