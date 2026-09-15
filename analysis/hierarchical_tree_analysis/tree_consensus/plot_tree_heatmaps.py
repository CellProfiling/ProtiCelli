"""Fig 6B: the consensus hierarchy tree of cell 134, U2OS, six levels

The tree is the hierarchical Leiden clustering of one generated U2OS cell's 12,800 proteins,
drawn with a per-node **co-occupancy heatmap** (where the cluster's proteins sit in the cell) framed in
the node's **co-clustering score** (cohesion: how consistently the cluster's proteins co-cluster across
single cells, from the multi-resolution consensus matrices of `consensus_leiden.py`). The
caches this needs are written by `cell_tree.py` and `build_tree_caches.py`; the script draws
`figures/F_consensus_tree.{png,svg}`.

| input | role |
|---|---|
| `data/cell_<id>/cluster_labels_generated.tsv` | per-protein cluster label at each of the 7 levels (levels 1–6 drawn) |
| `data/cell_<id>/cohesion_by_level_cache.json` | per-node cohesion (the frame colour) |
| `data/cell_<id>/heatmaps_cache.pkl.gz` | 169 node heatmaps keyed (level, cluster id), 512×512 float32 (177 MB, gzipped to 38 MB) |

The cell is `CELL_ID` (default 134). Nodes are labelled `<level>.<display id>`, display ids
counting left to right in the layout.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "utils")]
from paperlib import setup
setup(threads=4)       # SVG hash salt, SOURCE_DATE_EPOCH, BLAS threads -- before matplotlib is imported

import gzip, json, pickle
import numpy as np, pandas as pd
import treelib as tl

CELL = HERE / "data" / f"cell_{os.environ.get('CELL_ID', '134')}"
FIG = HERE / "figures"
N_LEVELS = 6

# ------------------------------------------------------------------------------
# 1. Labels, cohesion, heatmaps
# ------------------------------------------------------------------------------
# The 7-level labels, first six levels kept; the cohesion cache ({level column: {cluster id:
# cohesion}}) and the heatmaps. 12,800 proteins, cluster counts 1, 2, 6, 14, 22, 38 over the six
# levels.

df, cols = tl.load_cluster_labels(CELL / "cluster_labels_generated.tsv")
cols = cols[:N_LEVELS]
genes = df["gene_name"].tolist()
labels = {c: df[c].values.astype(int) for c in cols}
cohesion = {c: {int(k): v for k, v in vals.items()}
            for c, vals in json.loads((CELL / "cohesion_by_level_cache.json").read_text()).items()}
with gzip.open(CELL / "heatmaps_cache.pkl.gz", "rb") as fh:
    heatmaps = pickle.load(fh)
print(len(genes), "proteins; clusters per level:", [len(np.unique(labels[c])) for c in cols], "; heatmaps:", len(heatmaps))

# ------------------------------------------------------------------------------
# 2. The tree
# ------------------------------------------------------------------------------
# Level 1 (one cluster) is the root (1, 0); levels 2-6 become nodes (level, cluster id) carrying
# their genes, size and cohesion; a parent -> child edge needs > 10 % of the child's genes in the
# parent. Node insertion order is the layout's tie-breaker and is kept as shipped. 83 nodes.

G = tl.build_hierarchy(genes, [(lvl, labels[c]) for lvl, c in enumerate(cols, start=1)], threshold=0.1,
                       cohesion={lvl: cohesion[c] for lvl, c in enumerate(cols[1:], start=2)})
print(G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

# ------------------------------------------------------------------------------
# 3. The figure
# ------------------------------------------------------------------------------
# `dot` layout, rescaled so the tightest same-level gap equals one thumbnail step; each node is its
# heatmap tile behind a rounded frame coloured by cohesion (the root is 1.0 by definition); edge
# width follows the child's size (treelib.render_heatmap_tree and its constants). Cohesion per
# level is summarised after the figure.

FIG.mkdir(exist_ok=True)
out = FIG / "F_consensus_tree"
display_id = tl.render_heatmap_tree(G, heatmaps, out)
stats = pd.DataFrame({c: dict(n_clusters=len(v), mean=np.mean(list(v.values())), std=np.std(list(v.values())),
                              min=min(v.values()), max=max(v.values())) for c, v in cohesion.items()}).T.round(3)
print(stats)
