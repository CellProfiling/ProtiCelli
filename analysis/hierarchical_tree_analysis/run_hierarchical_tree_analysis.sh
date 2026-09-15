#!/usr/bin/env bash
# Hierarchical Leiden trees: the generated hierarchy tree, the subcluster UMAPs and the AMI ladder
# with its reproducibility ceiling (Fig 3C, 3D), the per-cell consensus tree of cell 134 with
# co-occupancy heatmaps (Fig 6B) and its co-clustering validation against real single-cell spread
# (Fig S10).
#
# Plain bash, no scheduler assumptions. The ladder (30 runs), the ceiling (30 runs) and the
# consensus sweep are the heavy steps; they are loops here, split them across a cluster as you
# like. The ladder and ceiling were run on AMD Zen 3 (EPYC 7763/7543) with four threads; the
# tuner's argmax over near-tied modularities is deterministic per CPU family, not across them.
#
#   bash run_hierarchical_tree_analysis.sh                 # every step, in order
#   STEPS="plot" bash run_hierarchical_tree_analysis.sh    # only re-plot from data/
#
# Inputs (env, defaults are the lab's paths): ARMS_DIR and INPUT_H5AD (see ../embedding_comparison),
# HALVES_DIR (split halves), EMBEDDINGS (the SubCell embeddings of every generated U2OS cell, one
# h5ad with obs cell_id and gene_name), STACK and PROTEIN_ORDER (the cell's Proteome2Cell image
# array and protein order), HPA_REAL_FEATURES (Fig S10), VALIDATION_CACHE.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
STEPS="${STEPS:-consensus cell_tree tree_caches halves ceiling ladder pick enrich panel_d_umap validation plot}"
THREADS="${THREADS:-4}"
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS NUMBA_NUM_THREADS=$THREADS
SUBCELL_EVAL_DIR="${SUBCELL_EVAL_DIR:-/home/groups/emmalu/konstantin/scratch/subcell_evaluation}"
export CELL_ID="${CELL_ID:-134}"
EMBEDDINGS="${EMBEDDINGS:-$SUBCELL_EVAL_DIR/multi_cell/U2OS/regenerated/aggregated/embeddings_all_cells.h5ad}"
CONSENSUS_DIR="${CONSENSUS_DIR:-$SUBCELL_EVAL_DIR/multi_cell/comparison/consensus_leiden/all_cells/uncentered/U2OS}"   # the sweep behind the shipped cohesion (uncentred per-cell Leiden)
STACK="${STACK:-/scratch/users/sunh/cytoVL/U2OS_superplexed_${CELL_ID}_stacked_images.npy}"                     # 512x512x12800 uint8
PROTEIN_ORDER="${PROTEIN_ORDER:-/scratch/groups/emmalu/konstantin/subcell/norm_v2/p2c_recovered/protein_order.pkl}"   # gene -> stack index
cd "$HERE"

for step in $STEPS; do
  echo "== $step  $(date -Is)"
  case "$step" in
    consensus)    "$PY" tree_consensus/consensus_leiden.py --embeddings "$EMBEDDINGS" --output-dir "$CONSENSUS_DIR" \
                    --resolutions 0.1 0.25 0.5 0.75 1.0 1.5 --n-neighbors 25 --metric cosine \
                    --n-cores "${N_CORES:-20}" --seed 42 ;;                        # -> CONSENSUS_DIR/res_*/
    cell_tree)    "$PY" tree_consensus/cell_tree.py --embeddings "$EMBEDDINGS" --cell-id "$CELL_ID" \
                    --out-dir tree_consensus/data ;;                                # -> tree_consensus/data/cell_<id>/
    tree_caches)  "$PY" tree_consensus/build_tree_caches.py --cell-id "$CELL_ID" --consensus-dir "$CONSENSUS_DIR" \
                    --stack "$STACK" --protein-order "$PROTEIN_ORDER" --out-dir tree_consensus/data ;;
    halves)       "$PY" -u tree_comparison/panelCD_cluster.py halves --splits 0 1 2 ;;   # split-half real embeddings (HALVES_DIR)
    ceiling)      for split in 0 1 2; do for clust in $(seq 0 9); do              # -> tree_comparison/data/ceiling/
                    "$PY" -u tree_comparison/panelCD_cluster.py ceiling --split "$split" --clust "$clust"
                  done; done ;;
    ladder)       for samp in 0 1 2; do for clust in $(seq 0 9); do               # -> tree_comparison/data/{ladders,labels}/
                    "$PY" -u tree_comparison/panelCD_cluster.py ladder --samp "$samp" --clust "$clust"
                  done; done ;;
    pick)         "$PY" tree_comparison/panelCD_cluster.py pick ;;                # ladder tables, representative run, renumbering
    enrich)       "$PY" tree_comparison/panelCD_cluster.py enrich ;;              # GO terms per tree node (g:Profiler, network)
    panel_d_umap) "$PY" tree_comparison/panelCD_cluster.py panel-d-umap ;;        # -> tree_comparison/data/umap_cache/
    validation)   for s in prep nodes spread coloc merge; do                       # -> cluster_validation/data/node_master.parquet
                    "$PY" cluster_validation/cluster_validation.py "$s"
                  done ;;
    plot)         "$PY" tree_comparison/plot_panels_C_D.py                        # Fig 3C, 3D
                  "$PY" tree_consensus/plot_tree_heatmaps.py                      # Fig 6B
                  "$PY" cluster_validation/plot_fig_S10.py ;;                     # Fig S10
    *) echo "unknown step: $step" >&2; exit 2 ;;
  esac
done
echo "done  $(date -Is)"
