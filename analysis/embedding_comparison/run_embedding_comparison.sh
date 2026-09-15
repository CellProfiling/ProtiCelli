#!/usr/bin/env bash
# Embedding comparison, real vs ProtiCelli-generated SubCell embeddings: the arms (per-line
# real and depth-matched generated protein means), the joint UMAP grids (Fig 3A, S5A) and the
# k-NN location label transfer (Fig 3B).
#
# Plain bash, no scheduler assumptions. The arms (36 jobs), the UMAP fits (12) and the
# transfer scoring (36) are independent per line and are written as loops here; split them
# across a cluster however you like.
#
#   bash run_embedding_comparison.sh                 # every step, in order
#   STEPS="plot" bash run_embedding_comparison.sh    # only re-plot from data/
#
# Inputs (env, defaults are the lab's paths): INPUT_H5AD (the published SubCell input h5ad),
# NORM_V2 (generated shards, location map), MORPH_DIR (morphology-only floor arms), ARMS_DIR.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
STEPS="${STEPS:-arms umaps transfer plot}"
THREADS="${THREADS:-4}"
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS NUMBA_NUM_THREADS=$THREADS
export ARMS_DIR="${ARMS_DIR:-/scratch/groups/emmalu/konstantin/proticelli/paper_v2/arms}"
LINES=("U2OS" "A-431" "U-251MG" "MCF-7" "HEK293" "PC-3" "Rh30" "HeLa" "SH-SY5Y" "CACO-2" "Hep-G2" "BJ [Human fibroblast]")
cd "$HERE"

for step in $STEPS; do
  echo "== $step  $(date -Is)"
  case "$step" in
    arms)      for line in "${LINES[@]}"; do for seed in 0 1 2; do          # 12 lines x 3 sampling seeds -> ARMS_DIR
                 "$PY" utils/build_arms.py build "$line" "$seed"
               done; done ;;
    umaps)     for line in "${LINES[@]}"; do "$PY" -u umaps/fit_umaps.py "$line"; done ;;   # -> umaps/data/panelA_parts/
    transfer)  for line in "${LINES[@]}"; do for samp in 0 1 2; do          # -> label_transfer/data/panelB_parts/
                 "$PY" -u label_transfer/score_transfer.py "$line" "$samp"
               done; done ;;
    plot)      "$PY" umaps/plot_panel_A.py                                  # Fig 3A, S5A
               "$PY" label_transfer/plot_panel_B.py ;;                      # Fig 3B
    *) echo "unknown step: $step" >&2; exit 2 ;;
  esac
done
echo "done  $(date -Is)"
