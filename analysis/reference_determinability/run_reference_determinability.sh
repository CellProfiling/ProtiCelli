#!/usr/bin/env bash
# Reference-channel determinability (Fig 2I, Fig S4B-C): how much of a protein's localisation the three
# landmark channels (MT, NUC, ER) determine, per single cell, then per protein and compartment.
#
# Plain bash, no scheduler assumptions. Run it as is on a workstation, or hand any step (the
# 1000-shard worker loop is the expensive one, ~1000 CPU-hours) to the scheduler of your choice.
#
#   bash run_reference_determinability.sh                 # every step, in order
#   STEPS="plot" bash run_reference_determinability.sh    # only re-plot from data/
#   STEPS="worker" SHARDS="$(seq 0 99)" bash ...          # a subset of shards (resumable: DONE_ markers)
#
# Input: HPA v25.1 single-cell crops (HPA_CROPS: five 16-bit PNGs per cell + crops_index.parquet).
# Outputs land in data/ (see determinability.py), figures in figures/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
STEPS="${STEPS:-index codebooks worker ceiling aggregate ksweep plot}"
NSHARDS="${NSHARDS:-1000}"
SHARDS="${SHARDS:-$(seq 0 $((NSHARDS - 1)))}"
N_JOBS="${N_JOBS:-16}"
export HPA_CROPS="${HPA_CROPS:-/scratch/users/ankitgpt/HPA_v25_1/crops}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$HERE"

for step in $STEPS; do
  echo "== $step  $(date -Is)"
  case "$step" in
    index)      "$PY" determinability.py index ;;                          # single-cell index (1.05 M cells)
    codebooks)  "$PY" determinability.py codebooks ;;                      # K=32 single + joint K=32..4096 codebooks
    worker)     for s in $SHARDS; do                                       # X per group, every cell
                  "$PY" determinability.py worker --shard "$s" --nshards "$NSHARDS"
                done ;;
    ceiling)    "$PY" determinability.py ceiling --n-jobs "$N_JOBS" ;;     # landmark-from-landmark ceiling
    aggregate)  "$PY" determinability.py aggregate --nshards "$NSHARDS" ;; # per-protein + compartment tables
    ksweep)     "$PY" determinability.py ksweep --n-jobs "$N_JOBS" ;;      # joint-K robustness (methods)
    plot)       "$PY" plot_determinability.py ;;                           # Fig 2I, S4B, S4C from data/
    *) echo "unknown step: $step" >&2; exit 2 ;;
  esac
done
echo "done  $(date -Is)"
