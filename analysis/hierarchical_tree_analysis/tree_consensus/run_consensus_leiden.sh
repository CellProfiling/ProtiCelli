#!/bin/bash
#SBATCH --job-name=consensus_leiden
#SBATCH --mem=128G
#SBATCH --cpus-per-task=20

#
# SLURM script for Multi-Resolution Consensus Leiden Clustering Pipeline
#
# Builds consensus co-occurrence matrices and generates clustermap heatmaps.
#
# Usage:
#   sbatch run_consensus_leiden.sh
#
#   # Quick test with 5 cells and 2 resolutions
#   MAX_CELLS=5 RESOLUTIONS="0.5 1.0" sbatch run_consensus_leiden.sh
#
#   # Different cell line
#   CELL_LINE=BJ sbatch run_consensus_leiden.sh
#
#   # Only generate clustermaps from existing matrices
#   CLUSTERMAPS_ONLY=1 sbatch run_consensus_leiden.sh
#
#   # Skip clustermaps (only compute consensus matrices)
#   SKIP_CLUSTERMAPS=1 sbatch run_consensus_leiden.sh
#

# ============================================================
# CONFIGURE: Set these paths for your environment
# ============================================================

BASE_DIR="${BASE_DIR:-/path/to/multi_cell}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/comparison/consensus_leiden/multi_res}"

CELL_LINE="${CELL_LINE:-U2OS}"
RESOLUTIONS="${RESOLUTIONS:-0.1 0.25 0.5 0.75 1.0 1.25 1.5}"
N_NEIGHBORS="${N_NEIGHBORS:-25}"
METRIC="${METRIC:-cosine}"

SKIP_CLUSTERMAPS="${SKIP_CLUSTERMAPS:-}"
CLUSTERMAP_RESOLUTIONS="${CLUSTERMAP_RESOLUTIONS:-}"
CLUSTERMAPS_ONLY="${CLUSTERMAPS_ONLY:-}"

NO_CENTER="${NO_CENTER:-}"
N_CORES="${N_CORES:-20}"
MAX_CELLS="${MAX_CELLS:-}"
SEED="${SEED:-42}"
SAVE_SVG="${SAVE_SVG:-0}"

# ============================================================
# Setup
# ============================================================

echo "=========================================="
echo "Multi-Resolution Consensus Leiden Pipeline"
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Base directory: ${BASE_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Cell line: ${CELL_LINE}"
echo "  Resolutions: ${RESOLUTIONS}"
echo "  kNN neighbors: ${N_NEIGHBORS}"
echo "  Metric: ${METRIC}"
echo "  Skip clustermaps: ${SKIP_CLUSTERMAPS:-no}"
echo "  Clustermap resolutions: ${CLUSTERMAP_RESOLUTIONS:-all}"
echo "  Cores: ${N_CORES}"
echo "  Max cells: ${MAX_CELLS:-'all'}"
echo "  Seed: ${SEED}"
echo "  Save SVG: ${SAVE_SVG}"
echo "=========================================="
echo ""

source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate scanpy
unset PYTHONPATH

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ============================================================
# Run Pipeline
# ============================================================

CMD="python ./analyze_consensus_leiden.py \
    --base-dir ${BASE_DIR} \
    --cell-line ${CELL_LINE} \
    --output-dir ${OUTPUT_DIR} \
    --resolutions ${RESOLUTIONS} \
    --n-neighbors ${N_NEIGHBORS} \
    --metric ${METRIC} \
    --n-cores ${N_CORES} \
    --seed ${SEED}"

if [ "${SAVE_SVG}" = "1" ]; then
    CMD="${CMD} --save-svg"
fi

if [ -n "${NO_CENTER}" ]; then
    CMD="${CMD} --no-center"
fi

if [ -n "${MAX_CELLS}" ]; then
    CMD="${CMD} --max-cells ${MAX_CELLS}"
fi

if [ -n "${SKIP_CLUSTERMAPS}" ]; then
    CMD="${CMD} --skip-clustermaps"
fi

if [ -n "${CLUSTERMAPS_ONLY}" ]; then
    CMD="${CMD} --clustermaps-only"
fi

if [ -n "${CLUSTERMAP_RESOLUTIONS}" ]; then
    CMD="${CMD} --clustermap-resolutions ${CLUSTERMAP_RESOLUTIONS}"
fi

echo "Command: ${CMD}"
echo ""

eval ${CMD}

if [ $? -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Pipeline failed!"
    echo "=========================================="
    exit 1
fi

# ============================================================
# Summary
# ============================================================

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  ${OUTPUT_DIR}/${CELL_LINE}/"
echo ""
echo "Output structure:"
echo "  - sweep_summary.json: All parameters and per-resolution diagnostics"
echo ""
echo "  Per-resolution (res_X.XX/):"
echo "    - consensus_matrix.npz: Consensus co-occurrence matrix"
echo "    - per_cell_labels.npz: Per-cell cluster labels"
echo "    - clustermap.png: Full protein-level clustermap"
echo ""
echo "End time: $(date)"
echo "=========================================="

exit 0
