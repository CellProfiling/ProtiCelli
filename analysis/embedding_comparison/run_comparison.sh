#!/bin/bash
#SBATCH --job-name=comparison_regen
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

#
# SLURM script for running embedding comparison (metrics + UMAP) on REGENERATED embeddings.
#
# Usage:
#   sbatch run_comparison.sh
#
# To run only metrics or only UMAP, set SKIP_METRICS=true or SKIP_VISUAL=true below.
#

# ============================================================
# CONFIGURE: Set these paths for your environment
# ============================================================

# Input paths
REAL_EMBEDDING="${REAL_EMBEDDING:-/path/to/combined_embeddings_harm_microscope.h5ad}"
HPA_CSV="${HPA_CSV:-/path/to/IF-image.csv}"
BASE_DIR="${BASE_DIR:-/path/to/multi_cell}"

# Regenerated embedding filename (relative path from aggregated/ dir)
GEN_EMBEDDING_FILENAME="../regenerated/aggregated/embeddings_aggregated.h5ad"

# Cell lines to compare (space-separated)
CELL_LINES="A-431 BJ CACO-2 HEK293 HeLa Hep-G2 MCF-7 PC-3 Rh30 SH-SY5Y U-251MG U2OS"

# Output directories (metrics and visual can differ)
METRICS_OUTPUT_DIR="${METRICS_OUTPUT_DIR:-${BASE_DIR}/comparison/metrics/restricted_regenerated/microscope_harm_only}"
VISUAL_OUTPUT_DIR="${VISUAL_OUTPUT_DIR:-${BASE_DIR}/comparison/visual_regenerated/microscope_harm_only}"

# ============================================================
# Analysis Control
# ============================================================

SKIP_METRICS=false      # Set to true to skip metrics computation
SKIP_VISUAL=false       # Set to true to skip UMAP visualization

# ============================================================
# Metric Options
# ============================================================

K_NEIGHBORS=15
PR_NEAREST_K=15
SKIP_PRECISION_RECALL=false
SPREAD_KNN_K=5
SAVE_SVG="${SAVE_SVG:-true}"  # Default to true if not set
N_JOBS="${SLURM_CPUS_PER_TASK:-1}"

# ============================================================
# Setup
# ============================================================

source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate scanpy
unset PYTHONPATH

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLAS_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${METRICS_OUTPUT_DIR}"
mkdir -p "${VISUAL_OUTPUT_DIR}"
mkdir -p logs

cd "${SCRIPT_DIR}"

# ============================================================
# Build command
# ============================================================

CMD="python ./compare_embeddings.py"

# Required arguments
CMD="${CMD} --real-embedding ${REAL_EMBEDDING}"
CMD="${CMD} --base-dir ${BASE_DIR}"
CMD="${CMD} --cell-lines ${CELL_LINES}"
CMD="${CMD} --output-dir ${METRICS_OUTPUT_DIR}"
CMD="${CMD} --visual-output-dir ${VISUAL_OUTPUT_DIR}"

# Regenerated embedding filename
CMD="${CMD} --gen-embedding-filename ${GEN_EMBEDDING_FILENAME}"

# Optional HPA CSV
if [ -n "$HPA_CSV" ]; then
    CMD="${CMD} --hpa-csv ${HPA_CSV}"
fi

# Analysis control
if [ "$SKIP_METRICS" = true ]; then
    CMD="${CMD} --skip-metrics"
fi
if [ "$SKIP_VISUAL" = true ]; then
    CMD="${CMD} --skip-visual"
fi

# Metrics options (ignored if --skip-metrics)
CMD="${CMD} --k-neighbors ${K_NEIGHBORS}"
CMD="${CMD} --pr-nearest-k ${PR_NEAREST_K}"
CMD="${CMD} --n-jobs ${N_JOBS}"
CMD="${CMD} --spread-knn-k ${SPREAD_KNN_K}"

if [ "$SKIP_PRECISION_RECALL" = true ]; then
    CMD="${CMD} --skip-precision-recall"
fi
if [ "$SAVE_SVG" = true ]; then
    CMD="${CMD} --save-svg"
fi

# ============================================================
# Run
# ============================================================

echo "=========================================="
echo "Embedding Comparison Analysis (REGENERATED)"
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""
echo "Real embedding: ${REAL_EMBEDDING}"
echo "Base directory: ${BASE_DIR}"
echo "Cell lines: ${CELL_LINES}"
echo "Metrics output: ${METRICS_OUTPUT_DIR}"
echo "Visual output:  ${VISUAL_OUTPUT_DIR}"
echo ""
echo "Options:"
echo "  Skip metrics: ${SKIP_METRICS}"
echo "  Skip visual:  ${SKIP_VISUAL}"
echo "  k-neighbors: ${K_NEIGHBORS}"
echo "  P/R k-neighbors: ${PR_NEAREST_K}"
echo "  Parallel jobs: ${N_JOBS}"
echo "  Skip P/R: ${SKIP_PRECISION_RECALL}"
echo "  Spread k-NN k: ${SPREAD_KNN_K}"
echo "  Save SVG: ${SAVE_SVG}"
echo ""
echo "Command:"
echo "${CMD}"
echo "=========================================="
echo ""

${CMD}

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed"
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
