#!/bin/bash
#SBATCH --job-name=plot_umap_paper
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# ============================================================
# CONFIGURE: Set these paths for your environment
# ============================================================
REAL_EMBEDDING="${REAL_EMBEDDING:-/path/to/combined_embeddings_harm_microscope.h5ad}"
HPA_CSV="${HPA_CSV:-/path/to/IF-image.csv}"
BASE_DIR="${BASE_DIR:-/path/to/multi_cell}"
CELL_LINES="A-431 BJ CACO-2 HEK293 HeLa Hep-G2 MCF-7 PC-3 Rh30 SH-SY5Y U-251MG U2OS"
OUTPUT_PATH="${OUTPUT_PATH:-${BASE_DIR}/comparison/visual_regenerated/umap_paper.png}"
REPLOT="${REPLOT:-false}"

source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate scanpy
unset PYTHONPATH

mkdir -p "$(dirname "${OUTPUT_PATH}")"
mkdir -p logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "Publication UMAP Plot"
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"
echo "Real embedding: ${REAL_EMBEDDING}"
echo "Output: ${OUTPUT_PATH}"
echo "Mode: $( [ "${REPLOT}" = "true" ] && echo "replot from cache" || echo "recompute from source" )"
echo "=========================================="

# Recompute from source by default.
# Set REPLOT=true to reuse cached h5ad files.
CMD=(
    python ./plot_umap_paper.py
    --real-embedding "${REAL_EMBEDDING}"
    --hpa-csv "${HPA_CSV}"
    --base-dir "${BASE_DIR}"
    --cell-lines ${CELL_LINES}
    --output-path "${OUTPUT_PATH}"
)

if [ "${REPLOT}" = "true" ]; then
    CMD+=(--replot)
fi

"${CMD[@]}"

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}
