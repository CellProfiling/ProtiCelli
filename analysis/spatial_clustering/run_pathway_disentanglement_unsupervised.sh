#!/bin/bash
#SBATCH --job-name=reactome_heatmap
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# ---------------------------------------------------------------------------
# Reactome sub-pathway heatmap wrapper.
#
# Usage:
#   sbatch run_pathway_disentanglement_unsupervised.sh
#   FOLDER="HeLa_vesicle_transport" \
#   PATHWAY_ID="R-HSA-5653656.4" \
#   TITLE="Vesicle-mediated transport - HeLa" \
#   sbatch run_pathway_disentanglement_unsupervised.sh
# ---------------------------------------------------------------------------

# ============================================================
# CONFIGURE: Set these paths for your environment
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/path/to/pathway_clustering}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

# Paper-default Hep-G2 case; override any of these via env vars for other runs.
DATA_SUBDIR="${DATA_SUBDIR:-reactome_subpathways/pathway_marker_clustering}"
FOLDER="${FOLDER:-Hep-G2_lipid}"
PATHWAY_ID="${PATHWAY_ID:-R-HSA-556833}"
TITLE="${TITLE:-Metabolism of lipids - Hep-G2}"
EXPRESSION_CSV="${EXPRESSION_CSV:-fold_change_global.csv}"
INPUT_MODE="${INPUT_MODE:-foldchange}"
DEPTH="${DEPTH:-2}"
PVAL_THRESHOLD="${PVAL_THRESHOLD:-0.05}"
COLOR_SCHEME="${COLOR_SCHEME:-paul}"
PLOT_TYPE="${PLOT_TYPE:-dotplot}"
OUTPUT_NAME="${OUTPUT_NAME:-fold_change_dotplot.png}"

source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate scanpy
unset PYTHONPATH

echo "====================================================================="
echo "Reactome Heatmap: ${FOLDER}"
echo "Pathway: ${PATHWAY_ID}"
echo "Input mode: ${INPUT_MODE}"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "====================================================================="

cd "${SCRIPT_DIR}"

DIR="${DATA_DIR}/${DATA_SUBDIR}/${FOLDER}"
OUTPUT_DIR="${OUTPUT_DIR:-${DIR}}"
CACHE="${DIR}/reactome_cache.json"

# Auto-detect the cell image (first PNG that is not a heatmap output)
CELL_IMAGE=$(find "${DIR}" -maxdepth 1 -name "*_clustered_cell_*.png" | sort | head -1)
IMAGE_ARGS=()
if [ -n "${CELL_IMAGE}" ]; then
    echo "Using cell image: ${CELL_IMAGE}"
    IMAGE_ARGS=(--image "${CELL_IMAGE}")
fi

echo "--- Generating Reactome plot ---"
python "${SCRIPT_DIR}/pathway_disentanglement.py" \
    --pathway-ids "${PATHWAY_ID}" \
    --expression-csv "${DIR}/${EXPRESSION_CSV}" \
    --input-mode "${INPUT_MODE}" \
    --plot-type "${PLOT_TYPE}" \
    --depth "${DEPTH}" \
    --pval-threshold "${PVAL_THRESHOLD}" \
    --color-scheme "${COLOR_SCHEME}" \
    --output "${OUTPUT_DIR}/${OUTPUT_NAME}" \
    --title "${TITLE}" \
    --cache "${CACHE}" \
    "${IMAGE_ARGS[@]}"

echo ""
echo "====================================================================="
echo "Finished ${FOLDER}: $(date)"
echo "====================================================================="
