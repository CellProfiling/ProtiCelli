#!/bin/bash
#SBATCH --job-name=tree_nodes_images_above
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

#
# SLURM script: Hierarchy tree with colored circle nodes + heatmap images above each node
#
# Usage:
#   sbatch run_tree_visualization.sh                            # default: cell 134
#   CELL_IDS="000 006 062" sbatch run_tree_visualization.sh     # explicit cell IDs
#   N_CELLS=20 sbatch run_tree_visualization.sh                 # sample 20 cells
#   SKIP_ENRICHMENT=0 sbatch run_tree_visualization.sh          # with GO enrichment
#   SAVE_SVG=1 sbatch run_tree_visualization.sh                 # also save SVG
#   REPLOT=1 CELL_IDS="134" sbatch run_tree_visualization.sh    # replot from cache
#   N_LEVELS=6 sbatch run_tree_visualization.sh                 # limit hierarchy depth
#   CELL_LINE=BJ sbatch run_tree_visualization.sh               # different cell line
#

# ============================================================
# CONFIGURE: Set these paths for your environment
# ============================================================

BASE_DIR="${BASE_DIR:-/path/to/multi_cell}"
CELL_LINE="${CELL_LINE:-U2OS}"

SINGLE_CELL_DIR="${SINGLE_CELL_DIR:-${BASE_DIR}/comparison/single_cell_trees/${CELL_LINE}}"
CONSENSUS_DIR="${CONSENSUS_DIR:-${BASE_DIR}/comparison/consensus_leiden/all_cells/uncentered/${CELL_LINE}}"

# Cell selection: set CELL_IDS (space-separated) OR N_CELLS (integer)
# CELL_IDS takes priority if both are set.
CELL_IDS="${CELL_IDS:-134}"
N_CELLS="${N_CELLS:-10}"
SEED="${SEED:-42}"

# 6 consensus resolutions mapping to Leiden levels 2-7
CONSENSUS_RESOLUTIONS="${CONSENSUS_RESOLUTIONS:-0.1 0.25 0.5 0.75 1.0 1.5}"

# Skip GO enrichment (0=run enrichment, 1=skip)
SKIP_ENRICHMENT="${SKIP_ENRICHMENT:-0}"

# Also save SVG (0=no, 1=yes)
SAVE_SVG="${SAVE_SVG:-0}"

# Minimum node size to display
MIN_NODE_SIZE="${MIN_NODE_SIZE:-3}"

# Overlap threshold for parent-child edges
OVERLAP_THRESHOLD="${OVERLAP_THRESHOLD:-0.1}"

# Output directory override (default: writes into each cell_{ID}/tree/)
OUTPUT_DIR="${OUTPUT_DIR:-}"

# Replot from cached outputs (0=no, 1=yes)
REPLOT="${REPLOT:-0}"

# Stack image directory
STACK_DIR="${STACK_DIR:-/path/to/cytoVL}"

# Protein order pickle
PROTEIN_ORDER="${PROTEIN_ORDER:-/path/to/protein_order.pkl}"

# Heatmap image zoom (default 0.12 -> ~61px thumbnails)
IMAGE_ZOOM="${IMAGE_ZOOM:-0.12}"

# Output DPI
DPI="${DPI:-150}"

# Heatmap colormap
HEATMAP_CMAP="${HEATMAP_CMAP:-hot}"

# Graphviz spacing
NODESEP="${NODESEP:-3.0}"
RANKSEP="${RANKSEP:-4.0}"

# Number of hierarchy levels to plot (empty = all)
N_LEVELS="${N_LEVELS:-}"

# Cohesion colormap (for node circle colors)
COHESION_CMAP="${COHESION_CMAP:-YlOrRd}"

# ============================================================
# Setup
# ============================================================

echo "============================================================"
echo "Hierarchy Tree: Colored Nodes + Heatmap Images Above"
echo "============================================================"
echo "Cell line:             ${CELL_LINE}"
echo "Single-cell dir:       ${SINGLE_CELL_DIR}"
echo "Consensus dir:         ${CONSENSUS_DIR}"
if [ -n "${CELL_IDS}" ]; then
    echo "Cell IDs:              ${CELL_IDS}"
else
    echo "N cells (sample):      ${N_CELLS}  (seed=${SEED})"
fi
echo "Consensus resolutions: ${CONSENSUS_RESOLUTIONS}"
echo "Skip enrichment:       ${SKIP_ENRICHMENT}"
echo "Save SVG:              ${SAVE_SVG}"
echo "Replot from cache:     ${REPLOT}"
echo "Min node size:         ${MIN_NODE_SIZE}"
echo "Stack dir:             ${STACK_DIR}"
echo "Protein order:         ${PROTEIN_ORDER}"
echo "Image zoom:            ${IMAGE_ZOOM}"
echo "DPI:                   ${DPI}"
echo "Heatmap cmap:          ${HEATMAP_CMAP}"
echo "Nodesep:               ${NODESEP}"
echo "Ranksep:               ${RANKSEP}"
echo "N levels:              ${N_LEVELS:-all}"
echo "Cohesion cmap:         ${COHESION_CMAP}"
if [ -n "${OUTPUT_DIR}" ]; then
    echo "Output dir override:   ${OUTPUT_DIR}"
fi
echo "Start time:            $(date)"
echo "============================================================"

source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate scanpy
unset PYTHONPATH

mkdir -p logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ============================================================
# Build command
# ============================================================

CMD="python ./plot_tree_with_heatmaps.py"
CMD="${CMD} --single-cell-dir ${SINGLE_CELL_DIR}"
CMD="${CMD} --consensus-dir ${CONSENSUS_DIR}"
CMD="${CMD} --base-dir ${BASE_DIR}"
CMD="${CMD} --cell-line ${CELL_LINE}"
CMD="${CMD} --consensus-resolutions ${CONSENSUS_RESOLUTIONS}"
CMD="${CMD} --overlap-threshold ${OVERLAP_THRESHOLD}"
CMD="${CMD} --min-node-size ${MIN_NODE_SIZE}"
CMD="${CMD} --seed ${SEED}"
CMD="${CMD} --stack-dir ${STACK_DIR}"
CMD="${CMD} --protein-order ${PROTEIN_ORDER}"
CMD="${CMD} --image-zoom ${IMAGE_ZOOM}"
CMD="${CMD} --dpi ${DPI}"
CMD="${CMD} --heatmap-cmap ${HEATMAP_CMAP}"
CMD="${CMD} --nodesep ${NODESEP}"
CMD="${CMD} --ranksep ${RANKSEP}"
CMD="${CMD} --cohesion-cmap ${COHESION_CMAP}"
CMD="${CMD} --images-above-nodes"

# Cell selection
if [ -n "${CELL_IDS}" ]; then
    CMD="${CMD} --cell-ids ${CELL_IDS}"
else
    CMD="${CMD} --n-cells ${N_CELLS}"
fi

# Optional output dir override
if [ -n "${OUTPUT_DIR}" ]; then
    CMD="${CMD} --output-dir ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
fi

if [ "${SKIP_ENRICHMENT}" = "1" ]; then
    CMD="${CMD} --skip-enrichment"
fi

if [ "${SAVE_SVG}" = "1" ]; then
    CMD="${CMD} --save-svg"
fi

if [ "${REPLOT}" = "1" ]; then
    CMD="${CMD} --replot"
fi

if [ -n "${N_LEVELS}" ]; then
    CMD="${CMD} --n-levels ${N_LEVELS}"
fi

# ============================================================
# Run
# ============================================================

echo ""
echo "Running: ${CMD}"
echo ""

eval ${CMD}
EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Finished with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "============================================================"

exit ${EXIT_CODE}
