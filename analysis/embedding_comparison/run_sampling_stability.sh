#!/bin/bash
#SBATCH --job-name=sampling_stability
#SBATCH --mem=300G
#SBATCH --cpus-per-task=20

#
# SLURM script for running cell sampling stability analysis
#
# This script analyzes how many cells are needed for stable mean protein embeddings
# by performing bootstrap sampling with varying sample sizes.
#
# Usage:
#   sbatch run_sampling_stability.sh
#
# Edit the variables below to customize for your dataset.
#

# ============================================================
# CONFIGURE: Set these paths for your environment
# ============================================================

# Input paths
BASE_DIR="${BASE_DIR:-/path/to/multi_cell}"

# Cell lines to analyze (space-separated)
# Available: A-431 BJ CACO-2 HEK293 HeLa Hep-G2 MCF-7 PC-3 Rh30 SH-SY5Y U-251MG U2OS
CELL_LINES="A-431 BJ CACO-2 HEK293 HeLa Hep-G2 MCF-7 PC-3 Rh30 SH-SY5Y U-251MG U2OS"

# Subdirectory under each cell line folder containing embeddings_all_cells.h5ad
GEN_SUBDIR="regenerated/aggregated"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/comparison/sampling_stability_regenerated}"

# ============================================================
# Analysis Options
# ============================================================

# Number of genes to randomly sample for analysis
N_GENES=200

# Number of bootstrap iterations per sample size
N_ITERATIONS=1000

# Random seed for reproducibility
RANDOM_SEED=42

# ============================================================
# Setup
# ============================================================

# Activate environment
source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate scanpy
unset PYTHONPATH

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ============================================================
# Build command
# ============================================================

CMD="python ./analyze_sampling_stability.py --source generated"

# Required arguments
CMD="${CMD} --base-dir ${BASE_DIR}"
CMD="${CMD} --cell-lines ${CELL_LINES}"
CMD="${CMD} --output-dir ${OUTPUT_DIR}"

# Gen subdir
CMD="${CMD} --gen-subdir ${GEN_SUBDIR}"

# Analysis options
CMD="${CMD} --n-genes ${N_GENES}"
CMD="${CMD} --n-iterations ${N_ITERATIONS}"
CMD="${CMD} --random-seed ${RANDOM_SEED}"

# ============================================================
# Run
# ============================================================

echo "=========================================="
echo "Cell Sampling Stability Analysis"
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: 300GB"
echo "Start time: $(date)"
echo ""
echo "Base directory: ${BASE_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Cell lines: ${CELL_LINES}"
echo ""
echo "Analysis Options:"
echo "  Genes to analyze: ${N_GENES}"
echo "  Iterations per sample size: ${N_ITERATIONS}"
echo "  Random seed: ${RANDOM_SEED}"
echo "  Sample sizes: [1, 2, 4, 8, 12, 16, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]"
echo ""
echo "Command:"
echo "${CMD}"
echo "=========================================="
echo ""

# Execute
${CMD}

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed"
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
