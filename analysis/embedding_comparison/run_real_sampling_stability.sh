#!/bin/bash
#SBATCH --job-name=real_sampling_stability
#SBATCH --mem=300G
#SBATCH --cpus-per-task=20

#
# SLURM script for running real HPA image sampling stability analysis
#
# This script analyzes how many images per gene are needed for stable mean protein embeddings
# by performing bootstrap sampling with varying sample sizes on real HPA data.
#
# Key features:
# - Selects top N genes by image count (not random)
# - Adaptive max sample size per cell line (90th percentile)
# - Samples IMAGES per gene (not cells)
# - Quality filtering applied (Enhanced/Supported/Approved genes only)
#
# Usage:
#   sbatch run_real_sampling_stability.sh
#
# Edit the variables below to customize for your dataset.
#

# ============================================================
# CONFIGURE: Set these paths for your environment
# ============================================================

# Input paths
REAL_EMBEDDING="${REAL_EMBEDDING:-/path/to/combined_embeddings_harm_microscope.h5ad}"
HPA_CSV="${HPA_CSV:-/path/to/IF-image.csv}"

# Cell lines to analyze (space-separated)
# Available: A-431 BJ CACO-2 HEK293 HeLa Hep-G2 MCF-7 PC-3 Rh30 SH-SY5Y U-251MG U2OS
CELL_LINES="A-431 BJ CACO-2 HEK293 HeLa Hep-G2 MCF-7 PC-3 Rh30 SH-SY5Y U-251MG U2OS"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/real_sampling_stability_regenerated}"

# ============================================================
# Analysis Options
# ============================================================

# Number of top genes (by image count) to analyze
N_GENES=100

# Number of bootstrap iterations per sample size
N_ITERATIONS=1000

# Percentile for adaptive max sample size (e.g., 90 = 90th percentile)
PERCENTILE=90

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

CMD="python ./analyze_sampling_stability.py --source real"

# Required arguments
CMD="${CMD} --real-embedding ${REAL_EMBEDDING}"
CMD="${CMD} --hpa-csv ${HPA_CSV}"
CMD="${CMD} --cell-lines ${CELL_LINES}"
CMD="${CMD} --output-dir ${OUTPUT_DIR}"

# Analysis options
CMD="${CMD} --n-genes ${N_GENES}"
CMD="${CMD} --n-iterations ${N_ITERATIONS}"
CMD="${CMD} --percentile ${PERCENTILE}"
CMD="${CMD} --random-seed ${RANDOM_SEED}"

# ============================================================
# Run
# ============================================================

echo "=========================================="
echo "Real HPA Image Sampling Stability Analysis"
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: 300GB"
echo "Start time: $(date)"
echo ""
echo "Real embeddings: ${REAL_EMBEDDING}"
echo "HPA CSV: ${HPA_CSV}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Cell lines: ${CELL_LINES}"
echo ""
echo "Analysis Options:"
echo "  Top genes to analyze: ${N_GENES}"
echo "  Iterations per sample size: ${N_ITERATIONS}"
echo "  Adaptive max sample size: ${PERCENTILE}th percentile"
echo "  Random seed: ${RANDOM_SEED}"
echo "  Sample sizes: Adaptive per cell line (base: [1, 2, 4, 8, 12, 16, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200])"
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
