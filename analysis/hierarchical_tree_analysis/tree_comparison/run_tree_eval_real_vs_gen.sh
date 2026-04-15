#!/bin/bash
#SBATCH --job-name=tree_eval_real-gen_regen
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#
# SLURM script for Real vs Generated Tree Evaluation Pipeline
#
# This script compares hierarchical clustering between REAL (HPA) and GENERATED embeddings
# for a single cell line (typically U2OS).
#
# IMPORTANT: This script includes a critical fix for the gene subsetting issue:
#   - Real tuning: Uses all HPA genes for the specified cell line
#   - Generated tuning: Subsets to ONLY genes present in real embeddings
#   - This ensures both achieve the same target cluster counts [1, 2, 6, 14, 22, 38, 86]
#   - Without this fix, cluster counts mismatch and comparisons are invalid
#
# Pipeline steps:
#   1. Tune clustering parameters for real embeddings
#   2. Generate clustering labels for real embeddings
#   3. Tune clustering parameters for generated (WITH gene subsetting)
#   4. Generate clustering labels for both real and generated
#   5. Compare clustering trees (ARI, AMI, V-Measure)
#   6. Visualize UMAPs (optional)
#
# Usage:
#   # Default run (uses harmonized embeddings)
#   sbatch run_tree_eval_real_vs_gen.sh
#
#   # Skip UMAP generation (faster)
#   GENERATE_UMAPS=false sbatch run_tree_eval_real_vs_gen.sh
#
#   # Force regenerate all steps (skip resume)
#   FORCE_REGENERATE=true sbatch run_tree_eval_real_vs_gen.sh
#
#   # Use different cell line
#   CELL_LINE="HeLa" sbatch run_tree_eval_real_vs_gen.sh
#
#   # Use different embedding file
#   REAL_EMBEDDING="/path/to/embeddings.pth" sbatch run_tree_eval_real_vs_gen.sh
#

# ============================================================
# CONFIGURE: Set these paths for your environment
# ============================================================

# Input paths (can be overridden via environment variables)
REAL_EMBEDDING="${REAL_EMBEDDING:-/path/to/harmonized_features_if_plate_id_microscope.pth}"
HPA_CSV="${HPA_CSV:-/path/to/IF-image.csv}"
BASE_DIR="${BASE_DIR:-/path/to/multi_cell}"

# Cell line to analyze (can be overridden)
CELL_LINE="${CELL_LINE:-U2OS}"

# Generated embeddings path
GEN_EMBEDDING="${GEN_EMBEDDING:-${BASE_DIR}/${CELL_LINE}/regenerated/aggregated/embeddings_aggregated.h5ad}"

# Output directory (can be overridden)
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/comparison/trees/${CELL_LINE}/harmonized/regenerated}"

# ============================================================
# Clustering Parameters
# ============================================================

# Target cluster counts at each hierarchy level (same as original HPA analysis)
TARGET_COUNTS="1 2 6 14 22 38 86"

# Distance metric (consistent with original analysis)
METRIC="euclidean"

# Maximum iterations for parameter tuning
MAX_ITER=1000

# Neighbors for embeddings (same as original HPA analysis)
GEN_NEIGHBORS="125 100 90 55 40 25 10"

# ============================================================
# Pipeline Control Options
# ============================================================

# Generate UMAP visualizations (can be slow, set to false to skip)
GENERATE_UMAPS="${GENERATE_UMAPS:-true}"

# Force regenerate all steps (set to true to ignore existing outputs)
FORCE_REGENERATE="${FORCE_REGENERATE:-true}"

# Maximum hierarchy levels for tree plot (set via env var, default: 7)
MAX_LEVELS="${MAX_LEVELS:-7}"

# Save SVG outputs (disabled by default)
SAVE_SVG="${SAVE_SVG:-true}"

# ============================================================
# Setup
# ============================================================

echo "=========================================="
echo "Real vs Generated Tree Evaluation Pipeline"
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Cell line: ${CELL_LINE}"
echo "  Real embedding: ${REAL_EMBEDDING}"
echo "  Generated embedding: ${GEN_EMBEDDING}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Target counts: ${TARGET_COUNTS}"
echo "  Metric: ${METRIC}"
echo "  Generate UMAPs: ${GENERATE_UMAPS}"
echo "  Force regenerate: ${FORCE_REGENERATE}"
echo "  Max tree levels: ${MAX_LEVELS}"
echo "  Save SVG: ${SAVE_SVG}"
echo "=========================================="
echo ""

# Activate environment
source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate scanpy
unset PYTHONPATH

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/real"
mkdir -p "${OUTPUT_DIR}/generated"
mkdir -p "${OUTPUT_DIR}/comparison_results"
if [ "${GENERATE_UMAPS}" == "true" ]; then
    mkdir -p "${OUTPUT_DIR}/comparison_results/umap_visualizations"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to script directory
cd "${SCRIPT_DIR}"

# ============================================================
# Step 1: Tune clustering parameters for REAL embeddings
# ============================================================

echo ""
echo "=========================================="
echo "STEP 1: Tuning clustering parameters for REAL embeddings"
echo "=========================================="
echo ""

REAL_RESOLUTIONS_FILE="${OUTPUT_DIR}/real/tuning_results.json"

# Check if already completed
if [ -f "${REAL_RESOLUTIONS_FILE}" ] && [ "${FORCE_REGENERATE}" != "true" ]; then
    echo "✓ Real tuning already completed, skipping..."
    echo "  Using existing: ${REAL_RESOLUTIONS_FILE}"
else
    echo "Tuning parameters for real ${CELL_LINE} embeddings..."

    python ../utils/tune_clustering_parameters.py \
        --mode real \
        --real-embedding "${REAL_EMBEDDING}" \
        --cell-line "${CELL_LINE}" \
        --hpa-csv "${HPA_CSV}" \
        --target-counts ${TARGET_COUNTS} \
        --metric "${METRIC}" \
        --output-dir "${OUTPUT_DIR}/real" \
        --max-iter ${MAX_ITER}

    if [ $? -ne 0 ]; then
        echo "ERROR: Real parameter tuning failed!"
        exit 1
    fi

    echo "  ✓ Tuning complete: ${REAL_RESOLUTIONS_FILE}"
fi

# Read the tuned resolutions from the JSON output file
if [ ! -f "${REAL_RESOLUTIONS_FILE}" ]; then
    echo "ERROR: Real resolutions file not found: ${REAL_RESOLUTIONS_FILE}"
    exit 1
fi

# Extract resolutions array from JSON and convert to space-separated string
REAL_RESOLUTIONS=$(python3 -c "import json; data=json.load(open('${REAL_RESOLUTIONS_FILE}')); print(' '.join(map(str, data['resolutions'])))")
REAL_CLUSTER_COUNTS=$(python3 -c "import json; data=json.load(open('${REAL_RESOLUTIONS_FILE}')); print(' '.join(map(str, data['cluster_counts'])))")

echo ""
echo "✓ Real embeddings ready for clustering"
echo "  Tuned resolutions: ${REAL_RESOLUTIONS}"
echo "  Achieved cluster counts: ${REAL_CLUSTER_COUNTS}"
echo ""

# ============================================================
# Step 2: Generate clustering labels for REAL embeddings
# ============================================================
# This must happen before Step 3 so we have the real gene list for subsetting

echo ""
echo "=========================================="
echo "STEP 2: Generating clustering labels for REAL embeddings"
echo "=========================================="
echo ""

REAL_LABELS="${OUTPUT_DIR}/cluster_labels_real_${CELL_LINE}.tsv"

# Check if already completed
if [ -f "${REAL_LABELS}" ] && [ "${FORCE_REGENERATE}" != "true" ]; then
    echo "✓ Real labels already generated, skipping..."
    echo "  Using existing: ${REAL_LABELS}"
else
    echo "Generating labels for real ${CELL_LINE} embeddings..."

    python ../utils/generate_clustering_labels.py \
        --mode real \
        --real-embedding "${REAL_EMBEDDING}" \
        --hpa-csv "${HPA_CSV}" \
        --cell-line "${CELL_LINE}" \
        --real-resolutions ${REAL_RESOLUTIONS} \
        --metric "${METRIC}" \
        --output-dir "${OUTPUT_DIR}"

    if [ $? -ne 0 ]; then
        echo "ERROR: Real label generation failed!"
        exit 1
    fi

    echo "  ✓ Labels generated: ${REAL_LABELS}"
fi

# Verify the file exists
if [ ! -f "${REAL_LABELS}" ]; then
    echo "ERROR: Real labels file not found: ${REAL_LABELS}"
    exit 1
fi

# Count real genes
REAL_GENE_COUNT=$(tail -n +2 "${REAL_LABELS}" | wc -l)

echo ""
echo "✓ Real labels ready"
echo "  Labels file: ${REAL_LABELS}"
echo "  Number of genes: ${REAL_GENE_COUNT}"
echo ""

# ============================================================
# Step 3: Tune clustering parameters for GENERATED embeddings
# ============================================================
# CRITICAL FIX: Subset to real genes for tuning to match label generation

echo ""
echo "=========================================="
echo "STEP 3: Tuning clustering parameters for GENERATED embeddings"
echo "=========================================="
echo "  WITH GENE SUBSETTING (critical fix for cluster count match)"
echo ""

GEN_RESOLUTIONS_FILE="${OUTPUT_DIR}/generated/tuning_results.json"

# Check if already completed
if [ -f "${GEN_RESOLUTIONS_FILE}" ] && [ "${FORCE_REGENERATE}" != "true" ]; then
    echo "✓ Generated tuning already completed, skipping..."
    echo "  Using existing: ${GEN_RESOLUTIONS_FILE}"
else
    echo "Tuning parameters for generated ${CELL_LINE} embeddings..."
    echo "  Subsetting to ${REAL_GENE_COUNT} genes from real embeddings"

    python ../utils/tune_clustering_parameters.py \
        --mode generated \
        --gen-embedding "${GEN_EMBEDDING}" \
        --subset-genes "${REAL_LABELS}" \
        --target-counts ${TARGET_COUNTS} \
        --metric "${METRIC}" \
        --output-dir "${OUTPUT_DIR}/generated" \
        --max-iter ${MAX_ITER}

    if [ $? -ne 0 ]; then
        echo "ERROR: Generated parameter tuning failed!"
        exit 1
    fi

    echo "  ✓ Tuning complete: ${GEN_RESOLUTIONS_FILE}"
fi

# Read the tuned resolutions from the JSON output file
if [ ! -f "${GEN_RESOLUTIONS_FILE}" ]; then
    echo "ERROR: Generated resolutions file not found: ${GEN_RESOLUTIONS_FILE}"
    exit 1
fi

# Extract resolutions and cluster counts
GEN_RESOLUTIONS=$(python3 -c "import json; data=json.load(open('${GEN_RESOLUTIONS_FILE}')); print(' '.join(map(str, data['resolutions'])))")
GEN_CLUSTER_COUNTS=$(python3 -c "import json; data=json.load(open('${GEN_RESOLUTIONS_FILE}')); print(' '.join(map(str, data['cluster_counts'])))")

echo ""
echo "✓ Generated embeddings ready for clustering"
echo "  Tuned resolutions: ${GEN_RESOLUTIONS}"
echo "  Achieved cluster counts: ${GEN_CLUSTER_COUNTS}"

# Validate cluster counts match targets
EXPECTED_COUNTS="1 2 6 14 22 38 86"
if [ "${GEN_CLUSTER_COUNTS}" != "${EXPECTED_COUNTS}" ]; then
    echo ""
    echo "⚠ WARNING: Generated tuning did not achieve exact target counts!"
    echo "  Expected: ${EXPECTED_COUNTS}"
    echo "  Achieved: ${GEN_CLUSTER_COUNTS}"
    echo "  This may cause comparison issues."
else
    echo "  ✓ Cluster counts match targets exactly"
fi
echo ""

# ============================================================
# Step 4: Generate clustering labels for GENERATED embeddings
# ============================================================
# Real labels were already generated in Step 2

echo ""
echo "=========================================="
echo "STEP 4: Generating clustering labels for GENERATED embeddings"
echo "=========================================="
echo ""

GEN_LABELS="${OUTPUT_DIR}/cluster_labels_generated_${CELL_LINE}.tsv"

# Check if already completed
if [ -f "${GEN_LABELS}" ] && [ "${FORCE_REGENERATE}" != "true" ]; then
    echo "✓ Generated labels already exist, skipping..."
    echo "  Using existing: ${GEN_LABELS}"
else
    echo "Generating labels for generated ${CELL_LINE} embeddings..."
    echo "  Will subset to ${REAL_GENE_COUNT} genes from real embeddings"

    python ../utils/generate_clustering_labels.py \
        --mode both \
        --real-embedding "${REAL_EMBEDDING}" \
        --hpa-csv "${HPA_CSV}" \
        --cell-line "${CELL_LINE}" \
        --gen-embedding "${GEN_EMBEDDING}" \
        --real-resolutions ${REAL_RESOLUTIONS} \
        --real-neighbors ${GEN_NEIGHBORS} \
        --gen-resolutions ${GEN_RESOLUTIONS} \
        --gen-neighbors ${GEN_NEIGHBORS} \
        --metric "${METRIC}" \
        --subset-to-real-genes \
        --output-dir "${OUTPUT_DIR}"

    if [ $? -ne 0 ]; then
        echo "ERROR: Generated label generation failed!"
        exit 1
    fi

    # Rename to include cell line name
    if [ -f "${OUTPUT_DIR}/cluster_labels_generated.tsv" ]; then
        mv "${OUTPUT_DIR}/cluster_labels_generated.tsv" "${GEN_LABELS}"
    fi

    echo "  ✓ Labels generated: ${GEN_LABELS}"
fi

# Verify the file exists
if [ ! -f "${GEN_LABELS}" ]; then
    echo "ERROR: Generated labels file not found: ${GEN_LABELS}"
    exit 1
fi

# Count generated genes
GEN_GENE_COUNT=$(tail -n +2 "${GEN_LABELS}" | wc -l)

echo ""
echo "✓ Generated labels ready"
echo "  Labels file: ${GEN_LABELS}"
echo "  Number of genes: ${GEN_GENE_COUNT}"
echo ""

# ============================================================
# Step 5: Compare clustering trees
# ============================================================

echo ""
echo "=========================================="
echo "STEP 5: Comparing clustering trees"
echo "=========================================="
echo ""

METRICS_FILE="${OUTPUT_DIR}/comparison_results/clustering_similarity_metrics.tsv"

# Check if already completed
if [ -f "${METRICS_FILE}" ] && [ "${FORCE_REGENERATE}" != "true" ]; then
    echo "✓ Tree comparison already completed, skipping..."
    echo "  Using existing: ${METRICS_FILE}"
else
    echo "Comparing real vs generated clustering trees..."
    echo "  Real labels: ${REAL_LABELS}"
    echo "  Generated labels: ${GEN_LABELS}"

    TREE_COMPARE_CMD=(
        python ../utils/compare_clustering_trees.py
        "${REAL_LABELS}" \
        "${GEN_LABELS}" \
        --output-dir "${OUTPUT_DIR}/comparison_results" \
        --plot-tree \
        --max-levels ${MAX_LEVELS:-7}
    )

    if [ "${SAVE_SVG}" = "true" ]; then
        TREE_COMPARE_CMD+=(--save-svg)
    fi

    "${TREE_COMPARE_CMD[@]}"

    if [ $? -ne 0 ]; then
        echo "ERROR: Tree comparison failed!"
        exit 1
    fi

    echo "  ✓ Comparison complete: ${METRICS_FILE}"
fi

# Display summary statistics
if [ -f "${METRICS_FILE}" ]; then
    echo ""
    echo "Clustering Similarity Metrics:"
    python3 << EOF
import pandas as pd
df = pd.read_csv("${METRICS_FILE}", sep='\t')

def resolve_column(*candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"None of the expected columns were found: {candidates}. Available columns: {list(df.columns)}")

ari_col = resolve_column("ARI", "ari")
ami_col = resolve_column("AMI", "ami")
v_col = resolve_column("V_Measure", "v_measure", "V-Measure")

print(f"  Mean ARI: {df[ari_col].mean():.4f}")
print(f"  Mean AMI: {df[ami_col].mean():.4f}")
print(f"  Mean V-Measure: {df[v_col].mean():.4f}")
EOF
fi

echo ""
echo "✓ Tree comparison complete"
echo ""

# ============================================================
# Step 6: Visualize UMAPs (optional)
# ============================================================

if [ "${GENERATE_UMAPS}" == "true" ]; then
    echo ""
    echo "=========================================="
    echo "STEP 6: Generating UMAP visualizations"
    echo "=========================================="
    echo ""

    UMAP_CHECK="${OUTPUT_DIR}/comparison_results/umap_visualizations/cluster_summary.tsv"

    # Check if already completed
    if [ -f "${UMAP_CHECK}" ] && [ "${FORCE_REGENERATE}" != "true" ]; then
        echo "✓ UMAPs already generated, skipping..."
        echo "  Using existing visualizations"
    else
        echo "Generating UMAP visualizations..."

        UMAP_CMD=(
            python ./visualize_clustering_umap.py
            --real-labels "${REAL_LABELS}" \
            --gen-labels "${GEN_LABELS}" \
            --real-embedding "${REAL_EMBEDDING}" \
            --hpa-csv "${HPA_CSV}" \
            --cell-line "${CELL_LINE}" \
            --gen-embedding "${GEN_EMBEDDING}" \
            --output-dir "${OUTPUT_DIR}/comparison_results/umap_visualizations"
        )

        if [ "${SAVE_SVG}" = "true" ]; then
            UMAP_CMD+=(--save-svg)
        fi

        "${UMAP_CMD[@]}"

        if [ $? -ne 0 ]; then
            echo "WARNING: UMAP visualization failed, continuing..."
        else
            echo "  ✓ UMAPs generated: ${OUTPUT_DIR}/comparison_results/umap_visualizations/"
        fi
    fi

    echo ""
    echo "✓ UMAP visualizations complete"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "STEP 6: UMAP generation skipped (GENERATE_UMAPS=false)"
    echo "=========================================="
    echo ""
fi

# ============================================================
# Summary
# ============================================================

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Cell line: ${CELL_LINE}"
echo "End time: $(date)"
echo ""
echo "All outputs saved to:"
echo "  ${OUTPUT_DIR}/"
echo ""

# ============================================================
# Cluster Count Verification
# ============================================================

echo "Cluster Count Verification:"
echo "  Expected target counts: 1 2 6 14 22 38 86"
echo ""

# Real embeddings
echo "  Real embeddings:"
python3 << EOF
import json
try:
    meta = json.load(open("${OUTPUT_DIR}/clustering_metadata_real.json"))
    counts = meta['cluster_counts']
    print(f"    Achieved: {' '.join(map(str, counts))}")
    if counts == [1, 2, 6, 14, 22, 38, 86]:
        print("    Status: ✓ EXACT MATCH")
    else:
        print("    Status: ✗ MISMATCH")
except FileNotFoundError:
    print("    Status: No metadata file found")
EOF

# Generated embeddings
echo ""
echo "  Generated embeddings:"
python3 << EOF
import json
try:
    meta = json.load(open("${OUTPUT_DIR}/clustering_metadata_generated.json"))
    counts = meta['cluster_counts']
    print(f"    Achieved: {' '.join(map(str, counts))}")
    if counts == [1, 2, 6, 14, 22, 38, 86]:
        print("    Status: ✓ EXACT MATCH")
    else:
        print("    Status: ✗ MISMATCH (cluster counts differ)")
        print("    WARNING: Comparison may be invalid due to count mismatch!")
except FileNotFoundError:
    print("    Status: No metadata file found")
EOF

echo ""

# ============================================================
# Output File Summary
# ============================================================

echo "Key outputs:"
echo ""
echo "  Tuning Results:"
echo "    - ${OUTPUT_DIR}/real/tuning_results.json"
echo "    - ${OUTPUT_DIR}/generated/tuning_results.json"
echo ""
echo "  Cluster Labels:"
echo "    - ${REAL_LABELS} (${REAL_GENE_COUNT} genes)"
echo "    - ${GEN_LABELS} (${GEN_GENE_COUNT} genes)"
echo ""
echo "  Clustering Metadata:"
echo "    - ${OUTPUT_DIR}/clustering_metadata_real.json"
echo "    - ${OUTPUT_DIR}/clustering_metadata_generated.json"
echo ""
echo "  Tree Comparison:"
echo "    - ${OUTPUT_DIR}/comparison_results/clustering_similarity_metrics.tsv"
echo "    - ${OUTPUT_DIR}/comparison_results/clustering_similarity_*.png"
echo ""

if [ "${GENERATE_UMAPS}" == "true" ]; then
    UMAP_COUNT=$(find "${OUTPUT_DIR}/comparison_results/umap_visualizations/" -name "*.png" 2>/dev/null | wc -l)
    echo "  UMAP Visualizations:"
    echo "    - ${OUTPUT_DIR}/comparison_results/umap_visualizations/ (${UMAP_COUNT} PNG files)"
    echo ""
fi

# ============================================================
# Final Recommendations
# ============================================================

echo "Next steps:"
echo "  1. Review cluster count verification above"
echo "  2. Check clustering similarity metrics in comparison_results/"
echo "  3. Examine UMAP visualizations (if generated)"
echo "  4. If cluster counts mismatch, re-run with FORCE_REGENERATE=true"
echo ""
echo "=========================================="

exit 0
