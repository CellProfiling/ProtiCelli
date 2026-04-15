#!/bin/bash
#SBATCH --job-name=single_cell_trees
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

#
# Single-Cell Hierarchy Tree Pipeline
#
# Per-cell: tunes Leiden resolutions, generates hierarchical cluster labels,
# and plots hierarchy trees. Then computes cross-cell clustering consistency
# (pairwise ARI/AMI) across all processed cells.
#
# Usage:
#   sbatch run_single_cell_trees.sh                            # random 50 cells
#   N_CELLS=20 sbatch run_single_cell_trees.sh                 # random 20 cells
#   CELL_IDS="0,50,90,134" sbatch run_single_cell_trees.sh     # specific cells
#   FORCE_REGENERATE=true sbatch run_single_cell_trees.sh      # reprocess all
#   RUN_GO_ENRICHMENT=true sbatch run_single_cell_trees.sh     # include GO enrichment
#   CELL_LINE=HeLa sbatch run_single_cell_trees.sh             # different cell line
#

# ============================================================
# CONFIGURE: Set these paths for your environment
# ============================================================

CELL_LINE="${CELL_LINE:-U2OS}"
N_CELLS="${N_CELLS:-50}"
SEED="${SEED:-42}"
BASE_DIR="${BASE_DIR:-/path/to/multi_cell}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/comparison/single_cell_trees/${CELL_LINE}}"
FORCE_REGENERATE="${FORCE_REGENERATE:-false}"
RUN_GO_ENRICHMENT="${RUN_GO_ENRICHMENT:-false}"

TARGET_COUNTS="1 2 6 14 22 38 86"
NEIGHBORS="125 100 90 55 40 25 10"
METRIC="euclidean"
MAX_ITER=1000

CELL_BASE="${BASE_DIR}/${CELL_LINE}/regenerated/cell_outputs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Setup
# ============================================================

echo "=========================================="
echo "Single-Cell Hierarchy Tree Pipeline"
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node:         ${SLURM_NODELIST:-local}"
echo "Start time:   $(date)"
echo ""
echo "Configuration:"
echo "  Cell line:        ${CELL_LINE}"
echo "  Output dir:       ${OUTPUT_DIR}"
echo "  Force regenerate: ${FORCE_REGENERATE}"
echo "  GO enrichment:    ${RUN_GO_ENRICHMENT}"
if [ -n "${CELL_IDS}" ]; then
    echo "  Cell IDs:         ${CELL_IDS}"
else
    echo "  N cells:          ${N_CELLS} (seed=${SEED})"
fi
echo "=========================================="
echo ""

source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate scanpy
unset PYTHONPATH

mkdir -p "${OUTPUT_DIR}"
cd "${SCRIPT_DIR}"

# ============================================================
# Cell selection
# ============================================================

if [ -n "${CELL_IDS}" ]; then
    # Specific cells mode: parse comma/space-separated IDs
    SAMPLED_CELLS=$(python3 -c "
ids = '${CELL_IDS}'.replace(',', ' ').split()
formatted = sorted(set(f'{int(x):03d}' for x in ids))
print(' '.join(formatted))
")
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to parse CELL_IDS='${CELL_IDS}'"
        exit 1
    fi

    # Validate source embeddings exist
    MISSING=0
    for CID in ${SAMPLED_CELLS}; do
        if [ ! -f "${CELL_BASE}/cell_${CID}/subcell_output/embeddings.h5ad" ]; then
            echo "WARNING: No embeddings for cell_${CID}"
            MISSING=$((MISSING + 1))
        fi
    done
    if [ "${MISSING}" -gt 0 ]; then
        echo "WARNING: ${MISSING} cells missing source embeddings (will be skipped)"
    fi
else
    # Random sampling mode: pick N_CELLS, skipping already-processed
    SAMPLED_CELLS=$(python3 - <<EOF
import random, os, sys, re
from pathlib import Path

random.seed(${SEED})
base   = "${CELL_BASE}"
outdir = Path("${OUTPUT_DIR}")
n_want = ${N_CELLS}

all_ids = [f"{i:03d}" for i in range(200)]
available = [c for c in all_ids
             if os.path.exists(f"{base}/cell_{c}/subcell_output/embeddings.h5ad")]
print(f"Available source cells: {len(available)}", file=sys.stderr)
if len(available) == 0:
    print("ERROR: No cells with embeddings found in: " + base, file=sys.stderr)
    sys.exit(1)

already_done = []
if outdir.exists():
    for d in sorted(outdir.glob("cell_*")):
        m = re.match(r'cell_(\d+)', d.name)
        if m and (d / "cluster_labels_generated.tsv").exists():
            already_done.append(m.group(1))
print(f"Already processed: {len(already_done)} cells", file=sys.stderr)

if len(already_done) >= n_want:
    print(f"Already have {len(already_done)} >= {n_want} cells processed", file=sys.stderr)
    print("")
    sys.exit(0)

already_set = set(already_done)
remaining   = [c for c in available if c not in already_set]
n_more      = n_want - len(already_done)
n_sample    = min(n_more, len(remaining))
if n_sample == 0:
    print("WARNING: No additional source cells available", file=sys.stderr)
    print("")
    sys.exit(0)

new_cells = sorted(random.sample(remaining, n_sample))
print(f"New cells to process: {new_cells}", file=sys.stderr)
print(" ".join(new_cells))
EOF
)
    if [ $? -ne 0 ]; then
        echo "ERROR: Cell sampling failed"
        exit 1
    fi
fi

TOTAL_CELLS=$(echo "${SAMPLED_CELLS}" | wc -w)
echo "Cells to process: ${SAMPLED_CELLS}"
echo "Total: ${TOTAL_CELLS}"
echo ""

if [ "${TOTAL_CELLS}" -eq 0 ]; then
    echo "No new cells to process — running cross-cell consistency only"
fi

# ============================================================
# Per-cell pipeline (Steps 1-5)
# ============================================================

CELL_COUNT=0
FAILED_CELLS=()

for CELL_ID in ${SAMPLED_CELLS}; do
    CELL_COUNT=$((CELL_COUNT + 1))
    CELL_OUTPUT="${OUTPUT_DIR}/cell_${CELL_ID}"
    CELL_H5AD="${CELL_BASE}/cell_${CELL_ID}/subcell_output/embeddings.h5ad"
    PROCESSED_H5AD="${CELL_OUTPUT}/embeddings_processed.h5ad"
    TUNING_DIR="${CELL_OUTPUT}/tuning"
    TUNING_RESULTS="${TUNING_DIR}/tuning_results.json"
    LABELS_FILE="${CELL_OUTPUT}/cluster_labels_generated.tsv"
    TREE_DIR="${CELL_OUTPUT}/tree"
    TREE_PNG="${TREE_DIR}/hierarchy_tree_generated.png"
    ENRICH_DIR="${CELL_OUTPUT}/go_enrichment"

    echo "=========================================="
    echo "Cell ${CELL_COUNT}/${TOTAL_CELLS}: cell_${CELL_ID}"
    echo "=========================================="

    if [ ! -f "${CELL_H5AD}" ]; then
        echo "  SKIPPING: No source embeddings at ${CELL_H5AD}"
        FAILED_CELLS+=("cell_${CELL_ID}")
        continue
    fi

    mkdir -p "${CELL_OUTPUT}" "${TUNING_DIR}" "${TREE_DIR}" "${ENRICH_DIR}"

    # ----------------------------------------------------------
    # Step 1: Preprocess h5ad (add gene_name column)
    # ----------------------------------------------------------

    if [ -f "${PROCESSED_H5AD}" ] && [ "${FORCE_REGENERATE}" != "true" ]; then
        echo "  [1/5] Preprocessed h5ad exists, skipping"
    else
        echo "  [1/5] Preprocessing h5ad..."
        python3 - <<EOF
import scanpy as sc, sys

def canonicalize_gene_ids(values):
    values = values.astype(str)
    return [value.split('_', 3)[-1] if '_' in value else value for value in values]

adata = sc.read_h5ad("${CELL_H5AD}")
print("  Loaded: " + str(adata.shape[0]) + " proteins x " + str(adata.shape[1]) + " features")

if "gene_name" not in adata.obs.columns:
    if "index" in adata.obs.columns:
        adata.obs["gene_name"] = canonicalize_gene_ids(adata.obs["index"])
        print("  Added canonical gene_name from obs['index']")
    else:
        adata.obs["gene_name"] = adata.obs_names.astype(str)
        print("  Added gene_name from obs_names")
else:
    print("  gene_name already present")

adata.write_h5ad("${PROCESSED_H5AD}")
print("  Saved: ${PROCESSED_H5AD}")
EOF
        if [ $? -ne 0 ]; then
            echo "  ERROR: Step 1 failed for cell_${CELL_ID}"
            FAILED_CELLS+=("cell_${CELL_ID}")
            continue
        fi
    fi

    # ----------------------------------------------------------
    # Step 2: Tune clustering parameters
    # ----------------------------------------------------------

    if [ -f "${TUNING_RESULTS}" ] && [ "${FORCE_REGENERATE}" != "true" ]; then
        echo "  [2/5] Tuning results exist, skipping"
    else
        echo "  [2/5] Tuning clustering parameters..."
        python ../utils/tune_clustering_parameters.py \
            --mode generated \
            --gen-embedding "${PROCESSED_H5AD}" \
            --target-counts ${TARGET_COUNTS} \
            --metric "${METRIC}" \
            --max-iter ${MAX_ITER} \
            --output-dir "${TUNING_DIR}"
        if [ $? -ne 0 ]; then
            echo "  ERROR: Step 2 failed for cell_${CELL_ID}"
            FAILED_CELLS+=("cell_${CELL_ID}")
            continue
        fi
    fi

    # ----------------------------------------------------------
    # Step 3: Generate clustering labels
    # ----------------------------------------------------------

    if [ -f "${LABELS_FILE}" ] && [ "${FORCE_REGENERATE}" != "true" ]; then
        echo "  [3/5] Cluster labels exist, skipping"
    else
        echo "  [3/5] Generating clustering labels..."
        RESOLUTIONS=$(python3 -c "import json; d=json.load(open('${TUNING_RESULTS}')); print(' '.join(map(str, d['resolutions'])))")
        if [ $? -ne 0 ]; then
            echo "  ERROR: Could not read resolutions from ${TUNING_RESULTS}"
            FAILED_CELLS+=("cell_${CELL_ID}")
            continue
        fi
        echo "  Resolutions: ${RESOLUTIONS}"

        python ../utils/generate_clustering_labels.py \
            --mode generated \
            --gen-embedding "${PROCESSED_H5AD}" \
            --gen-resolutions ${RESOLUTIONS} \
            --gen-neighbors ${NEIGHBORS} \
            --metric "${METRIC}" \
            --output-dir "${CELL_OUTPUT}"
        if [ $? -ne 0 ]; then
            echo "  ERROR: Step 3 failed for cell_${CELL_ID}"
            FAILED_CELLS+=("cell_${CELL_ID}")
            continue
        fi
    fi

    # ----------------------------------------------------------
    # Step 4: Plot hierarchy tree
    # ----------------------------------------------------------

    if [ -f "${TREE_PNG}" ] && [ "${FORCE_REGENERATE}" != "true" ]; then
        echo "  [4/5] Hierarchy tree exists, skipping"
    else
        echo "  [4/5] Plotting hierarchy tree..."
        python ../utils/compare_clustering_trees.py \
            "${LABELS_FILE}" \
            --output-dir "${TREE_DIR}" \
            --tree-only \
            --max-levels 7
        if [ $? -ne 0 ]; then
            echo "  ERROR: Step 4 failed for cell_${CELL_ID}"
            FAILED_CELLS+=("cell_${CELL_ID}")
            continue
        fi
    fi

    # ----------------------------------------------------------
    # Step 5: GO enrichment (optional, off by default)
    # ----------------------------------------------------------

    if [ "${RUN_GO_ENRICHMENT}" = "true" ]; then
        ENRICH_DONE=$(ls "${ENRICH_DIR}"/*.tsv 2>/dev/null | wc -l)
        if [ "${ENRICH_DONE}" -gt 0 ] && [ "${FORCE_REGENERATE}" != "true" ]; then
            echo "  [5/5] GO enrichment exists (${ENRICH_DONE} files), skipping"
        else
            echo "  [5/5] Running GO enrichment per cluster..."
            python3 - <<EOF
import pandas as pd, sys
from pathlib import Path

try:
    from gprofiler import GProfiler
except ImportError:
    print("WARNING: gprofiler not available, skipping GO enrichment", file=sys.stderr)
    sys.exit(0)

labels_file = "${LABELS_FILE}"
enrich_dir  = Path("${ENRICH_DIR}")
enrich_dir.mkdir(parents=True, exist_ok=True)

labels_df = pd.read_csv(labels_file, sep='\t')
gp = GProfiler(return_dataframe=True)

leiden_cols = [c for c in labels_df.columns if c.startswith('leiden_')]
print("  Found " + str(len(leiden_cols)) + " leiden columns")

total_enrichments = 0
for col in leiden_cols:
    for cluster_id in labels_df[col].unique():
        genes = labels_df.loc[labels_df[col] == cluster_id, 'gene_name'].tolist()
        if len(genes) < 3:
            continue
        try:
            results = gp.profile(
                organism='hsapiens',
                query=genes,
                sources=['GO:CC', 'GO:MF', 'GO:BP']
            )
            out_path = enrich_dir / (col + "_cluster_" + str(cluster_id) + ".tsv")
            results.to_csv(out_path, sep='\t', index=False)
            total_enrichments += 1
        except Exception as e:
            print("  WARNING: enrichment failed for " + col + " cluster " + str(cluster_id) + ": " + str(e),
                  file=sys.stderr)

print("  Saved " + str(total_enrichments) + " enrichment files to " + str(enrich_dir))
EOF
            if [ $? -ne 0 ]; then
                echo "  WARNING: GO enrichment failed for cell_${CELL_ID} (non-fatal)"
            fi
        fi
    else
        echo "  [5/5] GO enrichment skipped (set RUN_GO_ENRICHMENT=true to enable)"
    fi

    echo "  cell_${CELL_ID} complete"
    echo ""
done

# ============================================================
# Step 6: Cross-cell clustering consistency
# ============================================================

echo "=========================================="
echo "STEP 6: Cross-cell clustering consistency"
echo "=========================================="
echo ""

python ./cross_cell_consistency.py --output-dir "${OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "WARNING: Cross-cell consistency analysis failed"
fi

# ============================================================
# Summary
# ============================================================

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
TOTAL_DONE=$(find "${OUTPUT_DIR}" -maxdepth 2 -name "cluster_labels_generated.tsv" 2>/dev/null | wc -l)
echo "Cells processed this run: ${CELL_COUNT}"
echo "Total cells in output:    ${TOTAL_DONE}"
echo "Output directory:         ${OUTPUT_DIR}"

if [ ${#FAILED_CELLS[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: Failed cells:"
    for FC in "${FAILED_CELLS[@]}"; do
        echo "  - ${FC}"
    done
fi

echo ""
echo "End time: $(date)"
echo "=========================================="
