#!/usr/bin/env bash
# Reactome sub-pathway dotplot (Fig 5G): Hep-G2, Metabolism of lipids. Per (sub-pathway x cluster)
# mean log2 spatial enrichment with a one-sided Wilcoxon test, BH-corrected, from the protein-level
# spatial enrichment table of the pixel-level clustering and the Reactome hierarchy (ContentService,
# cached in data/ on first run). Inputs in inputs/.
#
#   bash run_spatial_clustering.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HERE"
"$PY" plot_dotplot.py
