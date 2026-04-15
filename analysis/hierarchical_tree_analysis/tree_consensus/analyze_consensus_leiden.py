#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Resolution Consensus Leiden Clustering Pipeline

Runs flat Leiden clustering at multiple resolutions on each cell independently,
then builds a consensus co-occurrence matrix for each resolution.

Algorithm:
1. For each resolution gamma in the sweep:
   a. Per-cell flat Leiden: For each cell, build kNN graph on embeddings, run Leiden at gamma
   b. Co-occurrence matrices: For each cell c, build binary matrix A^(c) where A_{ij}=1
      if proteins i,j are in the same cluster
   c. Consensus: M_gamma = (1/N) * sum(A^(c))
2. Generate hierarchically-ordered clustermap heatmaps of M at each resolution

Outputs consumed downstream:
- res_X.XX/consensus_matrix.npz: Used by tree coloring scripts (cohesion scores)
- res_X.XX/clustermap.png: Paper figure

Author: Konstantin Kahnert
"""

import numpy as np
import scanpy as sc
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import argparse
import sys
import json
import logging
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
from centering_utils import center_cell_embedding

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "embedding_comparison"))
from embedding_utils import save_figure as _save_figure

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================
# Data Loading
# ========================================

def _extract_gene_names(adata: sc.AnnData) -> List[str]:
    """Extract canonical gene names from an AnnData object."""
    if 'gene_name' in adata.obs.columns:
        return adata.obs['gene_name'].astype(str).tolist()
    if 'index' in adata.obs.columns:
        indices = adata.obs['index'].astype(str).tolist()
        return [idx.split('_', 3)[-1] if '_' in idx else idx for idx in indices]
    return adata.obs_names.astype(str).tolist()



def load_individual_cell_embeddings(
    cell_line_dir: Path,
    max_cells: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Load all individual cell embeddings for a cell line.

    Returns:
        embeddings_list: List of (n_proteins, n_features) arrays
        cell_ids: List of cell IDs
        gene_names: List of protein/gene names (consistent across cells)
    """
    logger.info(f"Loading cell embeddings from {cell_line_dir}")

    cell_outputs_dir = cell_line_dir / "cell_outputs"
    if not cell_outputs_dir.exists():
        raise ValueError(f"Cell outputs directory not found: {cell_outputs_dir}")

    cell_dirs = sorted(
        [d for d in cell_outputs_dir.iterdir() if d.is_dir() and d.name.startswith("cell_")],
        key=lambda x: int(x.name.split("_")[1])
    )

    if not cell_dirs:
        raise ValueError(f"No cell directories found in {cell_outputs_dir}")

    logger.info(f"Found {len(cell_dirs)} cell directories")

    if max_cells is not None and max_cells < len(cell_dirs):
        cell_dirs = cell_dirs[:max_cells]
        logger.info(f"Limiting to first {max_cells} cells")

    embeddings_list = []
    cell_ids = []
    gene_names_list = []

    for cell_dir in tqdm(cell_dirs, desc="Loading cells"):
        h5ad_path = cell_dir / "subcell_output" / "embeddings.h5ad"

        if not h5ad_path.exists():
            logger.warning(f"Embeddings file not found: {h5ad_path}, skipping")
            continue

        try:
            adata = sc.read_h5ad(h5ad_path)
            genes = _extract_gene_names(adata)

            embedding = adata.X
            if hasattr(embedding, 'toarray'):
                embedding = embedding.toarray()
            embedding = np.asarray(embedding, dtype=np.float32)

            embeddings_list.append(embedding)
            cell_ids.append(cell_dir.name)
            gene_names_list.append(genes)

        except Exception as e:
            logger.error(f"Error loading {h5ad_path}: {e}")
            continue

    if not embeddings_list:
        raise ValueError("No valid cell embeddings loaded")

    logger.info(f"Successfully loaded {len(embeddings_list)} cells")

    canonical_genes = _validate_gene_consistency(gene_names_list)

    return embeddings_list, cell_ids, canonical_genes


def _validate_gene_consistency(gene_lists: List[List[str]]) -> List[str]:
    """Validate all cells have same genes in same order."""
    if not gene_lists:
        raise ValueError("No gene lists provided")

    reference_genes = gene_lists[0]

    for i, genes in enumerate(gene_lists[1:], start=1):
        if len(genes) != len(reference_genes):
            raise ValueError(
                f"Cell {i} has {len(genes)} genes, but cell 0 has {len(reference_genes)}"
            )
        if genes != reference_genes:
            if set(genes) == set(reference_genes):
                raise ValueError(
                    f"Cell {i} has same genes but different order."
                )
            else:
                diff = set(genes) ^ set(reference_genes)
                raise ValueError(
                    f"Cell {i} has different genes. First few differences: "
                    f"{list(diff)[:5]}"
                )

    logger.info(f"Gene consistency validated: {len(reference_genes)} genes across all cells")
    return reference_genes


def load_gene_names_from_first_cell(cell_line_dir: Path) -> List[str]:
    """Load gene names by reading only the first cell's h5ad. Fast (~1s)."""
    cell_outputs_dir = cell_line_dir / "cell_outputs"
    cell_dirs = sorted(
        [d for d in cell_outputs_dir.iterdir() if d.is_dir() and d.name.startswith("cell_")],
        key=lambda x: int(x.name.split("_")[1])
    )
    for cell_dir in cell_dirs:
        h5ad_path = cell_dir / "subcell_output" / "embeddings.h5ad"
        if h5ad_path.exists():
            adata = sc.read_h5ad(h5ad_path)
            genes = _extract_gene_names(adata)
            logger.info(f"Loaded {len(genes)} gene names from {cell_dir.name}")
            return genes
    raise ValueError(f"No valid cell h5ad found in {cell_outputs_dir}")


# ========================================
# Per-Cell Leiden Clustering
# ========================================

def _cluster_single_cell(
    embedding: np.ndarray,
    resolution: float,
    n_neighbors: int,
    metric: str,
    seed: int,
    center: bool = True,
) -> np.ndarray:
    """
    Run Leiden clustering on a single cell's embedding.

    If center=True, embedding is centered by subtracting the per-cell centroid
    before kNN graph construction.

    Returns:
        labels: integer cluster labels for each protein
    """
    if center:
        embedding = center_cell_embedding(embedding)
    adata = sc.AnnData(X=embedding)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X', metric=metric)
    sc.tl.leiden(
        adata,
        resolution=resolution,
        flavor='leidenalg',
        n_iterations=2,
        random_state=seed,
    )
    labels = adata.obs['leiden'].astype(int).values
    return labels


# ========================================
# Consensus Matrix Construction
# ========================================

def build_consensus_matrix(
    all_labels: List[np.ndarray],
    n_proteins: int,
) -> np.ndarray:
    """
    Build consensus co-occurrence matrix from per-cell cluster labels.

    M_{ij} = fraction of cells where proteins i and j are in the same cluster.
    """
    n_cells = len(all_labels)
    M = np.zeros((n_proteins, n_proteins), dtype=np.float32)

    for labels in tqdm(all_labels, desc="Building consensus matrix"):
        for cluster_id in np.unique(labels):
            members = np.where(labels == cluster_id)[0]
            M[np.ix_(members, members)] += 1

    M /= n_cells
    np.fill_diagonal(M, 1.0)

    return M


# ========================================
# Multi-Resolution Sweep
# ========================================

def run_resolution_sweep(
    embeddings_list: List[np.ndarray],
    resolutions: List[float],
    n_neighbors: int,
    metric: str,
    seed: int,
    n_cores: int,
    output_dir: Path,
    center: bool = True,
) -> Tuple[Dict[float, Path], Dict[str, dict]]:
    """
    Run per-cell Leiden + consensus matrix for each resolution.

    Returns:
        Tuple of:
            consensus_paths: Dict mapping resolution to path of saved consensus matrix .npz
            diagnostics: Dict mapping resolution string to diagnostic stats
    """
    n_cells = len(embeddings_list)
    n_proteins = embeddings_list[0].shape[0]
    consensus_paths = {}
    all_diagnostics = {}

    for res in resolutions:
        res_str = f"{res:.2f}"
        res_dir = output_dir / f"res_{res_str}"
        res_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*50}")
        logger.info(f"Resolution gamma={res}")
        logger.info(f"{'='*50}")

        # Per-cell Leiden
        t0 = time.time()
        logger.info(f"  Running per-cell Leiden on {n_cores} cores...")

        all_labels = Parallel(n_jobs=n_cores, prefer='threads')(
            delayed(_cluster_single_cell)(
                emb, res, n_neighbors, metric, seed, center=center,
            )
            for emb in embeddings_list
        )

        t_leiden = time.time() - t0

        cluster_counts = [len(np.unique(labels)) for labels in all_labels]
        logger.info(f"  Per-cell clusters: mean={np.mean(cluster_counts):.1f}, "
                    f"min={np.min(cluster_counts)}, max={np.max(cluster_counts)}")

        # Save per-cell labels
        np.savez_compressed(
            res_dir / 'per_cell_labels.npz',
            labels=np.array(all_labels),
        )

        # Build consensus matrix
        t0 = time.time()
        M = build_consensus_matrix(all_labels, n_proteins)
        t_consensus = time.time() - t0

        # Diagnostics
        triu_idx = np.triu_indices_from(M, k=1)
        off_diag = M[triu_idx]
        diag = {
            'mean': float(off_diag.mean()),
            'median': float(np.median(off_diag)),
            'std': float(off_diag.std()),
            'max': float(off_diag.max()),
            'pct_above_0.3': float(np.mean(off_diag >= 0.3)),
            'pct_above_0.5': float(np.mean(off_diag >= 0.5)),
            'pct_above_0.7': float(np.mean(off_diag >= 0.7)),
            'pct_above_0.9': float(np.mean(off_diag >= 0.9)),
            'mean_clusters_per_cell': float(np.mean(cluster_counts)),
            'time_leiden_s': round(t_leiden, 1),
            'time_consensus_s': round(t_consensus, 1),
        }
        all_diagnostics[res_str] = diag
        logger.info(f"  Off-diagonal M: mean={diag['mean']:.4f}, "
                    f"median={diag['median']:.4f}, max={diag['max']:.4f}")
        logger.info(f"  Pairwise fractions above thresholds: "
                    f"0.3={diag['pct_above_0.3']:.4f}, "
                    f"0.5={diag['pct_above_0.5']:.4f}, "
                    f"0.7={diag['pct_above_0.7']:.4f}, "
                    f"0.9={diag['pct_above_0.9']:.4f}")

        # Save consensus matrix
        npz_path = res_dir / 'consensus_matrix.npz'
        np.savez_compressed(npz_path, M=M)
        consensus_paths[res] = npz_path
        logger.info(f"  Saved consensus matrix to {npz_path}")

        del M, off_diag, all_labels
        gc.collect()

    return consensus_paths, all_diagnostics


# ========================================
# Consensus Clustermaps
# ========================================

def plot_consensus_clustermap(
    M: np.ndarray,
    gene_names: List[str],
    resolution: float,
    output_dir: Path,
    save_svg: bool = False,
):
    """
    Plot hierarchically-ordered heatmap of the full consensus matrix.

    Uses average-linkage on condensed distance matrix, extracts leaf ordering
    via leaves_list (non-recursive), reorders M and plots as imshow.
    """
    res_str = f"{resolution:.2f}"
    res_dir = output_dir / f"res_{res_str}"
    res_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Generating clustermap for resolution {res_str} ({M.shape[0]} proteins)...")
    t0 = time.time()

    distance = 1.0 - M
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method='average')
    order = leaves_list(Z)

    M_ordered = M[np.ix_(order, order)]

    n = M.shape[0]
    fig_width = 12
    fig_height = 11
    min_dpi = max(150, int(np.ceil(n / fig_width)) + 1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(M_ordered, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Consensus Matrix, hierarchically ordered (gamma={res_str})', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.6, label='Co-clustering frequency')

    _save_figure(res_dir / 'clustermap.png', save_svg=save_svg, dpi=min_dpi)
    plt.close()

    elapsed = time.time() - t0
    logger.info(f"    Clustermap saved ({elapsed:.1f}s)")


# ========================================
# Helpers
# ========================================

def discover_existing_consensus_matrices(
    output_dir: Path,
    resolutions: List[float],
) -> Dict[float, Path]:
    """Find consensus_matrix.npz files already on disk for the given resolutions."""
    consensus_paths = {}
    for res in resolutions:
        res_str = f"{res:.2f}"
        npz_path = output_dir / f"res_{res_str}" / "consensus_matrix.npz"
        if npz_path.exists():
            consensus_paths[res] = npz_path
            logger.info(f"  Found existing matrix: {npz_path}")
        else:
            logger.warning(f"  Missing consensus matrix for resolution {res_str}: {npz_path}")
    return consensus_paths


# ========================================
# Main Pipeline
# ========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-Resolution Consensus Leiden Clustering Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--base-dir', type=str, required=True,
                        help='Base directory containing cell line directories')
    parser.add_argument('--cell-line', type=str, required=True,
                        help='Cell line to analyze (e.g., U2OS)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')

    parser.add_argument('--clustermaps-only', action='store_true',
                        help='Skip sweep entirely; load existing consensus matrices '
                             'and only generate clustermaps')

    parser.add_argument('--resolutions', type=float, nargs='+',
                        default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
                        help='Per-cell Leiden resolutions for multi-resolution sweep')

    parser.add_argument('--n-neighbors', type=int, default=50,
                        help='Number of neighbors for kNN graph')
    parser.add_argument('--metric', type=str, default='cosine',
                        help='Distance metric for kNN graph')

    parser.add_argument('--skip-clustermaps', action='store_true',
                        help='Skip clustermap generation (only compute consensus matrices)')
    parser.add_argument('--clustermap-resolutions', type=float, nargs='*', default=None,
                        help='Resolutions for clustermaps (default: all resolutions)')

    parser.add_argument('--n-cores', type=int, default=20,
                        help='Number of parallel cores')
    parser.add_argument('--max-cells', type=int, default=None,
                        help='Maximum number of cells to load (for testing)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-center', action='store_true',
                        help='Skip per-cell centering of embeddings before Leiden clustering')
    parser.add_argument('--save-svg', action='store_true',
                        help='Also save plots as SVG')

    return parser.parse_args()


def main():
    args = parse_args()
    timings = {}
    t_start = time.time()

    output_dir = Path(args.output_dir) / args.cell_line
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Multi-Resolution Consensus Leiden Clustering Pipeline")
    logger.info("=" * 60)
    logger.info(f"Cell line: {args.cell_line}")
    logger.info(f"Resolutions: {args.resolutions}")
    logger.info(f"kNN neighbors: {args.n_neighbors}")
    logger.info(f"Metric: {args.metric}")
    logger.info(f"Center embeddings: {not args.no_center}")
    logger.info(f"Skip clustermaps: {args.skip_clustermaps}")
    logger.info(f"Cores: {args.n_cores}")
    logger.info(f"Max cells: {args.max_cells or 'all'}")
    logger.info(f"Output: {output_dir}")

    cell_line_dir = Path(args.base_dir) / args.cell_line
    if not cell_line_dir.exists():
        logger.error(f"Cell line directory not found: {cell_line_dir}")
        sys.exit(1)

    # ========================================
    # Clustermaps-only fast path
    # ========================================

    if args.clustermaps_only:
        logger.info("")
        logger.info("Mode: clustermaps-only (loading existing consensus matrices)")
        logger.info("-" * 40)

        clustermap_resolutions = args.clustermap_resolutions or args.resolutions
        gene_names = load_gene_names_from_first_cell(cell_line_dir)
        consensus_paths = discover_existing_consensus_matrices(output_dir, clustermap_resolutions)

        if not consensus_paths:
            logger.error("No existing consensus matrices found. Run full sweep first.")
            sys.exit(1)

        t0 = time.time()
        for res in sorted(consensus_paths.keys()):
            data = np.load(consensus_paths[res])
            M = data['M']
            plot_consensus_clustermap(M, gene_names, res, output_dir, save_svg=args.save_svg)
            del M
            gc.collect()
        timings['clustermaps'] = time.time() - t0
        timings['total'] = time.time() - t_start

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Clustermaps complete! ({timings['clustermaps']:.1f}s)")
        logger.info(f"  Output: {output_dir}")
        logger.info("=" * 60)
        return

    # ========================================
    # Phase 1: Load data + multi-resolution sweep
    # ========================================

    logger.info("")
    logger.info("Phase 1: Data loading + multi-resolution per-cell Leiden")
    logger.info("-" * 40)

    t0 = time.time()
    embeddings_list, cell_ids, gene_names = load_individual_cell_embeddings(
        cell_line_dir, max_cells=args.max_cells,
    )
    timings['data_loading'] = time.time() - t0

    n_cells = len(embeddings_list)
    n_proteins = len(gene_names)
    n_features = embeddings_list[0].shape[1]
    logger.info(f"Loaded: {n_cells} cells, {n_proteins} proteins, {n_features} features")

    t0 = time.time()
    consensus_paths, sweep_diagnostics = run_resolution_sweep(
        embeddings_list=embeddings_list,
        resolutions=args.resolutions,
        n_neighbors=args.n_neighbors,
        metric=args.metric,
        seed=args.seed,
        n_cores=args.n_cores,
        output_dir=output_dir,
        center=not args.no_center,
    )
    timings['resolution_sweep'] = time.time() - t0

    del embeddings_list
    gc.collect()

    # ========================================
    # Phase 2: Clustermaps
    # ========================================

    if not args.skip_clustermaps:
        t0 = time.time()
        clustermap_resolutions = args.clustermap_resolutions or args.resolutions
        logger.info(f"\nPhase 2: Consensus Clustermaps")
        logger.info(f"  Resolutions: {clustermap_resolutions}")

        for res in clustermap_resolutions:
            if res not in consensus_paths:
                logger.warning(f"  Resolution {res} not in sweep, skipping clustermap")
                continue
            data = np.load(consensus_paths[res])
            M = data['M']
            plot_consensus_clustermap(M, gene_names, res, output_dir, save_svg=args.save_svg)
            del M
            gc.collect()

        timings['clustermaps'] = time.time() - t0

    # ========================================
    # Save run summary
    # ========================================

    timings['total'] = time.time() - t_start

    summary = {
        'parameters': {
            'cell_line': args.cell_line,
            'resolutions': args.resolutions,
            'n_neighbors': args.n_neighbors,
            'metric': args.metric,
            'center_embeddings': not args.no_center,
            'skip_clustermaps': args.skip_clustermaps,
            'clustermap_resolutions': args.clustermap_resolutions,
            'n_cores': args.n_cores,
            'max_cells': args.max_cells,
            'seed': args.seed,
        },
        'data': {
            'n_cells': n_cells,
            'n_proteins': n_proteins,
            'n_features': n_features,
            'cell_ids': cell_ids,
        },
        'sweep_diagnostics': sweep_diagnostics,
        'timings_seconds': {k: round(v, 1) for k, v in timings.items()},
    }

    with open(output_dir / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Total time: {timings['total']:.1f}s")
    for k, v in timings.items():
        if k != 'total':
            logger.info(f"    {k}: {v:.1f}s")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
