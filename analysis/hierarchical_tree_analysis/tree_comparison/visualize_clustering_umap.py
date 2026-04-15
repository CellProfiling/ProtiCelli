#!/usr/bin/env python3
"""Create 2x2 UMAP cluster comparison plots for real vs generated embeddings.

For each hierarchy level 2-7, generates a 2x2 grid:
  Rows = embedding source (Real, Generated)
  Cols = clustering source (Real clusters, Generated clusters)

Author: Konstantin Kahnert
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "embedding_comparison"))
from embedding_utils import (
    load_unnormalized_real_embeddings,
    filter_and_aggregate_real_embeddings_by_cell_line,
    save_figure as _save_figure,
)
from pipeline_utils import setup_logging, load_cluster_labels


def load_embeddings(real_path, gen_path, hpa_csv, cell_line, logger):
    """Load real and generated embeddings."""
    logger.info("Loading embeddings...")
    adata_real = load_unnormalized_real_embeddings(pth_path=real_path)
    adata_real = filter_and_aggregate_real_embeddings_by_cell_line(
        adata_real, cell_line, str(hpa_csv),
    )
    adata_gen = sc.read_h5ad(gen_path)
    logger.info(f"  Real: {adata_real.n_obs} genes, Generated: {adata_gen.n_obs} genes")
    return adata_real, adata_gen


def load_embeddings_gen_gen(gen_path1, gen_path2, cell1, cell2, logger):
    """Load two generated embeddings for gen-gen comparison."""
    logger.info("Loading generated embeddings...")
    adata1 = sc.read_h5ad(gen_path1)
    adata2 = sc.read_h5ad(gen_path2)
    logger.info(f"  {cell1}: {adata1.n_obs} genes, {cell2}: {adata2.n_obs} genes")
    return adata1, adata2


def align_data(adata_real, adata_gen, labels_real, labels_gen, logger):
    """Align all datasets to common gene set, sorted deterministically."""
    gene_sets = [
        set(adata_real.obs['gene_name'].astype(str).str.strip().unique()),
        set(adata_gen.obs['gene_name'].astype(str).str.strip().unique()),
        set(labels_real['gene_name'].astype(str).str.strip().unique()),
        set(labels_gen['gene_name'].astype(str).str.strip().unique()),
    ]
    common = sorted(set.intersection(*gene_sets))
    if not common:
        raise ValueError("No common genes found across all datasets")
    logger.info(f"Aligned to {len(common)} common genes")

    def subset_adata(adata):
        genes = adata.obs['gene_name'].astype(str).str.strip()
        a = adata[genes.isin(common)].copy()
        order = a.obs['gene_name'].astype(str).str.strip().argsort()
        return a[order].copy()

    def subset_labels(df):
        return (df[df['gene_name'].astype(str).str.strip().isin(common)]
                .sort_values('gene_name').reset_index(drop=True))

    return (subset_adata(adata_real), subset_adata(adata_gen),
            subset_labels(labels_real), subset_labels(labels_gen), common)


def compute_umaps(adata_real, adata_gen, logger, n_neighbors=125, random_state=42,
                  cache_dir=None):
    """Compute separate UMAPs for real and generated embeddings, with optional caching."""
    cache_real = Path(cache_dir) / "umap_coords_real.npy" if cache_dir else None
    cache_gen  = Path(cache_dir) / "umap_coords_generated.npy" if cache_dir else None

    if cache_real and cache_real.exists() and cache_gen and cache_gen.exists():
        logger.info("Loading cached UMAP coordinates...")
        adata_real.obsm['X_umap'] = np.load(cache_real)
        adata_gen.obsm['X_umap']  = np.load(cache_gen)
        logger.info(f"  real UMAP: {adata_real.obsm['X_umap'].shape} (cached)")
        logger.info(f"  generated UMAP: {adata_gen.obsm['X_umap'].shape} (cached)")
        return

    logger.info("Computing UMAPs...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, adata in [("real", adata_real), ("generated", adata_gen)]:
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric="cosine", use_rep="X")
            sc.tl.umap(adata, random_state=random_state, min_dist=0.1)
            logger.info(f"  {name} UMAP: {adata.obsm['X_umap'].shape}")

    if cache_dir:
        np.save(cache_real, adata_real.obsm['X_umap'])
        np.save(cache_gen,  adata_gen.obsm['X_umap'])
        logger.info(f"Cached UMAP coords to {cache_dir}")


def create_2x2_plot(adata_real, adata_gen, real_clusters, gen_clusters,
                    level_num, real_col, gen_col, output_path, logger, save_svg=False):
    """Create and save a 2x2 UMAP comparison plot for one hierarchy level."""
    n_real, n_gen = len(set(real_clusters)), len(set(gen_clusters))

    for adata in [adata_real, adata_gen]:
        adata.obs['real_clusters'] = pd.Categorical(real_clusters)
        adata.obs['gen_clusters'] = pd.Categorical(gen_clusters)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    sc.set_figure_params(vector_friendly=True, dpi=300)

    configs = [
        (0, 0, adata_real, 'real_clusters', 'none'),
        (0, 1, adata_real, 'gen_clusters',
         'right margin' if n_gen <= 15 else 'none'),
        (1, 0, adata_gen, 'real_clusters', 'none'),
        (1, 1, adata_gen, 'gen_clusters',
         'right margin' if n_gen <= 15 else 'none'),
    ]
    for r, c, adata, color, legend in configs:
        sc.pl.umap(adata, color=color, ax=axes[r, c], show=False,
                   title='', legend_loc=legend)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel('')
            ax.set_ylabel('')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_title(f"Clustering based on real\n({n_real} clusters)",
                         fontsize=13, fontweight='bold', pad=10)
    axes[0, 1].set_title(f"Clustering based on generated\n({n_gen} clusters)",
                         fontsize=13, fontweight='bold', pad=10)
    axes[0, 0].set_ylabel('Real', fontsize=13, fontweight='bold', labelpad=10)
    axes[1, 0].set_ylabel('Generated', fontsize=13, fontweight='bold', labelpad=10)

    fig.suptitle(f"Level {level_num}: {real_col} vs {gen_col}\n({len(real_clusters)} genes)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(output_path, save_svg=save_svg, dpi=300)
    plt.close()


def save_summary(labels_real, labels_gen, real_cols, gen_cols, common_genes, output_dir, logger):
    """Save cluster count summary for levels 2-7."""
    rows = []
    for i in range(1, 7):
        real_c = labels_real.set_index('gene_name').loc[common_genes, real_cols[i]]
        gen_c = labels_gen.set_index('gene_name').loc[common_genes, gen_cols[i]]
        rows.append({
            'level': i + 1,
            'real_column': real_cols[i],
            'gen_column': gen_cols[i],
            'n_real_clusters': real_c.nunique(),
            'n_gen_clusters': gen_c.nunique(),
            'n_genes': len(common_genes),
        })
    pd.DataFrame(rows).to_csv(output_dir / "cluster_summary.tsv", sep='\t', index=False)
    logger.info("Saved cluster_summary.tsv")


def main():
    parser = argparse.ArgumentParser(description='Create 2x2 UMAP cluster comparison plots')
    parser.add_argument('--mode', choices=['real-gen', 'gen-gen'], default='real-gen')
    parser.add_argument('--real-labels', type=str, required=True)
    parser.add_argument('--gen-labels', type=str, required=True)
    parser.add_argument('--real-embedding', type=str, default=None)
    parser.add_argument('--gen-embedding', type=str, required=True)
    parser.add_argument('--gen-embedding2', type=str, default=None)
    parser.add_argument('--hpa-csv', type=str, default=None)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--cell-line', type=str, default=None)
    parser.add_argument('--cell-line1', type=str, default=None)
    parser.add_argument('--cell-line2', type=str, default=None)
    parser.add_argument('--n-neighbors', type=int, default=125)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save-svg', action='store_true')
    parser.add_argument('--replot', action='store_true',
                        help='Skip UMAP recomputation, use cached coords and replot only')
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'real-gen':
        if not args.real_embedding or not args.hpa_csv:
            parser.error("--real-embedding and --hpa-csv required for mode=real-gen")
        args.cell_line = args.cell_line or 'U2OS'
    elif args.mode == 'gen-gen':
        if not args.gen_embedding2 or not args.cell_line1 or not args.cell_line2:
            parser.error("--gen-embedding2, --cell-line1, --cell-line2 required for mode=gen-gen")

    logger.info("=" * 60)
    logger.info(f"UMAP Cluster Comparison | mode={args.mode}")
    logger.info("=" * 60)

    labels_real, real_cols = load_cluster_labels(Path(args.real_labels))
    labels_gen, gen_cols = load_cluster_labels(Path(args.gen_labels))

    if args.mode == 'real-gen':
        adata_real, adata_gen = load_embeddings(
            Path(args.real_embedding), Path(args.gen_embedding),
            Path(args.hpa_csv), args.cell_line, logger,
        )
    else:
        adata_real, adata_gen = load_embeddings_gen_gen(
            Path(args.gen_embedding), Path(args.gen_embedding2),
            args.cell_line1, args.cell_line2, logger,
        )

    adata_real, adata_gen, labels_real, labels_gen, common_genes = align_data(
        adata_real, adata_gen, labels_real, labels_gen, logger,
    )

    if args.replot:
        cache_real = output_dir / "umap_coords_real.npy"
        cache_gen  = output_dir / "umap_coords_generated.npy"
        if not cache_real.exists() or not cache_gen.exists():
            logger.error("--replot requires cached UMAP coords — run without --replot first")
            sys.exit(1)

    compute_umaps(adata_real, adata_gen, logger, args.n_neighbors, cache_dir=output_dir)

    # Pre-index labels for efficient per-level lookup
    real_idx = labels_real.set_index('gene_name')
    gen_idx = labels_gen.set_index('gene_name')

    logger.info("Creating 2x2 plots for levels 2-7...")
    for level_i in range(1, 7):
        level_num = level_i + 1
        create_2x2_plot(
            adata_real, adata_gen,
            real_idx.loc[common_genes, real_cols[level_i]].values,
            gen_idx.loc[common_genes, gen_cols[level_i]].values,
            level_num, real_cols[level_i], gen_cols[level_i],
            output_dir / f"umap_cluster_comparison_level{level_num}.png",
            logger, save_svg=args.save_svg,
        )

    save_summary(labels_real, labels_gen, real_cols, gen_cols, common_genes, output_dir, logger)
    logger.info(f"Complete! Results: {output_dir}")


if __name__ == "__main__":
    main()
