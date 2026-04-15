#!/usr/bin/env python3
"""Generate hierarchical clustering labels for real and generated embeddings.

Author: Konstantin Kahnert
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Set, Tuple

import pandas as pd
import scanpy as sc

from analysis_functions import hierarchical_leiden

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "embedding_comparison"))
from embedding_utils import (
    load_unnormalized_real_embeddings,
    filter_and_aggregate_real_embeddings_by_cell_line,
)
from pipeline_utils import (
    setup_logging,
    get_ordered_leiden_columns,
    validate_clustering_parameters,
    require_file,
)


def cluster_and_extract(
    adata: sc.AnnData,
    resolutions: List[float],
    neighbors: List[int],
    random_seed: int,
    metric: str,
    logger,
) -> Tuple[pd.DataFrame, dict]:
    """Run hierarchical Leiden clustering and extract labels with metadata."""
    logger.info(f"Clustering {adata.n_obs:,} genes (resolutions={resolutions}, metric={metric})")
    adata = hierarchical_leiden(
        adata=adata, resolutions=resolutions, neighbors=neighbors,
        random_state=random_seed, metric=metric,
    )

    leiden_cols = get_ordered_leiden_columns(adata, expected_count=7)
    if 'gene_name' not in adata.obs.columns:
        raise ValueError("Missing 'gene_name' column in observations")

    output_df = adata.obs[['gene_name'] + leiden_cols].copy().reset_index(drop=True)
    cluster_counts = [output_df[col].nunique() for col in leiden_cols]

    for i, (col, count) in enumerate(zip(leiden_cols, cluster_counts)):
        logger.info(f"  Level {i}: {count} clusters ({col})")

    metadata = {
        "n_genes": len(output_df),
        "resolutions": resolutions,
        "neighbors": neighbors,
        "cluster_counts": cluster_counts,
        "random_seed": random_seed,
        "metric": metric,
        "creation_date": datetime.now().isoformat(),
        "leiden_columns": leiden_cols,
    }
    return output_df, metadata


def load_real(real_path: Path, hpa_csv: Path, cell_line: str, logger) -> sc.AnnData:
    """Load and aggregate real HPA embeddings for a cell line."""
    logger.info(f"Loading real embeddings for {cell_line}...")
    adata = load_unnormalized_real_embeddings(pth_path=real_path)
    return filter_and_aggregate_real_embeddings_by_cell_line(adata, cell_line, str(hpa_csv))


def load_generated(
    gen_path: Path,
    subset_genes: Optional[Set[str]],
    logger,
) -> sc.AnnData:
    """Load generated embeddings, optionally subsetting to a gene set."""
    logger.info(f"Loading generated embeddings from {gen_path}")
    adata = sc.read_h5ad(gen_path)
    logger.info(f"  Loaded: {adata.n_obs:,} genes, {adata.n_vars} features")

    if subset_genes is not None:
        genes = adata.obs['gene_name'].astype(str).str.strip()
        common = subset_genes & set(genes.unique())
        if not common:
            raise ValueError("No common genes between real and generated embeddings")
        logger.info(f"  Subsetting: {adata.n_obs:,} -> {len(common):,} common genes")
        adata = adata[genes.isin(common)].copy()

    return adata


def save_outputs(output_df, metadata, output_dir, prefix, cell_line, logger):
    """Save cluster labels TSV and metadata JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{cell_line}" if cell_line else ""
    tsv_path = output_dir / f"cluster_labels_{prefix}{suffix}.tsv"
    output_df.to_csv(tsv_path, sep='\t', index=False)
    logger.info(f"Saved: {tsv_path}")

    json_path = output_dir / f"clustering_metadata_{prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate hierarchical clustering labels for real and generated embeddings',
    )
    parser.add_argument('--mode', choices=['real', 'generated', 'both', 'gen-gen'], default='both')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--cell-line', type=str, default=None)
    parser.add_argument('--real-embedding', type=str, default=None)
    parser.add_argument('--hpa-csv', type=str, default=None)
    parser.add_argument('--gen-embedding', type=str, default=None)
    parser.add_argument('--gen-embedding2', type=str)
    parser.add_argument('--cell-line1', type=str)
    parser.add_argument('--cell-line2', type=str)
    parser.add_argument('--real-resolutions', type=float, nargs='+',
                        default=[0.0, 0.15, 0.2, 0.29, 0.30, 0.31, 0.33])
    parser.add_argument('--real-neighbors', type=int, nargs='+',
                        default=[125, 100, 90, 55, 40, 25, 10])
    parser.add_argument('--gen-resolutions', type=float, nargs='+', default=None)
    parser.add_argument('--gen-neighbors', type=int, nargs='+', default=None)
    parser.add_argument('--subset-to-real-genes', action='store_true', default=True)
    parser.add_argument('--no-subset', action='store_true')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--metric', choices=['cosine', 'euclidean'], default='cosine')
    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    gen_resolutions = args.gen_resolutions or args.real_resolutions
    gen_neighbors = args.gen_neighbors or args.real_neighbors
    subset_to_real = args.subset_to_real_genes and not args.no_subset

    validate_clustering_parameters(args.real_resolutions, args.real_neighbors)
    validate_clustering_parameters(gen_resolutions, gen_neighbors)

    output_dir = Path(args.output_dir)
    cell_line = args.cell_line or "U2OS"

    if args.mode in ('real', 'both'):
        require_file(args.real_embedding, 'real embedding')
        require_file(args.hpa_csv, 'HPA CSV')
    if args.mode in ('generated', 'both', 'gen-gen'):
        require_file(args.gen_embedding, 'generated embedding')
    if args.mode == 'gen-gen':
        require_file(args.gen_embedding2, 'second generated embedding')
        if not args.cell_line1 or not args.cell_line2:
            parser.error("--cell-line1 and --cell-line2 required for mode=gen-gen")

    logger.info("=" * 60)
    logger.info(f"Generate Clustering Labels | mode={args.mode} | output={output_dir}")
    logger.info("=" * 60)

    # Process real embeddings
    real_genes = None
    if args.mode in ('real', 'both'):
        adata_real = load_real(Path(args.real_embedding), Path(args.hpa_csv), cell_line, logger)
        real_df, real_meta = cluster_and_extract(
            adata_real, args.real_resolutions, args.real_neighbors,
            args.random_seed, args.metric, logger,
        )
        real_meta["embedding_source"] = f"real HPA {cell_line}"
        save_outputs(real_df, real_meta, output_dir, 'real', args.cell_line, logger)
        real_genes = set(real_df['gene_name'].unique())

    # Process generated embeddings
    if args.mode in ('generated', 'both'):
        subset = real_genes if (subset_to_real and real_genes) else None
        adata_gen = load_generated(Path(args.gen_embedding), subset, logger)
        gen_df, gen_meta = cluster_and_extract(
            adata_gen, gen_resolutions, gen_neighbors,
            args.random_seed, args.metric, logger,
        )
        gen_meta["embedding_source"] = "generated"
        gen_meta["subset_to_real_genes"] = subset is not None
        save_outputs(gen_df, gen_meta, output_dir, 'generated', args.cell_line, logger)

    # Process gen-gen mode (two generated embeddings)
    if args.mode == 'gen-gen':
        adata1 = load_generated(Path(args.gen_embedding), None, logger)
        adata2 = load_generated(Path(args.gen_embedding2), None, logger)

        df1, meta1 = cluster_and_extract(
            adata1, args.real_resolutions, args.real_neighbors,
            args.random_seed, args.metric, logger,
        )
        df2, meta2 = cluster_and_extract(
            adata2, gen_resolutions, gen_neighbors,
            args.random_seed, args.metric, logger,
        )

        # Align to common genes
        genes1, genes2 = set(df1['gene_name'].unique()), set(df2['gene_name'].unique())
        common = genes1 & genes2
        if not common:
            logger.error("No common genes between embeddings")
            sys.exit(1)
        logger.info(f"Common genes: {len(common):,} "
                     f"({args.cell_line1}: {len(genes1):,}, {args.cell_line2}: {len(genes2):,})")

        df1 = df1[df1['gene_name'].isin(common)].copy()
        df2 = df2[df2['gene_name'].isin(common)].copy()

        save_outputs(df1, meta1, output_dir, f'generated_{args.cell_line1}', None, logger)
        save_outputs(df2, meta2, output_dir, f'generated_{args.cell_line2}', None, logger)

    logger.info(f"All processing complete. Results: {output_dir}")


if __name__ == "__main__":
    main()
