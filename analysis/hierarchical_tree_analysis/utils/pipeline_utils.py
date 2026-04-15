#!/usr/bin/env python3
"""Shared utilities for the tree comparison pipeline.

Author: Konstantin Kahnert
"""

import re
import logging
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logger and return it."""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.DEBUG if verbose else logging.INFO,
    )
    return logging.getLogger()


def get_ordered_leiden_columns(df_or_adata, expected_count: int = 7) -> List[str]:
    """Return leiden columns sorted by hierarchy level (0..N)."""
    columns = df_or_adata.obs.columns if hasattr(df_or_adata, 'obs') else df_or_adata.columns
    leiden_cols = [c for c in columns if c.startswith('leiden_')]

    if len(leiden_cols) < expected_count:
        raise ValueError(f"Expected >= {expected_count} leiden columns, got {len(leiden_cols)}")

    parsed = []
    for col in leiden_cols:
        m = re.match(r'leiden_lvl(\d+)_res([\d.]+)', col)
        if m:
            parsed.append((int(m.group(1)), col))
        else:
            logging.warning(f"Column {col} uses old naming format, returning first {expected_count}")
            return leiden_cols[:expected_count]

    parsed.sort(key=lambda x: x[0])
    return [col for _, col in parsed][:expected_count]


def load_cluster_labels(file_path, expected_levels: int = 7) -> Tuple[pd.DataFrame, List[str]]:
    """Load cluster labels TSV, return (DataFrame, ordered leiden columns)."""
    file_path = Path(file_path)
    require_file(file_path, 'cluster labels')
    df = pd.read_csv(file_path, sep='\t')
    if 'gene_name' not in df.columns:
        raise ValueError(f"Missing 'gene_name' column in {file_path}")
    leiden_cols = get_ordered_leiden_columns(df, expected_count=expected_levels)
    logging.info(f"Loaded {len(df)} genes, {len(leiden_cols)} levels from {file_path.name}")
    return df, leiden_cols


def validate_clustering_parameters(resolutions: List[float], neighbors: List[int],
                                   n_levels: int = 7) -> None:
    """Validate resolution and neighbor parameter lists."""
    if len(resolutions) != n_levels:
        raise ValueError(f"Expected {n_levels} resolutions, got {len(resolutions)}")
    if len(neighbors) != n_levels:
        raise ValueError(f"Expected {n_levels} neighbors, got {len(neighbors)}")
    if any(r < 0 for r in resolutions):
        raise ValueError("Resolutions must be non-negative")
    if any(n <= 0 for n in neighbors):
        raise ValueError("Neighbor counts must be positive")


def require_file(path, description: str = "") -> Path:
    """Validate file exists; raise FileNotFoundError if missing."""
    path = Path(path)
    if not path.exists():
        label = f" ({description})" if description else ""
        raise FileNotFoundError(f"File not found{label}: {path}")
    return path


def load_gene_subset(subset_path) -> Set[str]:
    """Load a gene set from TSV/CSV (gene_name col), h5ad (obs.gene_name), or plain text."""
    subset_path = Path(subset_path)
    require_file(subset_path, 'gene subset')

    if subset_path.suffix == '.h5ad':
        import scanpy as sc
        adata = sc.read_h5ad(subset_path)
        if 'gene_name' not in adata.obs.columns:
            raise ValueError("h5ad missing 'gene_name' in obs")
        return set(adata.obs['gene_name'].astype(str).str.strip().unique())

    if subset_path.suffix in ('.tsv', '.csv'):
        sep = '\t' if subset_path.suffix == '.tsv' else ','
        df = pd.read_csv(subset_path, sep=sep)
        if 'gene_name' not in df.columns:
            raise ValueError(f"Missing 'gene_name' column. Available: {list(df.columns)}")
        return set(df['gene_name'].astype(str).str.strip().unique())

    with open(subset_path) as f:
        return {line.strip() for line in f if line.strip()}
