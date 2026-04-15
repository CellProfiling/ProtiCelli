#!/usr/bin/env python3
"""
Cross-cell clustering consistency analysis.

Computes pairwise ARI (Adjusted Rand Index) and AMI (Adjusted Mutual Information)
between cells at each hierarchy level, measuring how reproducible the clustering
is across independent cells.

Inputs:
    Directory containing cell_*/cluster_labels_generated.tsv files,
    each with columns: gene_name, leiden_lvl0_*, leiden_lvl1_*, ...

Outputs:
    cross_cell_consistency.tsv  — pairwise results (level, cell1, cell2, ARI, AMI)
    cross_cell_consistency.png  — line plot of mean ± std per level

Usage:
    python cross_cell_consistency.py --output-dir /path/to/single_cell_trees/U2OS

Author: Konstantin Kahnert
"""

import argparse
import logging
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-cell clustering consistency (ARI/AMI per hierarchy level)"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory containing cell_*/cluster_labels_generated.tsv"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    output_dir = Path(args.output_dir)

    label_files = sorted(output_dir.glob("cell_*/cluster_labels_generated.tsv"))
    n_cells = len(label_files)
    logger.info(f"Found {n_cells} cells with cluster label files")

    if n_cells < 2:
        logger.info("Need at least 2 cells for pairwise comparison, skipping")
        sys.exit(0)

    # Load all cells; set gene_name as index
    cell_data = {}
    for f in label_files:
        df = pd.read_csv(f, sep='\t').set_index('gene_name')
        cell_data[f.parent.name] = df

    # Align on common genes
    common_genes = sorted(set.intersection(*[set(df.index) for df in cell_data.values()]))
    logger.info(f"Common genes: {len(common_genes)}")

    # Identify hierarchy levels by lvl{N} prefix
    level_pat = re.compile(r'leiden_lvl(\d+)_')
    cell_names = list(cell_data.keys())
    ref_levels = sorted({
        int(m.group(1))
        for col in cell_data[cell_names[0]].columns
        for m in [level_pat.match(col)] if m
    })
    logger.info(f"Hierarchy levels: {[l + 1 for l in ref_levels]}")

    def get_col(df, lvl_idx):
        for col in df.columns:
            if col.startswith(f'leiden_lvl{lvl_idx}_'):
                return col
        return None

    pairs = list(combinations(cell_names, 2))
    logger.info(f"Computing {len(pairs)} pairwise comparisons across {len(ref_levels)} levels...")

    rows = []
    for lvl_idx in ref_levels:
        for c1, c2 in pairs:
            col1 = get_col(cell_data[c1], lvl_idx)
            col2 = get_col(cell_data[c2], lvl_idx)
            if col1 is None or col2 is None:
                continue
            l1 = cell_data[c1].loc[common_genes, col1].values
            l2 = cell_data[c2].loc[common_genes, col2].values
            rows.append({
                'level': lvl_idx + 1, 'cell1': c1, 'cell2': c2,
                'ARI': adjusted_rand_score(l1, l2),
                'AMI': adjusted_mutual_info_score(l1, l2),
            })

    results_df = pd.DataFrame(rows)
    tsv_path = output_dir / 'cross_cell_consistency.tsv'
    results_df.to_csv(tsv_path, sep='\t', index=False)
    logger.info(f"Saved {len(results_df)} pairwise results to {tsv_path}")

    # Line plot: mean ± std per level
    x_vals = sorted(results_df['level'].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric, color in [('ARI', 'steelblue'), ('AMI', 'darkorange')]:
        means = np.array([results_df[results_df['level'] == l][metric].mean() for l in x_vals])
        stds = np.array([results_df[results_df['level'] == l][metric].std() for l in x_vals])
        ax.plot(x_vals, means, marker='o', color=color, linewidth=2, label=f'{metric} (mean ± std)')
        ax.fill_between(x_vals, means - stds, means + stds, color=color, alpha=0.2)

    ax.set_xlabel('Hierarchy Level', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Cross-cell clustering consistency ({n_cells} cells, {len(pairs)} pairs)', fontsize=13)
    ax.set_xticks(x_vals)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_cell_consistency.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved cross_cell_consistency.png")


if __name__ == "__main__":
    main()
