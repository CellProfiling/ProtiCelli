#!/usr/bin/env python3
"""Compare hierarchical clustering results between two clustering trees.

Calculates ARI, AMI, V-Measure at each hierarchy level and optionally generates
hierarchy tree plots with GO enrichment.

Author: Konstantin Kahnert
"""

import sys
import logging
import pickle
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "embedding_comparison"))
from embedding_utils import save_figure as _save_figure
from pipeline_utils import setup_logging, load_cluster_labels


def align_datasets(df_real, df_gen, real_cols, gen_cols):
    """Align two datasets to common gene set, sorted deterministically."""
    common = sorted(set(df_real['gene_name']) & set(df_gen['gene_name']))
    if not common:
        raise ValueError("No common genes found between datasets")

    logging.info(f"Aligned: {len(common)} common genes "
                 f"(real={len(df_real)}, gen={len(df_gen)})")

    df_r = (df_real[df_real['gene_name'].isin(common)]
            .sort_values('gene_name').reset_index(drop=True))
    df_g = (df_gen[df_gen['gene_name'].isin(common)]
            .sort_values('gene_name').reset_index(drop=True))

    assert all(df_r['gene_name'] == df_g['gene_name']), "Gene alignment failed"
    return df_r[['gene_name'] + real_cols], df_g[['gene_name'] + gen_cols]


def compare_hierarchies(df_real, df_gen, real_cols, gen_cols):
    """Compare clustering at all 7 levels. Returns DataFrame with ARI, AMI, V_Measure."""
    results = []
    for level, (rc, gc) in enumerate(zip(real_cols, gen_cols)):
        lr, lg = df_real[rc].values, df_gen[gc].values
        metrics = {
            'Hierarchy_Level': level + 1,
            'ARI': adjusted_rand_score(lr, lg),
            'AMI': adjusted_mutual_info_score(lr, lg),
            'V_Measure': v_measure_score(lr, lg),
        }
        results.append(metrics)
        logging.info(f"  Level {level+1}: ARI={metrics['ARI']:.3f}, "
                     f"AMI={metrics['AMI']:.3f}, V={metrics['V_Measure']:.3f}")

    return pd.DataFrame(results)[['Hierarchy_Level', 'ARI', 'AMI', 'V_Measure']]


def plot_similarity_metrics(results_df, output_dir, save_svg=False):
    """Create line plot for AMI across hierarchy levels."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.3, palette="colorblind")
    fig, ax = plt.subplots(figsize=(7, 6))

    ami_df = results_df[results_df['Hierarchy_Level'] >= 2]
    palette = sns.color_palette("colorblind", 2)

    sns.lineplot(x=ami_df['Hierarchy_Level'].values, y=ami_df['AMI'].values, marker='s',
                 label='AMI', linewidth=2.5, markersize=9, color=palette[1], ax=ax)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Hierarchy Level')
    ax.set_ylabel('Similarity Score')
    ax.set_title('Clustering Similarity: Real vs Generated', fontweight='bold')
    ax.set_xticks(ami_df['Hierarchy_Level'].values)
    ax.set_xlim(1.5, ami_df['Hierarchy_Level'].max() + 0.5)
    ax.legend(loc='best', frameon=True, framealpha=0.9)

    y_min = ami_df['AMI'].min()
    ax.set_ylim(min(y_min - 0.05, -0.05), 0.6)

    sns.despine(ax=ax)
    ax.grid(True, axis='y', alpha=0.4, linewidth=0.7, color='#bbbbbb')
    plt.tight_layout()

    _save_figure(output_dir / 'clustering_similarity_lineplot.png', save_svg=save_svg, dpi=300)
    plt.close()
    logging.info(f"Saved line plot: {output_dir / 'clustering_similarity_lineplot.png'}")

    ami_df.to_csv(output_dir / 'ami_values.tsv', sep='\t', index=False)
    logging.info(f"Saved AMI values: {output_dir / 'ami_values.tsv'}")


def plot_hierarchy_trees(df_real, df_gen, real_cols, gen_cols, output_dir,
                         max_levels=7, save_svg=False, labels_to_plot=("real", "generated"),
                         skip_enrichment=False, replot=False):
    """Generate hierarchy tree plots using analysis.py helpers."""
    import re
    import anndata as ad

    from analysis_functions import build_hierarchy_graph, visualize_hierarchy, run_enrichment

    for label, df, cols in [("real", df_real, real_cols), ("generated", df_gen, gen_cols)]:
        if label not in labels_to_plot or df is None:
            continue

        resolutions = []
        for col in cols:
            m = re.match(r'leiden_lvl\d+_res([\d.]+)', col)
            if m:
                resolutions.append(float(m.group(1)))
        if not resolutions:
            logging.warning(f"No resolutions parsed for {label}, skipping tree")
            continue

        adata = ad.AnnData(obs=df.copy())
        cache_path = output_dir / f"enrichment_cache_{label}.pkl"

        if replot:
            if not cache_path.exists():
                raise FileNotFoundError(f"No enrichment cache: {cache_path}")
            with open(cache_path, "rb") as f:
                enrich_results = pickle.load(f)
        elif skip_enrichment:
            enrich_results = {}
        else:
            logging.info(f"  Running GO enrichment for {label} tree...")
            enrich_results = run_enrichment(adata, resolutions[:max_levels])
            with open(cache_path, "wb") as f:
                pickle.dump(enrich_results, f)

        G = build_hierarchy_graph(adata, resolutions, max_levels=max_levels)
        out_svg = str(output_dir / f"hierarchy_tree_{label}.svg")
        visualize_hierarchy(G, enrich_results, out_svg, resolutions, max_levels=max_levels)

        if not save_svg:
            svg_path = Path(out_svg)
            if svg_path.exists():
                svg_path.unlink()

        logging.info(f"  Saved hierarchy tree: {label}")


def save_summary(results_df, output_dir):
    """Save summary statistics text file."""
    with open(output_dir / 'comparison_summary.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Clustering Hierarchy Comparison Summary\n")
        f.write("=" * 60 + "\n\n")
        for metric in ['ARI', 'AMI', 'V_Measure']:
            vals = results_df[metric]
            f.write(f"{metric}: {vals.mean():.3f} +/- {vals.std():.3f} "
                    f"[{vals.min():.3f}, {vals.max():.3f}]\n")

        best = results_df.loc[results_df['ARI'].idxmax()]
        worst = results_df.loc[results_df['ARI'].idxmin()]
        f.write(f"\nBest (ARI): Level {int(best['Hierarchy_Level'])} "
                f"(ARI={best['ARI']:.3f}, AMI={best['AMI']:.3f}, V={best['V_Measure']:.3f})\n")
        f.write(f"Worst (ARI): Level {int(worst['Hierarchy_Level'])} "
                f"(ARI={worst['ARI']:.3f}, AMI={worst['AMI']:.3f}, V={worst['V_Measure']:.3f})\n")

    logging.info(f"Saved summary: {output_dir / 'comparison_summary.txt'}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare hierarchical clustering results between two clustering trees',
    )
    parser.add_argument('real_file', nargs='?', default=None,
                        help='Path to real cluster_labels.tsv')
    parser.add_argument('generated_file', nargs='?', default=None,
                        help='Path to generated cluster_labels.tsv')
    parser.add_argument('--output-dir', type=str, default='./comparison_output')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save-svg', action='store_true')
    parser.add_argument('--plot-tree', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-levels', type=int, default=7)
    parser.add_argument('--tree-only', action='store_true',
                        help='Plot hierarchy tree for a single file, skip comparison')
    parser.add_argument('--replot', action='store_true',
                        help='Replot trees from cached enrichment results')
    args = parser.parse_args()

    setup_logging(args.verbose)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve positional args (tree-only may use single arg in either slot)
    real_file = Path(args.real_file) if args.real_file else None
    _gen_arg = args.generated_file or (
        args.real_file if args.tree_only and not args.generated_file else None
    )
    gen_file = Path(_gen_arg) if _gen_arg else None

    # Tree-only mode: plot hierarchy tree for a single labels file
    if args.tree_only:
        if gen_file is None:
            logging.error("--tree-only requires a labels file as positional argument")
            sys.exit(1)
        df_gen, gen_cols = load_cluster_labels(gen_file)
        plot_hierarchy_trees(
            None, df_gen, [], gen_cols, output_dir,
            max_levels=args.max_levels, save_svg=args.save_svg,
            labels_to_plot=("generated",), skip_enrichment=True,
        )
        logging.info(f"Tree saved to: {output_dir}")
        return

    if real_file is None or gen_file is None:
        logging.error("Both real_file and generated_file required (or use --tree-only)")
        sys.exit(1)

    logging.info(f"Comparing: {real_file.name} vs {gen_file.name}")

    df_real, real_cols = load_cluster_labels(real_file)
    df_gen, gen_cols = load_cluster_labels(gen_file)
    df_real, df_gen = align_datasets(df_real, df_gen, real_cols, gen_cols)

    # Replot mode: skip metrics recomputation, regenerate all plots from cached TSV
    if args.replot:
        logging.info("--replot: skipping metrics, replotting all plots from cache")
        metrics_file = output_dir / 'clustering_similarity_metrics.tsv'
        if not metrics_file.exists():
            logging.error(f"--replot requires existing metrics file: {metrics_file}")
            sys.exit(1)
        results_df = pd.read_csv(metrics_file, sep='\t')
        results_df.to_csv(output_dir / 'clustering_similarity_metrics.tsv', sep='\t', index=False)
        plot_similarity_metrics(results_df, output_dir, save_svg=args.save_svg)
        save_summary(results_df, output_dir)
        if args.plot_tree:
            plot_hierarchy_trees(
                df_real, df_gen, real_cols, gen_cols, output_dir,
                max_levels=args.max_levels, save_svg=args.save_svg, replot=True,
            )
        return

    # Full comparison
    results_df = compare_hierarchies(df_real, df_gen, real_cols, gen_cols)
    results_df.to_csv(output_dir / 'clustering_similarity_metrics.tsv', sep='\t', index=False)
    logging.info(f"Saved: {output_dir / 'clustering_similarity_metrics.tsv'}")

    plot_similarity_metrics(results_df, output_dir, save_svg=args.save_svg)
    save_summary(results_df, output_dir)

    if args.plot_tree:
        plot_hierarchy_trees(
            df_real, df_gen, real_cols, gen_cols, output_dir,
            max_levels=args.max_levels, save_svg=args.save_svg,
        )

    logging.info(f"Complete! Results: {output_dir}")


if __name__ == "__main__":
    main()
