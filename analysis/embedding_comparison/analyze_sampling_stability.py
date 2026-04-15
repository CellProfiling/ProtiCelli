# -*- coding: utf-8 -*-
"""
Analyze embedding sampling stability for protein embeddings.

Two modes (--source):
  generated (default): Bootstrap sampling across cells for generated embeddings.
                        Loads from {base_dir}/{cell_line}/{gen_subdir}/embeddings_all_cells.h5ad.
                        Selects N random genes.
  real:                 Bootstrap sampling across images for real HPA embeddings.
                        Loads from --real-embedding.
                        Selects top N genes by image count. Adaptive sample sizes.

Shared workflow:
  - For each gene: sample 1, 2, 4, ..., N embeddings (with replacement, 1000 iterations)
  - Measure stability via cosine similarity convergence
  - --plot-only + --results-file: skip computation, reload TSV and regenerate plots

Outputs:
  - cosine_similarity_convergence_all_lines.png
  - genes_analyzed_{cell_line}.txt
  - results_for_replotting.tsv

Author: Konstantin Kahnert
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import argparse
import sys
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from embedding_utils import (
    load_multiple_embeddings,
    filter_real_embeddings_by_cell_line_imagelevel,
)

logger = logging.getLogger(__name__)


# Base sample sizes to test (may be truncated for real mode based on data availability)
BASE_SAMPLE_SIZES = [1, 2, 4, 8, 12, 16, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]


# -----------------------------------------------------------------------
# Real-Mode Helpers
# -----------------------------------------------------------------------


def select_top_genes_by_image_count(
    adata,
    n_genes: int = 100,
    gene_col: str = "gene_name",
) -> List[str]:
    """Select top N genes by image count (descending order)."""
    gene_counts = adata.obs[gene_col].value_counts()
    top_genes = gene_counts.head(n_genes).index.tolist()

    logger.info("Selected top %d genes by image count:", len(top_genes))
    logger.info("   Image counts range: %s to %s", f"{gene_counts.iloc[0]:,}", f"{gene_counts.iloc[n_genes-1]:,}")
    logger.info("   Mean images for selected genes: %.1f", gene_counts.head(n_genes).mean())
    logger.info("   Median images for selected genes: %.1f", gene_counts.head(n_genes).median())

    return top_genes


def compute_adaptive_sample_sizes(
    adata,
    genes: List[str],
    gene_col: str = "gene_name",
    percentile: int = 90,
    base_sizes: List[int] = BASE_SAMPLE_SIZES,
) -> List[int]:
    """Compute sample sizes up to the Nth percentile of per-gene image counts."""
    gene_counts = adata.obs[gene_col].value_counts()
    selected_counts = gene_counts[gene_counts.index.isin(genes)]
    max_size = int(np.percentile(selected_counts, percentile))

    adapted_sizes = [s for s in base_sizes if s <= max_size]
    if not adapted_sizes:
        adapted_sizes = [s for s in base_sizes if s <= selected_counts.min()]

    logger.info("Adaptive sample sizes (based on %dth percentile = %d):", percentile, max_size)
    logger.info("   Sample sizes: %s", adapted_sizes)
    logger.info("   Range: %d to %d", min(adapted_sizes), max(adapted_sizes))

    return adapted_sizes


# -----------------------------------------------------------------------
# Core Computation (shared)
# -----------------------------------------------------------------------


def compute_sampling_stability(
    adata,
    genes: List[str],
    sample_sizes: List[int],
    gene_col: str = "gene_name",
    n_iterations: int = 1000,
    random_seed: int = 42,
    unit: str = "samples",
) -> Dict:
    """
    Compute embedding stability across different sample sizes.

    For each gene, samples varying numbers of embeddings (with replacement) and
    computes mean embeddings. Measures stability via cosine similarity and
    Euclidean distance between bootstrapped means.

    Args:
        adata: AnnData with image-level embeddings (n_images x n_features)
        genes: List of gene names to analyze
        sample_sizes: List of sample sizes to test
        gene_col: Column name for gene identifiers
        n_iterations: Number of bootstrap iterations per sample size
        random_seed: Random seed for reproducibility
        unit: Label for the sampling unit in progress messages ("cells" or "images")

    Returns:
        Dictionary with cosine_sim_matrix, summary stats, genes_analyzed, and sample_sizes.
    """
    logger.info("Computing sampling stability for %d genes...", len(genes))
    logger.info("   Sample sizes: %s", sample_sizes)
    logger.info("   Iterations per size: %s", f"{n_iterations:,}")

    rng = np.random.default_rng(random_seed)

    cosine_sim_matrix = np.zeros((len(genes), len(sample_sizes)))

    for gene_idx, gene in enumerate(tqdm(genes, desc="   Genes")):
        gene_mask = adata.obs[gene_col] == gene
        gene_embeddings = adata.X[gene_mask].astype(np.float32)
        n_available = gene_embeddings.shape[0]

        for size_idx, sample_size in enumerate(sample_sizes):
            if sample_size > n_available or n_iterations <= 1:
                cosine_sim_matrix[gene_idx, size_idx] = np.nan
                continue

            sampled_means = np.array([
                gene_embeddings[rng.integers(0, n_available, sample_size)].mean(axis=0)
                for _ in range(n_iterations)
            ])
            sim_matrix = cosine_similarity(sampled_means)
            upper_tri = np.triu_indices(n_iterations, k=1)
            cosine_sim_matrix[gene_idx, size_idx] = sim_matrix[upper_tri].mean()

    mean_cosine_sim = np.nanmean(cosine_sim_matrix, axis=0)
    std_cosine_sim = np.nanstd(cosine_sim_matrix, axis=0)

    logger.info("Stability analysis complete")
    logger.info("   Mean Cosine Sim at 1 %s:   %.4f", unit, mean_cosine_sim[0])
    if 20 in sample_sizes:
        logger.info("   Mean Cosine Sim at 20 %ss: %.4f", unit, mean_cosine_sim[sample_sizes.index(20)])
    logger.info("   Mean Cosine Sim at %d %ss: %.4f", sample_sizes[-1], unit, mean_cosine_sim[-1])

    return {
        "cosine_sim_matrix": cosine_sim_matrix,
        "mean_cosine_sim": mean_cosine_sim,
        "std_cosine_sim": std_cosine_sim,
        "genes_analyzed": genes,
        "sample_sizes": np.array(sample_sizes),
    }


# -----------------------------------------------------------------------
# Results I/O (shared)
# -----------------------------------------------------------------------


def save_results_to_tsv(
    all_results: Dict[str, Dict],
    output_path: Path,
    unit: str = "cells",
):
    """Save per-gene stability results to TSV for later re-plotting."""
    rows = []

    for cell_line, results in sorted(all_results.items()):
        cosine_sim_matrix = results["cosine_sim_matrix"]
        genes = results["genes_analyzed"]
        sample_sizes = results["sample_sizes"]

        for gene_idx, gene in enumerate(genes):
            row = {"Cell_Line": cell_line, "Gene": gene}
            for size_idx, size in enumerate(sample_sizes):
                cosine_val = cosine_sim_matrix[gene_idx, size_idx]
                row[f"CosSim_{size}_{unit}"] = f"{cosine_val:.6f}" if not np.isnan(cosine_val) else "NaN"
            rows.append(row)

    df = pd.DataFrame(rows)

    with open(output_path, 'w') as f:
        f.write(f"# Sample sizes: {', '.join(map(str, sample_sizes))}\n")
        df.to_csv(f, sep="\t", index=False)

    logger.info("Saved results for re-plotting: %s", output_path.name)


def load_results_from_tsv(results_file: Path) -> Dict[str, Dict]:
    """Load per-gene stability results from TSV for re-plotting."""
    logger.info("Loading results from: %s", results_file)

    df = pd.read_csv(results_file, sep="\t", comment='#')

    cosine_cols = [col for col in df.columns if col.startswith("CosSim_")]
    if not cosine_cols:
        raise ValueError(f"No CosSim columns found in {results_file}")

    # Parse sample sizes, stripping known unit suffixes
    sample_sizes = []
    for col in cosine_cols:
        size_str = col.replace("CosSim_", "")
        for suffix in ("_cells", "_images", "_samples"):
            size_str = size_str.replace(suffix, "")
        sample_sizes.append(int(size_str))
    sample_sizes = np.array(sample_sizes)
    logger.info("   Sample sizes: %s", sample_sizes)

    all_results = {}
    for cell_line, group in df.groupby("Cell_Line"):
        genes = group["Gene"].values
        n_genes, n_sizes = len(genes), len(sample_sizes)

        cosine_sim_matrix = np.array([
            [float(row[col]) if str(row[col]) != "NaN" else np.nan for col in cosine_cols]
            for _, row in group.iterrows()
        ])

        all_results[cell_line] = {
            "cosine_sim_matrix": cosine_sim_matrix,
            "mean_cosine_sim": np.nanmean(cosine_sim_matrix, axis=0),
            "std_cosine_sim": np.nanstd(cosine_sim_matrix, axis=0),
            "genes_analyzed": genes,
            "sample_sizes": sample_sizes,
        }
        logger.info("   Loaded %s: %d genes x %d sample sizes", cell_line, n_genes, n_sizes)

    logger.info("Loaded %d cell lines", len(all_results))
    return all_results


# -----------------------------------------------------------------------
# Visualization (shared, parameterized by source)
# -----------------------------------------------------------------------


def plot_cosine_similarity_convergence(
    all_results: Dict[str, Dict],
    output_path: Path,
    x_label: str = "Number of Cells Sampled (n)",
    title_prefix: str = "",
):
    """Create convergence plot showing Mean Cosine Similarity vs sample size for all cell lines."""
    fig, ax = plt.subplots(figsize=(12, 7))

    tab20_colors = list(plt.cm.tab20.colors)
    colors = [tab20_colors[idx % len(tab20_colors)] for idx in range(len(all_results))]

    for idx, (cell_line, results) in enumerate(sorted(all_results.items())):
        sample_sizes = results["sample_sizes"]
        mean_cosine_sim = results["mean_cosine_sim"]
        std_cosine_sim = results["std_cosine_sim"]

        ax.plot(sample_sizes, mean_cosine_sim, marker='o', linewidth=2.5, label=cell_line, color=colors[idx])
        ax.fill_between(
            sample_sizes,
            mean_cosine_sim - std_cosine_sim,
            mean_cosine_sim + std_cosine_sim,
            alpha=0.2, color=colors[idx],
        )

    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Cosine Similarity", fontsize=12, fontweight='bold')
    ax.set_title(
        f"{title_prefix}Embedding Convergence: Mean Cosine Similarity vs Sample Size\n"
        "(Higher similarity = more consistent mean embeddings, ideally plateaus near 1.0)",
        fontsize=14, fontweight='bold',
    )
    ax.legend(loc="lower right", fontsize=10, ncol=2)
    ax.grid(alpha=0.3)

    max_sample_size = max([max(r["sample_sizes"]) for r in all_results.values()])
    ax.set_xlim(0, max_sample_size + 10)

    all_cosine_sims = [r["mean_cosine_sim"] for r in all_results.values()]
    min_cosine = min([np.nanmin(cs) for cs in all_cosine_sims])
    ax.set_ylim(max(0, min_cosine - 0.05), 1.05)

    ax.axhline(y=0.95, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Similarity=0.95 (converged)')
    ax.axhline(y=0.99, color='darkgreen', linestyle='--', linewidth=1, alpha=0.5, label='Similarity=0.99 (highly converged)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.close()

    logger.info("Saved cosine similarity convergence plot: %s", output_path.name)



# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze embedding sampling stability for protein embeddings"
    )
    parser.add_argument(
        "--source",
        choices=["generated", "real"],
        default="generated",
        help="Data source: 'generated' (default) or 'real' HPA embeddings",
    )
    # Shared args
    parser.add_argument(
        "--cell-lines",
        nargs="+",
        required=True,
        help="Cell line names to analyze",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (required unless --base-dir is given)",
    )
    parser.add_argument(
        "--gene-col",
        default="gene_name",
        help="Column name for gene identifiers (default: gene_name)",
    )
    parser.add_argument(
        "--n-genes",
        type=int,
        default=100,
        help="Number of genes to analyze: random (generated) or top by image count (real) (default: 100)",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=1000,
        help="Number of bootstrap iterations per sample size (default: 1000)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    # Shared plot-only (works for both sources)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip computation and regenerate plots from an existing results TSV",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        help="Path to existing results TSV (required with --plot-only)",
    )
    # Generated-mode args
    parser.add_argument(
        "--base-dir",
        help="Base directory containing cell line folders (required for --source generated)",
    )
    parser.add_argument(
        "--gen-subdir",
        type=str,
        default="aggregated",
        help="Subdirectory under each cell line folder containing embeddings_all_cells.h5ad (default: aggregated)",
    )
    # Real-mode args
    parser.add_argument(
        "--real-embedding",
        help="Path to real embeddings .h5ad file (required for --source real)",
    )
    parser.add_argument(
        "--hpa-csv",
        help="Path to HPA CSV file with gene reliability info (for --source real)",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=90,
        help="Percentile for adaptive max sample size in real mode (default: 90)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Validate args
    if args.plot_only and not args.results_file:
        logger.error("--results-file is required when using --plot-only")
        sys.exit(1)

    if not args.plot_only:
        if args.source == "generated" and not args.base_dir:
            logger.error("--base-dir is required for --source generated")
            sys.exit(1)
        if args.source == "real" and not args.real_embedding:
            logger.error("--real-embedding is required for --source real")
            sys.exit(1)

    # Source-specific display labels
    unit = "images" if args.source == "real" else "cells"
    x_label = "Number of Images Sampled per Gene" if args.source == "real" else "Number of Cells Sampled (n)"
    title_prefix = "Real HPA Data: " if args.source == "real" else ""

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.base_dir:
        output_dir = Path(args.base_dir) / "comparison" / "sampling_stability"
    else:
        logger.error("--output-dir is required when not using --base-dir")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_lines = args.cell_lines

    mode_label = "Generated" if args.source == "generated" else "Real HPA"
    logger.info("Embedding Sampling Stability Analysis (%s)", mode_label)
    logger.info("=" * 60)
    logger.info("Cell lines: %s", ", ".join(cell_lines))
    logger.info("Output directory: %s", output_dir)
    if args.plot_only:
        logger.info("Mode: Plot-only (loading from %s)", args.results_file)
    else:
        logger.info("Genes to analyze: %d", args.n_genes)
        logger.info("Iterations per sample size: %s", f"{args.n_iterations:,}")
        logger.info("Sample sizes: %s", BASE_SAMPLE_SIZES)

    # ----------------------------------------------------------------
    # Plot-only mode: load existing TSV and regenerate plots
    # ----------------------------------------------------------------
    if args.plot_only:
        results_file = Path(args.results_file)
        if not results_file.exists():
            logger.error("Results file not found: %s", results_file)
            sys.exit(1)
        try:
            all_results = load_results_from_tsv(results_file)
        except Exception as e:
            logger.error("Error loading results file: %s", e)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # ----------------------------------------------------------------
    # Generated mode: load from base_dir, randomly select genes
    # ----------------------------------------------------------------
    elif args.source == "generated":
        base_dir = Path(args.base_dir)
        if not base_dir.exists():
            logger.error("Base directory not found: %s", base_dir)
            sys.exit(1)

        generated_embedding_paths = {}
        missing = []
        for cell_line in cell_lines:
            path = base_dir / cell_line / args.gen_subdir / "embeddings_all_cells.h5ad"
            if path.exists():
                generated_embedding_paths[cell_line] = path
            else:
                missing.append(cell_line)

        if missing:
            logger.error("Embedding files not found for: %s", missing)
            logger.error("Expected: {base_dir}/{cell_line}/%s/embeddings_all_cells.h5ad", args.gen_subdir)
            sys.exit(1)

        logger.info("Loading embeddings...")
        try:
            generated_embeddings = load_multiple_embeddings(generated_embedding_paths)
        except Exception as e:
            logger.error("Error loading embeddings: %s", e)
            sys.exit(1)

        logger.info("Loaded %d cell lines", len(generated_embeddings))
        logger.info("Embeddings Summary:")
        for name, adata in sorted(generated_embeddings.items()):
            n_cells = adata.obs['cell_id'].nunique() if 'cell_id' in adata.obs.columns else 'unknown'
            logger.info("  %s: %s images x %d features (%s cells)", name, f"{adata.n_obs:,}", adata.n_vars, n_cells)

        logger.info("Computing stability analysis...")
        all_results = {}
        rng = np.random.default_rng(args.random_seed)

        for cell_line in cell_lines:
            logger.info("Analyzing %s...", cell_line)
            try:
                adata = generated_embeddings[cell_line]
                all_genes = adata.obs[args.gene_col].unique()
                logger.info("   Total genes available: %s", f"{len(all_genes):,}")

                if len(all_genes) < args.n_genes:
                    logger.warning("Only %d genes available, using all", len(all_genes))
                    genes_to_analyze = list(all_genes)
                else:
                    genes_to_analyze = list(rng.choice(all_genes, args.n_genes, replace=False))
                logger.info("   Randomly selected %d genes", len(genes_to_analyze))

                results = compute_sampling_stability(
                    adata,
                    genes=genes_to_analyze,
                    sample_sizes=BASE_SAMPLE_SIZES,
                    gene_col=args.gene_col,
                    n_iterations=args.n_iterations,
                    random_seed=args.random_seed,
                    unit=unit,
                )
                all_results[cell_line] = results

                genes_path = output_dir / f"genes_analyzed_{cell_line}.txt"
                with open(genes_path, 'w') as f:
                    for gene in results["genes_analyzed"]:
                        f.write(f"{gene}\n")

            except Exception as e:
                logger.error("Failed for %s: %s", cell_line, e)
                import traceback
                traceback.print_exc()
                continue

        if not all_results:
            logger.error("No stability analyses could be completed")
            sys.exit(1)

        save_results_to_tsv(all_results, output_dir / "results_for_replotting.tsv", unit=unit)

    # ----------------------------------------------------------------
    # Real mode: load from real_embedding, top-N genes, adaptive sizes
    # ----------------------------------------------------------------
    else:  # args.source == "real"
        import anndata as ad
        import torch
        import scanpy as sc

        real_embedding_path = Path(args.real_embedding)
        if not real_embedding_path.exists():
            logger.error("Real embeddings file not found: %s", real_embedding_path)
            sys.exit(1)

        hpa_csv_path = Path(args.hpa_csv) if args.hpa_csv else None
        if hpa_csv_path and not hpa_csv_path.exists():
            logger.error("HPA CSV file not found: %s", hpa_csv_path)
            sys.exit(1)

        logger.info("Loading real HPA embeddings...")
        is_h5ad = str(real_embedding_path).endswith(".h5ad")
        try:
            if is_h5ad:
                real_adata_raw = ad.read_h5ad(str(real_embedding_path))
                real_adata_raw.obs = real_adata_raw.obs.rename(columns={
                    "cell_line": "atlas_name",
                    "gene_name": "gene_names",
                })
                if "locations" not in real_adata_raw.obs.columns:
                    real_adata_raw.obs["locations"] = "placeholder"
            else:
                obs, embeddings = torch.load(str(real_embedding_path), map_location="cpu")
                real_adata_raw = sc.AnnData(embeddings.numpy(), obs=obs)
            logger.info("   Loaded %s images x %d features", f"{real_adata_raw.n_obs:,}", real_adata_raw.n_vars)
            logger.info("   Cell lines available: %d", real_adata_raw.obs['atlas_name'].nunique())
            logger.info("   Genes available: %s", f"{real_adata_raw.obs['gene_names'].nunique():,}")
        except Exception as e:
            logger.error("Error loading real embeddings: %s", e)
            sys.exit(1)

        logger.info("Computing stability analysis...")
        all_results = {}

        for cell_line in cell_lines:
            logger.info("=" * 60)
            logger.info("Analyzing %s...", cell_line)
            logger.info("=" * 60)
            try:
                logger.info("   Filtering to cell line with quality filters...")
                adata_cell_line = filter_real_embeddings_by_cell_line_imagelevel(
                    real_adata_raw.copy(),
                    cell_line,
                    None if is_h5ad else hpa_csv_path,
                )
                logger.info("   Filtered to %s images", f"{adata_cell_line.n_obs:,}")
                logger.info("      Unique genes: %s", f"{adata_cell_line.obs[args.gene_col].nunique():,}")

                top_genes = select_top_genes_by_image_count(
                    adata_cell_line, n_genes=args.n_genes, gene_col=args.gene_col,
                )
                sample_sizes = compute_adaptive_sample_sizes(
                    adata_cell_line, top_genes, gene_col=args.gene_col,
                    percentile=args.percentile, base_sizes=BASE_SAMPLE_SIZES,
                )

                results = compute_sampling_stability(
                    adata_cell_line,
                    genes=top_genes,
                    sample_sizes=sample_sizes,
                    gene_col=args.gene_col,
                    n_iterations=args.n_iterations,
                    random_seed=args.random_seed,
                    unit=unit,
                )
                all_results[cell_line] = results

                genes_path = output_dir / f"genes_analyzed_{cell_line}.txt"
                with open(genes_path, 'w') as f:
                    for gene in results["genes_analyzed"]:
                        f.write(f"{gene}\n")
                logger.info("   Saved gene list: %s", genes_path.name)

            except Exception as e:
                logger.error("Failed for %s: %s", cell_line, e)
                import traceback
                traceback.print_exc()
                continue

        if not all_results:
            logger.error("No stability analyses could be completed")
            sys.exit(1)

        save_results_to_tsv(all_results, output_dir / "results_for_replotting.tsv", unit=unit)

    # ----------------------------------------------------------------
    # Shared visualization
    # ----------------------------------------------------------------
    logger.info("Creating visualizations...")

    plot_cosine_similarity_convergence(
        all_results,
        output_dir / "cosine_similarity_convergence_all_lines.png",
        x_label=x_label,
        title_prefix=title_prefix,
    )

    # Print summary stats
    logger.info("Summary Statistics:")
    logger.info("=" * 60)
    for cell_line, results in sorted(all_results.items()):
        mean_cosine_sim = results["mean_cosine_sim"]
        sample_sizes = results["sample_sizes"]

        logger.info("%s:", cell_line)
        logger.info("  Cosine Sim at 1 %s:    %.4f", unit, mean_cosine_sim[0])
        for n in [20, 50, 100]:
            if n in list(sample_sizes):
                idx = list(sample_sizes).index(n)
                logger.info("  Cosine Sim at %d %ss:  %.4f", n, unit, mean_cosine_sim[idx])
        logger.info("  Cosine Sim at %d %ss: %.4f", sample_sizes[-1], unit, mean_cosine_sim[-1])

    logger.info("Analysis complete! Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
