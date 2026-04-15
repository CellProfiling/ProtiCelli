#!/usr/bin/env python3
"""Sequential greedy parameter tuning for hierarchical clustering.

Tunes resolution parameters level-by-level so that hierarchical Leiden clustering
matches target cluster counts at each level (e.g. [1, 2, 6, 14, 22, 38, 86]).

Author: Konstantin Kahnert
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc

from analysis_functions import hierarchical_leiden

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "embedding_comparison"))
from embedding_utils import (
    load_unnormalized_real_embeddings,
    filter_and_aggregate_real_embeddings_by_cell_line,
)
from pipeline_utils import setup_logging, require_file, load_gene_subset


# ---------------------------------------------------------------------------
# Core clustering algorithm (partial / resumable hierarchical Leiden)
# ---------------------------------------------------------------------------

def hierarchical_leiden_partial(
    adata,
    resolutions: List[float],
    neighbors: List[int],
    random_state: int = 42,
    max_level: Optional[int] = None,
    start_level: int = 0,
    metric: str = "cosine",
):
    """Perform hierarchical Leiden clustering with partial computation support.

    Allows computing only a range of levels, enabling level-by-level tuning
    with caching of already-computed parent levels.
    """

    def ensure_cat_col(adata, col):
        if col not in adata.obs.columns:
            adata.obs[col] = pd.Categorical(values=[None] * adata.shape[0])
        elif not isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].astype("category")

    def subcluster(adata, parent_cluster, res, n_neighbors, parent_label, level_idx):
        parent_mask = adata.obs[parent_cluster] == parent_label
        sub_adata = adata[parent_mask].copy()
        target_col = f"leiden_lvl{level_idx}_res{res}"
        ensure_cat_col(adata, target_col)

        if sub_adata.n_obs < n_neighbors:
            new_category = f"{parent_label}.0"
            if new_category not in adata.obs[target_col].cat.categories:
                adata.obs[target_col] = adata.obs[target_col].cat.add_categories([new_category])
            adata.obs.loc[parent_mask, target_col] = new_category
            return

        sc.pp.neighbors(sub_adata, n_neighbors=n_neighbors, n_pcs=None,
                        use_rep="X", metric=metric, random_state=random_state)
        sc.tl.leiden(sub_adata, flavor="leidenalg", resolution=res, n_iterations=2)

        for sub_label in sub_adata.obs["leiden"].unique():
            full_label = f"{parent_label}.{sub_label}"
            if full_label not in adata.obs[target_col].cat.categories:
                adata.obs[target_col] = adata.obs[target_col].cat.add_categories([full_label])

            subcluster_mask = sub_adata.obs["leiden"] == sub_label
            subcluster_positions = np.where(subcluster_mask)[0]
            original_mask = parent_mask.to_numpy().copy()
            original_mask[original_mask] = False
            parent_true_positions = np.where(parent_mask)[0]
            original_mask[parent_true_positions[subcluster_positions]] = True
            adata.obs.loc[original_mask, target_col] = full_label

    # Determine computation range
    stop_at = len(resolutions) if max_level is None else max_level + 1

    if start_level > 0:
        required_parent = f"leiden_lvl{start_level-1}_res{resolutions[start_level - 1]}"
        if required_parent not in adata.obs.columns:
            raise ValueError(
                f"Cannot resume from level {start_level}: "
                f"parent clustering '{required_parent}' not found"
            )

    # Level 0: base clustering
    if start_level == 0:
        sc.pp.neighbors(adata, n_neighbors=neighbors[0], n_pcs=None,
                        use_rep="X", metric=metric, random_state=random_state)
        sc.tl.leiden(adata, flavor="leidenalg", resolution=resolutions[0], n_iterations=2)
        adata.obs[f"leiden_lvl0_res{resolutions[0]}"] = adata.obs["leiden"].astype("category")
        if stop_at == 1:
            adata.uns["hierarchy_edges"] = []
            return adata

    # Hierarchical levels 1..max_level
    for i in range(max(1, start_level), stop_at):
        res = resolutions[i]
        n_neighbors = neighbors[i]
        parent_col = f"leiden_lvl{i-1}_res{resolutions[i - 1]}"
        ensure_cat_col(adata, parent_col)

        for parent_label in adata.obs[parent_col].unique():
            subcluster(adata, parent_col, res, n_neighbors, parent_label, level_idx=i)

        # Renumber labels consecutively
        col = f"leiden_lvl{i}_res{res}"
        ensure_cat_col(adata, col)
        uniq = pd.Series(adata.obs[col].astype(str).values, index=adata.obs.index).unique()
        mapping = {old: str(new) for new, old in enumerate(uniq)}
        adata.obs[col] = pd.Series(
            adata.obs[col].astype(str).map(mapping).values,
            index=adata.obs.index, dtype="category",
        )

    adata.uns["hierarchy_edges"] = []
    return adata


def calculate_cluster_counts(
    adata, resolutions, neighbors, random_seed=42,
    max_level=None, logger=None, metric="cosine",
) -> List[int]:
    """Run clustering and return cluster counts at each level."""
    adata_copy = hierarchical_leiden_partial(
        adata=adata.copy(), resolutions=resolutions, neighbors=neighbors,
        random_state=random_seed, max_level=max_level, metric=metric,
    )
    stop_at = len(resolutions) if max_level is None else max_level + 1
    counts = []
    for i in range(stop_at):
        col = f'leiden_lvl{i}_res{resolutions[i]}'
        if col in adata_copy.obs.columns:
            counts.append(adata_copy.obs[col].nunique())
        else:
            if logger:
                logger.warning(f"Cluster column for level {i} (res={resolutions[i]}) not found")
            counts.append(0)
    return counts


# ---------------------------------------------------------------------------
# Tuning helpers
# ---------------------------------------------------------------------------

def detect_oscillation(history: List[Tuple[float, int]]) -> bool:
    """Detect if last 4 iterations alternate between two cluster counts (A,B,A,B)."""
    if len(history) < 4:
        return False
    counts = [c for _, c in history[-4:]]
    return counts[0] == counts[2] and counts[1] == counts[3] and counts[0] != counts[1]


def choose_best_resolution(history: List[Tuple[float, int]], target: int) -> Tuple[float, int]:
    """From the oscillating pair, pick the resolution closer to target."""
    pair1, pair2 = history[-4], history[-3]
    return pair1 if abs(pair1[1] - target) <= abs(pair2[1] - target) else pair2


def tune_single_level(
    adata, resolutions, neighbors, level_idx, target_count,
    random_seed, logger, max_iterations=100, metric="cosine",
) -> Tuple[float, int, str]:
    """Tune resolution for a single level with progressive delta refinement.

    Returns (final_resolution, achieved_count, status).
    Status is 'exact', 'approximate', or 'max_iter'.
    """
    logger.info(f"Level {level_idx}: Target = {target_count}")

    # Cache base clustering (levels 0..level_idx-1)
    adata_base = hierarchical_leiden_partial(
        adata=adata.copy(), resolutions=resolutions[:level_idx],
        neighbors=neighbors[:level_idx], random_state=random_seed,
        max_level=level_idx - 1, metric=metric,
    )

    def compute_level_count(resolution):
        adata_temp = adata_base.copy()
        test_res = resolutions[:level_idx] + [resolution]
        test_nbrs = neighbors[:level_idx + 1]
        adata_temp = hierarchical_leiden_partial(
            adata=adata_temp, resolutions=test_res, neighbors=test_nbrs,
            random_state=random_seed, start_level=level_idx,
            max_level=level_idx, metric=metric,
        )
        return adata_temp.obs[f"leiden_lvl{level_idx}_res{resolution}"].nunique()

    delta_sequence = [0.1, 0.01, 0.001, 0.0001]
    max_iters_per_delta = max(20, max_iterations // len(delta_sequence))

    global_best_res = resolutions[level_idx]
    global_best_count = None
    global_best_diff = float('inf')
    current_res = resolutions[level_idx]

    for delta in delta_sequence:
        history = []
        best_exact_match = None
        logger.info(f"  Trying delta={delta}")

        for _ in range(max_iters_per_delta):
            try:
                current_count = compute_level_count(current_res)
            except Exception as e:
                logger.debug("Clustering failed at res=%.6f: %s", current_res, e)
                current_res = max(0.0, current_res + delta)
                continue

            diff = abs(current_count - target_count)
            if diff < global_best_diff:
                global_best_res, global_best_count, global_best_diff = (
                    current_res, current_count, diff,
                )

            if current_count == target_count:
                best_exact_match = current_res
                current_res += delta
                history.append((current_res, current_count))
                continue

            if current_count > target_count and best_exact_match is not None:
                logger.info(f"    Exact match: res={best_exact_match:.6f}")
                return best_exact_match, target_count, 'exact'

            history.append((current_res, current_count))

            if detect_oscillation(history):
                logger.info(f"    Oscillation at delta={delta}, refining")
                current_res = global_best_res
                break

            current_res = current_res + delta if current_count < target_count else max(0.0, current_res - delta)

            if current_res == 0.0 and current_count > target_count:
                break

        if best_exact_match is not None:
            logger.info(f"    Exact match: res={best_exact_match:.6f}")
            return best_exact_match, target_count, 'exact'

    # Exhausted all deltas
    if global_best_count is not None:
        if global_best_count == target_count:
            return global_best_res, global_best_count, 'exact'
        logger.warning(f"  Approximate: res={global_best_res:.6f} -> "
                       f"count={global_best_count} (target={target_count})")
        return global_best_res, global_best_count, 'approximate'

    logger.error(f"  No valid results for level {level_idx}")
    return resolutions[level_idx], 0, 'max_iter'


def sequential_greedy_tune(
    adata, target_counts, base_resolutions, base_neighbors,
    random_seed, logger, max_iter_per_level=100, metric="cosine",
) -> Dict:
    """Sequentially tune clustering parameters level by level."""
    logger.info("=" * 60)
    logger.info("Sequential Greedy Parameter Tuning")
    logger.info(f"  Targets: {target_counts}")
    logger.info(f"  Base resolutions: {[round(r, 4) for r in base_resolutions]}")
    logger.info(f"  Neighbors: {base_neighbors}, Metric: {metric}")
    logger.info("=" * 60)

    current_resolutions = base_resolutions.copy()
    tuning_details = [{
        "level": 0, "skipped": True,
        "final_res": current_resolutions[0],
        "target": target_counts[0], "status": "skipped",
    }]

    for level_idx in range(1, 7):
        final_res, achieved, status = tune_single_level(
            adata=adata, resolutions=current_resolutions, neighbors=base_neighbors,
            level_idx=level_idx, target_count=target_counts[level_idx],
            random_seed=random_seed, logger=logger,
            max_iterations=max_iter_per_level, metric=metric,
        )
        current_resolutions[level_idx] = final_res
        tuning_details.append({
            "level": level_idx, "final_res": round(final_res, 4),
            "target": target_counts[level_idx], "achieved": achieved, "status": status,
        })
        sym = "+" if achieved == target_counts[level_idx] else "~"
        logger.info(f"  {sym} Level {level_idx}: res={final_res:.4f}, "
                     f"count={achieved}/{target_counts[level_idx]}\n")

    # Final validation
    logger.info("Validating final parameters...")
    final_counts = calculate_cluster_counts(
        adata=adata, resolutions=current_resolutions,
        neighbors=base_neighbors, random_seed=random_seed, logger=logger, metric=metric,
    )

    result = {
        "creation_date": datetime.now().isoformat(),
        "target_counts": target_counts,
        "cluster_counts": final_counts,
        "resolutions": [round(r, 4) for r in current_resolutions],
        "neighbors": base_neighbors,
        "metric": metric,
        "tuning_details": tuning_details,
    }

    logger.info(f"Final resolutions: {result['resolutions']}")
    logger.info(f"Target:  {target_counts}")
    logger.info(f"Achieved: {final_counts}")
    for i, (t, a) in enumerate(zip(target_counts, final_counts)):
        logger.info(f"  Level {i}: {a:3d} / {t:3d} {'OK' if t == a else 'MISS'}")

    return result


def save_results(result, output_dir, logger):
    """Save tuning results to JSON and TSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "tuning_results.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {json_path}")

    rows = []
    for d in result['tuning_details']:
        achieved = d.get('achieved', d['target'])
        rows.append({
            "Level": d['level'], "Target": d['target'], "Achieved": achieved,
            "Resolution": d['final_res'], "Status": d['status'],
            "Match": "OK" if achieved == d['target'] else "MISS",
        })
    pd.DataFrame(rows).to_csv(output_dir / "tuning_summary.tsv", sep='\t', index=False)
    logger.info(f"Saved: {output_dir / 'tuning_summary.tsv'}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def load_embeddings(args, logger) -> sc.AnnData:
    """Load embeddings based on CLI mode, optionally subsetting genes."""
    if args.mode == 'generated':
        if not args.gen_embedding:
            logger.error("--gen-embedding required for mode=generated")
            sys.exit(1)
        require_file(args.gen_embedding, 'generated embedding')
        logger.info(f"Loading: {args.gen_embedding}")
        adata = sc.read_h5ad(args.gen_embedding)

    elif args.mode == 'real':
        if not args.real_embedding or not args.cell_line:
            logger.error("--real-embedding and --cell-line required for mode=real")
            sys.exit(1)
        require_file(args.real_embedding, 'real embedding')
        logger.info(f"Loading real embeddings for {args.cell_line}...")
        adata_raw = load_unnormalized_real_embeddings(Path(args.real_embedding))
        adata = filter_and_aggregate_real_embeddings_by_cell_line(
            adata_raw, args.cell_line, args.hpa_csv,
        )

    logger.info(f"  Loaded: {adata.n_obs:,} genes, {adata.n_vars} features")

    # Subset to reference gene set if specified
    if args.subset_genes:
        subset_genes = load_gene_subset(args.subset_genes)
        logger.info(f"Subsetting to {len(subset_genes)} reference genes...")
        gene_col = 'gene_name' if 'gene_name' in adata.obs.columns else 'gene_names'
        if gene_col not in adata.obs.columns:
            logger.error(f"No gene column in embeddings. Available: {list(adata.obs.columns)}")
            sys.exit(1)
        before = adata.n_obs
        adata = adata[adata.obs[gene_col].astype(str).str.strip().isin(subset_genes)].copy()
        logger.info(f"  Subsetted: {before:,} -> {adata.n_obs:,} genes")
        if adata.n_obs == 0:
            logger.error("No genes remaining after subsetting!")
            sys.exit(1)

    return adata


def main():
    parser = argparse.ArgumentParser(
        description='Tune clustering parameters to match target cluster counts',
    )
    parser.add_argument('--mode', choices=['generated', 'real'], default='generated')
    parser.add_argument('--gen-embedding', type=str)
    parser.add_argument('--real-embedding', type=str)
    parser.add_argument('--cell-line', type=str)
    parser.add_argument('--hpa-csv', type=str)
    parser.add_argument('--subset-genes', type=str,
                        help='Path to gene list file for subsetting before tuning')
    parser.add_argument('--target-counts', type=int, nargs=7, required=True)
    parser.add_argument('--output-dir', type=str, default='./tuning_results')
    parser.add_argument('--base-resolutions', type=float, nargs=7,
                        default=[0.0, 0.15, 0.2, 0.29, 0.30, 0.31, 0.33])
    parser.add_argument('--base-neighbors', type=int, nargs=7,
                        default=[125, 100, 90, 55, 40, 25, 10])
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Max iterations per level (default: 100)')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--metric', choices=['cosine', 'euclidean'], default='cosine')
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info(f"Clustering Parameter Tuning | mode={args.mode}")
    logger.info("=" * 60)

    try:
        adata = load_embeddings(args, logger)

        result = sequential_greedy_tune(
            adata=adata, target_counts=args.target_counts,
            base_resolutions=args.base_resolutions, base_neighbors=args.base_neighbors,
            random_seed=args.random_seed, logger=logger,
            max_iter_per_level=args.max_iter, metric=args.metric,
        )

        save_results(result, output_dir, logger)
        logger.info("Parameter tuning complete!")

    except Exception as e:
        logger.error(f"Parameter tuning failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
