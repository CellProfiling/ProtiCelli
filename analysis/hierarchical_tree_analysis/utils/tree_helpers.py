#!/usr/bin/env python3
"""
Helper functions for consensus-tree visualization and single-cell tree processing.

Extracted from color_eval_tree_by_consensus.py and
color_single_cell_tree_by_consensus.py for standalone use in
plot_tree_with_heatmaps.py.

Author: Konstantin Kahnert
"""

import re
import gc
import random
import logging
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cluster label loading
# ---------------------------------------------------------------------------

def load_eval_cluster_labels(
    tsv_path: Path,
) -> Tuple[List[str], Dict[str, np.ndarray], List[str], List[float]]:
    """Load cluster label TSV from the evaluation pipeline.

    Parses column names of the form ``leiden_lvl{N}_res{R}`` to extract
    level numbers and tuned resolutions.

    Args:
        tsv_path: Path to cluster_labels_real_U2OS.tsv.

    Returns:
        gene_names: Ordered list of gene names.
        cluster_assignments: {level_col: int array of cluster ids}.
        level_cols: Leiden column names sorted by level (7 entries).
        tuned_resolutions: Float resolutions extracted from column names (7 entries).
    """
    logger.info(f"Loading cluster labels from {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")

    if "gene_name" not in df.columns:
        raise ValueError(f"Missing 'gene_name' column in {tsv_path}")

    gene_names = df["gene_name"].tolist()

    leiden_cols = [col for col in df.columns if col.startswith("leiden_")]
    if not leiden_cols:
        raise ValueError(f"No 'leiden_*' columns found in {tsv_path}")

    col_with_level: List[Tuple[int, float, str]] = []
    for col in leiden_cols:
        match = re.match(r"leiden_lvl(\d+)_res([\d.]+)", col)
        if not match:
            raise ValueError(
                f"Column '{col}' doesn't match expected format "
                f"leiden_lvl{{N}}_res{{R}} in {tsv_path}"
            )
        level = int(match.group(1))
        res = float(match.group(2))
        col_with_level.append((level, res, col))

    col_with_level.sort(key=lambda x: x[0])

    if len(col_with_level) < 7:
        raise ValueError(
            f"Expected 7 leiden columns, got {len(col_with_level)} in {tsv_path}"
        )

    level_cols = [col for _, _, col in col_with_level]
    tuned_resolutions = [res for _, res, _ in col_with_level]

    cluster_assignments: Dict[str, np.ndarray] = {
        col: df[col].values.astype(int) for col in level_cols
    }

    n_clusters_per_level = [len(np.unique(cluster_assignments[c])) for c in level_cols]
    logger.info(
        f"Loaded {len(gene_names)} genes, {len(level_cols)} levels; "
        f"clusters per level: {n_clusters_per_level}; "
        f"tuned resolutions: {tuned_resolutions}"
    )
    return gene_names, cluster_assignments, level_cols, tuned_resolutions


# ---------------------------------------------------------------------------
# Cohesion computation
# ---------------------------------------------------------------------------

def compute_cluster_cohesion(M: np.ndarray, member_indices: np.ndarray) -> float:
    """Mean off-diagonal consensus value for proteins in a cluster.

    Single-protein clusters get cohesion 1.0.
    """
    n = len(member_indices)
    if n <= 1:
        return 1.0
    sub = M[np.ix_(member_indices, member_indices)]
    total = sub.sum() - np.trace(sub)
    n_pairs = n * (n - 1)
    return float(total / n_pairs)


def compute_cohesion_from_consensus(
    cluster_assignments: Dict[str, np.ndarray],
    level_cols: List[str],
    gene_names: List[str],
    consensus_dir: Path,
    consensus_resolutions: List[float],
    all_gene_names: List[str],
) -> Dict[str, Dict[int, float]]:
    """Compute per-cluster cohesion from consensus matrices.

    Maps eval levels 2-7 (``level_cols[1:]``) to ``consensus_resolutions``
    (6 entries) by position.  Level 1 (single all-genes cluster = root) is
    skipped.

    Args:
        cluster_assignments: {level_col: cluster id array} for all genes.
        level_cols: 7 leiden column names sorted by level.
        gene_names: Gene names from the TSV (same order as cluster_assignments).
        consensus_dir: Directory with ``res_X.XX/consensus_matrix.npz`` subdirs.
        consensus_resolutions: 6 resolutions mapping to levels 2-7.
        all_gene_names: Gene names from the consensus matrices (row/col order).

    Returns:
        cohesion_by_level: {level_col: {cluster_id: mean_cohesion}}.
    """
    if len(consensus_resolutions) != len(level_cols) - 1:
        raise ValueError(
            f"Expected {len(level_cols) - 1} consensus resolutions "
            f"(one per active level), got {len(consensus_resolutions)}"
        )

    consensus_gene_index: Dict[str, int] = {
        g: i for i, g in enumerate(all_gene_names)
    }

    tsv_gene_set = set(gene_names)
    consensus_gene_set = set(all_gene_names)
    common_genes = tsv_gene_set & consensus_gene_set
    logger.info(
        f"Gene overlap: {len(common_genes)} common genes; "
        f"{len(tsv_gene_set) - len(common_genes)} TSV-only; "
        f"{len(consensus_gene_set) - len(common_genes)} consensus-only"
    )

    cohesion_by_level: Dict[str, Dict[int, float]] = {}

    for level_idx, (level_col, res) in enumerate(
        zip(level_cols[1:], consensus_resolutions)
    ):
        res_str = f"{res:.2f}"
        npz_path = consensus_dir / f"res_{res_str}" / "consensus_matrix.npz"

        cluster_ids = np.unique(cluster_assignments[level_col])

        if not npz_path.exists():
            logger.warning(
                f"Missing consensus matrix: {npz_path} — "
                f"assigning 0.0 cohesion for all clusters at {level_col}"
            )
            cohesion_by_level[level_col] = {int(c): 0.0 for c in cluster_ids}
            continue

        logger.info(
            f"Loading consensus matrix res={res_str} for level {level_col} "
            f"({len(cluster_ids)} clusters) ..."
        )
        data = np.load(npz_path)
        M = data["M"]
        M = (M + M.T) / 2.0
        np.fill_diagonal(M, 1.0)

        cohesion: Dict[int, float] = {}
        n_dropped_total = 0

        for clust_id in cluster_ids:
            mask = cluster_assignments[level_col] == clust_id
            cluster_genes = [g for g, m in zip(gene_names, mask) if m]

            valid_genes = [g for g in cluster_genes if g in consensus_gene_index]
            n_dropped = len(cluster_genes) - len(valid_genes)
            n_dropped_total += n_dropped

            if len(valid_genes) == 0:
                cohesion[int(clust_id)] = 0.0
                continue

            indices = np.array([consensus_gene_index[g] for g in valid_genes])
            cohesion[int(clust_id)] = compute_cluster_cohesion(M, indices)

        if n_dropped_total > 0:
            logger.warning(
                f"  Level {level_col}: {n_dropped_total} genes dropped "
                f"(not in consensus matrix)"
            )

        cohesion_by_level[level_col] = cohesion

        vals = list(cohesion.values())
        logger.info(
            f"  Level {level_col}: {len(cluster_ids)} clusters, "
            f"mean cohesion={np.mean(vals):.3f} ± {np.std(vals):.3f}"
        )

        del M, data
        gc.collect()

    return cohesion_by_level


# ---------------------------------------------------------------------------
# Hierarchy construction
# ---------------------------------------------------------------------------

def build_eval_hierarchy(
    gene_names: List[str],
    cluster_assignments: Dict[str, np.ndarray],
    level_cols: List[str],
    tuned_resolutions: List[float],
    cohesion_by_level: Dict[str, Dict[int, float]],
    overlap_threshold: float = 0.1,
) -> nx.DiGraph:
    """Build directed hierarchy graph from evaluation pipeline cluster labels.

    Level 1 (single cluster = all genes) is represented by the root node.
    Levels 2-7 become hierarchy levels keyed by their tuned resolution.

    Node keys: ``("root", 0)`` or ``(tuned_res, cluster_id)``.
    Node attrs: ``genes`` (set), ``size`` (int), ``mean_consensus`` (float).

    Args:
        gene_names: Ordered list of gene names.
        cluster_assignments: {level_col: cluster id array}.
        level_cols: 7 leiden column names sorted by level.
        tuned_resolutions: 7 floats parsed from column names.
        cohesion_by_level: {level_col: {cluster_id: cohesion}}.
        overlap_threshold: Min fractional overlap for parent->child edges.

    Returns:
        nx.DiGraph
    """
    G = nx.DiGraph()

    all_genes = set(gene_names)
    root_node = ("root", 0)
    G.add_node(root_node, genes=all_genes, size=len(all_genes), mean_consensus=0.0)

    active_level_cols = level_cols[1:]
    active_tuned_res = tuned_resolutions[1:]

    clusters_genes: Dict[str, Dict[int, Set[str]]] = {}

    for level_col, tuned_res in zip(active_level_cols, active_tuned_res):
        clusters_genes[level_col] = {}
        cluster_ids = np.unique(cluster_assignments[level_col])
        level_cohesion = cohesion_by_level.get(level_col, {})

        for clust_id in cluster_ids:
            mask = cluster_assignments[level_col] == clust_id
            genes: Set[str] = {g for g, m in zip(gene_names, mask) if m}
            coh = level_cohesion.get(int(clust_id), 0.0)

            node = (tuned_res, int(clust_id))
            G.add_node(node, genes=genes, size=len(genes), mean_consensus=coh)
            clusters_genes[level_col][int(clust_id)] = genes

    first_col = active_level_cols[0]
    first_res = active_tuned_res[0]
    for clust_id in clusters_genes[first_col]:
        G.add_edge(root_node, (first_res, clust_id))

    for i in range(len(active_level_cols) - 1):
        parent_col = active_level_cols[i]
        child_col = active_level_cols[i + 1]
        parent_res = active_tuned_res[i]
        child_res = active_tuned_res[i + 1]

        for child_id, child_genes in clusters_genes[child_col].items():
            if not child_genes:
                continue
            for parent_id, parent_genes in clusters_genes[parent_col].items():
                overlap = len(child_genes & parent_genes) / len(child_genes)
                if overlap > overlap_threshold:
                    G.add_edge((parent_res, parent_id), (child_res, child_id))

    logger.info(
        f"Hierarchy graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges across {len(active_level_cols)} levels"
    )
    return G


# ---------------------------------------------------------------------------
# GO enrichment
# ---------------------------------------------------------------------------

def run_tree_enrichment(
    G: nx.DiGraph,
    tuned_resolutions: List[float],
    skip_enrichment: bool = True,
) -> Dict[Tuple, Optional[pd.DataFrame]]:
    """Run g:Profiler GO enrichment for each node in the tree.

    Skips nodes with < 5 genes or the root node.
    Uses GO:CC for non-final levels, GO:MF + GO:BP for the finest level.
    """
    enrich: Dict = {}

    if skip_enrichment:
        for node in G.nodes:
            enrich[node] = None
        logger.info("Skipping GO enrichment (--skip-enrichment)")
        return enrich

    from gprofiler import GProfiler  # type: ignore
    gp = GProfiler(return_dataframe=True)

    active_resolutions = sorted(r for r in tuned_resolutions if r != "root")
    finest_res = active_resolutions[-1] if active_resolutions else None

    total = len(G.nodes)
    done = 0

    for node in G.nodes:
        res = node[0]
        genes_set = G.nodes[node].get("genes", set())

        if len(genes_set) < 5 or res == "root":
            enrich[node] = None
            done += 1
            continue

        genes = list({g.split("+")[0] if "+" in g else g for g in genes_set})
        sources = ["GO:CC"] if res != finest_res else ["GO:MF", "GO:BP"]

        try:
            df = gp.profile(
                organism="hsapiens",
                query=genes,
                sources=sources,
                no_evidences=False,
                user_threshold=0.05,
            )
        except Exception as e:
            logger.warning(f"Enrichment failed for node {node}: {e}")
            df = None

        enrich[node] = df
        done += 1
        if done % 50 == 0:
            logger.info(f"  Enrichment progress: {done}/{total}")

    logger.info(f"Enrichment complete: {done}/{total} nodes processed")
    return enrich


# ---------------------------------------------------------------------------
# Single-cell tree cell-ID discovery
# ---------------------------------------------------------------------------

def discover_cell_ids(single_cell_dir: Path) -> List[str]:
    """Return sorted list of cell ID strings found in single_cell_dir.

    Looks for directories matching ``cell_*/`` that contain
    ``cluster_labels_generated.tsv``.
    """
    cell_ids = []
    for p in sorted(single_cell_dir.iterdir()):
        if p.is_dir() and p.name.startswith("cell_"):
            if (p / "cluster_labels_generated.tsv").exists():
                cell_ids.append(p.name[len("cell_"):])
    return cell_ids


def resolve_cell_ids(
    single_cell_dir: Path,
    cell_ids: Optional[List[str]],
    n_cells: Optional[int],
    seed: int,
) -> List[str]:
    """Resolve which cell IDs to process.

    Args:
        single_cell_dir: Directory with cell_* subdirectories.
        cell_ids: Explicit list (``--cell-ids``).
        n_cells: Sample this many cells randomly (``--n-cells``).
        seed: Random seed for sampling.

    Returns:
        Sorted list of cell ID strings (zero-padded, e.g. "062").
    """
    if cell_ids:
        missing = [
            cid for cid in cell_ids
            if not (single_cell_dir / f"cell_{cid}").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Cell directories not found in {single_cell_dir}: "
                + ", ".join(f"cell_{m}" for m in missing)
            )
        return sorted(cell_ids)

    available = discover_cell_ids(single_cell_dir)
    if not available:
        raise FileNotFoundError(
            f"No cell_* directories with cluster_labels_generated.tsv "
            f"found in {single_cell_dir}"
        )

    if n_cells is not None:
        n_cells = min(n_cells, len(available))
        rng = random.Random(seed)
        selected = sorted(rng.sample(available, n_cells))
        logger.info(
            f"Sampled {n_cells} cells (seed={seed}) from "
            f"{len(available)} available: {selected}"
        )
        return selected

    raise ValueError("Provide either --cell-ids or --n-cells")
