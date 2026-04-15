#!/usr/bin/env python3
"""
Hierarchical Leiden clustering and hierarchy visualization utilities.

Extracted from the SubCell evaluation pipeline for standalone use in the
paper's tree comparison workflow.

Author: Konstantin Kahnert
"""

import math
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple, Optional

from gprofiler import GProfiler


def hierarchical_leiden(
    adata, resolutions: List[float], neighbors: List[int], random_state: int = 42,
    metric: str = "cosine"
):
    """Perform hierarchical Leiden clustering on AnnData object.

    Args:
        adata: AnnData object with embeddings in X
        resolutions: List of resolution parameters for each level
        neighbors: List of neighbor counts for each level
        random_state: Random seed for reproducibility
        metric: Similarity metric for neighbor computation ('cosine' or 'euclidean')
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
            if f"leiden_lvl{level_idx}_res{res}" not in adata.obs.columns:
                adata.obs[f"leiden_lvl{level_idx}_res{res}"] = pd.Categorical(
                    values=[None] * adata.shape[0]
                )
            if new_category not in adata.obs[target_col].cat.categories:
                adata.obs[target_col] = adata.obs[target_col].cat.add_categories(
                    [new_category]
                )
            adata.obs.loc[parent_mask, target_col] = new_category
            return

        sc.pp.neighbors(
            sub_adata,
            n_neighbors=n_neighbors,
            n_pcs=None,
            use_rep="X",
            metric=metric,
            random_state=random_state,
        )
        sc.tl.leiden(sub_adata, flavor="leidenalg", resolution=res, n_iterations=2)

        for sub_label in sub_adata.obs["leiden"].unique():
            full_label = f"{parent_label}.{sub_label}"
            if f"leiden_lvl{level_idx}_res{res}" not in adata.obs.columns:
                adata.obs[f"leiden_lvl{level_idx}_res{res}"] = pd.Categorical(
                    values=[None] * adata.shape[0]
                )
            if full_label not in adata.obs[target_col].cat.categories:
                adata.obs[target_col] = adata.obs[target_col].cat.add_categories(
                    [full_label]
                )

            subcluster_mask = sub_adata.obs["leiden"] == sub_label
            subcluster_positions = np.where(subcluster_mask)[0]
            original_mask = parent_mask.to_numpy().copy()
            original_mask[original_mask] = False
            parent_true_positions = np.where(parent_mask)[0]
            original_mask[parent_true_positions[subcluster_positions]] = True

            adata.obs.loc[original_mask, target_col] = full_label

    # --- main flow ---
    sc.pp.neighbors(
        adata,
        n_neighbors=neighbors[0],
        n_pcs=None,
        use_rep="X",
        metric=metric,
        random_state=random_state,
    )
    sc.tl.leiden(adata, flavor="leidenalg", resolution=resolutions[0], n_iterations=2)

    adata.obs[f"leiden_lvl0_res{resolutions[0]}"] = adata.obs["leiden"].astype("category")

    for i in range(1, len(resolutions)):
        res = resolutions[i]
        n_neighbors = neighbors[i]
        parent_res = resolutions[i - 1]

        parent_col = f"leiden_lvl{i-1}_res{parent_res}"
        ensure_cat_col(adata, parent_col)

        parent_clusters = adata.obs[parent_col].unique()

        for parent_label in parent_clusters:
            subcluster(adata, parent_col, res, n_neighbors, parent_label, level_idx=i)

        col = f"leiden_lvl{i}_res{res}"
        ensure_cat_col(adata, col)
        uniq = pd.Series(
            adata.obs[col].astype(str).values, index=adata.obs.index
        ).unique()
        mapping = {old: str(new) for new, old in enumerate(uniq)}
        adata.obs[col] = pd.Series(
            adata.obs[col].astype(str).map(mapping).values,
            index=adata.obs.index,
            dtype="category",
        )

    adata.uns["hierarchy_edges"] = []

    return adata


def build_hierarchy_graph(adata, resolutions: List[float], max_levels: Optional[int] = None):
    """Build hierarchy graph from clustering results.

    Args:
        adata: AnnData object with clustering results
        resolutions: List of resolution parameters
        max_levels: If set, only use the first N hierarchy levels

    Returns:
        NetworkX directed graph representing the hierarchy
    """
    if max_levels is not None:
        resolutions = resolutions[:max_levels]

    G = nx.DiGraph()

    clusters_genes = {}
    for level_idx, res in enumerate(resolutions):
        clusters_genes[res] = {}
        col = f"leiden_lvl{level_idx}_res{res}"
        labels = np.array(adata.obs[col])
        for lab in np.unique(labels):
            members = np.where(labels == lab)[0]
            gn = adata.obs["gene_name"].to_numpy()
            genes = set(gn[members])
            clusters_genes[res][lab] = genes
            G.add_node((res, lab), genes=genes)

    for i in range(len(resolutions) - 1):
        res_i = resolutions[i]
        res_j = resolutions[i + 1]
        for cluster_i in clusters_genes[res_i]:
            genes_i = clusters_genes[res_i][cluster_i]
            for cluster_j in clusters_genes[res_j]:
                genes_j = clusters_genes[res_j][cluster_j]
                if len(genes_j.intersection(genes_i)) / len(genes_j) > 0.1:
                    G.add_edge((res_i, cluster_i), (res_j, cluster_j))

    return G


def run_enrichment(adata, resolutions: List[float]):
    """Run GO enrichment analysis for each cluster.

    Args:
        adata: AnnData object with clustering results
        resolutions: List of resolution parameters

    Returns:
        Dictionary mapping (resolution, cluster) to enrichment results
    """
    gp = GProfiler(return_dataframe=True)
    enrich: Dict[Tuple[float, str], Optional[object]] = {}

    total = 0
    res_to_labels: List[Tuple[float, List[str]]] = []
    for level_idx, res in enumerate(resolutions):
        labels = list(np.unique(adata.obs[f"leiden_lvl{level_idx}_res{res}"]))
        res_to_labels.append((res, labels))
        total += len(labels)

    done = 0

    jobs: List[Tuple[Tuple[float, str], List[str], List[str]]] = []
    for res, labels in res_to_labels:
        for lab in labels:
            level_idx = resolutions.index(res)
            members = np.where(adata.obs[f"leiden_lvl{level_idx}_res{res}"] == lab)[0]
            gene_names_full = adata.obs["gene_name"].to_numpy()[members]
            genes = []
            for gene_name in gene_names_full:
                if "+" in gene_name:
                    genes.append(gene_name.split("+")[0])
                else:
                    genes.append(gene_name)
            genes = list(set(genes))
            node = (res, lab)

            if len(genes) < 5 or res == resolutions[0]:
                enrich[node] = None
                done += 1
                continue

            sources = ["GO:CC"] if res != resolutions[-1] else ["GO:MF", "GO:BP"]
            jobs.append((node, genes, sources))

    def _do_job(job):
        node, genes, sources = job
        try:
            df = gp.profile(
                organism="hsapiens",
                query=genes,
                sources=sources,
                no_evidences=False,
                user_threshold=0.05,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "GO enrichment failed for node %s: %s", node, e
            )
            df = None
        return node, df

    for job in jobs:
        node, df = _do_job(job)
        enrich[node] = df
        done += 1

    return enrich


def visualize_hierarchy(G, enrich_results, out_svg: str, resolutions: List[float], max_levels: Optional[int] = None):
    """Visualize hierarchy graph with enrichment results.

    Args:
        G: NetworkX graph
        enrich_results: Dictionary of enrichment results
        out_svg: Output SVG file path
        resolutions: List of resolution parameters
        max_levels: If set, only use the first N hierarchy levels
    """
    if max_levels is not None:
        resolutions = resolutions[:max_levels]

    cmap = plt.get_cmap("Reds")
    truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
        "truncated_reds", cmap(np.linspace(0, 0.85, 256))
    )

    level_dict = {res: i + 1 for i, res in enumerate(resolutions)}
    node_sizes, node_colors = [], []
    label_dict = {}

    for node in G.nodes:
        num_genes = len(G.nodes[node].get("genes", []))
        node_sizes.append(400 + num_genes)
        level = level_dict[node[0]]

        df = enrich_results.get(node)
        if df is not None and hasattr(df, "empty") and not df.empty:
            term = df.iloc[0]
            label_dict[node] = f"{level}.{node[1]}: {term['name']}"
            pv = float(term["p_value"])
            node_colors.append(max(0.0, min(50.0, -math.log10(max(pv, 1e-300)))))
        else:
            label_dict[node] = f"{level}.{node[1]}: -"
            node_colors.append(0.0)

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB -Gnodesep=1 -Granksep=0.7")
    plt.figure(figsize=(58, 22))

    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color=node_colors, cmap=truncated_cmap
    )
    nx.draw_networkx_edges(G, pos, edge_color="#D3D3D3", width=2, arrows=False)

    last_resolution = resolutions[-1] if resolutions else None

    labels = {}
    for key, (x, y) in pos.items():
        labels[key] = label_dict.get(key, f"{level_dict[key[0]]}.{key[1]}")
        rotation = -90 if key[0] == last_resolution else -30
        plt.text(
            x,
            y - 7.5,
            labels[key],
            fontsize=16,
            fontweight="bold",
            ha="left",
            va="top",
            rotation=rotation,
        )

    plt.colorbar(
        nodes, label="-log10(p-value)", fraction=0.01, location="right", aspect=30
    )
    plt.title("Hierarchical Leiden clustering tree with most enriched GO terms")
    plt.axis("off")

    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    out_png = out_svg.replace(".svg", ".png")
    plt.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   Hierarchy visualization saved to: {out_svg}")
