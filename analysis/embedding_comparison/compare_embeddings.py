# -*- coding: utf-8 -*-
"""Compare protein embeddings using similarity metrics and UMAP visualizations.

Runs both analyses by default; use --skip-metrics or --skip-visual to run only one.

Metrics: CKA, Jaccard@k (raw + chance-corrected), Precision/Recall, MMD, SWD,
         per-dataset spread (Total Variance, ADC, Mean k-NN Distance).

Author: Konstantin Kahnert
"""

import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'
if 'MKL_NUM_THREADS' not in os.environ:
    os.environ['MKL_NUM_THREADS'] = '1'
if 'OPENBLAS_NUM_THREADS' not in os.environ:
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

matplotlib.rcParams['svg.fonttype'] = 'none'

try:
    from prdc import compute_prdc
    PRDC_AVAILABLE = True
except ImportError:
    PRDC_AVAILABLE = False

from embedding_utils import (
    _to_dense,
    _filter_and_align,
    zscore_fit_transform,
    load_unnormalized_real_embeddings,
    filter_and_aggregate_real_embeddings_by_cell_line,
    load_multiple_embeddings,
    save_figure as _save_figure,
    assign_primary_location,
)

logger = logging.getLogger(__name__)

MIN_SAMPLES = {'precision_recall': 50}


# -----------------------------------------------------------------------
# Metric Computation
# -----------------------------------------------------------------------

def _gram(E):
    return E @ E.T


def _center(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _cka_linear(E1, E2):
    K1, K2 = _center(_gram(E1)), _center(_gram(E2))
    num = np.sum(K1 * K2)
    den = np.sqrt(np.sum(K1 * K1) * np.sum(K2 * K2))
    return float(num / den)


def _knn_idx(E, k=20, metric="cosine"):
    k = min(k, max(1, E.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(E)
    return nn.kneighbors(E)[1][:, 1:]


def _jaccard_rows(idx1, idx2):
    out = np.empty(idx1.shape[0], float)
    for i, (a, b) in enumerate(zip(idx1, idx2)):
        sa, sb = set(a), set(b)
        out[i] = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
    return out


def _expected_jaccard(k: int, n: int) -> float:
    return k / (2 * n - k) if (n > 0 and k > 0 and k < n) else np.nan


def _chance_corrected_jaccard(observed: float, k: int, n: int) -> float:
    j_exp = _expected_jaccard(k, n)
    if np.isnan(j_exp) or j_exp >= 1.0:
        return np.nan
    return (observed - j_exp) / (1 - j_exp)


def _preservation_matrix(idx_x, idx_y, k_values):
    k_values = [k for k in k_values if 1 <= k <= idx_x.shape[1]]
    M = np.zeros((idx_x.shape[0], len(k_values)), dtype=float)
    for j, k in enumerate(k_values):
        for i in range(idx_x.shape[0]):
            M[i, j] = len(set(idx_x[i, :k]) & set(idx_y[i, :k])) / k
    return M, k_values


def compute_sliced_wasserstein(X1, X2, n_projections=1000, seed=42):
    """Sliced Wasserstein Distance between two embedding sets."""
    rng = np.random.default_rng(seed)
    d = X1.shape[1]
    distances = []
    for _ in range(n_projections):
        theta = rng.normal(0, 1, d)
        theta /= np.linalg.norm(theta)
        distances.append(wasserstein_distance(X1 @ theta, X2 @ theta))
    return float(np.mean(distances))


def compute_mmd(X1, X2, kernel='rbf', gamma=None):
    """Maximum Mean Discrepancy with RBF kernel."""
    n1, n2 = X1.shape[0], X2.shape[0]
    if gamma is None:
        sub = np.vstack([X1[:5000], X2[:5000]])
        if sub.shape[0] > 1:
            med = np.median(pdist(sub, metric='euclidean'))
            gamma = 1.0 if (not np.isfinite(med) or med <= 0) else 1.0 / (2 * med ** 2 + 1e-6)
        else:
            gamma = 1.0

    K_XX = rbf_kernel(X1, X1, gamma=gamma)
    K_YY = rbf_kernel(X2, X2, gamma=gamma)
    K_XY = rbf_kernel(X1, X2, gamma=gamma)

    t1 = (np.sum(K_XX) - np.trace(K_XX)) / (n1 * (n1 - 1))
    t2 = (np.sum(K_YY) - np.trace(K_YY)) / (n2 * (n2 - 1))
    t3 = np.sum(K_XY) / (n1 * n2)
    return {'mmd': float(np.sqrt(max(0.0, t1 + t2 - 2 * t3))), 'gamma': float(gamma)}


def compute_precision_recall(real_features, fake_features, nearest_k=5):
    """Distributional Precision and Recall via prdc library."""
    if not PRDC_AVAILABLE:
        return {'precision': float('nan'), 'recall': float('nan')}
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)
    if len(real_features) < nearest_k + 1 or len(fake_features) < nearest_k + 1:
        return {'precision': float('nan'), 'recall': float('nan')}
    try:
        m = compute_prdc(real_features=real_features, fake_features=fake_features, nearest_k=nearest_k)
        return {'precision': float(m['precision']), 'recall': float(m['recall'])}
    except (ValueError, IndexError):
        return {'precision': float('nan'), 'recall': float('nan')}


# -----------------------------------------------------------------------
# Per-Dataset Spread Metrics
# -----------------------------------------------------------------------

def compute_total_variance(X): return float(np.sum(np.var(X, axis=0, ddof=1)))
def compute_adc(X): return float(np.mean(np.linalg.norm(X - X.mean(axis=0), axis=1)))

def compute_mean_knn_distance(X, k=5):
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X)
    return float(np.mean(nn.kneighbors(X)[0][:, k]))


# -----------------------------------------------------------------------
# Main Comparison Function
# -----------------------------------------------------------------------

def get_min_samples_thresholds(k_neighbors: int) -> Dict[str, int]:
    return {"cka": 20, "sliced_wasserstein": 20, "mmd": 50,
            "jaccard": 3 * k_neighbors, "precision_recall": 50}


def compare_embeddings_on_shared_real_gene_universe(
    adata_A, adata_B, gene_col="gene_name", k_neighbors=20,
    k_curve=(5, 10, 20, 50), knn_metric="cosine",
    run_precision_recall=True, pr_nearest_k=5,
    cast_float32=True, is_X1_real=None,
):
    """Compare two embeddings after restricting to a shared real-gene universe."""
    X1, X2, genes = _filter_and_align(adata_A, adata_B, gene_col=gene_col, drop_dups=True)
    if cast_float32:
        X1, X2 = X1.astype(np.float32), X2.astype(np.float32)

    if is_X1_real is True:
        X1, X2 = zscore_fit_transform(X1, X2)
    elif is_X1_real is False:
        X2, X1 = zscore_fit_transform(X2, X1)
    else:
        X1, X2 = zscore_fit_transform(X1, X2)

    n = X1.shape[0]
    thresholds = get_min_samples_thresholds(k_neighbors)

    def _safe(name, fn):
        if n < thresholds[name]:
            logger.warning(f"Skipping {name}: {n} < {thresholds[name]} samples")
            return float('nan')
        return float(fn())

    result = {
        "n": n, "gene_order": genes,
        "cka": _safe("cka", lambda: _cka_linear(X1, X2)),
        "sliced_wasserstein": _safe("sliced_wasserstein", lambda: compute_sliced_wasserstein(X1, X2)),
        "mmd": _safe("mmd", lambda: compute_mmd(X1, X2)["mmd"]),
    }

    # Precision / Recall
    if run_precision_recall and PRDC_AVAILABLE and n >= thresholds["precision_recall"]:
        real, fake = (X1, X2) if is_X1_real is not False else (X2, X1)
        pr = compute_precision_recall(real, fake, nearest_k=pr_nearest_k)
    else:
        pr = {'precision': float('nan'), 'recall': float('nan')}
    result.update({"precision": pr['precision'], "recall": pr['recall']})

    # Jaccard
    if n >= thresholds["jaccard"]:
        idx1 = _knn_idx(X1, k_neighbors, knn_metric)
        idx2 = _knn_idx(X2, k_neighbors, knn_metric)
        jacc_mean = float(_jaccard_rows(idx1, idx2).mean())
        result.update({
            "jaccard_mean": jacc_mean,
            "jaccard_corrected": _chance_corrected_jaccard(jacc_mean, k_neighbors, n),
            "jaccard_expected": _expected_jaccard(k_neighbors, n),
            "preservation_matrix": _preservation_matrix(idx1, idx2, list(k_curve))[0],
            "k_curve": list(k_curve),
        })
    else:
        result.update({
            "jaccard_mean": float('nan'), "jaccard_corrected": float('nan'),
            "jaccard_expected": float('nan'),
            "preservation_matrix": np.array([[float('nan')]]),
            "k_curve": list(k_curve),
        })

    return result


# -----------------------------------------------------------------------
# Batch Comparison
# -----------------------------------------------------------------------

def _compare_worker(name1, name2, adata_1, adata_2, gene_col, k_neighbors,
                    k_curve, knn_metric, run_pr, pr_k, cast_f32):
    """Worker for parallel pairwise comparison."""
    try:
        is1r, is2r = name1.startswith("Real_"), name2.startswith("Real_")
        is_X1_real = True if (is1r and not is2r) else (False if (is2r and not is1r) else (name1 <= name2))
        result = compare_embeddings_on_shared_real_gene_universe(
            adata_1, adata_2, gene_col=gene_col, k_neighbors=k_neighbors,
            k_curve=k_curve, knn_metric=knn_metric, run_precision_recall=run_pr,
            pr_nearest_k=pr_k, cast_float32=cast_f32, is_X1_real=is_X1_real,
        )
        return ((name1, name2), result)
    except Exception as e:
        return ((name1, name2), {
            "n": 0, "cka": float('nan'), "sliced_wasserstein": float('nan'),
            "mmd": float('nan'), "jaccard_mean": float('nan'),
            "jaccard_corrected": float('nan'), "jaccard_expected": float('nan'),
            "precision": float('nan'), "recall": float('nan'), "error": str(e),
        })


def compare_embeddings_pairwise(
    embeddings, shared_real_gene_universes, gene_col="gene_name",
    k_neighbors=20, k_curve=(5, 10, 20, 50), knn_metric="cosine",
    run_precision_recall=True, pr_nearest_k=5, cast_float32=True, n_jobs=1,
):
    """Compare all pairs of embeddings. Returns {(name1, name2): result_dict}."""
    names = list(embeddings.keys())
    tasks = []
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i >= j:
                continue
            cl1, cl2 = n1.replace('Real_', ''), n2.replace('Real_', '')
            shared = shared_real_gene_universes[cl1] & shared_real_gene_universes[cl2]
            a1 = embeddings[n1][embeddings[n1].obs[gene_col].astype(str).str.strip().isin(shared)].copy()
            a2 = embeddings[n2][embeddings[n2].obs[gene_col].astype(str).str.strip().isin(shared)].copy()
            tasks.append((n1, n2, a1, a2, gene_col, k_neighbors, k_curve,
                          knn_metric, run_precision_recall, pr_nearest_k, cast_float32))

    logger.info(f"Computing {len(tasks)} pairwise comparisons (n_jobs={n_jobs})...")
    if n_jobs == 1:
        results = {}
        for t in tasks:
            key, res = _compare_worker(*t)
            results[key] = res
            if "error" not in res:
                logger.info(f"  {t[0]} vs {t[1]}: n={res['n']}, CKA={res['cka']:.3f}")
    else:
        par = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
            delayed(_compare_worker)(*t) for t in tasks)
        results = dict(par)
    return results


def compute_spread_metrics(all_embeddings, gene_col="gene_name", knn_k=5):
    """Compute per-dataset spread metrics (variance, ADC, k-NN distance)."""
    results = []
    real_names = sorted(n for n in all_embeddings if n.startswith("Real_"))
    for real_name in real_names:
        cl = real_name.replace("Real_", "")
        gen_name = cl
        if gen_name not in all_embeddings:
            continue
        real_genes = set(all_embeddings[real_name].obs[gene_col].astype(str).str.strip())
        gen_genes = set(all_embeddings[gen_name].obs[gene_col].astype(str).str.strip())
        common = real_genes & gen_genes
        if len(common) < 10:
            continue
        for name, adata in [(real_name, all_embeddings[real_name]), (gen_name, all_embeddings[gen_name])]:
            X = _to_dense(adata[adata.obs[gene_col].astype(str).str.strip().isin(common)].X)
            eff_k = min(knn_k, X.shape[0] - 1)
            results.append({
                'embedding': name, 'cell_line': cl,
                'type': "Real" if name.startswith("Real_") else "Generated",
                'n_genes': X.shape[0], 'n_features': X.shape[1],
                'total_variance': compute_total_variance(X),
                'adc': compute_adc(X),
                'mean_knn_distance': compute_mean_knn_distance(X, k=eff_k) if eff_k >= 1 else float('nan'),
            })
    return results


def load_results_from_summary(summary_path: Path) -> Dict:
    """Load comparison results from a saved summary TSV."""
    df = pd.read_csv(summary_path, sep="\t")
    results = {}
    for _, row in df.iterrows():
        r = {c: float(row[c]) if c != "N_Genes" else int(row[c])
             for c in ["N_Genes", "CKA", "Sliced_Wasserstein", "MMD", "Jaccard_Mean"]
             if c in df.columns}
        r["n"] = r.pop("N_Genes", 0)
        r["cka"] = r.pop("CKA", float('nan'))
        r["sliced_wasserstein"] = r.pop("Sliced_Wasserstein", float('nan'))
        r["mmd"] = r.pop("MMD", float('nan'))
        r["jaccard_mean"] = r.pop("Jaccard_Mean", float('nan'))
        r["jaccard_corrected"] = float(row.get("Jaccard_Corrected", float('nan')))
        r["jaccard_expected"] = float(row.get("Jaccard_Expected", float('nan')))
        r["precision"] = float(row.get("Precision", float('nan')))
        r["recall"] = float(row.get("Recall", float('nan')))
        results[(row["Embedding_1"], row["Embedding_2"])] = r
    logger.info(f"Loaded {len(results)} comparison results from {summary_path}")
    return results


# -----------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------

def create_summary_table(results, output_path):
    rows = []
    for (n1, n2), r in results.items():
        row = {"Embedding_1": n1, "Embedding_2": n2, "N_Genes": r["n"],
               "CKA": r["cka"], "Sliced_Wasserstein": r["sliced_wasserstein"],
               "MMD": r["mmd"], "Jaccard_Mean": r["jaccard_mean"],
               "Jaccard_Corrected": r.get("jaccard_corrected", float('nan')),
               "Jaccard_Expected": r.get("jaccard_expected", float('nan'))}
        if not pd.isna(r.get("precision", float('nan'))):
            row.update({"Precision": r["precision"], "Recall": r["recall"]})
        rows.append(row)
    pd.DataFrame(rows).sort_values(["Embedding_1", "Embedding_2"]).to_csv(
        output_path, sep="\t", index=False)
    logger.info(f"Saved summary table: {output_path}")


def create_comparison_heatmap(results, metric, output_path, save_svg=False):
    """Heatmap of one metric across all embedding pairs."""
    all_names = set()
    for n1, n2 in results:
        all_names.update([n1, n2])
    real = sorted(n for n in all_names if "real" in n.lower())
    other = sorted(n for n in all_names if "real" not in n.lower())
    all_names = real + other

    matrix = np.full((len(all_names), len(all_names)), np.nan)
    for (n1, n2), r in results.items():
        i1, i2 = all_names.index(n1), all_names.index(n2)
        v = r.get(metric, np.nan)
        if not pd.isna(v):
            matrix[i1, i2] = matrix[i2, i1] = v
    np.fill_diagonal(matrix, 0.0 if metric in ['sliced_wasserstein', 'mmd'] else 1.0)

    plt.figure(figsize=(16, 14))
    cmap = sns.color_palette("Blues", as_cmap=True)
    cmap.set_bad(color='lightgray')
    sns.heatmap(matrix, xticklabels=all_names, yticklabels=all_names,
                annot=True, fmt=".2f", cmap=cmap, center=0.5, square=True)
    plt.title(metric.replace('_', ' ').title())
    plt.tight_layout()
    _save_figure(output_path, save_svg)
    # plt.close()


def _build_metric_groups(metric, gr_results, gg_results, rr_results, min_n=None):
    """Collect metric values by comparison group."""
    groups = {}
    for label, src in [('Gen-Real', gr_results), ('Gen-Gen', gg_results), ('Real-Real', rr_results)]:
        if not src:
            continue
        vals = [r[metric] for r in src.values()
                if not pd.isna(r.get(metric, np.nan))
                and (min_n is None or r.get('n', 0) >= min_n)]
        if vals:
            groups[label] = vals
    return groups


def _plot_metric_boxplot(metric, title, groups, gr_results, output_dir,
                         save_svg=False, ylim_01=False, interpretation=None):
    """Single metric boxplot with same-vs-diff cell line coloring."""
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(3.3, 5))
    labels = list(groups.keys())
    data = [groups[l] for l in labels]
    colors = ['steelblue', 'darkorange', 'forestgreen'][:len(labels)]
    positions = [1, 1.7, 2.4][:len(labels)]
    display = {'Gen-Real': 'Gen vs\nReal', 'Gen-Gen': 'Gen vs\nGen', 'Real-Real': 'Real vs\nReal'}

    bp = ax.boxplot(data, positions=positions, labels=[display.get(l, l) for l in labels],
                    patch_artist=True, showmeans=True, meanline=True, showfliers=False, widths=0.44)
    ax.set_xlim(positions[0] - 0.52, positions[-1] + 0.52)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    for pos, label in zip(positions, labels):
        if label == 'Gen-Real' and gr_results:
            same, diff = [], []
            for (n1, n2), r in gr_results.items():
                if pd.isna(r.get(metric, np.nan)):
                    continue
                (same if n1.replace('Real_', '') == n2.replace('Real_', '') else diff).append(r[metric])
            if diff:
                ax.scatter(np.random.normal(pos, 0.04, len(diff)), diff, alpha=0.4, s=30,
                           color='black', label='Diff cell line')
            if same:
                ax.scatter(np.random.normal(pos, 0.04, len(same)), same, alpha=0.5, s=30,
                           color='#C0504D', edgecolors='#8B0000', linewidths=0.5, label='Same cell line')
        else:
            vals = groups[label]
            ax.scatter(np.random.normal(pos, 0.04, len(vals)), vals, alpha=0.4, s=30, color='black')

    is_sim = metric in ['cka', 'jaccard_mean', 'jaccard_corrected', 'precision', 'recall']
    interp = interpretation or ('(Higher is better)' if is_sim else '(Lower is more similar)')
    ax.set_ylabel(title, fontsize=9)
    ax.set_title(f'{title}\n{interp}', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim((0, 1.05) if ylim_01 else (0, None))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles, leg_labels = ax.get_legend_handles_labels()
    if leg_labels:
        ax.legend(handles, leg_labels, loc='best', fontsize=10, framealpha=0.9)

    ax.tick_params(labelsize=9)
    ax.set_position([0.18, 0.12, 0.72, 0.62])
    _save_figure(output_dir / f'{metric}_boxplot.png', save_svg, tight_bbox=False)
    # plt.close()


def create_spread_box_plot(spread_results, output_path, save_svg=False):
    """Box+strip plots comparing Real vs Generated spread metrics."""
    df = pd.DataFrame(spread_results)
    if df.empty:
        return
    metrics = [('total_variance', 'Total Variance'), ('adc', 'Avg Dist to Centroid'),
               ('mean_knn_distance', 'Mean k-NN Distance')]
    palette = {'Real': '#2ca02c', 'Generated': '#1f77b4'}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (key, label) in zip(axes, metrics):
        sns.boxplot(data=df, x='type', y=key, order=['Real', 'Generated'], palette=palette,
                    width=0.45, showfliers=False, ax=ax)
        sns.stripplot(data=df, x='type', y=key, order=['Real', 'Generated'],
                      color='black', size=6, alpha=0.7, jitter=True, ax=ax)
        ax.set_xlabel(''); ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Embedding Spread: Real vs Generated', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_figure(output_path, save_svg)
    # plt.close()


def create_umap_plots(real_embeddings, generated_embeddings, output_path, save_svg=False):
    """Multi-panel UMAP of embeddings by cell line and subcellular location."""
    import scanpy as sc
    import matplotlib.gridspec as gridspec

    logger.info("Creating UMAP visualizations...")
    all_embs = {}
    for name, adata in real_embeddings.items():
        a = adata.copy()
        a.obs["embedding_type"], a.obs["cell_line"] = "Real", name.replace("Real_", "")
        all_embs[name] = a
    for name, adata in generated_embeddings.items():
        a = adata.copy()
        a.obs["embedding_type"], a.obs["cell_line"] = "Generated", name
        all_embs[name] = a

    cell_lines = sorted(set(n.replace("Real_", "") for n in real_embeddings))
    tab20 = list(plt.cm.tab20.colors)
    tab20b = list(plt.cm.tab20b.colors)
    ext = tab20 + [tab20b[14], tab20b[15], tab20b[18], tab20b[19]]
    cmap_type = {}
    for i, cl in enumerate(cell_lines):
        cmap_type[f"{cl} (Real)"] = ext[i * 2]
        cmap_type[f"{cl} (Generated)"] = ext[i * 2 + 1]

    combined = sc.concat(list(all_embs.values()), join="outer", fill_value=0)
    combined.obs_names_make_unique()
    combined.obs["cell_line_type"] = combined.obs["cell_line"] + " (" + combined.obs["embedding_type"] + ")"
    sc.pp.neighbors(combined, n_neighbors=25, n_pcs=150)
    sc.tl.umap(combined, random_state=42)

    real_sub = combined[combined.obs["embedding_type"] == "Real"].copy()
    gen_sub = combined[combined.obs["embedding_type"] == "Generated"].copy()

    gene_col = "gene_name" if "gene_name" in real_sub.obs.columns else "gene_names"
    g2l = dict(zip(real_sub.obs[gene_col], real_sub.obs.get("locations", "Unknown")))
    gen_sub.obs["locations"] = (gen_sub.obs[gene_col].map(g2l) if gene_col in gen_sub.obs.columns
                                else gen_sub.obs.index.map(g2l)).fillna("Unknown")
    for a in [real_sub, gen_sub]:
        a.obs["location_label"] = a.obs.get("locations", pd.Series("Unknown")).apply(assign_primary_location)

    fig = plt.figure(figsize=(16, 24))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    axes = [fig.add_subplot(gs[0, :]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
            fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]

    sc.pl.umap(combined, color="cell_line_type", palette=cmap_type, ax=axes[0],
               title="All Embeddings", show=False, legend_loc="right margin")
    cmap_r = {cl: ext[i * 2] for i, cl in enumerate(cell_lines)}
    cmap_g = {cl: ext[i * 2 + 1] for i, cl in enumerate(cell_lines)}
    sc.pl.umap(real_sub, color="cell_line", palette=cmap_r, ax=axes[1],
               title="Real", show=False, legend_loc="right margin")
    sc.pl.umap(gen_sub, color="cell_line", palette=cmap_g, ax=axes[2],
               title="Generated", show=False, legend_loc="right margin")
    sc.pl.umap(real_sub, color="location_label", ax=axes[3],
               title="Real by Location", show=False, legend_loc="right margin")
    sc.pl.umap(gen_sub, color="location_label", ax=axes[4],
               title="Generated by Location", show=False, legend_loc="right margin")

    for sep_name, sep_list, palette, ax_idx in [
        ("real", [a for n, a in all_embs.items() if n.startswith("Real_")], cmap_r, 5),
        ("gen", [a for n, a in all_embs.items() if not n.startswith("Real_")], cmap_g, 6),
    ]:
        if sep_list:
            sep = sc.concat(sep_list, join="outer", fill_value=0)
            sep.obs_names_make_unique()
            sc.pp.neighbors(sep, n_neighbors=25, n_pcs=150)
            sc.tl.umap(sep)
            sc.pl.umap(sep, color="cell_line", palette=palette, ax=axes[ax_idx],
                       title=f"{sep_name.title()} (Separate UMAP)", show=False, legend_loc="right margin")

    for ax in axes:
        for c in ax.collections:
            c.set_rasterized(True)
        ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout(pad=2.0)
    _save_figure(output_path, save_svg)
    # plt.close()
    logger.info(f"Saved UMAP: {output_path}")


# -----------------------------------------------------------------------
# Categorize results
# -----------------------------------------------------------------------

def _categorize_results(all_results, real_embeddings=None, generated_embeddings=None):
    """Split results into gen-vs-real, gen-vs-gen, real-vs-real."""
    gr, gg, rr = {}, {}, {}
    for (n1, n2), r in all_results.items():
        is1r = n1.startswith("Real_") if real_embeddings is None else n1 in real_embeddings
        is2r = n2.startswith("Real_") if real_embeddings is None else n2 in real_embeddings
        if is1r != is2r:
            gr[(n1, n2)] = r
        elif not is1r:
            gg[(n1, n2)] = r
        else:
            rr[(n1, n2)] = r
    return gr, gg, rr


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def _run_metrics(args, all_embeddings, real_embeddings, generated_embeddings,
                 shared_real_gene_universes, metrics_dir):
    """Run all metrics computations and generate plots."""
    all_results = compare_embeddings_pairwise(
        all_embeddings, shared_real_gene_universes, gene_col=args.gene_col,
        k_neighbors=args.k_neighbors, run_precision_recall=not args.skip_precision_recall,
        pr_nearest_k=args.pr_nearest_k, n_jobs=args.n_jobs,
    )
    if not all_results:
        logger.error("No comparisons completed")
        sys.exit(1)

    gr, gg, rr = _categorize_results(all_results, real_embeddings, generated_embeddings)
    logger.info(f"Gen-Real: {len(gr)}, Gen-Gen: {len(gg)}, Real-Real: {len(rr)}")

    if not args.skip_spread_metrics:
        spread = compute_spread_metrics(all_embeddings, args.gene_col, knn_k=args.spread_knn_k)
        if spread:
            pd.DataFrame(spread).to_csv(metrics_dir / "spread_metrics_summary.tsv", sep='\t', index=False)
            create_spread_box_plot(spread, metrics_dir / "spread_metrics_box_plot.png", args.save_svg)

    # Heatmaps
    for m in ['cka', 'jaccard_corrected', 'sliced_wasserstein', 'mmd']:
        if any(m in r and not pd.isna(r.get(m)) for r in all_results.values()):
            create_comparison_heatmap(all_results, m, metrics_dir / f"{m}_heatmap.png", args.save_svg)

    # Boxplots
    for m, title in [('cka', 'CKA'), ('jaccard_corrected', 'Jaccard (Corrected)'),
                     ('sliced_wasserstein', 'SWD'), ('mmd', 'MMD')]:
        groups = _build_metric_groups(m, gr, gg, rr)
        _plot_metric_boxplot(m, title, groups, gr, metrics_dir, args.save_svg,
                             ylim_01=m in ['cka', 'jaccard_corrected'])
    if not args.skip_precision_recall:
        for m, title in [('precision', 'Precision'), ('recall', 'Recall')]:
            groups = _build_metric_groups(m, gr, gg, rr, min_n=MIN_SAMPLES['precision_recall'])
            _plot_metric_boxplot(m, title, groups, gr, metrics_dir, args.save_svg,
                                ylim_01=True, interpretation='(Higher is better)')

    create_summary_table(all_results, metrics_dir / "comparison_summary.tsv")
    return all_results, gr, gg, rr


def _run_replot(args, metrics_dir):
    """Replot from saved results."""
    summary_path = metrics_dir / "comparison_summary.tsv"
    if not summary_path.exists():
        logger.error(f"Summary not found: {summary_path}")
        sys.exit(1)
    all_results = load_results_from_summary(summary_path)
    gr, gg, rr = _categorize_results(all_results)

    for m, title in [('cka', 'CKA'), ('jaccard_corrected', 'Jaccard (Corrected)'),
                     ('sliced_wasserstein', 'SWD'), ('mmd', 'MMD')]:
        groups = _build_metric_groups(m, gr, gg, rr)
        _plot_metric_boxplot(m, title, groups, gr, metrics_dir, args.save_svg,
                             ylim_01=m in ['cka', 'jaccard_corrected'])
    for m in ['cka', 'jaccard_corrected', 'sliced_wasserstein', 'mmd']:
        if any(m in r for r in all_results.values()):
            create_comparison_heatmap(all_results, m, metrics_dir / f"{m}_heatmap.png", args.save_svg)
    if not args.skip_precision_recall:
        for m, title in [('precision', 'Precision'), ('recall', 'Recall')]:
            groups = _build_metric_groups(m, gr, gg, rr, min_n=MIN_SAMPLES['precision_recall'])
            _plot_metric_boxplot(m, title, groups, gr, metrics_dir, args.save_svg,
                                ylim_01=True, interpretation='(Higher is better)')


def main():
    parser = argparse.ArgumentParser(description="Compare protein embeddings")
    parser.add_argument("--real-embedding")
    parser.add_argument("--hpa-csv")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--gen-embedding-filename", default="embeddings_aggregated.h5ad")
    parser.add_argument("--cell-lines", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--visual-output-dir")
    parser.add_argument("--gene-col", default="gene_name")
    parser.add_argument("--save-svg", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--k-neighbors", type=int, default=25)
    parser.add_argument("--skip-precision-recall", action="store_true")
    parser.add_argument("--pr-nearest-k", type=int, default=5)
    parser.add_argument("--replot-only", action="store_true")
    parser.add_argument("--spread-knn-k", type=int, default=5)
    parser.add_argument("--skip-spread-metrics", action="store_true")
    parser.add_argument("--replot-spread-only", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_dir = Path(args.base_dir)
    metrics_dir = Path(args.output_dir) if args.output_dir else base_dir / "comparison" / "metrics"
    visual_dir = Path(args.visual_output_dir) if args.visual_output_dir else base_dir / "comparison" / "visual"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Spread-only replot
    if args.replot_spread_only:
        tsv = metrics_dir / "spread_metrics_summary.tsv"
        if not tsv.exists():
            logger.error(f"Not found: {tsv}"); sys.exit(1)
        create_spread_box_plot(pd.read_csv(tsv, sep='\t').to_dict('records'),
                               metrics_dir / "spread_metrics_box_plot.png", args.save_svg)
        return

    # Replot mode
    if args.replot_only:
        _run_replot(args, metrics_dir)
        return

    # Validate required args
    if not args.skip_metrics:
        if not args.real_embedding:
            logger.error("--real-embedding required unless --skip-metrics"); sys.exit(1)
        if not args.cell_lines:
            logger.error("--cell-lines required unless --skip-metrics"); sys.exit(1)

    # Load embeddings
    real_embeddings, generated_embeddings = {}, {}
    if not args.skip_metrics or not args.skip_visual:
        real_adata_raw = load_unnormalized_real_embeddings(Path(args.real_embedding))
        for cl in args.cell_lines:
            try:
                real_embeddings[f"Real_{cl}"] = filter_and_aggregate_real_embeddings_by_cell_line(
                    real_adata_raw, cl, args.hpa_csv)
            except Exception as e:
                logger.error(f"Failed real for {cl}: {e}")
        gen_paths = {cl: base_dir / cl / "aggregated" / args.gen_embedding_filename
                     for cl in args.cell_lines}
        gen_paths = {k: v for k, v in gen_paths.items() if v.exists()}
        generated_embeddings = load_multiple_embeddings(gen_paths)
        all_embeddings = {**generated_embeddings, **real_embeddings}

    # Metrics
    if not args.skip_metrics:
        shared = {cl.replace('Real_', ''): set(a.obs[args.gene_col].astype(str).str.strip())
                  for cl, a in real_embeddings.items()}
        _run_metrics(args, all_embeddings, real_embeddings, generated_embeddings, shared, metrics_dir)

    # Visual
    if not args.skip_visual:
        visual_dir.mkdir(parents=True, exist_ok=True)
        create_umap_plots(real_embeddings, generated_embeddings,
                          visual_dir / "umap_embeddings.png", args.save_svg)

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
