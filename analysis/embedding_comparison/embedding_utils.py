"""Shared utilities for loading, filtering, aligning, and aggregating protein embeddings.

Author: Konstantin Kahnert
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

def save_figure(output_path: Path, save_svg: bool = False, dpi: int = 300,
                tight_bbox: bool = True, **kwargs):
    """Save current figure as PNG and optionally SVG."""
    bbox = "tight" if tight_bbox else None
    plt.savefig(output_path, dpi=dpi, bbox_inches=bbox, **kwargs)
    if save_svg:
        plt.savefig(Path(output_path).with_suffix('.svg'), bbox_inches=bbox, **kwargs)


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def _to_dense(X):
    """Convert sparse matrix to dense numpy array if needed."""
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _l2norm_rows(M):
    """L2-normalize each row of M (for cosine similarity via dot product)."""
    d = np.linalg.norm(M, axis=1, keepdims=True)
    d[d == 0] = 1.0
    return M / d


def zscore_fit_transform(X_real: np.ndarray, X_gen: np.ndarray) -> tuple:
    """Z-score normalize per feature, fitting on X_real only.

    Constant features (std==0) are zeroed out. Returns float64 arrays.
    """
    X_real = np.asarray(X_real, dtype=np.float64)
    X_gen = np.asarray(X_gen, dtype=np.float64)
    mean_ = X_real.mean(axis=0)
    std_ = X_real.std(axis=0)
    X_real_z = (X_real - mean_) / (std_ + 1e-5)
    X_gen_z = (X_gen - mean_) / (std_ + 1e-5)
    return X_real_z, X_gen_z


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _filter_and_align(adata_A, adata_B, gene_col="gene_name", drop_dups=True):
    """Align two AnnData objects to the intersection of gene_col.

    Returns (X1, X2, gene_order) as aligned dense matrices.
    """
    gA = adata_A.obs[gene_col].astype(str).str.strip()
    gB = adata_B.obs[gene_col].astype(str).str.strip()
    common = np.intersect1d(gA.values, gB.values)
    if common.size == 0:
        raise ValueError("No overlapping genes between the two AnnData objects.")

    A = adata_A[gA.isin(common)].copy()
    B = adata_B[gB.isin(common)].copy()

    if drop_dups:
        A = A[~A.obs[gene_col].astype(str).str.strip().duplicated(keep="first")].copy()
        B = B[~B.obs[gene_col].astype(str).str.strip().duplicated(keep="first")].copy()

    a_order = A.obs[gene_col].astype(str).str.strip().to_numpy()
    b_lookup = pd.Series(np.arange(B.n_obs), index=B.obs[gene_col].astype(str).str.strip())

    keep = np.array([g in b_lookup.index for g in a_order])
    if not np.all(keep):
        A = A[keep].copy()
        a_order = a_order[keep]
    B = B[b_lookup.loc[a_order].values].copy()

    return _to_dense(A.X), _to_dense(B.X), a_order


def align_and_subsample_by_gene(adata1, adata2, gene_col="gene_name", seed=42):
    """Align two AnnData objects by gene and subsample to equal counts per gene.

    Returns (X1_aligned, X2_aligned, gene_names).
    """
    logger.info("Aligning and subsampling by gene...")
    genes1 = adata1.obs[gene_col].astype(str).str.strip()
    genes2 = adata2.obs[gene_col].astype(str).str.strip()
    common_genes = np.intersect1d(genes1.unique(), genes2.unique())
    if len(common_genes) == 0:
        raise ValueError(f"No overlapping genes in {gene_col} column!")
    logger.info(f"Found {len(common_genes)} common genes")

    rng = np.random.default_rng(seed)
    X1_parts, X2_parts, gene_parts = [], [], []

    for gene in common_genes:
        X1_gene = _to_dense(adata1[genes1 == gene].X)
        X2_gene = _to_dense(adata2[genes2 == gene].X)
        n_samples = min(X1_gene.shape[0], X2_gene.shape[0])
        if n_samples == 0:
            continue
        if X1_gene.shape[0] > n_samples:
            X1_gene = X1_gene[rng.choice(X1_gene.shape[0], n_samples, replace=False)]
        if X2_gene.shape[0] > n_samples:
            X2_gene = X2_gene[rng.choice(X2_gene.shape[0], n_samples, replace=False)]
        X1_parts.append(X1_gene)
        X2_parts.append(X2_gene)
        gene_parts.extend([gene] * n_samples)

    X1_aligned = np.vstack(X1_parts)
    X2_aligned = np.vstack(X2_parts)
    logger.info(f"Aligned: {X1_aligned.shape[0]:,} images x {X1_aligned.shape[1]} features")
    return X1_aligned, X2_aligned, np.array(gene_parts)


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_embedding_file(file_path: Path, name: str = None) -> Tuple:
    """Load an h5ad embedding file. Returns (adata, name)."""
    import scanpy as sc
    adata = sc.read_h5ad(file_path)
    embedding_name = name if name else file_path.stem
    logger.info(f"Loaded {embedding_name}: {adata.n_obs} genes, {adata.n_vars} features")
    return adata, embedding_name


def load_multiple_embeddings(embedding_paths: Dict[str, Path]) -> Dict:
    """Load multiple h5ad embedding files. Returns {name: adata}."""
    logger.info("Loading multiple embeddings...")
    embeddings = {}
    for name, path in embedding_paths.items():
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue
        try:
            adata, loaded_name = load_embedding_file(path, name)
            embeddings[loaded_name] = adata
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
    logger.info(f"Loaded {len(embeddings)} embeddings")
    return embeddings


def load_unnormalized_real_embeddings(
    pth_path: Path,
    common_genes: Optional[Set[str]] = None,
    backed_mode: bool = False,
):
    """Load real embeddings from .pth or .h5ad file. Returns AnnData (float32)."""
    import scanpy as sc
    pth_path = Path(pth_path)

    if pth_path.suffix == ".h5ad":
        adata = sc.read_h5ad(pth_path)
        adata.X = adata.X.astype(np.float32)
        logger.info(f"Loaded real embeddings (h5ad): {adata.n_obs:,} images, {adata.n_vars} features")
        return adata

    obs, embeddings = torch.load(pth_path, map_location="cpu")
    embeddings = embeddings.cpu().numpy()
    adata = sc.AnnData(embeddings, obs=obs)

    if backed_mode:
        logger.info(f"Loaded real embeddings metadata: {adata.n_obs} images")
        return adata

    if common_genes and 'gene_names' in adata.obs.columns:
        gene_mask = adata.obs['gene_names'].isin(common_genes)
        before = adata.n_obs
        adata = adata[gene_mask].copy()
        logger.info(f"Filtered to {adata.n_obs:,} / {before:,} images for common genes")

    adata.X = adata.X.astype(np.float32)
    logger.info(f"Loaded real embeddings: {adata.n_obs:,} images, {adata.n_vars} features")

    if "atlas_name" in adata.obs.columns:
        logger.info(f"Cell lines: {list(adata.obs['atlas_name'].unique()[:10])}...")

    return adata


# ---------------------------------------------------------------------------
# Shared filtering helpers
# ---------------------------------------------------------------------------

def _match_cell_line(adata, cell_line: str, cell_line_col: str = "atlas_name"):
    """Subset adata to a cell line, with fuzzy matching fallback."""
    subset = adata[adata.obs[cell_line_col] == cell_line].copy()
    if subset.n_obs > 0:
        return subset
    available = adata.obs[cell_line_col].unique()
    matches = [n for n in available if cell_line in str(n)]
    if len(matches) == 1:
        logger.info(f"Fuzzy matched '{cell_line}' to '{matches[0]}'")
        return adata[adata.obs[cell_line_col] == matches[0]].copy()
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for '{cell_line}': {matches}")
    raise ValueError(f"No cells found for cell line: {cell_line}. Available: {list(available)}")


def _apply_quality_filters(adata, gene_col="gene_names"):
    """Apply standard quality filters: remove NaN locations, multi-gene antibodies, uncertain reliability."""
    n_before = adata.n_obs
    if "locations" in adata.obs.columns:
        adata = adata[~adata.obs["locations"].isna()]
    if gene_col in adata.obs.columns:
        adata = adata[~adata.obs[gene_col].str.contains(",")]
    if "gene_reliability" in adata.obs.columns:
        adata = adata[
            adata.obs["gene_reliability"].notna()
            & (adata.obs["gene_reliability"] != "Uncertain")
        ]
    logger.info(f"Quality filters: {n_before:,} -> {adata.n_obs:,} images")
    return adata


def _join_hpa_metadata(adata, hpa_csv_path: str, gene_col="gene_names"):
    """Join HPA metadata (locations, reliability, antibody) onto adata.obs."""
    if not hpa_csv_path or not Path(hpa_csv_path).exists():
        adata.obs.setdefault("locations", np.nan)
        adata.obs.setdefault("gene_reliability", "Supported")
        return adata

    df_hpa = pd.read_csv(hpa_csv_path)
    reliability_order = {"Enhanced": 0, "Supported": 1, "Approved": 2, "Uncertain": 3}

    if "if_plate_id" in adata.obs.columns and "if_plate_id" in df_hpa.columns:
        # Basename-based join for pth-format embeddings
        for df in [df_hpa, adata.obs]:
            df["basename"] = (
                df["if_plate_id"].astype(str) + "_"
                + df["position"].astype(str) + "_"
                + df["sample"].astype(str)
            )
        adata.obs["gene_reliability"] = adata.obs["basename"].map(
            df_hpa.set_index("basename")["Gene reliability (in release)"]
        )
        adata.obs["reliability_rank"] = adata.obs["gene_reliability"].map(reliability_order)
    else:
        # Gene-name-based join for h5ad-format embeddings
        df_hpa["_rank"] = df_hpa["Gene reliability (in release)"].map(reliability_order)
        df_best = (
            df_hpa.sort_values("_rank")
            .drop_duplicates(subset=["gene_names"], keep="first")
            .set_index("gene_names")
        )
        for col in ["locations", "ensembl_ids", "antibody"]:
            if col in df_best.columns:
                adata.obs[col] = adata.obs[gene_col].map(df_best[col])
        adata.obs["gene_reliability"] = adata.obs[gene_col].map(
            df_best["Gene reliability (in release)"]
        )
        adata.obs["reliability_rank"] = adata.obs["gene_reliability"].map(reliability_order)

    return adata


# ---------------------------------------------------------------------------
# Cell-line filtering and aggregation
# ---------------------------------------------------------------------------

def filter_and_aggregate_real_embeddings_by_cell_line(
    adata, cell_line: str, hpa_csv_path: str = None,
):
    """Filter real embeddings for a cell line and aggregate by gene.

    Applies quality filters (reliability, location annotation, multi-gene),
    selects best antibody per gene, and averages embeddings per gene.
    """
    import scanpy as sc

    # Normalize schema to atlas_name/gene_names
    if "atlas_name" not in adata.obs.columns:
        adata = adata.copy()
        adata.obs["atlas_name"] = adata.obs.get("cell_line", adata.obs.index)
        adata.obs["gene_names"] = adata.obs.get("gene_name", adata.obs.index)
        adata = _join_hpa_metadata(adata, hpa_csv_path, gene_col="gene_names")

    adata_cl = _match_cell_line(adata, cell_line, "atlas_name")
    logger.info(f"Found {adata_cl.n_obs} cells for {cell_line}")

    # Join reliability if using pth-format data
    if "if_plate_id" in adata_cl.obs.columns and hpa_csv_path:
        adata_cl = _join_hpa_metadata(adata_cl, hpa_csv_path, gene_col="gene_names")

    adata_cl = _apply_quality_filters(adata_cl, gene_col="gene_names")

    # Select best antibody per gene (by reliability rank)
    best = adata_cl.obs.sort_values(
        by=["reliability_rank", "ensembl_ids", "antibody"], kind="stable"
    ).drop_duplicates(subset=["atlas_name", "ensembl_ids"], keep="first")
    adata_cl = adata_cl[adata_cl.obs["antibody"].isin(best["antibody"])]

    # Aggregate by gene
    adata_cl = adata_cl.copy()
    adata_cl.obs["atlas_name_ensembl_ids"] = (
        adata_cl.obs["atlas_name"].astype(str) + "_" + adata_cl.obs["ensembl_ids"].astype(str)
    ).values

    df = pd.DataFrame(adata_cl.X, columns=adata_cl.var_names)
    df["atlas_name_ensembl_ids"] = adata_cl.obs["atlas_name_ensembl_ids"].values
    df_grouped = df.groupby("atlas_name_ensembl_ids").mean()

    def _agg_locations(locations):
        unique = set()
        for loc in locations:
            unique.update(loc.split(","))
        return ",".join(sorted(unique))

    locations_agg = (
        adata_cl.obs.groupby("atlas_name_ensembl_ids")
        .agg({"locations": _agg_locations}).reset_index()
    )
    df_unique = (
        adata_cl.obs.drop_duplicates(subset="atlas_name_ensembl_ids", keep="first")
        .set_index("atlas_name_ensembl_ids")
        .drop(columns=["locations"])
        .merge(locations_agg, on="atlas_name_ensembl_ids", how="left")
    )

    adata_agg = sc.AnnData(
        X=df_grouped.values,
        var=pd.DataFrame(index=adata_cl.var_names),
        obs=pd.DataFrame(df_grouped.index, columns=["atlas_name_ensembl_ids"]),
    )
    adata_agg.obs = adata_agg.obs.merge(df_unique, on="atlas_name_ensembl_ids", how="left")
    adata_agg.obs["gene_name"] = adata_agg.obs["gene_names"]

    logger.info(f"Processed {cell_line}: {adata_agg.n_obs} genes after aggregation")
    return adata_agg


def filter_real_embeddings_by_cell_line_imagelevel(
    adata, cell_line: str, hpa_csv_path: str = None, backed_mode: bool = False,
):
    """Filter real embeddings for a cell line WITHOUT aggregation (image-level)."""
    adata_cl = _match_cell_line(adata, cell_line, "atlas_name")
    logger.info(f"Found {adata_cl.n_obs} images for {cell_line}")

    if backed_mode:
        adata_cl.obs["gene_name"] = adata_cl.obs["gene_names"]
        return adata_cl

    if hpa_csv_path:
        adata_cl = _join_hpa_metadata(adata_cl, hpa_csv_path, gene_col="gene_names")
        if "gene_reliability" in adata_cl.obs.columns:
            adata_cl = adata_cl[adata_cl.obs["gene_reliability"] != "Uncertain"]

    if "locations" in adata_cl.obs.columns:
        adata_cl = adata_cl[~adata_cl.obs["locations"].isna()]
    if "gene_names" in adata_cl.obs.columns:
        adata_cl = adata_cl[~adata_cl.obs["gene_names"].str.contains(",")]

    if "gene_name" not in adata_cl.obs.columns and "gene_names" in adata_cl.obs.columns:
        adata_cl.obs["gene_name"] = adata_cl.obs["gene_names"]

    logger.info(f"Processed {cell_line} (image-level): {adata_cl.n_obs} images")
    if adata_cl.n_obs < 100:
        logger.warning(f"Only {adata_cl.n_obs} images — may be too few for robust analysis")

    return adata_cl


def aggregate_embeddings(adata, gene_col="gene_name", cell_line_col="cell_line",
                         cell_lines=None, embedding_type="aggregated"):
    """Aggregate image-level embeddings to gene-level by averaging.

    Works for both real (gene_names/atlas_name) and generated (gene_name/cell_line)
    schemas — just pass the appropriate column names.
    """
    import scanpy as sc

    if cell_lines:
        adata = adata[adata.obs[cell_line_col].isin(cell_lines)].copy()
        logger.info(f"Filtered to {len(cell_lines)} cell lines: {adata.n_obs:,} images")

    if gene_col not in adata.obs.columns:
        raise ValueError(f"'{gene_col}' not found. Available: {adata.obs.columns.tolist()}")

    logger.info(f"Aggregating {adata.n_obs:,} images by {gene_col}...")
    df = pd.DataFrame(adata.X, columns=[f"feat_{i}" for i in range(adata.n_vars)])
    df[gene_col] = adata.obs[gene_col].values
    df_grouped = df.groupby(gene_col, observed=True).mean(numeric_only=True)

    n_images = adata.obs.groupby(gene_col, observed=True).size()

    agg_adata = sc.AnnData(
        X=df_grouped.values,
        obs=pd.DataFrame({gene_col: df_grouped.index}).reset_index(drop=True),
    )
    agg_adata.obs["n_images"] = agg_adata.obs[gene_col].map(n_images).values
    agg_adata.obs["gene_name"] = agg_adata.obs[gene_col]
    agg_adata.uns["embedding_type"] = embedding_type
    agg_adata.uns["aggregation_method"] = "mean"
    agg_adata.uns["creation_date"] = datetime.now().isoformat()

    logger.info(f"Aggregated to {agg_adata.shape[0]:,} genes")
    return agg_adata


# ---------------------------------------------------------------------------
# Subcellular location mapping (HPA annotations -> grouped categories)
# ---------------------------------------------------------------------------

LOCATION_MAPPING = {
    "Actin filaments": "Cytoskeleton",
    "Aggresome": "Cytosol",
    "Centriolar satellite": "Cytoskeleton",
    "Centrosome": "Cytoskeleton",
    "Cytokinetic bridge": "Cytokinetic bridge",
    "Cytoplasmic bodies": "Cytosol",
    "Cytosol": "Cytosol",
    "Focal adhesion sites": "Cytoskeleton",
    "Intermediate filaments": "Cytoskeleton",
    "Microtubules": "Cytoskeleton",
    "Microtubule ends": "Cytoskeleton",
    "Midbody": "Cytokinetic bridge",
    "Midbody ring": "Cytokinetic bridge",
    "Mitochondria": "Mitochondria",
    "Mitotic spindle": "Mitotic structures",
    "Rods & Rings": "Mitotic structures",
    "Nuclear bodies": "Nucleus",
    "Nuclear membrane": "Nuclear membrane",
    "Nuclear speckles": "Nuclear speckles",
    "Nucleoli": "Nucleoli",
    "Nucleoli fibrillar center": "Nucleoli",
    "Nucleoli rim": "Nucleoli",
    "Nucleoplasm": "Nucleoplasm",
    "Cell Junctions": "Plasma membrane",
    "Endoplasmic reticulum": "Endomembrane system",
    "Endosomes": "Vesicles",
    "Golgi apparatus": "Endomembrane system",
    "Lipid droplets": "Vesicles",
    "Lysosomes": "Vesicles",
    "Peroxisomes": "Vesicles",
    "Plasma membrane": "Plasma membrane",
    "Vesicles": "Endomembrane system",
}


def assign_primary_location(locations_str):
    """Assign a single primary location category from comma-separated HPA locations."""
    if pd.isna(locations_str):
        return "Unknown"
    locs = [loc.strip() for loc in str(locations_str).split(",")]
    if len(locs) == 1:
        return LOCATION_MAPPING.get(locs[0], "Unknown")
    return "Multi-localizing"
