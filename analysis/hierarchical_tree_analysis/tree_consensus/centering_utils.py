"""Per-cell embedding centering for within-cell comparison analyses.

Author: Konstantin Kahnert
"""

import numpy as np
from typing import Tuple, Optional


def compute_per_cell_centroid(cell_embedding: np.ndarray) -> np.ndarray:
    """Compute centroid (mean) for a single cell's embedding matrix."""
    return np.mean(cell_embedding, axis=0, dtype=np.float32)


def center_cell_embedding(
    cell_embedding: np.ndarray,
    cell_centroid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Center a cell's embeddings by subtracting its own centroid.

    Removes cell-specific morphology signals so protein-protein similarities
    reflect localization differences rather than shared cell effects.
    """
    if cell_centroid is None:
        cell_centroid = compute_per_cell_centroid(cell_embedding)
    return cell_embedding - cell_centroid


def compute_per_cell_cosine_similarities(
    cell_embedding: np.ndarray,
    use_centering: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute pairwise cosine similarities for one cell.

    Returns (standard_sims, centered_sims) where each is an upper-triangle
    vector of length n_proteins*(n_proteins-1)/2. centered_sims is None
    if use_centering=False.
    """
    def _cosine_upper_tri(embedding):
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embedding / norms
        sim_matrix = normed @ normed.T
        return sim_matrix[np.triu_indices(embedding.shape[0], k=1)]

    standard_sims = _cosine_upper_tri(cell_embedding)
    centered_sims = None
    if use_centering:
        centered = center_cell_embedding(cell_embedding)
        centered_sims = _cosine_upper_tri(centered)
    return standard_sims, centered_sims
