"""Ensemble uncertainty scoring used by ``Model.predict_with_reliability``."""

import numpy as np


def compute_medoid_and_reliability(ensemble, cell_mask=None, eps=1e-8):
    """Pick the ensemble medoid and a scale-free reliability score.

    The point prediction is NOT an average. It is the medoid: the single real
    member that agrees most with the rest, so sharp/punctate morphology (e.g.
    vesicles) is preserved. Reliability is the ensemble's mean pairwise Pearson
    correlation (restricted to ``cell_mask`` if given), clipped to [0, 1] --
    scale- and offset-invariant, so it's comparable across proteins of
    different expression levels.

    Args:
        ensemble:  np.ndarray [N, H, W], per-member predicted images (same
                   conditioning, independent initial noise).
        cell_mask: optional bool [H, W] cell footprint; agreement is computed
                   inside it so background pixels don't inflate the score.

    Returns:
        (medoid_idx, reliability_score): medoid_idx indexes the member to use
        as the prediction; reliability_score is NaN when it can't be computed
        (N == 1, or fewer than 2 members with nonzero variance in the mask).
    """
    raw = np.asarray(ensemble)
    if raw.ndim != 3 or min(raw.shape) < 1 or raw.dtype.kind not in "buif":
        raise ValueError("ensemble must be a nonempty real [N, H, W] array")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    N = raw.shape[0]
    if cell_mask is not None and np.asarray(cell_mask).shape != raw.shape[1:]:
        raise ValueError("cell_mask must match the image spatial dimensions")
    mask = np.ones(raw.shape[1:], dtype=bool) if cell_mask is None else np.asarray(cell_mask, dtype=bool)
    if mask.sum() < 2:
        mask = np.ones(raw.shape[1:], dtype=bool)
    if N == 1 or mask.sum() < 2:
        return 0, float("nan")
    # Stable block covariance avoids float32 dot-product error on 512px images
    # without allocating several complete float64 copies of the ensemble.
    flat, selected = raw.reshape(N, -1), mask.reshape(-1)
    mean = np.zeros(N, dtype=np.float64)
    cross = np.zeros((N, N), dtype=np.float64)
    count = 0
    for start in range(0, flat.shape[1], 32768):
        block_mask = selected[start:start + 32768]
        if not block_mask.any():
            continue
        block = flat[:, start:start + 32768][:, block_mask].astype(np.float64)
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        block_count = block.shape[1]
        block_mean = block.mean(axis=1)
        block -= block_mean[:, None]
        difference = block_mean - mean
        total = count + block_count
        cross += block @ block.T + np.outer(difference, difference) * (count * block_count / total)
        mean += difference * (block_count / total)
        count = total
    variance = np.maximum(np.diag(cross) / count, 0)
    # Members that are flat inside the mask (zero variance) have undefined
    # correlation; exclude them so they neither poison the matrix nor win.
    valid = np.flatnonzero(np.sqrt(variance) > eps)
    if valid.size < 2:
        return (int(valid[0]) if valid.size == 1 else 0), float("nan")

    cross = cross[np.ix_(valid, valid)]
    norm = np.sqrt(np.diag(cross))
    corr = np.clip(cross / norm[:, None] / norm[None, :], -1, 1)
    np.fill_diagonal(corr, np.nan)
    centrality = np.nanmean(corr, axis=1)
    local = int(np.nanargmax(centrality))
    medoid_idx = int(valid[local])
    reliability_score = float(np.clip(np.nanmean(corr), 0.0, 1.0))
    return medoid_idx, reliability_score
