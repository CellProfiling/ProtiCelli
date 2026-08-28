"""Ensemble uncertainty scoring used by ``Model.predict_with_uncertainty``."""

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
    N = ensemble.shape[0]
    raw = ensemble.astype(np.float32, copy=False)
    if not np.isfinite(raw).all():
        # A single inf/nan member would otherwise propagate through the dot
        # product and make the whole correlation matrix NaN.
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    if cell_mask is None or np.asarray(cell_mask).sum() < 2:
        mask = np.ones(raw.shape[1:], dtype=bool)
    else:
        mask = np.asarray(cell_mask, dtype=bool)

    if N == 1 or mask.sum() < 2:
        return 0, float("nan")

    X = raw.reshape(N, -1)[:, mask.reshape(-1)]           # [N, P]
    Xc = X - X.mean(axis=1, keepdims=True)
    denom = np.sqrt((Xc**2).mean(axis=1, keepdims=True))  # [N, 1]
    # Members that are flat inside the mask (zero variance) have undefined
    # correlation; exclude them so they neither poison the matrix nor win.
    valid = np.where(denom[:, 0] > eps)[0]
    if valid.size < 2:
        return (int(valid[0]) if valid.size == 1 else 0), float("nan")

    Xn = Xc[valid] / (denom[valid] + eps)
    corr = (Xn @ Xn.T) / Xn.shape[1]                       # [V, V]
    np.fill_diagonal(corr, np.nan)
    centrality = np.nanmean(corr, axis=1)
    local = int(np.nanargmax(centrality))
    medoid_idx = int(valid[local])
    reliability_score = float(np.clip(np.nanmean(corr), 0.0, 1.0))
    return medoid_idx, reliability_score
