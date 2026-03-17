"""Preprocessing utilities for assembling and normalizing multi-channel microscopy images.

Expected channel order for the ProtVS package [H, W, 4]:
  Channel 0 = microtubules (MT)
  Channel 1 = protein of interest (target / empty placeholder)
  Channel 2 = nucleus
  Channel 3 = ER
"""

from __future__ import annotations

import numpy as np
from tifffile import imread, imwrite
from typing import Optional


def assemble_image(
    microtubules: str | np.ndarray,
    nucleus: str | np.ndarray,
    er: str | np.ndarray,
    protein: Optional[str | np.ndarray] = None,
) -> np.ndarray:
    """Assemble single-channel images into a [H, W, 4] stack for ProtVS.

    Parameters
    ----------
    microtubules : path or ndarray
        Microtubule channel (channel 0).
    nucleus : path or ndarray
        Nucleus channel (channel 2).
    er : path or ndarray
        ER channel (channel 3).
    protein : path or ndarray, optional
        Protein-of-interest channel (channel 1). If None, a zero-filled
        placeholder is inserted so the stack can be used for inference.

    Returns
    -------
    np.ndarray
        Array of shape [H, W, 4] with dtype matching the input channels.
    """

    def _load(src):
        if isinstance(src, (str, bytes)):
            img = imread(src)
        else:
            img = np.asarray(src)
        # Collapse to 2-D if the file was saved as (1, H, W) or (H, W, 1)
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        elif img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        if img.ndim != 2:
            raise ValueError(
                f"Expected a single-channel 2-D image, got shape {img.shape}"
            )
        return img

    mt = _load(microtubules)
    nu = _load(nucleus)
    er_ = _load(er)

    h, w = mt.shape
    for name, ch in [("nucleus", nu), ("er", er_)]:
        if ch.shape != (h, w):
            raise ValueError(
                f"Channel '{name}' has shape {ch.shape}, expected ({h}, {w})"
            )

    if protein is None:
        prot = np.zeros((h, w), dtype=mt.dtype)
    else:
        prot = _load(protein)
        if prot.shape != (h, w):
            raise ValueError(
                f"Channel 'protein' has shape {prot.shape}, expected ({h}, {w})"
            )

    # Stack in ProtVS order: [MT, protein, nucleus, ER]
    stack = np.stack([mt, prot, nu, er_], axis=-1)  # [H, W, 4]
    return stack


def normalize_image(image: np.ndarray, scale_threshold: float = 0.1) -> np.ndarray:
    """Normalize a [H, W, C] image to the range [-1, 1].

    Strategy
    --------
    1. Clip each channel at its 99.5th percentile to suppress outlier pixels.
    2. Attempt global normalization: divide all channels by the microtubule
       channel maximum (channel 0), which acts as a shared intensity reference.
    3. If any channel's clipped maximum is less than ``scale_threshold`` times
       the MT maximum (i.e., channels are not on a comparable intensity scale),
       fall back to per-channel normalization instead.
    4. Rescale from [0, 1] to [-1, 1].

    Parameters
    ----------
    image : np.ndarray
        Input image of shape [H, W, C], any numeric dtype.
    scale_threshold : float
        Ratio below which a channel is considered to be on a different
        intensity scale from the MT channel, triggering per-channel
        normalization. Default 0.1 (10 %).

    Returns
    -------
    np.ndarray
        Float32 array of shape [H, W, C] with values in [-1, 1].
    """
    image = image.astype(np.float32)
    n_channels = image.shape[2]

    # Step 1: clip each channel at 99.5th percentile
    channel_maxes = np.zeros(n_channels, dtype=np.float32)
    for i in range(n_channels):
        p99 = np.percentile(image[:, :, i], 99.5)
        image[:, :, i] = np.clip(image[:, :, i], 0, p99)
        channel_maxes[i] = np.max(image[:, :, i])

    # Step 2: decide normalization strategy
    mt_max = channel_maxes[0]  # channel 0 = microtubules

    use_global = mt_max > 0
    if use_global:
        # Check whether all channels are on a comparable scale
        for i in range(n_channels):
            if channel_maxes[i] < scale_threshold * mt_max:
                use_global = False
                break

    if use_global:
        # Global normalization by MT max
        image /= mt_max
    else:
        # Per-channel normalization
        for i in range(n_channels):
            if channel_maxes[i] > 0:
                image[:, :, i] /= channel_maxes[i]
            # channels that are all zero remain zero

    # Step 3: rescale [0, 1] → [-1, 1]
    image -= 0.5
    image *= 2.0
    return image


def assemble_and_normalize(
    microtubules: str | np.ndarray,
    nucleus: str | np.ndarray,
    er: str | np.ndarray,
    protein: Optional[str | np.ndarray] = None,
    scale_threshold: float = 0.1,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Convenience wrapper: assemble channels then normalize.

    Parameters
    ----------
    microtubules, nucleus, er, protein :
        See :func:`assemble_image`.
    scale_threshold : float
        See :func:`normalize_image`.
    save_path : str, optional
        If provided, save the normalized stack as a TIFF at this path.

    Returns
    -------
    np.ndarray
        Normalized [H, W, 4] float32 array in [-1, 1].
    """
    stack = assemble_image(microtubules, nucleus, er, protein)
    stack = normalize_image(stack, scale_threshold=scale_threshold)
    if save_path is not None:
        imwrite(save_path, stack)
    return stack
