"""Preprocessing utilities for assembling and normalizing multi-channel microscopy images.

Expected channel order for the ProtiCelli package [H, W, 4]:
  Channel 0 = microtubules (MT)
  Channel 1 = protein of interest (target / empty placeholder)
  Channel 2 = nucleus
  Channel 3 = ER

Classes follow a scikit-learn-style API:
  - Parameters are set in ``__init__``.
  - ``fit(X)`` learns statistics from data and returns ``self``.
  - ``transform(X)`` applies the transformation.
  - ``fit_transform(X)`` is a convenience shortcut.
  - Fitted attributes are suffixed with ``_``.
"""

from __future__ import annotations

import numpy as np
from tifffile import imread, imwrite


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _load_channel(src: str | np.ndarray) -> np.ndarray:
    """Load a single-channel image from a file path or array.

    Squeezes (1, H, W) and (H, W, 1) shapes to (H, W).
    """
    if isinstance(src, (str, bytes)):
        img = imread(src)
    else:
        img = np.asarray(src)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim != 2:
        raise ValueError(
            f"Expected a single-channel 2-D image, got shape {img.shape}"
        )
    return img


# ---------------------------------------------------------------------------
# ChannelAssembler
# ---------------------------------------------------------------------------

class ChannelAssembler:
    """Assemble separate single-channel images into a [H, W, 4] stack.

    Channel order matches the ProtiCelli convention:
      0 = microtubules, 1 = protein (or zeros), 2 = nucleus, 3 = ER.

    ``fit`` is a no-op (nothing to learn); the class exists for API
    consistency and to keep channel-order configuration in one place.

    Parameters
    ----------
    has_protein : bool
        If ``True``, ``transform`` expects a ``"protein"`` key in the input
        dict. If ``False``, channel 1 is filled with zeros (inference mode).
        Default ``True``.

    Examples
    --------
    >>> assembler = ChannelAssembler(has_protein=False)
    >>> stack = assembler.fit_transform({
    ...     "microtubules": "mt.tif",
    ...     "nucleus": "nucleus.tif",
    ...     "er": "er.tif",
    ... })
    >>> stack.shape   # (H, W, 4)
    """

    def __init__(self, has_protein: bool = True):
        self.has_protein = has_protein

    def fit(self, X=None, _y=None):
        """No-op; returns self for API compatibility."""
        return self

    def transform(self, X: dict) -> np.ndarray:
        """Assemble channels from a dict of paths or arrays.

        Parameters
        ----------
        X : dict
            Must contain keys ``"microtubules"``, ``"nucleus"``, ``"er"``.
            If ``has_protein=True``, must also contain ``"protein"``.

        Returns
        -------
        np.ndarray
            Shape [H, W, 4], dtype matching the input channels.
        """
        mt = _load_channel(X["microtubules"])
        nu = _load_channel(X["nucleus"])
        er = _load_channel(X["er"])

        h, w = mt.shape
        for name, ch in [("nucleus", nu), ("er", er)]:
            if ch.shape != (h, w):
                raise ValueError(
                    f"Channel '{name}' has shape {ch.shape}, expected ({h}, {w})"
                )

        if self.has_protein:
            prot = _load_channel(X["protein"])
            if prot.shape != (h, w):
                raise ValueError(
                    f"Channel 'protein' has shape {prot.shape}, expected ({h}, {w})"
                )
        else:
            prot = np.zeros((h, w), dtype=mt.dtype)

        return np.stack([mt, prot, nu, er], axis=-1)  # [H, W, 4]

    def fit_transform(self, X: dict, y=None) -> np.ndarray:
        """Fit (no-op) and transform in one step."""
        return self.fit(X, y).transform(X)


# ---------------------------------------------------------------------------
# ImageNormalizer
# ---------------------------------------------------------------------------

class ImageNormalizer:
    """Normalize a [H, W, C] image to the range [-1, 1].

    Each image is normalized independently using its own pixel statistics.

    Normalization strategy (per image)
    -----------------------------------
    1. Compute a clip threshold from the **MT channel** (channel 0) at
       ``percentile`` (default 99.95), capped at the bit-depth maximum.
       By default this value is applied to all channels; set
       ``clip_channel=None`` to clip each channel at its own percentile.
    2. **Global mode** — divide all channels by the clipped MT-channel max
       to preserve relative intensities across channels.
    3. **Per-channel fallback** — if any channel's clipped max is less than
       ``scale_threshold × MT_max``, each channel is normalized by its own
       max instead.
    4. Rescale [0, 1] → [-1, 1].

    ``fit`` is a no-op kept for API consistency. All statistics are computed
    on-the-fly per image inside ``transform``.

    Parameters
    ----------
    bit_depth : {8, 16}
        Bit depth of the input images. Caps the clip threshold at 255 or
        65535. Default ``8``.
    percentile : float
        Percentile of the reference channel used to compute the clip
        threshold. Default ``99.95``.
    clip_channel : int or None
        Channel whose percentile sets the clip for all channels. Default
        ``0`` (MT channel). Set to ``None`` to clip each channel
        independently.
    scale_threshold : float
        Fraction of MT max below which per-channel normalization replaces
        global normalization. Default ``0.1`` (10 %).

    Examples
    --------
    >>> normalizer = ImageNormalizer(bit_depth=16)
    >>> norm = normalizer.transform(stack)            # single image [H, W, 4]
    >>> norms = normalizer.transform(batch)           # batch [N, H, W, 4]
    >>> norm = normalizer.transform(stack, save_path="cell_norm.tif")
    """

    _BIT_DEPTH_MAX = {8: 255.0, 16: 65535.0}

    def __init__(
        self,
        bit_depth: int = 8,
        percentile: float = 99.95,
        clip_channel: int | None = 0,
        scale_threshold: float = 0.1,
    ):
        if bit_depth not in self._BIT_DEPTH_MAX:
            raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}")
        self.bit_depth = bit_depth
        self.percentile = percentile
        self.clip_channel = clip_channel
        self.scale_threshold = scale_threshold

    def fit(self, X=None, y=None) -> ImageNormalizer:
        """No-op; returns self for API consistency."""
        return self

    def _normalize_one(self, img: np.ndarray) -> np.ndarray:
        """Normalize a single [H, W, C] float32 image in-place and return it."""
        max_val = self._BIT_DEPTH_MAX[self.bit_depth]
        n_channels = img.shape[2]

        # Step 1: compute clip threshold
        if self.clip_channel is not None:
            ref_clip = min(
                float(np.percentile(img[..., self.clip_channel], self.percentile)),
                max_val,
            )
            clip_values = np.full(n_channels, ref_clip, dtype=np.float32)
        else:
            clip_values = np.array([
                min(float(np.percentile(img[..., c], self.percentile)), max_val)
                for c in range(n_channels)
            ], dtype=np.float32)

        # Step 2: clip and find per-channel maxes
        for c in range(n_channels):
            img[..., c] = np.clip(img[..., c], 0, clip_values[c])
        channel_maxes = np.array([img[..., c].max() for c in range(n_channels)], dtype=np.float32)

        # Step 3: choose global vs per-channel scale
        mt_max = channel_maxes[0]
        use_global = mt_max > 0 and all(
            channel_maxes[c] >= self.scale_threshold * mt_max
            for c in range(n_channels)
        )
        scale = (
            np.full(n_channels, mt_max, dtype=np.float32) if use_global
            else np.where(channel_maxes > 0, channel_maxes, 1.0).astype(np.float32)
        )

        # Step 4: normalize and rescale to [-1, 1]
        for c in range(n_channels):
            img[..., c] /= scale[c]
        img = img * 2.0 - 1.0
        return img

    def transform(self, X: np.ndarray, save_path: str | None = None) -> np.ndarray:
        """Normalize each image independently.

        Parameters
        ----------
        X : np.ndarray
            Single image [H, W, C] or batch [N, H, W, C].
        save_path : str, optional
            Save the normalized result as a float32 TIFF. For batches,
            one file per image is written as ``{stem}_{i}.tif``.

        Returns
        -------
        np.ndarray
            Float32 array of the same shape, values in [-1, 1].
        """
        single = X.ndim == 3
        X = np.asarray(X, dtype=np.float32).copy()
        if single:
            X = X[np.newaxis]

        for i in range(len(X)):
            X[i] = self._normalize_one(X[i])

        if save_path is not None:
            from pathlib import Path
            if single:
                imwrite(save_path, X[0])
            else:
                stem = Path(save_path).stem
                parent = Path(save_path).parent
                suffix = Path(save_path).suffix or ".tif"
                for i, img in enumerate(X):
                    imwrite(parent / f"{stem}_{i}{suffix}", img)

        return X[0] if single else X

    def fit_transform(self, X: np.ndarray, y=None, save_path: str | None = None) -> np.ndarray:
        """Fit (no-op) and transform in one step."""
        return self.transform(X, save_path=save_path)
