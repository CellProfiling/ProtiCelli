"""Preprocessing utilities for assembling and normalizing multi-channel microscopy images.

Expected channel order for the ProtVS package [H, W, 4]:
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
from tifffile import imread


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

    Channel order matches the ProtVS convention:
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

    Normalization strategy
    ----------------------
    1. Compute a clip threshold from the **MT channel** (channel 0) at
       ``percentile``, capped at the bit-depth maximum. By default this
       single value is applied to all channels; set ``clip_channel=None``
       to clip each channel independently.
    2. **Global mode** — divide all channels by the clipped MT-channel max
       to preserve relative intensities.
    3. **Per-channel fallback** — if any channel's clipped max is less than
       ``scale_threshold × MT_max``, each channel is normalized by its own
       max instead.
    4. Rescale [0, 1] → [-1, 1].

    ``fit`` computes the clip values and scale factor from one or more images
    and stores them as fitted attributes (``clip_values_``, ``scale_``,
    ``use_global_``). ``transform`` applies those stored values, so the same
    statistics can be reused across a dataset.

    Parameters
    ----------
    bit_depth : {8, 16}
        Bit depth of the input images. Sets the absolute maximum pixel value
        used to cap the clip threshold (255 for 8-bit, 65535 for 16-bit).
        Default ``8``.
    percentile : float
        Percentile of the MT channel used to compute the clip threshold.
        Default ``99.95``.
    clip_channel : int or None
        Channel whose percentile is used as the clip threshold for **all**
        channels. Default ``0`` (MT channel). Set to ``None`` to clip each
        channel at its own percentile.
    scale_threshold : float
        Fraction of MT max below which per-channel normalization is used.
        Default ``0.1`` (10 %).

    Attributes
    ----------
    clip_values_ : np.ndarray, shape (C,)
        Per-channel clip threshold learned during ``fit``.
    scale_ : np.ndarray, shape (C,)
        Per-channel divisor learned during ``fit``.
    use_global_ : bool
        Whether global (MT-referenced) or per-channel normalization was chosen.
    n_channels_ : int
        Number of channels seen during ``fit``.

    Examples
    --------
    Normalize a single 16-bit image:

    >>> normalizer = ImageNormalizer(bit_depth=16)
    >>> norm = normalizer.fit_transform(stack)   # stack: [H, W, 4]

    Fit on training images, apply to test images with the same statistics:

    >>> normalizer = ImageNormalizer(bit_depth=16)
    >>> normalizer.fit(train_stack)
    >>> norm_train = normalizer.transform(train_stack)
    >>> norm_test  = normalizer.transform(test_stack)
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

    def fit(self, X: np.ndarray, y=None) -> ImageNormalizer:
        """Learn clip values and scale from ``X``.

        Parameters
        ----------
        X : np.ndarray
            A single image [H, W, C] or a batch [N, H, W, C].

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 3:
            X = X[np.newaxis]   # → [1, H, W, C]
        if X.ndim != 4:
            raise ValueError(f"Expected [H,W,C] or [N,H,W,C], got shape {X.shape}")

        n_channels = X.shape[-1]
        self.n_channels_ = n_channels

        max_val = self._BIT_DEPTH_MAX[self.bit_depth]

        if self.clip_channel is not None:
            # Single clip threshold derived from one reference channel
            ref_clip = min(
                float(np.percentile(X[..., self.clip_channel], self.percentile)),
                max_val,
            )
            self.clip_values_ = np.full(n_channels, ref_clip, dtype=np.float32)
        else:
            # Independent clip per channel
            self.clip_values_ = np.array([
                min(float(np.percentile(X[..., i], self.percentile)), max_val)
                for i in range(n_channels)
            ], dtype=np.float32)

        # Channel maxes after clipping (used to decide global vs per-channel)
        channel_maxes = np.array([
            np.max(np.clip(X[..., i], 0, self.clip_values_[i]))
            for i in range(n_channels)
        ], dtype=np.float32)

        mt_max = channel_maxes[0]
        use_global = mt_max > 0 and all(
            channel_maxes[i] >= self.scale_threshold * mt_max
            for i in range(n_channels)
        )
        self.use_global_ = use_global

        if use_global:
            self.scale_ = np.full(n_channels, mt_max, dtype=np.float32)
        else:
            self.scale_ = np.where(channel_maxes > 0, channel_maxes, 1.0).astype(np.float32)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization using fitted statistics.

        Parameters
        ----------
        X : np.ndarray
            Image [H, W, C] or batch [N, H, W, C].

        Returns
        -------
        np.ndarray
            Float32 array of the same shape, values in [-1, 1].
        """
        if not hasattr(self, "clip_values_"):
            raise RuntimeError("Call fit() before transform().")

        single = X.ndim == 3
        X = np.asarray(X, dtype=np.float32)
        if single:
            X = X[np.newaxis]

        X = X.copy()
        for i in range(self.n_channels_):
            X[..., i] = np.clip(X[..., i], 0, self.clip_values_[i]) / self.scale_[i]

        X = X * 2.0 - 1.0
        return X[0] if single else X

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """Save fitted statistics to a ``.npz`` file."""
        if not hasattr(self, "clip_values_"):
            raise RuntimeError("Nothing to save — call fit() first.")
        np.savez(
            path,
            clip_values_=self.clip_values_,
            scale_=self.scale_,
            use_global_=self.use_global_,
            n_channels_=self.n_channels_,
            bit_depth=self.bit_depth,
            percentile=self.percentile,
            clip_channel=-1 if self.clip_channel is None else self.clip_channel,
            scale_threshold=self.scale_threshold,
        )

    @classmethod
    def load(cls, path: str) -> ImageNormalizer:
        """Load a previously saved normalizer from a ``.npz`` file."""
        data = np.load(path)
        clip_channel = int(data["clip_channel"])
        obj = cls(
            bit_depth=int(data["bit_depth"]),
            percentile=float(data["percentile"]),
            clip_channel=None if clip_channel == -1 else clip_channel,
            scale_threshold=float(data["scale_threshold"]),
        )
        obj.clip_values_ = data["clip_values_"]
        obj.scale_ = data["scale_"]
        obj.use_global_ = bool(data["use_global_"])
        obj.n_channels_ = int(data["n_channels_"])
        return obj
