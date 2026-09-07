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
from pathlib import Path

import os   

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _load_channel(src: str | np.ndarray) -> np.ndarray:
    """Load a single-channel image from a file path or array.

    Squeezes (1, H, W) and (H, W, 1) shapes to (H, W).
    """
    if isinstance(src, (str, bytes, os.PathLike)):
        from tifffile import imread

        img = imread(src, is_ome=False)  # OME-TIFFs may have extra dimensions we don't want
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
    Channel order: 0 = MT, 1 = protein (POI), 2 = Nucleus, 3 = ER.

    Normalization strategy (per image)
    -----------------------------------
    1. Clip all channels at ``percentile`` of the **Nucleus channel**
       (``ref_channel``, default 2), capped at the bit-depth maximum.
    2. Per-channel ratio ``r_c = max_c / nuc_max``.
    3. Gain

           f_c = r_c                                    if r_c >= r_floor
               = r_floor * (r_c / r_floor) ** dim_gamma  if r_c <  r_floor

       Continuous at ``r_floor``, monotone increasing. For ``r_c >= r_floor``
       this reduces to dividing by ``nuc_max``, i.e. the old global mode,
       bit-identically. Channels whose max is below ``noise_floor`` counts get
       no lift, so empty channels are not amplified into noise.
    4. ``v_c = I_c / max_c * f_c``, then rescale [0, 1] -> [-1, 1].

    ``dim_gamma=1.0`` reverts step 3 entirely and reproduces the previous
    global-mode behaviour for all channels. The old ``scale_threshold``
    all-or-nothing fallback is gone: it flipped every channel in the image to
    self-normalization whenever any single channel crossed the threshold, and
    it was discontinuous in that channel's max.

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
    ref_channel : int
        Channel whose percentile sets the clip for all channels and whose max
        is the scale reference. Default ``2`` (Nucleus). Both uses must agree,
        otherwise normalized values can exceed 1.
    r_floor : float
        Ratio at or above which a channel is left exactly as the old global
        mode. Default ``0.3``.
    dim_gamma : float
        Compression exponent applied below ``r_floor``. Default ``0.35``.
        Set to ``1.0`` to disable.
    noise_floor : float or None
        Raw-count max below which a channel receives no lift. ``None``
        (default) selects 3.0 for 8-bit and 20.0 for 16-bit.

    Examples
    --------
    >>> normalizer = ImageNormalizer(bit_depth=16)
    >>> norm = normalizer.transform(stack)                  # [H, W, 4]
    >>> norms = normalizer.transform(batch)                 # [N, H, W, 4]
    >>> norm, f = normalizer.transform(stack, return_gains=True)
    >>> recon = normalizer.transform(gen, gains=f, clamp_gains=False)
    """

    _BIT_DEPTH_MAX = {8: 255.0, 16: 65535.0}

    def __init__(
        self,
        bit_depth: int = 8,
        percentile: float = 99.95,
        ref_channel: int = 2,
        r_floor: float = 0.3,
        dim_gamma: float = 0.35,
        noise_floor: float | None = None,
    ):
        if bit_depth not in self._BIT_DEPTH_MAX:
            raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}")
        self.bit_depth = bit_depth
        self.percentile = percentile
        self.ref_channel = ref_channel
        self.r_floor = r_floor
        self.dim_gamma = dim_gamma
        self.noise_floor = (
            noise_floor if noise_floor is not None
            else (3.0 if bit_depth == 8 else 20.0)
        )

    def fit(self, X=None, y=None) -> ImageNormalizer:
        """No-op; returns self for API consistency."""
        return self

    def _normalize_one(
        self,
        img: np.ndarray,
        gains: np.ndarray | None = None,
        clamp_gains: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize a single [H, W, C] image. Returns (normalized, gains)."""
        max_val = self._BIT_DEPTH_MAX[self.bit_depth]
        img = np.asarray(img, dtype=np.float32).copy()
        n_channels = img.shape[2]
        ref = self.ref_channel

        # Step 1: clip all channels at the nucleus percentile
        ref_clip = min(
            float(np.percentile(img[..., ref], self.percentile)), max_val
        )
        if ref_clip <= 0:                     # reference channel empty
            ref_clip = min(float(img.max()), max_val)
        if ref_clip <= 0:                     # whole stack empty
            return (np.full(img.shape, -1.0, np.float32),
                    np.zeros(n_channels, np.float32))
        np.clip(img, 0.0, ref_clip, out=img)
        channel_maxes = np.array(
            [img[..., c].max() for c in range(n_channels)], dtype=np.float32
        )

        # Step 2: ratio to the nucleus channel
        nuc_max = channel_maxes[ref]
        if nuc_max <= 0:
            nuc_max = max(float(channel_maxes.max()), 1.0)
        r = channel_maxes / nuc_max

        # Step 3: gain, smooth lift below r_floor
        r0, g = self.r_floor, self.dim_gamma
        f = np.where(
            r >= r0, r, r0 * (np.maximum(r, 1e-8) / r0) ** g
        ).astype(np.float32)
        f = np.where(channel_maxes < self.noise_floor, r, f).astype(np.float32)

        if gains is not None:
            g_in = np.asarray(gains, dtype=np.float32)
            if clamp_gains:
                observed = channel_maxes > 0
                g_in = np.where(observed, np.clip(g_in, r, 1.0), g_in)
            f = np.where(np.isnan(g_in), f, g_in).astype(np.float32)

        # Step 4: apply and rescale to [-1, 1]
        for c in range(n_channels):
            if channel_maxes[c] > 0:
                img[..., c] *= f[c] / channel_maxes[c]
        np.clip(img, 0.0, 1.0, out=img)
        return img * 2.0 - 1.0, f

    def transform(
        self,
        X: np.ndarray,
        save_path: str | None = None,
        gains: np.ndarray | None = None,
        clamp_gains: bool = True,
        return_gains: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Normalize each image independently.

        Parameters
        ----------
        X : np.ndarray
            Single image [H, W, C] or batch [N, H, W, C].
        save_path : str, optional
            Save the normalized result as a float32 TIFF. For batches,
            one file per image is written as ``{stem}_{i}.tif``.
        gains : np.ndarray, optional
            [C] or [N, C]. Entries override the computed gain; ``np.nan``
            means "compute this channel". A [C] vector is broadcast over the
            batch. Required at sampling time, where the protein channel has no
            observed max.
        clamp_gains : bool
            Clamp supplied gains to ``[r_c, 1]`` for channels with a nonzero
            max, so a supplied gain cannot amplify a dim channel's noise
            floor. Set ``False`` on generated images, where there is no true
            ``r_c`` to clamp against.
        return_gains : bool
            Also return the gains actually applied, [C] or [N, C]. Cache these
            to invert generated samples back to counts.

        Returns
        -------
        np.ndarray
            Float32 array of the same shape, values in [-1, 1].
        """
        single = X.ndim == 3
        X = np.asarray(X)
        if single:
            X = X[np.newaxis]

        if gains is None:
            G = [None] * len(X)
        else:
            G = np.atleast_2d(np.asarray(gains, dtype=np.float32))
            if len(G) == 1 and len(X) > 1:
                G = np.repeat(G, len(X), axis=0)
            if len(G) != len(X):
                raise ValueError(f"gains has {len(G)} rows, X has {len(X)}")

        out = np.empty(X.shape, dtype=np.float32)
        f_all = np.empty((len(X), X.shape[-1]), dtype=np.float32)
        for i in range(len(X)):
            out[i], f_all[i] = self._normalize_one(X[i], G[i], clamp_gains)

        if save_path is not None:
            from tifffile import imwrite

            p = Path(save_path)
            if single:
                imwrite(save_path, out[0])
            else:
                suffix = p.suffix or ".tif"
                for i, img in enumerate(out):
                    imwrite(p.parent / f"{p.stem}_{i}{suffix}", img)

        result = out[0] if single else out
        f = f_all[0] if single else f_all
        return (result, f) if return_gains else result

    def fit_transform(
        self, X: np.ndarray, y=None, save_path: str | None = None, **kwargs
    ) -> np.ndarray:
        """Fit (no-op) and transform in one step."""
        return self.transform(X, save_path=save_path, **kwargs)


# ---------------------------------------------------------------------------
# ResolutionResampler
# ---------------------------------------------------------------------------

class ResolutionResampler:
    """Resample a [H, W, C] image so its pixel size matches the model resolution.

    Computes a scale factor as ``xy_resolution / model_resolution`` and
    applies bilinear interpolation via :func:`skimage.transform.resize`.
    Channels are resampled jointly, preserving relative spatial structure.

    ``fit`` is a no-op kept for API consistency. Resolution metadata is
    passed at transform time because it is an image-level property, not a
    dataset-level statistic.

    Parameters
    ----------
    model_resolution : float
        Target pixel size in µm/px. Default ``0.1067`` (ProtiCelli native).
    order : int
        Spline interpolation order passed to :func:`skimage.transform.resize`.
        ``1`` = bilinear (default, fast, no ringing). Use ``3`` for cubic
        upscaling if sharpness matters.
    atol : float
        Absolute tolerance (µm/px) within which resampling is skipped as a
        no-op. Default ``1e-3``.

    Examples
    --------
    >>> resampler = ResolutionResampler()
    >>> img_resampled = resampler.transform(stack, xy_resolution=0.0707)
    >>> img_resampled.shape  # spatially rescaled, still [H', W', 4]
    """

    MODEL_RESOLUTION = 0.1067  # µm/px

    def __init__(
        self,
        model_resolution: float = MODEL_RESOLUTION,
        order: int = 1,
        atol: float = 1e-3,
    ):
        self.model_resolution = model_resolution
        self.order = order
        self.atol = atol

    def fit(self, X=None, y=None) -> ResolutionResampler:
        """No-op; returns self for API consistency."""
        return self

    def transform(self, X: np.ndarray, xy_resolution: float, save_path: str | None = None) -> np.ndarray:
        """Resample X to the model's native pixel size.

        Parameters
        ----------
        X : np.ndarray
            Single image [H, W, C] or batch [N, H, W, C].
        xy_resolution : float
            Pixel size of the input image in µm/px.
        save_path : str, optional
            Save the resampled result as a float32 TIFF. Batch files are
            written as ``{stem}_{i}.tif``, matching :class:`ImageNormalizer`
            convention.

        Returns
        -------
        np.ndarray
            Float32 array, shape [H', W', C] or [N, H', W', C].
        """

        scale = xy_resolution / self.model_resolution

        single = X.ndim == 3
        X = np.asarray(X, dtype=np.float32)
        if single:
            X = X[np.newaxis]  # [1, H, W, C]

        if np.isclose(xy_resolution, self.model_resolution, atol=self.atol):
            result = X
        else:
            from skimage.transform import resize as sk_resize

            n, h, w, c = X.shape
            out_h = round(h * scale)
            out_w = round(w * scale)
            result = np.empty((n, out_h, out_w, c), dtype=np.float32)
            for i in range(n):
                result[i] = sk_resize(
                    X[i],
                    (out_h, out_w, c),
                    order=self.order,
                    mode="reflect",
                    anti_aliasing=scale < 1.0,
                    preserve_range=True,
                ).astype(np.float32)

        if save_path is not None:
            from tifffile import imwrite

            if single:
                imwrite(save_path, result[0])
            else:
                stem = Path(save_path).stem
                parent = Path(save_path).parent
                suffix = Path(save_path).suffix or ".tif"
                for i, img in enumerate(result):
                    imwrite(parent / f"{stem}_{i}{suffix}", img)

        return result[0] if single else result

    def fit_transform(self, X: np.ndarray, xy_resolution: float, y=None, save_path: str | None = None) -> np.ndarray:
        """Fit (no-op) and transform in one step."""
        return self.transform(X, xy_resolution=xy_resolution, save_path=save_path)
