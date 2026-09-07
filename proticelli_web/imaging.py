"""Scientific image validation, statistics, and browser preview helpers."""

from __future__ import annotations

import hashlib
import io
import itertools
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


MODEL_IMAGE_SIZE = 512
CHANNEL_KEYS = ("microtubules", "nucleus", "er")
CHANNEL_LABELS = ("Microtubules", "Nucleus", "ER")
NATIVE_PIXEL_SIZE_UM = 0.1067
SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".webp", ".gif"}
)


class ImageValidationError(ValueError):
    """Raised when an uploaded image cannot be used as model conditioning."""


def _ome_metadata(xml: str | None) -> tuple[float | None, list[str]]:
    """Extract physical pixel size and channel names without assuming a namespace."""

    if not xml:
        return None, []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None, []
    pixels = next((node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "Pixels"), None)
    if pixels is None:
        return None, []
    try:
        pixel_size = float(pixels.attrib["PhysicalSizeX"])
    except (KeyError, TypeError, ValueError):
        pixel_size = None
    names = []
    for node in pixels:
        if node.tag.rsplit("}", 1)[-1] != "Channel":
            continue
        names.append(node.attrib.get("Name") or node.attrib.get("Fluor") or f"Channel {len(names) + 1}")
    return pixel_size, names


def suggest_channel_role(text: str) -> str | None:
    """Suggest a landmark role from a filename or OME channel label."""

    value = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    rules = {
        "microtubules": ("microtubule", "tubulin", " alpha tub", " beta tub", " mt "),
        "nucleus": ("nucleus", "nuclei", "nuclear", "dapi", "hoechst", " dna "),
        "er": ("endoplasmic", "calreticulin", "calnexin", "sec61", " er "),
    }
    padded = f" {value} "
    for role, tokens in rules.items():
        if any(token in padded for token in tokens):
            return role
    return None


def _extract_planes(array: np.ndarray, axes: str, channel_names: list[str]) -> list[dict[str, Any]]:
    """Flatten a TIFF series into addressable YX planes with useful labels."""

    axes = (axes or "").upper()
    values = np.asarray(array)
    if values.ndim < 2:
        raise ImageValidationError(f"Expected an image plane, but found shape {tuple(values.shape)}.")
    if len(axes) != values.ndim or "Y" not in axes or "X" not in axes:
        axes = "Q" * max(0, values.ndim - 2) + "YX"
    y_axis, x_axis = axes.index("Y"), axes.index("X")
    other_axes = [index for index in range(values.ndim) if index not in (y_axis, x_axis)]
    ordered = np.transpose(values, other_axes + [y_axis, x_axis])
    leading_shape = ordered.shape[:-2]
    flattened = ordered.reshape((-1,) + ordered.shape[-2:])
    results: list[dict[str, Any]] = []
    coordinates = itertools.product(*(range(size) for size in leading_shape)) if leading_shape else [()]
    for plane_index, coordinate in enumerate(coordinates):
        parts = []
        channel_index = None
        for source_axis, value in zip(other_axes, coordinate):
            axis_name = axes[source_axis]
            parts.append(f"{axis_name}{value}")
            if axis_name in {"C", "S"}:
                channel_index = value
        channel_name = channel_names[channel_index] if channel_index is not None and channel_index < len(channel_names) else None
        label = channel_name or (" · ".join(parts) if parts else "Image")
        results.append(
            {
                "index": plane_index,
                "label": label,
                "channel_index": channel_index,
                "shape": [int(flattened.shape[-2]), int(flattened.shape[-1])],
                "dtype": str(flattened.dtype),
            }
        )
    return results


def inspect_tiff(path: str | Path) -> dict[str, Any]:
    """Inspect the first TIFF series and expose each addressable YX plane."""

    try:
        from tifffile import TiffFile

        with TiffFile(path) as tif:
            if not tif.series:
                raise ImageValidationError("The TIFF does not contain an image series.")
            series = tif.series[0]
            array = np.asarray(series.asarray())
            axes = str(getattr(series, "axes", ""))
            pixel_size, channel_names = _ome_metadata(getattr(tif, "ome_metadata", None))
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError(f"Could not inspect TIFF: {exc}") from exc
    planes = _extract_planes(array, axes, channel_names)
    source_name = Path(path).name
    for plane in planes:
        plane["suggested_role"] = suggest_channel_role(f"{source_name} {plane['label']}")
    return {
        "name": source_name,
        "format": "TIFF",
        "lossy": False,
        "shape": [int(value) for value in array.shape],
        "axes": axes or None,
        "dtype": str(array.dtype),
        "pixel_size_um": pixel_size,
        "planes": planes,
    }


def _read_raster_array(path: str | Path) -> tuple[np.ndarray, str, list[str]]:
    """Read a common browser image without discarding grayscale bit depth."""

    try:
        with Image.open(path) as source:
            if int(getattr(source, "n_frames", 1)) != 1:
                raise ImageValidationError(
                    "Animated or multi-frame non-TIFF images are not supported. "
                    "Export one frame or use TIFF for image stacks."
                )
            image = ImageOps.exif_transpose(source)
            if image.mode == "P":
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            elif image.mode in {"CMYK", "YCbCr", "HSV"}:
                image = image.convert("RGB")
            elif image.mode == "1":
                image = image.convert("L")
            array = np.asarray(image)
            image_format = str(source.format or Path(path).suffix.lstrip(".") or "raster").upper()
            mode = image.mode
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError(f"Could not inspect image: {exc}") from exc

    if array.ndim == 2:
        axes = "YX"
        channel_names: list[str] = []
    elif array.ndim == 3 and array.shape[-1] in {2, 3, 4}:
        axes = "YXS"
        channel_names = {
            2: ["Luminance", "Alpha"],
            3: ["Red", "Green", "Blue"],
            4: ["Red", "Green", "Blue", "Alpha"],
        }[int(array.shape[-1])]
    else:
        raise ImageValidationError(
            f"Expected a grayscale, RGB, or RGBA image, but found shape {tuple(array.shape)} "
            f"in mode {mode}."
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ImageValidationError("The image contains unsupported or non-finite pixel values.")
    return np.ascontiguousarray(array), image_format, channel_names


def inspect_raster(path: str | Path) -> dict[str, Any]:
    """Inspect PNG, JPEG, BMP, or WebP and expose grayscale/color planes."""

    array, image_format, channel_names = _read_raster_array(path)
    axes = "YX" if array.ndim == 2 else "YXS"
    planes = _extract_planes(array, axes, channel_names)
    default_color_roles = {"Red": "microtubules", "Green": "er", "Blue": "nucleus"}
    source_name = Path(path).name
    filename_role = suggest_channel_role(source_name)
    for plane in planes:
        plane["suggested_role"] = filename_role or default_color_roles.get(plane["label"])
    return {
        "name": source_name,
        "format": image_format,
        "lossy": image_format in {"JPEG", "JPG"},
        "shape": [int(value) for value in array.shape],
        "axes": axes,
        "dtype": str(array.dtype),
        "pixel_size_um": None,
        "planes": planes,
    }


def inspect_image(path: str | Path) -> dict[str, Any]:
    """Inspect any supported scientific or conventional input image."""

    suffix = Path(path).suffix.casefold()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise ImageValidationError(f"Unsupported image format: {suffix or 'missing extension'}.")
    return inspect_tiff(path) if suffix in {".tif", ".tiff"} else inspect_raster(path)


def read_tiff_plane(path: str | Path, plane_index: int) -> np.ndarray:
    """Read one flattened YX plane using the same ordering as :func:`inspect_tiff`."""

    try:
        from tifffile import TiffFile

        with TiffFile(path) as tif:
            series = tif.series[0]
            values = np.asarray(series.asarray())
            axes = str(getattr(series, "axes", ""))
    except Exception as exc:
        raise ImageValidationError(f"Could not read TIFF plane: {exc}") from exc
    if len(axes) != values.ndim or "Y" not in axes or "X" not in axes:
        axes = "Q" * max(0, values.ndim - 2) + "YX"
    y_axis, x_axis = axes.index("Y"), axes.index("X")
    other_axes = [index for index in range(values.ndim) if index not in (y_axis, x_axis)]
    ordered = np.transpose(values, other_axes + [y_axis, x_axis]).reshape((-1, values.shape[y_axis], values.shape[x_axis]))
    if plane_index < 0 or plane_index >= ordered.shape[0]:
        raise ImageValidationError(f"Plane {plane_index} is outside the TIFF series.")
    plane = np.asarray(ordered[plane_index])
    if not np.issubdtype(plane.dtype, np.number) or not np.isfinite(plane).all():
        raise ImageValidationError("The selected plane contains unsupported or non-finite pixel values.")
    return np.ascontiguousarray(plane)


def read_raster_plane(path: str | Path, plane_index: int) -> np.ndarray:
    """Read one grayscale or component plane from a conventional image."""

    values, _, _ = _read_raster_array(path)
    if values.ndim == 2:
        if plane_index != 0:
            raise ImageValidationError(f"Plane {plane_index} is outside the image.")
        plane = values
    else:
        if plane_index < 0 or plane_index >= values.shape[-1]:
            raise ImageValidationError(f"Plane {plane_index} is outside the image.")
        plane = values[..., plane_index]
    return np.ascontiguousarray(plane)


def read_image_plane(path: str | Path, plane_index: int) -> np.ndarray:
    """Read one plane from any supported image using inspection ordering."""

    suffix = Path(path).suffix.casefold()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise ImageValidationError(f"Unsupported image format: {suffix or 'missing extension'}.")
    return read_tiff_plane(path, plane_index) if suffix in {".tif", ".tiff"} else read_raster_plane(path, plane_index)


def prepare_reference_channels(
    channels: dict[str, np.ndarray],
    *,
    pixel_size_um: float,
    resample: bool,
    crop_x: int | None = None,
    crop_y: int | None = None,
    normalize: bool = False,
    normalization_bit_depth: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assemble mapped planes into the checkpoint's canonical 512px input.

    When requested, the package's :class:`ImageNormalizer` is applied after
    resampling, cropping, and padding. A zero protein plane is inserted so its
    four-channel convention remains ``[MT, protein, nucleus, ER]``.
    """

    missing = [key for key in CHANNEL_KEYS if key not in channels]
    if missing:
        raise ImageValidationError(f"Missing channel mapping: {', '.join(missing)}.")
    planes = [np.asarray(channels[key]) for key in CHANNEL_KEYS]
    if any(plane.ndim != 2 for plane in planes):
        raise ImageValidationError("Every mapped channel must be a 2D image plane.")
    if len({plane.shape for plane in planes}) != 1:
        raise ImageValidationError("Mapped landmark channels must have identical dimensions.")
    if not np.isfinite(planes).all():
        raise ImageValidationError("A mapped channel contains NaN or infinite values.")

    original_shape = tuple(int(value) for value in planes[0].shape)
    transformed = planes
    effective_pixel_size = float(pixel_size_um)
    if resample:
        from skimage.transform import resize

        scale = effective_pixel_size / NATIVE_PIXEL_SIZE_UM
        target_shape = tuple(max(1, int(round(value * scale))) for value in original_shape)
        transformed = [
            resize(plane, target_shape, order=1, preserve_range=True, anti_aliasing=scale < 1).astype(np.float32)
            for plane in planes
        ]
        effective_pixel_size = NATIVE_PIXEL_SIZE_UM
    transformed_shape = tuple(int(value) for value in transformed[0].shape)
    height, width = transformed_shape
    x = (width - MODEL_IMAGE_SIZE) // 2 if crop_x is None else int(crop_x)
    y = (height - MODEL_IMAGE_SIZE) // 2 if crop_y is None else int(crop_y)

    # Keep the window center on the source image while allowing the fixed model
    # canvas to extend beyond any edge. This supports both small images and
    # deliberately offset crops from a 512px source without allowing an empty crop.
    half = MODEL_IMAGE_SIZE // 2
    if not (-half <= x <= width - half and -half <= y <= height - half):
        raise ImageValidationError(
            "The 512 × 512 crop window must remain centered somewhere on the source image."
        )

    source_x0 = max(0, x)
    source_y0 = max(0, y)
    source_x1 = min(width, x + MODEL_IMAGE_SIZE)
    source_y1 = min(height, y + MODEL_IMAGE_SIZE)
    target_x0 = source_x0 - x
    target_y0 = source_y0 - y
    target_x1 = target_x0 + (source_x1 - source_x0)
    target_y1 = target_y0 + (source_y1 - source_y0)
    cropped = []
    for plane in transformed:
        canvas = np.zeros((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), dtype=plane.dtype)
        canvas[target_y0:target_y1, target_x0:target_x1] = plane[
            source_y0:source_y1,
            source_x0:source_x1,
        ]
        cropped.append(canvas)
    array = np.ascontiguousarray(np.stack(cropped, axis=-1))
    padding = {
        "left": max(0, -x),
        "top": max(0, -y),
        "right": max(0, x + MODEL_IMAGE_SIZE - width),
        "bottom": max(0, y + MODEL_IMAGE_SIZE - height),
    }
    normalization: dict[str, Any] = {
        "applied": False,
        "method": "ImageNormalizer",
        "stage": "after_crop",
    }
    if normalize:
        if float(array.min()) < 0:
            raise ImageValidationError(
                "ProtiCelli normalization expects non-negative raw intensities. "
                "Disable it when the selected image is already normalized."
            )
        if normalization_bit_depth is not None and normalization_bit_depth not in {8, 16}:
            raise ImageValidationError("Normalization bit depth must be 8 or 16.")
        if normalization_bit_depth is None:
            bit_depth = 8 if array.dtype == np.uint8 or float(array.max()) <= 255 else 16
            bit_depth_source = "auto"
        else:
            bit_depth = int(normalization_bit_depth)
            bit_depth_source = "user"

        from proticelli.data import ImageNormalizer

        four_channel = np.zeros(array.shape[:2] + (4,), dtype=array.dtype)
        four_channel[..., 0] = array[..., 0]
        four_channel[..., 2] = array[..., 1]
        four_channel[..., 3] = array[..., 2]
        normalizer = ImageNormalizer(bit_depth=bit_depth)
        normalized_four, gains = normalizer.transform(four_channel, return_gains=True)
        array = np.ascontiguousarray(normalized_four[..., [0, 2, 3]])
        normalization = {
            "applied": True,
            "method": "ImageNormalizer",
            "stage": "after_crop",
            "bit_depth": bit_depth,
            "bit_depth_source": bit_depth_source,
            "parameters": {
                "percentile": float(normalizer.percentile),
                "ref_channel": int(normalizer.ref_channel),
                "r_floor": float(normalizer.r_floor),
                "dim_gamma": float(normalizer.dim_gamma),
                "noise_floor": float(normalizer.noise_floor),
            },
            "gains": {
                key: float(value)
                for key, value in zip(
                    ("microtubules", "protein", "nucleus", "er"), gains
                )
            },
            "output_range": [-1.0, 1.0],
        }
    manifest = {
        "original_shape": list(original_shape),
        "resampled": bool(resample),
        "source_pixel_size_um": float(pixel_size_um),
        "effective_pixel_size_um": effective_pixel_size,
        "shape_before_crop": list(transformed_shape),
        "crop": {
            "x": x,
            "y": y,
            "width": MODEL_IMAGE_SIZE,
            "height": MODEL_IMAGE_SIZE,
            "padding": padding,
        },
        "normalization": normalization,
    }
    return array, manifest


@dataclass(frozen=True)
class ChannelStats:
    key: str
    label: str
    minimum: float
    maximum: float
    p005: float
    p995: float
    mean: float
    histogram: list[int]
    histogram_edges: list[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "p005": self.p005,
            "p995": self.p995,
            "mean": self.mean,
            "histogram": self.histogram,
            "histogram_edges": self.histogram_edges,
        }


def read_reference_tiff(path: str | Path) -> np.ndarray:
    """Read a 3- or 4-channel TIFF and return canonical [H, W, 3] channels.

    Three-channel input is interpreted as ``[MT, nucleus, ER]``. Four-channel
    input follows the package convention ``[MT, protein, nucleus, ER]`` and
    discards the observed protein channel.
    """

    try:
        from tifffile import imread

        array = np.asarray(imread(path, is_ome=False))
    except Exception as exc:  # tifffile gives several format-specific errors
        raise ImageValidationError(f"Could not read TIFF: {exc}") from exc

    array = np.squeeze(array)
    if array.ndim != 3:
        raise ImageValidationError(
            f"Expected a 3- or 4-channel TIFF, but found shape {tuple(array.shape)}."
        )

    if array.shape[-1] not in (3, 4) and array.shape[0] in (3, 4):
        array = np.moveaxis(array, 0, -1)

    if array.shape[-1] == 4:
        array = array[..., [0, 2, 3]]
    elif array.shape[-1] != 3:
        raise ImageValidationError(
            f"Expected 3 or 4 channels on the last axis, but found {array.shape[-1]}."
        )

    height, width, _ = array.shape
    if (height, width) != (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE):
        raise ImageValidationError(
            f"This checkpoint requires {MODEL_IMAGE_SIZE} × {MODEL_IMAGE_SIZE} input; "
            f"the uploaded image is {width} × {height}. Resample and crop explicitly "
            "before inference so the spatial transformation remains reproducible."
        )

    if not np.issubdtype(array.dtype, np.number):
        raise ImageValidationError(f"Unsupported pixel type {array.dtype}.")
    if not np.isfinite(array).all():
        raise ImageValidationError("The image contains NaN or infinite pixel values.")

    return np.ascontiguousarray(array)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def channel_statistics(array: np.ndarray, bins: int = 128) -> list[ChannelStats]:
    stats: list[ChannelStats] = []
    for index, (key, label) in enumerate(zip(CHANNEL_KEYS, CHANNEL_LABELS)):
        values = np.asarray(array[..., index], dtype=np.float64)
        minimum = float(values.min())
        maximum = float(values.max())
        p005, p995 = (float(v) for v in np.percentile(values, (0.5, 99.5)))
        if maximum <= minimum:
            edges = np.linspace(minimum, minimum + 1.0, bins + 1)
            hist = np.zeros(bins, dtype=np.int64)
            hist[0] = values.size
        else:
            hist, edges = np.histogram(values, bins=bins, range=(minimum, maximum))
        stats.append(
            ChannelStats(
                key=key,
                label=label,
                minimum=minimum,
                maximum=maximum,
                p005=p005,
                p995=p995,
                mean=float(values.mean()),
                histogram=hist.astype(int).tolist(),
                histogram_edges=[float(v) for v in edges],
            )
        )
    return stats


def scalar_statistics(array: np.ndarray, bins: int = 128) -> dict[str, Any]:
    """Return JSON-safe display statistics for one scalar image."""

    values = np.asarray(array, dtype=np.float64)
    minimum = float(values.min())
    maximum = float(values.max())
    p005, p995 = (float(v) for v in np.percentile(values, (0.5, 99.5)))
    if maximum <= minimum:
        edges = np.linspace(minimum, minimum + 1.0, bins + 1)
        hist = np.zeros(bins, dtype=np.int64)
        hist[0] = values.size
    else:
        hist, edges = np.histogram(values, bins=bins, range=(minimum, maximum))
    return {
        "minimum": minimum,
        "maximum": maximum,
        "p005": p005,
        "p995": p995,
        "mean": float(values.mean()),
        "histogram": hist.astype(int).tolist(),
        "histogram_edges": [float(value) for value in edges],
    }


def normalize_channel(channel: np.ndarray) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - minimum) / (maximum - minimum)
    return np.rint(np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def channel_png_bytes(channel: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(normalize_channel(channel), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def channel_preview_png_bytes(channel: np.ndarray, max_size: int = 1024) -> bytes:
    """Create a bounded grayscale preview without changing crop coordinates."""

    image = Image.fromarray(normalize_channel(channel), mode="L")
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), resample=Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def prediction_png_bytes(prediction: np.ndarray) -> bytes:
    """Create a non-destructive display preview for a predicted float image."""

    buffer = io.BytesIO()
    Image.fromarray(normalize_channel(prediction), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def normalize_reference_for_demo(reference: np.ndarray) -> np.ndarray:
    channels = [normalize_channel(reference[..., index]).astype(np.float32) / 255.0 for index in range(3)]
    return np.stack(channels, axis=-1)
