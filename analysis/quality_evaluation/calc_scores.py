"""Shared image-quality evaluation for the ProtiCelli publication.

The location, B2AI, and general profiles retain the three original workflows.
See README.md for scaling, historical MS-SSIM, and FID resampling details.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import pickle
import platform
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter
from skimage import filters
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity
from tifffile import imread
from tqdm import tqdm


TREATMENTS = ("Paclitaxel", "untreated", "Vorinostat")
METRICS = ("pearson", "spearman", "psnr", "ms_ssim", "mutual_info", "haralick")
PROFILES = {
    "location": ("latent_diffusion_edm/gen_model_test_images_new", "protVL_mixed_scores", 0),
    "b2ai": ("generated_images/mda_Aug_11", "ProtiCelli_MDA_scores_new", 0),
    "general": ("./MAC_3D_Jul_27", "MAC_3D_Jul_27_results", 32500),
}
REAL_TOKEN = re.compile(r"(?<![A-Za-z0-9])real(?![A-Za-z0-9])")


def _image(image):
    array = np.asarray(image)
    if array.ndim != 2 or not array.size or min(array.shape) < 1:
        raise ValueError("Expected a nonempty 2D image")
    if array.dtype.kind not in "buif" or not np.isfinite(array).all():
        raise ValueError("Images must contain finite real numbers")
    if array.dtype.kind in "ui" and (array.min() < -(2 ** 53) or array.max() > 2 ** 53):
        raise ValueError("Integer pixels outside [-2**53, 2**53] exceed float64 exact precision")
    return array.astype(np.float64, copy=False)


def _pair(input, target):
    first, second = _image(input), _image(target)
    if first.shape != second.shape:
        raise ValueError(f"Input images must have the same shape: {first.shape}, {second.shape}")
    return first, second


def _positive_range(data_range):
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")


def psnr(input, target, data_range=1.0):
    """PSNR in dB; cast before subtraction to avoid unsigned integer wraparound."""
    _positive_range(data_range)
    first, second = _pair(input, target)
    # Scale before squaring, avoiding overflow/underflow in ordinary MSE.
    with np.errstate(over="raise", invalid="raise"):
        difference = np.abs(first - second)
        scale = float(difference.max())
        if scale == 0:
            return float("inf")
        log_mse = 2 * np.log10(scale) + np.log10(np.mean((difference / scale) ** 2))
    return float(20 * np.log10(data_range) - 10 * log_mse)


def mutual_info_(input, target, bins=256):
    """Histogram mutual information in nats, using the original bin edges."""
    first, second = _pair(input, target)
    if not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    edges = np.linspace(0, 1, bins)
    labels = []
    for image in (first, second):
        if image.max() > 1:
            image = image / image.max()
        labels.append(np.clip(np.digitize(image.ravel(), edges) - 1, 0, bins - 1))
    pairs, counts = np.unique(np.column_stack(labels), axis=0, return_counts=True)
    marginal_x = np.bincount(labels[0], minlength=bins)
    marginal_y = np.bincount(labels[1], minlength=bins)
    probability = counts.astype(np.float64) / first.size
    return float(np.sum(probability * (
        np.log(probability) - np.log(marginal_x[pairs[:, 0]] / first.size)
        - np.log(marginal_y[pairs[:, 1]] / first.size)
    )))


def haralick_features(image, distances=(1,), angles=(0, 45, 90, 135),
                      levels=256, symmetric=True, normed=True):
    """Mean/std of six GLCM features, with historical per-image integer scaling."""
    original = np.asarray(image)
    image = _image(image)
    if not isinstance(levels, (int, np.integer)) or not 2 <= levels <= 256:
        raise ValueError("levels must be an integer in [2, 256]")
    if original.dtype.kind in "bui":
        span = image.max() - image.min()
        image = (image - image.min()) / span if span else np.zeros_like(image)
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Floating-point Haralick inputs must be in [0, 1]")
    quantized = (image * (levels - 1)).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=distances, angles=np.deg2rad(angles),
                       levels=levels, symmetric=symmetric, normed=normed)
    result = {}
    for name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        values = graycoprops(glcm, name)
        result[name] = float(values.mean())
        result[f"{name}_std"] = float(values.std())
    return result


def haralick_similarity(image1, image2, **kwargs):
    first, second = haralick_features(image1, **kwargs), haralick_features(image2, **kwargs)
    result = {f"haralick_sim_{key}": 1 - abs(value - second[key]) /
              max(abs(value), abs(second[key]), 1e-10) for key, value in first.items()}
    result["haralick_overall_similarity"] = float(np.mean([
        value for key, value in result.items() if not key.endswith("_std")
    ]))
    return result


def _gabor_responses(image, frequencies, angles, sigma_x=2, sigma_y=2):
    for frequency in frequencies:
        for angle in angles:
            real, imaginary = filters.gabor(image, frequency=frequency,
                                           theta=np.deg2rad(angle), sigma_x=sigma_x, sigma_y=sigma_y)
            yield frequency, angle, real, imaginary, np.hypot(real, imaginary)


def gabor_filter_bank(image, frequencies=(0.1, 0.3, 0.5), angles=(0, 45, 90, 135),
                      sigma_x=2, sigma_y=2):
    """Compatibility helper returning features and individual filter responses."""
    image = _image(image)
    features, responses = {}, []
    for frequency, angle, real, imaginary, magnitude in _gabor_responses(
            image, frequencies, angles, sigma_x, sigma_y):
        prefix = f"gabor_f{frequency}_a{angle}"
        features.update({f"{prefix}_mean": float(magnitude.mean()),
                         f"{prefix}_std": float(magnitude.std()),
                         f"{prefix}_energy": float(np.sum(magnitude ** 2)),
                         f"{prefix}_entropy": float(-np.sum(magnitude * np.log(magnitude + 1e-10)))})
        responses.append({"frequency": frequency, "angle": angle, "real": real,
                          "imaginary": imaginary, "magnitude": magnitude})
    if not responses:
        raise ValueError("At least one Gabor frequency and angle are required")
    features.update({"gabor_global_mean": float(np.mean([r["magnitude"].mean() for r in responses])),
                     "gabor_global_std": float(np.mean([r["magnitude"].std() for r in responses])),
                     "gabor_global_max": float(max(r["magnitude"].max() for r in responses))})
    return features, responses


def gabor_similarity(image1, image2, frequencies=(0.1, 0.3, 0.5), angles=(0, 45, 90, 135)):
    """Correlate magnitudes one filter at a time, avoiding a stored filter bank."""
    first, second = _pair(image1, image2)
    result = {}
    for response1, response2 in zip(_gabor_responses(first, frequencies, angles),
                                    _gabor_responses(second, frequencies, angles)):
        frequency, angle = response1[:2]
        first_flat = response1[-1].ravel()
        second_flat = response2[-1].ravel()
        if first_flat.std() == 0 or second_flat.std() == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(first_flat, second_flat)[0, 1])
        result[f"gabor_sim_f{frequency}_a{angle}"] = correlation
    if not result:
        raise ValueError("At least one Gabor frequency and angle are required")
    result["gabor_overall_similarity"] = float(np.mean(list(result.values())))
    return result


def ms_ssim(input, target, win_size=11, data_range=1.0, method="historical"):
    """Historical approximation, or corrected contrast-structure product.

    Original scripts use the full SSIM map as the contrast-structure map. Keep
    that definition by default for manuscript reproducibility. 'standard' uses
    actual contrast-structure terms with the same uniform-window settings.
    Both methods truncate/renormalize weights for images below five scales.
    """
    _positive_range(data_range)
    first, second = _pair(input, target)
    if not isinstance(win_size, (int, np.integer)) or win_size < 3 or win_size % 2 == 0:
        raise ValueError("win_size must be an odd integer >= 3")
    if min(first.shape) < win_size:
        raise ValueError("Image is smaller than the SSIM window")
    if method not in ("historical", "standard"):
        raise ValueError("MS-SSIM method must be historical or standard")
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    similarities, contrasts = [], []
    for scale in range(len(weights)):
        similarity, ssim_map = structural_similarity(
            first, second, win_size=win_size, data_range=data_range, full=True)
        similarities.append(similarity)
        if method == "historical":
            contrasts.append(float(ssim_map.mean()))
        else:
            mean1, mean2 = uniform_filter(first, win_size), uniform_filter(second, win_size)
            correction = win_size ** 2 / (win_size ** 2 - 1)
            var1 = correction * (uniform_filter(first * first, win_size) - mean1 * mean1)
            var2 = correction * (uniform_filter(second * second, win_size) - mean2 * mean2)
            cov = correction * (uniform_filter(first * second, win_size) - mean1 * mean2)
            constant = (0.03 * data_range) ** 2
            contrast = (2 * cov + constant) / (var1 + var2 + constant)
            pad = (win_size - 1) // 2
            contrasts.append(float(contrast[pad:-pad, pad:-pad].mean()))
        if scale == len(weights) - 1:
            break
        if min(first.shape) < 2 * win_size:
            weights = weights[:scale + 1]
            weights = weights / weights.sum()
            break
        pooled = []
        for image in (first, second):
            height, width = image.shape
            image = image[:height // 2 * 2, :width // 2 * 2]
            pooled.append(image.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3)))
        first, second = pooled
    floor = 1e-8 if method == "historical" else 0.0
    contrasts = np.clip(contrasts, floor, 1)
    similarities = np.clip(similarities, floor, 1)
    if len(similarities) == 1:
        return float(similarities[0])
    return float(np.prod(contrasts[:-1] ** weights[:-1]) * similarities[-1] ** weights[-1])


class InceptionFeatureExtractor:
    """Lazy, bounded InceptionV3 extraction; weights are needed only when used."""

    def __init__(self, device=None):
        import torch
        from torch import nn
        from torchvision import models, transforms

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT,
                                         transform_input=False)
        self.model.fc = nn.Identity()
        self.model = self.model.to(self.device).eval()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((299, 299)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _to_uint8_rgb(image):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.ndim != 3 or image.shape[-1] != 3 or not image.size:
            raise ValueError("Inception expects a grayscale or RGB image")
        if image.dtype.kind not in "buif" or not np.isfinite(image).all() or image.min() < 0:
            raise ValueError("Inception inputs must be finite and nonnegative")
        if image.dtype == np.uint8:
            return image
        if image.dtype.kind in "bui":
            maximum = 1 if image.dtype.kind == "b" else np.iinfo(image.dtype).max
            image = image.astype(np.float64) / maximum
        elif image.max() > 1:
            raise ValueError("Floating-point Inception inputs must be in [0, 1]")
        return (image * 255).astype(np.uint8)

    def extract_features(self, image):
        return self.extract_features_batch([image])[0]

    def extract_features_batch(self, images):
        import torch

        if not images:
            raise ValueError("Feature extraction requires at least one image")
        batch = None
        try:
            batch = torch.stack([self.preprocess(self._to_uint8_rgb(img)) for img in images])
            batch = batch.to(self.device)
            with torch.no_grad():
                return self.model(batch).cpu().numpy()
        except torch.OutOfMemoryError:
            del batch
            if len(images) == 1:
                raise
        # Leave the exception handler first: its traceback retains forward-pass
        # tensors, which otherwise stay allocated during the smaller retries.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        middle = len(images) // 2
        return np.concatenate([self.extract_features_batch(images[:middle]),
                               self.extract_features_batch(images[middle:])])


def calculate_pairwise_fid(extractor, real_image, generated_image):
    """Return feature L2 distance (a proxy, not distributional FID) and features."""
    real = np.asarray(extractor.extract_features(real_image), dtype=np.float64)
    generated = np.asarray(extractor.extract_features(generated_image), dtype=np.float64)
    return float(np.linalg.norm(real - generated)), real, generated


def calculate_fid_score(real_features, gen_features, *, sample_counts=None):
    """Gaussian FID with sample covariances, including singular/1D feature sets.

    Integer sample_counts are frequency weights, exactly equivalent to repeating
    paired rows. Resampling therefore does not allocate padded feature matrices.
    """
    real, generated = (np.asarray(features) for features in (real_features, gen_features))
    if real.dtype.kind not in "buif" or generated.dtype.kind not in "buif":
        raise ValueError("FID requires real numeric features")
    real, generated = real.astype(np.float64, copy=False), generated.astype(np.float64, copy=False)
    if (real.ndim != 2 or generated.ndim != 2 or not real.shape[1]
            or real.shape[1] != generated.shape[1]):
        raise ValueError("FID requires two sample-by-feature matrices with equal feature dimensions")
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError("FID features must be finite")
    if sample_counts is not None:
        counts = np.asarray(sample_counts)
        if (len(real) != len(generated) or counts.shape != (len(real),)
                or counts.dtype.kind not in "iu" or np.any(counts < 0)):
            raise ValueError("sample_counts must be nonnegative integer frequencies for paired rows")
        counts = counts.astype(np.float64)
    else:
        counts = None
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        centered, means, traces = [], [], []
        for features in (real, generated):
            frequencies = np.ones(len(features)) if counts is None else counts
            total = frequencies.sum()
            if total < 2:
                raise ValueError("FID requires at least two samples in each feature set")
            mean = np.average(features, axis=0, weights=frequencies)
            factor = (features - mean) * np.sqrt(frequencies[:, None] / (total - 1))
            means.append(mean)
            centered.append(factor)
            traces.append(float(np.sum(factor * factor)))
        first, second = centered
        # For small sample sets the nuclear norm avoids large, singular D x D
        # covariance matrices. Both routes compute the same Bures cross term.
        if len(first) * len(second) < real.shape[1] ** 2:
            cross_trace = np.linalg.svd(first @ second.T, compute_uv=False).sum()
        else:
            covariance1, covariance2 = first.T @ first, second.T @ second
            eigenvalues, eigenvectors = np.linalg.eigh(covariance1)
            root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
            product = root @ covariance2 @ root
            cross_trace = np.sqrt(np.maximum(np.linalg.eigvalsh((product + product.T) / 2), 0)).sum()
        difference = means[0] - means[1]
        mean_distance = float(difference @ difference)
        fid = float(mean_distance + sum(traces) - 2 * cross_trace)
    tolerance = 1e-7 * max(1.0, mean_distance + sum(traces))
    if not np.isfinite(fid) or fid < -tolerance:
        raise ValueError(f"Numerically invalid FID: {fid}")
    return max(0.0, fid)


def _read_image(path, channel, data_range):
    image = np.asarray(imread(path))
    if image.ndim == 3:
        if channel >= image.shape[2]:
            raise ValueError(f"Channel {channel} unavailable in {image.shape}")
        image = image[..., channel]
    image = _image(image)
    if image.min() < 0 or image.max() > data_range:
        raise ValueError(f"Pixel intensities must lie in [0, {data_range}]; set --data-range explicitly")
    return image / data_range


def _metadata(filename, profile, location_map, mode, treatment_groups):
    if profile == "general":
        return {}, [], None
    parts = filename.split("_", 2)
    if len(parts) < 3 or not parts[0] or not parts[1]:
        return {}, [], "invalid_metadata"
    metadata = {"cell_line": parts[0], "gene_name": parts[1]}
    if profile == "location":
        locations = location_map.get(parts[1], {}).get(parts[0], []) or ["unknown"]
        locations = [locations] if isinstance(locations, str) else list(locations)
        locations = list(dict.fromkeys(locations))
        if ((mode == "single" and len(locations) != 1)
                or (mode == "multi" and len(locations) == 1)):
            return metadata, [], "filtered_localization"
        metadata["location"] = locations
        return metadata, locations, None
    matches = [group for group in treatment_groups
               if re.search(r"(?<![A-Za-z0-9])" + re.escape(group) + r"(?![A-Za-z0-9])", filename)]
    if len(matches) != 1:
        return metadata, [], "unmatched_treatment"
    metadata.update({"treatment": matches[0], "location": matches[0]})
    return metadata, matches, None


def _json_safe(value):
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, allow_nan=False)


def _write_pickle(path, value):
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _package_versions():
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "scikit-image", "tifffile", "torch", "torchvision"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _save_group_results(directory, frame, groups, metric_names, mode):
    raw = {group: {**{metric: [record[metric] for record in records] for metric in metric_names},
                   "count": len(records)} for group, records in groups.items()}
    averaged = {group: {metric: float(np.mean(values[metric])) for metric in metric_names}
                for group, values in raw.items()}
    _write_pickle(directory / "scores_raw_by_location.pkl", raw)
    _write_pickle(directory / f"scores_by_location.pkl_{mode}", averaged)
    genes = frame.groupby("gene_name")[list(metric_names)].mean()
    genes.to_csv(directory / "gene_average_scores.csv")
    genes.to_pickle(directory / "gene_average_scores.pkl")
    rankings = {}
    for gene, row in genes.iterrows():
        ranks = {metric: float(stats.percentileofscore(genes[metric], row[metric])) for metric in metric_names}
        rankings[gene] = {"avg_percentile_rank": float(np.mean(list(ranks.values()))),
                          "individual_ranks": ranks, "avg_scores": row.to_dict()}
    rankings = dict(sorted(rankings.items(), key=lambda pair: pair[1]["avg_percentile_rank"], reverse=True))
    _write_json(directory / "gene_rankings_detailed.json", rankings)
    with (directory / "ranked_genes.txt").open("w", encoding="utf-8") as handle:
        handle.write("Gene Rankings (by average percentile across all metrics)\n" + "=" * 60 + "\n\n")
        for rank, (gene, values) in enumerate(rankings.items(), 1):
            handle.write(f"{rank:2d}. {gene}: Avg Percentile = {values['avg_percentile_rank']:.1f}%\n")
            handle.write("    " + ", ".join(f"{key}: {value:.1f}%" for key, value in values["individual_ranks"].items()) + "\n\n")
    return averaged, len(genes)


def _save_plots(directory, frame, metric_names, grouped, overall_fid):
    import matplotlib.pyplot as plt

    panels = len(metric_names) + 1
    rows = (panels + 3) // 4
    figure, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows), squeeze=False)
    for axis, metric in zip(axes.flat, metric_names):
        if grouped:
            names = list(grouped)
            values = np.array([grouped[name][metric] for name in names])
            axis.bar(names, np.where(np.isfinite(values), values, np.nan))
            axis.tick_params(axis="x", labelrotation=90)
        else:
            values = frame[metric].to_numpy(dtype=float)
            axis.hist(values[np.isfinite(values)], bins=40)
        axis.set_title(metric)
    axis = axes.flat[len(metric_names)]
    axis.set_title("Overall FID")
    if overall_fid is not None:
        axis.scatter([0], [overall_fid])
        axis.set_xticks([0], ["All images"])
        axis.annotate(f"{overall_fid:.3f}", (0, overall_fid), xytext=(0, 10), textcoords="offset points", ha="center")
    else:
        axis.text(0.5, 0.5, "Fewer than two valid pairs", ha="center", transform=axis.transAxes)
    for axis in list(axes.flat)[panels:]:
        axis.axis("off")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"scores_plot.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def evaluate(image_dir, results_dir, *, profile="general", location_map=None, mode="mixed",
             treatment_groups=TREATMENTS, channel=1, data_range=1.0, batch_size=16,
             fid_target_n=0, seed=0, extractor=None, include_gabor=None, plots=True,
             ms_ssim_method="historical"):
    """Evaluate real/pred TIFF pairs and return the saved summary statistics.

    All metrics and features for a pair enter the results together. Only positive
    infinite PSNR (an exact match) is permitted among otherwise finite scores.
    The caller may inject a feature extractor for offline testing.
    """
    if profile not in PROFILES or mode not in ("single", "multi", "mixed"):
        raise ValueError("Unknown evaluation profile or localization mode")
    _positive_range(data_range)
    for name, value, minimum in (("batch_size", batch_size, 1), ("channel", channel, 0),
                                 ("fid_target_n", fid_target_n, 0), ("seed", seed, 0)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if ms_ssim_method not in ("historical", "standard"):
        raise ValueError("MS-SSIM method must be historical or standard")
    map_provenance = None
    if profile == "location":
        if isinstance(location_map, (str, Path)):
            with Path(location_map).open("rb") as handle:
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                map_provenance = {"path": str(Path(location_map).resolve()), "sha256": digest.hexdigest()}
                handle.seek(0)
                location_map = pickle.load(handle)
        if not isinstance(location_map, Mapping):
            raise ValueError("The location profile requires a location_map mapping or pickle path")
    if profile == "b2ai" and (not treatment_groups or any(not group for group in treatment_groups)):
        raise ValueError("At least one nonempty treatment group is required")
    include_gabor = profile == "location" if include_gabor is None else include_gabor
    metric_names = METRICS + (("gabor",) if include_gabor else ())
    score_columns = list(metric_names) + ["fid_distance"]
    metadata_columns = [] if profile == "general" else ["cell_line", "gene_name", "location"]
    if profile == "b2ai":
        metadata_columns.append("treatment")
    source, destination = Path(image_dir), Path(results_dir)
    candidates = sorted(path for path in source.iterdir() if path.is_file()
                        and path.suffix.lower() in (".tif", ".tiff") and REAL_TOKEN.search(path.stem))
    destination.mkdir(parents=True, exist_ok=True)
    scores, metadata, groups = {}, {}, {}
    details = {"per_image_scores": {}, "per_image_metadata": {}, "haralick_detailed": {}}
    if include_gabor:
        details["gabor_detailed"] = {}
    real_features, generated_features, pending = [], [], []
    skipped = Counter()

    def flush():
        nonlocal extractor
        if not pending:
            return
        if extractor is None:
            extractor = InceptionFeatureExtractor()
        real_batch = np.asarray(extractor.extract_features_batch([item[1] for item in pending]))
        gen_batch = np.asarray(extractor.extract_features_batch([item[2] for item in pending]))
        if (real_batch.ndim != 2 or real_batch.shape != gen_batch.shape
                or real_batch.shape[0] != len(pending) or real_batch.shape[1] == 0):
            raise ValueError("Extractor returned invalid or misaligned feature matrices")
        if real_batch.dtype.kind not in "buif" or gen_batch.dtype.kind not in "buif":
            raise ValueError("Extractor must return real numeric features")
        if real_features and real_batch.shape[1] != real_features[0].size:
            raise ValueError("Extractor feature dimension changed between batches")
        for item, real_feature, gen_feature in zip(pending, real_batch, gen_batch):
            filename, _, _, record, image_metadata, image_groups, texture, gabor = item
            if not np.isfinite(real_feature).all() or not np.isfinite(gen_feature).all():
                skipped["nonfinite_features"] += 1
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                distance = float(np.linalg.norm(real_feature.astype(np.float64) - gen_feature.astype(np.float64)))
            if not np.isfinite(distance):
                skipped["nonfinite_features"] += 1
                continue
            record["fid_distance"] = distance
            scores[filename], metadata[filename] = record, image_metadata
            real_features.append(real_feature.copy())
            generated_features.append(gen_feature.copy())
            details["per_image_scores"][filename] = record.copy()
            details["per_image_metadata"][filename] = image_metadata
            details["haralick_detailed"][filename] = texture
            if include_gabor:
                details["gabor_detailed"][filename] = gabor
            for group in image_groups:
                groups.setdefault(group, []).append(record)
        pending.clear()

    for path in tqdm(candidates, desc="Processing files"):
        if len(REAL_TOKEN.findall(path.stem)) != 1:
            skipped["invalid_metadata"] += 1
            continue
        image_metadata, image_groups, reason = _metadata(path.name, profile, location_map, mode, treatment_groups)
        if reason:
            skipped[reason] += 1
            continue
        prediction = path.with_name(REAL_TOKEN.sub("pred", path.stem) + path.suffix)
        if not prediction.is_file():
            skipped["missing_prediction"] += 1
            continue
        try:
            real, generated = _pair(_read_image(path, channel, data_range),
                                    _read_image(prediction, channel, data_range))
            if real.min() == real.max() or generated.min() == generated.max():
                skipped["constant_image"] += 1
                continue
            texture = haralick_similarity(real, generated)
            gabor = gabor_similarity(real, generated) if include_gabor else {}
            record = {
                "pearson": float(stats.pearsonr(real.ravel(), generated.ravel()).statistic),
                "spearman": float(stats.spearmanr(real.ravel(), generated.ravel()).statistic),
                "psnr": psnr(real, generated),
                "ms_ssim": ms_ssim(real, generated, method=ms_ssim_method),
                "mutual_info": mutual_info_(real, generated),
                "haralick": texture["haralick_overall_similarity"],
            }
            if include_gabor:
                record["gabor"] = gabor["gabor_overall_similarity"]
        except (ValueError, OSError, FloatingPointError) as error:
            skipped["invalid_image"] += 1
            warnings.warn(f"Skipping {path.name}: {error}", RuntimeWarning, stacklevel=2)
            continue
        if any(not np.isfinite(value) and not (key == "psnr" and value == np.inf)
               for key, value in record.items()):
            skipped["nonfinite_metrics"] += 1
            continue
        pending.append((path.name, real, generated, record, image_metadata, image_groups, texture, gabor))
        if len(pending) >= batch_size:
            flush()
    flush()

    frame = pd.DataFrame.from_dict(scores, orient="index").reindex(columns=score_columns)
    frame.index.name = "filename"
    for column in metadata_columns:
        frame[column] = [metadata[name][column] for name in frame.index]
    frame.to_csv(destination / "individual_image_scores.csv")
    frame.to_pickle(destination / "individual_image_scores.pkl")
    real_array, gen_array = np.asarray(real_features), np.asarray(generated_features)
    real_features.clear()
    generated_features.clear()
    n_valid, n_bootstrap, overall_fid = len(scores), 0, None
    if n_valid >= 2:
        counts = None
        if n_valid < fid_target_n:
            indices = np.random.default_rng(seed).integers(0, n_valid, size=fid_target_n - n_valid)
            counts = np.bincount(indices, minlength=n_valid) + 1
            n_bootstrap = fid_target_n - n_valid
        overall_fid = calculate_fid_score(real_array, gen_array, sample_counts=counts)
    n_fid_used = n_valid + n_bootstrap if overall_fid is not None else 0
    fid_data = {"real_features": real_array, "generated_features": gen_array,
                "overall_fid": overall_fid, "feature_filenames": list(scores),
                "n_valid_pairs": n_valid, "n_fid_target": fid_target_n,
                "n_bootstrap_pairs": n_bootstrap, "n_fid_samples_used": n_fid_used}
    _write_pickle(destination / "fid_features_and_scores.pkl", fid_data)
    with (destination / "overall_fid_score.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Overall FID Score: {overall_fid}\nValid pairs: {n_valid}\n"
                     f"FID target N: {fid_target_n}\nBootstrapped (resampled) pairs: {n_bootstrap}\n"
                     f"Total samples used for FID: {n_fid_used}\n")
    _write_json(destination / "detailed_texture_analysis.json", details)
    grouped, n_genes = {}, 0
    if profile != "general":
        grouped, n_genes = _save_group_results(destination, frame, groups, metric_names, mode)
    distributions = {}
    for metric in score_columns:
        values = frame[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        distributions[metric] = {
            "mean": float(finite.mean()) if len(finite) else None,
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "median": float(np.median(finite)) if len(finite) else None,
            "finite_count": len(finite), "positive_infinity_count": int(np.isposinf(values).sum()),
        }
    summary = {
        "overall_statistics": {
            "total_images_processed": len(candidates), "valid_images_analyzed": n_valid,
            "images_skipped_nan": skipped["nonfinite_metrics"] + skipped["nonfinite_features"],
            "skip_reasons": dict(skipped), "overall_fid_score": overall_fid,
            "mode": mode, "profile": profile, "unique_genes": n_genes,
            "unique_locations": len(groups), "fid_target_n": fid_target_n,
            "fid_bootstrap_pairs": n_bootstrap, "fid_samples_used": n_fid_used,
        },
        "configuration": {"channel": channel, "data_range": data_range, "batch_size": batch_size,
                          "fid_bootstrap_seed": seed, "include_gabor": include_gabor,
                          "ms_ssim_method": ms_ssim_method,
                          "treatment_groups": list(treatment_groups) if profile == "b2ai" else None},
        "provenance": {"image_dir": str(source.resolve()), "location_map": map_provenance,
                       "package_versions": _package_versions(),
                       "feature_extractor": type(extractor).__name__ if extractor is not None else None,
                       "weights": "torchvision.Inception_V3_Weights.DEFAULT"
                       if type(extractor) is InceptionFeatureExtractor else None},
        "metric_distributions": distributions,
    }
    _write_json(destination / "summary_statistics.json", summary)
    if plots:
        plot_metrics = metric_names if profile != "general" else tuple(score_columns)
        _save_plots(destination, frame, plot_metrics, grouped, overall_fid)
    print(f"Saved {n_valid} valid pairs ({sum(skipped.values())} skipped) to {destination}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="location")
    parser.add_argument("--image-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--location-map", default="location_map.pkl")
    parser.add_argument("--mode", choices=("single", "multi", "mixed"), default="mixed")
    parser.add_argument("--treatment-groups", nargs="+", default=TREATMENTS)
    parser.add_argument("--channel", type=int, default=1, help="Protein channel of HWC TIFFs; 2D TIFFs are used directly")
    parser.add_argument("--data-range", type=float, default=1.0, help="Intensity maximum (1, 255, 65535, etc.); inputs must be nonnegative")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fid-target-n", type=int, help="Pad FID with resampled pairs to N; 0 disables (general default: 32500)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gabor", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ms-ssim-method", choices=("historical", "standard"), default="historical")
    args = parser.parse_args(argv)
    image_dir, results_dir, target = PROFILES[args.profile]
    try:
        return evaluate(args.image_dir or image_dir, args.results_dir or results_dir,
                        profile=args.profile, location_map=args.location_map, mode=args.mode,
                        treatment_groups=args.treatment_groups, channel=args.channel, data_range=args.data_range,
                        batch_size=args.batch_size, fid_target_n=target if args.fid_target_n is None else args.fid_target_n,
                        seed=args.seed, include_gabor=args.gabor, plots=args.plots, ms_ssim_method=args.ms_ssim_method)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
