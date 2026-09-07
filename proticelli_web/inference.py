"""Inference engines for real checkpoints and an explicitly labeled demo mode."""

from __future__ import annotations

import hashlib
import gc
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .imaging import normalize_reference_for_demo


if platform.system() == "Darwin":
    # Let unsupported MPS operators fall back to CPU instead of aborting a run.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


ProgressCallback = Callable[[float, str], None]


def runtime_diagnostics(requested_device: str = "auto") -> dict[str, object]:
    """Report accelerator availability across macOS, Linux, and Windows."""

    details: dict[str, object] = {
        "requested_device": requested_device,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "torch_version": None,
        "torch_cuda_build": None,
        "torch_hip_build": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "mps_built": False,
        "mps_available": False,
        "nvidia_driver_detected": False,
        "nvidia_gpus": [],
        "status": "torch_missing",
    }
    try:
        import torch

        details["torch_version"] = str(getattr(torch, "__version__", "unknown"))
        details["torch_cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
        details["torch_hip_build"] = getattr(getattr(torch, "version", None), "hip", None)
        details["cuda_available"] = bool(torch.cuda.is_available())
        device_count = getattr(torch.cuda, "device_count", lambda: 1)
        details["cuda_device_count"] = int(device_count()) if details["cuda_available"] else 0
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None:
            details["mps_built"] = bool(getattr(mps_backend, "is_built", lambda: False)())
            details["mps_available"] = bool(getattr(mps_backend, "is_available", lambda: False)())
    except (ImportError, RuntimeError):
        torch = None

    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if completed.returncode == 0:
                gpus = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
                details["nvidia_driver_detected"] = bool(gpus)
                details["nvidia_gpus"] = gpus
        except (OSError, subprocess.SubprocessError):
            pass

    requested = requested_device.strip().lower()
    if torch is None:
        details["status"] = "torch_missing"
    elif requested == "cpu":
        details["status"] = "cpu_requested"
    elif requested == "auto":
        if details["cuda_available"]:
            details["status"] = "rocm_ready" if details["torch_hip_build"] else "cuda_ready"
        elif details["mps_available"]:
            details["status"] = "mps_ready"
        elif details["nvidia_driver_detected"] and details["torch_cuda_build"] is None:
            details["status"] = "pytorch_cpu_build"
        elif details["torch_cuda_build"] is not None and not details["nvidia_driver_detected"]:
            details["status"] = "nvidia_driver_missing"
        elif details["torch_hip_build"] is not None:
            details["status"] = "cuda_runtime_unavailable"
        else:
            details["status"] = "cpu_ready"
    elif requested == "mps":
        if details["mps_available"]:
            details["status"] = "mps_ready"
        elif details["mps_built"]:
            details["status"] = "mps_unavailable"
        else:
            details["status"] = "mps_not_built"
    elif requested == "cuda" or requested.startswith("cuda:"):
        if details["cuda_available"]:
            details["status"] = "rocm_ready" if details["torch_hip_build"] else "cuda_ready"
        elif details["torch_cuda_build"] is None and details["torch_hip_build"] is None:
            details["status"] = "pytorch_cpu_build"
        elif not details["nvidia_driver_detected"] and not details["torch_hip_build"]:
            details["status"] = "nvidia_driver_missing"
        else:
            details["status"] = "cuda_runtime_unavailable"
    else:
        details["status"] = "unsupported_device"
    return details


def resolve_device(requested_device: str, diagnostics: dict[str, object]) -> str:
    """Resolve an explicit or automatic request to a usable Torch device."""

    requested = requested_device.strip().lower()
    if requested == "auto":
        if diagnostics["cuda_available"]:
            return "cuda"
        if diagnostics["mps_available"]:
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        return "mps" if diagnostics["mps_available"] else "cpu"
    if requested == "cuda" or requested.startswith("cuda:"):
        return requested if diagnostics["cuda_available"] else "cpu"
    raise ValueError("PROTICELLI_WEB_DEVICE must be auto, cpu, mps, cuda, or cuda:N")


@dataclass
class InferenceOutput:
    images: list[np.ndarray]
    medoid_index: int
    reliability: float
    engine: str


class InferenceEngine(Protocol):
    name: str

    def generate(
        self,
        reference: np.ndarray,
        protein: str,
        cell_line: str | None,
        num_samples: int,
        num_inference_steps: int,
        seed: int,
        progress: ProgressCallback,
        cancel_check: Callable[[], bool] | None = None,
        dtype: str = "float32",
    ) -> InferenceOutput: ...


def compute_medoid_and_reliability(
    ensemble: np.ndarray, cell_mask: np.ndarray | None = None, eps: float = 1e-8
) -> tuple[int, float]:
    """Local, NumPy-only equivalent of ProtiCelli's ensemble statistic."""

    raw = np.nan_to_num(np.asarray(ensemble, dtype=np.float32))
    count = raw.shape[0]
    if cell_mask is None or np.asarray(cell_mask, dtype=bool).sum() < 2:
        mask = np.ones(raw.shape[1:], dtype=bool)
    else:
        mask = np.asarray(cell_mask, dtype=bool)
    if count == 1 or mask.sum() < 2:
        return 0, float("nan")
    pixels = raw.reshape(count, -1)[:, mask.reshape(-1)]
    centered = pixels - pixels.mean(axis=1, keepdims=True)
    denom = np.sqrt((centered**2).mean(axis=1, keepdims=True))
    valid = np.where(denom[:, 0] > eps)[0]
    if valid.size < 2:
        return (int(valid[0]) if valid.size else 0), float("nan")
    standardized = centered[valid] / (denom[valid] + eps)
    corr = (standardized @ standardized.T) / standardized.shape[1]
    np.fill_diagonal(corr, np.nan)
    centrality = np.nanmean(corr, axis=1)
    medoid = int(valid[int(np.nanargmax(centrality))])
    return medoid, float(np.clip(np.nanmean(corr), 0.0, 1.0))


def _box_blur(image: np.ndarray, rounds: int = 2) -> np.ndarray:
    result = np.asarray(image, dtype=np.float32)
    for _ in range(rounds):
        padded = np.pad(result, 1, mode="reflect")
        result = sum(
            padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ) / 9.0
    return result


class DemoInferenceEngine:
    """Deterministic synthetic renderer for developing the UI without weights.

    Its outputs are deliberately identified as simulated everywhere. They are
    never represented as model predictions.
    """

    name = "demo"

    def generate(
        self,
        reference: np.ndarray,
        protein: str,
        cell_line: str | None,
        num_samples: int,
        num_inference_steps: int,
        seed: int,
        progress: ProgressCallback,
        cancel_check: Callable[[], bool] | None = None,
        dtype: str = "float32",
    ) -> InferenceOutput:
        del num_inference_steps, dtype
        ref = normalize_reference_for_demo(reference)
        mt, nucleus, er = (ref[..., index] for index in range(3))
        cell_mask = np.maximum.reduce((mt, nucleus, er)) > 0.06
        token = hashlib.sha256(f"{protein}|{cell_line or ''}".encode()).digest()
        weights = np.asarray([token[0], token[1], token[2]], np.float32) + 32
        weights /= weights.sum()
        shared = weights[0] * _box_blur(mt, 1) + weights[1] * _box_blur(nucleus, 2) + weights[2] * _box_blur(er, 1)
        edge = np.abs(shared - _box_blur(shared, 3))
        shared = np.clip(0.78 * shared + 1.8 * edge, 0.0, 1.0) * cell_mask

        images: list[np.ndarray] = []
        for index in range(num_samples):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Inference cancelled")
            progress(0.08 + 0.82 * (index / max(num_samples, 1)), f"Rendering sample {index + 1} of {num_samples}")
            rng = np.random.default_rng(seed + index * 1009 + int.from_bytes(token[3:7], "little"))
            noise = _box_blur(rng.random(shared.shape, dtype=np.float32), 2) - 0.5
            impulses = np.zeros(shared.shape, dtype=np.float32)
            eligible = np.flatnonzero(cell_mask)
            if eligible.size:
                selected = rng.choice(eligible, size=min(36, eligible.size), replace=False)
                impulses.flat[selected] = rng.uniform(0.4, 1.0, selected.size)
            puncta = _box_blur(impulses, 2) * (8.0 + token[7] / 32.0)
            image = np.clip(shared + 0.12 * noise + puncta, 0.0, None)
            high = float(np.percentile(image[cell_mask], 99.5)) if cell_mask.any() else float(image.max())
            if high > 0:
                image = np.clip(image / high, 0.0, 1.3)
            images.append(image.astype(np.float32))

        ensemble = np.stack(images)
        medoid, reliability = compute_medoid_and_reliability(ensemble, cell_mask)
        progress(0.96, "Summarizing ensemble")
        return InferenceOutput(images, medoid, reliability, self.name)


class RealInferenceEngine:
    name = "proticelli"

    def __init__(self, package_root: Path):
        self.package_root = package_root
        self._model = None
        requested_device = os.getenv("PROTICELLI_WEB_DEVICE", "auto").strip().lower()
        diagnostics = runtime_diagnostics(requested_device)
        self.device = resolve_device(requested_device, diagnostics)
        device_name = "CPU"
        accelerator = "cpu"
        if self.device.startswith("cuda"):
            import torch

            device_index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
            device_name = torch.cuda.get_device_name(device_index)
            accelerator = "rocm" if diagnostics["torch_hip_build"] else "cuda"
        elif self.device == "mps":
            import torch

            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            get_name = getattr(mps_backend, "get_name", None)
            try:
                device_name = str(get_name()) if get_name else "Apple GPU (MPS)"
            except RuntimeError:
                device_name = "Apple GPU (MPS)"
            accelerator = "mps"
        default_dtype = "float32"
        self.dtype = os.getenv("PROTICELLI_WEB_DTYPE", default_dtype).strip().lower()
        if self.dtype not in {"float16", "float32"}:
            raise ValueError("PROTICELLI_WEB_DTYPE must be float16 or float32")
        default_batch = 4 if self.device.startswith("cuda") else 1
        try:
            configured_batch = int(os.getenv("PROTICELLI_WEB_GPU_BATCH", str(default_batch)))
        except ValueError:
            configured_batch = default_batch
        self.trajectory_batch_size = max(1, min(16, configured_batch))
        self.runtime_info = {
            **diagnostics,
            "device": self.device,
            "device_name": device_name,
            "accelerator": accelerator,
            "dtype": self.dtype,
            "trajectory_batch_size": self.trajectory_batch_size,
        }

    @property
    def model(self):
        if self._model is None:
            from proticelli import Model

            self._model = Model(device=self.device, dtype=self.dtype)
        return self._model

    def _select_precision(self, dtype: str, progress: ProgressCallback) -> None:
        precision = dtype.strip().lower()
        if precision not in {"float16", "float32"}:
            raise ValueError("Inference precision must be float16 or float32")
        if precision == "float16" and not self.device.startswith("cuda"):
            raise ValueError("Float16 web inference requires CUDA or ROCm; use float32 on CPU/MPS")
        if precision == self.dtype:
            return
        had_loaded_model = self._model is not None
        if had_loaded_model:
            progress(0.02, f"Switching model precision to {precision}")
            previous = self._model
            self._model = None
            del previous
            gc.collect()
            if self.device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
            elif self.device == "mps":
                import torch

                empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
                if empty_cache is not None:
                    empty_cache()
        self.dtype = precision
        self.runtime_info["dtype"] = precision

    def generate(
        self,
        reference: np.ndarray,
        protein: str,
        cell_line: str | None,
        num_samples: int,
        num_inference_steps: int,
        seed: int,
        progress: ProgressCallback,
        cancel_check: Callable[[], bool] | None = None,
        dtype: str = "float32",
    ) -> InferenceOutput:
        self._select_precision(dtype, progress)
        progress(
            0.08,
            "Running ProtiCelli prediction"
            if num_samples == 1
            else f"Running {num_samples}-state ProtiCelli ensemble",
        )
        result = self.model.predict_with_reliability(
            images=[reference],
            protein_names=[protein],
            cell_line_names=[cell_line] if cell_line else None,
            num_samples=num_samples,
            num_inference_steps=num_inference_steps,
            batch_size=min(self.trajectory_batch_size, num_samples),
            seed=seed,
            return_ensembles=True,
            show_progress=False,
            cancel_check=cancel_check,
        )
        progress(0.90, "Finalizing ensemble statistics")
        if not result.ensembles:
            raise RuntimeError("ProtiCelli did not return the requested ensemble")
        ensemble = np.asarray(result.ensembles[0], dtype=np.float32)
        ref = normalize_reference_for_demo(reference)
        cell_mask = ref.max(axis=-1) > 0.1
        medoid, _ = compute_medoid_and_reliability(ensemble, cell_mask)
        reliability = float(result.reliability_scores[0])
        return InferenceOutput(list(ensemble), medoid, reliability, self.name)


def real_assets_ready(package_root: Path) -> bool:
    from proticelli.utils.download import assets_ready

    return assets_ready(package_root / "proticelli")


def create_engine(package_root: Path, requested_mode: str) -> InferenceEngine:
    mode = requested_mode.lower()
    if mode not in {"auto", "real", "demo"}:
        raise ValueError("PROTICELLI_WEB_MODE must be auto, real, or demo")
    ready = real_assets_ready(package_root)
    if mode == "real" and not ready:
        raise RuntimeError(
            "Real inference was requested, but checkpoint/unet and vae assets are missing. "
            "Run Model.download_checkpoints() or start with --mode demo."
        )
    if ready and mode != "demo":
        return RealInferenceEngine(package_root)
    return DemoInferenceEngine()
