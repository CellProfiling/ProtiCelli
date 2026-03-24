"""
ProtiCelli: Protein Visual Language Model for microscopy image generation.

Scikit-learn style API for predicting protein localization patterns
from reference microscopy images.

All model assets (checkpoints, VAE, label maps) live inside the
``proticelli/`` package directory so the user's working directory stays clean.

Example usage::

    from proticelli import ProtiCelli

    # Download checkpoints (first time only)
    ProtiCelli.download_checkpoints()

    # Initialize model
    model = ProtiCelli()

    # Predict protein localization
    results = model.predict(
        images=[img1, img2],
        protein_names=["ACTB", "TUBB"],
        cell_line_names=["U-2 OS", "A-431"],
    )
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from .utils.download import download_checkpoints

# Package root directory (where __init__.py lives)
_PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass
class PredictionResult:
    """Container for prediction outputs.

    Attributes:
        images: List of predicted protein channel images, each [H, W] float32 in [0, 1].
        latents: Optional raw latent tensors before decoding.
        metadata: Per-sample metadata (protein name, cell line, etc.).
    """
    images: List[np.ndarray]
    latents: Optional[List[np.ndarray]] = None
    metadata: List[Dict] = field(default_factory=list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

    def __iter__(self):
        return iter(self.images)

    def show_prediction(self):
        """Display all predicted images using matplotlib."""
        import matplotlib.pyplot as plt

        n = len(self.images)
        _, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for i, (ax, img) in enumerate(zip(axes, self.images)):
            img8 = (img * 255).clip(0, 255).astype(np.uint8)
            meta = self.metadata[i] if i < len(self.metadata) else {}
            ax.imshow(img8, cmap="gray")
            ax.set_title(
                f"{meta.get('cell_line_name', '')} / {meta.get('protein_name', '')}",
                fontsize=9,
            )
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def save_prediction(self, prefix: str = "", directory: str = "./"):
        """Save predicted images as TIFF files.

        Parameters
        ----------
        prefix : str
            Filename prefix. If non-empty, files are named
            ``{prefix}_{index}_{cell_line}_cell_{protein}.tif``.
        directory : str
            Output directory. Created if it does not exist.
        """
        from tifffile import imwrite

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(self.images):
            img8 = (img * 255).clip(0, 255).astype(np.uint8)
            meta = self.metadata[i] if i < len(self.metadata) else {}
            cell_line = (meta.get("cell_line_name") or "unknown").replace(" ", "_")
            protein = (meta.get("protein_name") or "unknown").replace(" ", "_")
            stem = f"{i}_{cell_line}_cell_{protein}"
            if prefix:
                stem = f"{prefix}_{stem}"
            imwrite(str(out_dir / f"{stem}.tif"), img8)


class ProtiCelli:
    """Protein Visual Language model for conditional protein image generation.

    Generates predicted protein localization images given:
    - Reference microscopy channels (nucleus, ER, microtubules)
    - A target protein/antibody name
    - Optionally, a cell line name

    All model assets live inside the ``proticelli/`` package directory:
    ``proticelli/checkpoint/``, ``proticelli/vae/``, ``proticelli/data/*.pkl``.

    Parameters
    ----------
    checkpoint_dir : str or Path, optional
        Path to model checkpoint. Default: ``proticelli/checkpoint/``.
    vae_dir : str or Path, optional
        Path to VAE checkpoint. Default: ``proticelli/vae/``.
    device : str, optional
        Device to run on. Default ``"cuda"`` if available, else ``"cpu"``.
    dtype : str, optional
        Weight precision: ``"float32"``, ``"float16"``, or ``"bfloat16"``.
    protein_map : str, Path, or dict, optional
        Protein label map. Default: ``proticelli/data/antibody_map.pkl``.
    cellline_map : str, Path, or dict, optional
        Cell line label map. Default: ``proticelli/data/cell_line_map.pkl``.

    Examples
    --------
    >>> ProtiCelli.download_checkpoints()   # first time only
    >>> model = ProtiCelli()
    >>> preds = model.predict(images, protein_names=["ACTB"])
    """

    # ------------------------------------------------------------------ #
    #  Construction helpers
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        checkpoint_dir: Union[str, Path, None] = None,
        vae_dir: Union[str, Path, None] = None,
        device: Optional[str] = None,
        dtype: str = "float32",
        protein_map: Union[str, Path, dict, None] = None,
        cellline_map: Union[str, Path, dict, None] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else _PACKAGE_DIR / "checkpoint"
        self.vae_dir = Path(vae_dir) if vae_dir else _PACKAGE_DIR / "vae"
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = _parse_dtype(dtype)

        # Lazy-loaded components
        self._model = None
        self._vae = None
        self._scheduler = None
        self._protein_map: Optional[Dict[str, int]] = None
        self._cellline_map: Optional[Dict[str, int]] = None

        # Resolve label maps
        self._protein_map_src = protein_map
        self._cellline_map_src = cellline_map

    # ---- Lazy loading ------------------------------------------------ #

    @property
    def model(self):
        """DiT transformer, loaded on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def vae(self):
        """VAE encoder/decoder, loaded on first access."""
        if self._vae is None:
            self._load_vae()
        return self._vae

    @property
    def scheduler(self):
        """EDM noise scheduler, loaded on first access."""
        if self._scheduler is None:
            self._load_scheduler()
        return self._scheduler

    @property
    def protein_map(self) -> Dict[str, int]:
        if self._protein_map is None:
            self._protein_map = self._resolve_map(
                self._protein_map_src, "antibody_map.pkl"
            )
        return self._protein_map

    @property
    def cellline_map(self) -> Dict[str, int]:
        if self._cellline_map is None:
            self._cellline_map = self._resolve_map(
                self._cellline_map_src, "cell_line_map.pkl"
            )
        return self._cellline_map

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def predict(
        self,
        images: Sequence[np.ndarray],
        protein_names: Sequence[str],
        cell_line_names: Optional[Sequence[str]] = None,
        *,
        num_inference_steps: int = 50,
        batch_size: int = 4,
        seed: Optional[int] = None,
        return_latents: bool = False,
        show_progress: bool = True,
    ) -> PredictionResult:
        """Generate predicted protein localization images.

        Parameters
        ----------
        images : list of np.ndarray
            Reference channel images. Each should be one of:
            - ``[H, W, 3]`` float32 array with 3 reference channels
              (nucleus, ER/microtubules channels), values in [0, 1].
            - ``[H, W, 4]`` float32 array (4-channel TIFF); channel 1 is
              ignored and channels 0, 2, 3 are used as conditioning.
        protein_names : list of str
            Target protein/antibody name for each image. Must be present
            in the model's protein vocabulary.
        cell_line_names : list of str, optional
            Cell line name for each image.  If ``None``, a default
            (index 0) is used for all samples.
        num_inference_steps : int
            Number of denoising steps (higher = better quality, slower).
        batch_size : int
            Inference batch size.
        seed : int, optional
            Random seed for reproducibility.
        return_latents : bool
            If True, include raw latent arrays in the result.
        show_progress : bool
            Show a progress bar.

        Returns
        -------
        PredictionResult
            Container with ``.images`` list of ``[H, W]`` float32 arrays.
        """
        # ---- Input validation ---------------------------------------- #
        images = list(images)
        protein_names = list(protein_names)
        n = len(images)
        if len(protein_names) != n:
            raise ValueError(
                f"len(images)={n} != len(protein_names)={len(protein_names)}"
            )
        if cell_line_names is not None:
            cell_line_names = list(cell_line_names)
            if len(cell_line_names) != n:
                raise ValueError(
                    f"len(images)={n} != len(cell_line_names)={len(cell_line_names)}"
                )
        else:
            cell_line_names = [None] * n

        # ---- Preprocess inputs --------------------------------------- #
        cond_tensors = []
        for img in images:
            cond_tensors.append(self._preprocess_image(img))

        protein_indices = []
        for name in protein_names:
            if name not in self.protein_map:
                raise KeyError(
                    f"Protein '{name}' not found in vocabulary "
                    f"({len(self.protein_map)} proteins available). "
                    f"Use model.protein_map.keys() to see valid names."
                )
            protein_indices.append(self.protein_map[name])

        cellline_indices = []
        for name in cell_line_names:
            if name is None:
                cellline_indices.append(0)
            elif name not in self.cellline_map:
                warnings.warn(
                    f"Cell line '{name}' not in vocabulary, using default (0)."
                )
                cellline_indices.append(0)
            else:
                cellline_indices.append(self.cellline_map[name])

        # ---- Run inference in batches -------------------------------- #
        from ._sampling import sample_edm

        all_images = []
        all_latents = []
        all_meta = []

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        num_batches = (n + batch_size - 1) // batch_size
        batch_iter = range(num_batches)

        for b in batch_iter:
            start = b * batch_size
            end = min(start + batch_size, n)
            bs = end - start

            cond_batch = torch.stack(cond_tensors[start:end]).to(
                self.device, dtype=self.dtype
            )
            prot_batch = torch.tensor(
                protein_indices[start:end], device=self.device, dtype=torch.long
            )
            cl_batch = torch.tensor(
                cellline_indices[start:end], device=self.device, dtype=torch.long
            )

            # Encode conditioning to latent space
            with torch.no_grad():
                ref_latents = (
                    self.vae.encode(cond_batch).latent_dist.sample().to(self.dtype)
                    * self.vae.config.scaling_factor / 4
                )

            latents = sample_edm(
                model=self.model,
                scheduler=self.scheduler,
                batch_size=bs,
                image_size=64,
                num_inference_steps=num_inference_steps,
                protein_labels=prot_batch,
                cell_line_labels=cl_batch,
                generator=generator,
                device=self.device,
                weight_dtype=self.dtype,
                reference_channels=ref_latents,
            )

            # Decode latents to images
            decoded = self._decode_latents(latents)

            for i in range(bs):
                pred_np = decoded[i].cpu().numpy()  # [1, H, W]
                pred_np = pred_np.squeeze(0)        # [H, W]
                all_images.append(pred_np)
                if return_latents:
                    all_latents.append(latents[i].cpu().numpy())
                all_meta.append({
                    "protein_name": protein_names[start + i],
                    "cell_line_name": cell_line_names[start + i],
                })

        return PredictionResult(
            images=all_images,
            latents=all_latents if return_latents else None,
            metadata=all_meta,
        )

    def fit(
        self,
        image_dir: Union[str, Path],
        image_files: Sequence[str],
        protein_names: Sequence[str],
        cell_line_names: Optional[Sequence[str]] = None,
        *,
        output_dir: str = "./proticelli_finetune",
        num_epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        resume_from: Optional[str] = None,
        **kwargs,
    ) -> "ProtiCelli":
        """Fine-tune the model on a new dataset.

        Images are loaded from disk on-the-fly to avoid memory issues
        with large datasets.

        Parameters
        ----------
        image_dir : str or Path
            Directory containing training TIFF images.  Each image is
            a 4-channel TIFF ``[H, W, 4]`` where channels are
            ``[nucleus, protein, ER, microtubules]``.  Channels 0, 2, 3
            are used as conditioning; channel 1 is the ground-truth
            protein target.
        image_files : list of str
            Filenames within ``image_dir``, e.g.
            ``["img1.tiff", "img2.tiff", ...]``.
        protein_names : list of str
            Target protein name per image, e.g.
            ``["CDT1", "CD8", ...]``.  Must have the same length as
            ``image_files``.
        cell_line_names : list of str, optional
            Cell line name per image, e.g.
            ``["U-2 OS", "A-431", ...]``.  If ``None``, defaults to
            label index 0 (unconditioned).
        output_dir : str
            Directory to save fine-tuned checkpoints.
        num_epochs : int
            Number of training epochs.
        batch_size : int
            Training batch size.
        learning_rate : float
            Peak learning rate.
        resume_from : str, optional
            Path to a checkpoint directory to resume from.
        **kwargs
            Additional training arguments (see ``proticelli._training``).

        Returns
        -------
        self
            The model instance (for method chaining).

        Examples
        --------
        >>> model.fit(
        ...     image_dir="./data/train",
        ...     image_files=["cell_0.tiff", "cell_1.tiff", "cell_2.tiff"],
        ...     protein_names=["CDT1", "CD8", "CTNNB1"],
        ...     cell_line_names=["U-2 OS", "U-2 OS", "A-431"],
        ...     output_dir="./finetuned",
        ...     num_epochs=50,
        ... )
        """
        from ._training import run_finetuning

        if len(image_files) != len(protein_names):
            raise ValueError(
                f"image_files ({len(image_files)}) and protein_names "
                f"({len(protein_names)}) must have the same length."
            )

        # Load EMA weights for training if not already loaded
        if self._model is None:
            self._load_model(use_ema=True)

        run_finetuning(
            model=self,
            image_dir=str(image_dir),
            image_files=list(image_files),
            protein_names=list(protein_names),
            cell_line_names=list(cell_line_names) if cell_line_names else None,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            resume_from=resume_from,
            **kwargs,
        )
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Save the full model (DiT + config + label maps) to a directory.

        Parameters
        ----------
        path : str or Path
            Destination directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save DiT
        self.model.save_pretrained(path / "unet")

        # Save label maps
        with open(path / "antibody_map.pkl", "wb") as f:
            pickle.dump(self.protein_map, f)
        with open(path / "cell_line_map.pkl", "wb") as f:
            pickle.dump(self.cellline_map, f)

        print(f"Model saved to {path}")

    @staticmethod
    def download_checkpoints(
        dest_dir: Union[str, Path, None] = None,
        *,
        checkpoint_url: str = "https://ell-vault.stanford.edu/dav/public/ProtiCelli/checkpoint.zip",
        vae_url: str = "http://ell-vault.stanford.edu/dav/public/ProtiCelli/vae.zip",
    ) -> dict:
        """Download pre-trained checkpoints into the package directory.

        Creates ``proticelli/checkpoint/`` and ``proticelli/vae/``.

        Parameters
        ----------
        dest_dir : str or Path, optional
            Override destination. Default: the ``proticelli/`` package directory.
        checkpoint_url : str
            URL to the model checkpoint zip.
        vae_url : str
            URL to the VAE checkpoint zip.

        Returns
        -------
        dict
            ``{"checkpoint_dir": ..., "vae_dir": ...}``.
        """
        dest = Path(dest_dir) if dest_dir else _PACKAGE_DIR
        return download_checkpoints(dest, checkpoint_url, vae_url)

    @property
    def available_proteins(self) -> List[str]:
        """List all protein names the model can predict."""
        return sorted(self.protein_map.keys())

    @property
    def available_cell_lines(self) -> List[str]:
        """List all cell line names the model recognizes."""
        return sorted(self.cellline_map.keys())

    def summary(self) -> str:
        """Return a human-readable model summary."""
        n_params = sum(p.numel() for p in self.model.parameters())
        lines = [
            "ProtiCelli Model Summary",
            f"  Parameters:    {n_params:,}",
            f"  Proteins:      {len(self.protein_map):,}",
            f"  Cell lines:    {len(self.cellline_map):,}",
            f"  Device:        {self.device}",
            f"  Dtype:         {self.dtype}",
            f"  Checkpoint:    {self.checkpoint_dir}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        loaded = "loaded" if self._model is not None else "not loaded"
        return f"ProtiCelli(checkpoint='{self.checkpoint_dir}', {loaded})"

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _load_model(self, use_ema=False):
        from .models.dit import DiTTransformer2DModel, create_dit_model

        unet_path = self.checkpoint_dir / "unet"
        unet_ema_path = self.checkpoint_dir / "unet_ema"

        if use_ema and unet_ema_path.exists():
            self._model = DiTTransformer2DModel.from_pretrained(
                str(self.checkpoint_dir), subfolder="unet_ema"
            )
        elif unet_path.exists():
            self._model = DiTTransformer2DModel.from_pretrained(
                str(self.checkpoint_dir), subfolder="unet"
            )
        else:
            # No checkpoint found, create model from scratch with default config
            self._model = create_dit_model(config=None, resolution=64)

        self._model.to(self.device, dtype=self.dtype)
        self._model.eval()
        self._model.requires_grad_(False)

    def _load_vae(self):
        from diffusers import AutoencoderKL

        self._vae = AutoencoderKL.from_pretrained(str(self.vae_dir))
        self._vae.to(self.device, dtype=self.dtype)
        self._vae.eval()
        self._vae.requires_grad_(False)

    def _load_scheduler(self):
        from .schedulers.edm_scheduler import create_edm_scheduler
        from .config.default_config import EDM_CONFIG

        self._scheduler = create_edm_scheduler(
            sigma_min=EDM_CONFIG["SIGMA_MIN"],
            sigma_max=EDM_CONFIG["SIGMA_MAX"],
            sigma_data=EDM_CONFIG["SIGMA_DATA"],
            num_train_timesteps=1000,
            prediction_type="sample",
        )
        self._scheduler.sigmas = self._scheduler.sigmas.to(self.device)

    def _resolve_map(self, src, default_filename: str) -> dict:
        if isinstance(src, dict):
            return src
        if src is not None:
            path = Path(src)
            if not path.exists():
                raise FileNotFoundError(f"Label map not found at {path}")
            with open(path, "rb") as f:
                return pickle.load(f)

        # Search inside the package: proticelli/data/ first, then checkpoint_dir
        candidates = [
            _PACKAGE_DIR / "data" / default_filename,
            self.checkpoint_dir / default_filename,
            self.checkpoint_dir.parent / default_filename,
        ]
        for path in candidates:
            if path.exists():
                with open(path, "rb") as f:
                    return pickle.load(f)

        raise FileNotFoundError(
            f"Label map '{default_filename}' not found. Searched:\n"
            + "\n".join(f"  - {p}" for p in candidates)
            + "\nProvide it via the constructor argument."
        )

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Convert a user-provided numpy image to a [3, H, W] tensor in [-1, 1]."""
        if img.ndim == 2:
            raise ValueError(
                f"Expected 3- or 4-channel image, got shape {img.shape}. "
                "Provide [H, W, 3] reference channels."
            )
        if img.ndim != 3:
            raise ValueError(f"Expected [H, W, C] image, got shape {img.shape}")

        if img.shape[2] == 4:
            # 4-channel TIFF: use channels 0, 2, 3 as conditioning
            img = img[:, :, [0, 2, 3]]
        elif img.shape[2] == 3:
            pass
        else:
            raise ValueError(
                f"Expected 3 or 4 channels, got {img.shape[2]}"
            )

        # Convert to float32 if needed
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32)

        # Normalize to [-1, 1]
        if img.min() >= 0 and img.max() <= 1.0:
            img = img * 2.0 - 1.0

        # To tensor [C, H, W]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return tensor

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor to images. Returns [B, 1, H, W] in [0, 1]."""
        vae = self.vae
        prev_dtype = vae.dtype
        vae.to(torch.float32)

        decoded_all = []
        for i in range(latents.shape[0]):
            sample = latents[i, :16, :, :].unsqueeze(0).to(torch.float32)
            sample = sample * 4 / vae.config.scaling_factor
            decoded = vae.decode(sample).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1.3)
            # Average across 3 decoded channels → 1 channel
            decoded = decoded.mean(dim=1, keepdim=True)
            decoded_all.append(decoded.squeeze(0))  # [1, H, W]

        vae.to(prev_dtype)
        return torch.stack(decoded_all)


def _parse_dtype(s: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }[s]
