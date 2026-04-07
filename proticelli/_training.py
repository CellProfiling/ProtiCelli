"""Fine-tuning logic for Model.fit().

Images are loaded from disk on-the-fly to handle large datasets
without running out of memory.
"""

import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tifffile import imread
from tqdm.auto import tqdm

from .utils.edm_utils import (
    edm_clean_image_to_model_input,
    edm_model_output_to_x_0_hat,
    edm_loss_weight,
)


class FinetuneDataset(Dataset):
    """Loads 4-channel TIFFs from disk for fine-tuning.

    Each TIFF is ``[H, W, 4]`` with channels ``[nucleus, protein, ER,
    microtubules]``.  Channels 0, 2, 3 become the 3-channel
    conditioning input; channel 1 is the ground-truth protein target.

    Parameters
    ----------
    image_dir : str
        Directory containing the TIFF files.
    image_files : list of str
        Filenames within ``image_dir``.
    protein_labels : list of int
        Integer protein label per image.
    cellline_labels : list of int
        Integer cell line label per image.
    """

    def __init__(
        self,
        image_dir: str,
        image_files: List[str],
        protein_labels: List[int],
        cellline_labels: List[int],
    ):
        self.image_dir = image_dir
        self.image_files = image_files
        self.protein_labels = protein_labels
        self.cellline_labels = cellline_labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.image_files[idx])
        img = imread(path).astype(np.float32)

        # Handle [H, W, 4] TIFF: ch0=nucleus, ch1=protein(GT), ch2=ER, ch3=MT
        if img.ndim == 3 and img.shape[-1] == 4:
            # Normalize to [-1, 1].
            # Three cases: raw uint8/uint16, already [0, 1], already [-1, 1].
            if img.max() > 1.0:
                img = img / 255.0 if img.max() <= 255 else img / 65535.0
                img = img * 2.0 - 1.0
            elif img.min() >= 0.0:
                # Already in [0, 1] — only rescale, don't divide again
                img = img * 2.0 - 1.0
            # else already in [-1, 1] (e.g. from assemble_and_normalize)

            cond = img[:, :, [0, 2, 3]]   # [H, W, 3]
            gt = img[:, :, 1:2]            # [H, W, 1]

        elif img.ndim == 3 and img.shape[-1] == 3:
            raise ValueError(
                f"{self.image_files[idx]}: 3-channel image has no protein "
                f"ground truth. Fine-tuning requires 4-channel TIFFs."
            )
        else:
            raise ValueError(
                f"{self.image_files[idx]}: unexpected shape {img.shape}. "
                f"Expected [H, W, 4]."
            )

        # [H, W, C] -> [C, H, W]
        cond_tensor = torch.from_numpy(cond).permute(2, 0, 1).float()
        gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).float()

        return {
            "cond_image": cond_tensor,
            "gt_image": gt_tensor,
            "label": self.protein_labels[idx],
            "cell_line": self.cellline_labels[idx],
        }


def run_finetuning(
    model,  # Model instance
    image_dir: str,
    image_files: List[str],
    protein_names: List[str],
    cell_line_names: Optional[List[str]] = None,
    *,
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    output_dir: str = "./proticelli_finetune",
    resume_from: Optional[str] = None,
    label_dropout_prob: float = 0.2,
    lr_scheduler_type: str = "cosine",
    lr_warmup_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    checkpointing_steps: int = 500,
    save_model_epochs: int = 10,
    max_grad_norm: float = 1.0,
    adam_beta1: float = 0.95,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-6,
    adam_epsilon: float = 1e-8,
    use_ema: bool = False,
    mixed_precision: str = "no",
    num_workers: int = 4,
):
    """Run fine-tuning loop.

    Called by ``Model.fit()``.  Loads images from ``image_dir`` via a
    PyTorch ``DataLoader`` so only one batch is in memory at a time.
    """
    from diffusers.optimization import get_scheduler as get_lr_scheduler
    from .config.default_config import EDM_CONFIG
    from .schedulers.edm_scheduler import create_edm_scheduler

    device = model.device
    dtype = model.dtype
    n = len(image_files)

    # ---- Resolve labels ----------------------------------------------- #
    protein_labels = []
    for name in protein_names:
        key = model._resolve_protein_name(name)
        protein_labels.append(model.protein_map[key])

    if cell_line_names is None:
        cellline_labels = [0] * n
    else:
        if len(cell_line_names) != n:
            raise ValueError(
                f"cell_line_names ({len(cell_line_names)}) must match "
                f"image_files ({n})."
            )
        cellline_labels = [
            model.cellline_map.get(name, 0) for name in cell_line_names
        ]

    # ---- Dataset & DataLoader ----------------------------------------- #
    dataset = FinetuneDataset(
        image_dir=image_dir,
        image_files=image_files,
        protein_labels=protein_labels,
        cellline_labels=cellline_labels,
    )
    import sys
    if sys.platform == "win32":
        num_workers = 0
    else:
        num_workers = min(num_workers, 4)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---- Setup model for training ------------------------------------- #
    dit = model.model  # triggers lazy load
    dit.train()
    dit.requires_grad_(True)

    vae = model.vae
    vae.eval()
    vae.requires_grad_(False)

    noise_scheduler = create_edm_scheduler(
        sigma_min=EDM_CONFIG["SIGMA_MIN"],
        sigma_max=EDM_CONFIG["SIGMA_MAX"],
        sigma_data=EDM_CONFIG["SIGMA_DATA"],
        num_train_timesteps=1000,
        prediction_type="epsilon",
    )
    noise_scheduler.sigmas = noise_scheduler.sigmas.to(device)

    # ---- Optimizer & LR scheduler ------------------------------------- #
    optimizer = torch.optim.AdamW(
        dit.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / gradient_accumulation_steps
    )
    max_train_steps = num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_lr_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps,
    )

    os.makedirs(output_dir, exist_ok=True)

    # ---- Training loop ------------------------------------------------ #
    global_step = 0
    print(
        f"Fine-tuning: {n} images, {num_epochs} epochs, "
        f"{max_train_steps} steps, batch_size={batch_size}"
    )

    for epoch in range(num_epochs):
        dit.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

        for step, batch in enumerate(pbar):
            cond_imgs = batch["cond_image"].to(device, dtype=dtype)
            gt_imgs = batch["gt_image"].repeat(1, 3, 1, 1).to(device, dtype=dtype)
            protein_label = batch["label"].to(device, dtype=torch.long)
            cellline_label = batch["cell_line"].to(device, dtype=torch.long)

            with torch.no_grad():
                cond_latents = vae.encode(cond_imgs).latent_dist.sample()
                cond_latents = cond_latents * vae.config.scaling_factor / 4
                gt_latents = vae.encode(gt_imgs).latent_dist.sample()
                gt_latents = gt_latents * vae.config.scaling_factor / 4

            clean_images = gt_latents.to(dtype)

            # Label dropout for classifier-free guidance
            dropout_mask = (
                torch.rand(protein_label.shape, device=device) > label_dropout_prob
            )
            protein_label = protein_label * dropout_mask.long()
            cellline_label = cellline_label * dropout_mask.long()

            # Sample noise
            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,),
                device=device,
            ).long()

            sigmas = (
                noise_scheduler.sigmas[timesteps]
                .reshape(-1, 1, 1, 1)
                .to(dtype)
            )
            x_noisy = noise * sigmas + clean_images

            # Forward
            model_input, timestep_input = edm_clean_image_to_model_input(
                x_noisy, sigmas
            )
            timestep_input = timestep_input.squeeze()

            model_input_concat = torch.cat([model_input, cond_latents], dim=1)

            model_output = dit(
                model_input_concat,
                timestep_input,
                protein_labels=protein_label.reshape(-1),
                cell_line_labels=cellline_label.reshape(-1),
                encoder_hidden_states=None,
            ).sample

            x_0_hat = edm_model_output_to_x_0_hat(x_noisy, sigmas, model_output)

            weights = edm_loss_weight(sigmas)
            loss = (
                weights * (x_0_hat.float() - clean_images.float()) ** 2
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(dit.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

            # Periodic checkpoint
            if global_step % checkpointing_steps == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_path, exist_ok=True)
                dit.save_pretrained(os.path.join(ckpt_path, "unet_ema"))
                print(f"  Saved checkpoint to {ckpt_path}")

        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f}")

        if epoch % save_model_epochs == 0 or epoch == num_epochs - 1:
            dit.save_pretrained(os.path.join(output_dir, "unet_ema"))

    # Restore eval mode
    dit.eval()
    dit.requires_grad_(False)

    # Update the Model instance
    model.checkpoint_dir = Path(output_dir)
    print(f"Fine-tuning complete. Model saved to {output_dir}")
