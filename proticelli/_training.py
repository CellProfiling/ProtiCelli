"""Fine-tuning with explicit intensity ranges and resumable optimizer state."""

import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tifffile import imread
from tqdm.auto import tqdm

from .data.preprocessing import _normalize_image_range
from .utils.edm_utils import (
    edm_clean_image_to_model_input, edm_model_output_to_x_0_hat, edm_loss_weight,
)
from ._training_state import (
    resolve_resume_path, restore_rng, update_ema, save_training_checkpoint,
)


class FinetuneDataset(Dataset):
    """Load HWC TIFFs ordered [microtubules, protein, nucleus, ER].

    Conditioning uses channels 0, 2, 3; channel 1 supplies the protein target.
    Integer scale comes from acquisition dtype or an explicit bit_depth, never
    observed brightness. Specify input_range='normalized' for prepared float
    data on the nominal [-1,1] scale, especially for all-positive images.
    Finite overshoot in previously prepared data is preserved.
    """

    def __init__(self, image_dir: str, image_files: List[str],
                 protein_labels: List[int], cellline_labels: List[int], *,
                 input_range="auto", bit_depth=None):
        if not (len(image_files) == len(protein_labels) == len(cellline_labels)):
            raise ValueError("Image files and labels must have matching lengths")
        self.image_dir = image_dir
        self.image_files = image_files
        self.protein_labels = protein_labels
        self.cellline_labels = cellline_labels
        self.input_range = input_range
        self.bit_depth = bit_depth

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = np.asarray(imread(os.path.join(self.image_dir, self.image_files[idx])))
        if img.ndim != 3 or img.shape[-1] != 4:
            raise ValueError(f"{self.image_files[idx]}: fine-tuning requires [H, W, 4] TIFFs")
        img = _normalize_image_range(img, input_range=self.input_range, bit_depth=self.bit_depth)
        return {"cond_image": torch.from_numpy(img[:, :, [0, 2, 3]]).permute(2, 0, 1).float(),
                "gt_image": torch.from_numpy(img[:, :, 1:2]).permute(2, 0, 1).float(),
                "label": self.protein_labels[idx], "cell_line": self.cellline_labels[idx]}


def _positive_integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def run_finetuning(
    model, image_dir: str, image_files: List[str], protein_names: List[str],
    cell_line_names: Optional[List[str]] = None, *, num_epochs: int = 100,
    batch_size: int = 16, learning_rate: float = 1e-4,
    output_dir: str = "./proticelli_finetune", resume_from: Optional[str] = None,
    label_dropout_prob: float = 0.2, lr_scheduler_type: str = "cosine",
    lr_warmup_steps: int = 500, gradient_accumulation_steps: int = 1,
    checkpointing_steps: int = 500, save_model_epochs: int = 10,
    max_grad_norm: float = 1.0, adam_beta1: float = 0.95, adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-6, adam_epsilon: float = 1e-8,
    use_ema: bool = False, mixed_precision: str = "no", num_workers: int = 4,
    ema_decay: float = 0.999, input_range="auto", bit_depth=None, seed=None,
):
    """Train with FP32 master parameters, sample-mean accumulation, EMA and resume.

    Checkpoints contain raw/optional EMA exports plus training_state.pt with
    model, optimizer, LR scheduler, scaler, RNG and next-batch cursor. Resume
    requires the same data and update configuration. Extending num_epochs is
    allowed; the LR schedule then uses the newly requested total epoch count.
    fp16 autocast/scaling requires CUDA; bf16 autocast supports CPU and CUDA.
    """
    resume_path = resolve_resume_path(resume_from)
    for name, value, minimum in (("num_epochs", num_epochs, 1), ("batch_size", batch_size, 1),
                                 ("gradient_accumulation_steps", gradient_accumulation_steps, 1),
                                 ("checkpointing_steps", checkpointing_steps, 1),
                                 ("save_model_epochs", save_model_epochs, 1),
                                 ("lr_warmup_steps", lr_warmup_steps, 0), ("num_workers", num_workers, 0)):
        _positive_integer(name, value, minimum)
    if seed is not None:
        _positive_integer("seed", seed, 0)
        if seed >= 2 ** 63:
            raise ValueError("seed must be less than 2**63")
    for name, value in (("learning_rate", learning_rate), ("max_grad_norm", max_grad_norm),
                        ("adam_epsilon", adam_epsilon)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (("label_dropout_prob", label_dropout_prob), ("ema_decay", ema_decay),
                        ("adam_beta1", adam_beta1), ("adam_beta2", adam_beta2)):
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must lie in [0, 1)")
    if not math.isfinite(adam_weight_decay) or adam_weight_decay < 0:
        raise ValueError("adam_weight_decay must be finite and nonnegative")
    if mixed_precision not in ("no", "fp16", "bf16"):
        raise ValueError("mixed_precision must be no, fp16, or bf16")
    device = torch.device(model.device)
    if mixed_precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA; use bf16 or no on CPU")
    if mixed_precision != "no" and device.type not in ("cpu", "cuda"):
        raise ValueError("Autocast training is supported on CPU and CUDA")
    n = len(image_files)
    if n == 0 or len(protein_names) != n:
        raise ValueError("Provide nonempty image files with one protein name per image")
    if cell_line_names is not None and len(cell_line_names) != n:
        raise ValueError("Provide one cell line name per image")

    from diffusers.optimization import get_scheduler as get_lr_scheduler
    from .config.default_config import EDM_CONFIG
    from .schedulers.edm_scheduler import create_edm_scheduler

    protein_labels = [model.protein_map[model._resolve_protein_name(name)] for name in protein_names]
    if cell_line_names is None:
        cellline_labels = [0] * n
    else:
        from .model import _resolve_cell_line
        lower = {name.lower(): name for name in model.cellline_map}
        cellline_labels = [model.cellline_map.get(_resolve_cell_line(name, model.cellline_map, lower), 0)
                           if name is not None else 0 for name in cell_line_names]
    dataset = FinetuneDataset(image_dir, image_files, protein_labels, cellline_labels,
                              input_range=input_range, bit_depth=bit_depth)
    import sys
    num_workers = 0 if sys.platform == "win32" else min(num_workers, 4)
    full_batches = math.ceil(n / batch_size)
    updates_per_epoch = math.ceil(full_batches / gradient_accumulation_steps)
    configuration = {"image_dir": str(Path(image_dir).resolve()), "image_files": list(image_files),
                     "protein_labels": protein_labels, "cellline_labels": cellline_labels,
                     "batch_size": int(batch_size), "gradient_accumulation_steps": int(gradient_accumulation_steps),
                     "learning_rate": learning_rate, "lr_scheduler_type": lr_scheduler_type,
                     "lr_warmup_steps": int(lr_warmup_steps), "label_dropout_prob": label_dropout_prob,
                     "max_grad_norm": max_grad_norm, "adam_beta1": adam_beta1, "adam_beta2": adam_beta2,
                     "adam_weight_decay": adam_weight_decay, "adam_epsilon": adam_epsilon,
                     "use_ema": bool(use_ema), "ema_decay": ema_decay, "mixed_precision": mixed_precision,
                     "input_range": input_range, "bit_depth": bit_depth}
    state = None
    if resume_path is not None:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state.get("version") != 1 or state.get("configuration") != configuration:
            raise ValueError("Resume checkpoint does not match the data or training configuration")
    data_seed = int(state["data_seed"] if state is not None else
                    (torch.initial_seed() if seed is None else seed)) % (2 ** 63)
    if state is None and seed is not None:
        torch.random.default_generator.manual_seed(data_seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(data_seed)
        elif device.type == "mps" and hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(data_seed)

    dit, vae = model.model, model.vae
    vae_dtype = getattr(vae, "dtype", model.dtype)
    try:
        dit.to(device=device, dtype=torch.float32).train().requires_grad_(True)
        vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(dit.parameters(), lr=learning_rate,
                                     betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                                     eps=adam_epsilon)
        scheduler = get_lr_scheduler(lr_scheduler_type, optimizer=optimizer,
                                     num_warmup_steps=lr_warmup_steps,
                                     num_training_steps=num_epochs * updates_per_epoch)
        # torch.amp.GradScaler is absent in the oldest supported Torch versions.
        modern_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
        scaler = (modern_scaler("cuda", enabled=mixed_precision == "fp16") if modern_scaler is not None
                  else torch.cuda.amp.GradScaler(enabled=mixed_precision == "fp16"))
        noise_scheduler = create_edm_scheduler(
            sigma_min=EDM_CONFIG["SIGMA_MIN"], sigma_max=EDM_CONFIG["SIGMA_MAX"],
            sigma_data=EDM_CONFIG["SIGMA_DATA"], num_train_timesteps=1000, prediction_type="epsilon")
        sigmas_all = noise_scheduler.sigmas.to(device=device, dtype=torch.float32)
        # Resume loads its saved EMA directly; allocating a fresh full shadow
        # first would temporarily retain two shadows on the accelerator.
        ema = ({name: value.detach().clone() for name, value in dit.state_dict().items()}
               if use_ema and state is None else None)
        epoch_start, next_batch, global_step = 0, 0, 0
        if state is not None:
            dit.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["lr_scheduler"])
            scaler.load_state_dict(state["scaler"])
            if use_ema:
                if state["ema"] is None:
                    raise ValueError("Resume checkpoint has no EMA state")
                ema = {name: value.to(device) for name, value in state["ema"].items()}
            epoch_start, next_batch = int(state["epoch"]), int(state["next_batch"])
            global_step = int(state["global_step"])
            if not 0 <= next_batch < full_batches or epoch_start < 0:
                raise ValueError("Invalid training checkpoint cursor")
            restore_rng(state["rng"], device)
            del state
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        optimizer.zero_grad(set_to_none=True)
        amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
        print(f"Fine-tuning {n} images for {num_epochs} epochs; {updates_per_epoch} updates/epoch")

        def save(path, epoch, batch):
            save_training_checkpoint(path, model, dit, optimizer, scheduler, scaler, ema,
                                     epoch, batch, global_step, data_seed, configuration)

        for epoch in range(epoch_start, num_epochs):
            start_batch = next_batch if epoch == epoch_start else 0
            permutation = torch.randperm(n, generator=torch.Generator().manual_seed((data_seed + epoch) % (2 ** 63)))
            subset = Subset(dataset, permutation.tolist()[start_batch * batch_size:])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=device.type == "cuda",
                                generator=torch.Generator().manual_seed((data_seed + epoch) % (2 ** 63)))
            epoch_loss, samples_seen = 0.0, 0
            for batch_index, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", leave=False), start_batch):
                cond = batch["cond_image"].to(device, dtype=torch.float32)
                gt = batch["gt_image"].repeat(1, 3, 1, 1).to(device, dtype=torch.float32)
                protein = batch["label"].to(device, dtype=torch.long)
                cellline = batch["cell_line"].to(device, dtype=torch.long)
                with torch.no_grad():
                    condition = vae.encode(cond).latent_dist.sample() * vae.config.scaling_factor / 4
                    clean = vae.encode(gt).latent_dist.sample() * vae.config.scaling_factor / 4
                clean, condition = clean.float(), condition.float()
                keep = (torch.rand(protein.shape, device=device) > label_dropout_prob).long()
                protein, cellline = protein * keep, cellline * keep
                noise = torch.randn_like(clean)
                steps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (len(clean),), device=device)
                sigmas = sigmas_all[steps].reshape(-1, 1, 1, 1)
                noisy = noise * sigmas + clean
                model_input, timestep = edm_clean_image_to_model_input(noisy, sigmas)
                autocast = (torch.autocast(device_type=device.type, dtype=amp_dtype)
                            if mixed_precision != "no" else nullcontext())
                with autocast:
                    prediction = dit(torch.cat([model_input, condition], dim=1), timestep.reshape(-1),
                                     protein_labels=protein.reshape(-1), cell_line_labels=cellline.reshape(-1),
                                     encoder_hidden_states=None).sample
                estimate = edm_model_output_to_x_0_hat(noisy, sigmas, prediction.float())
                loss = (edm_loss_weight(sigmas) * (estimate.float() - clean).square()).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite fine-tuning loss")
                group_start = (batch_index // gradient_accumulation_steps) * gradient_accumulation_steps
                group_end = min(group_start + gradient_accumulation_steps, full_batches)
                group_samples = min(n, group_end * batch_size) - group_start * batch_size
                scaled_loss = loss * (len(clean) / group_samples)
                scaler.scale(scaled_loss).backward()
                epoch_loss += loss.item() * len(clean)
                samples_seen += len(clean)
                if batch_index + 1 == group_end:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(dit.parameters(), max_grad_norm,
                                                  error_if_nonfinite=not scaler.is_enabled())
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    updated = scaler.get_scale() >= previous_scale
                    optimizer.zero_grad(set_to_none=True)
                    if updated:
                        scheduler.step()
                        global_step += 1
                        if ema is not None:
                            update_ema(ema, dit, ema_decay)
                        following_epoch, following_batch = epoch, batch_index + 1
                        if following_batch == full_batches:
                            following_epoch, following_batch = epoch + 1, 0
                        if global_step % checkpointing_steps == 0:
                            save(output / f"checkpoint-{global_step}", following_epoch, following_batch)
            print(f"Epoch {epoch}: avg_loss={epoch_loss / max(samples_seen, 1):.6f}")
            if (epoch + 1) % save_model_epochs == 0 or epoch == num_epochs - 1:
                save(output, epoch + 1, 0)
        if ema is not None:
            dit.load_state_dict(ema)
        model.checkpoint_dir = output
        print(f"Fine-tuning complete. Model saved to {output}")
    finally:
        dit.eval().requires_grad_(False).to(device=device, dtype=model.dtype)
        vae.to(device=device, dtype=vae_dtype)
