"""Checkpoint save/load utilities for ProtVL training."""

import os
import shutil
from pathlib import Path

from diffusers.training_utils import EMAModel


def setup_checkpoint_hooks(accelerator, args, ema_model=None):
    """Register custom save/load hooks with Accelerate."""

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if args.use_ema and ema_model is not None:
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))
            for model in models:
                model.save_pretrained(os.path.join(output_dir, "unet"))
                weights.pop()

    def load_model_hook(models, input_dir):
        from .models.dit import DiTTransformer2DModel

        if args.use_ema and ema_model is not None:
            try:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), DiTTransformer2DModel
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model
            except Exception:
                print("No EMA model found in checkpoint")

        for _ in range(len(models)):
            model = models.pop()
            load_model = DiTTransformer2DModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    return accelerator


def save_checkpoint(accelerator, args, global_step, save_path=None):
    """Save a training checkpoint."""
    if not accelerator.is_main_process:
        return None
    if save_path is None:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    return save_path


def cleanup_checkpoints(output_dir, checkpoints_total_limit):
    """Remove old checkpoints exceeding the limit."""
    if not checkpoints_total_limit or checkpoints_total_limit <= 0:
        return
    checkpoints = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint")],
        key=lambda x: int(x.split("-")[1]),
    )
    if len(checkpoints) > checkpoints_total_limit:
        for ckpt in checkpoints[: len(checkpoints) - checkpoints_total_limit]:
            shutil.rmtree(os.path.join(output_dir, ckpt))


def find_latest_checkpoint(output_dir):
    """Find the most recent checkpoint directory."""
    dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if not dirs:
        return None
    return os.path.join(output_dir, sorted(dirs, key=lambda x: int(x.split("-")[1]))[-1])


def resume_from_checkpoint(accelerator, args, num_update_steps_per_epoch=0):
    """Resume training from a checkpoint. Returns (global_step, first_epoch, resume_step)."""
    global_step = first_epoch = resume_step = 0

    if args.resume_from_checkpoint is None:
        return global_step, first_epoch, resume_step

    path = (
        find_latest_checkpoint(args.output_dir)
        if args.resume_from_checkpoint == "latest"
        else args.resume_from_checkpoint
    )
    if path is None:
        return global_step, first_epoch, resume_step

    accelerator.load_state(path)

    if os.path.isdir(path):
        global_step = int(path.split("-")[-1])
        if num_update_steps_per_epoch > 0:
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch

    return global_step, first_epoch, resume_step
