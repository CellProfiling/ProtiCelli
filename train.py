"""Training script for conditional cell generation."""
import argparse
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
import accelerate
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration
from packaging import version
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMPipeline
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.utils import is_tensorboard_available, is_wandb_available

from config.config import EDM_CONFIG
from data.dataset import FullFieldDataset
from models.dit import create_dit_model
from schedulers.edm_scheduler import create_edm_scheduler
from utils.logging_utils import setup_logging, configure_diffusers_logging, log_training_parameters
from utils.checkpoint_utils import setup_checkpoint_hooks, save_checkpoint, cleanup_checkpoints, resume_from_checkpoint
from utils.edm_utils import (
    edm_clean_image_to_model_input,
    edm_model_output_to_x_0_hat,
    edm_loss_weight,
    prepare_latent_sample,
    prepare_model_inputs,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DiT model for cell generation.")

    # Data paths
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for training data.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV file.")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV file.")
    parser.add_argument("--cellline_map", type=str, required=True, help="Path to cell line label dict.")
    parser.add_argument("--antibody_map", type=str, required=True, help="Path to antibody/protein dict.")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to VAE model.")

    # Output
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory.")

    # Training
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--label_dropout_prob", type=float, default=0.2, help="Label dropout for CFG.")

    # Optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    # LR scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)

    # EMA
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=0.75)
    parser.add_argument("--ema_max_decay", type=float, default=0.9995)

    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--finetune", action="store_true", 
                   help="Load model weights only, reset training state for new dataset")

    # Logging
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])

    # Misc
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--prediction_type", type=str, default="epsilon")
    parser.add_argument("--resolution", type=int, default=64)

    return parser.parse_args()


def load_vae(vae_path, accelerator, weight_dtype):
    """Load VAE model."""
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    return vae



def main():
    """Main training function."""
    args = parse_args()

    # Get EDM config
    sigma_min = EDM_CONFIG.sigma_min
    sigma_max = EDM_CONFIG.sigma_max
    sigma_data = EDM_CONFIG.sigma_data

    # Setup logging
    logger = setup_logging(__name__)

    # Setup accelerator
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Validate logger availability
    if args.logger == "tensorboard" and not is_tensorboard_available():
        raise ImportError("Install tensorboard for logging.")
    elif args.logger == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb for logging.")

    logger.info(accelerator.state, main_process_only=False)
    configure_diffusers_logging(accelerator.is_local_main_process)

    # Create output directory
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Set weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Create DiT model
    model = create_dit_model(resolution=args.resolution)
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Create EMA model
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=type(model),
            model_config=model.config,
        )

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        accelerator = setup_checkpoint_hooks(accelerator, args, ema_model)

    # Create scheduler
    noise_scheduler = create_edm_scheduler(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
        num_train_timesteps=args.ddpm_num_steps,
        prediction_type=args.prediction_type,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Create dataset
    dataset = FullFieldDataset(
        data_root=args.data_root,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        cellline_map=args.cellline_map,
        antibody_map=args.antibody_map,
        is_train=True,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=False,
    )

    # Create LR scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    # Load VAE and CLIP
    vae = load_vae(args.vae_path, accelerator, weight_dtype)

    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)
        accelerator.ema_model = ema_model

    noise_scheduler.sigmas = noise_scheduler.sigmas.to(accelerator.device)

    # Initialize trackers
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    # Calculate training parameters
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    log_training_parameters(logger, args, total_batch_size, num_update_steps_per_epoch, max_train_steps, len(dataset))

    # Full checkpoint resumption (same dataset)
    global_step, first_epoch, resume_step = resume_from_checkpoint(
        accelerator, args, num_update_steps_per_epoch
    )
    if args.finetune:
        global_step = 0
        first_epoch = 0
        resume_step = 0



    # Training loop
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            # Skip to resume step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # Encode images to latent space
            with torch.no_grad():
                cond_images = batch["cond_image"].to(weight_dtype)
                cond_images_latent = prepare_latent_sample(vae, cond_images, weight_dtype)

                gt_images = batch["gt_image"].repeat(1, 3, 1, 1).to(weight_dtype)
                gt_images_latent = prepare_latent_sample(vae, gt_images, weight_dtype)

                # Prepare inputs with label dropout
                (clean_images, cond_latents), (protein_label, cellline_label), encoder_hidden_states, _ = prepare_model_inputs(
                    gt_images_latent,
                    cond_images_latent,
                    batch["cell_line"],
                    batch["label"],
                    dropout_prob=args.label_dropout_prob,
                    weight_dtype=weight_dtype,
                    encoder_hidden_states=None,
                )

            # Sample noise and timesteps
            noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device).long()

            # Get sigmas
            sigmas = noise_scheduler.sigmas[timesteps].reshape(-1, 1, 1, 1).to(weight_dtype)
            x_noisy = noise * sigmas + clean_images

            with accelerator.accumulate(model):
                # Precondition input
                model_input, timestep_input = edm_clean_image_to_model_input(x_noisy, sigmas)
                timestep_input = timestep_input.squeeze()

                protein_label = protein_label.reshape(-1)
                cellline_label = cellline_label.reshape(-1)

                # Concatenate with conditioning
                model_input_concat = torch.cat([model_input, cond_latents], dim=1)

                # Model forward pass with labels
                model_output = model(
                    model_input_concat,
                    timestep_input,
                    protein_labels=protein_label,
                    cell_line_labels=cellline_label,
                    encoder_hidden_states=None,
                ).sample

                # Get denoised prediction
                x_0_hat = edm_model_output_to_x_0_hat(x_noisy, sigmas, model_output)

                # Calculate loss
                target = clean_images
                weights = edm_loss_weight(sigmas)
                loss = weights * ((x_0_hat.float() - target.float()) ** 2)
                loss = loss.mean()

                # Backward pass
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update EMA
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                # Save checkpoint
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    cleanup_checkpoints(args.output_dir, args.checkpoints_total_limit)
                    save_path = save_checkpoint(accelerator, args, global_step)
                    logger.info(f"Saved state to {save_path}")

            # Log metrics
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.close()
        accelerator.wait_for_everyone()

        # Save model at end of epoch
        if accelerator.is_main_process:
            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
                pipeline.save_pretrained(args.output_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

    accelerator.end_training()


if __name__ == "__main__":
    main()
