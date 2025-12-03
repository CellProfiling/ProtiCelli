import os
import argparse

import torch
from torch.distributions.normal import Normal
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from diffusers import AutoencoderKL
from accelerate import Accelerator
from accelerate.utils import set_seed

import math
import numpy as np
from tifffile import imread, imwrite
import pandas as pd
from scipy.ndimage import rotate

import json
from datetime import datetime 
from collections import defaultdict
import gc
from tqdm.auto import tqdm

import pickle


from schedulers.edm_scheduler import create_edm_scheduler
from utils.edm_utils import edm_clean_image_to_model_input, edm_model_output_to_x_0_hat
from config.default_config import EDM_CONFIG
from models.dit import create_dit_model, DiTTransformer2DModel


### LOGGER ###
class SynchronizedLogger:
    """Logger that collects metrics from all processes and writes them sequentially on process 0"""
    
    def __init__(self, log_dir, accelerator):
        self.accelerator = accelerator
        self.log_dir = log_dir
        self.metrics_buffer = []
        self.batch_metrics = defaultdict(list)
        
        if accelerator.is_main_process:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(log_dir, f"inference_log_{timestamp}.txt")
            self.metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.json")
            
            with open(self.log_file, 'w') as f:
                f.write(f"Inference Log - {accelerator.num_processes} GPUs\n")
                f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
    
    def log(self, metrics_dict, batch_idx=None):
        """Buffer metrics for later synchronization"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'process_index': self.accelerator.process_index,
            'batch_idx': batch_idx,
            'metrics': metrics_dict.copy()
        }
        self.metrics_buffer.append(log_entry)
        
        # If batch_idx is provided, also store in batch_metrics for easy aggregation
        if batch_idx is not None:
            self.batch_metrics[batch_idx].append(log_entry)
    
    def sync_and_log_batch(self, batch_idx):
        """Synchronize and log metrics for a specific batch across all processes"""
        # Gather metrics from all processes for this batch
        batch_data = self.batch_metrics.get(batch_idx, [])
        all_batch_data = self.accelerator.gather_for_metrics(batch_data)
        
        if self.accelerator.is_main_process:
            # Sort by process index for consistent ordering
            sorted_data = sorted(all_batch_data, key=lambda x: x.get('process_index', 0))
            
            with open(self.log_file, 'a') as f:
                f.write(f"=== Batch {batch_idx} Results ===\n")
                for entry in sorted_data:
                    f.write(f"[Process {entry['process_index']}] [{entry['timestamp']}] ")
                    for key, value in entry['metrics'].items():
                        f.write(f"{key}: {value} | ")
                    f.write("\n")
                f.write("\n")
            
            # Clear processed batch data
            if batch_idx in self.batch_metrics:
                del self.batch_metrics[batch_idx]
    
    def sync_and_log_all(self):
        """Synchronize all buffered metrics and write them to file"""
        # Gather all metrics from all processes
        all_metrics = self.accelerator.gather_for_metrics(self.metrics_buffer)
        
        if self.accelerator.is_main_process:
            # Sort by timestamp for chronological order
            sorted_metrics = sorted(all_metrics, key=lambda x: x['timestamp'])
            
            # Write to text log
            with open(self.log_file, 'a') as f:
                f.write("\n=== Complete Metrics Log ===\n")
                for entry in sorted_metrics:
                    f.write(f"[Process {entry['process_index']}] [{entry['timestamp']}] ")
                    if entry['batch_idx'] is not None:
                        f.write(f"[Batch {entry['batch_idx']}] ")
                    for key, value in entry['metrics'].items():
                        f.write(f"{key}: {value} | ")
                    f.write("\n")
            
            # Also save as JSON for easy parsing
            with open(self.metrics_file, 'w') as f:
                json.dump(sorted_metrics, f, indent=2, default=str)
        
        # Clear buffer after syncing
        self.metrics_buffer.clear()
    
    def finish(self):
        """Final synchronization and cleanup"""
        self.sync_and_log_all()
        
        if self.accelerator.is_main_process:
            with open(self.log_file, 'a') as f:
                f.write(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


### CLASSIFIER & VAE ###
class LocationClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, model_type='vit_small_patch16_224'):
        super().__init__()
        
        self.model_type = model_type
        
        # Load pretrained ViT from timm
        self.vit = timm.create_model(
            model_type, 
            pretrained=pretrained, 
            num_classes=num_classes,
            in_chans=16  # timm supports direct specification of input channels
        )
        
        # Model expects 224x224 input, so add an adapter for 32x32
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        x1 = self.resize(x)  # Resize to [batch_size, 32, 224, 224]
        x1 = self.vit(x1)
        return x1


def load_classifier(checkpoint_path, weight_dtype):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint['config']
    
    classifier = LocationClassifier(
        num_classes=config['num_classes'],
        pretrained=False,
        model_type=config['model_type']
    )
    
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    classifier.to(weight_dtype)
    classifier.requires_grad_(False)
    
    return classifier


def load_vae(vae_path, weight_dtype):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.to(weight_dtype)
    vae.eval()
    vae.requires_grad_(False)
    
    return vae


### HELPER FUNCTIONS ###
def compute_ess(w, dim=0):
    ess = (w.sum(dim=dim))**2 / torch.sum(w**2, dim=dim)
    return ess


def compute_ess_from_log_w(log_w, dim=0):
    return compute_ess(normalize_weights(log_w, dim=dim), dim=dim)


def normalize_weights(log_weights, dim=0):
    return torch.exp(normalize_log_weights(log_weights, dim=dim))


def normalize_log_weights(log_weights, dim):
    log_weights = log_weights - log_weights.max(dim=dim, keepdims=True)[0]
    log_weights = log_weights - torch.logsumexp(log_weights, dim=dim, keepdims=True)
    return log_weights


def log_normal_density(sample, mean, var):
    return Normal(loc=mean, scale=torch.sqrt(var)).log_prob(sample)


def systematic_resampling(particles, weights):
    """Perform systematic resampling on a set of particles based on their weights."""
    device = particles.device
    N = len(weights)
    # Normalize weights
    weights /= torch.sum(weights)
    
    # Calculate cumulative sum of weights
    cumsum = torch.cumsum(weights, dim=0)
    
    # Generate systematic noise (one random number)
    u = torch.distributions.Uniform(low=0.0, high=1.0/N).sample()
    
    # Generate points for systematic sampling
    points = torch.zeros(N, device=device)
    for i in range(N):
        points[i] = u + i/N
    
    # Initialize arrays for results
    indexes = torch.zeros(N, dtype=torch.long, device=device)
    cumsum = torch.cat([torch.tensor([0.0], device=device), cumsum])
    
    # Perform systematic resampling
    i, j = 0, 0
    while i < N:
        while points[i] > cumsum[j+1]:
            j += 1
        indexes[i] = j
        i += 1
    
    # Resample particles and reset weights
    resampled_particles = particles[indexes]
    new_weights = torch.zeros(N, device=device)
    return resampled_particles, new_weights, indexes   


def prepare_latent_sample(vae, images, weight_dtype=torch.float32):
    """Encode images to latent space using VAE"""
    with torch.no_grad():
        latent = vae.encode(images).latent_dist.sample()
    return latent


def decode_latents(vae, latents, scaling_factor=4.0):
    """Decode latent samples to images using VAE"""
    with torch.no_grad():
        # Scale latents
        latents = latents * 4 / vae.config.scaling_factor
        
        # Decode the latents to images
        images = vae.decode(latents).sample
        
        # Normalize images to [0, 1.3] range
        images = (images / 2 + 0.5).clamp(0, 1.3)
        
    return images


def get_log_probs(logits, true_labels, pred_labels=None, threshold=0.5):
    """Get log probabilities for multi-label classification."""
    batch_size, num_classes = logits.shape
    
    # Convert logits to probabilities using sigmoid (for multi-label)
    probs = torch.sigmoid(logits)
    
    # Get predicted labels if not provided
    if pred_labels is None:
        pred_labels = (probs > threshold).float()
    
    # Convert to log probabilities
    log_probs_positive = torch.log(probs + 1e-8)
    log_probs_negative = torch.log(1 - probs + 1e-8)
    
    # Get log probabilities for true labels
    true_log_probs = true_labels * log_probs_positive + (1 - true_labels) * log_probs_negative
    
    # Get log probabilities for predicted labels
    pred_log_probs = pred_labels * log_probs_positive + (1 - pred_labels) * log_probs_negative
    
    # Calculate joint log probability for true configuration
    true_joint_log_prob = true_log_probs.sum(dim=1)
    
    # Calculate joint log probability for predicted configuration
    pred_joint_log_prob = pred_log_probs.sum(dim=1)
    
    # Calculate log probabilities for only the positive (true) labels
    positive_mask = true_labels == 1
    true_positive_log_probs = torch.where(positive_mask, log_probs_positive, torch.zeros_like(log_probs_positive))
    true_positive_joint_log_prob = true_positive_log_probs.sum(dim=1)
    
    # Calculate average log probability per true positive label
    num_true_positives = true_labels.sum(dim=1)
    avg_true_positive_log_prob = torch.where(
        num_true_positives > 0,
        true_positive_joint_log_prob / num_true_positives,
        torch.zeros_like(true_positive_joint_log_prob)
    )
    
    return {
        'true_log_probs': true_joint_log_prob,
        'pred_log_probs': pred_joint_log_prob,
        'all_log_probs': true_log_probs,
        'true_log_probs_per_class': true_log_probs,
        'pred_log_probs_per_class': pred_log_probs,
        'true_joint_log_prob': true_joint_log_prob,
        'pred_joint_log_prob': pred_joint_log_prob,
        'true_positive_log_probs': true_positive_log_probs,
        'true_positive_joint_log_prob': true_positive_joint_log_prob,
        'avg_true_positive_log_prob': avg_true_positive_log_prob,
        'all_probs': probs,
        'all_log_probs_positive': log_probs_positive,
        'all_log_probs_negative': log_probs_negative,
    }


def twisting_classifier(x_0_hat_scaled_to_vae, y_true, weight_dtype, classifier):
    classifier_input = x_0_hat_scaled_to_vae.to(weight_dtype)
    x_logit = classifier(classifier_input)
    return get_log_probs(x_logit, y_true)["true_log_probs"]


def get_guidance_scale(timestep, total_timesteps, schedule_type="cosine", base_scale=0.1, max_scale=1.0):
    """Get guidance scale based on timestep and schedule type."""
    t_normalized = timestep / (total_timesteps - 1)
    
    if schedule_type == "cosine":
        scale = base_scale + (max_scale - base_scale) * (1 + math.cos(math.pi * t_normalized)) / 2
    elif schedule_type == "linear":
        scale = base_scale + (max_scale - base_scale) * t_normalized
    elif schedule_type == "exponential":
        scale = base_scale + (max_scale - base_scale) * (t_normalized ** 2)
    elif schedule_type == "inverse_exponential":
        scale = base_scale + (max_scale - base_scale) * ((1 - t_normalized) ** 2)
    elif schedule_type == "constant":
        scale = base_scale
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return scale


def safe_gradient_computation(log_prob, target_tensor, max_grad_norm=1.0, gradient_checkpointing=True):
    """Safely compute gradients with proper memory management and clipping."""
    try:
        if gradient_checkpointing:
            grad = torch.autograd.grad(
                log_prob.sum(), 
                target_tensor, 
                retain_graph=False,
                create_graph=False,
                only_inputs=True
            )[0]
        else:
            grad = torch.autograd.grad(
                log_prob.sum(), 
                target_tensor, 
                retain_graph=False
            )[0]
        
        # Clip gradients to prevent instability
        if max_grad_norm > 0:
            grad_norm = torch.norm(grad)
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
        
        grad = grad.detach()
        return grad
    
    except RuntimeError as e:
        print(f"Gradient computation failed: {e}")
        return torch.zeros_like(target_tensor)


def cleanup_gradients(*tensors):
    """Properly detach tensors and clean up gradient computation."""
    cleaned_tensors = []
    
    for tensor in tensors:
        if tensor is not None:
            cleaned_tensor = tensor.detach()
            cleaned_tensor.requires_grad_(False)
            cleaned_tensors.append(cleaned_tensor)
        else:
            cleaned_tensors.append(None)
    
    return tuple(cleaned_tensors) if len(cleaned_tensors) > 1 else cleaned_tensors[0]

### SAMPLER FUNCTIONS ###
def sample_edm(
    model,
    scheduler,
    image_size=32,
    batch_size=1,
    num_inference_steps=50,
    protein_labels=None,
    cell_line_labels=None,
    generator=None,
    unconditional_sample=False,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    device=None,
    weight_dtype=None,
    reference_channels=None,
):
    """
    Ordinary EDM sampling without classifier guidance or particle filtering.
    
    Args:
        model: The diffusion model
        scheduler: The noise scheduler
        image_size: Size of the generated images
        batch_size: Number of samples to generate
        num_inference_steps: Number of denoising steps
        protein_labels: Protein conditioning labels
        cell_line_labels: Cell line conditioning labels
        generator: Random number generator
        unconditional_sample: Whether to generate unconditional samples
        s_churn, s_tmin, s_tmax, s_noise: Stochastic sampling parameters
        device: Device to run on
        weight_dtype: Data type for weights
        reference_channels: Reference channels to concatenate with latents
    
    Returns:
        Generated latents
    """
    
    # Create random noise
    latent_channels = 16
    latents = torch.randn(
        (batch_size, latent_channels, image_size, image_size),
        generator=generator,
        device=device,
        dtype=weight_dtype
    )
    
    # Scale initial noise by first sigma
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.sigmas[0].to(device)
    
    # Setup progress bar
    progress_bar = tqdm(range(num_inference_steps))
    progress_bar.set_description("Sampling")
    
    # Prepare conditioning inputs
    if protein_labels is not None:
        protein_labels = protein_labels.to(device)
    if cell_line_labels is not None:
        cell_line_labels = cell_line_labels.to(device)
    if reference_channels is not None:
        reference_channels = reference_channels.to(device, dtype=weight_dtype)
        # Expand reference channels to match batch size if needed
        if reference_channels.shape[0] != batch_size:
            reference_channels = reference_channels.repeat(batch_size, 1, 1, 1)
    
    # Main sampling loop - no gradients needed for ordinary inference
    with torch.no_grad():
        for i, t in enumerate(progress_bar):
            sigma = scheduler.sigmas[i].to(device)
            sigma_next = scheduler.sigmas[i + 1].to(device) if i < len(scheduler.sigmas) - 1 else torch.tensor(0.0, device=device)
            
            # Calculate gamma for stochastic sampling
            gamma = min(s_churn / (len(scheduler.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
            sigma_hat = sigma * (gamma + 1)
            sigma_hat = sigma_hat.to(device)
            
            # Add noise if gamma > 0
            if gamma > 0:
                noise = torch.randn_like(latents, generator=generator, device=device, dtype=latents.dtype)
                eps = noise * s_noise
                latents = latents + eps * (sigma_hat**2 - sigma**2) ** 0.5
        
            # Prepare model input - ensure sigma_hat_view is on correct device
            sigma_hat_view = sigma_hat.view(-1, 1, 1, 1).expand(batch_size, -1, -1, -1).to(device)
            model_input, timestep_input = edm_clean_image_to_model_input(latents, sigma_hat_view)
            
            # Ensure timestep is properly shaped as 1D array
            if len(timestep_input.shape) == 0:
                # Scalar timestep - expand to batch size
                timestep_input = timestep_input.unsqueeze(0).repeat(batch_size)
            elif len(timestep_input.shape) > 1:
                # Multi-dimensional - flatten to 1D
                timestep_input = timestep_input.flatten()
            
            # Ensure we have the right number of timesteps for the batch
            if timestep_input.shape[0] != batch_size:
                if timestep_input.shape[0] == 1:
                    timestep_input = timestep_input.repeat(batch_size)
                else:
                    # Take first element and repeat
                    timestep_input = timestep_input[0].unsqueeze(0).repeat(batch_size)
            
            # Concatenate reference channels if provided
            if reference_channels is not None:
                model_input = torch.cat([model_input, reference_channels], dim=1)
            
            model_input = model_input.to(weight_dtype)
            timestep_input = timestep_input.to(weight_dtype)
            
            # Enable unconditional sampling if requested
            if unconditional_sample:
                if cell_line_labels is not None:
                    cell_line_labels = torch.zeros_like(cell_line_labels)
                if protein_labels is not None:
                    protein_labels = torch.zeros_like(protein_labels)
            
            # Get model prediction
            model_output = model(
                model_input,
                timestep_input,
                protein_labels=protein_labels,
                cell_line_labels=cell_line_labels,
                encoder_hidden_states=None,
            ).sample
            
            # Convert model output to denoised latent
            predicted_x_start = edm_model_output_to_x_0_hat(latents, sigma_hat_view, model_output)
            
            # Calculate step size and direction - ensure all tensors are on correct device
            step_sigma = (sigma - sigma_next).to(device)
            step_size = step_sigma / sigma
            direction = (predicted_x_start - latents) / sigma_hat_view
            
            # Update latents
            latents = latents + step_size.view(-1, 1, 1, 1) * sigma_hat_view * direction
    
    return latents

### DISTRIBUTED INFERENCE ###
def run_inference_accelerate(args):
    """Run inference using Accelerate for multi-GPU support with synchronized logging"""
    
    # Initialize accelerator with explicit configuration
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        split_batches=False,
        step_scheduler_with_optimizer=False,
    )
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    device = accelerator.device
    
    logger = SynchronizedLogger(args.log_dir, accelerator)
    logger.log({
        "status": "Starting inference", 
        "num_processes": accelerator.num_processes,
        "device": str(device),
        "process_index": accelerator.process_index
    })
    
    # Load models
    accelerator.print("Loading models...")
    
    model = DiTTransformer2DModel.from_pretrained(args.model_path, subfolder="unet")
    model.to(device)
    model.to(args.weight_dtype)
    model.eval()
    
    vae = load_vae(args.vae_path, args.weight_dtype)
    vae.to(device)
    vae.eval()
    
    # Load scheduler
    scheduler = create_edm_scheduler(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=args.sigma_data,
        num_train_timesteps=1000,
        prediction_type="sample"
    )
    scheduler.sigmas = scheduler.sigmas.to(device)
    
    accelerator.print("Models loaded successfully")
    
    # Load data dictionaries
    cell_line_dict = pickle.load(open(args.cell_line_map_path, "rb"))
    label_dict = pickle.load(open(args.antibody_map_path, "rb"))
    # No annotation dict needed without classifier
    
    # Load dataframe
    df = pd.read_csv(args.csv_file_path)
    accelerator.print(f"Loaded {len(df)} samples from CSV")
    
    # Create dataset and dataloader
    dataset = CSVDataset(args.csv_file_path)
    collate_fn = collate_fn_factory(df, label_dict, cell_line_dict)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # Use simple batch_size instead of effective_batch_size
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    # Prepare model and dataloader with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)
    
    # Create output directory
    output_dir = args.output_dir
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process batches
    total_processed = 0
    progress_bar = tqdm(
        dataloader, 
        desc=f"Process {accelerator.process_index}", 
        disable=not accelerator.is_local_main_process
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        if batch is None:
            continue
        
        # Move batch to device and dtype
        cond_images = batch["cond_image"].to(device).to(args.weight_dtype)
        gt_images = batch["gt_image"].to(device).to(args.weight_dtype)
        cell_line = batch["cell_line"].to(device).long()
        protein_label = batch["label"].to(device).long()
        
        protein_names = batch["protein_name"]
        cell_line_names = batch["cell_line_name"]
        image_paths = batch["image_paths"]
        
        batch_size = len(batch['cond_image'])
        
        # Set random seed for reproducible generation
        generator = torch.Generator(device=device).manual_seed(
            args.seed + accelerator.process_index + batch_idx
        )
        
        # Prepare reference channels if needed
        with torch.no_grad():
            reference_channels = vae.encode(cond_images).latent_dist.sample().to(args.weight_dtype) * vae.config.scaling_factor / 4

        # Sample from model
        generated_latents = sample_edm(
            model=accelerator.unwrap_model(model),
            scheduler=scheduler,
            batch_size=batch_size,
            image_size=64,
            num_inference_steps=args.num_inference_steps,
            protein_labels=protein_label,
            cell_line_labels=cell_line,
            generator=generator,
            unconditional_sample=getattr(args, 'unconditional_sample', False),
            s_churn=0,
            device=device,
            weight_dtype=args.weight_dtype,
            reference_channels=reference_channels,
        )
        
        
        # Process and save results (existing code)
        with torch.no_grad():
            vae_type = vae.dtype
            vae.to(torch.float32)
            
            current_batch_size = len(batch['cond_image'])
            for sample_idx in range(current_batch_size):
                sample_latents = generated_latents[sample_idx].to(args.weight_dtype)
                # Fix: sample_latents is 3D [channels, height, width], so slice channels only
                generated_images_gt = decode_latents(vae, sample_latents[:16,:,:].unsqueeze(0).to(torch.float32))

                assert generated_images_gt.shape[0] == 1, "Generated images should have batch size of 1"
                assert generated_images_gt.shape[1] == 3, "Generated images should have 3 channels"

                protein_name = protein_names[sample_idx]
                cell_line_name = cell_line_names[sample_idx]
                image_path = image_paths[sample_idx]

                synthetic_image = generated_images_gt.cpu()
                # Remove batch dimension after mean
                synthetic_image = synthetic_image.squeeze(0)
                synthetic_image = synthetic_image.mean(dim=0, keepdim=True)
                
                # Save images
                cond_image_single = cond_images[sample_idx].to(torch.float32).cpu().numpy()
                cond_image_single += 1
                cond_image_single /= 2
                synthetic_image_np = synthetic_image.numpy()
                
                synthetic_stack = np.concatenate([synthetic_image_np, cond_image_single], axis=0)
                synthetic_stack = synthetic_stack[[1, 0, 2, 3], :, :]
                synthetic_stack = np.moveaxis(synthetic_stack, 0, -1)
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                # if len(base_name) > 20:
                #     base_name = base_name[:20]
                # randomly generated 10-char string to avoid conflicts
                # base_name = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=10))
                if len(protein_name) > 30:
                    protein_name = protein_name[:30]
                output_filename = f"{output_dir}/{base_name}_{cell_line_name}_{protein_name}_pred.tif"
                imwrite(output_filename, synthetic_stack)
                
                # Save ground truth
                if gt_images is None:
                    continue
                gt_image_single = gt_images[sample_idx].to(torch.float32).cpu().numpy()
                gt_image_single += 1
                gt_image_single /= 2
                real_stack = np.concatenate([gt_image_single, cond_image_single], axis=0)
                real_stack = real_stack[[1, 0, 2, 3], :, :]
                real_stack = np.moveaxis(real_stack, 0, -1)
                real_output_filename = f"{output_dir}/{base_name}_{cell_line_name}_{protein_name}_real.tif"
                imwrite(real_output_filename, real_stack)
            
            vae.to(vae_type)
            total_processed += current_batch_size
        
        # Log batch metrics
        logger.log({
            "batch_completed": batch_idx,
            "samples_in_batch": current_batch_size,
            "cumulative_processed": total_processed
        }, batch_idx=batch_idx)
        
        # Synchronize logs after EACH batch for real-time monitoring
        logger.sync_and_log_batch(batch_idx)
        accelerator.wait_for_everyone()
    
    logger.finish()
    
    accelerator.print(f"Process {accelerator.process_index} completed!")
    

### DATALOADER ###
class CSVDataset(Dataset):
    """Custom dataset for loading CSV data"""
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return idx


def create_batch_from_csv(df, start_idx, batch_size, label_dict, cell_line_dict):
    """Create a batch from CSV data starting at start_idx with better error handling"""
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx]
    
    batch = {
        'cond_image': [],
        'gt_image': [],
        'label': [],
        'cell_line': [],
        'protein_name': [],
        'cell_line_name': [],
        'image_paths': []
    }
    
    for _, row in batch_df.iterrows():
        image_path = row['image_path']
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                continue
            img = imread(image_path)
                
            # randomly rotate 90, 180 or 270 deg for the HW dimension
            # degree = np.random.choice([90, 180, 270])
            # img = rotate(img, degree, axes=(0, 1), reshape=False)
            
            img = torch.from_numpy(img)
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            continue
        
        
        image_path = image_path.split('/')[-1]
        if img.shape[2] == 4:
            gt_img = img[:, :, [1]]
            cond_img = img[:, :, [0, 2, 3]]
        elif img.shape[2] == 3:
            gt_img = None
            cond_img = img
        else:
            print(f"Warning: Unexpected number of channels in image {image_path}: {img.shape[2]}")
            continue


        # move to channel first
        gt_img = torch.permute(gt_img, (2, 0, 1))
        cond_img = torch.permute(cond_img, (2, 0, 1))

        cell_line = row['cell_line_name']
        ab = row['gene_name']
        
        # Check if keys exist in dictionaries
        if ab not in label_dict:
            print(f"Warning: Protein {ab} not found in label_dict")
            continue
        # if cell_line not in cell_line_dict:
        #     print(f"Warning: Cell line {cell_line} not found in cell_line_dict")
        #     continue


        # Add to batch
        batch['cond_image'].append(cond_img)
        batch['gt_image'].append(gt_img)
        batch['label'].append(label_dict[ab])
        if cell_line not in cell_line_dict:
            batch['cell_line'].append(0)  # Unknown cell line
        else:
            batch['cell_line'].append(cell_line_dict[cell_line])
        batch['protein_name'].append(ab)
        batch['cell_line_name'].append(cell_line)
        batch['image_paths'].append(image_path)
    
    # Convert lists to tensors
    try:
        if len(batch['cond_image']) == 0:
            return None
            
        batch['cond_image'] = torch.stack(batch['cond_image'])
        batch['gt_image'] = torch.stack(batch['gt_image'])
        batch['label'] = torch.tensor(batch['label'])
        batch['cell_line'] = torch.tensor(batch['cell_line'])
    except RuntimeError as e:
        print(f"Warning: Failed to create batch tensors: {e}")
        return None

    return batch


def collate_fn_factory(df, label_dict, cell_line_dict):
    """Factory function to create a collate_fn with access to the dictionaries"""
    def collate_fn(indices):
        start_idx = min(indices)
        batch_size = len(indices)
        batch = create_batch_from_csv(df, start_idx, batch_size, label_dict, cell_line_dict)
        return batch
    return collate_fn


### MAIN FUNCTION ###
def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Diffusion Inference with Accelerate')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vae_path', type=str, required=True)
    parser.add_argument('--csv_file_path', type=str, required=True)
    parser.add_argument('--cell_line_map_path', type=str, required=True)
    parser.add_argument('--antibody_map_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--unconditional_sample', action='store_true')
    parser.add_argument('--sigma_min', type=float, default=0.002)
    parser.add_argument('--sigma_max', type=float, default=80.0)
    parser.add_argument('--sigma_data', type=float, default=0.5)
    parser.add_argument('--weight_dtype', type=str, default='float32')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory for output files')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for log files')
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Convert weight_dtype string to torch dtype
    if args.weight_dtype == 'float32':
        args.weight_dtype = torch.float32
    elif args.weight_dtype == 'bfloat16':
        args.weight_dtype = torch.bfloat16
    else:
        args.weight_dtype = torch.float16
    
    # Run inference with Accelerate
    run_inference_accelerate(args)


if __name__ == "__main__":
    main()