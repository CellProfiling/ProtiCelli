import math
import torch
import torch.nn.functional as F
from diffusers.schedulers import EDMEulerScheduler

def create_edm_scheduler(sigma_min=0.002, sigma_max=80.0, sigma_data=0.5, num_train_timesteps=1000, prediction_type="epsilon"):
    """
    Create an EDM scheduler for diffusion training.
    
    Args:
        sigma_min (float): Minimum noise level
        sigma_max (float): Maximum noise level
        sigma_data (float): Standard deviation of the data distribution
        num_train_timesteps (int): Number of training timesteps
        prediction_type (str): Model prediction type ("epsilon" or "sample")
        
    Returns: 
        EDMEulerScheduler: Configured scheduler
    """
    noise_scheduler = EDMEulerScheduler(
        sigma_min=sigma_min,
        sigma_max=sigma_max, 
        sigma_data=sigma_data,
        prediction_type=prediction_type,
        num_train_timesteps=num_train_timesteps,
    )
    
    return noise_scheduler

def edm_precondition(x, sigma, sigma_data=0.5):
    """
    Apply EDM preconditioning to the input x based on noise level sigma.
    
    Args:
        x (torch.Tensor): Input tensor
        sigma (torch.Tensor): Noise level
        sigma_data (float): Data noise level
        
    Returns:
        tuple: (c_skip, c_out, c_in) preconditioning factors
    """
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
    
    return c_skip, c_out, c_in

def edm_loss_weight(sigma, sigma_data=0.5):
    """
    Calculate the importance sampling weight for EDM loss.
    
    Args:
        sigma (torch.Tensor): Noise level
        sigma_data (float): Data noise level
        
    Returns:
        torch.Tensor: EDM loss weights
    """
    # Weight function from EDM paper
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
    # Normalize weights to prevent numerical issues
    return weight / weight.mean()

def prepare_input_with_noise(clean_images, sigmas, noise=None):
    """
    Prepare model input by adding noise according to EDM formulation.
    
    Args:
        clean_images (torch.Tensor): Clean input images
        sigmas (torch.Tensor): Noise levels
        noise (torch.Tensor, optional): Optional noise tensor. If None, random noise is generated.
        
    Returns:
        tuple: (noisy_images, noise, c_skip, c_out, c_in)
    """
    # Reshape sigma for broadcasting
    sigmas = sigmas.reshape(-1, 1, 1, 1)
    
    # Generate random noise if not provided
    if noise is None:
        noise = torch.randn_like(clean_images)
    
    # Apply EDM preconditioning
    c_skip, c_out, c_in = edm_precondition(clean_images, sigmas)
    
    # Add noise according to EDM formulation
    noisy_images = c_skip * clean_images + c_out * noise
    
    return noisy_images, noise, c_skip, c_out, c_in

def calculate_edm_loss(model_output, target, weights):
    """
    Calculate the EDM-weighted loss.
    
    Args:
        model_output (torch.Tensor): Model predictions
        target (torch.Tensor): Target values
        weights (torch.Tensor): EDM importance weights
        
    Returns:
        torch.Tensor: Weighted loss value
    """
    # Calculate MSE loss between model output and target
    loss = F.mse_loss(model_output.float(), target.float(), reduction='none')
    
    # Apply weights and mean across batch
    loss = (loss.mean(dim=[1, 2, 3]) * weights).mean()
    
    return loss

def get_timestep_from_sigma(sigma, noise_scheduler):
    """
    Convert sigma values to timestep indices.
    
    Args:
        sigma (torch.Tensor): Sigma values
        noise_scheduler: EDM noise scheduler
        
    Returns:
        torch.Tensor: Corresponding timestep indices
    """
    # Find closest sigma in the scheduler's discretization
    sigmas = noise_scheduler.sigmas.to(sigma.device)
    dists = torch.abs(sigma.unsqueeze(1) - sigmas.unsqueeze(0))
    timesteps = torch.argmin(dists, dim=1)
    
    return timesteps

def extract_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D tensor for a batch of indices.
    
    Args:
        arr (torch.Tensor): 1-D tensor to extract values from
        timesteps (torch.Tensor): Tensor of indices
        broadcast_shape (tuple): Shape to broadcast the extracted values to
        
    Returns:
        torch.Tensor: Extracted and broadcasted values
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)