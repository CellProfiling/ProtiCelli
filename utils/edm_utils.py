import torch
import torch.nn.functional as F

def edm_precondition(sigma, sigma_data=0.5):
    """
    Apply EDM preconditioning to the input x based on noise level sigma.
    
    Args:
        sigma (torch.Tensor): Noise level tensor
        sigma_data (float): Standard deviation of the data distribution
        
    Returns:
        tuple: (c_skip, c_out, c_in) preconditioning factors
    """
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in = 1 / (sigma**2 + sigma_data**2).sqrt() 
    c_noise = sigma.log() / 4
    
    return c_skip, c_out, c_in, c_noise



def edm_clean_image_to_model_input(x_noisy, sigma):
    """
    Precondition the input x_clean based on noise level sigma and noise.
    
    Args:
        x_clean (torch.Tensor): Clean input tensor
        sigma (torch.Tensor): Noise level tensor
        noise (torch.Tensor): Noise tensor
        
    Returns:
        torch.Tensor: Preconditioned input tensor
    """
    c_skip, c_out, c_in, c_noise = edm_precondition(sigma)
    
    model_input = c_in * x_noisy
    timestep_input = c_noise

    return model_input, timestep_input


def edm_model_output_to_x_0_hat(x_noisy, sigma, model_output):
    """
    Precondition the noisy input x_noisy based on noise level sigma and noise.
    
    Args:
        x_noisy (torch.Tensor): Noisy input tensor
        sigma (torch.Tensor): Noise level tensor
        
    Returns:
        torch.Tensor: Preconditioned x_0_hat tensor
    """
    c_skip, c_out, c_in, c_noise = edm_precondition(sigma)
    
    x_0_hat = c_skip * x_noisy + c_out * model_output
    
    return x_0_hat



def edm_loss_weight(sigma, sigma_data=0.5):
    """
    Calculate the importance sampling weight for EDM loss.
    
    Args:
        sigma (torch.Tensor): Noise level tensor
        sigma_data (float): Standard deviation of the data distribution
        
    Returns:
        torch.Tensor: Normalized weight tensor
    """
    # Weight function from EDM paper
    (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
    # Normalize weights to prevent numerical issues
    return weight # / weight.mean()



def get_noise_weight(timesteps, schedule_type, max_weight, min_weight, num_timesteps, sigmas=None):
    """
    Compute noise weights based on timesteps and selected schedule.
    
    Args:
        timesteps (torch.Tensor): Current timesteps in the diffusion process
        schedule_type (str): Type of weighting schedule to use
        max_weight (float): Maximum weight value
        min_weight (float): Minimum weight value
        num_timesteps (int): Total number of timesteps in diffusion
        sigmas (torch.Tensor, optional): Sigma values for SNR-based weighting
        
    Returns:
        torch.Tensor: Tensor of weights for the current batch
    """
    if schedule_type == "constant":
        return torch.ones_like(timesteps, dtype=torch.float32) * max_weight
    
    # Normalize timesteps to [0, 1]
    t_normalized = timesteps.float() / num_timesteps
    
    if schedule_type == "linear":
        # Linear schedule from max_weight to min_weight
        weights = max_weight - t_normalized * (max_weight - min_weight)
    
    elif schedule_type == "exponential":
        # Exponential schedule from max_weight to min_weight
        weights = min_weight + (max_weight - min_weight) * torch.exp(-5 * t_normalized)
    
    elif schedule_type == "snr" and sigmas is not None:
        # SNR-based weighting using sigma values
        weights = 1.0 / (sigmas ** 2)  # EDM-style SNR weighting
    
    else:
        # Default to constant weights
        weights = torch.ones_like(timesteps, dtype=torch.float32) * max_weight
        
    return weights.to(timesteps.device)



def prepare_latent_sample(vae, images, weight_dtype):
    """
    Encode images to latent space using VAE.
    
    Args:
        vae: VAE model
        images (torch.Tensor): Input images
        weight_dtype: Data type for weights
        
    Returns:
        torch.Tensor: Latent representation
    """
    with torch.no_grad():
        # Convert images to proper dtype
        images = images.to(weight_dtype)
        
        # Encode images to latent space
        latents = vae.encode(images).latent_dist.sample()
        
        # Scale latents according to the VAE scaling factor
        latents = latents * vae.scaling_factor
        
    return latents



def prepare_model_inputs(gt_latents, cond_latents, cell_line, protein_label, dropout_prob=1, weight_dtype=torch.float32, encoder_hidden_states=None):
    """
    Prepare model inputs including class labels with dropout.
    
    Args:
        gt_latents (torch.Tensor): Ground truth latents
        cond_latents (torch.Tensor): Conditioning latents
        cell_line (torch.Tensor): Cell line indices
        protein_label (torch.Tensor): Protein label indices
        dropout_prob (float): Probability of label dropout
        weight_dtype: Data type for weights
        
    Returns:
        tuple: (clean_images, total_label)
    """
    # Concatenate latents along the channel dimension
    #clean_images = torch.cat([gt_latents, cond_latents], dim=1).to(weight_dtype) / 4

    #Do Not USE cond_latents as input for now
    clean_images = gt_latents.to(weight_dtype) / 4
    cond_latents = cond_latents.to(weight_dtype) / 4

    # Create dropout mask
    dropout_mask = torch.rand(protein_label.shape, dtype=weight_dtype, device=clean_images.device) > dropout_prob
    
    # Create one-hot encodings
    # label_onehot = F.one_hot(protein_label, num_classes=13348).to(weight_dtype)
    # cell_line_onehot = F.one_hot(cell_line, num_classes=40).to(weight_dtype)
    
    # # Apply dropout
    # label_onehot = label_onehot * dropout_mask.reshape(-1, 1)
    # cell_line_onehot = cell_line_onehot * dropout_mask.reshape(-1, 1)
    
    # Concatenate labels
    protein_label = protein_label.reshape(-1,1) * dropout_mask.reshape(-1, 1)
    cell_line = cell_line.reshape(-1,1) * dropout_mask.reshape(-1, 1)

    protein_label = protein_label.long()
    cell_line = cell_line.long()
    total_label = (protein_label, cell_line)#torch.cat([label_onehot, cell_line_onehot], dim=1)
    
    if encoder_hidden_states is not None:
        # Concatenate encoder hidden states if provided
        encoder_hidden_states = encoder_hidden_states # * dropout_mask.reshape(-1, 1,1)

    return (clean_images, cond_latents), total_label, encoder_hidden_states, dropout_mask

def decode_latents(vae, latents):
    """
    Decode latents to images using VAE.
    
    Args:
        vae: VAE model
        latents (torch.Tensor): Latent representations
        
    Returns:
        torch.Tensor: Decoded images
    """
    with torch.no_grad():
        # Scale latents
        latents = latents / vae.scaling_factor
        
        # Decode latents
        images = vae.decode(latents).sample
        
    return images