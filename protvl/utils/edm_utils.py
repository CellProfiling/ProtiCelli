"""EDM (Elucidating Diffusion Models) preconditioning and loss utilities.

Implements the preconditioning scheme from Karras et al. (2022),
"Elucidating the Design Space of Diffusion-Based Generative Models."
"""

import torch
import torch.nn.functional as F


def edm_precondition(sigma, sigma_data=0.5):
    """Compute EDM preconditioning factors.

    Parameters
    ----------
    sigma : torch.Tensor
        Noise level tensor.
    sigma_data : float
        Standard deviation of the data distribution.

    Returns
    -------
    tuple of (c_skip, c_out, c_in, c_noise)
    """
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
    c_noise = sigma.log() / 4
    return c_skip, c_out, c_in, c_noise


def edm_clean_image_to_model_input(x_noisy, sigma):
    """Apply EDM input preconditioning.

    Parameters
    ----------
    x_noisy : torch.Tensor
        Noisy input tensor.
    sigma : torch.Tensor
        Noise level tensor.

    Returns
    -------
    tuple of (model_input, timestep_input)
    """
    c_skip, c_out, c_in, c_noise = edm_precondition(sigma)
    model_input = c_in * x_noisy
    timestep_input = c_noise
    return model_input, timestep_input


def edm_model_output_to_x_0_hat(x_noisy, sigma, model_output):
    """Convert raw model output to denoised prediction x_0_hat.

    Parameters
    ----------
    x_noisy : torch.Tensor
        Noisy input tensor.
    sigma : torch.Tensor
        Noise level tensor.
    model_output : torch.Tensor
        Raw model output.

    Returns
    -------
    torch.Tensor
        Denoised prediction.
    """
    c_skip, c_out, c_in, c_noise = edm_precondition(sigma)
    x_0_hat = c_skip * x_noisy + c_out * model_output
    return x_0_hat


def edm_loss_weight(sigma, sigma_data=0.5):
    """Compute per-sample loss weights for EDM training.

    Parameters
    ----------
    sigma : torch.Tensor
        Noise level tensor.
    sigma_data : float
        Standard deviation of the data distribution.

    Returns
    -------
    torch.Tensor
        Loss weight per sample.
    """
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    return weight


def prepare_latent_sample(vae, images, weight_dtype):
    """Encode images to latent space using VAE.

    Parameters
    ----------
    vae : AutoencoderKL
        VAE model.
    images : torch.Tensor
        Input images [B, C, H, W].
    weight_dtype : torch.dtype
        Data type for weights.

    Returns
    -------
    torch.Tensor
        Scaled latent representation.
    """
    with torch.no_grad():
        images = images.to(weight_dtype)
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    return latents


def prepare_model_inputs(
    gt_latents,
    cond_latents,
    cell_line,
    protein_label,
    dropout_prob=0.2,
    weight_dtype=torch.float32,
    encoder_hidden_states=None,
):
    """Prepare model inputs with label dropout for classifier-free guidance.

    Parameters
    ----------
    gt_latents : torch.Tensor
        Ground truth latents.
    cond_latents : torch.Tensor
        Conditioning latents.
    cell_line : torch.Tensor
        Cell line indices.
    protein_label : torch.Tensor
        Protein label indices.
    dropout_prob : float
        Probability of dropping labels (for CFG training).
    weight_dtype : torch.dtype
        Data type for weights.
    encoder_hidden_states : torch.Tensor, optional
        Additional encoder hidden states.

    Returns
    -------
    tuple
        ((clean_images, cond_latents), (protein_label, cellline_label),
         encoder_hidden_states, dropout_mask)
    """
    clean_images = gt_latents.to(weight_dtype) / 4
    cond_latents = cond_latents.to(weight_dtype) / 4

    dropout_mask = (
        torch.rand(protein_label.shape, dtype=weight_dtype, device=clean_images.device)
        > dropout_prob
    )

    protein_label = (protein_label.reshape(-1, 1) * dropout_mask.reshape(-1, 1)).long()
    cell_line = (cell_line.reshape(-1, 1) * dropout_mask.reshape(-1, 1)).long()

    return (clean_images, cond_latents), (protein_label, cell_line), encoder_hidden_states, dropout_mask
