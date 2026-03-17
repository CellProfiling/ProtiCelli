"""EDM sampling for ProtVS inference."""

import math
import torch
from tqdm.auto import tqdm

from .utils.edm_utils import edm_clean_image_to_model_input, edm_model_output_to_x_0_hat


def sample_edm(
    model,
    scheduler,
    batch_size: int = 1,
    image_size: int = 64,
    num_inference_steps: int = 50,
    protein_labels=None,
    cell_line_labels=None,
    generator=None,
    unconditional_sample: bool = False,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    device=None,
    weight_dtype=None,
    reference_channels=None,
):
    """EDM sampling loop for generating protein localization images.

    Parameters
    ----------
    model : DiTTransformer2DModel
        The diffusion transformer model.
    scheduler : EDMDPMSolverMultistepScheduler
        The noise scheduler with precomputed sigmas.
    batch_size : int
        Number of samples to generate.
    image_size : int
        Spatial size of the latent (typically 64).
    num_inference_steps : int
        Number of denoising steps.
    protein_labels : torch.Tensor, optional
        Protein conditioning labels [B].
    cell_line_labels : torch.Tensor, optional
        Cell line conditioning labels [B].
    generator : torch.Generator, optional
        Random number generator for reproducibility.
    unconditional_sample : bool
        If True, zero out all conditioning (unconditional generation).
    s_churn, s_tmin, s_tmax, s_noise : float
        Stochastic sampling parameters (Karras et al.).
    device : torch.device
        Target device.
    weight_dtype : torch.dtype
        Data type for computation.
    reference_channels : torch.Tensor, optional
        Reference channel latents to concatenate with model input [B, C, H, W].

    Returns
    -------
    torch.Tensor
        Generated latents [B, C, H, W].
    """
    latent_channels = 16

    # Initial noise
    latents = torch.randn(
        (batch_size, latent_channels, image_size, image_size),
        generator=generator,
        device=device,
        dtype=weight_dtype,
    )

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.sigmas[0].to(device)

    # Move labels to device
    if protein_labels is not None:
        protein_labels = protein_labels.to(device)
    if cell_line_labels is not None:
        cell_line_labels = cell_line_labels.to(device)
    if reference_channels is not None:
        reference_channels = reference_channels.to(device, dtype=weight_dtype)
        if reference_channels.shape[0] != batch_size:
            reference_channels = reference_channels.expand(batch_size, -1, -1, -1)

    with torch.no_grad():
        for i in range(num_inference_steps):
            sigma = scheduler.sigmas[i].to(device)
            sigma_next = (
                scheduler.sigmas[i + 1].to(device)
                if i < len(scheduler.sigmas) - 1
                else torch.tensor(0.0, device=device)
            )

            # Stochastic sampling (gamma)
            gamma = 0.0
            if s_tmin <= sigma <= s_tmax:
                gamma = min(s_churn / (len(scheduler.sigmas) - 1), 2**0.5 - 1)
            sigma_hat = sigma * (gamma + 1)
            sigma_hat = sigma_hat.to(device)

            if gamma > 0:
                noise = torch.randn_like(latents, generator=generator, dtype=latents.dtype)
                latents = latents + noise * s_noise * (sigma_hat**2 - sigma**2) ** 0.5

            # Precondition
            sigma_view = sigma_hat.view(-1, 1, 1, 1).expand(batch_size, -1, -1, -1).to(device)
            model_input, timestep_input = edm_clean_image_to_model_input(latents, sigma_view)

            # Fix timestep shape
            if timestep_input.dim() == 0:
                timestep_input = timestep_input.unsqueeze(0).repeat(batch_size)
            elif timestep_input.dim() > 1:
                timestep_input = timestep_input.flatten()
            if timestep_input.shape[0] != batch_size:
                timestep_input = timestep_input[0:1].repeat(batch_size)

            # Concatenate reference channels
            if reference_channels is not None:
                model_input = torch.cat([model_input, reference_channels], dim=1)

            model_input = model_input.to(weight_dtype)
            timestep_input = timestep_input.to(weight_dtype)

            # Zero conditioning for unconditional sampling
            if unconditional_sample:
                if cell_line_labels is not None:
                    cell_line_labels = torch.zeros_like(cell_line_labels)
                if protein_labels is not None:
                    protein_labels = torch.zeros_like(protein_labels)

            # Forward
            model_output = model(
                model_input,
                timestep_input,
                protein_labels=protein_labels,
                cell_line_labels=cell_line_labels,
                encoder_hidden_states=None,
            ).sample

            # Denoise
            predicted_x_start = edm_model_output_to_x_0_hat(latents, sigma_view, model_output)

            step_sigma = (sigma - sigma_next).to(device)
            step_size = step_sigma / sigma
            direction = (predicted_x_start - latents) / sigma_view
            latents = latents + step_size.view(-1, 1, 1, 1) * sigma_view * direction

    return latents
