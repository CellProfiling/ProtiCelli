import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

def save_images(images, output_dir="generated_images", prefix="sample", format="png"):
    """
    Save a batch of images to disk.
    
    Args:
        images (torch.Tensor or list): Batch of images to save. Can be a tensor of shape [B, C, H, W] 
                                       or a list of individual images.
        output_dir (str): Directory to save images to. Will be created if it doesn't exist.
        prefix (str): Prefix for the saved image filenames.
        format (str): Image format to save as (png, jpg, etc.)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
     
    # Convert to list if not already a list
    if not isinstance(images, list):
        images = [images]
    
    # Save each image
    for i, img in enumerate(images):
        # Convert tensor to numpy if needed
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        
        # Handle different shapes and channel configurations
        if img.ndim == 4 and img.shape[0] == 1:  # [1, C, H, W]
            img = img[0]
        
        if img.shape[0] == 3 or img.shape[0] == 1:  # [C, H, W]
            img = img.transpose(1, 2, 0)
        
        # Scale to 0-255 range if in 0-1 range
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Create PIL image based on number of channels
        if img.shape[-1] == 1:  # Single channel
            img = img.squeeze(-1)
            pil_img = Image.fromarray(img, mode='L')
        elif img.shape[-1] == 3:  # RGB
            pil_img = Image.fromarray(img, mode='RGB')
        else:
            raise ValueError(f"Unsupported number of channels: {img.shape[-1]}")
        
        # Save the image
        filename = f"{prefix}_{i:04d}.{format}"
        filepath = os.path.join(output_dir, filename)
        pil_img.save(filepath)
    
    print(f"Saved {len(images)} images to {output_dir}")


def plot_and_save_images(images, output_dir=None, prefix=None, format="png", row_title=None, **kwargs):
    """
    Plot images and optionally save them to disk.
    
    Args:
        images: Images to plot/save
        output_dir: If provided, save images to this directory
        prefix: Prefix for saved image filenames
        format: Image format for saving
        row_title: Title for the plot
        **kwargs: Additional arguments for matplotlib
    """
    # Plot the images first
    if not isinstance(images, list):
        images = [images]
    
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(12, 12 // num_images))
    
    if num_images == 1:
        axs = [axs]
    
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        
        # Handle different shapes and channel configurations
        if img.ndim == 4 and img.shape[0] == 1:  # [1, C, H, W]
            img = img[0]
        
        if img.shape[0] == 3 or img.shape[0] == 1:  # [C, H, W]
            img = img.transpose(1, 2, 0)
        
        if img.shape[-1] == 1:  # Single channel
            img = img.squeeze(-1)
            axs[i].imshow(img, cmap='gray')
        else:  # RGB
            axs[i].imshow(img)
        
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    
    if row_title is not None:
        plt.suptitle(row_title)
    
    plt.tight_layout()
    
    # Save images if output_dir is provided
    if output_dir and prefix:
        # Save the plot as a single image
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{prefix}_plot.{format}")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Also save individual images
        save_images(images, output_dir=output_dir, prefix=prefix, format=format)
    
    plt.show() 