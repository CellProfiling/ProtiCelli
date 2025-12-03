import os
import logging
from accelerate.logging import get_logger

def setup_logging(name, log_level="INFO"):
    """
    Set up a logger with the specified name and level.
    
    Args:
        name (str): Logger name
        log_level (str): Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        
    Returns:
        logger: Configured logger
    """
    logger = get_logger(name, log_level=log_level)
    
    # Configure basic logging format
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=getattr(logging, log_level), 
    )
    
    return logger

def configure_diffusers_logging(is_main_process):
    """
    Configure the diffusers library logging.
    
    Args:
        is_main_process (bool): Whether this is the main process
    """
    import diffusers.utils.logging
    
    if is_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

def log_training_parameters(logger, args, total_batch_size, num_update_steps_per_epoch, max_train_steps, dataset_size):
    """
    Log the training parameters.
    
    Args:
        logger: Logger to use
        args: Training arguments
        total_batch_size (int): Total batch size
        num_update_steps_per_epoch (int): Number of update steps per epoch
        max_train_steps (int): Maximum number of training steps
        dataset_size (int): Size of the dataset
    """
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {dataset_size}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
def setup_tensorboard_logging(output_dir, logging_dir):
    """
    Set up tensorboard logging directory.
    
    Args:
        output_dir (str): Output directory for model checkpoints
        logging_dir (str): Logging directory
        
    Returns:
        str: Full path to the logging directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logging directory
    full_logging_dir = os.path.join(output_dir, logging_dir)
    os.makedirs(full_logging_dir, exist_ok=True)
    
    return full_logging_dir

def log_images_to_tensorboard(tracker, images, epoch, tag="test_samples"):
    """
    Log images to tensorboard.
    
    Args:
        tracker: Tensorboard tracker
        images (numpy.ndarray): Images to log of shape [N, H, W, C]
        epoch (int): Current epoch
        tag (str): Tag for the images
    """
    # Processing images for tensorboard
    # TensorBoard expects images in [N, C, H, W] format
    images_processed = (images * 255).round().astype("uint8")
    images_for_tb = images_processed.transpose(0, 3, 1, 2)
    
    # Log to tensorboard
    tracker.add_images(tag, images_for_tb, epoch)

def log_images_to_wandb(tracker, images, epoch, global_step, tag="test_samples"):
    """
    Log images to wandb.
    
    Args:
        tracker: W&B tracker
        images (numpy.ndarray): Images to log
        epoch (int): Current epoch
        global_step (int): Global step
        tag (str): Tag for the images
    """
    import wandb
    
    # Processing images for wandb
    images_processed = (images * 255).round().astype("uint8")
    
    # Log to wandb
    tracker.log(
        {tag: [wandb.Image(img) for img in images_processed], "epoch": epoch},
        step=global_step,
    )