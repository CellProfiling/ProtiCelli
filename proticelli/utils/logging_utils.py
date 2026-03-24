"""Logging utilities for ProtiCelli training."""

import logging
import diffusers


def setup_logging(name: str) -> logging.Logger:
    """Set up a logger with the given name."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(name)


def configure_diffusers_logging(is_main_process: bool):
    """Configure diffusers logging verbosity."""
    if is_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()


def log_training_parameters(logger, args, total_batch_size, num_update_steps, max_train_steps, dataset_size):
    """Log training configuration."""
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {dataset_size}")
    logger.info(f"  Num epochs = {args.num_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
