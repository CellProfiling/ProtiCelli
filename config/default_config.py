import argparse
import os

def parse_args():
    """Parse command line arguments for diffusion model training."""
    parser = argparse.ArgumentParser(description="Diffusion model training script.")
    
    # Dataset parameters
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the Dataset (from the HuggingFace hub) to train on."
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config."
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored."
    )
    
    # Model parameters
    parser.add_argument(
        "--use_VIT",
        type=bool,
        default=True,
        help="If true, using vit for backbone, else use unet."
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard configuration."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="The resolution for input images, all images will be resized to this resolution."
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Whether to center crop the input images to the resolution."
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="Whether to randomly flip images horizontally."
    )
    
    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--overwrite_output_dir", 
        action="store_true",
        help="Overwrite the content of the output directory."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory."
    )
    
    # Training parameters
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=16, 
        help="Batch size per device for training."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=16, 
        help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading."
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=100,
        help="Total number of training epochs."
    )
    parser.add_argument(
        "--save_images_epochs", 
        type=int, 
        default=10, 
        help="How often to save images during training."
    )
    parser.add_argument(
        "--save_model_epochs", 
        type=int, 
        default=10, 
        help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    
    # Optimizer parameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="The scheduler type to use. Choose between [linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup]"
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=500, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--adam_beta1", 
        type=float, 
        default=0.95, 
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", 
        type=float, 
        default=0.999, 
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", 
        type=float, 
        default=1e-6, 
        help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_epsilon", 
        type=float, 
        default=1e-08, 
        help="Epsilon value for the Adam optimizer."
    )
    
    # EMA parameters
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights."
    )
    parser.add_argument(
        "--ema_inv_gamma", 
        type=float, 
        default=1.0, 
        help="The inverse gamma value for the EMA decay."
    )
    parser.add_argument(
        "--ema_power", 
        type=float, 
        default=3/4, 
        help="The power value for the EMA decay."
    )
    parser.add_argument(
        "--ema_max_decay", 
        type=float, 
        default=0.9995, 
        help="The maximum decay magnitude for EMA."
    )
    
    # Diffusion model parameters
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the noise error or directly the reconstructed image."
    )
    parser.add_argument(
        "--ddpm_num_steps", 
        type=int, 
        default=1000,
        help="Number of denoising steps."
    )
    parser.add_argument(
        "--ddpm_num_inference_steps", 
        type=int, 
        default=1000,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--ddpm_beta_schedule", 
        type=str, 
        default="linear",
        help="Beta schedule for the noise."
    )
    
    # Hugging Face Hub parameters
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_token", 
        type=str, 
        default=None, 
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local output_dir."
    )
    parser.add_argument(
        "--hub_private_repo", 
        action="store_true", 
        help="Whether to create a private repository."
    )
    
    # Logging parameters
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help="Whether to use tensorboard or wandb for experiment tracking and logging."
    )
    
    parser.add_argument(
        "--transfer_weights",
        type=str,
        default=None,
        help="Whether training should be initialized from a previous checkpoint."
    )


    
    # Checkpointing parameters
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates."
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint."
    )
    
    # Miscellaneous parameters
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision."
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether to use xformers."
    )
    
    args = parser.parse_args()
    
    # Set up environment variables for distributed training
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    # Validate arguments
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")
    
    return args

# EDM-specific constants
EDM_CONFIG = {
    "SIGMA_MIN": 0.002,  # Minimum noise level
    "SIGMA_MAX": 80.0,   # Maximum noise level
    "SIGMA_DATA": 0.5,   # Standard deviation of the data distribution
    "RHO": 7            # EDM-specific parameter
}