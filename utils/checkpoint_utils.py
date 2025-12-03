import os
import shutil
from pathlib import Path
from diffusers import DiffusionPipeline, UNet2DConditionModel
from models.dit import DiTTransformer2DModelWithCrossAttention
from diffusers.training_utils import EMAModel
from huggingface_hub import create_repo, upload_folder
import torch
def setup_checkpoint_hooks(accelerator, args, ema_model=None):
    """
    Set up custom hooks for saving and loading checkpoints with accelerate.
    
    Args:
        accelerator: Accelerator instance
        args: Training arguments
        ema_model: Optional EMA model
         
    Returns:
        accelerator: Modified accelerator with hooks registered
    """
    # Custom saving hook for accelerate
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # Save EMA model if used
            if args.use_ema and ema_model is not None:
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))
                
            # Save each model
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))
                
                # Pop weight so model is not saved again
                weights.pop()

    # Custom loading hook for accelerate
    def load_model_hook(models, input_dir):
        # Load EMA model if used
        if args.use_ema and ema_model is not None:
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), DiTTransformer2DModelWithCrossAttention)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model
        if args.use_ema:
            try:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), DiTTransformer2DModelWithCrossAttention)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                accelerator.ema_model = ema_model
                del load_model
            except:
                print("No EMA model found")
        # Load main model
        for i in range(len(models)):
            # Pop model to avoid loading it again
            model = models.pop()
            
            # Load diffusers style into model
            load_model = DiTTransformer2DModelWithCrossAttention.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

    # Register hooks with accelerator
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    return accelerator

def save_checkpoint(accelerator, args, global_step, models=None, save_path=None):
    """
    Save a training checkpoint.
    
    Args:
        accelerator: Accelerator instance
        args: Training arguments
        global_step (int): Current global step
        models: Optional models to save separately
        save_path (str): Optional custom save path
        
    Returns:
        str: Path where checkpoint was saved
    """
    if not accelerator.is_main_process:
        return None
        
    # Use default path if none provided
    if save_path is None:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    
    # Save state using accelerator
    accelerator.save_state(save_path)
    
    # If models provided, save them separately
    if models is not None and isinstance(models, dict):
        for name, model in models.items():
            model_path = os.path.join(save_path, name)
            os.makedirs(model_path, exist_ok=True)
            model.save_pretrained(model_path)
    
    return save_path

def cleanup_checkpoints(output_dir, checkpoints_total_limit):
    """
    Clean up old checkpoints to stay under the total limit.
    
    Args:
        output_dir (str): Output directory containing checkpoints
        checkpoints_total_limit (int): Maximum number of checkpoints to keep
    """
    if checkpoints_total_limit is None or checkpoints_total_limit <= 0:
        return
        
    # Find all checkpoint directories
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    
    # Sort by step number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    
    # If we're over the limit, remove oldest checkpoints
    if len(checkpoints) > checkpoints_total_limit:
        num_to_remove = len(checkpoints) - checkpoints_total_limit
        removing_checkpoints = checkpoints[0:num_to_remove]
        
        for checkpoint in removing_checkpoints:
            checkpoint_path = os.path.join(output_dir, checkpoint)
            shutil.rmtree(checkpoint_path)

def find_latest_checkpoint(output_dir):
    """
    Find the most recent checkpoint in the output directory.
    
    Args:
        output_dir (str): Directory containing checkpoints
        
    Returns:
        str: Path to the latest checkpoint, or None if no checkpoints found
    """
    # Find all checkpoint directories
    dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    
    if not dirs:
        return None
        
    # Sort by step number and get the latest
    latest_dir = sorted(dirs, key=lambda x: int(x.split("-")[1]))[-1]
    
    return os.path.join(output_dir, latest_dir)

def resume_from_checkpoint(accelerator, args, num_update_steps_per_epoch=0):
    """
    Resume training from a checkpoint.
    
    Args:
        accelerator: Accelerator instance
        args: Training arguments
        
    Returns:
        tuple: (global_step, first_epoch, resume_step)
    """
    global_step = 0
    first_epoch = 0
    resume_step = 0
    
    if args.resume_from_checkpoint is None:
        return global_step, first_epoch, resume_step
        
    # Find checkpoint path
    if args.resume_from_checkpoint == "latest":
        checkpoint_path = find_latest_checkpoint(args.output_dir)
        if checkpoint_path is None:
            return global_step, first_epoch, resume_step
    else:
        checkpoint_path = args.resume_from_checkpoint
    
    # Load state
    accelerator.load_state(checkpoint_path)
    
    # Extract step from checkpoint name
    if os.path.isdir(checkpoint_path):
        global_step = int(checkpoint_path.split("-")[-1])
        
        # Calculate epoch and step to resume from
        #num_update_steps_per_epoch = args.get("num_update_steps_per_epoch", 0)
        if num_update_steps_per_epoch > 0:
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch
    
    return global_step, first_epoch, resume_step
def transfer_from_checkpoint(args, model, ema_model):
    """
    transfer training from a checkpoint.
    
    Args:
        accelerator: Accelerator instance
        args: Training arguments
        
    Returns:
        tuple: (global_step, first_epoch, resume_step)
    """
    
    if args.transfer_weights is None:
        return model, ema_model
    else:
        input_dir = args.transfer_weights
    
    
    # Load diffusers style into model
    # Fix: Add low_cpu_mem_usage=False to ensure actual tensors are loaded
    load_model = DiTTransformer2DModelWithCrossAttention.from_pretrained(
        input_dir, 
        subfolder="unet", 
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,  # This ensures actual tensors are loaded
        torch_dtype=torch.float32  # Specify dtype to avoid meta tensors
    )

    # Load state dict with key matching
    load_state_dict = load_model.state_dict()
    model_state_dict = model.state_dict()
    
    # Find matching keys
    matched_keys = []
    mismatched_keys = []
    missing_keys = []
    unexpected_keys = []
    
    for key in model_state_dict.keys():
        if key in load_state_dict:
            if model_state_dict[key].shape == load_state_dict[key].shape:
                matched_keys.append(key)
            else:
                mismatched_keys.append((key, model_state_dict[key].shape, load_state_dict[key].shape))
        else:
            missing_keys.append(key)
    
    for key in load_state_dict.keys():
        if key not in model_state_dict:
            unexpected_keys.append(key)
    
    # Load only matching keys
    filtered_state_dict = {k: load_state_dict[k] for k in matched_keys}
    model.load_state_dict(filtered_state_dict, strict=False)
    
    # Report mismatches
    if mismatched_keys:
        print(f"Mismatched keys (skipped): {len(mismatched_keys)}")
        for key, model_shape, load_shape in mismatched_keys[:5]:  # Show first 5
            print(f"  {key}: model {model_shape} vs checkpoint {load_shape}")
    if missing_keys:
        print(f"Missing keys in checkpoint: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")
    print(f"Successfully loaded {len(matched_keys)} matching keys")
    del load_model
    if args.use_ema and ema_model is not None:
        try:
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), DiTTransformer2DModelWithCrossAttention)
            ema_model.load_state_dict(load_model.state_dict())
            del load_model
        except:
            ema_model = EMAModel(
                model.parameters(),
                decay=args.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=args.ema_inv_gamma,
                power=args.ema_power,
                model_cls=type(model),
                model_config=model.config,
            )
            print("no ema found. init ema with model weights")
    return model, ema_model

def save_pipeline_to_hub(args, model, scheduler, output_dir):
    """
    Save the pipeline to the Hugging Face Hub.
    
    Args:
        args: Training arguments
        model: UNet model
        scheduler: Noise scheduler
        output_dir (str): Output directory
    """
    # Only proceed if pushing to hub is enabled
    if not args.push_to_hub:
        return
        
    # Create or get the repo
    repo_id = args.hub_model_id or Path(args.output_dir).name
    repo_id = create_repo(
        repo_id=repo_id,
        exist_ok=True,
        token=args.hub_token,
        private=args.hub_private_repo,
    ).repo_id
    
    # Create pipeline
    pipeline = DiffusionPipeline(
        unet=model,
        scheduler=scheduler,
    )
    
    # Save locally
    pipeline.save_pretrained(output_dir)
    
    # Upload to hub
    upload_folder(
        repo_id=repo_id,
        folder_path=args.output_dir,
        commit_message=f"Upload model",
        ignore_patterns=["step_*", "epoch_*"],
    )