import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import pickle
from diffusers import AutoencoderKL
from tqdm import tqdm

class LocationClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, model_type='vit_small_patch16_224'):
        super().__init__()
         
        self.model_type = model_type
        
        # Load pretrained ViT from timm
        # Available options include:
        # - vit_tiny_patch16_224
        # - vit_small_patch16_224
        # - vit_base_patch16_224
        # - deit_tiny_patch16_224
        # - deit_small_patch16_224
        self.vit = timm.create_model(
            model_type, 
            pretrained=pretrained, 
            num_classes=num_classes,
            in_chans=32  # timm supports direct specification of input channels
        )
        
        # Model expects 224x224 input, so add an adapter for 32x32
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Input shape: [batch_size, 32, 32, 32]
        x = self.resize(x)  # Resize to [batch_size, 32, 224, 224]
        x1 = self.vit(x)
        return x1

class Trainer:
    def __init__(self, model, train_loader, val_loader, config=None):
        # Initialize config with defaults if not provided
        self.config = {
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'num_epochs': 10,
            'batch_size': 32,
            'model_type': 'vit_small_patch16_224',
            'optimizer': 'Adam',
            'scheduler': 'linear_with_warmup',
            'input_channels': 32,
            'input_size': 32,
        } if config is None else config
        
        # Initialize project tracking
        # Note: With accelerator.log, explicit wandb init is not needed
        # The tracking will be handled by the accelerator
        
        # Modified accelerator initialization
        accelerator_project_config = None  # Define this if needed
        args = type('Args', (), {
            'gradient_accumulation_steps': 1,  # Default value, adjust as needed
            'mixed_precision': 'bf16',  # Default value, adjust as needed
            'logger': 'wandb'  # Using wandb as logger
        })()
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.logger,
            project_config=accelerator_project_config
        )
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Define weight_dtype before using it
        weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        vae = AutoencoderKL.from_pretrained("/scratch/groups/emmalu/marvinli/twisted_diffusion/stable-diffusion-3.5-large-turbo/vae")
        #disable gradients, move to accelerator device, convert to the same precision as model
        vae.to(self.accelerator.device)
        vae.to(weight_dtype)
        vae.requires_grad_(False)
        self.vae = vae
        self.weight_dtype = weight_dtype

        # Initialize optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Prepare everything with accelerator
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Model architecture logging is handled through accelerator
        # No need for explicit wandb.watch with accelerator.log approach

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        # Create a progress bar for the training loop
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Train]",
            leave=True,
            dynamic_ncols=True
        )
        
        for batch_idx, data in enumerate(progress_bar):
            target = data['annotation']

            cond_images = data['cond_image'].to(self.weight_dtype)
            gt_images = data["gt_image"].repeat(1, 3, 1, 1).to(self.weight_dtype)
            cond_images_latent = self.vae.encode(cond_images).latent_dist.sample()
            gt_images_latent = self.vae.encode(gt_images).latent_dist.sample()

            images = torch.concat([gt_images_latent, cond_images_latent], dim=1).to(self.weight_dtype)
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, target)
            
            self.accelerator.backward(loss)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar with current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Log training metrics using accelerator.log
            if batch_idx % 10 == 0:
                self.accelerator.log({
                    'train/batch_loss': loss.item(),
                    'train/batch': batch_idx + epoch * len(self.train_loader),
                    'train/epoch': epoch,
                })
        
        avg_loss = total_loss / len(self.train_loader)
        self.accelerator.log({
            'train/epoch_loss': avg_loss,
            'train/epoch': epoch,
        })
        
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        correct = 0
        all_preds = []
        all_targets = []
        
        # Create a progress bar for the validation loop
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Valid]",
            leave=True,
            dynamic_ncols=True
        )
        
        with torch.no_grad():
            for data in progress_bar:
                target = data['annotation']

                cond_images = data['cond_image'].to(self.weight_dtype)
                gt_images = data["gt_image"].repeat(1, 3, 1, 1).to(self.weight_dtype)
                
                cond_images_latent = self.vae.encode(cond_images).latent_dist.sample()
                gt_images_latent = self.vae.encode(gt_images).latent_dist.sample()

                images = torch.concat([gt_images_latent, cond_images_latent], dim=1).to(self.weight_dtype)

                output = self.model(images)
                batch_loss = self.criterion(output, target).item()
                val_loss += batch_loss
                
                # For binary classification
                pred = (output > 0).float()
                batch_correct = (pred == target).sum().item()
                correct += batch_correct
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
                
                # Update progress bar with current loss
                batch_accuracy = batch_correct / (target.size(0) * target.size(1))
                progress_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_accuracy:.4f}")
        
        val_loss /= len(self.val_loader)
        accuracy = correct / (len(self.val_loader.dataset) * target.size(1))  # Adjusting for multi-label
        
        # Log validation metrics using accelerator.log
        self.accelerator.log({
            'val/loss': val_loss,
            'val/accuracy': accuracy,
            'val/epoch': epoch,
        })
        
        return val_loss, accuracy

    def save_model(self, epoch, val_loss):
        # Save model checkpoint locally
        checkpoint_path = Path('checkpoints_classifier') / f'model_epoch_{epoch}.pth'
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'config': self.config,
        }
        
        self.accelerator.save(checkpoint, checkpoint_path)

    def train(self):
        best_val_loss = float('inf')
        
        # Create a progress bar for epochs
        epoch_progress = tqdm(
            range(self.config['num_epochs']), 
            desc="Training Progress",
            leave=True,
            dynamic_ncols=True
        )
        
        for epoch in epoch_progress:
            train_loss = self.train_epoch(epoch)
            val_loss, val_accuracy = self.validate(epoch)
            
            # Update epoch progress bar with metrics
            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_accuracy:.4f}"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch, val_loss)
                epoch_progress.write(f"✓ Checkpoint saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
            
            # Print a summary for this epoch
            self.accelerator.print(
                f"\nEpoch {epoch+1}/{self.config['num_epochs']} Summary:\n"
                f"Train Loss: {train_loss:.4f}\n"
                f"Val Loss: {val_loss:.4f}\n"
                f"Val Accuracy: {val_accuracy:.4f}\n"
            )

def main():
    # Define your hyperparameters and configuration
    config = {
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'num_epochs': 10,
        'batch_size': 256,
        'num_classes': 36,  # Replace with your number of classes
        'model_type': 'vit_small_patch16_224',  # Using timm's smaller ViT variant
        'optimizer': 'AdamW',
        'scheduler': 'linear_with_warmup',
        'input_channels': 32,
        'input_size': 32,
    }
    
    # Initialize your model
    model = LocationClassifier(
        num_classes=config['num_classes'],
        pretrained=True,
        model_type=config['model_type']
    )
    
    train_dataset = FullFieldDataset(data_root='/scratch/groups/emmalu/multimodal_phenotyping/prot_imp/datasets/training_images')
    test_dataset = FullFieldDataset(data_root='/scratch/groups/emmalu/multimodal_phenotyping/prot_imp/datasets/test_images')
    
    # Initialize your dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Initialize trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()