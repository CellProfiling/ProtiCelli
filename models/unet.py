import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


from diffusers import UNet2DConditionModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel
import timm

class CustomUNetWithEmbeddings(UNet2DConditionModel):
    @register_to_config
    def __init__(
        self, 
        sample_size=32,
        in_channels=32,
        out_channels=32,
        layers_per_block=2,
        block_out_channels=(256, 512, 512),
        down_block_types=("AttnDownBlock2D","AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D","AttnUpBlock2D", "AttnUpBlock2D", ),
        cross_attention_dim=768,
        mid_block_type="UNetMidBlock2DCrossAttn",
        class_embed_type="projection",
        projection_class_embeddings_input_dim=512,
        # Custom parameters
        protein_embedding_dim=384,
        cell_embedding_dim=128,
        num_protein_embeddings=13349,  # 13348 + 1
        num_cell_embeddings=41,        # 40 + 1
        **kwargs
    ):
        super().__init__(
                    sample_size=sample_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    layers_per_block=layers_per_block,
                    block_out_channels=block_out_channels,
                    down_block_types=down_block_types,
                    up_block_types=up_block_types,
                    cross_attention_dim=cross_attention_dim,
                    mid_block_type=mid_block_type,
                    class_embed_type=class_embed_type,
                    projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
                    **kwargs
                )
        # Define embeddings within the model
        self.embedding_protein = nn.Embedding(num_protein_embeddings, protein_embedding_dim, padding_idx=0)
        self.embedding_cell_label = nn.Embedding(num_cell_embeddings, cell_embedding_dim, padding_idx=0)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor,
        protein_labels: torch.Tensor = None, # Accept the specific labels
        cell_line_labels: torch.Tensor = None, # Accept the specific labels
        encoder_hidden_states: torch.Tensor = None,
        class_labels: torch.Tensor = None, # Keep standard arg, but we'll compute it
        **kwargs # Pass other args through
    ):
        # --- Compute combined embedding internally ---
        computed_class_labels = None
        if protein_labels is not None and cell_line_labels is not None:
            protein_embed = self.embedding_protein(protein_labels)
            cell_embed = self.embedding_cell_label(cell_line_labels)
            protein_embed = protein_embed.squeeze()
            cell_embed = cell_embed.squeeze()
            if len(protein_embed.shape) == 1:
                protein_embed = protein_embed.unsqueeze(0)
            if len(cell_embed.shape) == 1:
                cell_embed = cell_embed.unsqueeze(0)
            combined_embed = torch.cat([protein_embed, cell_embed], dim=1)
            computed_class_labels = F.silu(combined_embed)
            
        # If user explicitly passed class_labels, maybe prioritize it? Or raise error?
        # Here, we prioritize the computed one if labels were given.
        final_class_labels = computed_class_labels if computed_class_labels is not None else class_labels

        # --- Call the original UNet2DConditionModel forward ---
        # Pass the computed embedding as 'class_labels'
        return super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=final_class_labels, # Use the computed embedding
            **kwargs # Pass through any other args
        )


class LocationClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, model_type='vit_small_patch16_224'):
        super().__init__()
        
        self.model_type = model_type
        
        # Load pretrained ViT from timm
        self.vit = timm.create_model(
            model_type, 
            pretrained=pretrained, 
            num_classes=num_classes,
            in_chans=16  # timm supports direct specification of input channels
        )
        
        # Model expects 224x224 input, so add an adapter for 32x32
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Input shape: [batch_size, 32, 32, 32]
        x = self.resize(x)  # Resize to [batch_size, 32, 224, 224]
        x1 = self.vit(x)
        return x1
    

def create_unet_model(config=None, resolution=32):
    """
    Create a UNet model for conditional diffusion.
    
    Args:
        config: Optional model configuration.
        resolution: Image resolution expected by the model.
        
    Returns:
        UNet2DConditionModel: The configured UNet model.
    """
    if config is None:
        model = CustomUNetWithEmbeddings(
                sample_size=32,
                in_channels=32,          # Total channels for combined input (gt + cond)
                out_channels=32,         # Output channels should match input dimension
                layers_per_block=2,
                block_out_channels=(256, 512, 512),
                down_block_types=(
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "AttnDownBlock2D",
                ),
                up_block_types=(
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "AttnUpBlock2D",
                ),
                cross_attention_dim = 768,
                
                mid_block_type="UNetMidBlock2DCrossAttn",
                # Add parameters for conditioning
                class_embed_type="projection",
                projection_class_embeddings_input_dim = 512,
            )

    else:
        print("no model")
        # Load configuration from file
        # config = UNet2DConditionModel.load_config(config)
        # model = UNet2DConditionModel.from_config(config)
    
    return model

def setup_model_devices(model, accelerator, weight_dtype, enable_xformers=False):
    """
    Set up model devices and optimizations.
    
    Args:
        model: The model to set up.
        accelerator: Accelerator instance.
        weight_dtype: Data type for weights.
        enable_xformers: Whether to enable xformers for memory-efficient attention.
        
    Returns:
        model: The set up model.
    """
    # Move model to device and set weight type
    model = model.to(accelerator.device)
    model = model.to(weight_dtype)
    
    # Enable xformers if requested and available
    if enable_xformers:
        try:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                model.enable_xformers_memory_efficient_attention()
            else:
                raise ImportError("xformers is not available")
        except ImportError as e:
            print(f"Warning: {e}. xformers not enabled.")
    
    return model

def load_vae(vae_path, accelerator, weight_dtype):
    """
    Load a pre-trained VAE model.
    
    Args:
        vae_path: Path to the VAE model.
        accelerator: Accelerator instance.
        weight_dtype: Data type for weights.
        
    Returns:
        vae: The loaded VAE model.
    """
    from diffusers import AutoencoderKL
    
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.to(accelerator.device)
    vae.to(weight_dtype)
    vae.eval()
    vae.requires_grad_(False)
    
    return vae

def load_classifier(checkpoint_path, accelerator, weight_dtype):
    """
    Load a pre-trained classifier model.
    
    Args:
        checkpoint_path: Path to the classifier checkpoint.
        accelerator: Accelerator instance.
        weight_dtype: Data type for weights.
        
    Returns:
        classifier: The loaded classifier model.
    """
    # from .location_classifier import LocationClassifier
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint['config']
    
    classifier = LocationClassifier(
        num_classes=config['num_classes'],
        pretrained=False,
        model_type=config['model_type']
    )
    
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    classifier.to(accelerator.device)
    classifier.to(weight_dtype)
    classifier.requires_grad_(False)
    
    return classifier

def load_clip_model(clip_model_path, accelerator, weight_dtype):
    """
    Load a pre-trained CLIP model.
    
    Args:
        clip_model_path: Path to the CLIP model.
        accelerator: Accelerator instance.
        weight_dtype: Data type for weights.
        
    Returns:
        clip_model: The loaded CLIP model.
    """
    from .clip_image_encoder import OpenCLIPVisionEncoder
    
    clip_model = OpenCLIPVisionEncoder(
        model_name=clip_model_path,
        device=accelerator.device,
        weight_dtype=weight_dtype
    )
    clip_model.to(accelerator.device)
    clip_model.to(weight_dtype)
    clip_model.eval()
    clip_model.requires_grad_(False)
    return clip_model