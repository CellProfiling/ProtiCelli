import torch
import torch.nn as nn
import open_clip
import os
from PIL import Image
model_name_or_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
class mc cp -r dataset/images_v2 runai-subcell-vs/subcell-vs-pr
ovl-pvc(nn.Module):
    """
    Wrapper for OpenCLIP vision encoder to extract the second-to-last layer features.
    
    This class loads the vision encoder portion of an OpenCLIP model
    and extracts the second-to-last layer representation for conditioning.
    """
    
    def __init__(self, model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
                device=None, weight_dtype=None,
                get_last_n_layer=1):
        """
        Initialize the OpenCLIP vision encoder.
        
        Args:
            model_name (str): The name of the OpenCLIP model architecture
            pretrained (str): The name of the pretrained weights
            device: Device to load the model on
        """
        super().__init__()
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if weight_dtype is None:
            self.weight_dtype = torch.float32
        self.model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.get_last_n_layer = get_last_n_layer
        # Freeze the model parameters
        for param in self.model.visual.parameters():
            param.requires_grad = False
            
        # Move model to device
        self.model.visual.to(self.device)
        self.model.visual.eval()
        
        # Get output dimension of the model
        # For ViT-B-16, this should be 768
        #self.output_dim = self.model.visual.output_dim
        
        # Save hooks for extracting intermediate representations
        self.hooks = []
        self.saved_features = {}
        
        # Register hooks to capture intermediate outputs
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate layer outputs"""
        # Clear existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        n = self.get_last_n_layer
        # Function to capture outputs
        def hook_fn(name):
            def _hook(module, input, output):
                self.saved_features[name] = output
            return _hook
        
        # Register hook on the second-to-last transformer block
        # For most OpenCLIP vision transformers, blocks are indexed
        second_last_block = len(self.model.visual.trunk.blocks) - n
        self.hooks.append(
            self.model.visual.trunk.blocks[second_last_block].register_forward_hook(
                hook_fn(f"block_{second_last_block}")
            )
        )
    
    def get_layer(self):
        """
        Extract the second-to-last layer representation from captured features.
        
        Returns:
            torch.Tensor: Second-to-last layer representation
        """
        n = self.get_last_n_layer
        # Get the feature from the second-to-last block
        second_last_block = len(self.model.visual.trunk.blocks) - n
        block_name = f"block_{second_last_block}"
        
        if block_name not in self.saved_features:
            raise ValueError("Second-to-last layer features not captured. Run forward pass first.")
        
        # Get features and extract the CLS token (first token)
        features = self.saved_features[block_name]
        
        # Note: The first token is usually the CLS token in transformer models
        # We want the second-to-last layer features, so we take all tokens except the CLS token
        embedding = features[:, 1:, :]
        
        return embedding
    
    def forward(self, x):
        """
        Process images and extract the second-to-last layer embeddings.
        
        Args:
            images: Input images (PIL images or tensors)
            
        Returns:
            torch.Tensor: Second-to-last layer representations
        """
        # Clear saved features
        self.saved_features = {}
    
        
        # Extract features with second-to-last hidden state
        with torch.no_grad():
            # Run model forward pass to collect features via hooks
            _ = self.model.visual(x)
            
            # Get second-to-last layer representation
            features = self.get_layer()
            
        return features
    
    def encode_images(self, x, batch_size=32):
        """
        Encode images in batches to avoid memory issues.
        
        Args:
            images: List of images or tensor of images
            batch_size: Batch size for processing
            
        Returns:
            torch.Tensor: Encoded features from the second-to-last layer
        """

        
        return self.forward(x)
            
    def __del__(self):
        """Clean up hooks when the object is deleted"""
        for hook in self.hooks:
            hook.remove()