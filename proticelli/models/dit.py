"""DiT (Diffusion Transformer) model for ProtiCelli.

A 2D Transformer model based on DiT (https://arxiv.org/abs/2212.09748),
extended with dual label embeddings for protein and cell line conditioning.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.utils import is_torch_version, logging
from diffusers.models.embeddings import PatchEmbed, CombinedTimestepLabelEmbeddings
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from .basic_transformer_block import BasicTransformerBlock

logger = logging.get_logger(__name__)


class DiTTransformer2DModel(ModelMixin, ConfigMixin):
    """2D Transformer model with protein and cell line conditioning.

    Parameters
    ----------
    num_attention_heads : int
        Number of attention heads.
    attention_head_dim : int
        Dimension per attention head.
    in_channels : int
        Input channels (latent channels + conditioning channels).
    out_channels : int, optional
        Output channels. Defaults to ``in_channels``.
    num_layers : int
        Number of transformer blocks.
    dropout : float
        Dropout probability.
    norm_num_groups : int
        Groups for group normalization.
    attention_bias : bool
        Whether attention layers use bias.
    sample_size : int
        Spatial size of the latent input.
    patch_size : int
        Patch size for patchification.
    activation_fn : str
        Activation function in feed-forward blocks.
    num_embeds_ada_norm : int
        Number of AdaLayerNorm embeddings.
    upcast_attention : bool
        Whether to upcast attention to float32.
    norm_type : str
        Normalization type (e.g. ``"ada_norm_zero_continuous"``).
    norm_elementwise_affine : bool
        Element-wise affine in normalization.
    norm_eps : float
        Epsilon for normalization.
    cross_attention_dim : int, optional
        Cross-attention dimension (None to disable).
    num_protein_labels : int
        Size of protein label vocabulary (including null label at 0).
    num_cell_labels : int
        Size of cell line label vocabulary (including null label at 0).
    positional_embeddings : torch.Tensor, optional
        Custom positional embeddings.
    """

    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _supports_gradient_checkpointing = True
    _supports_group_offloading = False

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        attention_bias: bool = True,
        sample_size: int = 32,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_zero",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        cross_attention_dim=None,
        num_protein_labels: int = 13349,
        num_cell_labels: int = 41,
        positional_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False

        # Dual label embeddings: protein + cell line
        self.emb_protein_labels = CombinedTimestepLabelEmbeddings(
            num_protein_labels, self.inner_dim // 2
        )
        self.emb_cell_line_labels = CombinedTimestepLabelEmbeddings(
            num_cell_labels, self.inner_dim // 2
        )

        # Patch embedding
        self.height = self.config.sample_size
        self.width = self.config.sample_size
        self.patch_size = self.config.patch_size
        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
        )

        # Optional cross-attention projection
        self.encoder_hidden_states_compress = None
        if self.config.cross_attention_dim is not None:
            self.encoder_hidden_states_compress = nn.Sequential(
                nn.Linear(cross_attention_dim, cross_attention_dim // 2),
                nn.SiLU(),
                nn.Linear(cross_attention_dim // 2, cross_attention_dim // 4),
            )
            self.cross_attention_dim = cross_attention_dim // 4

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    cross_attention_dim=self.config.cross_attention_dim,
                    num_protein_labels=num_protein_labels,
                    num_cell_labels=num_cell_labels,
                    positional_embeddings=positional_embeddings,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # Output layers
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(
            self.inner_dim,
            self.config.patch_size * self.config.patch_size * self.out_channels,
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        protein_labels: Optional[torch.Tensor] = None,
        cell_line_labels: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput:
        """Forward pass.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input latents ``[B, C, H, W]``.
        timestep : torch.LongTensor
            Denoising timestep (EDM c_noise).
        protein_labels : torch.Tensor
            Protein label indices ``[B]``.
        cell_line_labels : torch.Tensor
            Cell line label indices ``[B]``.
        encoder_hidden_states : torch.Tensor, optional
            Cross-attention conditioning.
        cross_attention_kwargs : dict, optional
            Extra kwargs for attention processors.
        return_dict : bool
            Whether to return a dataclass or tuple.

        Returns
        -------
        Transformer2DModelOutput or tuple
        """
        # Embed labels
        emb_protein = self.emb_protein_labels(
            timestep, protein_labels, hidden_dtype=timestep.dtype
        )
        emb_cell_line = self.emb_cell_line_labels(
            timestep, cell_line_labels, hidden_dtype=timestep.dtype
        )
        emb = torch.cat([emb_protein, emb_cell_line], dim=1)

        # Patchify + position embed
        height = hidden_states.shape[-2] // self.patch_size
        width = hidden_states.shape[-1] // self.patch_size
        hidden_states = self.pos_embed(hidden_states)

        # Optional cross-attention compression
        if self.encoder_hidden_states_compress is not None:
            encoder_hidden_states = self.encoder_hidden_states_compress(
                encoder_hidden_states
            )

        # Transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward

                ckpt_kwargs = (
                    {"use_reentrant": False}
                    if is_torch_version(">=", "1.11.0")
                    else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    encoder_hidden_states,
                    None,
                    emb,
                    cross_attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    embedding=emb,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

        # Output projection + unpatchify
        conditioning = emb.to(hidden_states.dtype)
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states)

        height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            -1, height, width, self.patch_size, self.patch_size, self.out_channels
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            -1, self.out_channels, height * self.patch_size, width * self.patch_size
        )

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def create_dit_model(config=None, resolution=32):
    """Create a DiT model with default ProtiCelli configuration.

    Parameters
    ----------
    config : dict, optional
        Custom configuration. If None, uses default ProtiCelli settings.
    resolution : int
        Spatial resolution (used for sample_size).

    Returns
    -------
    DiTTransformer2DModel
    """
    if config is None:
        model = DiTTransformer2DModel(
            num_attention_heads=16,
            attention_head_dim=64,
            in_channels=32,
            out_channels=16,
            num_layers=24,
            dropout=0.0,
            norm_num_groups=32,
            attention_bias=True,
            sample_size=64,
            patch_size=2,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            upcast_attention=False,
            norm_type="ada_norm_zero_continuous",
            norm_elementwise_affine=False,
            norm_eps=1e-5,
            cross_attention_dim=None,
            num_protein_labels=12810,
            num_cell_labels=42,
            positional_embeddings=None,
        )
    else:
        model = DiTTransformer2DModel(**config)
    return model
