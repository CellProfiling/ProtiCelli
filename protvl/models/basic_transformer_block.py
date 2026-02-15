"""Custom BasicTransformerBlock with continuous AdaLayerNorm for ProtVL.

Extends the diffusers BasicTransformerBlock to support ``ada_norm_zero_continuous``
normalization, which uses a continuous embedding (concatenated protein + cell line
embeddings) instead of discrete class labels.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from diffusers.models.attention import BasicTransformerBlock as _BaseBlock
from diffusers.models.normalization import LayerNorm, FP32LayerNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph


class AdaLayerNormZeroContinuous(nn.Module):
    """Adaptive LayerNorm-Zero with continuous conditioning.

    Produces shift/scale/gate parameters from a continuous embedding
    vector (e.g. concatenated protein + cell line embeddings) rather
    than discrete class labels.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the conditioning embedding.
    norm_type : str
        Underlying norm type: ``"layer_norm"`` or ``"fp32_layer_norm"``.
    bias : bool
        Whether the linear projection uses bias.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_type: str = "layer_norm",
        bias: bool = True,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported norm_type '{norm_type}'. "
                "Use 'layer_norm' or 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


@maybe_allow_in_graph
class BasicTransformerBlock(_BaseBlock):
    """Transformer block with ``ada_norm_zero_continuous`` support.

    Inherits from ``diffusers.models.attention.BasicTransformerBlock`` and
    overrides normalization to support continuous AdaLN-Zero conditioning
    via a concatenated protein + cell line embedding.

    The extra constructor parameters ``num_protein_labels`` and
    ``num_cell_labels`` are accepted for API compatibility with
    ``DiTTransformer2DModel`` but are not used directly here (the
    embedding happens at the model level).
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_protein_labels: int = 12810,
        num_cell_labels: int = 42,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        # The parent class doesn't know about "ada_norm_zero_continuous",
        # so we pass "ada_norm_zero" to it and override norm1 afterwards.
        super().__init__(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            only_cross_attention=only_cross_attention,
            double_self_attention=double_self_attention,
            upcast_attention=upcast_attention,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_type=(
                "ada_norm_zero"
                if norm_type == "ada_norm_zero_continuous"
                else norm_type
            ),
            norm_eps=norm_eps,
            final_dropout=final_dropout,
            attention_type=attention_type,
            positional_embeddings=positional_embeddings,
            num_positional_embeddings=num_positional_embeddings,
            ada_norm_continous_conditioning_embedding_dim=ada_norm_continous_conditioning_embedding_dim,
            ada_norm_bias=ada_norm_bias,
            ff_inner_dim=ff_inner_dim,
            ff_bias=ff_bias,
            attention_out_bias=attention_out_bias,
        )

        # Replace norm layers for the continuous variant
        if norm_type == "ada_norm_zero_continuous":
            self.norm_type = "ada_norm_zero_continuous"
            self.norm1 = AdaLayerNormZeroContinuous(dim)
            self.norm2 = LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )
            self.norm3 = LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        embedding: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Sequence of patch embeddings ``[B, N, D]``.
        attention_mask : torch.Tensor, optional
            Self-attention mask.
        encoder_hidden_states : torch.Tensor, optional
            Cross-attention context.
        encoder_attention_mask : torch.Tensor, optional
            Cross-attention mask.
        embedding : torch.Tensor, optional
            Continuous conditioning embedding (concatenated protein + cell line).
        cross_attention_kwargs : dict, optional
            Extra kwargs for attention processors.
        added_cond_kwargs : dict, optional
            Additional conditioning kwargs.

        Returns
        -------
        torch.Tensor
            Transformed hidden states.
        """
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            cross_attention_kwargs.pop("scale", None)

        batch_size = hidden_states.shape[0]

        # ---- Norm 1 + Self-Attention --------------------------------- #
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ("layer_norm", "layer_norm_i2vgen"):
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        elif self.norm_type == "ada_norm_zero_continuous":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, embedding, hidden_dtype=hidden_states.dtype
            )
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # GLIGEN support
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.norm_type in ("ada_norm_zero", "ada_norm_zero_continuous"):
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # ---- Norm 2 + Cross-Attention -------------------------------- #
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in (
                "ada_norm_zero_continuous",
                "ada_norm_zero",
                "layer_norm",
                "layer_norm_i2vgen",
            ):
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(
                    hidden_states, added_cond_kwargs["pooled_text_emb"]
                )
            else:
                raise ValueError(f"Unsupported norm_type: {self.norm_type}")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # ---- Norm 3 + Feed-Forward ----------------------------------- #
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif self.norm_type != "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type in ("ada_norm_zero", "ada_norm_zero_continuous"):
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            from diffusers.models.attention import _chunked_feed_forward

            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type in ("ada_norm_zero", "ada_norm_zero_continuous"):
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states
