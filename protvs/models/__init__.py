"""ProtVS model components.

Exports:
    DiTTransformer2DModel: Main diffusion transformer.
    create_dit_model: Factory function.

Note: Requires ``basic_transformer_block.py`` to be populated first.
"""

try:
    from .dit import DiTTransformer2DModel, create_dit_model
except ImportError:
    pass  # BasicTransformerBlock not yet populated
