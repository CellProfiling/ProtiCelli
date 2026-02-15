"""ProtVL — Protein Visual Language Model for microscopy image generation.

Quick start::

    from protvl import ProtVL

    ProtVL.download_checkpoints()  # downloads into protvl/ package dir
    model = ProtVL()               # loads from protvl/checkpoint, protvl/vae
    results = model.predict(images=[ref_img], protein_names=["ACTB"])
"""

from .model import ProtVL, PredictionResult

__version__ = "0.1.0"
__all__ = ["ProtVL", "PredictionResult"]
