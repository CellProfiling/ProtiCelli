"""ProtVS — Protein Visual Language Model for microscopy image generation.

Quick start::

    from protvs import ProtVS

    ProtVS.download_checkpoints()  # downloads into protvs/ package dir
    model = ProtVS()               # loads from protvs/checkpoint, protvs/vae
    results = model.predict(images=[ref_img], protein_names=["ACTB"])
"""

from .model import ProtVS, PredictionResult

__version__ = "0.1.0"
__all__ = ["ProtVS", "PredictionResult"]
