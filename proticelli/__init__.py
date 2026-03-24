"""ProtiCelli — Protein Visual Language Model for microscopy image generation.

Quick start::

    from proticelli import Model

    Model.download_checkpoints()  # downloads into proticelli/ package dir
    model = Model()               # loads from proticelli/checkpoint, proticelli/vae
    results = model.predict(images=[ref_img], protein_names=["ACTB"])
"""

from .model import Model, PredictionResult

__version__ = "0.1.0"
__all__ = ["Model", "PredictionResult"]
