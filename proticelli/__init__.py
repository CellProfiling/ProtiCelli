"""ProtiCelli — Protein Visual Language Model for microscopy image generation.

Quick start::

    from proticelli import Model

    Model.download_checkpoints()  # downloads into proticelli/ package dir
    model = Model()               # loads from proticelli/checkpoint, proticelli/vae
    results = model.predict(images=[ref_img], protein_names=["ACTB"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import Model, PredictionResult

__version__ = "0.1.0"
__all__ = ["Model", "PredictionResult"]


def __getattr__(name: str) -> Any:
    """Load GPU model classes only when the public API requests them."""

    if name in {"Model", "PredictionResult"}:
        from .model import Model, PredictionResult

        return {"Model": Model, "PredictionResult": PredictionResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
