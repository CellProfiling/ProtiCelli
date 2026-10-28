"""Checkpoint, EMA, and random-state helpers for resumable fine-tuning."""

from contextlib import contextmanager
import json
from pathlib import Path
import pickle
import random

import numpy as np
import torch


def resolve_resume_path(resume_from):
    if resume_from is None:
        return None
    path = Path(resume_from)
    if path.is_dir():
        path = path / "training_state.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Training state not found: {path}")
    return path


def capture_rng(device):
    state = {"torch_cpu": torch.random.get_rng_state(),
             "numpy": np.random.get_state(), "python": random.getstate()}
    if device.type == "cuda":
        state["device"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["device"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device):
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "device" in state:
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["device"].cpu(), device)
        elif device.type == "mps" and hasattr(torch.mps, "set_rng_state"):
            torch.mps.set_rng_state(state["device"].cpu())


@torch.no_grad()
def update_ema(shadow, model, decay):
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            shadow[name].lerp_(value.detach(), 1 - decay)
        else:
            shadow[name].copy_(value)


@contextmanager
def ema_weights(model, shadow):
    """Temporarily swap existing tensor storage, avoiding a third model copy."""
    original, seen = [], set()
    try:
        for name, tensor in model.state_dict(keep_vars=True).items():
            if id(tensor) not in seen:
                seen.add(id(tensor))
                original.append((tensor, tensor.data))
                tensor.data = shadow[name]
        yield
    finally:
        for tensor, data in original:
            tensor.data = data


def save_training_checkpoint(path, model, dit, optimizer, scheduler, scaler, ema,
                             epoch, next_batch, global_step, data_seed, configuration):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    dit.save_pretrained(path / "unet")
    if ema is not None:
        with ema_weights(dit, ema):
            dit.save_pretrained(path / "unet_ema")
    for filename, mapping in (("antibody_map.pkl", model.protein_map),
                              ("cell_line_map.pkl", model.cellline_map)):
        with (path / filename).open("wb") as handle:
            pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (path / "proticelli_checkpoint.json").open("w", encoding="utf-8") as handle:
        json.dump({"version": 1, "inference_subfolder": "unet_ema" if ema is not None else "unet",
                   "channel_order": ["microtubules", "protein", "nucleus", "ER"]}, handle)
    state = {"version": 1, "model": dit.state_dict(), "optimizer": optimizer.state_dict(),
             "lr_scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
             "ema": ema, "epoch": epoch, "next_batch": next_batch,
             "global_step": global_step, "data_seed": data_seed,
             "rng": capture_rng(model.device), "configuration": configuration}
    temporary = path / ".training_state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(path / "training_state.pt")
