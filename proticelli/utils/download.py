"""Utilities for downloading ProtiCelli pre-trained checkpoints.

After downloading, the ``proticelli/`` package directory contains::

    proticelli/
    ├── checkpoint/       # model weights (from checkpoint.zip)
    │   └── unet_ema/
    ├── vae/              # VAE weights (from vae.zip)
    ├── data/
    │   ├── antibody_map.pkl
    │   ├── cell_line_map.pkl
    │   └── dataset.py
    └── ...
"""

import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Union


def download_checkpoints(
    dest_dir: Union[str, Path, None] = None,
    checkpoint_url: str = "https://ell-vault.stanford.edu/dav/public/ProtiCelli/checkpoint.zip",
    vae_url: str = "https://ell-vault.stanford.edu/dav/public/ProtiCelli/vae.zip",
) -> dict:
    """Download and extract ProtiCelli checkpoints into the package directory.

    Parameters
    ----------
    dest_dir : str or Path, optional
        Directory where ``checkpoint/`` and ``vae/`` will be created.
        Default: the ``proticelli/`` package directory.
    checkpoint_url : str
        URL to the model checkpoint zip.
    vae_url : str
        URL to the VAE zip.

    Returns
    -------
    dict
        ``{"checkpoint_dir": str, "vae_dir": str}``.
    """
    if dest_dir is None:
        dest_dir = Path(__file__).resolve().parent.parent  # proticelli/
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = dest_dir / "checkpoint"
    vae_dir = dest_dir / "vae"

    if not checkpoint_dir.exists():
        print(f"Downloading model checkpoint from {checkpoint_url} ...")
        _download_and_extract(checkpoint_url, dest_dir, "checkpoint.zip")
    else:
        print(f"Checkpoint already exists at {checkpoint_dir}")

    if not vae_dir.exists():
        print(f"Downloading VAE from {vae_url} ...")
        _download_and_extract(vae_url, dest_dir, "vae.zip")
    else:
        print(f"VAE already exists at {vae_dir}")

    # Verify
    for d, label in [(checkpoint_dir, "checkpoint"), (vae_dir, "vae")]:
        if not d.exists():
            contents = [p.name for p in dest_dir.iterdir()]
            print(
                f"Warning: Expected {d} but not found. "
                f"Contents of {dest_dir}: {contents}"
            )

    paths = {
        "checkpoint_dir": str(checkpoint_dir),
        "vae_dir": str(vae_dir),
    }
    print(f"Ready: {paths}")
    return paths


def _download_and_extract(url: str, dest_dir: Path, zip_name: str):
    """Download a zip file and extract it."""
    zip_path = dest_dir / zip_name

    # Try wget → curl → urllib
    downloaded = False
    for cmd in [
        ["wget", "-q", "--show-progress", "-O", str(zip_path), url],
        ["curl", "-L", "-o", str(zip_path), url],
    ]:
        try:
            subprocess.run(cmd, check=True)
            downloaded = True
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    if not downloaded:
        import urllib.request
        print("  Downloading with urllib (no progress bar)...")
        urllib.request.urlretrieve(url, str(zip_path))

    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file from {url} is not a valid zip (got HTML error page?)")

    print(f"  Extracting {zip_name} ...")
    target_names = {"checkpoint", "vae"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)
        # Walk all extracted paths and move any dir whose name matches target_names
        for entry in tmp_path.rglob("*"):
            if entry.is_dir() and entry.name in target_names:
                dest = dest_dir / entry.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(entry), str(dest))

    zip_path.unlink()
    print("  Done.")
