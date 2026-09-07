"""Download and validate ProtiCelli's pre-trained inference assets."""

from __future__ import annotations

import argparse
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Union


DEFAULT_CHECKPOINT_URL = (
    "https://ell-vault.stanford.edu/dav/public/ProtiCelli/checkpoint.zip"
)
DEFAULT_VAE_URL = "https://ell-vault.stanford.edu/dav/public/ProtiCelli/vae.zip"
_MODEL_SUFFIXES = {".bin", ".safetensors"}
_CA_BUNDLE_ENV_VARS = (
    "PROTICELLI_CA_BUNDLE",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
)


def _diffusers_asset_ready(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and any(
            candidate.is_file() and candidate.suffix.lower() in _MODEL_SUFFIXES
            for candidate in path.rglob("*")
        )
    )


def checkpoint_ready(dest_dir: Union[str, Path]) -> bool:
    """Return whether the DiT checkpoint contains a loadable ``unet`` asset."""
    return _diffusers_asset_ready(Path(dest_dir) / "checkpoint" / "unet")


def vae_ready(dest_dir: Union[str, Path]) -> bool:
    """Return whether the VAE directory contains config and model weights."""
    return _diffusers_asset_ready(Path(dest_dir) / "vae")


def assets_ready(dest_dir: Union[str, Path, None] = None) -> bool:
    """Return whether both inference assets are complete and loadable."""
    destination = (
        Path(dest_dir)
        if dest_dir is not None
        else Path(__file__).resolve().parent.parent
    )
    return checkpoint_ready(destination) and vae_ready(destination)


def download_checkpoints(
    dest_dir: Union[str, Path, None] = None,
    checkpoint_url: str = DEFAULT_CHECKPOINT_URL,
    vae_url: str = DEFAULT_VAE_URL,
) -> dict[str, str]:
    """Download missing model assets and install them atomically.

    Existing complete assets are retained. Empty or incomplete asset directories
    are replaced only after a newly downloaded archive passes validation.
    """
    destination = (
        Path(dest_dir)
        if dest_dir is not None
        else Path(__file__).resolve().parent.parent
    )
    destination.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = destination / "checkpoint"
    vae_dir = destination / "vae"

    if checkpoint_ready(destination):
        print(f"Model checkpoint is ready at {checkpoint_dir}")
    else:
        print("Downloading the ProtiCelli model checkpoint (1 of 2)…")
        _download_and_install(
            checkpoint_url,
            destination,
            asset_name="checkpoint",
            validator=lambda path: _diffusers_asset_ready(path / "unet"),
        )

    if vae_ready(destination):
        print(f"VAE is ready at {vae_dir}")
    else:
        print("Downloading the ProtiCelli VAE (2 of 2)…")
        _download_and_install(
            vae_url,
            destination,
            asset_name="vae",
            validator=_diffusers_asset_ready,
        )

    if not assets_ready(destination):
        raise RuntimeError(
            "Downloaded model assets did not pass validation. Remove any manually "
            "copied checkpoint/vae folders and run the setup again."
        )

    paths = {
        "checkpoint_dir": str(checkpoint_dir),
        "vae_dir": str(vae_dir),
    }
    print("ProtiCelli model assets are ready.")
    return paths


def _download_ssl_context() -> ssl.SSLContext:
    """Build a verified TLS context without requiring macOS certificate setup.

    A user- or institution-provided CA bundle takes precedence. Otherwise use
    Certifi when available, because Python.org macOS installations do not
    always connect their OpenSSL trust store to the macOS Keychain.
    """

    for variable in _CA_BUNDLE_ENV_VARS:
        configured = os.getenv(variable, "").strip()
        if configured:
            bundle = Path(configured).expanduser()
            if not bundle.is_file():
                raise FileNotFoundError(
                    f"{variable} points to a missing CA bundle: {bundle}"
                )
            return ssl.create_default_context(cafile=str(bundle))

    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _download_with_urllib(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ProtiCelli/0.1"})
    context = _download_ssl_context() if url.lower().startswith("https://") else None
    with urllib.request.urlopen(
        request,
        timeout=60,
        context=context,
    ) as response, target.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        last_percent = -1
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = min(100, int(downloaded * 100 / total))
                if percent != last_percent:
                    print(
                        f"  {percent:3d}% · {downloaded / (1024 ** 2):.1f} "
                        f"of {total / (1024 ** 2):.1f} MiB",
                        end="\r",
                        flush=True,
                    )
                    last_percent = percent
        if total and downloaded != total:
            raise RuntimeError(
                f"Download ended early: received {downloaded} of {total} bytes"
            )
    if total:
        print()
    else:
        print(f"  Downloaded {target.stat().st_size / (1024 ** 2):.1f} MiB")


def _download_with_system_tool(url: str, target: Path) -> str:
    """Use a system TLS client as a secure, non-interactive fallback."""

    attempts: list[str] = []
    commands = (
        (
            "curl",
            [
                "--fail",
                "--location",
                "--show-error",
                "--progress-bar",
                "--output",
                str(target),
                "--",
                url,
            ],
        ),
        (
            "wget",
            [
                "--https-only",
                "--output-document",
                str(target),
                "--",
                url,
            ],
        ),
    )
    for name, arguments in commands:
        executable = shutil.which(name)
        if executable is None:
            attempts.append(f"{name} is not installed")
            continue
        target.unlink(missing_ok=True)
        print(f"  Retrying automatically with system {name}…")
        try:
            completed = subprocess.run(
                [executable, *arguments],
                check=False,
            )
        except OSError as exc:
            attempts.append(f"{name} could not start: {exc}")
            continue
        if completed.returncode == 0 and target.is_file() and target.stat().st_size:
            print(f"  Downloaded {target.stat().st_size / (1024 ** 2):.1f} MiB with {name}")
            return name
        target.unlink(missing_ok=True)
        attempts.append(f"{name} exited with status {completed.returncode}")

    raise RuntimeError("; ".join(attempts))


def _download_file(url: str, target: Path) -> None:
    """Download without prompts, retaining TLS verification on every route."""

    try:
        _download_with_urllib(url, target)
        return
    except (OSError, RuntimeError) as primary_error:
        target.unlink(missing_ok=True)
        print(
            "  Python's secure downloader could not complete the transfer; "
            "trying the operating-system certificate store."
        )
        try:
            _download_with_system_tool(url, target)
        except RuntimeError as fallback_error:
            raise RuntimeError(
                "Secure asset download failed with Python and all available "
                "system downloaders. TLS verification was not disabled. "
                "On a managed network, set PROTICELLI_CA_BUNDLE to the path of "
                "your institution's PEM certificate bundle. "
                f"Python error: {primary_error}. Fallbacks: {fallback_error}"
            ) from primary_error


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extractall(destination)


def _find_asset_directory(
    extracted_root: Path,
    asset_name: str,
    validator: Callable[[Path], bool],
) -> Path:
    candidates = [
        path
        for path in extracted_root.rglob(asset_name)
        if path.is_dir() and validator(path)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Archive must contain exactly one valid {asset_name}/ directory; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _install_directory(
    source: Path,
    destination: Path,
    validator: Callable[[Path], bool],
) -> None:
    backup = destination.parent / f".{destination.name}.previous-{uuid.uuid4().hex}"
    had_previous = destination.exists()
    if had_previous:
        destination.replace(backup)
    try:
        source.replace(destination)
        if not validator(destination):
            raise ValueError(f"Installed {destination.name} asset failed validation")
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        if had_previous and backup.exists():
            backup.replace(destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def _download_and_install(
    url: str,
    dest_dir: Path,
    *,
    asset_name: str,
    validator: Callable[[Path], bool],
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=f".proticelli-{asset_name}-",
        dir=dest_dir,
    ) as temporary:
        temporary_dir = Path(temporary)
        archive_path = temporary_dir / f"{asset_name}.zip"
        extracted_dir = temporary_dir / "extracted"
        extracted_dir.mkdir()

        _download_file(url, archive_path)
        if not zipfile.is_zipfile(archive_path):
            raise ValueError(f"{asset_name} download is not a valid ZIP archive")
        print(f"  Validating and installing {asset_name}…")
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extracted_dir)
        source = _find_asset_directory(extracted_dir, asset_name, validator)
        _install_directory(source, dest_dir / asset_name, validator)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download or verify ProtiCelli inference checkpoints"
    )
    parser.add_argument("--dest-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-url",
        default=os.getenv("PROTICELLI_CHECKPOINT_URL", DEFAULT_CHECKPOINT_URL),
    )
    parser.add_argument(
        "--vae-url",
        default=os.getenv("PROTICELLI_VAE_URL", DEFAULT_VAE_URL),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit successfully only when both model assets are ready",
    )
    args = parser.parse_args(argv)
    if args.check:
        return 0 if assets_ready(args.dest_dir) else 1
    download_checkpoints(args.dest_dir, args.checkpoint_url, args.vae_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
