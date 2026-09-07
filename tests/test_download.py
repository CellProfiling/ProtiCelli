from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from proticelli.utils.download import (
    _download_file,
    assets_ready,
    download_checkpoints,
    main,
)


def make_asset_archive(path: Path, asset_name: str) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if asset_name == "checkpoint":
            root = "release/checkpoint/unet"
        else:
            root = "release/vae"
        archive.writestr(f"{root}/config.json", "{}")
        archive.writestr(f"{root}/diffusion_pytorch_model.safetensors", b"weights")


class DownloadTests(unittest.TestCase):
    def test_secure_download_falls_back_to_system_curl_without_prompting(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            target = Path(directory) / "asset.zip"

            def fake_run(command, **_kwargs):
                self.assertIn("--fail", command)
                self.assertNotIn("--insecure", command)
                target.write_bytes(b"downloaded by curl")
                return SimpleNamespace(returncode=0)

            with patch(
                "proticelli.utils.download._download_with_urllib",
                side_effect=OSError("certificate verify failed"),
            ), patch(
                "proticelli.utils.download.shutil.which",
                side_effect=lambda name: "/usr/bin/curl" if name == "curl" else None,
            ), patch(
                "proticelli.utils.download.subprocess.run",
                side_effect=fake_run,
            ):
                _download_file("https://example.invalid/asset.zip", target)

            self.assertEqual(target.read_bytes(), b"downloaded by curl")

    def test_wget_is_the_automatic_backup_when_curl_fails(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            target = Path(directory) / "asset.zip"
            attempted: list[str] = []

            def fake_run(command, **_kwargs):
                attempted.append(command[0])
                if command[0].endswith("curl"):
                    return SimpleNamespace(returncode=60)
                target.write_bytes(b"downloaded by wget")
                return SimpleNamespace(returncode=0)

            with patch(
                "proticelli.utils.download._download_with_urllib",
                side_effect=OSError("certificate verify failed"),
            ), patch(
                "proticelli.utils.download.shutil.which",
                side_effect=lambda name: f"/usr/bin/{name}",
            ), patch(
                "proticelli.utils.download.subprocess.run",
                side_effect=fake_run,
            ):
                _download_file("https://example.invalid/asset.zip", target)

            self.assertEqual(attempted, ["/usr/bin/curl", "/usr/bin/wget"])
            self.assertEqual(target.read_bytes(), b"downloaded by wget")

    def test_download_failure_keeps_tls_verification_enabled(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            target = Path(directory) / "asset.zip"
            with patch(
                "proticelli.utils.download._download_with_urllib",
                side_effect=OSError("certificate verify failed"),
            ), patch(
                "proticelli.utils.download.shutil.which",
                return_value=None,
            ):
                with self.assertRaisesRegex(RuntimeError, "TLS verification was not disabled"):
                    _download_file("https://example.invalid/asset.zip", target)

    def test_downloads_missing_assets_repairs_partial_directory_and_then_skips(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            destination = root / "package"
            destination.mkdir()
            partial = destination / "checkpoint"
            partial.mkdir()
            (partial / "incomplete.txt").write_text("partial", encoding="utf-8")

            checkpoint_archive = root / "checkpoint.zip"
            vae_archive = root / "vae.zip"
            make_asset_archive(checkpoint_archive, "checkpoint")
            make_asset_archive(vae_archive, "vae")

            paths = download_checkpoints(
                destination,
                checkpoint_archive.resolve().as_uri(),
                vae_archive.resolve().as_uri(),
            )
            self.assertTrue(assets_ready(destination))
            self.assertFalse((partial / "incomplete.txt").exists())
            self.assertEqual(Path(paths["checkpoint_dir"]), destination / "checkpoint")

            missing_url = (root / "does-not-exist.zip").resolve().as_uri()
            download_checkpoints(destination, missing_url, missing_url)
            self.assertEqual(main(["--check", "--dest-dir", str(destination)]), 0)

    def test_check_fails_when_model_weights_are_incomplete(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            destination = Path(directory)
            (destination / "checkpoint" / "unet").mkdir(parents=True)
            (destination / "checkpoint" / "unet" / "config.json").write_text(
                "{}", encoding="utf-8"
            )
            (destination / "vae").mkdir()
            (destination / "vae" / "config.json").write_text("{}", encoding="utf-8")
            self.assertFalse(assets_ready(destination))
            self.assertEqual(main(["--check", "--dest-dir", str(destination)]), 1)


if __name__ == "__main__":
    unittest.main()
