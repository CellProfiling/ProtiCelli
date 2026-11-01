"""Run with ``python -I tests/smoke_installed_package.py`` after installation.

Isolated mode prevents the checkout from satisfying package imports. No learned
weights are downloaded: a tiny real DiT checks dependency and checkpoint APIs.
"""

from importlib import metadata, resources
import contextlib
import io
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import proticelli
import proticelli_web
from proticelli import Model
from proticelli.models.dit import DiTTransformer2DModel


class InstalledPackageTests(unittest.TestCase):
    def test_imports_use_installed_distribution_and_matching_version(self):
        checkout = Path(__file__).resolve().parents[1]
        self.assertNotEqual(Path(proticelli.__file__).resolve().parent, checkout / "proticelli")
        self.assertNotEqual(Path(proticelli_web.__file__).resolve().parent, checkout / "proticelli_web")
        distribution = metadata.distribution("proticelli")
        self.assertEqual(proticelli.__version__, distribution.version)
        self.assertEqual(distribution.metadata["License-Expression"], "GPL-3.0-only")
        self.assertIn("LICENSE", " ".join(distribution.metadata.get_all("License-File", [])))
        self.assertIn("Repository, https://github.com/CellProfiling/ProtiCelli",
                      distribution.metadata.get_all("Project-URL", []))

    def test_bundled_vocabularies_and_gallery_assets_are_usable(self):
        data = resources.files("proticelli.data")
        wrapper = Model(device="cpu", dtype="float32")
        for name, actual in (("antibody_map.pkl", wrapper.protein_map),
                             ("cell_line_map.pkl", wrapper.cellline_map)):
            with self.subTest(name=name):
                mapping = pickle.loads((data / name).read_bytes())
                self.assertTrue(mapping)
                self.assertEqual(mapping, actual)
                self.assertTrue(all(isinstance(k, str) and isinstance(v, int)
                                    for k, v in mapping.items()))
        static = resources.files("proticelli_web") / "static"
        for name in ("index.html", "app.js", "styles.css", "brand.svg", "proticelli-logo.png"):
            with self.subTest(name=name):
                self.assertTrue((static / name).read_bytes())

    def test_real_dit_forward_and_public_checkpoint_roundtrip(self):
        with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as directory:
            torch.random.default_generator.manual_seed(73)
            network = DiTTransformer2DModel(
                num_attention_heads=2, attention_head_dim=4, in_channels=4,
                out_channels=2, num_layers=1, sample_size=8, patch_size=2,
                num_embeds_ada_norm=16, num_protein_labels=5, num_cell_labels=3,
                norm_type="ada_norm_zero_continuous",
            ).eval()
            image = torch.linspace(-1, 1, 2 * 4 * 8 * 8).reshape(2, 4, 8, 8)
            conditions = {"timestep": torch.tensor([0.1, 0.5]),
                          "protein_labels": torch.tensor([1, 2]),
                          "cell_line_labels": torch.tensor([1, 2])}
            with torch.no_grad():
                before = network(image, **conditions).sample
            # Output geometry follows configured channels and input patch grid.
            self.assertEqual(before.shape, (2, 2, 8, 8))
            self.assertTrue(torch.isfinite(before).all().item())
            self.assertGreater(before.std().item(), 0)
            wrapper = Model(device="cpu", dtype="float32", protein_map={"CUSTOM": 2},
                            cellline_map={"CUSTOM_CELL": 1})
            wrapper._model = network
            with contextlib.redirect_stdout(io.StringIO()):
                wrapper.save(directory)
            restored = Model(checkpoint_dir=directory, device="cpu", dtype="float32")
            self.assertEqual(restored.protein_map, {"CUSTOM": 2})
            self.assertEqual(restored.cellline_map, {"CUSTOM_CELL": 1})
            for name, value in network.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[name], rtol=0, atol=0)
            with torch.no_grad():
                after = restored.model(image, **conditions).sample
            # Independent serialization invariant: preserving every weight and
            # configuration must preserve the result for the same real input.
            np.testing.assert_array_equal(after.numpy(), before.numpy())

    def test_installed_console_entry_point_supports_help_without_assets(self):
        distribution = metadata.distribution("proticelli")
        entry = next(ep for ep in distribution.entry_points if ep.name == "proticelli-web")
        output = io.StringIO()
        with patch.object(sys, "argv", ["proticelli-web", "--help"]), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as stopped:
                entry.load()()
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--mode", output.getvalue())


if __name__ == "__main__":
    unittest.main()
