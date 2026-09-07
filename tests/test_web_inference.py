from __future__ import annotations

import unittest
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from proticelli_web.inference import (
    DemoInferenceEngine,
    RealInferenceEngine,
    compute_medoid_and_reliability,
    runtime_diagnostics,
)


class WebInferenceTests(unittest.TestCase):
    def test_medoid_prefers_the_consensus_member(self):
        base = np.arange(64, dtype=np.float32).reshape(8, 8)
        ensemble = np.stack([base, base * 1.1 + 2, np.flipud(base)])
        medoid, reliability = compute_medoid_and_reliability(ensemble)
        self.assertIn(medoid, (0, 1))
        self.assertGreaterEqual(reliability, 0.0)
        self.assertLessEqual(reliability, 1.0)

    def test_demo_engine_is_reproducible_and_labeled(self):
        y, x = np.mgrid[:512, :512]
        radius = np.sqrt((x - 256) ** 2 + (y - 256) ** 2)
        reference = np.stack(
            [np.clip(1 - radius / 240, 0, 1), np.exp(-(radius / 75) ** 2), np.exp(-(radius / 165) ** 2)],
            axis=-1,
        ).astype(np.float32)
        engine = DemoInferenceEngine()
        first = engine.generate(reference, "TOMM20", "A-431", 3, 10, 42, lambda *_: None)
        second = engine.generate(reference, "TOMM20", "A-431", 3, 10, 42, lambda *_: None)
        self.assertEqual(first.engine, "demo")
        self.assertEqual(len(first.images), 3)
        self.assertTrue(np.array_equal(first.images[0], second.images[0]))
        self.assertEqual(first.medoid_index, second.medoid_index)
        self.assertTrue(np.isfinite(first.reliability))

    def test_demo_engine_honors_cooperative_cancellation(self):
        reference = np.zeros((512, 512, 3), dtype=np.float32)
        with self.assertRaises(InterruptedError):
            DemoInferenceEngine().generate(
                reference, "TOMM20", "A-431", 3, 10, 42, lambda *_: None,
                cancel_check=lambda: True,
            )

    def test_web_runtime_prefers_cuda_float32_and_trajectory_batching(self):
        fake_cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _index: "Test GPU")
        fake_torch = SimpleNamespace(cuda=fake_cuda)
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.dict(os.environ, {}, clear=False):
            for key in ("PROTICELLI_WEB_DEVICE", "PROTICELLI_WEB_DTYPE", "PROTICELLI_WEB_GPU_BATCH"):
                os.environ.pop(key, None)
            engine = RealInferenceEngine(Path("."))
        self.assertEqual(engine.runtime_info["device"], "cuda")
        self.assertEqual(engine.runtime_info["dtype"], "float32")
        self.assertEqual(engine.runtime_info["trajectory_batch_size"], 4)

    def test_web_runtime_allows_explicit_float16_override(self):
        fake_cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _index: "Test GPU")
        fake_torch = SimpleNamespace(cuda=fake_cuda)
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.dict(
            os.environ,
            {"PROTICELLI_WEB_DTYPE": "float16"},
            clear=False,
        ):
            engine = RealInferenceEngine(Path("."))
        self.assertEqual(engine.runtime_info["dtype"], "float16")

    def test_web_runtime_uses_mps_on_apple_silicon_when_cuda_is_unavailable(self):
        fake_mps = SimpleNamespace(
            is_built=lambda: True,
            is_available=lambda: True,
            get_name=lambda: "Apple M-series GPU",
        )
        fake_torch = SimpleNamespace(
            __version__="2.x",
            version=SimpleNamespace(cuda=None, hip=None),
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=fake_mps),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROTICELLI_WEB_DEVICE", None)
            os.environ.pop("PROTICELLI_WEB_DTYPE", None)
            os.environ.pop("PROTICELLI_WEB_GPU_BATCH", None)
            engine = RealInferenceEngine(Path("."))
        self.assertEqual(engine.runtime_info["device"], "mps")
        self.assertEqual(engine.runtime_info["accelerator"], "mps")
        self.assertEqual(engine.runtime_info["device_name"], "Apple M-series GPU")
        self.assertEqual(engine.runtime_info["dtype"], "float32")
        self.assertEqual(engine.runtime_info["trajectory_batch_size"], 1)
        self.assertEqual(engine.runtime_info["status"], "mps_ready")

    def test_runtime_diagnostics_distinguish_cpu_only_torch(self):
        fake_torch = SimpleNamespace(
            __version__="2.x+cpu",
            version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}), patch(
            "proticelli_web.inference.shutil.which", return_value=None
        ):
            details = runtime_diagnostics("cuda")
        self.assertEqual(details["status"], "pytorch_cpu_build")
        self.assertFalse(details["cuda_available"])

    def test_precision_switch_releases_the_resident_model(self):
        emptied = []
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "Test GPU",
            device_count=lambda: 1,
            empty_cache=lambda: emptied.append(True),
        )
        fake_torch = SimpleNamespace(
            __version__="2.x+cu121",
            version=SimpleNamespace(cuda="12.1"),
            cuda=fake_cuda,
        )
        messages = []
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROTICELLI_WEB_DTYPE", None)
            engine = RealInferenceEngine(Path("."))
            engine._model = object()
            engine._select_precision("float16", lambda _value, message: messages.append(message))
        self.assertIsNone(engine._model)
        self.assertEqual(engine.dtype, "float16")
        self.assertTrue(emptied)
        self.assertIn("Switching model precision to float16", messages)

    def test_real_engine_exposes_the_scored_ensemble(self):
        base = np.arange(64, dtype=np.float32).reshape(8, 8)
        ensemble = np.stack([base, base + 0.1, base + 0.2])

        class FakeModel:
            def predict_with_reliability(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(ensembles=[ensemble], reliability_scores=[0.91])

        engine = RealInferenceEngine(Path("."))
        engine._model = FakeModel()
        reference = np.zeros((8, 8, 3), dtype=np.float32)
        result = engine.generate(reference, "TOMM20", "A-431", 3, 20, 42, lambda *_: None)
        self.assertTrue(engine._model.kwargs["return_ensembles"])
        self.assertEqual(result.engine, "proticelli")
        self.assertEqual(len(result.images), 3)
        self.assertAlmostEqual(result.reliability, 0.91)


if __name__ == "__main__":
    unittest.main()
