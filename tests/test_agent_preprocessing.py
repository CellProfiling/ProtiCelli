"""File-to-prediction integration at declared pixel size and channel order."""

from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
from tifffile import imwrite

from proticelli import PredictionResult
from proticelli.agent_tools import run_tool
from proticelli.data import center_crop_or_pad


class CapturingPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return PredictionResult(
            images=[np.zeros((512, 512), np.float32)],
            metadata=[{"protein_name": "A", "cell_line_name": "C"}],
        )


class AgentPreprocessingTests(unittest.TestCase):
    def test_real_web_engine_forwards_the_known_normalized_range(self):
        from proticelli_web.inference import RealInferenceEngine

        base = np.linspace(0, 1, 512 * 512, dtype=np.float32).reshape(512, 512)
        ensemble = np.stack([base, base * 0.9, np.flipud(base)])
        received = []

        class ModelBoundary:
            def predict_with_reliability(self, **kwargs):
                received.append(kwargs)
                return types.SimpleNamespace(ensembles=[ensemble], reliability_scores=[0.9])

        engine = RealInferenceEngine.__new__(RealInferenceEngine)
        engine._model = ModelBoundary()
        engine.device, engine.dtype = 'cpu', 'float32'
        engine.trajectory_batch_size = 1
        result = engine.generate(np.full((512, 512, 3), 0.4, np.float32),
                                 'A', 'C', 3, 10, 7, lambda *_: None,
                                 input_range='normalized')
        self.assertEqual(received[0]['input_range'], 'normalized')
        self.assertEqual(len(result.images), 3)

    def test_ema_only_inference_assets_are_ready_without_a_raw_checkpoint(self):
        from proticelli.utils.download import assets_ready

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('checkpoint/unet_ema', 'vae'):
                path = root / name
                path.mkdir(parents=True)
                (path / 'config.json').write_text('{}', encoding='utf-8')
                (path / 'diffusion_pytorch_model.safetensors').write_bytes(b'weight-boundary')
            self.assertTrue(assets_ready(root))

    def test_resampling_preserves_physical_scale_and_pads_normalized_background(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, counts in (("mt", 70), ("nucleus", 100), ("er", 70)):
                imwrite(root / f"{name}.tif", np.full((512, 512), counts, np.uint16))
            predictor = CapturingPredictor()
            result = run_tool(predictor, "predict_from_files", {
                "microtubules_path": str(root / "mt.tif"),
                "nucleus_path": str(root / "nucleus.tif"), "er_path": str(root / "er.tif"),
                "protein_names": ["A"], "cell_line_names": ["C"],
                "xy_resolution": 0.0707, "bit_depth": 16,
                "output_directory": str(root / "output"),
            })
            self.assertEqual(result["status"], "ok", result)
            self.assertEqual(len(predictor.calls), 1)
            supplied = predictor.calls[0]
            self.assertEqual(supplied["input_range"], "normalized")
            image = supplied["images"][0]
            self.assertEqual(image.shape, (512, 512, 4))
            # Physical resampling gives round(512*.0707/.1067)=339 pixels.
            # Centering173extra pixels gives86before and87after each axis.
            np.testing.assert_array_equal(image[:86], -1)
            np.testing.assert_array_equal(image[425:], -1)
            np.testing.assert_array_equal(image[:, :86], -1)
            np.testing.assert_array_equal(image[:, 425:], -1)
            expected = np.broadcast_to(np.array([0.4, -1, 1, 0.4]), (339, 339, 4))
            np.testing.assert_allclose(image[86:425, 86:425], expected, atol=1e-6)
            geometry = result["preprocessing"]["geometry"]
            self.assertEqual(geometry["source_shape"], [339, 339])
            self.assertEqual(geometry["padding_tblr"], [86, 87, 86, 87])
            self.assertTrue(all(Path(name).is_file() for name in result["saved_files"]))

    def test_center_crop_preserves_pixels_without_resizing(self):
        source = np.arange(5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
        cropped, geometry = center_crop_or_pad(source, size=3, return_metadata=True)
        np.testing.assert_array_equal(cropped, source[1:4, 2:5])
        self.assertEqual(geometry["crop_origin_yx"], [1, 2])
        self.assertEqual(geometry["padding_tblr"], [0, 0, 0, 0])

    def test_center_padding_places_unequal_margins_consistently(self):
        source = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        result, geometry = center_crop_or_pad(source, size=6, return_metadata=True)
        expected = np.full((6, 6, 4), -1, np.float32)
        expected[2:4, 1:4] = source
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(geometry["padding_tblr"], [2, 2, 1, 2])

    def test_invalid_crop_inputs_are_rejected(self):
        for source in (np.empty((0, 3, 4)), np.full((2, 3, 4), np.nan)):
            with self.subTest(shape=source.shape):
                with self.assertRaises(ValueError):
                    center_crop_or_pad(source)

    def test_web_auto_bit_depth_uses_uint16_dtype_before_optional_resampling(self):
        from proticelli_web.imaging import prepare_reference_channels

        channels = {"microtubules": np.full((512, 512), 70, np.uint16),
                    "nucleus": np.full((512, 512), 100, np.uint16),
                    "er": np.full((512, 512), 70, np.uint16)}
        _, metadata = prepare_reference_channels(channels, normalize=True,
                                                  pixel_size_um=0.1067, resample=False)
        self.assertEqual(metadata["normalization"]["bit_depth"], 16)


if __name__ == '__main__':
    unittest.main()
