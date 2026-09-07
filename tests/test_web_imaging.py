from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from proticelli_web.imaging import (
    ImageValidationError,
    channel_statistics,
    inspect_image,
    inspect_tiff,
    normalize_channel,
    prepare_reference_channels,
    read_image_plane,
    read_reference_tiff,
    read_tiff_plane,
    scalar_statistics,
    suggest_channel_role,
)


class WebImagingTests(unittest.TestCase):
    def test_rgb_png_is_exposed_as_named_component_planes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "composite.png"
            array = np.zeros((12, 16, 3), dtype=np.uint8)
            array[..., 0] = 11
            array[..., 1] = 22
            array[..., 2] = 33
            Image.fromarray(array, mode="RGB").save(path)

            inspected = inspect_image(path)
            self.assertEqual(inspected["format"], "PNG")
            self.assertFalse(inspected["lossy"])
            self.assertEqual([plane["label"] for plane in inspected["planes"]], ["Red", "Green", "Blue"])
            self.assertEqual(
                [plane["suggested_role"] for plane in inspected["planes"]],
                ["microtubules", "er", "nucleus"],
            )
            np.testing.assert_array_equal(read_image_plane(path, 1), array[..., 1])

    def test_grayscale_jpeg_is_supported_and_marked_lossy(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "nucleus.jpg"
            Image.fromarray(np.full((10, 14), 80, dtype=np.uint8), mode="L").save(path)

            inspected = inspect_image(path)
            self.assertEqual(inspected["format"], "JPEG")
            self.assertTrue(inspected["lossy"])
            self.assertEqual(len(inspected["planes"]), 1)
            self.assertEqual(inspected["planes"][0]["suggested_role"], "nucleus")
            self.assertEqual(read_image_plane(path, 0).shape, (10, 14))

    def test_channel_normalization_preserves_order(self):
        image = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        display = normalize_channel(image)
        self.assertEqual(display.dtype, np.uint8)
        self.assertEqual(display[0, 0], 0)
        self.assertEqual(display[-1, -1], 255)
        self.assertTrue(np.all(np.diff(display.ravel().astype(int)) > 0))

    def test_statistics_are_json_safe(self):
        reference = np.stack(
            [np.arange(16).reshape(4, 4), np.ones((4, 4)) * 4, np.eye(4)], axis=-1
        )
        stats = channel_statistics(reference, bins=8)
        self.assertEqual([item.key for item in stats], ["microtubules", "nucleus", "er"])
        self.assertEqual(len(stats[0].histogram), 8)
        scalar = scalar_statistics(reference[..., 0], bins=4)
        self.assertEqual(sum(scalar["histogram"]), 16)
        self.assertIsInstance(scalar["minimum"], float)

    def test_four_channel_input_drops_observed_protein(self):
        array = np.zeros((512, 512, 4), dtype=np.uint16)
        for channel in range(4):
            array[..., channel] = channel
        fake = types.SimpleNamespace(imread=lambda *_args, **_kwargs: array)
        with patch.dict(sys.modules, {"tifffile": fake}):
            result = read_reference_tiff("example.tiff")
        self.assertEqual(result.shape, (512, 512, 3))
        self.assertEqual([int(result[0, 0, index]) for index in range(3)], [0, 2, 3])

    def test_non_native_shape_is_rejected(self):
        array = np.zeros((256, 512, 3), dtype=np.uint16)
        fake = types.SimpleNamespace(imread=lambda *_args, **_kwargs: array)
        with patch.dict(sys.modules, {"tifffile": fake}):
            with self.assertRaisesRegex(ImageValidationError, "requires 512 × 512"):
                read_reference_tiff("example.tiff")

    def test_channel_role_suggestions_cover_common_landmarks(self):
        self.assertEqual(suggest_channel_role("cell_07_DAPI.ome.tiff"), "nucleus")
        self.assertEqual(suggest_channel_role("alpha-tubulin"), "microtubules")
        self.assertEqual(suggest_channel_role("Calreticulin channel"), "er")
        self.assertIsNone(suggest_channel_role("unknown stain"))

    def test_mapped_planes_are_center_cropped_and_documented(self):
        y, x = np.mgrid[:620, :700]
        channels = {
            "microtubules": x.astype(np.float32),
            "nucleus": y.astype(np.float32),
            "er": (x + y).astype(np.float32),
        }
        prepared, manifest = prepare_reference_channels(
            channels,
            pixel_size_um=0.1067,
            resample=False,
        )
        self.assertEqual(prepared.shape, (512, 512, 3))
        self.assertEqual(
            manifest["crop"],
            {
                "x": 94,
                "y": 54,
                "width": 512,
                "height": 512,
                "padding": {"left": 0, "top": 0, "right": 0, "bottom": 0},
            },
        )
        self.assertFalse(manifest["resampled"])
        self.assertEqual(float(prepared[0, 0, 0]), 94.0)

    def test_small_mapped_planes_are_centered_and_zero_padded(self):
        base = np.ones((200, 300), dtype=np.float32)
        channels = {key: base * (index + 1) for index, key in enumerate(("microtubules", "nucleus", "er"))}
        prepared, manifest = prepare_reference_channels(
            channels,
            pixel_size_um=0.1067,
            resample=False,
        )
        self.assertEqual(prepared.shape, (512, 512, 3))
        self.assertEqual(manifest["crop"]["x"], -106)
        self.assertEqual(manifest["crop"]["y"], -156)
        self.assertEqual(
            manifest["crop"]["padding"],
            {"left": 106, "top": 156, "right": 106, "bottom": 156},
        )
        self.assertTrue(np.all(prepared[156:356, 106:406, 0] == 1))
        self.assertEqual(float(prepared[0, 0, 0]), 0.0)

    def test_optional_proticelli_normalization_runs_after_crop_and_records_gains(self):
        base = np.linspace(0, 240, 300 * 400, dtype=np.float32).reshape(300, 400)
        channels = {
            "microtubules": base * 0.45,
            "nucleus": base,
            "er": base * 0.2,
        }
        prepared, manifest = prepare_reference_channels(
            channels,
            pixel_size_um=0.1067,
            resample=False,
            normalize=True,
            normalization_bit_depth=8,
        )

        self.assertEqual(prepared.shape, (512, 512, 3))
        self.assertEqual(prepared.dtype, np.float32)
        self.assertGreaterEqual(float(prepared.min()), -1.0)
        self.assertLessEqual(float(prepared.max()), 1.0)
        self.assertTrue(np.all(prepared[0, 0] == -1.0))
        normalization = manifest["normalization"]
        self.assertTrue(normalization["applied"])
        self.assertEqual(normalization["stage"], "after_crop")
        self.assertEqual(normalization["bit_depth"], 8)
        self.assertEqual(normalization["parameters"]["ref_channel"], 2)
        self.assertEqual(
            set(normalization["gains"]),
            {"microtubules", "protein", "nucleus", "er"},
        )

    def test_normalization_rejects_already_signed_input(self):
        channels = {
            key: np.full((512, 512), -0.5, dtype=np.float32)
            for key in ("microtubules", "nucleus", "er")
        }
        with self.assertRaisesRegex(ImageValidationError, "non-negative raw intensities"):
            prepare_reference_channels(
                channels,
                pixel_size_um=0.1067,
                resample=False,
                normalize=True,
            )

    def test_offset_window_on_native_image_records_padding(self):
        base = np.ones((512, 512), dtype=np.float32)
        channels = {key: base for key in ("microtubules", "nucleus", "er")}
        prepared, manifest = prepare_reference_channels(
            channels,
            pixel_size_um=0.1067,
            resample=False,
            crop_x=-128,
            crop_y=64,
        )
        self.assertEqual(
            manifest["crop"]["padding"],
            {"left": 128, "top": 0, "right": 0, "bottom": 64},
        )
        self.assertTrue(np.all(prepared[:, :128] == 0))

    def test_ome_axes_names_and_pixel_size_are_inspected(self):
        array = np.zeros((3, 32, 48), dtype=np.uint16)
        array[1] = 7
        xml = """<OME><Image><Pixels PhysicalSizeX="0.1067">
          <Channel Name="alpha tubulin"/><Channel Name="DAPI"/><Channel Name="calreticulin"/>
        </Pixels></Image></OME>"""

        class Series:
            axes = "CYX"
            def asarray(self):
                return array

        class FakeTiffFile:
            def __init__(self, _path):
                self.series = [Series()]
                self.ome_metadata = xml
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False

        fake = types.SimpleNamespace(TiffFile=FakeTiffFile)
        with patch.dict(sys.modules, {"tifffile": fake}):
            inspected = inspect_tiff("study.ome.tiff")
            plane = read_tiff_plane("study.ome.tiff", 1)
        self.assertEqual(inspected["axes"], "CYX")
        self.assertEqual(inspected["pixel_size_um"], 0.1067)
        self.assertEqual([item["suggested_role"] for item in inspected["planes"]], ["microtubules", "nucleus", "er"])
        self.assertTrue(np.all(plane == 7))


if __name__ == "__main__":
    unittest.main()
