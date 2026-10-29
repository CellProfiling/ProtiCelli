"""Normalization and dtype contract, written before production changes.

Expected values follow declared acquisition bit depth or the explicitly chosen
intensity range. Auto mode uses array dtype and the whole four-channel stack,
including a signed normalized protein placeholder, before selecting references.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from tifffile import imwrite

from proticelli import Model
from proticelli._training import FinetuneDataset
from proticelli.data import ChannelAssembler, ImageNormalizer


def small_model():
    return Model(device="cpu", protein_map={"A": 1}, cellline_map={"C": 1})


class InputRangeTests(unittest.TestCase):
    def test_auto_uint8_one_count_is_scaled_by_dtype(self):
        image = np.broadcast_to(np.asarray([0, 1, 1], dtype=np.uint8), (2, 3, 3))
        expected = np.broadcast_to(np.asarray([-1, 2 / 255 - 1, 2 / 255 - 1])[:, None, None], (3, 2, 3))
        np.testing.assert_allclose(small_model()._preprocess_image(image).numpy(), expected, atol=1e-7)

    def test_auto_uint16_dim_values_keep_sixteen_bit_scale(self):
        image = np.broadcast_to(np.asarray([20, 128, 255], dtype=np.uint16), (2, 3, 3))
        expected = np.moveaxis(image.astype(np.float64) / 65535 * 2 - 1, -1, 0)
        np.testing.assert_allclose(small_model()._preprocess_image(image).numpy(), expected, atol=1e-7)

    def test_auto_preserves_actual_normalizer_pipeline_before_dropping_placeholder(self):
        stack = ChannelAssembler(has_protein=False).transform({
            "microtubules": np.full((2, 3), 70, np.uint8),
            "nucleus": np.full((2, 3), 100, np.uint8),
            "er": np.full((2, 3), 70, np.uint8),
        })
        normalized = ImageNormalizer(bit_depth=8).transform(stack)
        # Normalizer's signed placeholder establishes the entire stack's range.
        self.assertEqual(normalized[0, 0, 1], -1)
        expected = np.broadcast_to(np.asarray([0.4, 1, 0.4])[:, None, None], (3, 2, 3))
        np.testing.assert_allclose(small_model()._preprocess_image(normalized).numpy(), expected, atol=1e-7)

    def test_explicit_normalized_positive_floats_are_preserved(self):
        image = np.broadcast_to(np.asarray([0.2, 0.4, 0.8], np.float32), (2, 3, 3))
        actual = small_model()._preprocess_image(image, input_range="normalized")
        np.testing.assert_array_equal(actual.numpy(), np.moveaxis(image, -1, 0))

    def test_explicit_normalized_overshoot_is_preserved_without_clipping(self):
        # A shared training normalization anchor does not bound every channel
        # at 1. Already normalized intensities must reach the encoder as-is.
        image = np.broadcast_to(np.asarray([0.2, 1.2511415, 1.5], np.float32), (2, 3, 3))
        actual = small_model()._preprocess_image(image, input_range="normalized")
        np.testing.assert_array_equal(actual.numpy(), np.moveaxis(image, -1, 0))

    def test_auto_signed_normalized_overshoot_is_preserved(self):
        image = np.broadcast_to(np.asarray([-1.02, 1.0, 1.5], np.float32), (2, 3, 3))
        np.testing.assert_array_equal(small_model()._preprocess_image(image).numpy(),
                                      np.moveaxis(image, -1, 0))

    def test_normalized_float64_values_must_remain_finite_after_float32_cast(self):
        with self.assertRaises(ValueError):
            small_model()._preprocess_image(np.full((2, 3, 3), 1e40, np.float64),
                                            input_range="normalized")

    def test_explicit_unit_floats_are_scaled_once(self):
        image = np.broadcast_to(np.asarray([0.0, 0.25, 1.0], np.float32), (2, 3, 3))
        actual = small_model()._preprocess_image(image, input_range="unit")
        expected = np.broadcast_to(np.asarray([-1, -0.5, 1])[:, None, None], (3, 2, 3))
        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_auto_positive_unit_floats_keep_legacy_unit_interpretation(self):
        image = np.broadcast_to(np.asarray([0.2, 0.4, 0.8], np.float32), (2, 3, 3))
        expected = np.moveaxis(image * 2 - 1, -1, 0)
        np.testing.assert_array_equal(small_model()._preprocess_image(image).numpy(), expected)

    def test_explicit_raw_float_counts_require_and_use_bit_depth(self):
        for depth, maximum in ((8, 255), (16, 65535)):
            with self.subTest(depth=depth):
                image = np.broadcast_to(np.asarray([0, 128, 200], np.float32), (2, 3, 3))
                actual = small_model()._preprocess_image(image, input_range="raw", bit_depth=depth)
                expected = np.moveaxis(image.astype(np.float64) / maximum * 2 - 1, -1, 0)
                np.testing.assert_allclose(actual.numpy(), expected, atol=1e-7)

    def test_raw_float_without_acquisition_depth_is_rejected(self):
        with self.assertRaises(ValueError):
            small_model()._preprocess_image(np.full((2, 3, 3), 128.0), input_range="raw")

    def test_auto_nonunit_floats_preserve_pre_session_behavior(self):
        # Before this review, only floats wholly within [0,1] were rescaled.
        # Positive training-normalized values above1 passed through unchanged.
        image = np.broadcast_to(np.asarray([0.2, 1.25, 1.5], np.float32), (2, 3, 3))
        np.testing.assert_array_equal(small_model()._preprocess_image(image).numpy(),
                                      np.moveaxis(image, -1, 0))

    def test_auto_float_range_is_decided_after_float32_rounding_as_before(self):
        # float32 rounds the tiny upper overshoot to1 and the tiny negative to0.
        # The old API then treats each array as unit data and applies2*x-1.
        for values, expected in (
            ([0.25, 0.5, 1 + 1e-8], [-0.5, 0, 1]),
            ([-1e-50, 0.25, 0.5], [-1, -0.5, 0]),
        ):
            with self.subTest(values=values):
                image = np.asarray([[values]], dtype=np.float64)
                actual = small_model()._preprocess_image(image).numpy()[:, 0, 0]
                np.testing.assert_array_equal(actual, expected)

    def test_invalid_explicit_ranges_and_depths_are_rejected(self):
        for options in (
            {"input_range": "other"}, {"input_range": "raw", "bit_depth": 12},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    small_model()._preprocess_image(np.zeros((2, 3, 3)), **options)

    def test_finite_real_inputs_are_required_including_ignored_protein_channel(self):
        cases = [
            np.full((2, 3, 3), np.nan), np.full((2, 3, 3), np.inf),
            np.ones((2, 3, 3), dtype=np.complex64) * (1 + 1j),
        ]
        ignored_target = np.zeros((2, 3, 4), np.float32)
        ignored_target[..., 1] = np.nan
        cases.append(ignored_target)
        for image in cases:
            with self.subTest(dtype=image.dtype, channels=image.shape[-1]):
                with self.assertRaises(ValueError):
                    small_model()._preprocess_image(image)

    def test_explicit_range_limits_are_checked_before_scaling(self):
        for image, options in (
            (np.full((2, 3, 3), 1.1), {"input_range": "unit"}),
            (np.full((2, 3, 3), 1 + 1e-8), {"input_range": "unit"}),
            (np.full((2, 3, 3), -1e-50), {"input_range": "unit"}),
            (np.full((2, 3, 3), -0.1), {"input_range": "unit"}),
            (np.full((2, 3, 3), 256.0), {"input_range": "raw", "bit_depth": 8}),
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    small_model()._preprocess_image(image, **options)

    def test_positive_extents_correct_channel_axis_and_numeric_dtype_are_required(self):
        for image in (
            np.empty((0, 3, 3)), np.zeros((2, 3)), np.zeros((2, 3, 2)),
            np.zeros((1, 2, 3, 3)), np.full((2, 3, 3), "0.5"),
            np.full((2, 3, 3), 0.5, dtype=object),
        ):
            with self.subTest(shape=image.shape, dtype=image.dtype):
                with self.assertRaises(ValueError):
                    small_model()._preprocess_image(image)


class PublicInputValidationTests(unittest.TestCase):
    def test_preflight_accepts_prepared_reference_overshoot(self):
        image = np.broadcast_to(np.asarray([-1, 1.0639269, 1.2511415], np.float32), (512, 512, 3))
        report = small_model().validate_inputs([image], ["A"], input_range="normalized")
        self.assertTrue(report["valid"], report["errors"])

    def test_native_512_pixel_input_passes_preflight_without_loading_weights(self):
        model = small_model()
        report = model.validate_inputs([np.zeros((512, 512, 3), np.uint8)], ["A"], ["C"])
        self.assertTrue(report["valid"], report["errors"])
        self.assertIsNone(model._model)
        self.assertIsNone(model._vae)

    def test_non_native_geometry_is_a_preflight_error(self):
        report = small_model().validate_inputs([np.zeros((256, 512, 3), np.float32)], ["A"])
        self.assertFalse(report["valid"], "Public inference requires the native 512x512 geometry")

    def test_nonfinite_input_is_a_preflight_error(self):
        image = np.zeros((512, 512, 3), np.float32)
        image[0, 0, 0] = np.nan
        report = small_model().validate_inputs([image], ["A"])
        self.assertFalse(report["valid"])

    def test_preflight_accepts_explicit_positive_normalized_input(self):
        report = small_model().validate_inputs(
            [np.full((512, 512, 3), 0.4, np.float32)], ["A"],
            input_range="normalized",
        )
        self.assertTrue(report["valid"], report["errors"])

    def check_public_range(self, reliability, input_range, value, expected):
        from tests.test_core_regressions import prediction_boundaries

        with prediction_boundaries() as model:
            captured = []
            original = model.vae.encode

            def encode(condition):
                captured.append(condition.detach().clone())
                return original(condition)

            with patch.object(model.vae, "encode", side_effect=encode):
                method = model.predict_with_reliability if reliability else model.predict
                options = {"num_samples": 2} if reliability else {}
                result = method(
                    [np.full((512, 512, 3), value, np.float32)], ["A"], ["C"],
                    input_range=input_range, seed=23, show_progress=False, **options,
                )
        self.assertEqual(len(result.images), 1)
        np.testing.assert_allclose(captured[0].numpy(), expected, atol=1e-7)

    def test_predict_forwards_explicit_positive_normalized_range(self):
        self.check_public_range(False, "normalized", 0.4, 0.4)

    def test_reliability_forwards_explicit_unit_range(self):
        self.check_public_range(True, "unit", 0.25, -0.5)

    def test_non_native_inference_is_rejected_before_learned_components_load(self):
        for reliability in (False, True):
            with self.subTest(reliability=reliability):
                model = small_model()
                with patch.object(model, "_load_model", side_effect=AssertionError("unexpected weight load")), patch.object(
                    model, "_load_vae", side_effect=AssertionError("unexpected VAE load")
                ):
                    method = model.predict_with_reliability if reliability else model.predict
                    with self.assertRaises(ValueError):
                        method([np.zeros((256, 512, 3), np.float32)], ["A"], show_progress=False)


class SharedTrainingRangeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def load(self, image, **options):
        imwrite(self.root / "cell.tif", image, photometric="minisblack")
        return FinetuneDataset(str(self.root), ["cell.tif"], [1], [1], **options)[0]

    def test_dim_uint16_training_and_inference_use_the_same_dtype_scale(self):
        image = np.broadcast_to(np.asarray([20, 128, 70, 200], np.uint16), (2, 3, 4)).copy()
        expected = image.astype(np.float64) / 65535 * 2 - 1
        sample = self.load(image)
        np.testing.assert_allclose(sample["gt_image"].numpy(), expected[..., 1:2].transpose(2, 0, 1), atol=1e-7)
        inference = small_model()._preprocess_image(image).numpy()
        np.testing.assert_allclose(sample["cond_image"].numpy(), inference, atol=1e-7)

    def test_training_explicit_positive_normalized_data_is_preserved(self):
        image = np.broadcast_to(np.asarray([0.2, 0.3, 0.4, 0.8], np.float32), (2, 3, 4)).copy()
        sample = self.load(image, input_range="normalized")
        np.testing.assert_array_equal(sample["cond_image"].numpy(), image[..., [0, 2, 3]].transpose(2, 0, 1))
        np.testing.assert_array_equal(sample["gt_image"].numpy(), image[..., 1:2].transpose(2, 0, 1))

    def test_prepared_training_tiff_preserves_reference_and_target_overshoot(self):
        image = np.broadcast_to(np.asarray([-1, 1.3, 1.0639269, 1.2511415], np.float32), (2, 3, 4)).copy()
        sample = self.load(image, input_range="normalized")
        np.testing.assert_array_equal(sample["cond_image"].numpy(), image[..., [0, 2, 3]].transpose(2, 0, 1))
        np.testing.assert_array_equal(sample["gt_image"].numpy(), image[..., 1:2].transpose(2, 0, 1))

    def test_training_unit_floats_and_target_are_scaled_once(self):
        image = np.broadcast_to(np.asarray([0, 0.25, 0.5, 1], np.float32), (2, 3, 4)).copy()
        sample = self.load(image, input_range="unit")
        np.testing.assert_array_equal(sample["cond_image"].numpy()[:, 0, 0], [-1, 0, 1])
        np.testing.assert_array_equal(sample["gt_image"].numpy(), np.full((1, 2, 3), -0.5))

    def test_training_raw_float_counts_use_explicit_sixteen_bit_depth(self):
        image = np.full((2, 3, 4), 128.0, np.float32)
        sample = self.load(image, input_range="raw", bit_depth=16)
        np.testing.assert_allclose(sample["gt_image"].numpy(), 128 / 65535 * 2 - 1, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
