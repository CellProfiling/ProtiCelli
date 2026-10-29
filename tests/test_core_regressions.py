"""Tests-first baseline for the public ProtiCelli inference API.

These tests intentionally expose existing defects. They do not mark failures as
expected, replace core behavior with fixes, load learned weights, or use a GPU.
Only the heavyweight checkpoint, VAE, denoiser, and prediction-sampler boundaries
are replaced. Numerical sampling tests execute the real EDM sampling functions.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

from proticelli import Model
from proticelli._sampling import sample_edm, sample_edm_uncertainty
from proticelli.data import ChannelAssembler, ImageNormalizer


class TinyCheckpoint(torch.nn.Module):
    """A serializable learned-weight boundary with actual Torch state/dtypes."""

    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))
        self.config = {}

    def register_to_config(self, **options):
        self.config.update(options)

    @classmethod
    def from_pretrained(cls, directory, *, subfolder):
        values = np.load(Path(directory) / subfolder / "boundary_weights.npy")
        return cls(float(values[0]))

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "boundary_weights.npy", self.weight.detach().cpu().numpy())
        (directory / "config.json").write_text("{}", encoding="utf-8")


@contextlib.contextmanager
def checkpoint_boundary():
    """Keep unsupported diffusers/transformers imports outside these unit tests."""
    package = types.ModuleType("proticelli.models")
    package.__path__ = []
    module = types.ModuleType("proticelli.models.dit")
    module.DiTTransformer2DModel = TinyCheckpoint
    # An accidental random initialization must remain observable as weight zero.
    module.create_dit_model = lambda **_kwargs: TinyCheckpoint(0)
    replacements = {"proticelli.models": package, "proticelli.models.dit": module}
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        # Restore only the checkpoint boundary modules. Restoring the entire
        # module registry would discard lazy Torch registrations loaded meanwhile.
        for name, original in previous.items():
            if original is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class CheckpointRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def model(self, checkpoint=None, **kwargs):
        return Model(
            checkpoint_dir=self.directory if checkpoint is None else checkpoint,
            device="cpu", **kwargs,
        )

    def test_raw_unet_checkpoint_loads_known_weights(self):
        TinyCheckpoint(3.25).save_pretrained(self.directory / "unet")
        with checkpoint_boundary():
            loaded = self.model().model
        self.assertEqual(loaded.weight.item(), 3.25)
        self.assertFalse(loaded.training)
        self.assertFalse(loaded.weight.requires_grad)

    def test_explicit_ema_load_selects_ema_over_raw_weights(self):
        TinyCheckpoint(3.25).save_pretrained(self.directory / "unet")
        TinyCheckpoint(7.5).save_pretrained(self.directory / "unet_ema")
        instance = self.model()
        with checkpoint_boundary():
            instance._load_model(use_ema=True)
        self.assertEqual(instance.model.weight.item(), 7.5)

    def test_ema_only_checkpoint_is_loaded_for_public_inference(self):
        TinyCheckpoint(7.5).save_pretrained(self.directory / "unet_ema")
        with checkpoint_boundary():
            loaded = self.model().model
        self.assertEqual(
            loaded.weight.item(), 7.5,
            "An EMA-only trained checkpoint must not become random initialization",
        )

    def test_missing_checkpoint_cannot_silently_create_an_untrained_predictor(self):
        with checkpoint_boundary():
            with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):
                _ = self.model(self.directory / "missing").model

    def test_existing_unet_directory_without_weights_fails(self):
        (self.directory / "unet").mkdir()
        with checkpoint_boundary():
            with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):
                _ = self.model().model

    def test_saved_weights_reload_through_the_public_constructor(self):
        instance = self.model(protein_map={"CUSTOM": 37}, cellline_map={"CUSTOM_CELL": 9})
        instance._model = TinyCheckpoint(6.75)
        destination = self.directory / "saved"
        with contextlib.redirect_stdout(io.StringIO()):
            instance.save(destination)
        with checkpoint_boundary():
            reloaded = self.model(destination).model
        self.assertEqual(reloaded.weight.item(), 6.75)

    def test_saving_over_an_ema_checkpoint_reloads_the_new_weights(self):
        TinyCheckpoint(7.5).save_pretrained(self.directory / "unet_ema")
        (self.directory / "proticelli_checkpoint.json").write_text(
            json.dumps({"version": 1, "inference_subfolder": "unet_ema"}), encoding="utf-8"
        )
        instance = self.model(protein_map={"CUSTOM": 1}, cellline_map={"C": 1})
        instance._model = TinyCheckpoint(3.25)
        with contextlib.redirect_stdout(io.StringIO()):
            instance.save(self.directory)
        with checkpoint_boundary():
            reloaded = self.model().model
        self.assertEqual(reloaded.weight.item(), 3.25)

    def test_registered_accelerate_load_hook_restores_saved_weights(self):
        from proticelli.utils.checkpoint_utils import setup_checkpoint_hooks

        TinyCheckpoint(4.5).save_pretrained(self.directory / "unet")
        target = TinyCheckpoint(0)
        hooks = {}
        accelerator = types.SimpleNamespace(
            is_main_process=True, device="cpu",
            register_save_state_pre_hook=lambda hook: hooks.update(save=hook),
            register_load_state_pre_hook=lambda hook: hooks.update(load=hook),
        )
        setup_checkpoint_hooks(accelerator, types.SimpleNamespace(use_ema=False))
        models = [target]
        with checkpoint_boundary():
            hooks["load"](models, str(self.directory))
        self.assertEqual(models, [])
        self.assertEqual(target.weight.item(), 4.5)

    def test_saved_custom_label_maps_override_bundled_defaults_when_reloaded(self):
        instance = self.model(protein_map={"CUSTOM": 37}, cellline_map={"CUSTOM_CELL": 9})
        instance._model = TinyCheckpoint(6.75)
        destination = self.directory / "saved"
        with contextlib.redirect_stdout(io.StringIO()):
            instance.save(destination)
        reloaded = self.model(destination)
        self.assertEqual(
            (reloaded.protein_map.get("CUSTOM"), reloaded.cellline_map.get("CUSTOM_CELL")),
            (37, 9),
            "Model.save() label maps must accompany the saved model on reload",
        )
        self.assertEqual(set(reloaded.protein_map), {"CUSTOM"})
        self.assertEqual(set(reloaded.cellline_map), {"CUSTOM_CELL"})


class SigmaSchedule:
    def __init__(self, sigmas):
        self._values = tuple(sigmas)

    def set_timesteps(self, count):
        if count != len(self._values) - 1:
            raise AssertionError("The scalar reference schedule has a fixed step count")
        self.sigmas = torch.tensor(self._values, dtype=torch.float64)


class ZeroRawDenoiser:
    def __call__(self, image, timestep, **_kwargs):
        return types.SimpleNamespace(sample=torch.zeros_like(image[:, :16]))


def scalar_zero_denoiser_reference(initial, sigmas, solver):
    """Integrate dx/dsigma = sigma*x/(sigma**2 + .25) with scalar arithmetic.

    A zero raw EDM network gives D(x,sigma)=.25*x/(sigma**2+.25).
    The terminal transition to zero uses Euler: denoiser conditioning log(sigma)
    is undefined at zero, so a second denoiser evaluation is not permitted there.
    """
    value = float(initial) * sigmas[0]
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        derivative = sigma * value / (sigma * sigma + 0.25)
        step = next_sigma - sigma
        predicted = value + step * derivative
        if solver == "heun" and next_sigma > 0:
            next_derivative = next_sigma * predicted / (next_sigma**2 + 0.25)
            value += step * (derivative + next_derivative) / 2
        else:
            value = predicted
    return value


class SamplingRegressionTests(unittest.TestCase):
    def uncertainty(self, sigmas, *, solver="euler", **kwargs):
        return sample_edm_uncertainty(
            ZeroRawDenoiser(), SigmaSchedule(sigmas), batch_size=1, image_size=2,
            num_inference_steps=len(sigmas) - 1, device="cpu", weight_dtype=torch.float64,
            initial_latents=torch.ones((1, 16, 2, 2), dtype=torch.float64),
            solver=solver, **kwargs,
        )

    def test_plain_edm_euler_matches_independent_scalar_ode_reference(self):
        sigmas = (1.0, 0.5, 0.0)
        seed = 17
        initial = torch.randn(
            (2, 16, 2, 2), generator=torch.Generator().manual_seed(seed), dtype=torch.float64
        )
        actual = sample_edm(
            ZeroRawDenoiser(), SigmaSchedule(sigmas), batch_size=2, image_size=2,
            num_inference_steps=2, device="cpu", weight_dtype=torch.float64,
            generator=torch.Generator().manual_seed(seed),
        )
        expected_factor = scalar_zero_denoiser_reference(1.0, sigmas, "euler")
        self.assertAlmostEqual(expected_factor, 0.3)
        torch.testing.assert_close(actual, initial * expected_factor, rtol=1e-12, atol=1e-12)

    def test_uncertainty_euler_matches_independent_scalar_ode_reference(self):
        sigmas = (1.0, 0.5, 0.25, 0.0)
        actual = self.uncertainty(sigmas)
        expected = scalar_zero_denoiser_reference(1.0, sigmas, "euler")
        torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=1e-12, atol=1e-12)

    def test_heun_positive_endpoint_matches_independent_scalar_reference(self):
        sigmas = (1.0, 0.5, 0.25)
        actual = self.uncertainty(sigmas, solver="heun")
        expected = scalar_zero_denoiser_reference(1.0, sigmas, "heun")
        self.assertAlmostEqual(expected, 0.52)
        torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=1e-12, atol=1e-12)

    def test_heun_terminal_zero_uses_the_final_euler_transition(self):
        sigmas = (1.0, 0.5, 0.0)
        actual = self.uncertainty(sigmas, solver="heun")
        expected = scalar_zero_denoiser_reference(1.0, sigmas, "heun")
        self.assertAlmostEqual(expected, 0.325)
        self.assertTrue(torch.isfinite(actual).all().item(), "Terminal sigma zero produced non-finite latents")
        torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=1e-12, atol=1e-12)

    def test_single_heun_step_to_zero_remains_finite(self):
        actual = self.uncertainty((1.0, 0.0), solver="heun")
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, torch.full_like(actual, 0.2), rtol=1e-12, atol=1e-12)

    def test_sampler_cancellation_propagates_before_denoising(self):
        with self.assertRaises(InterruptedError):
            self.uncertainty((1.0, 0.5, 0.0), cancel_check=lambda: True)


def bare_model():
    return Model(device="cpu", protein_map={"A": 1}, cellline_map={"C": 1})


class PreprocessingRegressionTests(unittest.TestCase):
    def test_raw_uint8_channels_use_the_full_dtype_range(self):
        image = np.asarray([[[0, 128, 255], [255, 64, 0]]], dtype=np.uint8)
        expected = np.moveaxis(image.astype(np.float32) / 255 * 2 - 1, -1, 0)
        np.testing.assert_allclose(bare_model()._preprocess_image(image).numpy(), expected)

    def test_dim_raw_uint16_channels_do_not_infer_eight_bit_depth(self):
        image = np.asarray([[[0, 128, 255], [32, 64, 200]]], dtype=np.uint16)
        expected = np.moveaxis(image.astype(np.float32) / 65535 * 2 - 1, -1, 0)
        np.testing.assert_allclose(bare_model()._preprocess_image(image).numpy(), expected)

    def test_signed_normalized_float_channels_are_preserved(self):
        image = np.asarray([[[-0.8, 0.2, 0.9], [0.1, -0.3, 0.7]]], dtype=np.float32)
        np.testing.assert_array_equal(
            bare_model()._preprocess_image(image).numpy(), np.moveaxis(image, -1, 0)
        )

    def test_four_channel_preprocessing_ignores_the_target_channel(self):
        image = np.asarray([[[0, 255, 128, 255], [255, 0, 64, 32]]], dtype=np.uint8)
        expected = np.moveaxis(image[..., [0, 2, 3]].astype(np.float32) / 255 * 2 - 1, -1, 0)
        actual = bare_model()._preprocess_image(image).numpy()
        np.testing.assert_allclose(actual, expected)

    def test_normalizer_pipeline_preserves_conditioning_with_negative_pixels(self):
        ramp = np.asarray([[0, 20], [40, 100]], dtype=np.uint8)
        stack = ChannelAssembler(has_protein=False).transform({
            "microtubules": ramp, "nucleus": ramp, "er": ramp,
        })
        normalized = ImageNormalizer(bit_depth=8).transform(stack)
        expected = np.moveaxis(normalized[..., [0, 2, 3]], -1, 0)
        np.testing.assert_array_equal(bare_model()._preprocess_image(normalized).numpy(), expected)

    def test_normalizer_pipeline_preserves_conditioning_when_all_reference_pixels_are_positive(self):
        stack = ChannelAssembler(has_protein=False).transform({
            "microtubules": np.full((2, 2), 70, dtype=np.uint8),
            "nucleus": np.full((2, 2), 100, dtype=np.uint8),
            "er": np.full((2, 2), 70, dtype=np.uint8),
        })
        normalized = ImageNormalizer(bit_depth=8).transform(stack)
        # These are known signed-normalized values, not raw [0,1] inputs:
        # 2*(70/100)-1=.4 for MT/ER, and 1 for the nucleus reference.
        expected = np.broadcast_to(np.asarray([0.4, 1, 0.4], dtype=np.float32)[:, None, None], (3, 2, 2))
        np.testing.assert_allclose(bare_model()._preprocess_image(normalized).numpy(), expected, atol=1e-7)


class LatentDistribution:
    def __init__(self, mean, stochastic):
        self.mean = mean
        self.stochastic = stochastic

    def sample(self, generator=None):
        if not self.stochastic:
            return self.mean.clone()
        return self.mean + torch.randn(
            self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype
        )

    def mode(self):
        return self.mean.clone()


class SmallVAE:
    config = types.SimpleNamespace(scaling_factor=4)

    def __init__(self, *, stochastic=False, before_encode=None):
        self.stochastic = stochastic
        self.before_encode = before_encode
        self.encoded_batches = []

    def encode(self, condition):
        if self.before_encode is not None:
            self.before_encode()
        self.encoded_batches.append(len(condition))
        downsampled = torch.nn.functional.avg_pool2d(condition, kernel_size=8)
        mean = downsampled.mean(dim=1, keepdim=True).repeat(1, 16, 1, 1)
        return types.SimpleNamespace(latent_dist=LatentDistribution(mean, self.stochastic))


def conditioning_image(index=0):
    ramp = np.linspace(-0.5, 0.5, 512 * 512, dtype=np.float32).reshape(512, 512)
    return np.stack((ramp, ramp + 0.05, ramp - 0.05), axis=-1) + index * 0.01


def boundary_sampler(**kwargs):
    """Deterministic conditional response given reference latents and noise."""
    reference = kwargs["reference_channels"]
    initial = kwargs.get("initial_latents")
    if initial is None:
        noise = torch.randn(reference.shape, generator=kwargs.get("generator"), dtype=reference.dtype)
    else:
        noise = initial[..., :reference.shape[-2], :reference.shape[-1]]
    return reference + 0.1 * noise


@contextlib.contextmanager
def prediction_boundaries(*, stochastic=False, before_encode=None):
    instance = bare_model()
    instance._vae = SmallVAE(stochastic=stochastic, before_encode=before_encode)
    instance._model = object()
    instance._scheduler = object()
    instance._decode_latents = lambda latents: torch.nn.functional.interpolate(
        latents[:, :1].detach(), scale_factor=8, mode="nearest"
    )
    with patch("proticelli._sampling.sample_edm", side_effect=boundary_sampler), patch(
        "proticelli._sampling.sample_edm_uncertainty", side_effect=boundary_sampler
    ):
        yield instance


class PredictionOrchestrationRegressionTests(unittest.TestCase):
    def setUp(self):
        state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, state)
        torch.random.default_generator.manual_seed(193)

    def predict(self, instance, **kwargs):
        return instance.predict([conditioning_image()], ["A"], ["C"], show_progress=False, **kwargs)

    def reliability(self, instance, **kwargs):
        options = {"num_samples": 5, "batch_size": 2, "seed": 23, "show_progress": False}
        options.update(kwargs)
        return instance.predict_with_reliability([conditioning_image()], ["A"], ["C"], **options)

    def test_seeded_plain_prediction_control_with_deterministic_encoder(self):
        with prediction_boundaries() as instance:
            first = self.predict(instance, seed=23)
            second = self.predict(instance, seed=23)
        np.testing.assert_array_equal(first.images[0], second.images[0])

    def test_different_plain_prediction_seeds_change_noise(self):
        with prediction_boundaries() as instance:
            first = self.predict(instance, seed=23)
            second = self.predict(instance, seed=24)
        self.assertFalse(np.array_equal(first.images[0], second.images[0]))

    def test_bfloat16_latent_export_preserves_representable_values_in_numpy(self):
        # A known latent at the learned sampler boundary isolates export from
        # the diffusion algorithm. 0.3125 is exactly representable in BF16.
        latent = torch.full((1, 16, 64, 64), 0.3125, dtype=torch.bfloat16)
        with prediction_boundaries() as instance:
            instance.dtype = torch.bfloat16
            instance._decode_latents = lambda value: torch.nn.functional.interpolate(
                value[:, :1].float(), scale_factor=8, mode="nearest"
            )
            with patch("proticelli._sampling.sample_edm", return_value=latent):
                result = self.predict(instance, return_latents=True)
        self.assertEqual(result.latents[0].dtype, np.float32)
        np.testing.assert_array_equal(result.latents[0], np.full((16, 64, 64), 0.3125))

    def test_seeded_prediction_includes_stochastic_conditioning_encode(self):
        with prediction_boundaries(stochastic=True) as instance:
            first = self.predict(instance, seed=23)
            second = self.predict(instance, seed=23)
        np.testing.assert_array_equal(first.images[0], second.images[0])

    def test_seeded_reliability_includes_stochastic_conditioning_encode(self):
        with prediction_boundaries(stochastic=True) as instance:
            first = self.reliability(instance, return_ensembles=True)
            second = self.reliability(instance, return_ensembles=True)
        np.testing.assert_array_equal(first.ensembles[0], second.ensembles[0])
        np.testing.assert_array_equal(first.images[0], second.images[0])
        np.testing.assert_array_equal(first.reliability_scores, second.reliability_scores)

    def test_requested_ensemble_medoid_and_score_match_independent_correlations(self):
        with prediction_boundaries() as instance:
            result = self.reliability(instance, return_ensembles=True)
        ensemble = result.ensembles[0]
        correlation = np.corrcoef(ensemble.reshape(len(ensemble), -1).astype(np.float64))
        centrality = (correlation.sum(axis=1) - 1) / (len(ensemble) - 1)
        medoid = int(np.argmax(centrality))
        score = np.clip((correlation.sum() - len(ensemble)) / (len(ensemble) * (len(ensemble) - 1)), 0, 1)
        np.testing.assert_array_equal(result.images[0], ensemble[medoid])
        self.assertAlmostEqual(result.reliability_scores[0], score, places=6)

    def test_unrequested_ensemble_storage_is_not_retained_by_the_medoid(self):
        with prediction_boundaries() as instance:
            result = self.reliability(instance, return_ensembles=False)
        self.assertIsNone(result.ensembles)
        retained_roots = {}
        for image in result.images:
            root = image
            while isinstance(root.base, np.ndarray):
                root = root.base
            retained_roots[id(root)] = root
        retained_bytes = sum(root.nbytes for root in retained_roots.values())
        output_bytes = sum(image.nbytes for image in result.images)
        self.assertLessEqual(
            retained_bytes, output_bytes,
            "A returned medoid retains backing storage for unrequested ensemble members",
        )

    def test_encoder_batches_follow_requested_limit_and_preserve_output_count(self):
        images = [conditioning_image(index) for index in range(7)]
        with prediction_boundaries() as instance:
            result = instance.predict(images, ["A"] * 7, ["C"] * 7,
                                      batch_size=2, seed=23, show_progress=False)
            batches = instance.vae.encoded_batches
        self.assertEqual(batches, [2, 2, 2, 1])
        self.assertEqual(len(result.images), 7)

    def assert_bounded_preprocessing(self, reliability):
        references = []
        batch_size = 2

        def before_encode():
            live = sum(reference() is not None for reference in references)
            # Allow an active and previous batch. Retaining all eight inputs
            # violates a bound proportional to batch_size, even on tiny images.
            self.assertLessEqual(live, 2 * batch_size, "All preprocessed inputs were retained at once")

        with prediction_boundaries(before_encode=before_encode) as instance:
            original = instance._preprocess_image

            def monitored(image, **options):
                tensor = original(image, **options)
                references.append(weakref.ref(tensor))
                return tensor

            with patch.object(instance, "_preprocess_image", side_effect=monitored):
                method = instance.predict_with_reliability if reliability else instance.predict
                kwargs = {"num_samples": 2} if reliability else {}
                result = method(
                    [conditioning_image(index) for index in range(8)], ["A"] * 8, ["C"] * 8,
                    batch_size=batch_size, seed=23, show_progress=False, **kwargs,
                )
        self.assertEqual(len(result.images), 8)

    def test_plain_prediction_preprocesses_a_bounded_number_of_images(self):
        self.assert_bounded_preprocessing(reliability=False)

    def test_reliability_preprocesses_a_bounded_number_of_images(self):
        self.assert_bounded_preprocessing(reliability=True)


if __name__ == "__main__":
    unittest.main()
