"""Test-first checks for core fine-tuning behavior; production is unchanged.

Numerical references come from TIFF dtype ranges and the algebraic EDM loss
weight. Integration cases exercise the actual data loader, autograd, optimizer,
and training loop with tiny replacements for the learned VAE and denoiser.
Known defects deliberately fail; they are not marked as expected failures.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
from numpy.testing import assert_allclose
import torch
from tifffile import imwrite

from proticelli import _training
from proticelli.utils.edm_utils import edm_loss_weight


class FineTuneInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def load(self, array):
        name = 'cell.tif'
        imwrite(self.root / name, array, photometric='minisblack')
        return _training.FinetuneDataset(str(self.root), [name], [7], [3])[0]

    def assert_normalization(self, dtype, pixels):
        raw = np.broadcast_to(np.asarray(pixels, dtype=dtype), (3, 5, 4)).copy()
        sample = self.load(raw)
        # A raw integer count is normalized using its acquisition dtype, not
        # the brightest observed pixel. Channel1 is the target, 0/2/3 condition.
        expected = raw.astype(np.float64) / np.iinfo(dtype).max * 2 - 1
        assert_allclose(sample['cond_image'].numpy(), expected[..., [0, 2, 3]].transpose(2, 0, 1), atol=1e-7)
        assert_allclose(sample['gt_image'].numpy(), expected[..., 1:2].transpose(2, 0, 1), atol=1e-7)
        self.assertEqual(sample['label'], 7)
        self.assertEqual(sample['cell_line'], 3)

    def test_raw_uint8_with_nontrivial_channel_values(self):
        self.assert_normalization(np.uint8, [20, 255, 80, 160])

    def test_raw_uint16_with_full_scale_reference(self):
        self.assert_normalization(np.uint16, [200, 65535, 2048, 40000])

    def test_dim_uint16_retains_its_sixteen_bit_scale(self):
        self.assert_normalization(np.uint16, [20, 128, 70, 200])

    def test_one_count_uint8_is_not_a_unit_range_float(self):
        self.assert_normalization(np.uint8, [0, 1, 1, 0])

    def test_unit_range_float_tiff_is_scaled_once(self):
        raw = np.broadcast_to(np.array([0.0, 0.25, 0.5, 1.0], np.float32), (3, 5, 4)).copy()
        sample = self.load(raw)
        assert_allclose(sample['cond_image'].numpy()[:, 0, 0], [-1, 0, 1], atol=1e-7)
        assert_allclose(sample['gt_image'].numpy(), -0.5, atol=1e-7)

    def test_signed_float_tiff_keeps_reference_order_and_target(self):
        raw = np.broadcast_to(np.array([-0.8, 0.25, 0.4, 0.6], np.float32), (3, 5, 4)).copy()
        sample = self.load(raw)
        assert_allclose(sample['cond_image'].numpy()[:, 0, 0], [-0.8, 0.4, 0.6], atol=1e-7)
        assert_allclose(sample['gt_image'].numpy(), 0.25, atol=1e-7)


class EDMLossWeightTests(unittest.TestCase):
    def check_weights(self, dtype):
        sigma = torch.tensor([0.002, 0.5, 80.0], dtype=dtype)
        # (s^2+d^2)/(s*d)^2 = 1/s^2 + 1/d^2. Use represented sigma
        # values in a float64 reference, so half quantization is accounted for.
        expected = 1 / sigma.double().numpy() ** 2 + 4
        actual = edm_loss_weight(sigma)
        self.assertTrue(torch.isfinite(actual).all().item(), f'Non-finite weights: {actual}')
        assert_allclose(actual.double().numpy(), expected, rtol=5e-7, atol=1e-7)

    def test_float32_weights_match_independent_algebraic_reference(self):
        self.check_weights(torch.float32)

    def test_half_precision_input_still_yields_finite_accurate_loss_weights(self):
        self.check_weights(torch.float16)


class _TinyDenoiser(torch.nn.Module):
    """Consumes one timestep per sample, as the real timestep embedding does."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.forward_calls = 0
        self.saved_weights = []

    def forward(self, image, timestep, **kwargs):
        self.forward_calls += 1
        # Normal batched embedding/broadcasting requires a one-dimensional
        # timestep vector. There is no assertion tailored to the implementation.
        temporal = timestep[:, None, None, None]
        prediction = self.weight * torch.ones_like(image[:, :16]) + temporal * 0.01
        return types.SimpleNamespace(sample=prediction)

    def save_pretrained(self, path):
        self.saved_weights.append((str(path), self.weight.detach().clone()))


class _TinyVAE(torch.nn.Module):
    config = types.SimpleNamespace(scaling_factor=4.0)

    def encode(self, image):
        latent = image.mean(dim=1, keepdim=True).repeat(1, 16, 1, 1)
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(sample=lambda: latent.clone()))


class _QuietProgress:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **kwargs):
        pass


class _StatefulSchedule:
    """Constant-rate scheduler boundary that supports actual checkpoint state."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.steps = 0
        self.step = Mock(wraps=self._step)

    def _step(self):
        self.steps += 1

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {'steps': self.steps}

    def load_state_dict(self, state):
        self.steps = state['steps']


class FineTuneLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        self.optimizers = []
        self.schedules = []

    def run_loop(self, n_images, *, batch_size, accumulation=1, **options):
        names = []
        for index in range(n_images):
            name = f'cell_{index}.tif'
            image = np.broadcast_to(np.array([-0.6, -0.2, 0.3, 0.7], np.float32), (2, 2, 4)).copy()
            imwrite(self.images / name, image, photometric='minisblack')
            names.append(name)
        network = _TinyDenoiser()
        model = types.SimpleNamespace(
            device=torch.device('cpu'), dtype=torch.float32, model=network, vae=_TinyVAE(),
            protein_map={'P': 1}, cellline_map={'C': 1}, _resolve_protein_name=lambda name: name,
        )
        original_optimizer = torch.optim.AdamW

        def optimizer_factory(*args, **kwargs):
            optimizer = original_optimizer(*args, **kwargs)
            optimizer.step = Mock(wraps=optimizer.step)
            self.optimizers.append(optimizer)
            return optimizer

        def schedule_factory(name, *, optimizer, **kwargs):
            schedule = _StatefulSchedule(optimizer)
            self.schedules.append(schedule)
            return schedule

        optimization_module = importlib.import_module('diffusers.optimization')
        scheduler_module = importlib.import_module('proticelli.schedulers.edm_scheduler')

        def noise_schedule_factory(**kwargs):
            return types.SimpleNamespace(
                config=types.SimpleNamespace(num_train_timesteps=1000),
                sigmas=torch.full((1001,), 0.5),
            )

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(optimization_module, 'get_scheduler', schedule_factory))
            stack.enter_context(patch.object(scheduler_module, 'create_edm_scheduler', noise_schedule_factory))
            stack.enter_context(patch.object(_training.torch.optim, 'AdamW', optimizer_factory))
            stack.enter_context(patch.object(_training, 'tqdm', lambda iterable, **kwargs: _QuietProgress(iterable)))
            stack.enter_context(torch.random.fork_rng(devices=[]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(warnings.catch_warnings())
            warnings.filterwarnings('ignore', message=".*pin_memory.*", category=UserWarning)
            torch.random.default_generator.manual_seed(19)
            _training.run_finetuning(
                model, str(self.images), names, ['P'] * n_images, ['C'] * n_images,
                num_epochs=1, batch_size=batch_size, gradient_accumulation_steps=accumulation,
                num_workers=0, lr_warmup_steps=0, learning_rate=1e-3,
                output_dir=str(self.root / 'output'), checkpointing_steps=1000,
                **options,
            )
        return model, network

    def test_regular_batches_update_parameters_and_finish_in_eval_mode(self):
        model, network = self.run_loop(4, batch_size=2)
        self.assertEqual(network.forward_calls, 2)
        self.assertEqual(self.optimizers[0].step.call_count, 2)
        self.assertEqual(self.schedules[0].step.call_count, 2)
        self.assertTrue(math.isfinite(network.weight.item()))
        self.assertNotEqual(network.weight.item(), torch.tensor(0.1).item())
        self.assertFalse(network.training)
        self.assertFalse(network.weight.requires_grad)
        self.assertEqual(model.checkpoint_dir, self.root / 'output')
        self.assertTrue(network.saved_weights)

    def test_batch_size_one_is_a_valid_training_batch(self):
        _, network = self.run_loop(1, batch_size=1)
        self.assertEqual(network.forward_calls, 1)
        self.assertEqual(self.optimizers[0].step.call_count, 1)
        self.assertTrue(torch.isfinite(network.weight).item())

    def test_final_singleton_remainder_is_trained_without_shape_failure(self):
        _, network = self.run_loop(3, batch_size=2)
        self.assertEqual(network.forward_calls, 2)
        self.assertEqual(self.optimizers[0].step.call_count, 2)

    def test_gradient_accumulation_updates_once_per_group_including_remainder(self):
        self.run_loop(10, batch_size=2, accumulation=2)
        # Ten samples / two samples per minibatch = five minibatches.
        # Two minibatches per update require three updates, including the tail.
        self.assertEqual(self.optimizers[0].step.call_count, 3)
        self.assertEqual(self.schedules[0].step.call_count, 3)

    def test_missing_resume_checkpoint_cannot_silently_start_training(self):
        with self.assertRaises((FileNotFoundError, NotImplementedError)):
            self.run_loop(2, batch_size=2, resume_from=str(self.root / 'missing-checkpoint'))


if __name__ == '__main__':
    unittest.main()
