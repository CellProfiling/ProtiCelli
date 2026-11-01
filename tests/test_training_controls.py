"""Tiny CPU integrations for accumulation, resumable state, EMA, and AMP.

The optimizer, autograd, TIFF dataset, checkpoint files, and training loop are
real. Only learned networks, the fixed sigma schedule, and a deterministic LR
factory are controlled. Accumulation gradients have an independent algebraic
reference; resume is compared to an uninterrupted stochastic training run.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import io
import json
import math
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch
from tifffile import imwrite

from proticelli import Model, _training


class ControlledDenoiser(torch.nn.Module):
    def __init__(self, *, analytic=False, fail_on=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.1]], dtype=torch.float32))
        self.register_buffer("forward_count", torch.tensor(0, dtype=torch.int64))
        self.analytic = analytic
        self.fail_on = fail_on
        self.calls = 0
        self.traces = []
        self.projection_dtypes = []
        self.parameter_dtypes = []

    def forward(self, image, timestep, **_kwargs):
        self.calls += 1
        if self.calls == self.fail_on:
            raise InterruptedError("controlled interruption at a model boundary")
        self.forward_count.add_(1)
        self.traces.append({
            "target": image[:, 16:].detach().float().mean(dim=(1, 2, 3)).clone(),
            "input": image[:, :16].detach().float().clone(),
        })
        ones = torch.ones_like(image[:, :1]).movedim(1, -1)
        projection = torch.nn.functional.linear(ones, self.weight).movedim(-1, 1)
        self.projection_dtypes.append(projection.dtype)
        self.parameter_dtypes.append(self.weight.dtype)
        prediction = projection.repeat(1, 16, 1, 1)
        if self.analytic:
            # At sigma=.5, c_in=sqrt(2), c_skip=.5, c_out=1/(2*sqrt(2)).
            # F=w-c_in*x cancels the noisy contribution in D=c_skip*x+c_out*F.
            prediction = prediction - image[:, :16]
        else:
            prediction = prediction + 0.01 * timestep[:, None, None, None]
        return types.SimpleNamespace(sample=prediction)

    def save_pretrained(self, directory, **_kwargs):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "test_weights.pt")
        (directory / "config.json").write_text("{}", encoding="utf-8")


class ControlledVAE(torch.nn.Module):
    config = types.SimpleNamespace(scaling_factor=4.0)

    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.tensor(0.0))

    @property
    def dtype(self):
        return self.anchor.dtype

    def encode(self, image):
        mean = image.mean(dim=1, keepdim=True).repeat(1, 16, 1, 1)
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(sample=lambda: mean.clone()))


class QuietProgress:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **_kwargs):
        pass


class TrainingHarness:
    def __init__(self, images, names, output, *, analytic=False, fail_on=None, dtype="float32"):
        self.images, self.names, self.output = images, names, output
        self.model = Model(device="cpu", dtype=dtype, protein_map={"P": 1}, cellline_map={"C": 1})
        self.network = ControlledDenoiser(analytic=analytic, fail_on=fail_on)
        self.network.to(self.model.dtype).eval().requires_grad_(False)
        self.model._model = self.network
        self.model._vae = ControlledVAE().to(self.model.dtype).eval().requires_grad_(False)
        self.optimizers, self.schedules, self.steps = [], [], []

    def run(self, **overrides):
        options = {
            "num_epochs": 1, "batch_size": 2, "gradient_accumulation_steps": 1,
            "learning_rate": 0.01, "lr_scheduler_type": "linear", "lr_warmup_steps": 0,
            "checkpointing_steps": 1, "save_model_epochs": 1, "num_workers": 0,
            "max_grad_norm": 1e6, "adam_weight_decay": 0,
            "input_range": "normalized", "seed": 139,
        }
        options.update(overrides)
        optimizer_class = torch.optim.AdamW

        def optimizer_factory(*args, **kwargs):
            optimizer = optimizer_class(*args, **kwargs)
            self.optimizers.append(optimizer)
            return optimizer

        def lr_factory(_name, *, optimizer, **_kwargs):
            # A real stateful schedule with nonconstant LR makes missing LR
            # restoration observable when comparing resumed optimization.
            schedule = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: max(0.1, 1 - step / 20)
            )
            original_step = optimizer.step

            @functools.wraps(original_step)
            def step(*args, **kwargs):
                record = {
                    "before": self.network.weight.detach().float().clone(),
                    "gradient": self.network.weight.grad.detach().float().clone(),
                    "through_trace": len(self.network.traces),
                }
                result = original_step(*args, **kwargs)
                record["after"] = self.network.weight.detach().float().clone()
                self.steps.append(record)
                return result

            optimizer.step = step
            self.schedules.append(schedule)
            return schedule

        optimization = importlib.import_module("diffusers.optimization")
        edm = importlib.import_module("proticelli.schedulers.edm_scheduler")
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(_training.torch.optim, "AdamW", optimizer_factory))
                stack.enter_context(patch.object(optimization, "get_scheduler", lr_factory))
                stack.enter_context(patch.object(edm, "create_edm_scheduler", lambda **_kwargs: types.SimpleNamespace(
                    config=types.SimpleNamespace(num_train_timesteps=1000),
                    sigmas=torch.full((1001,), 0.5),
                )))
                stack.enter_context(patch.object(_training, "tqdm", lambda iterable, **_kwargs: QuietProgress(iterable)))
                stack.enter_context(torch.random.fork_rng(devices=[]))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(warnings.catch_warnings())
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*GradScaler.*")
                # Intentionally unrelated initial state. Explicit seed or resumed
                # checkpoint RNG must control the actual training draws.
                torch.random.default_generator.manual_seed(991)
                self.model.fit(
                    self.images, self.names, ["P"] * len(self.names), ["C"] * len(self.names),
                    output_dir=str(self.output), **options,
                )
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        return self

    def state(self, path=None):
        path = self.output / "training_state.pt" if path is None else Path(path)
        return torch.load(path, map_location="cpu", weights_only=False)


class TrainingControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.names = []
        for index, value in enumerate((-0.6, -0.1, 0.2, 0.7, 0.9, 0.4)):
            filename = f"cell_{index:03d}.tif"
            imwrite(self.images / filename, np.full((2, 2, 4), value, np.float32), photometric="minisblack")
            self.names.append(filename)

    def harness(self, name, n=6, **kwargs):
        return TrainingHarness(self.images, self.names[:n], self.root / name, **kwargs)

    def assert_gradient_reference(self, run):
        previous = 0
        for step in run.steps:
            groups = run.network.traces[previous:step["through_trace"]]
            targets = torch.cat([group["target"] for group in groups]).double().numpy()
            # With the analytic denoiser, D=w/(2*sqrt(2)), weight(sigma)=8.
            # The sample-mean squared loss therefore has dL/dw=2w-4sqrt(2)*mean(y).
            expected = 2 * step["before"].double().numpy() - 4 * math.sqrt(2) * targets.mean()
            np.testing.assert_allclose(step["gradient"].numpy(), expected, rtol=2e-6, atol=2e-6)
            previous = step["through_trace"]

    def test_accumulation_weights_unequal_minibatches_by_samples(self):
        run = self.harness("unequal", n=3, analytic=True).run(
            batch_size=2, gradient_accumulation_steps=2
        )
        self.assertEqual(len(run.steps), 1)
        self.assert_gradient_reference(run)

    def test_accumulation_tail_uses_its_actual_sample_count(self):
        run = self.harness("tail", n=5, analytic=True).run(
            batch_size=2, gradient_accumulation_steps=2
        )
        self.assertEqual(len(run.steps), 2)
        self.assert_gradient_reference(run)

    def test_seeded_fresh_training_has_a_reproducible_passing_control(self):
        first = self.harness("first", n=4).run()
        second = self.harness("second", n=4).run()
        torch.testing.assert_close(first.state()["model"]["weight"], second.state()["model"]["weight"], rtol=0, atol=0)
        for a, b in zip(first.network.traces, second.network.traces):
            torch.testing.assert_close(a["input"], b["input"], rtol=0, atol=0)

    def test_mid_epoch_resume_restores_model_optimizer_lr_cursor_and_rng(self):
        uninterrupted = self.harness("uninterrupted").run(num_epochs=2, use_ema=True, ema_decay=0.5)
        interrupted = self.harness("interrupted", fail_on=3)
        with self.assertRaises(InterruptedError):
            interrupted.run(num_epochs=2, use_ema=True, ema_decay=0.5)
        checkpoint = interrupted.output / "checkpoint-2"
        partial = interrupted.state(checkpoint / "training_state.pt")
        self.assertEqual((partial["epoch"], partial["next_batch"], partial["global_step"]), (0, 2, 2))
        resumed = self.harness("resumed").run(
            num_epochs=2, resume_from=str(checkpoint), use_ema=True, ema_decay=0.5
        )
        expected, actual = uninterrupted.state(), resumed.state()
        required = {"version", "model", "optimizer", "lr_scheduler", "scaler", "ema",
                    "epoch", "next_batch", "global_step", "data_seed", "rng", "configuration"}
        self.assertTrue(required.issubset(actual))
        self.assertEqual((actual["epoch"], actual["next_batch"], actual["global_step"]), (2, 0, 6))
        for name in expected["model"]:
            torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=0, atol=0)
            torch.testing.assert_close(actual["ema"][name], expected["ema"][name], rtol=0, atol=0)
        self.assertEqual(actual["lr_scheduler"]["last_epoch"], expected["lr_scheduler"]["last_epoch"])
        self.assertEqual(actual["optimizer"]["param_groups"], expected["optimizer"]["param_groups"])
        for parameter, state in expected["optimizer"]["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(actual["optimizer"]["state"][parameter][key], value, rtol=0, atol=0)
        torch.testing.assert_close(actual["rng"]["torch_cpu"], expected["rng"]["torch_cpu"], rtol=0, atol=0)
        self.assertEqual(len(resumed.network.traces), len(uninterrupted.network.traces) - 2)
        for a, b in zip(resumed.network.traces, uninterrupted.network.traces[2:]):
            torch.testing.assert_close(a["input"], b["input"], rtol=0, atol=0)

    def test_resume_accepts_a_training_state_file_path(self):
        complete = self.harness("complete", n=4).run()
        checkpoint = complete.output / "checkpoint-1" / "training_state.pt"
        resumed = self.harness("file_resume", n=4).run(resume_from=str(checkpoint))
        torch.testing.assert_close(resumed.state()["model"]["weight"], complete.state()["model"]["weight"], rtol=0, atol=0)

    def test_ema_updates_follow_the_declared_recurrence_and_export_separately(self):
        decay = 0.5
        run = self.harness("ema", n=4).run(use_ema=True, ema_decay=decay)
        expected = torch.tensor([[0.1]], dtype=torch.float32)
        for step in run.steps:
            expected = decay * expected + (1 - decay) * step["after"]
        state = run.state()
        torch.testing.assert_close(state["ema"]["weight"], expected)
        torch.testing.assert_close(run.model.model.weight, expected)
        raw = torch.load(run.output / "unet" / "test_weights.pt", weights_only=True)
        ema = torch.load(run.output / "unet_ema" / "test_weights.pt", weights_only=True)
        torch.testing.assert_close(raw["weight"], state["model"]["weight"])
        torch.testing.assert_close(ema["weight"], expected)
        self.assertEqual(state["ema"]["forward_count"].dtype, torch.int64)
        self.assertEqual(state["ema"]["forward_count"].item(), len(run.network.traces))
        torch.testing.assert_close(ema["forward_count"], raw["forward_count"])
        self.assertFalse(torch.equal(raw["weight"], ema["weight"]))
        metadata = json.loads((run.output / "proticelli_checkpoint.json").read_text())
        self.assertEqual(metadata["inference_subfolder"], "unet_ema")

    def test_bf16_cpu_uses_autocast_with_fp32_master_parameters(self):
        run = self.harness("bf16", n=4).run(mixed_precision="bf16")
        self.assertTrue(all(dtype == torch.bfloat16 for dtype in run.network.projection_dtypes))
        self.assertTrue(all(dtype == torch.float32 for dtype in run.network.parameter_dtypes))
        self.assertEqual(run.state()["model"]["weight"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(run.network.weight).all().item())

    def test_fp16_cpu_is_rejected_before_a_training_forward(self):
        run = self.harness("invalid_fp16", n=4)
        with self.assertRaisesRegex(ValueError, "(?i)fp16|cuda"):
            run.run(mixed_precision="fp16")
        self.assertEqual(run.network.calls, 0)
        self.assertFalse(run.network.training)
        self.assertFalse(run.network.weight.requires_grad)

    def test_half_precision_inference_model_trains_in_fp32_and_restores_dtype(self):
        run = self.harness("half_inference", n=4, dtype="float16").run()
        self.assertTrue(all(dtype == torch.float32 for dtype in run.network.parameter_dtypes))
        self.assertEqual(run.state()["model"]["weight"].dtype, torch.float32)
        self.assertEqual(run.network.weight.dtype, torch.float16)
        self.assertEqual(run.model.vae.dtype, torch.float16)
        self.assertFalse(run.network.training)
        self.assertFalse(run.network.weight.requires_grad)

    def test_training_failure_restores_inference_dtype_eval_and_frozen_parameters(self):
        run = self.harness("failed_half", n=4, dtype="float16", fail_on=1)
        with self.assertRaises(InterruptedError):
            run.run()
        self.assertEqual(run.network.weight.dtype, torch.float16)
        self.assertEqual(run.model.vae.dtype, torch.float16)
        self.assertFalse(run.network.training)
        self.assertFalse(run.network.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
