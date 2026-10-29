"""Bound temporary model-sized storage while restoring an EMA checkpoint.

The training loop and checkpoint I/O are real. A large inert buffer makes the
model/EMA copies distinguishable from tiny optimizer and activation tensors,
without measuring process RSS or allocating a production-sized network.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import weakref

import numpy as np
import torch
from tifffile import imwrite
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from tests.test_training_controls import ControlledDenoiser, TrainingHarness


class _StorageBudget(TorchDispatchMode):
    """Observe distinct live large storages without owning their tensors."""

    def __init__(self, minimum_bytes):
        super().__init__()
        self.minimum_bytes = minimum_bytes
        self.references = {}
        self.peak_bytes = 0
        self.frozen = False

    def observe(self, value):
        if self.frozen:
            return
        for tensor in tree_flatten(value)[0]:
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                continue
            storage = tensor.untyped_storage()
            if storage.nbytes() >= self.minimum_bytes:
                key = (storage.data_ptr(), storage.nbytes())
                self.references.setdefault(key, []).append(weakref.ref(tensor))
        live_bytes = 0
        for key, references in list(self.references.items()):
            live = []
            for reference in references:
                tensor = reference()
                if tensor is not None:
                    storage = tensor.untyped_storage()
                    if (storage.data_ptr(), storage.nbytes()) == key:
                        live.append(reference)
            if live:
                self.references[key] = live
                live_bytes += key[1]
            else:
                del self.references[key]
        self.peak_bytes = max(self.peak_bytes, live_bytes)

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        self.observe(result)
        return result

    def freeze(self):
        self.frozen = True


class _BufferedDenoiser(ControlledDenoiser):
    def __init__(self):
        super().__init__()
        self.register_buffer("memory_probe", torch.full((65536,), 0.75, dtype=torch.float32))


class ResumeMemoryTests(unittest.TestCase):
    def test_ema_resume_has_no_discarded_full_model_shadow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            filename = "cell.tif"
            imwrite(images / filename, np.full((2, 2, 4), 0.2, np.float32),
                    photometric="minisblack")

            def harness(output):
                run = TrainingHarness(images, [filename], root / output)
                run.network = _BufferedDenoiser().eval().requires_grad_(False)
                run.model._model = run.network
                return run

            checkpoint = harness("checkpoint").run(use_ema=True)
            resumed = harness("resumed")
            buffer_bytes = resumed.network.memory_probe.nbytes
            observer = _StorageBudget(buffer_bytes)
            observer.observe(resumed.network.memory_probe)
            # The budget applies to restoring state, before activations and
            # subsequent training/checkpoint exports introduce other lifetimes.
            hook = resumed.network.register_forward_pre_hook(
                lambda _module, _inputs: observer.freeze()
            )
            try:
                with observer:
                    resumed.run(use_ema=True, num_epochs=2,
                                resume_from=str(checkpoint.output))
            finally:
                hook.remove()

            self.assertTrue(observer.frozen, "The resumed training did not reach its first forward pass")
            # CPU resume can retain the live network, raw checkpoint network,
            # and its saved EMA. These three distinct buffers are sufficient;
            # an additional full shadow, even transiently, is unnecessary.
            self.assertLessEqual(
                observer.peak_bytes, 3 * buffer_bytes,
                "EMA resume exceeded the live model + raw checkpoint + saved EMA storage budget",
            )


if __name__ == "__main__":
    unittest.main()
