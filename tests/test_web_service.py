from __future__ import annotations

import sys
import io
import tempfile
import threading
import time
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from proticelli_web.inference import DemoInferenceEngine
from proticelli_web.service import InputRecord, InputStore, JobManager, UploadStore, WorkspaceRunStore


class StubInputs:
    def __init__(self, root: Path):
        self.reference = np.zeros((512, 512, 3), dtype=np.float32)
        y, x = np.mgrid[:512, :512]
        self.reference[..., 1] = np.exp(-(((x - 256) ** 2 + (y - 256) ** 2) / 8000))
        source = root / "source.tiff"
        array = root / "source.npy"
        source.write_bytes(b"source")
        np.save(array, self.reference)
        self.record = InputRecord(
            id="input-1",
            name="source.tiff",
            source_path=source,
            array_path=array,
            sha256="abc123",
            shape=(512, 512, 3),
            dtype="float32",
            channels=[],
            created_at="2026-01-01T00:00:00Z",
        )

    def get(self, input_id: str) -> InputRecord:
        if input_id != self.record.id:
            raise KeyError(input_id)
        return self.record

    def array(self, input_id: str) -> np.ndarray:
        self.get(input_id)
        return self.reference


class BlockingInferenceEngine:
    name = "blocking"

    def __init__(self):
        self.started = threading.Event()

    def generate(self, *, cancel_check, **_):
        self.started.set()
        while not cancel_check():
            time.sleep(0.01)
        raise InterruptedError("Inference cancelled")


class WebServiceTests(unittest.TestCase):
    def test_png_upload_can_be_mapped_cropped_and_padded(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            array = np.zeros((120, 160, 3), dtype=np.uint8)
            array[..., 0] = 10
            array[..., 1] = 20
            array[..., 2] = 30
            payload = io.BytesIO()
            Image.fromarray(array, mode="RGB").save(payload, format="PNG")
            uploads = UploadStore(root / "uploads", InputStore(root / "inputs"))

            session = uploads.add_files([("reference.png", payload.getvalue())])
            public = session.public()
            self.assertEqual(public["files"][0]["format"], "PNG")
            self.assertNotIn("stored_file", public["files"][0])
            record = uploads.assemble(
                upload_id=session.id,
                mapping={
                    "microtubules": {"file_index": 0, "plane_index": 0},
                    "nucleus": {"file_index": 0, "plane_index": 2},
                    "er": {"file_index": 0, "plane_index": 1},
                },
                pixel_size_um=0.1067,
                resample=False,
                crop_x=None,
                crop_y=None,
            )
            prepared = np.load(record.array_path)
            self.assertEqual(prepared.shape, (512, 512, 3))
            self.assertEqual(record.preprocessing["mode"], "mapped_image_planes")
            self.assertEqual([int(value) for value in prepared[256, 256]], [10, 30, 20])

    def test_prepared_inputs_are_restored_after_restart(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory) / "inputs"
            original = np.zeros((512, 512, 3), dtype=np.float32)
            store = InputStore(root)
            record = store.add_array(
                name="mapped-reference.tiff",
                array=original,
                source_sha256="source-sha",
                preprocessing={"mode": "mapped_tiff_planes"},
            )

            restored = InputStore(root).get(record.id)
            self.assertEqual(restored.name, "mapped-reference.tiff")
            self.assertEqual(restored.shape, (512, 512, 3))
            self.assertEqual(restored.preprocessing["mode"], "mapped_tiff_planes")
            np.testing.assert_array_equal(np.load(restored.array_path), original)

    def test_workspace_runs_persist_and_duplicate_without_predictions(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "gallery.sqlite3"
            store = WorkspaceRunStore(path)
            run = store.create(name="Organelle survey", cell_line=None)
            self.assertEqual(run["defaults"]["num_samples"], 1)
            updated = store.update(run["id"], notes="same reference context")
            duplicate = store.duplicate(run["id"])
            store.close()

            reopened = WorkspaceRunStore(path)
            try:
                restored = reopened.get(run["id"])
                self.assertEqual(restored["name"], "Organelle survey")
                self.assertIsNone(restored["cell_line"])
                self.assertEqual(restored["notes"], "same reference context")
                self.assertEqual(duplicate["input_id"], restored["input_id"])
                self.assertEqual(len(reopened.list()), 2)
                reopened.delete(run["id"])
                with self.assertRaises(KeyError):
                    reopened.get(run["id"])
                self.assertEqual(len(reopened.list()), 1)
            finally:
                reopened.close()

    def test_job_export_contains_raw_samples_and_manifest(self):
        fake_tifffile = types.SimpleNamespace(
            imwrite=lambda path, array: Path(path).write_bytes(np.asarray(array).tobytes())
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            manager = JobManager(root / "runs", StubInputs(root), DemoInferenceEngine())
            try:
                with patch.dict(sys.modules, {"tifffile": fake_tifffile}):
                    job = manager.submit(
                        run_id="run-1",
                        input_id="input-1",
                        protein="TOMM20",
                        cell_line="A-431",
                        num_samples=2,
                        num_inference_steps=10,
                        seed=42,
                        pixel_size_um=0.1067,
                    )
                    deadline = time.time() + 8
                    while job["status"] not in {"succeeded", "failed"} and time.time() < deadline:
                        time.sleep(0.05)
                        job = manager.public(job["id"])
                self.assertEqual(job["status"], "succeeded", job.get("error"))
                self.assertEqual(job["result"]["engine"], "demo")
                self.assertEqual(job["result"]["manifest"]["sampling"]["compute_dtype"], "float32")
                archive = manager.archive(job["id"])
                with zipfile.ZipFile(archive) as exported:
                    self.assertIn("manifest.json", exported.namelist())
                    self.assertIn("sample-00.tiff", exported.namelist())
                    self.assertIn("sample-01.png", exported.namelist())
                manager.close()
                restored = JobManager(root / "runs", StubInputs(root), DemoInferenceEngine())
                try:
                    restored_job = restored.public(job["id"])
                    self.assertEqual(restored_job["run_id"], "run-1")
                    self.assertEqual(restored_job["status"], "succeeded")
                    self.assertEqual(restored.delete_run("run-1"), 1)
                    self.assertFalse((root / "runs" / job["id"]).exists())
                    with self.assertRaises(KeyError):
                        restored.public(job["id"])
                finally:
                    restored.close()
            finally:
                manager.close()

    def test_run_deletion_is_blocked_until_active_inference_stops(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            engine = BlockingInferenceEngine()
            manager = JobManager(root / "runs", StubInputs(root), engine)
            try:
                job = manager.submit(
                    run_id="run-active",
                    input_id="input-1",
                    protein="TOMM20",
                    cell_line=None,
                    num_samples=1,
                    num_inference_steps=10,
                    seed=42,
                    pixel_size_um=0.1067,
                )
                self.assertTrue(engine.started.wait(timeout=2))
                with self.assertRaisesRegex(RuntimeError, "Cancel active inference"):
                    manager.delete_run("run-active")

                manager.cancel(job["id"])
                deadline = time.time() + 2
                status = manager.public(job["id"])["status"]
                while status != "cancelled" and time.time() < deadline:
                    time.sleep(0.02)
                    status = manager.public(job["id"])["status"]
                self.assertEqual(status, "cancelled")
                self.assertEqual(manager.delete_run("run-active"), 1)
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
