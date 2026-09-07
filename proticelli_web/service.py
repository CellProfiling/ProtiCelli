"""State, persistence, and single-worker job orchestration for the web app."""

from __future__ import annotations

import json
import hashlib
import os
import pickle
import shutil
import sqlite3
import threading
import uuid
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .imaging import (
    CHANNEL_KEYS,
    channel_statistics,
    file_sha256,
    inspect_image,
    prepare_reference_channels,
    prediction_png_bytes,
    read_image_plane,
    read_reference_tiff,
    scalar_statistics,
    suggest_channel_role,
)
from .inference import InferenceEngine


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


@dataclass
class InputRecord:
    id: str
    name: str
    source_path: Path
    array_path: Path
    sha256: str
    shape: tuple[int, int, int]
    dtype: str
    channels: list[dict[str, Any]]
    created_at: str
    preprocessing: dict[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "sha256": self.sha256,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "channels": self.channels,
            "created_at": self.created_at,
            "preprocessing": self.preprocessing or {},
        }

    def stored(self) -> dict[str, Any]:
        return {
            **self.public(),
            "source_file": self.source_path.name,
            "array_file": self.array_path.name,
        }


class InputStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, InputRecord] = {}
        self._lock = threading.RLock()
        self._restore()

    def _record_path(self, input_id: str) -> Path:
        return self.root / f"{input_id}.record.json"

    def _persist(self, record: InputRecord) -> None:
        _json_dump(self._record_path(record.id), record.stored())

    def _restore(self) -> None:
        for path in self.root.glob("*.record.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                record = InputRecord(
                    id=payload["id"],
                    name=payload["name"],
                    source_path=self.root / payload["source_file"],
                    array_path=self.root / payload["array_file"],
                    sha256=payload["sha256"],
                    shape=tuple(int(value) for value in payload["shape"]),
                    dtype=payload["dtype"],
                    channels=payload["channels"],
                    created_at=payload["created_at"],
                    preprocessing=payload.get("preprocessing", {}),
                )
                if record.array_path.is_file():
                    self._records[record.id] = record
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    def add_bytes(self, name: str, payload: bytes) -> InputRecord:
        input_id = uuid.uuid4().hex
        source_path = self.root / f"{input_id}.tiff"
        source_path.write_bytes(payload)
        try:
            array = read_reference_tiff(source_path)
            array_path = self.root / f"{input_id}.npy"
            np.save(array_path, array, allow_pickle=False)
            record = InputRecord(
                id=input_id,
                name=Path(name).name,
                source_path=source_path,
                array_path=array_path,
                sha256=file_sha256(source_path),
                shape=tuple(int(v) for v in array.shape),
                dtype=str(array.dtype),
                channels=[item.as_dict() for item in channel_statistics(array)],
                created_at=utc_now(),
                preprocessing={"mode": "canonical_tiff", "channel_order": list(CHANNEL_KEYS)},
            )
        except Exception:
            source_path.unlink(missing_ok=True)
            raise
        with self._lock:
            self._records[input_id] = record
            self._persist(record)
        return record

    def add_array(
        self,
        *,
        name: str,
        array: np.ndarray,
        source_sha256: str,
        preprocessing: dict[str, Any],
    ) -> InputRecord:
        """Persist an explicitly mapped, resampled, and cropped reference array."""

        values = np.asarray(array)
        if values.shape != (512, 512, 3):
            raise ValueError(f"Prepared input has unexpected shape {tuple(values.shape)}")
        input_id = uuid.uuid4().hex
        array_path = self.root / f"{input_id}.npy"
        source_path = self.root / f"{input_id}.source.json"
        np.save(array_path, values, allow_pickle=False)
        _json_dump(source_path, {"sha256": source_sha256, "preprocessing": preprocessing})
        record = InputRecord(
            id=input_id,
            name=Path(name).name,
            source_path=source_path,
            array_path=array_path,
            sha256=source_sha256,
            shape=tuple(int(value) for value in values.shape),
            dtype=str(values.dtype),
            channels=[item.as_dict() for item in channel_statistics(values)],
            created_at=utc_now(),
            preprocessing=preprocessing,
        )
        with self._lock:
            self._records[input_id] = record
            self._persist(record)
        return record

    def add_path(self, path: Path) -> InputRecord:
        return self.add_bytes(path.name, path.read_bytes())

    def get(self, input_id: str) -> InputRecord:
        with self._lock:
            record = self._records.get(input_id)
        if record is None:
            raise KeyError(input_id)
        return record

    def array(self, input_id: str) -> np.ndarray:
        return np.load(self.get(input_id).array_path, allow_pickle=False)


@dataclass
class UploadRecord:
    id: str
    directory: Path
    files: list[dict[str, Any]]
    created_at: str

    def public(self) -> dict[str, Any]:
        files = [
            {key: value for key, value in item.items() if key != "stored_file"}
            for item in self.files
        ]
        return {"id": self.id, "files": files, "created_at": self.created_at}


class UploadStore:
    """Temporary multi-format inspection sessions used by the channel mapper."""

    def __init__(self, root: Path, inputs: InputStore):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.inputs = inputs
        self._records: dict[str, UploadRecord] = {}
        self._lock = threading.RLock()

    def add_files(self, files: list[tuple[str, bytes]]) -> UploadRecord:
        upload_id = uuid.uuid4().hex
        directory = self.root / upload_id
        directory.mkdir(parents=True, exist_ok=False)
        inspected: list[dict[str, Any]] = []
        try:
            for file_index, (name, payload) in enumerate(files):
                suffix = Path(name).suffix.casefold()
                path = directory / f"{file_index:03d}{suffix}"
                path.write_bytes(payload)
                details = inspect_image(path)
                details.update(
                    {
                        "file_index": file_index,
                        "name": Path(name).name,
                        "sha256": file_sha256(path),
                        "stored_file": path.name,
                    }
                )
                for plane in details["planes"]:
                    plane["suggested_role"] = (
                        suggest_channel_role(f"{name} {plane['label']}")
                        or plane.get("suggested_role")
                    )
                inspected.append(details)
        except Exception:
            for path in directory.glob("*"):
                path.unlink(missing_ok=True)
            directory.rmdir()
            raise
        record = UploadRecord(upload_id, directory, inspected, utc_now())
        with self._lock:
            self._records[upload_id] = record
        return record

    def get(self, upload_id: str) -> UploadRecord:
        with self._lock:
            record = self._records.get(upload_id)
        if record is None:
            raise KeyError(upload_id)
        return record

    def plane(self, upload_id: str, file_index: int, plane_index: int) -> np.ndarray:
        record = self.get(upload_id)
        if file_index < 0 or file_index >= len(record.files):
            raise IndexError(file_index)
        return read_image_plane(record.directory / record.files[file_index]["stored_file"], plane_index)

    def assemble(
        self,
        *,
        upload_id: str,
        mapping: dict[str, dict[str, int]],
        pixel_size_um: float,
        resample: bool,
        crop_x: int | None,
        crop_y: int | None,
        normalize: bool = False,
        normalization_bit_depth: int | None = None,
    ) -> InputRecord:
        record = self.get(upload_id)
        channels: dict[str, np.ndarray] = {}
        mapping_manifest: dict[str, Any] = {}
        for role in CHANNEL_KEYS:
            selected = mapping.get(role)
            if selected is None:
                raise ValueError(f"Missing mapping for {role}")
            file_index = int(selected["file_index"])
            plane_index = int(selected["plane_index"])
            channels[role] = self.plane(upload_id, file_index, plane_index)
            file_info = record.files[file_index]
            plane_info = file_info["planes"][plane_index]
            mapping_manifest[role] = {
                "file": file_info["name"],
                "file_sha256": file_info["sha256"],
                "plane_index": plane_index,
                "plane_label": plane_info["label"],
            }
        array, transformation = prepare_reference_channels(
            channels,
            pixel_size_um=pixel_size_um,
            resample=resample,
            crop_x=crop_x,
            crop_y=crop_y,
            normalize=normalize,
            normalization_bit_depth=normalization_bit_depth,
        )
        digest = hashlib.sha256()
        for item in record.files:
            digest.update(item["sha256"].encode("ascii"))
        digest.update(json.dumps(mapping_manifest, sort_keys=True).encode("utf-8"))
        preprocessing = {
            "mode": "mapped_image_planes",
            "channel_order": list(CHANNEL_KEYS),
            "mapping": mapping_manifest,
            **transformation,
        }
        names = ", ".join(item["name"] for item in record.files)
        return self.inputs.add_array(
            name=f"Mapped channels · {names}",
            array=array,
            source_sha256=digest.hexdigest(),
            preprocessing=preprocessing,
        )


class WorkspaceRunStore:
    """Persistent user-facing workspaces backed by a small local SQLite file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_runs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    input_id TEXT,
                    cell_line TEXT,
                    notes TEXT NOT NULL DEFAULT '',
                    defaults_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        defaults = json.loads(row["defaults_json"])
        defaults.setdefault("cell_line", row["cell_line"])
        return {
            "id": row["id"],
            "name": row["name"],
            "input_id": row["input_id"],
            "cell_line": row["cell_line"],
            "notes": row["notes"],
            "defaults": defaults,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create(
        self,
        *,
        name: str,
        input_id: str | None = None,
        cell_line: str | None = None,
        notes: str = "",
        defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        timestamp = utc_now()
        values = defaults or {
            "num_samples": 1,
            "num_inference_steps": 50,
            "seed": 42,
            "pixel_size_um": 0.1067,
            "cell_line": None,
            "dtype": "float32",
        }
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO workspace_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    name.strip() or "Untitled run",
                    input_id,
                    cell_line,
                    notes.strip(),
                    json.dumps(values, allow_nan=False),
                    timestamp,
                    timestamp,
                ),
            )
        return self.get(run_id)

    def get(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM workspace_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._public(row)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM workspace_runs ORDER BY updated_at DESC"
            ).fetchall()
        return [self._public(row) for row in rows]

    def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {"name", "input_id", "cell_line", "notes", "defaults"}
        values = {key: value for key, value in changes.items() if key in allowed}
        if not values:
            return self.get(run_id)
        if "name" in values:
            values["name"] = str(values["name"]).strip() or "Untitled run"
        if "notes" in values:
            values["notes"] = str(values["notes"]).strip()
        if "defaults" in values:
            values["defaults_json"] = json.dumps(values.pop("defaults"), allow_nan=False)
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                f"UPDATE workspace_runs SET {assignments} WHERE id = ?",
                (*values.values(), run_id),
            )
        if cursor.rowcount == 0:
            raise KeyError(run_id)
        return self.get(run_id)

    def duplicate(self, run_id: str, *, name: str | None = None) -> dict[str, Any]:
        source = self.get(run_id)
        return self.create(
            name=name or f"{source['name']} copy",
            input_id=source["input_id"],
            cell_line=source["cell_line"],
            notes=source["notes"],
            defaults=source["defaults"],
        )

    def delete(self, run_id: str) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM workspace_runs WHERE id = ?", (run_id,)
            )
        if cursor.rowcount == 0:
            raise KeyError(run_id)

    def touch(self, run_id: str) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE workspace_runs SET updated_at = ? WHERE id = ?",
                (utc_now(), run_id),
            )
        if cursor.rowcount == 0:
            raise KeyError(run_id)


class JobManager:
    def __init__(self, root: Path, inputs: InputStore, engine: InferenceEngine):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.inputs = inputs
        self.engine = engine
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="proticelli-gpu")
        self._restore()

    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _record_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _persist_locked(self, job_id: str) -> None:
        directory = self._job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        _json_dump(self._record_path(job_id), self._jobs[job_id])

    def _restore(self) -> None:
        for path in self.root.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                job_id = str(job["id"])
                if job.get("status") in {"queued", "running"}:
                    job.update(
                        status="cancelled",
                        progress=0.0,
                        message="Interrupted by the previous application shutdown",
                        completed_at=utc_now(),
                    )
                    _json_dump(path, job)
                self._jobs[job_id] = job
                self._cancel_events[job_id] = threading.Event()
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    def close(self) -> None:
        with self._lock:
            for event in self._cancel_events.values():
                event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def submit(
        self,
        *,
        run_id: str | None = None,
        input_id: str,
        protein: str,
        cell_line: str | None,
        num_samples: int,
        num_inference_steps: int,
        seed: int,
        pixel_size_um: float,
        dtype: str = "float32",
    ) -> dict[str, Any]:
        input_record = self.inputs.get(input_id)
        job_id = uuid.uuid4().hex
        record: dict[str, Any] = {
            "id": job_id,
            "run_id": run_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Waiting for inference worker",
            "created_at": utc_now(),
            "started_at": None,
            "completed_at": None,
            "input": input_record.public(),
            "request": {
                "protein": protein,
                "cell_line": cell_line,
                "num_samples": num_samples,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                "pixel_size_um": pixel_size_um,
                "dtype": dtype,
            },
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = record
            self._cancel_events[job_id] = threading.Event()
            self._persist_locked(job_id)
            self._futures[job_id] = self._executor.submit(self._run, job_id)
        return self.public(job_id)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(values)
            self._persist_locked(job_id)

    def _progress(self, job_id: str, value: float, message: str) -> None:
        self._update(job_id, progress=max(0.0, min(float(value), 0.99)), message=message)

    def _run(self, job_id: str) -> None:
        try:
            with self._lock:
                job = dict(self._jobs[job_id])
            self._update(
                job_id,
                status="running",
                started_at=utc_now(),
                progress=0.02,
                message="Preparing inference request",
            )
            request = job["request"]
            reference = self.inputs.array(job["input"]["id"])
            output = self.engine.generate(
                reference=reference,
                protein=request["protein"],
                cell_line=request["cell_line"],
                num_samples=request["num_samples"],
                num_inference_steps=request["num_inference_steps"],
                seed=request["seed"],
                progress=lambda value, message: self._progress(job_id, value, message),
                cancel_check=self._cancel_events[job_id].is_set,
                dtype=request.get("dtype", "float32"),
            )
            if self._cancel_events[job_id].is_set():
                raise InterruptedError("Inference cancelled")

            job_dir = self._job_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            from tifffile import imwrite

            image_records = []
            for index, image in enumerate(output.images):
                raw_name = f"sample-{index:02d}.tiff"
                preview_name = f"sample-{index:02d}.png"
                imwrite(job_dir / raw_name, np.asarray(image, dtype=np.float32))
                (job_dir / preview_name).write_bytes(prediction_png_bytes(image))
                image_records.append(
                    {
                        "index": index,
                        "seed": request["seed"] + index * 1009,
                        "is_medoid": index == output.medoid_index,
                        "raw_tiff": raw_name,
                        "preview_png": preview_name,
                        "statistics": scalar_statistics(image),
                    }
                )

            manifest = {
                "schema_version": "1.0",
                "created_at": utc_now(),
                "engine": output.engine,
                "model_version": "0.1.0",
                "run_id": job.get("run_id"),
                "input": {
                    "name": job["input"]["name"],
                    "sha256": job["input"]["sha256"],
                    "shape": job["input"]["shape"],
                    "dtype": job["input"]["dtype"],
                    "channel_order": list(CHANNEL_KEYS),
                    "pixel_size_um": request["pixel_size_um"],
                    "preprocessing": job["input"].get("preprocessing", {}),
                },
                "conditioning": {
                    "protein": request["protein"],
                    "cell_line": request["cell_line"],
                },
                "sampling": {
                    "num_samples": request["num_samples"],
                    "num_inference_steps": request["num_inference_steps"],
                    "base_seed": request["seed"],
                    "compute_dtype": request.get("dtype", "float32"),
                },
                "ensemble": {
                    "medoid_index": output.medoid_index,
                    "reliability": None if not np.isfinite(output.reliability) else output.reliability,
                },
                "images": image_records,
                "display_note": "Intensity adjustments are display-only; raw TIFF values are unchanged.",
            }
            _json_dump(job_dir / "manifest.json", manifest)
            archive_path = job_dir / "proticelli-export.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(job_dir / "manifest.json", "manifest.json")
                for image_record in image_records:
                    archive.write(job_dir / image_record["raw_tiff"], image_record["raw_tiff"])
                    archive.write(job_dir / image_record["preview_png"], image_record["preview_png"])

            self._update(
                job_id,
                status="succeeded",
                progress=1.0,
                message="Prediction complete" if request["num_samples"] == 1 else "Ensemble complete",
                completed_at=utc_now(),
                result={
                    "engine": output.engine,
                    "medoid_index": output.medoid_index,
                    "reliability": manifest["ensemble"]["reliability"],
                    "images": image_records,
                    "manifest": manifest,
                    "job_dir": str(job_dir),
                    "archive_path": str(archive_path),
                },
            )
        except InterruptedError:
            self._update(
                job_id,
                status="cancelled",
                completed_at=utc_now(),
                progress=0.0,
                message="Inference stopped",
                error=None,
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                completed_at=utc_now(),
                message="Inference failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def public(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = dict(self._jobs.get(job_id) or {})
        if not job:
            raise KeyError(job_id)
        result = job.get("result")
        if result:
            result = {key: value for key, value in result.items() if key not in {"job_dir", "archive_path"}}
            job["result"] = result
        return job

    def list(self, run_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            ids = [
                job_id for job_id, job in self._jobs.items()
                if run_id is None or job.get("run_id") == run_id
            ]
        return sorted((self.public(job_id) for job_id in ids), key=lambda item: item["created_at"], reverse=True)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
            if job["status"] not in {"queued", "running"}:
                return self.public(job_id)
            future = self._futures[job_id]
            self._cancel_events[job_id].set()
            if future.cancel():
                job.update(status="cancelled", completed_at=utc_now(), message="Cancelled", progress=0.0)
            else:
                job["message"] = "Stopping after the current denoising step"
            self._persist_locked(job_id)
        return self.public(job_id)

    def delete_run(self, run_id: str) -> int:
        """Delete completed job records and artifacts owned by one workspace run."""
        with self._lock:
            job_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.get("run_id") == run_id
            ]
            active = [
                job_id
                for job_id in job_ids
                if self._jobs[job_id].get("status") in {"queued", "running"}
                or (
                    (future := self._futures.get(job_id)) is not None
                    and not future.done()
                )
            ]
            if active:
                raise RuntimeError(
                    "Cancel active inference and wait for it to stop before deleting this run"
                )

            root = self.root.resolve()
            directories: list[Path] = []
            for job_id in job_ids:
                directory = self._job_dir(job_id).resolve()
                if directory.parent != root:
                    raise ValueError("Invalid job storage path")
                directories.append(directory)

            for directory in directories:
                if directory.exists():
                    shutil.rmtree(directory)
            for job_id in job_ids:
                self._jobs.pop(job_id, None)
                self._futures.pop(job_id, None)
                self._cancel_events.pop(job_id, None)
        return len(job_ids)

    def result_file(self, job_id: str, index: int, kind: str) -> Path:
        with self._lock:
            result = (self._jobs.get(job_id) or {}).get("result")
        if not result:
            raise KeyError(job_id)
        images = result["images"]
        if index < 0 or index >= len(images):
            raise IndexError(index)
        key = "preview_png" if kind == "preview" else "raw_tiff"
        return Path(result["job_dir"]) / images[index][key]

    def archive(self, job_id: str) -> Path:
        with self._lock:
            result = (self._jobs.get(job_id) or {}).get("result")
        if not result:
            raise KeyError(job_id)
        return Path(result["archive_path"])


def load_vocab(package_root: Path) -> tuple[list[str], list[str]]:
    data_dir = package_root / "proticelli" / "data"
    with (data_dir / "antibody_map.pkl").open("rb") as handle:
        proteins = sorted(pickle.load(handle).keys())
    with (data_dir / "cell_line_map.pkl").open("rb") as handle:
        cell_lines = sorted(pickle.load(handle).keys())
    return proteins, cell_lines


def default_data_dir() -> Path:
    configured = os.getenv("PROTICELLI_WEB_DATA_DIR")
    return Path(configured).expanduser() if configured else Path.cwd() / "proticelli_web_data"
