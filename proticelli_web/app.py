"""FastAPI application serving ProtiCelli inference and the browser client."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .imaging import (
    CHANNEL_KEYS,
    ImageValidationError,
    SUPPORTED_IMAGE_SUFFIXES,
    channel_png_bytes,
    channel_preview_png_bytes,
)
from .inference import create_engine, real_assets_ready
from .service import (
    InputStore,
    JobManager,
    UploadStore,
    WorkspaceRunStore,
    default_data_dir,
    load_vocab,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 128 * 1024 * 1024


class GenerationRequest(BaseModel):
    run_id: str
    input_id: str
    protein: str = Field(min_length=1, max_length=300)
    cell_line: str | None = Field(default=None, max_length=120)
    num_samples: int = Field(default=1, ge=1, le=50)
    num_inference_steps: int = Field(default=50, ge=5, le=200)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    pixel_size_um: float = Field(default=0.1067, gt=0, le=100)
    dtype: Literal["float32", "float16"] = "float32"


class PlaneSelection(BaseModel):
    file_index: int = Field(ge=0)
    plane_index: int = Field(ge=0)


class AssembleInputRequest(BaseModel):
    upload_id: str
    mapping: dict[str, PlaneSelection]
    pixel_size_um: float = Field(default=0.1067, gt=0, le=100)
    resample: bool = False
    crop_x: int | None = None
    crop_y: int | None = None
    normalize: bool = False
    normalization_bit_depth: Literal[8, 16] | None = None


class RunCreateRequest(BaseModel):
    name: str = Field(default="Untitled run", min_length=1, max_length=160)
    input_id: str | None = None
    notes: str = Field(default="", max_length=2000)


class RunDefaultsRequest(BaseModel):
    num_samples: int = Field(default=1, ge=1, le=50)
    num_inference_steps: int = Field(default=50, ge=5, le=200)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    pixel_size_um: float = Field(default=0.1067, gt=0, le=100)
    cell_line: str | None = Field(default=None, max_length=120)
    dtype: Literal["float32", "float16"] = "float32"


class RunUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    input_id: str | None = None
    notes: str | None = Field(default=None, max_length=2000)
    defaults: RunDefaultsRequest | None = None


class RunDuplicateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)


def create_app() -> FastAPI:
    data_dir = default_data_dir()
    mode = os.getenv("PROTICELLI_WEB_MODE", "auto")
    engine = create_engine(PACKAGE_ROOT, mode)
    inputs = InputStore(data_dir / "inputs")
    uploads = UploadStore(data_dir / "uploads", inputs)
    jobs = JobManager(data_dir / "runs", inputs, engine)
    workspaces = WorkspaceRunStore(data_dir / "gallery.sqlite3")
    proteins, cell_lines = load_vocab(PACKAGE_ROOT)
    protein_set = set(proteins)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        jobs.close()
        workspaces.close()

    app = FastAPI(
        title="ProtiCelli Interactive Gallery",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "engine": engine.name, "runtime": getattr(engine, "runtime_info", None)}

    @app.get("/api/config")
    def config() -> dict:
        return {
            "engine": engine.name,
            "real_assets_ready": real_assets_ready(PACKAGE_ROOT),
            "model_version": "0.1.0",
            "native_pixel_size_um": 0.1067,
            "required_shape": [512, 512],
            "channel_order": list(CHANNEL_KEYS),
            "cell_lines": cell_lines,
            "protein_count": len(proteins),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "supports_mapped_uploads": True,
            "supported_input_extensions": sorted(SUPPORTED_IMAGE_SUFFIXES),
            "runtime": getattr(engine, "runtime_info", None),
        }

    @app.get("/api/vocabulary/proteins")
    def protein_search(
        q: str = Query(default="", max_length=100),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        term = q.strip().casefold()
        if not term:
            matches = proteins[:limit]
        else:
            starts = [name for name in proteins if name.casefold().startswith(term)]
            contains = [name for name in proteins if term in name.casefold() and name not in starts]
            matches = (starts + contains)[:limit]
        return {"items": matches, "total": len(proteins)}

    def run_payload(run: dict, *, include_predictions: bool = False) -> dict:
        predictions = jobs.list(run["id"])
        enriched = dict(run)
        defaults = dict(enriched.get("defaults") or {})
        if "cell_line" not in defaults:
            defaults["cell_line"] = enriched.get("cell_line")
        enriched["defaults"] = defaults
        enriched["prediction_count"] = len(predictions)
        enriched["active_count"] = sum(
            prediction["status"] in {"queued", "running"} for prediction in predictions
        )
        enriched["succeeded_count"] = sum(
            prediction["status"] == "succeeded" for prediction in predictions
        )
        enriched["conditions"] = sorted(
            {
                prediction.get("request", {}).get("cell_line") or "Unconditional"
                for prediction in predictions
            }
        )
        if run.get("input_id"):
            try:
                enriched["input"] = inputs.get(run["input_id"]).public()
            except KeyError:
                enriched["input"] = None
        else:
            enriched["input"] = None
        if include_predictions:
            enriched["predictions"] = predictions
        return enriched

    @app.get("/api/runs")
    def list_runs() -> dict:
        return {"items": [run_payload(run) for run in workspaces.list()]}

    @app.post("/api/runs", status_code=201)
    def create_run(payload: RunCreateRequest) -> dict:
        if payload.input_id:
            try:
                inputs.get(payload.input_id)
            except KeyError as exc:
                raise HTTPException(404, "Input not found") from exc
        values = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        return run_payload(workspaces.create(**values), include_predictions=True)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        try:
            return run_payload(workspaces.get(run_id), include_predictions=True)
        except KeyError as exc:
            raise HTTPException(404, "Run not found") from exc

    @app.patch("/api/runs/{run_id}")
    def update_run(run_id: str, payload: RunUpdateRequest) -> dict:
        values = (
            payload.model_dump(exclude_unset=True)
            if hasattr(payload, "model_dump")
            else payload.dict(exclude_unset=True)
        )
        defaults = values.get("defaults")
        if defaults and defaults.get("cell_line") and defaults["cell_line"] not in cell_lines:
            raise HTTPException(422, "Cell line is not in the ProtiCelli vocabulary")
        if values.get("input_id"):
            try:
                inputs.get(values["input_id"])
            except KeyError as exc:
                raise HTTPException(404, "Input not found") from exc
        try:
            current = workspaces.get(run_id)
        except KeyError as exc:
            raise HTTPException(404, "Run not found") from exc
        if jobs.list(run_id):
            if "input_id" in values and values["input_id"] != current["input_id"]:
                raise HTTPException(409, "Reference input is locked after the first prediction; create or duplicate a run")
        return run_payload(workspaces.update(run_id, **values), include_predictions=True)

    @app.post("/api/runs/{run_id}/duplicate", status_code=201)
    def duplicate_run(run_id: str, payload: RunDuplicateRequest) -> dict:
        try:
            return run_payload(
                workspaces.duplicate(run_id, name=payload.name),
                include_predictions=True,
            )
        except KeyError as exc:
            raise HTTPException(404, "Run not found") from exc

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str) -> dict:
        try:
            workspaces.get(run_id)
            deleted_predictions = jobs.delete_run(run_id)
            workspaces.delete(run_id)
        except KeyError as exc:
            raise HTTPException(404, "Run not found") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {
            "deleted": True,
            "run_id": run_id,
            "deleted_predictions": deleted_predictions,
        }

    @app.post("/api/inputs", status_code=201)
    async def upload_input(file: UploadFile = File(...)) -> dict:
        filename = file.filename or "reference.tiff"
        if Path(filename).suffix.lower() not in {".tif", ".tiff"}:
            raise HTTPException(415, "Upload a .tif or .tiff file")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        if not payload:
            raise HTTPException(400, "The uploaded file is empty")
        try:
            return inputs.add_bytes(filename, payload).public()
        except ImageValidationError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/uploads", status_code=201)
    async def inspect_uploads(files: list[UploadFile] = File(...)) -> dict:
        if not files:
            raise HTTPException(400, "Choose at least one image file")
        if len(files) > 24:
            raise HTTPException(413, "A mapping session can contain at most 24 files")
        collected: list[tuple[str, bytes]] = []
        total = 0
        for file in files:
            filename = file.filename or "reference"
            if Path(filename).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
                raise HTTPException(415, f"{filename}: unsupported image type. Use {supported}")
            payload = await file.read(MAX_UPLOAD_BYTES + 1)
            total += len(payload)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"Combined upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            if not payload:
                raise HTTPException(400, f"{filename} is empty")
            collected.append((filename, payload))
        try:
            return uploads.add_files(collected).public()
        except ImageValidationError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/uploads/{upload_id}/planes/{file_index}/{plane_index}.png")
    def upload_plane(
        upload_id: str,
        file_index: int,
        plane_index: int,
        max_size: int = Query(default=1024, ge=128, le=2048),
    ) -> Response:
        try:
            plane = uploads.plane(upload_id, file_index, plane_index)
        except KeyError as exc:
            raise HTTPException(404, "Upload session not found") from exc
        except (IndexError, ImageValidationError) as exc:
            raise HTTPException(404, str(exc)) from exc
        return Response(
            channel_preview_png_bytes(plane, max_size=max_size),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/inputs/assemble", status_code=201)
    def assemble_input(payload: AssembleInputRequest) -> dict:
        mapping = {
            role: (selection.model_dump() if hasattr(selection, "model_dump") else selection.dict())
            for role, selection in payload.mapping.items()
        }
        if set(mapping) != set(CHANNEL_KEYS):
            raise HTTPException(422, "Map exactly one plane to MT, nucleus, and ER")
        try:
            return uploads.assemble(
                upload_id=payload.upload_id,
                mapping=mapping,
                pixel_size_um=payload.pixel_size_um,
                resample=payload.resample,
                crop_x=payload.crop_x,
                crop_y=payload.crop_y,
                normalize=payload.normalize,
                normalization_bit_depth=payload.normalization_bit_depth,
            ).public()
        except KeyError as exc:
            raise HTTPException(404, "Upload session not found") from exc
        except (ValueError, IndexError, ImageValidationError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/inputs/example", status_code=201)
    def example_input() -> dict:
        example = PACKAGE_ROOT / "example_cell_reference_input" / "cell_0.tiff"
        if not example.is_file():
            raise HTTPException(404, "Bundled example image is unavailable")
        try:
            return inputs.add_path(example).public()
        except ImageValidationError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/inputs/{input_id}")
    def get_input(input_id: str) -> dict:
        try:
            return inputs.get(input_id).public()
        except KeyError as exc:
            raise HTTPException(404, "Input not found") from exc

    @app.get("/api/inputs/{input_id}/channels/{channel}.png")
    def input_channel(input_id: str, channel: str) -> Response:
        if channel not in CHANNEL_KEYS:
            raise HTTPException(404, "Unknown channel")
        try:
            array = inputs.array(input_id)
        except KeyError as exc:
            raise HTTPException(404, "Input not found") from exc
        return Response(channel_png_bytes(array[..., CHANNEL_KEYS.index(channel)]), media_type="image/png")

    @app.post("/api/jobs", status_code=202)
    def create_job(payload: GenerationRequest) -> dict:
        if payload.protein not in protein_set:
            raise HTTPException(422, "Protein is not in the ProtiCelli vocabulary")
        if payload.cell_line and payload.cell_line not in cell_lines:
            raise HTTPException(422, "Cell line is not in the ProtiCelli vocabulary")
        try:
            run = workspaces.get(payload.run_id)
            if run["input_id"] is None:
                run = workspaces.update(payload.run_id, input_id=payload.input_id)
            elif run["input_id"] != payload.input_id:
                raise HTTPException(409, "Prediction input does not match the active run")
            values = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
            workspaces.touch(payload.run_id)
            return jobs.submit(**values)
        except KeyError as exc:
            raise HTTPException(404, "Run or input not found") from exc

    @app.get("/api/jobs")
    def list_jobs(run_id: str | None = Query(default=None)) -> dict:
        return {"items": jobs.list(run_id)}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        try:
            return jobs.public(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Run not found") from exc

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str) -> dict:
        try:
            return jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Run not found") from exc

    @app.get("/api/jobs/{job_id}/images/{index}.png")
    def job_preview(job_id: str, index: int) -> FileResponse:
        try:
            path = jobs.result_file(job_id, index, "preview")
        except KeyError as exc:
            raise HTTPException(404, "Completed run not found") from exc
        except IndexError as exc:
            raise HTTPException(404, "Sample not found") from exc
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/jobs/{job_id}/images/{index}.tiff")
    def job_raw(job_id: str, index: int) -> FileResponse:
        try:
            path = jobs.result_file(job_id, index, "raw")
        except KeyError as exc:
            raise HTTPException(404, "Completed run not found") from exc
        except IndexError as exc:
            raise HTTPException(404, "Sample not found") from exc
        return FileResponse(path, media_type="image/tiff", filename=path.name)

    @app.get("/api/jobs/{job_id}/export")
    def export_job(job_id: str) -> FileResponse:
        try:
            path = jobs.archive(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Completed run not found") from exc
        return FileResponse(path, media_type="application/zip", filename=f"proticelli-{job_id[:8]}.zip")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
