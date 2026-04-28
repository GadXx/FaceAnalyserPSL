from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.face_aesthetics.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    HealthResponse,
    ReferenceGroupsResponse,
)
from src.face_aesthetics.services.inference_service import (
    ServiceDependencyError,
    ServiceValidationError,
    analyze_image,
    build_preprocess_overrides,
    get_api_runtime_settings,
    health_status,
    list_reference_groups,
    persist_uploaded_image,
    serialize_inference_artifacts,
)


def _multipart_runtime_available() -> bool:
    return importlib.util.find_spec("multipart") is not None


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ServiceValidationError(f"Invalid boolean value: {value}")


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ServiceValidationError(f"Invalid integer value: {value}") from exc


def create_app(*, project_root: Path | None = None) -> FastAPI:
    resolved_project_root = (project_root or Path.cwd()).resolve()
    runtime_settings = get_api_runtime_settings(project_root=resolved_project_root)
    app = FastAPI(
        title="FaceAnalyserPSL Morphometric API",
        version="1.0.0",
        description="Objective morphometric face analysis API. No beauty or attractiveness scoring is produced.",
    )
    app.state.project_root = resolved_project_root
    cors_allowed_origins = list(runtime_settings["api"].get("cors_allowed_origins", []))
    if cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _resolve_project_file(path_value: str) -> Path:
        if not path_value or not path_value.strip():
            raise ServiceValidationError("Query parameter 'path' is required.")
        candidate = Path(path_value.strip())
        resolved = candidate.resolve() if candidate.is_absolute() else (app.state.project_root / candidate).resolve()
        project_root_resolved = app.state.project_root.resolve()
        if not resolved.is_relative_to(project_root_resolved):
            raise ServiceValidationError(
                f"Requested file is outside the project root and cannot be served: {resolved}"
            )
        if not resolved.exists() or not resolved.is_file():
            raise ServiceValidationError(f"Requested file does not exist: {resolved}")
        return resolved

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={500: {"model": ErrorResponse}},
    )
    def get_health() -> dict:
        return health_status(project_root=app.state.project_root)

    @app.get(
        "/reference-groups",
        response_model=ReferenceGroupsResponse,
        responses={424: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    def get_reference_groups() -> dict:
        try:
            return list_reference_groups(project_root=app.state.project_root)
        except ServiceDependencyError as exc:
            raise HTTPException(status_code=424, detail={"detail": str(exc), "error_type": "dependency_error"}) from exc
        except Exception as exc:  # pragma: no cover - unexpected failures should surface clearly
            raise HTTPException(status_code=500, detail={"detail": str(exc), "error_type": "internal_error"}) from exc

    @app.get(
        "/artifact-file",
        responses={422: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    )
    def get_artifact_file(path: str = Query(..., description="Project-relative or absolute artifact path.")) -> FileResponse:
        try:
            resolved = _resolve_project_file(path)
            return FileResponse(resolved)
        except ServiceValidationError as exc:
            detail = str(exc)
            status_code = 404 if "does not exist" in detail else 422
            raise HTTPException(status_code=status_code, detail={"detail": detail, "error_type": "validation_error"}) from exc

    @app.post(
        "/analyze",
        response_model=AnalyzeResponse,
        responses={
            422: {"model": ErrorResponse},
            424: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def post_analyze(request: AnalyzeRequest) -> dict:
        preprocess_overrides = build_preprocess_overrides(
            preferred_execution_provider=request.preferred_execution_provider,
            device_id=request.device_id,
            strict_gpu=request.strict_gpu,
        )
        try:
            artifacts = analyze_image(
                image_path=Path(request.image_path),
                output_dir=Path(request.output_dir) if request.output_dir else None,
                group_override=request.group,
                project_root=app.state.project_root,
                preprocess_overrides=preprocess_overrides,
            )
            return serialize_inference_artifacts(artifacts)
        except ServiceValidationError as exc:
            raise HTTPException(status_code=422, detail={"detail": str(exc), "error_type": "validation_error"}) from exc
        except ServiceDependencyError as exc:
            raise HTTPException(status_code=424, detail={"detail": str(exc), "error_type": "dependency_error"}) from exc
        except Exception as exc:  # pragma: no cover - unexpected failures should surface clearly
            raise HTTPException(status_code=500, detail={"detail": str(exc), "error_type": "internal_error"}) from exc

    @app.post(
        "/analyze-upload",
        response_model=AnalyzeResponse,
        responses={
            422: {"model": ErrorResponse},
            424: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def post_analyze_upload(request: Request) -> dict:
        if not _multipart_runtime_available():
            raise HTTPException(
                status_code=424,
                detail={
                    "detail": "python-multipart is required for file upload support. Install dependencies from requirements.txt.",
                    "error_type": "dependency_error",
                },
            )

        try:
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                raise ServiceValidationError("Multipart form field 'file' is required.")
            if not hasattr(upload, "filename"):
                raise ServiceValidationError("Multipart form field 'file' must be a file upload.")

            file_bytes = await upload.read()
            stored_image_path = persist_uploaded_image(
                file_name=str(getattr(upload, "filename", "upload")),
                content_type=getattr(upload, "content_type", None),
                file_bytes=file_bytes,
                project_root=app.state.project_root,
            )
            preprocess_overrides = build_preprocess_overrides(
                preferred_execution_provider=str(form.get("preferred_execution_provider")).strip()
                if form.get("preferred_execution_provider")
                else None,
                device_id=_parse_optional_int(str(form.get("device_id"))) if form.get("device_id") is not None else None,
                strict_gpu=_parse_optional_bool(str(form.get("strict_gpu"))) if form.get("strict_gpu") is not None else None,
            )
            artifacts = analyze_image(
                image_path=stored_image_path,
                output_dir=Path(str(form.get("output_dir"))) if form.get("output_dir") else None,
                group_override=str(form.get("group")).strip() if form.get("group") else None,
                project_root=app.state.project_root,
                preprocess_overrides=preprocess_overrides,
            )
            serialized = serialize_inference_artifacts(artifacts)
            serialized.setdefault("input", {})
            serialized["input"]["upload_original_filename"] = getattr(upload, "filename", None)
            serialized["input"]["upload_content_type"] = getattr(upload, "content_type", None)
            return serialized
        except ServiceValidationError as exc:
            raise HTTPException(status_code=422, detail={"detail": str(exc), "error_type": "validation_error"}) from exc
        except ServiceDependencyError as exc:
            raise HTTPException(status_code=424, detail={"detail": str(exc), "error_type": "dependency_error"}) from exc
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - unexpected failures should surface clearly
            raise HTTPException(status_code=500, detail={"detail": str(exc), "error_type": "internal_error"}) from exc

    return app


app = create_app()
