from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.face_aesthetics.pipelines.inference import InferenceArtifacts, run_inference
from src.face_aesthetics.utils.config import load_named_configs


API_CONFIG_FILES = (
    "data.yaml",
    "preprocess.yaml",
    "train_baseline.yaml",
    "infer.yaml",
)


class ServiceValidationError(ValueError):
    pass


class ServiceDependencyError(RuntimeError):
    pass


def _load_api_configs(project_root: Path) -> dict[str, dict[str, Any]]:
    return load_named_configs(project_root / "configs", file_names=API_CONFIG_FILES)


def get_api_runtime_settings(*, project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or Path.cwd()).resolve()
    configs = _load_api_configs(root)
    infer_config = dict(configs.get("infer", {}).get("infer", {}))
    return {
        "project_root": root,
        "infer": infer_config,
        "api": dict(infer_config.get("api", {})),
        "upload": dict(infer_config.get("upload", {})),
    }


def build_preprocess_overrides(
    *,
    preferred_execution_provider: str | None,
    device_id: int | None,
    strict_gpu: bool | None,
) -> dict[str, Any]:
    preprocess_overrides = {
        "preferred_execution_provider": preferred_execution_provider,
        "device_id": device_id,
        "strict_gpu_mode": strict_gpu,
    }
    if (
        strict_gpu is None
        and preferred_execution_provider is not None
        and preferred_execution_provider.strip() == "CUDAExecutionProvider"
    ):
        preprocess_overrides["strict_gpu_mode"] = True
    return preprocess_overrides


def _sanitize_upload_filename(filename: str) -> str:
    candidate = Path(filename or "upload").name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._")
    return safe or "upload"


def persist_uploaded_image(
    *,
    file_name: str,
    content_type: str | None,
    file_bytes: bytes,
    project_root: Path | None = None,
) -> Path:
    runtime_settings = get_api_runtime_settings(project_root=project_root)
    root = Path(runtime_settings["project_root"])
    upload_config = runtime_settings["upload"]

    if upload_config.get("enabled", True) is False:
        raise ServiceDependencyError("Upload analysis is disabled by config.")

    safe_name = _sanitize_upload_filename(file_name)
    suffix = Path(safe_name).suffix.lower()
    allowed_extensions = {str(item).lower() for item in upload_config.get("allowed_extensions", [])}
    allowed_content_types = {str(item).lower() for item in upload_config.get("allowed_content_types", [])}
    max_file_size_mb = float(upload_config.get("max_file_size_mb", 15))

    if allowed_extensions and suffix not in allowed_extensions:
        raise ServiceValidationError(
            f"Unsupported file extension '{suffix or '<none>'}'. Allowed: {sorted(allowed_extensions)}"
        )

    normalized_content_type = (content_type or "").strip().lower()
    if normalized_content_type and allowed_content_types and normalized_content_type not in allowed_content_types:
        raise ServiceValidationError(
            "Unsupported file content type "
            f"'{normalized_content_type}'. Allowed: {sorted(allowed_content_types)}"
        )

    max_size_bytes = int(max_file_size_mb * 1024 * 1024)
    if len(file_bytes) == 0:
        raise ServiceValidationError("Uploaded file is empty.")
    if len(file_bytes) > max_size_bytes:
        raise ServiceValidationError(
            f"Uploaded file is too large: {len(file_bytes)} bytes. Limit: {max_size_bytes} bytes."
        )

    upload_dir = (root / str(upload_config.get("temp_dir", "data/interim/runtime_uploads"))).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(safe_name).stem or "upload"
    final_name = f"{stem}-{uuid.uuid4().hex[:12]}{suffix or '.bin'}"
    destination = upload_dir / final_name
    destination.write_bytes(file_bytes)
    return destination


def analyze_image(
    *,
    image_path: Path,
    output_dir: Path | None = None,
    group_override: str | None = None,
    project_root: Path | None = None,
    preprocess_overrides: dict[str, Any] | None = None,
) -> InferenceArtifacts:
    root = (project_root or Path.cwd()).resolve()
    resolved_image_path = image_path.resolve()
    if not resolved_image_path.exists():
        raise ServiceValidationError(f"Input image was not found: {resolved_image_path}")

    try:
        return run_inference(
            image_path=resolved_image_path,
            output_dir=output_dir,
            group_override=group_override,
            project_root=root,
            preprocess_overrides=preprocess_overrides,
        )
    except FileNotFoundError as exc:
        raise ServiceDependencyError(str(exc)) from exc
    except ValueError as exc:
        raise ServiceValidationError(str(exc)) from exc
    except RuntimeError as exc:
        raise ServiceDependencyError(str(exc)) from exc


def serialize_inference_artifacts(artifacts: InferenceArtifacts) -> dict[str, Any]:
    report = dict(artifacts.report)
    report["generated_artifacts"] = {
        "report_json_path": str(artifacts.report_json_path),
        "report_markdown_path": str(artifacts.report_markdown_path),
        "reporting_summary_path": str(artifacts.reporting_summary_path),
        "analysis_image_path": str(artifacts.analysis_image_path) if artifacts.analysis_image_path else None,
        "normalized_frame_image_path": (
            str(artifacts.normalized_frame_image_path) if artifacts.normalized_frame_image_path else None
        ),
        "aligned_image_path": str(artifacts.aligned_image_path) if artifacts.aligned_image_path else None,
        "measurement_image_path": str(artifacts.measurement_image_path) if artifacts.measurement_image_path else None,
        "overlay_image_path": str(artifacts.analysis_image_path) if artifacts.analysis_image_path else None,
        "annotated_image_path": str(artifacts.annotated_image_path) if artifacts.annotated_image_path else None,
    }
    return report


def _dependency_item(*, name: str, path: Path, required: bool, exists: bool, message: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path),
        "required": required,
        "available": exists,
        "message": message,
    }


def health_status(*, project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or Path.cwd()).resolve()
    configs = _load_api_configs(root)
    runtime_settings = get_api_runtime_settings(project_root=root)
    preprocess_config = configs["preprocess"]["preprocess"]
    infer_config = runtime_settings["infer"]
    upload_config = runtime_settings["upload"]

    insightface_config = dict(preprocess_config["insightface"])
    model_root = (root / str(insightface_config["model_root"])).resolve()
    model_name = str(insightface_config["model_name"])
    required_model_files = list(insightface_config.get("required_model_files", []))
    reference_model_path = (root / str(infer_config["reference_model_path"])).resolve()
    reference_statistics_path = (root / str(infer_config["reference_statistics_path"])).resolve()
    optional_three_d_template = (
        root / str(preprocess_config.get("three_d", {}).get("template_path", "data/interim/3d_models/canonical_face_5pt.npy"))
    ).resolve()

    dependencies: list[dict[str, Any]] = [
        _dependency_item(
            name="reference_model",
            path=reference_model_path,
            required=True,
            exists=reference_model_path.exists(),
        ),
        _dependency_item(
            name="reference_statistics",
            path=reference_statistics_path,
            required=True,
            exists=reference_statistics_path.exists(),
        ),
    ]
    for model_file in required_model_files:
        candidate = model_root / "models" / model_name / model_file
        dependencies.append(
            _dependency_item(
                name=f"insightface:{model_file}",
                path=candidate,
                required=True,
                exists=candidate.exists(),
            )
        )
    dependencies.append(
        _dependency_item(
            name="optional_3d_template",
            path=optional_three_d_template,
            required=False,
            exists=optional_three_d_template.exists(),
            message="Template-based 3D branch uses this asset when enabled.",
        )
    )
    upload_dir = (root / str(upload_config.get("temp_dir", "data/interim/runtime_uploads"))).resolve()
    dependencies.append(
        _dependency_item(
            name="upload_runtime_dir",
            path=upload_dir,
            required=False,
            exists=upload_dir.exists(),
            message="POST /analyze-upload stores incoming images here before analysis.",
        )
    )

    missing_required = [item for item in dependencies if item["required"] and not item["available"]]
    return {
        "status": "ok" if not missing_required else "degraded",
        "can_analyze": not missing_required,
        "project_root": str(root),
        "dependencies": dependencies,
    }


def list_reference_groups(*, project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or Path.cwd()).resolve()
    configs = _load_api_configs(root)
    infer_config = configs["infer"]["infer"]
    reference_model_path = (root / str(infer_config["reference_model_path"])).resolve()
    reference_statistics_path = (root / str(infer_config["reference_statistics_path"])).resolve()

    if not reference_model_path.exists():
        raise ServiceDependencyError(f"Reference model artifact is missing: {reference_model_path}")
    if not reference_statistics_path.exists():
        raise ServiceDependencyError(f"Reference statistics artifact is missing: {reference_statistics_path}")

    reference_payload = joblib.load(reference_model_path)
    statistics_frame = pd.read_parquet(reference_statistics_path)

    groups: list[dict[str, Any]] = []
    for combination_name, bucket in dict(reference_payload.get("reference_groups", {})).items():
        for group_key, payload in dict(bucket).items():
            stats_subset = statistics_frame.loc[statistics_frame["group_key"] == group_key]
            groups.append(
                {
                    "combination_name": combination_name,
                    "group_key": group_key,
                    "group_values": dict(payload.get("group_values", {})),
                    "sample_count": int(payload.get("sample_count", 0)),
                    "complete_case_count": int(payload.get("complete_case_count", 0)),
                    "feature_count": int(len(payload.get("feature_columns", []))),
                    "statistics_feature_rows": int(len(stats_subset)),
                }
            )
    groups.sort(key=lambda item: (item["combination_name"], item["group_key"]))
    return {
        "status": "ok",
        "reference_model_path": str(reference_model_path),
        "reference_statistics_path": str(reference_statistics_path),
        "groupings": reference_payload.get("groupings", []),
        "reference_group_count": len(groups),
        "groups": groups,
    }
