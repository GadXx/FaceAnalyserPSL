from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.face_aesthetics.data.io import (
    iter_parquet_batches,
    load_image,
    read_parquet,
    save_image,
    write_json,
    write_parquet_shard,
)
from src.face_aesthetics.services.preprocessing import (
    AutoAlignmentOutcome,
    DetectedFace,
    MediaPipeDetector,
    compute_blur_variance,
    select_detection_with_auto_alignment,
)
from src.face_aesthetics.landmarks.mediapipe_source import get_mediapipe_runtime_startup_diagnostics
from src.face_aesthetics.services.three_d import (
    OptionalFace3DProcessor,
    ensure_default_canonical_face_template,
    save_3d_artifact,
)
from src.face_aesthetics.utils.config import load_named_configs
from src.face_aesthetics.utils.filesystem import ensure_directory, to_relative_path


PREPROCESS_COLUMNS = [
    "dataset_name",
    "sample_id",
    "subject_id",
    "image_path",
    "split",
    "raw_labels",
    "normalized_labels",
    "quality_gate_status",
    "adapter_version",
    "image_id",
    "reference_model",
    "reference_deviation",
    "reference_deviation_std",
    "processing_status",
    "processing_error_message",
    "face_detected",
    "num_faces",
    "bbox",
    "landmarks",
    "landmarks_dense",
    "det_score",
    "detection_backend",
    "yaw",
    "pitch",
    "roll",
    "original_yaw",
    "original_pitch",
    "original_roll",
    "blur_variance",
    "occlusion_score",
    "alignment_attempted",
    "alignment_applied",
    "alignment_improved_quality",
    "alignment_source",
    "alignment_status",
    "alignment_rotation_deg",
    "alignment_transform",
    "original_quality_pass",
    "original_quality_fail_reason",
    "aligned_quality_pass",
    "aligned_quality_fail_reason",
    "quality_flags",
    "quality_pass",
    "quality_fail_reason",
    "aligned_image_path",
]

THREE_D_MANIFEST_COLUMNS = [
    "image_id",
    "artifact_path",
    "status",
    "reason",
    "artifact_version",
    "template_path",
]

STATUS_PROCESSED = "processed"
STATUS_SUCCESS = "success"
STATUS_IMAGE_READ_ERROR = "image_read_error"
STATUS_DETECTION_ERROR = "detection_error"
STATUS_ALIGNED_SAVE_ERROR = "aligned_save_error"
STATUS_MISSING_IMAGE_PATH = "missing_image_path"
PROCESSED_STATUSES = {STATUS_PROCESSED, STATUS_SUCCESS}


@dataclass
class PreprocessingArtifacts:
    output_manifest_path: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass
class SingleImagePreprocessingArtifacts:
    row: dict[str, Any]
    aligned_image_path: Path | None
    analysis_image_path: Path | None
    measurement_image_path: Path | None
    measurement_image_role: str
    provider_diagnostics: dict[str, Any]


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return not math.isnan(value)
    if isinstance(value, str):
        stripped = value.strip()
        return bool(stripped and stripped.lower() not in {"null", "nan"})
    return True


def _bbox_size(bbox: list[float] | None) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = bbox
    return float(min(max(x2 - x1, 0.0), max(y2 - y1, 0.0)))


def _compute_blur_variance(detection: DetectedFace | None) -> float | None:
    if detection is None or detection.aligned_image is None:
        return None
    return compute_blur_variance(detection.aligned_image)


def _evaluate_quality(
    detection: DetectedFace | None,
    num_faces: int,
    quality_gate: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []

    if num_faces == 0:
        return ["no_face"]
    if quality_gate.get("require_single_face", True) and num_faces > 1:
        return ["multiple_faces"]
    if detection is None:
        return ["no_face"]
    if detection.aligned_image is None:
        reasons.append("alignment_unavailable")
    if _bbox_size(detection.bbox) < float(quality_gate["min_bbox_size"]):
        reasons.append("bbox_too_small")
    if detection.det_score is not None and detection.det_score < float(quality_gate["min_detector_score"]):
        reasons.append("detector_score_too_low")

    blur_variance = _compute_blur_variance(detection)
    if blur_variance is not None and blur_variance < float(quality_gate["min_blur_variance"]):
        reasons.append("blur_too_strong")

    pose_limits = (
        ("yaw_extreme", detection.yaw, float(quality_gate["max_abs_yaw"])),
        ("pitch_extreme", detection.pitch, float(quality_gate["max_abs_pitch"])),
        ("roll_extreme", detection.roll, float(quality_gate["max_abs_roll"])),
    )
    for reason, value, limit in pose_limits:
        if value is not None and abs(value) > limit:
            reasons.append(reason)

    occlusion_limit = quality_gate.get("max_occlusion_score")
    if occlusion_limit is not None and detection.occlusion_score is not None:
        if detection.occlusion_score > float(occlusion_limit):
            reasons.append("severe_occlusion")

    return reasons


def _quality_gate_status(processing_status: str, quality_reasons: list[str]) -> str:
    if processing_status not in PROCESSED_STATUSES:
        return "preprocessing_error"
    if quality_reasons:
        return "failed"
    return "passed"


def _normalize_manifest_row(input_row: dict[str, Any], default_reference_model: str | None) -> dict[str, Any]:
    row = dict(input_row)
    row["reference_model"] = row.get("reference_model") or default_reference_model
    row["reference_deviation"] = row.get("reference_deviation")
    row["reference_deviation_std"] = row.get("reference_deviation_std")
    row["quality_gate_status"] = row.get("quality_gate_status") or "pending_preprocessing"
    return row


def _build_output_row(
    input_row: dict[str, Any],
    detection: DetectedFace | None,
    num_faces: int,
    quality_reasons: list[str],
    alignment_outcome: AutoAlignmentOutcome | None,
    aligned_image_path: str | None,
    processing_status: str,
    processing_error_message: str | None = None,
) -> dict[str, Any]:
    blur_variance = _compute_blur_variance(detection)
    quality_pass = processing_status in PROCESSED_STATUSES and not quality_reasons

    row = dict(input_row)
    row.update(
        {
            "quality_gate_status": _quality_gate_status(processing_status, quality_reasons),
            "processing_status": processing_status,
            "processing_error_message": processing_error_message,
            "face_detected": num_faces > 0,
            "num_faces": num_faces,
            "bbox": _json_dumps(detection.bbox if detection is not None else None),
            "landmarks": _json_dumps(detection.landmarks if detection is not None else None),
            "landmarks_dense": _json_dumps(detection.landmarks_dense if detection is not None else None),
            "det_score": detection.det_score if detection is not None else None,
            "detection_backend": detection.detection_backend if detection is not None else None,
            "yaw": detection.yaw if detection is not None else None,
            "pitch": detection.pitch if detection is not None else None,
            "roll": detection.roll if detection is not None else None,
            "original_yaw": (alignment_outcome.original_pose.get("yaw") if alignment_outcome is not None else None),
            "original_pitch": (alignment_outcome.original_pose.get("pitch") if alignment_outcome is not None else None),
            "original_roll": (alignment_outcome.original_pose.get("roll") if alignment_outcome is not None else None),
            "blur_variance": blur_variance,
            "occlusion_score": detection.occlusion_score if detection is not None else None,
            "alignment_attempted": (alignment_outcome.attempted if alignment_outcome is not None else False),
            "alignment_applied": (alignment_outcome.applied if alignment_outcome is not None else False),
            "alignment_improved_quality": (alignment_outcome.improved_quality if alignment_outcome is not None else False),
            "alignment_source": (alignment_outcome.source if alignment_outcome is not None else None),
            "alignment_status": (alignment_outcome.status if alignment_outcome is not None else None),
            "alignment_rotation_deg": (alignment_outcome.rotation_deg if alignment_outcome is not None else None),
            "alignment_transform": _json_dumps(alignment_outcome.transform if alignment_outcome is not None else None),
            "original_quality_pass": (alignment_outcome.original_quality_pass if alignment_outcome is not None else None),
            "original_quality_fail_reason": (
                alignment_outcome.original_quality_fail_reason if alignment_outcome is not None else None
            ),
            "aligned_quality_pass": (alignment_outcome.aligned_quality_pass if alignment_outcome is not None else None),
            "aligned_quality_fail_reason": (
                alignment_outcome.aligned_quality_fail_reason if alignment_outcome is not None else None
            ),
            "quality_flags": _json_dumps(quality_reasons),
            "quality_pass": quality_pass,
            "quality_fail_reason": "|".join(quality_reasons) if quality_reasons else None,
            "aligned_image_path": aligned_image_path,
        }
    )
    return row


def _build_detector(preprocess_config: dict[str, Any], runtime_config: dict[str, Any], root: Path) -> MediaPipeDetector:
    landmarks_config = dict(preprocess_config.get("landmarks", {}))
    mediapipe_config = dict(landmarks_config.get("mediapipe_face_mesh", {}))
    if not mediapipe_config:
        mediapipe_config = {
            "enabled": True,
            "optional": False,
            "model": "face_landmarker",
            "max_num_faces": 1,
            "refine_landmarks": True,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "model_asset_path": "data/interim/mediapipe/face_landmarker.task",
            "delegate": runtime_config.get("mediapipe_delegate", "auto"),
            "fallback_to_cpu": True,
        }
    if "delegate" not in mediapipe_config and runtime_config.get("mediapipe_delegate") is not None:
        mediapipe_config["delegate"] = runtime_config["mediapipe_delegate"]
    return MediaPipeDetector(
        mediapipe_config=mediapipe_config,
        project_root=root,
        target_image_size=list(preprocess_config.get("target_image_size", [224, 224])),
    )


def _runtime_config(preprocess_config: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    runtime = dict(preprocess_config.get("runtime", {}))
    if overrides:
        runtime.update({key: value for key, value in overrides.items() if value is not None})

    if runtime.get("smoke_mode") and runtime.get("limit") is None:
        runtime["limit"] = int(runtime.get("smoke_limit", 16))

    runtime.setdefault("batch_size", 64)
    runtime.setdefault("offset", 0)
    runtime.setdefault("resume", True)
    runtime.setdefault("overwrite", False)
    runtime.setdefault("num_workers", 1)
    runtime.setdefault("preferred_execution_provider", "CUDAExecutionProvider")
    runtime.setdefault("fallback_execution_provider", "CPUExecutionProvider")
    runtime.setdefault("device_id", 0)
    runtime.setdefault("strict_gpu_mode", False)
    runtime.setdefault("smoke_mode", False)
    runtime.setdefault("smoke_limit", 16)
    return runtime


def _resolve_runtime_check_model_path(preprocess_config: dict[str, Any], root: Path) -> Path:
    landmarks_config = dict(preprocess_config.get("landmarks", {}))
    mediapipe_config = dict(landmarks_config.get("mediapipe_face_mesh", {}))
    model_asset_path = str(mediapipe_config.get("model_asset_path", "data/interim/mediapipe/face_landmarker.task"))
    path = Path(model_asset_path)
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _runtime_check_output_path(preprocess_config: dict[str, Any], root: Path) -> Path:
    gpu_runtime_check_config = dict(preprocess_config.get("gpu_runtime_check", {}))
    output_path = str(gpu_runtime_check_config.get("output_path", "reports/metrics/gpu_runtime_check.json"))
    return (root / output_path).resolve()


def _ensure_strict_gpu_runtime_or_raise(runtime_check: dict[str, Any], runtime_config: dict[str, Any]) -> None:
    strict = bool(runtime_config.get("strict_gpu_mode", False))
    preferred = str(runtime_config.get("preferred_execution_provider", "")).strip()
    if not strict or preferred != "CUDAExecutionProvider":
        return

    missing_libraries = list(runtime_check.get("missing_cuda_runtime_libraries", []))
    gpu_active = bool(runtime_check.get("gpu_provider_active", False))
    error = runtime_check.get("error")
    if missing_libraries or not gpu_active or error:
        raise RuntimeError(
            "Strict GPU mode is enabled and CUDA runtime is not ready. "
            f"gpu_provider_active={gpu_active}, "
            f"missing_cuda_runtime_libraries={missing_libraries}, "
            f"runtime_error={error}. "
            "Install missing CUDA runtime dependencies and ensure ONNX Runtime can initialize CUDAExecutionProvider."
        )


def run_preprocessing_gpu_runtime_check(
    project_root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = (project_root or Path.cwd()).resolve()
    configs = load_named_configs(root / "configs")
    preprocess_config = configs["preprocess"]["preprocess"]
    runtime_config = _runtime_config(preprocess_config, overrides)
    auto_alignment_config = dict(preprocess_config.get("auto_alignment", {}))
    runtime_check_path = _runtime_check_output_path(preprocess_config, root)
    test_model_path = _resolve_runtime_check_model_path(preprocess_config, root)

    landmarks_config = dict(preprocess_config.get("landmarks", {}))
    mediapipe_config = dict(landmarks_config.get("mediapipe_face_mesh", {}))
    payload = get_mediapipe_runtime_startup_diagnostics(
        config=mediapipe_config,
        project_root=root,
    )
    payload["test_model_path"] = str(test_model_path)
    payload["runtime"] = runtime_config
    write_json(payload, runtime_check_path)
    _ensure_strict_gpu_runtime_or_raise(payload, runtime_config)
    return payload


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_parquet(path)


def _prepare_output_manifest_path(
    output_manifest_path: Path,
    *,
    overwrite: bool,
    resume: bool,
) -> list[dict[str, Any]]:
    existing_rows: list[dict[str, Any]] = []
    if not output_manifest_path.exists():
        ensure_directory(output_manifest_path)
        return existing_rows

    if output_manifest_path.is_dir():
        if overwrite:
            shutil.rmtree(output_manifest_path)
            ensure_directory(output_manifest_path)
            return []
        return _load_existing_rows(output_manifest_path)

    existing_rows = _load_existing_rows(output_manifest_path)
    if overwrite:
        output_manifest_path.unlink()
        ensure_directory(output_manifest_path)
        return []

    if not resume:
        raise FileExistsError(
            f"Output path already exists: {output_manifest_path}. "
            "Use overwrite=true to replace it or resume=true to continue from existing results."
        )

    output_manifest_path.unlink()
    ensure_directory(output_manifest_path)
    if existing_rows:
        write_parquet_shard(existing_rows, output_manifest_path, PREPROCESS_COLUMNS, "part-legacy-000000.parquet")
    return existing_rows


def _successful_image_ids(rows: list[dict[str, Any]]) -> set[str]:
    processed: set[str] = set()
    for row in rows:
        if str(row.get("processing_status")).strip().lower() in PROCESSED_STATUSES and row.get("image_id"):
            processed.add(str(row["image_id"]))
    return processed


def _build_existing_counters(
    rows: list[dict[str, Any]]
) -> tuple[Counter[str], Counter[str], list[str], int, int, int, int, int]:
    rejection_counter: Counter[str] = Counter()
    processing_status_counter: Counter[str] = Counter()
    aligned_examples: list[str] = []
    aligned_nonempty_count = 0
    landmarks_nonempty_count = 0
    alignment_attempted_count = 0
    alignment_applied_count = 0
    alignment_improved_quality_count = 0

    for row in rows:
        processing_status = row.get("processing_status")
        if _is_nonempty(processing_status):
            processing_status_counter[str(processing_status)] += 1
        quality_fail_reason = row.get("quality_fail_reason")
        reasons = str(quality_fail_reason).strip() if _is_nonempty(quality_fail_reason) else ""
        if reasons:
            for reason in reasons.split("|"):
                if reason:
                    rejection_counter[reason] += 1
        aligned_path = row.get("aligned_image_path")
        if _is_nonempty(aligned_path):
            aligned_nonempty_count += 1
            if len(aligned_examples) < 5:
                aligned_examples.append(str(aligned_path))
        if _is_nonempty(row.get("landmarks")):
            landmarks_nonempty_count += 1
        if bool(row.get("alignment_attempted")):
            alignment_attempted_count += 1
        if bool(row.get("alignment_applied")):
            alignment_applied_count += 1
        if bool(row.get("alignment_improved_quality")):
            alignment_improved_quality_count += 1

    return (
        rejection_counter,
        processing_status_counter,
        aligned_examples,
        aligned_nonempty_count,
        landmarks_nonempty_count,
        alignment_attempted_count,
        alignment_applied_count,
        alignment_improved_quality_count,
    )


def _next_shard_index(output_manifest_path: Path) -> int:
    if not output_manifest_path.exists() or not output_manifest_path.is_dir():
        return 0
    shard_paths = sorted(output_manifest_path.glob("part-*.parquet"))
    return len(shard_paths)


def _summary_payload(
    *,
    root: Path,
    input_manifest_path: Path,
    output_manifest_path: Path,
    aligned_dir: Path,
    summary_path: Path,
    progress_path: Path,
    provider_diagnostics: dict[str, Any] | None,
    runtime_config: dict[str, Any],
    input_total_rows: int,
    skipped_existing_rows: int,
    processed_rows: int,
    successful_rows: int,
    rejected_rows: int,
    error_rows: int,
    quality_pass_true_count: int,
    aligned_nonempty_count: int,
    landmarks_nonempty_count: int,
    alignment_attempted_count: int,
    alignment_applied_count: int,
    alignment_improved_quality_count: int,
    rejection_counter: Counter[str],
    processing_status_counter: Counter[str],
    aligned_examples: list[str],
    shards_written: int,
) -> dict[str, Any]:
    return {
        "input_manifest_path": input_manifest_path.relative_to(root).as_posix(),
        "output_manifest_path": output_manifest_path.relative_to(root).as_posix(),
        "aligned_dir": aligned_dir.relative_to(root).as_posix(),
        "summary_path": summary_path.relative_to(root).as_posix(),
        "progress_path": progress_path.relative_to(root).as_posix(),
        "input_total_rows": input_total_rows,
        "total_rows": processed_rows + error_rows,
        "processed_rows": processed_rows,
        "successful_rows": successful_rows,
        "rejected_rows": rejected_rows,
        "error_rows": error_rows,
        "quality_pass_true_count": quality_pass_true_count,
        "aligned_nonempty_count": aligned_nonempty_count,
        "landmarks_nonempty_count": landmarks_nonempty_count,
        "alignment_attempted_count": alignment_attempted_count,
        "alignment_applied_count": alignment_applied_count,
        "alignment_improved_quality_count": alignment_improved_quality_count,
        "skipped_existing_rows": skipped_existing_rows,
        "rejection_counts": dict(sorted(rejection_counter.items())),
        "processing_status_counts": dict(sorted(processing_status_counter.items())),
        "aligned_examples": aligned_examples[:5],
        "shards_written": shards_written,
        "runtime": runtime_config,
        "provider_diagnostics": provider_diagnostics,
    }


def _three_d_summary_payload(
    *,
    root: Path,
    artifacts_dir: Path,
    manifest_path: Path,
    summary_path: Path,
    branch_payload: dict[str, Any],
    processed_rows: int,
    success_rows: int,
    skipped_rows: int,
) -> dict[str, Any]:
    return {
        "artifacts_dir": artifacts_dir.relative_to(root).as_posix(),
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "summary_path": summary_path.relative_to(root).as_posix(),
        "branch": branch_payload,
        "processed_rows": processed_rows,
        "success_rows": success_rows,
        "skipped_rows": skipped_rows,
    }


def _process_single_image_load(row: dict[str, Any], root: Path) -> tuple[dict[str, Any], Any | None, str | None]:
    image_path_value = row.get("image_path")
    if not image_path_value:
        return row, None, "Input manifest row does not contain a valid image_path."
    image_path = (root / str(image_path_value)).resolve()
    try:
        return row, load_image(image_path), None
    except Exception as exc:
        return row, None, str(exc)


def preprocess_single_image(
    *,
    image_path: Path,
    project_root: Path | None = None,
    output_aligned_path: Path | None = None,
    output_analysis_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> SingleImagePreprocessingArtifacts:
    root = (project_root or Path.cwd()).resolve()
    configs = load_named_configs(root / "configs")
    preprocess_config = configs["preprocess"]["preprocess"]
    runtime_config = _runtime_config(preprocess_config, overrides)
    quality_gate = preprocess_config["quality_gate"]
    auto_alignment_config = dict(preprocess_config.get("auto_alignment", {}))
    target_image_size = list(preprocess_config["target_image_size"])
    default_reference_model = configs["data"]["dataset"].get("default_reference_model")
    test_model_path = _resolve_runtime_check_model_path(preprocess_config, root)

    landmarks_config = dict(preprocess_config.get("landmarks", {}))
    mediapipe_config = dict(landmarks_config.get("mediapipe_face_mesh", {}))
    gpu_runtime = get_mediapipe_runtime_startup_diagnostics(
        config=mediapipe_config,
        project_root=root,
    )
    gpu_runtime["test_model_path"] = str(test_model_path)
    gpu_runtime["runtime"] = runtime_config
    _ensure_strict_gpu_runtime_or_raise(gpu_runtime, runtime_config)

    metadata_payload = dict(metadata or {})
    resolved_image_path = image_path.resolve()
    input_row = _normalize_manifest_row(
        {
            "dataset_name": "single_inference",
            "sample_id": resolved_image_path.stem,
            "subject_id": metadata_payload.get("subject_id"),
            "image_path": resolved_image_path.relative_to(root).as_posix()
            if resolved_image_path.is_relative_to(root)
            else str(resolved_image_path),
            "split": "inference",
            "raw_labels": _json_dumps(metadata_payload.get("raw_labels")),
            "normalized_labels": _json_dumps(metadata_payload.get("normalized_labels")),
            "quality_gate_status": "pending_preprocessing",
            "adapter_version": "single_inference@1",
            "image_id": f"single_inference::{resolved_image_path.stem}",
            "reference_model": metadata_payload.get("reference_model"),
        },
        default_reference_model,
    )

    detector = _build_detector(preprocess_config, runtime_config, root)
    provider_diagnostics = {
        **gpu_runtime,
        "detector_provider_diagnostics": detector.diagnostics_payload(),
    }

    try:
        image = load_image(resolved_image_path)
    except Exception as exc:
        row = _build_output_row(
            input_row=input_row,
            detection=None,
            num_faces=0,
            quality_reasons=[],
            alignment_outcome=None,
            aligned_image_path=None,
            processing_status=STATUS_IMAGE_READ_ERROR,
            processing_error_message=str(exc),
        )
        return SingleImagePreprocessingArtifacts(
            row=row,
            aligned_image_path=None,
            analysis_image_path=None,
            measurement_image_path=None,
            measurement_image_role="analysis_image",
            provider_diagnostics=provider_diagnostics,
        )

    try:
        detections, primary_detection, quality_reasons, alignment_outcome, analysis_image = select_detection_with_auto_alignment(
            detector=detector,
            image_bgr=image,
            target_image_size=target_image_size,
            quality_gate=quality_gate,
            auto_alignment_config=auto_alignment_config,
            quality_evaluator=_evaluate_quality,
        )
    except Exception as exc:
        row = _build_output_row(
            input_row=input_row,
            detection=None,
            num_faces=0,
            quality_reasons=[],
            alignment_outcome=None,
            aligned_image_path=None,
            processing_status=STATUS_DETECTION_ERROR,
            processing_error_message=str(exc),
        )
        return SingleImagePreprocessingArtifacts(
            row=row,
            aligned_image_path=None,
            analysis_image_path=None,
            measurement_image_path=None,
            measurement_image_role="analysis_image",
            provider_diagnostics=provider_diagnostics,
        )

    num_faces = len(detections)
    processing_status = STATUS_PROCESSED
    processing_error_message: str | None = None
    aligned_image_path_value: str | None = None
    analysis_image_path_value: str | None = None
    aligned_output_path: Path | None = None
    analysis_output_path: Path | None = None
    measurement_output_path: Path | None = None
    measurement_image_role = "analysis_image"
    measurement_detection = primary_detection
    measurement_quality_reasons = quality_reasons
    measurement_num_faces = num_faces

    if output_analysis_path is not None:
        analysis_output_path = output_analysis_path.resolve()
        try:
            save_image(analysis_output_path, analysis_image)
            analysis_image_path_value = (
                analysis_output_path.relative_to(root).as_posix()
                if analysis_output_path.is_relative_to(root)
                else str(analysis_output_path)
            )
        except Exception:
            analysis_output_path = None

    if primary_detection is not None and primary_detection.aligned_image is not None and output_aligned_path is not None:
        aligned_output_path = output_aligned_path.resolve()
        try:
            save_image(aligned_output_path, primary_detection.aligned_image)
            aligned_image_path_value = (
                aligned_output_path.relative_to(root).as_posix()
                if aligned_output_path.is_relative_to(root)
                else str(aligned_output_path)
            )
        except Exception as exc:
            processing_status = STATUS_ALIGNED_SAVE_ERROR
            processing_error_message = str(exc)
            aligned_output_path = None

    if primary_detection is not None and primary_detection.aligned_image is not None:
        try:
            aligned_measurement_detections = detector.detect(
                image_bgr=primary_detection.aligned_image,
                target_image_size=target_image_size,
            )
        except Exception:
            aligned_measurement_detections = []

        aligned_measurement_num_faces = len(aligned_measurement_detections)
        aligned_measurement_primary = (
            aligned_measurement_detections[0] if aligned_measurement_num_faces == 1 else None
        )
        aligned_measurement_quality = _evaluate_quality(
            aligned_measurement_primary,
            aligned_measurement_num_faces,
            quality_gate,
        )

        if aligned_measurement_primary is not None and aligned_measurement_num_faces == 1:
            measurement_detection = aligned_measurement_primary
            measurement_quality_reasons = aligned_measurement_quality
            measurement_num_faces = aligned_measurement_num_faces
            measurement_output_path = aligned_output_path
            measurement_image_role = "aligned_face"
        else:
            measurement_output_path = analysis_output_path
    else:
        measurement_output_path = analysis_output_path

    row = _build_output_row(
        input_row=input_row,
        detection=measurement_detection,
        num_faces=measurement_num_faces,
        quality_reasons=measurement_quality_reasons,
        alignment_outcome=alignment_outcome,
        aligned_image_path=aligned_image_path_value,
        processing_status=processing_status,
        processing_error_message=processing_error_message,
    )
    return SingleImagePreprocessingArtifacts(
        row=row,
        aligned_image_path=aligned_output_path,
        analysis_image_path=analysis_output_path,
        measurement_image_path=measurement_output_path,
        measurement_image_role=measurement_image_role,
        provider_diagnostics=provider_diagnostics,
    )


def _load_batch_images(batch_rows: list[dict[str, Any]], root: Path, num_workers: int) -> list[tuple[dict[str, Any], Any | None, str | None]]:
    if num_workers <= 1:
        return [_process_single_image_load(row, root) for row in batch_rows]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(lambda item: _process_single_image_load(item, root), batch_rows))


def run_preprocessing(
    project_root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> PreprocessingArtifacts:
    root = (project_root or Path.cwd()).resolve()
    configs = load_named_configs(root / "configs")
    preprocess_config = configs["preprocess"]["preprocess"]
    runtime_config = _runtime_config(preprocess_config, overrides)

    input_manifest_path = (root / preprocess_config["input_manifest_path"]).resolve()
    aligned_dir = (root / preprocess_config["aligned_dir"]).resolve()
    three_d_artifacts_dir = (root / preprocess_config.get("three_d_artifacts_dir", "data/processed/3d_artifacts")).resolve()
    output_manifest_path = (root / preprocess_config["output_manifest_path"]).resolve()
    summary_path = (root / preprocess_config["summary_path"]).resolve()
    progress_path = (root / preprocess_config["progress_path"]).resolve()
    three_d_summary_path = (root / preprocess_config.get("three_d_summary_path", "reports/metrics/preprocessing_3d_summary.json")).resolve()
    gpu_runtime_check_path = _runtime_check_output_path(preprocess_config, root)
    quality_gate = preprocess_config["quality_gate"]
    auto_alignment_config = dict(preprocess_config.get("auto_alignment", {}))
    three_d_config = dict(preprocess_config.get("three_d", {}))
    three_d_manifest_path = (root / three_d_config.get("artifacts_manifest_path", "data/processed/3d_artifacts/manifest.parquet")).resolve()
    target_image_size = list(preprocess_config["target_image_size"])
    default_reference_model = configs["data"]["dataset"].get("default_reference_model")

    if bool(runtime_config["overwrite"]):
        if three_d_artifacts_dir.exists():
            shutil.rmtree(three_d_artifacts_dir)
        if three_d_manifest_path.exists():
            shutil.rmtree(three_d_manifest_path)
    ensure_directory(aligned_dir)
    ensure_directory(three_d_artifacts_dir)
    ensure_directory(three_d_manifest_path)
    test_model_path = _resolve_runtime_check_model_path(preprocess_config, root)
    landmarks_config = dict(preprocess_config.get("landmarks", {}))
    mediapipe_config = dict(landmarks_config.get("mediapipe_face_mesh", {}))
    gpu_runtime = get_mediapipe_runtime_startup_diagnostics(
        config=mediapipe_config,
        project_root=root,
    )
    gpu_runtime["test_model_path"] = str(test_model_path)
    gpu_runtime["runtime"] = runtime_config
    write_json(gpu_runtime, gpu_runtime_check_path)
    _ensure_strict_gpu_runtime_or_raise(gpu_runtime, runtime_config)
    existing_rows = _prepare_output_manifest_path(
        output_manifest_path,
        overwrite=bool(runtime_config["overwrite"]),
        resume=bool(runtime_config["resume"]),
    )
    existing_processed_ids = _successful_image_ids(existing_rows) if runtime_config["resume"] else set()
    (
        rejection_counter,
        processing_status_counter,
        aligned_examples,
        aligned_nonempty_count,
        landmarks_nonempty_count,
        alignment_attempted_count,
        alignment_applied_count,
        alignment_improved_quality_count,
    ) = _build_existing_counters(existing_rows)

    processed_rows_count = sum(
        1 for row in existing_rows if str(row.get("processing_status")).strip().lower() in PROCESSED_STATUSES
    )
    successful_rows_count = sum(1 for row in existing_rows if bool(row.get("quality_pass")))
    error_rows_count = sum(
        1 for row in existing_rows if str(row.get("processing_status")).strip().lower() not in PROCESSED_STATUSES
    )
    shards_written = _next_shard_index(output_manifest_path)
    resolved_three_d_template_path = ensure_default_canonical_face_template(
        template_path=(root / str(three_d_config.get("template_path", "data/interim/3d_models/canonical_face_5pt.npy"))).resolve(),
        auto_create=bool(three_d_config.get("auto_create_default_template", True)),
    )
    three_d_processor = OptionalFace3DProcessor(
        enabled=bool(three_d_config.get("enabled", False)),
        template_path=resolved_three_d_template_path,
    )
    three_d_branch_payload = three_d_processor.availability_payload()
    three_d_processed_rows = 0
    three_d_success_rows = 0
    three_d_skipped_rows = 0

    input_total_rows = len(read_parquet(input_manifest_path, columns=["image_id"]))
    if input_total_rows == 0:
        summary = _summary_payload(
            root=root,
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            aligned_dir=aligned_dir,
            summary_path=summary_path,
            progress_path=progress_path,
            provider_diagnostics=None,
            runtime_config=runtime_config,
            input_total_rows=0,
            skipped_existing_rows=0,
            processed_rows=processed_rows_count,
            successful_rows=successful_rows_count,
            rejected_rows=processed_rows_count - successful_rows_count,
            error_rows=error_rows_count,
            quality_pass_true_count=successful_rows_count,
            aligned_nonempty_count=aligned_nonempty_count,
            landmarks_nonempty_count=landmarks_nonempty_count,
            alignment_attempted_count=alignment_attempted_count,
            alignment_applied_count=alignment_applied_count,
            alignment_improved_quality_count=alignment_improved_quality_count,
            rejection_counter=rejection_counter,
            processing_status_counter=processing_status_counter,
            aligned_examples=aligned_examples,
            shards_written=shards_written,
        )
        summary["gpu_runtime_check"] = gpu_runtime
        write_json(summary, summary_path)
        write_json(summary, progress_path)
        write_json(
            _three_d_summary_payload(
                root=root,
                artifacts_dir=three_d_artifacts_dir,
                manifest_path=three_d_manifest_path,
                summary_path=three_d_summary_path,
                branch_payload=three_d_branch_payload,
                processed_rows=three_d_processed_rows,
                success_rows=three_d_success_rows,
                skipped_rows=three_d_skipped_rows,
            ),
            three_d_summary_path,
        )
        return PreprocessingArtifacts(
            output_manifest_path=output_manifest_path,
            summary_path=summary_path,
            summary=summary,
        )

    detector = _build_detector(preprocess_config, runtime_config, root)
    provider_diagnostics = detector.diagnostics_payload()

    skipped_existing_rows = 0
    batch_size = int(runtime_config["batch_size"])
    limit = runtime_config.get("limit")
    offset = int(runtime_config["offset"])
    num_workers = int(runtime_config["num_workers"])

    for source_batch in iter_parquet_batches(
        input_manifest_path,
        batch_size=batch_size,
        offset=offset,
        limit=limit,
    ):
        batch_rows = [
            _normalize_manifest_row(row, default_reference_model)
            for row in source_batch
            if not (runtime_config["resume"] and str(row.get("image_id")) in existing_processed_ids)
        ]
        skipped_existing_rows += len(source_batch) - len(batch_rows)
        if not batch_rows:
            continue

        loaded_rows = _load_batch_images(batch_rows, root, num_workers)
        batch_output_rows: list[dict[str, Any]] = []
        batch_three_d_rows: list[dict[str, Any]] = []

        for input_row, image, image_load_error in loaded_rows:
            if image_load_error is not None:
                processing_status = STATUS_IMAGE_READ_ERROR
                processing_status_counter[processing_status] += 1
                batch_output_rows.append(
                    _build_output_row(
                        input_row=input_row,
                        detection=None,
                        num_faces=0,
                        quality_reasons=[],
                        alignment_outcome=None,
                        aligned_image_path=None,
                        processing_status=processing_status,
                        processing_error_message=image_load_error,
                    )
                )
                error_rows_count += 1
                continue

            assert image is not None
            try:
                detections, primary_detection, quality_reasons, alignment_outcome, _analysis_image = select_detection_with_auto_alignment(
                    detector=detector,
                    image_bgr=image,
                    target_image_size=target_image_size,
                    quality_gate=quality_gate,
                    auto_alignment_config=auto_alignment_config,
                    quality_evaluator=_evaluate_quality,
                )
            except Exception as exc:
                processing_status = STATUS_DETECTION_ERROR
                processing_status_counter[processing_status] += 1
                batch_output_rows.append(
                    _build_output_row(
                        input_row=input_row,
                        detection=None,
                        num_faces=0,
                        quality_reasons=[],
                        alignment_outcome=None,
                        aligned_image_path=None,
                        processing_status=processing_status,
                        processing_error_message=str(exc),
                    )
                )
                error_rows_count += 1
                continue

            num_faces = len(detections)
            aligned_image_path: str | None = None
            processing_status = STATUS_PROCESSED
            processing_error_message: str | None = None

            if primary_detection is not None and primary_detection.aligned_image is not None:
                aligned_path = aligned_dir / f"{input_row['image_id']}.jpg"
                try:
                    save_image(aligned_path, primary_detection.aligned_image)
                    aligned_image_path = to_relative_path(aligned_path, root)
                except Exception as exc:
                    processing_status = STATUS_ALIGNED_SAVE_ERROR
                    processing_error_message = str(exc)
                    aligned_image_path = None

            row = _build_output_row(
                input_row=input_row,
                detection=primary_detection,
                num_faces=num_faces,
                quality_reasons=quality_reasons,
                alignment_outcome=alignment_outcome,
                aligned_image_path=aligned_image_path,
                processing_status=processing_status,
                processing_error_message=processing_error_message,
            )
            batch_output_rows.append(row)

            if three_d_processor.availability.enabled:
                three_d_processed_rows += 1
                artifact = three_d_processor.build_artifact(
                    image_id=str(input_row["image_id"]),
                    landmarks_2d=(primary_detection.landmarks if primary_detection is not None else None),
                    yaw=(primary_detection.yaw if primary_detection is not None else None),
                    pitch=(primary_detection.pitch if primary_detection is not None else None),
                    roll=(primary_detection.roll if primary_detection is not None else None),
                )
                artifact_path: str | None = None
                if artifact.get("status") == "success":
                    artifact_file_path = three_d_artifacts_dir / f"{input_row['image_id']}.json"
                    save_3d_artifact(artifact_file_path, artifact)
                    artifact_path = to_relative_path(artifact_file_path, root)
                    three_d_success_rows += 1
                else:
                    three_d_skipped_rows += 1
                batch_three_d_rows.append(
                    {
                        "image_id": str(input_row["image_id"]),
                        "artifact_path": artifact_path,
                        "status": artifact.get("status"),
                        "reason": artifact.get("reason"),
                        "artifact_version": artifact.get("artifact_version"),
                        "template_path": artifact.get("template_path"),
                    }
                )

            processing_status_counter[processing_status] += 1
            if processing_status in PROCESSED_STATUSES:
                processed_rows_count += 1
                existing_processed_ids.add(str(input_row["image_id"]))
                if row.get("alignment_attempted"):
                    alignment_attempted_count += 1
                if row.get("alignment_applied"):
                    alignment_applied_count += 1
                if row.get("alignment_improved_quality"):
                    alignment_improved_quality_count += 1
                if row["quality_pass"]:
                    successful_rows_count += 1
                else:
                    for reason in quality_reasons:
                        rejection_counter[reason] += 1
                if _is_nonempty(row.get("aligned_image_path")):
                    aligned_nonempty_count += 1
                    if len(aligned_examples) < 5:
                        aligned_examples.append(str(row["aligned_image_path"]))
                if _is_nonempty(row.get("landmarks")):
                    landmarks_nonempty_count += 1
            else:
                error_rows_count += 1

        if not batch_output_rows:
            continue

        shard_name = f"part-{shards_written:06d}.parquet"
        write_parquet_shard(batch_output_rows, output_manifest_path, PREPROCESS_COLUMNS, shard_name)
        if batch_three_d_rows:
            write_parquet_shard(batch_three_d_rows, three_d_manifest_path, THREE_D_MANIFEST_COLUMNS, shard_name)
        shards_written += 1

        summary = _summary_payload(
            root=root,
            input_manifest_path=input_manifest_path,
            output_manifest_path=output_manifest_path,
            aligned_dir=aligned_dir,
            summary_path=summary_path,
            progress_path=progress_path,
            provider_diagnostics=provider_diagnostics,
            runtime_config=runtime_config,
            input_total_rows=input_total_rows,
            skipped_existing_rows=skipped_existing_rows,
            processed_rows=processed_rows_count,
            successful_rows=successful_rows_count,
            rejected_rows=processed_rows_count - successful_rows_count,
            error_rows=error_rows_count,
            quality_pass_true_count=successful_rows_count,
            aligned_nonempty_count=aligned_nonempty_count,
            landmarks_nonempty_count=landmarks_nonempty_count,
            alignment_attempted_count=alignment_attempted_count,
            alignment_applied_count=alignment_applied_count,
            alignment_improved_quality_count=alignment_improved_quality_count,
            rejection_counter=rejection_counter,
            processing_status_counter=processing_status_counter,
            aligned_examples=aligned_examples,
            shards_written=shards_written,
        )
        summary["gpu_runtime_check"] = gpu_runtime
        write_json(summary, summary_path)
        write_json(summary, progress_path)
        write_json(
            _three_d_summary_payload(
                root=root,
                artifacts_dir=three_d_artifacts_dir,
                manifest_path=three_d_manifest_path,
                summary_path=three_d_summary_path,
                branch_payload=three_d_branch_payload,
                processed_rows=three_d_processed_rows,
                success_rows=three_d_success_rows,
                skipped_rows=three_d_skipped_rows,
            ),
            three_d_summary_path,
        )

    final_summary = _summary_payload(
        root=root,
        input_manifest_path=input_manifest_path,
        output_manifest_path=output_manifest_path,
        aligned_dir=aligned_dir,
        summary_path=summary_path,
        progress_path=progress_path,
        provider_diagnostics=provider_diagnostics,
        runtime_config=runtime_config,
        input_total_rows=input_total_rows,
        skipped_existing_rows=skipped_existing_rows,
        processed_rows=processed_rows_count,
        successful_rows=successful_rows_count,
        rejected_rows=processed_rows_count - successful_rows_count,
        error_rows=error_rows_count,
        quality_pass_true_count=successful_rows_count,
        aligned_nonempty_count=aligned_nonempty_count,
        landmarks_nonempty_count=landmarks_nonempty_count,
        alignment_attempted_count=alignment_attempted_count,
        alignment_applied_count=alignment_applied_count,
        alignment_improved_quality_count=alignment_improved_quality_count,
        rejection_counter=rejection_counter,
        processing_status_counter=processing_status_counter,
        aligned_examples=aligned_examples,
        shards_written=shards_written,
    )
    final_summary["gpu_runtime_check"] = gpu_runtime
    write_json(final_summary, summary_path)
    write_json(final_summary, progress_path)
    write_json(
        _three_d_summary_payload(
            root=root,
            artifacts_dir=three_d_artifacts_dir,
            manifest_path=three_d_manifest_path,
            summary_path=three_d_summary_path,
            branch_payload=three_d_branch_payload,
            processed_rows=three_d_processed_rows,
            success_rows=three_d_success_rows,
            skipped_rows=three_d_skipped_rows,
        ),
        three_d_summary_path,
    )

    return PreprocessingArtifacts(
        output_manifest_path=output_manifest_path,
        summary_path=summary_path,
        summary=final_summary,
    )
