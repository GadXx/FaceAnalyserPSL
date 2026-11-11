from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any

import cv2
import numpy as np

from src.face_aesthetics.data.io import save_image, write_json
from src.face_aesthetics.landmarks import LandmarkSourceResult


MEDIAPIPE_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
MEDIAPIPE_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MEDIAPIPE_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MEDIAPIPE_LEFT_UPPER_EYELID = [246, 161, 160, 159, 158, 157, 173]
MEDIAPIPE_RIGHT_UPPER_EYELID = [466, 388, 387, 386, 385, 384, 398]
MEDIAPIPE_LEFT_LOWER_EYELID = [33, 7, 163, 144, 145, 153, 154, 155, 133]
MEDIAPIPE_RIGHT_LOWER_EYELID = [362, 382, 381, 380, 374, 373, 390, 249, 263]
MEDIAPIPE_LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
MEDIAPIPE_RIGHT_BROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
MEDIAPIPE_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
MEDIAPIPE_LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
MEDIAPIPE_NOSE = [6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 168]

MEDIAPIPE_REGION_INDICES = {
    "dense_face_oval": MEDIAPIPE_FACE_OVAL,
    "dense_eyes": sorted(set(MEDIAPIPE_LEFT_EYE + MEDIAPIPE_RIGHT_EYE)),
    "dense_brows": sorted(set(MEDIAPIPE_LEFT_BROW + MEDIAPIPE_RIGHT_BROW)),
    "dense_lips": sorted(set(MEDIAPIPE_LIPS_OUTER + MEDIAPIPE_LIPS_INNER)),
    "dense_nose": MEDIAPIPE_NOSE,
    "approximate_cheeks": [234, 454, 93, 323, 136, 365],
}

MEDIAPIPE_DIRECT_REGIONS = [
    "dense_face_oval",
    "dense_eyes",
    "dense_brows",
    "dense_lips",
    "dense_nose",
]

MEDIAPIPE_PROXY_REGIONS = [
    "approximate_cheeks",
]

MEDIAPIPE_UNAVAILABLE_REGIONS = [
    "ears",
    "hairline",
    "skin_mask",
]

_TASKS_LANDMARKER_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_TASKS_LANDMARKER_CACHE_LOCK = threading.Lock()


def _import_mediapipe() -> tuple[Any | None, str | None]:
    try:
        import mediapipe as mp  # type: ignore

        return mp, None
    except Exception as exc:  # pragma: no cover - optional dependency
        return None, f"{type(exc).__name__}:{exc}"


def _normalize_delegate_name(value: Any) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "gpu"}:
        return "auto"
    return normalized


def _gpu_globally_disabled() -> bool:
    normalized = str(os.environ.get("MEDIAPIPE_DISABLE_GPU", "")).strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _build_solutions_face_mesh(mp: Any, config: dict[str, Any]) -> Any:
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
        return None
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=bool(config.get("static_image_mode", True)),
        max_num_faces=int(config.get("max_num_faces", 1)),
        refine_landmarks=bool(config.get("refine_landmarks", True)),
        min_detection_confidence=float(config.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
    )


def _build_tasks_face_landmarker(
    mp: Any,
    config: dict[str, Any],
    project_root: Path | None,
) -> tuple[Any | None, dict[str, Any]]:
    if not hasattr(mp, "tasks") or not hasattr(mp.tasks, "vision"):
        return None, {"error": "mediapipe_tasks_api_unavailable"}
    model_asset_path = config.get("model_asset_path")
    if not model_asset_path:
        return None, {"error": "mediapipe_face_landmarker_model_asset_missing"}
    resolved_model_path = Path(str(model_asset_path))
    if project_root is not None and not resolved_model_path.is_absolute():
        resolved_model_path = (project_root / resolved_model_path).resolve()
    if not resolved_model_path.exists():
        return None, {"error": f"mediapipe_face_landmarker_model_not_found:{resolved_model_path}"}

    requested_delegate = _normalize_delegate_name(config.get("delegate"))
    fallback_to_cpu = bool(config.get("fallback_to_cpu", True))
    gpu_disabled = _gpu_globally_disabled()
    effective_requested_delegate = "cpu" if gpu_disabled else requested_delegate
    delegate_enum = getattr(mp.tasks.BaseOptions, "Delegate", None)

    def _delegate_value(name: str) -> Any | None:
        if delegate_enum is None:
            return None
        return getattr(delegate_enum, name.upper(), None)

    def _create(delegate_name: str) -> tuple[dict[str, Any] | None, str | None]:
        cache_key = (str(resolved_model_path), delegate_name)
        with _TASKS_LANDMARKER_CACHE_LOCK:
            cached = _TASKS_LANDMARKER_CACHE.get(cache_key)
            if cached is not None:
                return cached, None
        try:  # pragma: no cover - optional runtime branch
            from mediapipe.tasks.python import vision  # type: ignore

            kwargs = {"model_asset_path": str(resolved_model_path)}
            delegate_value = _delegate_value(delegate_name)
            if delegate_name == "gpu" and delegate_value is None:
                return None, "mediapipe_gpu_delegate_enum_unavailable"
            if delegate_name in {"gpu", "cpu"} and delegate_value is not None:
                kwargs["delegate"] = delegate_value
            base_options = mp.tasks.BaseOptions(**kwargs)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=int(config.get("max_num_faces", 1)),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            landmarker = vision.FaceLandmarker.create_from_options(options)
            cached_entry = {"landmarker": landmarker, "lock": threading.Lock()}
            with _TASKS_LANDMARKER_CACHE_LOCK:
                _TASKS_LANDMARKER_CACHE[cache_key] = cached_entry
            return cached_entry, None
        except Exception as exc:
            return None, f"mediapipe_face_landmarker_init_failed:{delegate_name}:{type(exc).__name__}:{exc}"

    delegate_attempts: list[str] = []
    selected_delegate = None
    fallback_reason = None
    warnings: list[str] = []

    if gpu_disabled:
        delegate_attempts = ["cpu"]
        if requested_delegate != "cpu":
            warnings.append("mediapipe_gpu_disabled_by_env")
            fallback_reason = "gpu_disabled_by_env"
    elif effective_requested_delegate == "auto":
        delegate_attempts = ["gpu", "cpu"]
    else:
        delegate_attempts = [effective_requested_delegate]
        if effective_requested_delegate == "gpu" and fallback_to_cpu:
            delegate_attempts.append("cpu")

    last_error = None
    for index, delegate_name in enumerate(delegate_attempts):
        cached_entry, error = _create(delegate_name)
        if cached_entry is not None:
            selected_delegate = delegate_name
            if index > 0:
                fallback_reason = last_error
                warnings.append("mediapipe_delegate_fallback_to_cpu")
            return cached_entry, {
                "requested_delegate": requested_delegate,
                "selected_delegate": selected_delegate,
                "fallback_to_cpu": bool(index > 0 and selected_delegate == "cpu"),
                "fallback_reason": fallback_reason,
                "model_asset_path": str(resolved_model_path),
                "gpu_disabled": gpu_disabled,
                "warnings": warnings,
            }
        last_error = error

    return None, {
        "error": last_error or "mediapipe_face_landmarker_init_failed",
        "requested_delegate": requested_delegate,
        "selected_delegate": None,
        "fallback_to_cpu": False,
        "fallback_reason": fallback_reason,
        "model_asset_path": str(resolved_model_path),
        "gpu_disabled": gpu_disabled,
        "warnings": warnings,
    }


def get_mediapipe_runtime_startup_diagnostics(
    *,
    config: dict[str, Any],
    project_root: Path | None = None,
) -> dict[str, Any]:
    requested_delegate = _normalize_delegate_name(config.get("delegate"))
    payload = {
        "mediapipe_enabled": bool(config.get("enabled", False)),
        "mediapipe_available": False,
        "requested_delegate": requested_delegate,
        "selected_delegate": None,
        "gpu_disabled": _gpu_globally_disabled(),
        "fallback_to_cpu": bool(config.get("fallback_to_cpu", True)),
        "fallback_reason": None,
        "landmark_count": None,
        "model_asset_path": str(config.get("model_asset_path") or ""),
        "runtime_backend": None,
        "warnings": [],
    }
    if not payload["mediapipe_enabled"]:
        payload["warnings"] = ["mediapipe_disabled_in_config"]
        return payload

    mp, error = _import_mediapipe()
    if mp is None:
        payload["warnings"] = ["mediapipe_dependency_unavailable"]
        payload["fallback_reason"] = error
        return payload

    cached_entry, diagnostics = _build_tasks_face_landmarker(mp, config, project_root)
    if cached_entry is None:
        payload["warnings"] = list(diagnostics.get("warnings", []))
        payload["fallback_reason"] = diagnostics.get("error")
        payload["selected_delegate"] = diagnostics.get("selected_delegate")
        payload["mediapipe_available"] = False
        return payload

    payload.update(
        {
            "mediapipe_available": True,
            "selected_delegate": diagnostics.get("selected_delegate"),
            "gpu_disabled": bool(diagnostics.get("gpu_disabled", payload["gpu_disabled"])),
            "fallback_to_cpu": bool(diagnostics.get("fallback_to_cpu", payload["fallback_to_cpu"])),
            "fallback_reason": diagnostics.get("fallback_reason"),
            "model_asset_path": diagnostics.get("model_asset_path") or payload["model_asset_path"],
            "runtime_backend": "tasks_face_landmarker",
            "landmark_count": 478,
            "warnings": list(diagnostics.get("warnings", [])),
        }
    )
    return payload


def _draw_debug_indices(image_bgr: np.ndarray, landmarks_xy: list[list[float]]) -> np.ndarray:
    canvas = image_bgr.copy()
    for index, point in enumerate(landmarks_xy):
        x = int(round(point[0]))
        y = int(round(point[1]))
        cv2.circle(canvas, (x, y), 1, (0, 255, 255), -1)
        cv2.putText(canvas, str(index), (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 180, 255), 1, cv2.LINE_AA)
    return canvas


def write_mediapipe_debug_artifacts(
    *,
    project_root: Path,
    report_paths: dict[str, Any],
    image_bgr: np.ndarray,
    landmarks_xyz: list[list[float]],
) -> dict[str, str]:
    json_path = (project_root / str(report_paths["mediapipe_debug_landmarks_path"])).resolve()
    indices_image_path = (project_root / str(report_paths["mediapipe_debug_indices_image_path"])).resolve()
    region_report_path = (project_root / str(report_paths["mediapipe_region_report_path"])).resolve()
    write_json({"landmarks": landmarks_xyz}, json_path)
    write_json({"regions": MEDIAPIPE_REGION_INDICES}, region_report_path)
    xy = [[point[0], point[1]] for point in landmarks_xyz]
    debug_image = _draw_debug_indices(image_bgr=image_bgr, landmarks_xy=xy)
    save_image(indices_image_path, debug_image)
    return {
        "mediapipe_debug_landmarks_path": json_path.relative_to(project_root).as_posix(),
        "mediapipe_debug_indices_image_path": indices_image_path.relative_to(project_root).as_posix(),
        "mediapipe_region_report_path": region_report_path.relative_to(project_root).as_posix(),
    }


def build_mediapipe_source_result(
    *,
    image_bgr: np.ndarray | None,
    config: dict[str, Any],
    project_root: Path | None = None,
) -> LandmarkSourceResult:
    enabled = bool(config.get("enabled", False))
    mp, error = _import_mediapipe()
    warnings: list[str] = []
    metadata = {
        "model": config.get("model"),
        "output_landmarks": config.get("output_landmarks"),
        "region_indices": MEDIAPIPE_REGION_INDICES,
        "mediapipe_unavailable_reason": None,
        "runtime_backend": None,
        "requested_delegate": _normalize_delegate_name(config.get("delegate")),
        "selected_delegate": None,
        "fallback_to_cpu": bool(config.get("fallback_to_cpu", True)),
        "fallback_reason": None,
        "gpu_disabled": _gpu_globally_disabled(),
        "model_asset_path": str(config.get("model_asset_path") or ""),
    }
    if not enabled:
        return LandmarkSourceResult(
            source="mediapipe_face_mesh",
            enabled=False,
            available=False,
            optional=True,
            unavailable_regions=MEDIAPIPE_DIRECT_REGIONS + MEDIAPIPE_PROXY_REGIONS + MEDIAPIPE_UNAVAILABLE_REGIONS,
            dependency_message="disabled_in_config",
            metadata=metadata,
        )
    if mp is None:
        warnings.append("mediapipe_dependency_unavailable")
        metadata["mediapipe_unavailable_reason"] = error
        return LandmarkSourceResult(
            source="mediapipe_face_mesh",
            enabled=True,
            available=False,
            optional=True,
            unavailable_regions=MEDIAPIPE_DIRECT_REGIONS + MEDIAPIPE_PROXY_REGIONS + MEDIAPIPE_UNAVAILABLE_REGIONS,
            warnings=warnings,
            dependency_message=f"mediapipe_import_failed:{error}",
            metadata=metadata,
        )
    if image_bgr is None:
        metadata["mediapipe_unavailable_reason"] = "image_not_provided_for_runtime_extraction"
        return LandmarkSourceResult(
            source="mediapipe_face_mesh",
            enabled=True,
            available=False,
            optional=True,
            unavailable_regions=MEDIAPIPE_UNAVAILABLE_REGIONS,
            direct_regions=MEDIAPIPE_DIRECT_REGIONS,
            proxy_regions=MEDIAPIPE_PROXY_REGIONS,
            dependency_message="image_not_provided_for_runtime_extraction",
            metadata=metadata,
        )

    try:  # pragma: no cover - dependency optional
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        face_landmarks: list[Any] = []
        backend = None
        prefer_tasks_backend = bool(config.get("model_asset_path")) or str(config.get("model") or "").strip().lower() == "face_landmarker"
        cached_landmarker = None
        init_error = None
        if prefer_tasks_backend:
            cached_landmarker, init_error = _build_tasks_face_landmarker(mp, config, project_root)
        face_mesh = None if cached_landmarker is not None or prefer_tasks_backend else _build_solutions_face_mesh(mp, config)
        if cached_landmarker is None and face_mesh is None:
            cached_landmarker, init_error = _build_tasks_face_landmarker(mp, config, project_root)
        if cached_landmarker is not None:
            backend = "tasks_face_landmarker"
            metadata["requested_delegate"] = init_error.get("requested_delegate", metadata["requested_delegate"])
            metadata["selected_delegate"] = init_error.get("selected_delegate")
            metadata["fallback_to_cpu"] = init_error.get("fallback_to_cpu", metadata["fallback_to_cpu"])
            metadata["fallback_reason"] = init_error.get("fallback_reason")
            metadata["gpu_disabled"] = init_error.get("gpu_disabled", metadata["gpu_disabled"])
            warnings.extend(list(init_error.get("warnings", [])))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            with cached_landmarker["lock"]:
                detection_result = cached_landmarker["landmarker"].detect(mp_image)
            face_landmarks = list(getattr(detection_result, "face_landmarks", []) or [])
        elif face_mesh is not None:
            backend = "solutions_face_mesh"
            result = face_mesh.process(image_rgb)
            face_mesh.close()
            face_landmarks = list(result.multi_face_landmarks or [])
        else:
            if init_error is None:
                _, init_error = _build_tasks_face_landmarker(mp, config, project_root)
            if init_error is not None:
                metadata["mediapipe_unavailable_reason"] = init_error.get("error")
                metadata["requested_delegate"] = init_error.get("requested_delegate", metadata["requested_delegate"])
                metadata["selected_delegate"] = init_error.get("selected_delegate")
                metadata["fallback_to_cpu"] = init_error.get("fallback_to_cpu", metadata["fallback_to_cpu"])
                metadata["fallback_reason"] = init_error.get("fallback_reason")
                metadata["gpu_disabled"] = init_error.get("gpu_disabled", metadata["gpu_disabled"])
                return LandmarkSourceResult(
                    source="mediapipe_face_mesh",
                    enabled=True,
                    available=False,
                    optional=True,
                    unavailable_regions=MEDIAPIPE_DIRECT_REGIONS + MEDIAPIPE_PROXY_REGIONS + MEDIAPIPE_UNAVAILABLE_REGIONS,
                    warnings=["mediapipe_runtime_backend_unavailable", *list(init_error.get("warnings", []))],
                    dependency_message=str(init_error.get("error")),
                    metadata=metadata,
                )
        if not face_landmarks:
            metadata["mediapipe_unavailable_reason"] = "no_face_detected"
            return LandmarkSourceResult(
                source="mediapipe_face_mesh",
                enabled=True,
                available=False,
                optional=True,
                direct_regions=MEDIAPIPE_DIRECT_REGIONS,
                proxy_regions=MEDIAPIPE_PROXY_REGIONS,
                unavailable_regions=MEDIAPIPE_UNAVAILABLE_REGIONS,
                warnings=["mediapipe_no_face_detected"],
                dependency_message="no_face_detected",
                metadata=metadata,
            )
        if len(face_landmarks) != 1:
            metadata["mediapipe_unavailable_reason"] = "multiple_faces_detected"
            return LandmarkSourceResult(
                source="mediapipe_face_mesh",
                enabled=True,
                available=False,
                optional=True,
                direct_regions=MEDIAPIPE_DIRECT_REGIONS,
                proxy_regions=MEDIAPIPE_PROXY_REGIONS,
                unavailable_regions=MEDIAPIPE_UNAVAILABLE_REGIONS,
                warnings=["mediapipe_multiple_faces_detected"],
                dependency_message="multiple_faces_detected",
                metadata=metadata,
            )

        height, width = image_bgr.shape[:2]
        points: list[list[float]] = []
        face_landmarks_payload = face_landmarks[0]
        if hasattr(face_landmarks_payload, "landmark"):
            iterable_points = face_landmarks_payload.landmark
        else:
            iterable_points = face_landmarks_payload
        for point in iterable_points:
            points.append(
                [
                    float(point.x * width),
                    float(point.y * height),
                    float(getattr(point, "z", 0.0)),
                ]
            )
        metadata.update(
            {
                "mediapipe_unavailable_reason": None,
                "landmark_regions": MEDIAPIPE_REGION_INDICES,
                "runtime_backend": backend,
                "landmark_count": len(points),
            }
        )
        return LandmarkSourceResult(
            source="mediapipe_face_mesh",
            enabled=True,
            available=True,
            optional=True,
            coordinate_space="analysis_image",
            confidence=1.0,
            landmark_count=len(points),
            landmarks=points,
            available_regions=MEDIAPIPE_DIRECT_REGIONS + MEDIAPIPE_PROXY_REGIONS,
            direct_regions=MEDIAPIPE_DIRECT_REGIONS,
            proxy_regions=MEDIAPIPE_PROXY_REGIONS,
            unavailable_regions=MEDIAPIPE_UNAVAILABLE_REGIONS,
            warnings=warnings,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - dependency optional
        metadata["mediapipe_unavailable_reason"] = f"runtime_failed:{exc}"
        return LandmarkSourceResult(
            source="mediapipe_face_mesh",
            enabled=True,
            available=False,
            optional=True,
            unavailable_regions=MEDIAPIPE_DIRECT_REGIONS + MEDIAPIPE_PROXY_REGIONS + MEDIAPIPE_UNAVAILABLE_REGIONS,
            warnings=["mediapipe_runtime_failed"],
            dependency_message=f"mediapipe_runtime_failed:{exc}",
            metadata=metadata,
        )
