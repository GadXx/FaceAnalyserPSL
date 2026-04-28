from __future__ import annotations

import ctypes
import ctypes.util
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from insightface.app.common import Face
from src.face_aesthetics.features.extraction import derive_semantic_landmark_bundle
from src.face_aesthetics.landmarks.insightface_106 import (
    LEFT_EYE_INDICES,
    MOUTH_LEFT_CORNER_INDEX,
    MOUTH_RIGHT_CORNER_INDEX,
    NOSE_TIP_INDEX,
    RIGHT_EYE_INDICES,
    has_106_landmarks,
    point_for_index,
    points_for_indices,
)


POSE_MODEL_POINTS_5 = np.array(
    [
        [-30.0, 35.0, 30.0],
        [30.0, 35.0, 30.0],
        [0.0, 0.0, 0.0],
        [-25.0, -35.0, 30.0],
        [25.0, -35.0, 30.0],
    ],
    dtype=np.float64,
)

SUPPORTED_OCCLUSION_ATTRIBUTES = ("occlusion", "occlusion_score", "mask_score", "mask_prob")
CUDA_RUNTIME_LIBRARIES = (
    "libcublasLt.so.12",
    "libcublas.so",
    "libcudnn.so",
    "libcudart.so",
)


def _wrap_signed_angle(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _normalize_pose_angles(yaw_deg: float, pitch_deg: float, roll_deg: float) -> tuple[float, float, float]:
    yaw = _wrap_signed_angle(yaw_deg)
    pitch = _wrap_signed_angle(pitch_deg)
    roll = _wrap_signed_angle(roll_deg)

    # solvePnP + Euler decomposition can encode a mild negative pitch as a
    # mathematically equivalent angle close to +180 deg. Fold pitch into an
    # interpretable signed head-tilt range for quality gating/reporting.
    if pitch > 90.0:
        pitch -= 180.0
    elif pitch < -90.0:
        pitch += 180.0

    return yaw, pitch, roll


def _heuristic_pose_from_landmarks(landmarks: list[list[float]]) -> tuple[float | None, float | None, float | None]:
    if len(landmarks) != 5:
        return None, None, None

    left_eye, right_eye, nose, left_mouth, right_mouth = [np.asarray(point, dtype=np.float64) for point in landmarks]
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0
    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    vertical_span = float(mouth_center[1] - eye_center[1])

    if eye_distance < 1e-6 or abs(vertical_span) < 1e-6:
        return None, None, None

    roll = math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0])))

    center_x = float((eye_center[0] + mouth_center[0]) / 2.0)
    yaw_ratio = float((nose[0] - center_x) / eye_distance)
    yaw = float(np.clip(yaw_ratio * 90.0, -45.0, 45.0))

    pitch_ratio = float((nose[1] - eye_center[1]) / vertical_span)
    pitch = float(np.clip((pitch_ratio - 0.5) * 100.0, -45.0, 45.0))

    return yaw, pitch, roll


def estimate_eye_line_angle_degrees(landmarks: list[list[float]]) -> float | None:
    if len(landmarks) < 2:
        return None
    left_eye = np.asarray(landmarks[0], dtype=np.float64)
    right_eye = np.asarray(landmarks[1], dtype=np.float64)
    vector = right_eye - left_eye
    if float(np.linalg.norm(vector)) < 1e-6:
        return None
    return float(math.degrees(math.atan2(float(vector[1]), float(vector[0]))))


def compute_auto_alignment_rotation(
    detection: DetectedFace | None,
    auto_alignment_config: dict[str, Any],
) -> tuple[float | None, str | None]:
    if detection is None or not detection.landmarks or len(detection.landmarks) < 5:
        return None, None

    source = str(auto_alignment_config.get("alignment_source", "auto")).strip() or "auto"
    eye_line_angle = estimate_eye_line_angle_degrees(detection.landmarks)
    pose_roll = float(detection.roll) if detection.roll is not None and np.isfinite(float(detection.roll)) else None

    if source == "eye_line":
        rotation = -eye_line_angle if eye_line_angle is not None else None
        resolved_source = "eye_line" if rotation is not None else None
    elif source == "landmarks":
        rotation = -pose_roll if pose_roll is not None else (-eye_line_angle if eye_line_angle is not None else None)
        resolved_source = "landmarks" if pose_roll is not None else ("eye_line" if rotation is not None else None)
    else:
        if eye_line_angle is not None:
            rotation = -eye_line_angle
            resolved_source = "eye_line"
        elif pose_roll is not None:
            rotation = -pose_roll
            resolved_source = "landmarks"
        else:
            rotation = None
            resolved_source = None

    if rotation is None:
        return None, None

    max_rotation = float(auto_alignment_config.get("max_auto_rotation_deg", 20.0))
    rotation = float(np.clip(rotation, -max_rotation, max_rotation))
    min_rotation = float(auto_alignment_config.get("min_auto_rotation_deg", 3.0))
    if abs(rotation) < min_rotation:
        return None, resolved_source
    return rotation, resolved_source


@dataclass
class DetectedFace:
    bbox: list[float]
    landmarks: list[list[float]]
    landmarks_dense: list[list[float]] | None
    det_score: float | None
    aligned_image: np.ndarray | None
    yaw: float | None
    pitch: float | None
    roll: float | None
    occlusion_score: float | None
    detection_backend: str


@dataclass
class AutoAlignmentOutcome:
    attempted: bool
    applied: bool
    improved_quality: bool
    source: str | None
    status: str
    rotation_deg: float | None
    transform: dict[str, Any] | None
    original_pose: dict[str, float | None]
    aligned_pose: dict[str, float | None]
    original_quality_pass: bool | None
    original_quality_fail_reason: str | None
    aligned_quality_pass: bool | None
    aligned_quality_fail_reason: str | None


@dataclass
class ProviderDiagnostics:
    onnxruntime_version: str
    preload_dlls_supported: bool
    preload_dlls_success: bool
    preload_dlls_error: str | None
    cuda_runtime_libraries: dict[str, dict[str, Any]]
    missing_cuda_runtime_libraries: list[str]
    available_providers: list[str]
    preferred_execution_provider: str
    requested_providers: list[str]
    selected_execution_provider: str
    fallback_to_cpu: bool
    fallback_reason: str | None
    device_id: int
    actual_model_providers: dict[str, list[str]]
    strict_gpu_mode: bool


def _rotation_matrix_to_euler_angles(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0.0

    return tuple(math.degrees(angle) for angle in (yaw, pitch, roll))


def estimate_pose_from_landmarks(
    image_shape: tuple[int, int, int], landmarks: list[list[float]]
) -> tuple[float | None, float | None, float | None]:
    if len(landmarks) != 5:
        return None, None, None

    image_points = np.asarray(landmarks, dtype=np.float64)
    height, width = image_shape[:2]
    focal_length = float(width)
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    solved, rotation_vector, _translation_vector = cv2.solvePnP(
        POSE_MODEL_POINTS_5,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not solved:
        return _heuristic_pose_from_landmarks(landmarks)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    raw_yaw, raw_pitch, raw_roll = _rotation_matrix_to_euler_angles(rotation_matrix)
    normalized_pose = _normalize_pose_angles(raw_yaw, raw_pitch, raw_roll)

    if any(value is None or not np.isfinite(value) for value in normalized_pose):
        return _heuristic_pose_from_landmarks(landmarks)

    if abs(float(normalized_pose[1])) > 25.0:
        heuristic_pose = _heuristic_pose_from_landmarks(landmarks)
        if all(value is not None and np.isfinite(value) for value in heuristic_pose):
            return heuristic_pose

    return normalized_pose


def compute_blur_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _extract_occlusion_score(face: Any) -> float | None:
    for attribute_name in SUPPORTED_OCCLUSION_ATTRIBUTES:
        if hasattr(face, attribute_name):
            try:
                return float(getattr(face, attribute_name))
            except (TypeError, ValueError):
                return None
        if isinstance(face, dict) and attribute_name in face:
            try:
                return float(face[attribute_name])
            except (TypeError, ValueError):
                return None
    return None


def _largest_bbox(boxes: np.ndarray) -> np.ndarray | None:
    if boxes is None or len(boxes) == 0:
        return None
    largest = None
    largest_area = -1.0
    for box in boxes:
        x, y, w, h = [float(value) for value in box[:4]]
        area = max(w, 0.0) * max(h, 0.0)
        if area > largest_area:
            largest_area = area
            largest = np.array([x, y, x + w, y + h], dtype=np.float32)
    return largest


def _extract_eye_and_nose_from_dense_landmarks(
    points: np.ndarray,
    bbox: list[float],
    detector_landmarks: list[list[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if has_106_landmarks(points):
        left_eye_indices, left_eye_points = points_for_indices(points, LEFT_EYE_INDICES)
        right_eye_indices, right_eye_points = points_for_indices(points, RIGHT_EYE_INDICES)
        nose_tip = point_for_index(points, NOSE_TIP_INDEX)
        if len(left_eye_points) > 0 and len(right_eye_points) > 0 and nose_tip is not None:
            return (
                np.asarray(left_eye_points.mean(axis=0), dtype=np.float32),
                np.asarray(right_eye_points.mean(axis=0), dtype=np.float32),
                np.asarray(nose_tip, dtype=np.float32),
            )

    if detector_landmarks and len(detector_landmarks) >= 3:
        detector_points = np.asarray(detector_landmarks, dtype=np.float32)
        return detector_points[0], detector_points[1], detector_points[2]

    x1, y1, x2, y2 = [float(value) for value in bbox]
    center_x = (x1 + x2) / 2.0
    y_values = points[:, 1]
    q60 = float(np.quantile(y_values, 0.60))

    upper_points = points[points[:, 1] <= q60]
    left_eye_candidates = upper_points[upper_points[:, 0] <= center_x]
    right_eye_candidates = upper_points[upper_points[:, 0] > center_x]

    def _cluster_center(candidates: np.ndarray) -> np.ndarray:
        if len(candidates) == 0:
            return np.array([center_x, (y1 + y2) / 2.0], dtype=np.float32)
        c_y_low = float(np.quantile(candidates[:, 1], 0.20))
        c_y_high = float(np.quantile(candidates[:, 1], 0.70))
        band = candidates[(candidates[:, 1] >= c_y_low) & (candidates[:, 1] <= c_y_high)]
        if len(band) == 0:
            band = candidates
        return band.mean(axis=0)

    left_eye = _cluster_center(left_eye_candidates)
    right_eye = _cluster_center(right_eye_candidates)

    eye_center_y = float((left_eye[1] + right_eye[1]) / 2.0)
    face_width = max(float(x2 - x1), 1.0)
    vertical_span = max(float(y2 - y1), 1.0)
    target_nose_y = eye_center_y + 0.38 * vertical_span
    central_band_half_width = 0.28 * face_width
    lower_bound = eye_center_y + 0.10 * vertical_span
    upper_bound = y1 + 0.78 * vertical_span

    nose_candidates = points[
        (points[:, 1] >= lower_bound)
        & (points[:, 1] <= upper_bound)
        & (np.abs(points[:, 0] - center_x) <= central_band_half_width)
    ]
    if len(nose_candidates) == 0:
        nose_candidates = points
    nose_scores = (
        np.abs(nose_candidates[:, 0] - center_x) / face_width
        + np.abs(nose_candidates[:, 1] - target_nose_y) / vertical_span
    )
    nose = nose_candidates[int(np.argmin(nose_scores))]
    return left_eye, right_eye, nose


def _extract_mouth_corners_from_dense_landmarks(
    points: np.ndarray,
    bbox: list[float],
    *,
    center_x: float,
    nose_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width = max(float(x2 - x1), 1.0)
    face_height = max(float(y2 - y1), 1.0)

    if has_106_landmarks(points):
        return (
            np.asarray(points[MOUTH_LEFT_CORNER_INDEX], dtype=np.float32),
            np.asarray(points[MOUTH_RIGHT_CORNER_INDEX], dtype=np.float32),
        )

    mouth_band_low = max(float(nose_y + 0.08 * face_height), float(y1 + 0.58 * face_height))
    mouth_band_high = min(float(y2 - 0.08 * face_height), float(y1 + 0.86 * face_height))
    horizontal_half_width = 0.30 * face_width

    mouth_candidates = points[
        (points[:, 1] >= mouth_band_low)
        & (points[:, 1] <= mouth_band_high)
        & (np.abs(points[:, 0] - center_x) <= horizontal_half_width)
    ]
    if len(mouth_candidates) < 4:
        mouth_candidates = points[
            (points[:, 1] >= max(float(nose_y), float(y1 + 0.54 * face_height)))
            & (points[:, 1] <= float(y1 + 0.88 * face_height))
            & (np.abs(points[:, 0] - center_x) <= 0.34 * face_width)
        ]
    if len(mouth_candidates) == 0:
        return (
            np.array([x1 + 0.35 * face_width, y1 + 0.75 * face_height], dtype=np.float32),
            np.array([x1 + 0.65 * face_width, y1 + 0.75 * face_height], dtype=np.float32),
        )

    y_low = float(np.quantile(mouth_candidates[:, 1], 0.15))
    y_high = float(np.quantile(mouth_candidates[:, 1], 0.75))
    lip_band = mouth_candidates[(mouth_candidates[:, 1] >= y_low) & (mouth_candidates[:, 1] <= y_high)]
    if len(lip_band) >= 4:
        mouth_candidates = lip_band

    left_candidates = mouth_candidates[mouth_candidates[:, 0] <= center_x]
    right_candidates = mouth_candidates[mouth_candidates[:, 0] >= center_x]

    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return (
            np.array([x1 + 0.35 * face_width, y1 + 0.75 * face_height], dtype=np.float32),
            np.array([x1 + 0.65 * face_width, y1 + 0.75 * face_height], dtype=np.float32),
        )

    left_mouth = left_candidates[np.argmin(left_candidates[:, 0])]
    right_mouth = right_candidates[np.argmax(right_candidates[:, 0])]
    return left_mouth, right_mouth


def _derive_five_point_landmarks(
    landmarks: list[list[float]],
    bbox: list[float],
    detector_landmarks: list[list[float]] | None = None,
) -> list[list[float]]:
    points = np.asarray(landmarks, dtype=np.float32)
    if len(points) == 5:
        return [[float(x), float(y)] for x, y in points]

    x1, _y1, x2, _y2 = [float(value) for value in bbox]
    center_x = (x1 + x2) / 2.0
    left_eye, right_eye, nose = _extract_eye_and_nose_from_dense_landmarks(
        points,
        bbox,
        detector_landmarks=detector_landmarks,
    )
    provisional_left_mouth, provisional_right_mouth = _extract_mouth_corners_from_dense_landmarks(
        points,
        bbox,
        center_x=center_x,
        nose_y=float(nose[1]),
    )
    semantic_bundle = derive_semantic_landmark_bundle(
        landmarks=[
            [float(left_eye[0]), float(left_eye[1])],
            [float(right_eye[0]), float(right_eye[1])],
            [float(nose[0]), float(nose[1])],
            [float(provisional_left_mouth[0]), float(provisional_left_mouth[1])],
            [float(provisional_right_mouth[0]), float(provisional_right_mouth[1])],
        ],
        landmarks_dense=points.tolist(),
        bbox=bbox,
    )
    semantic_points = semantic_bundle.get("semantic_points", {})
    left_mouth_payload = semantic_points.get("mouth_left")
    right_mouth_payload = semantic_points.get("mouth_right")
    left_mouth = (
        np.asarray([left_mouth_payload["x"], left_mouth_payload["y"]], dtype=np.float32)
        if isinstance(left_mouth_payload, dict)
        else provisional_left_mouth
    )
    right_mouth = (
        np.asarray([right_mouth_payload["x"], right_mouth_payload["y"]], dtype=np.float32)
        if isinstance(right_mouth_payload, dict)
        else provisional_right_mouth
    )

    return [
        [float(left_eye[0]), float(left_eye[1])],
        [float(right_eye[0]), float(right_eye[1])],
        [float(nose[0]), float(nose[1])],
        [float(left_mouth[0]), float(left_mouth[1])],
        [float(right_mouth[0]), float(right_mouth[1])],
    ]


def _crop_and_resize(image_bgr: np.ndarray, bbox: list[float], image_size: int) -> np.ndarray | None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_bgr.shape[1], x2)
    y2 = min(image_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def _rotate_image_with_canvas(
    image_bgr: np.ndarray,
    rotation_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_bgr.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)

    cos_value = abs(rotation_matrix[0, 0])
    sin_value = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_value) + (width * cos_value))
    new_height = int((height * cos_value) + (width * sin_value))

    rotation_matrix[0, 2] += (new_width / 2.0) - center[0]
    rotation_matrix[1, 2] += (new_height / 2.0) - center[1]

    rotated = cv2.warpAffine(
        image_bgr,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, rotation_matrix


def _apply_affine_to_points(points: list[list[float]] | None, transform: np.ndarray) -> list[list[float]] | None:
    if not points:
        return None
    points_array = np.asarray(points, dtype=np.float64)
    ones = np.ones((points_array.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([points_array, ones])
    transformed = homogeneous @ transform.T
    return [[float(x), float(y)] for x, y in transformed.tolist()]


def _transform_bbox(bbox: list[float] | None, transform: np.ndarray) -> list[float] | None:
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    corners = [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]
    transformed = _apply_affine_to_points(corners, transform)
    if not transformed:
        return None
    xs = [float(point[0]) for point in transformed]
    ys = [float(point[1]) for point in transformed]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _build_aligned_image(
    *,
    image_bgr: np.ndarray,
    bbox: list[float] | None,
    landmarks: list[list[float]] | None,
    image_size: int,
    face_align: Any,
) -> np.ndarray | None:
    aligned: np.ndarray | None = None
    if landmarks and len(landmarks) >= 5:
        try:
            aligned = face_align.norm_crop(
                image_bgr,
                landmark=np.asarray(landmarks[:5], dtype=np.float32),
                image_size=int(image_size),
            )
        except Exception:
            aligned = None
    if aligned is None:
        aligned = _crop_and_resize(image_bgr, bbox or [], int(image_size))
    return aligned


def _quality_fail_reason(reasons: list[str]) -> str | None:
    return "|".join(reasons) if reasons else None


def _alignment_helped_quality(
    original_quality_reasons: list[str],
    aligned_quality_reasons: list[str],
    original_num_faces: int,
    aligned_num_faces: int,
) -> bool:
    original_pass = original_num_faces == 1 and not original_quality_reasons
    aligned_pass = aligned_num_faces == 1 and not aligned_quality_reasons
    if aligned_pass and not original_pass:
        return True
    return len(aligned_quality_reasons) < len(original_quality_reasons)


def _resolve_insightface_model_dir(model_name: str, model_root: Path) -> Path:
    candidate_dirs = [
        model_root / "models" / model_name,
        model_root / model_name,
    ]
    for candidate in candidate_dirs:
        if candidate.exists() and candidate.is_dir():
            return candidate

    expected = candidate_dirs[0]
    raise FileNotFoundError(
        "InsightFace model bundle was not found. "
        f"Expected local files under {expected}. "
        f"Copy the '{model_name}' bundle to that directory before running preprocessing."
    )


def _validate_insightface_model_dir(
    model_name: str,
    model_root: Path,
    required_model_files: list[str] | None,
) -> Path:
    model_dir = _resolve_insightface_model_dir(model_name, model_root)
    required_patterns = required_model_files or ["*.onnx"]
    missing_patterns = [
        pattern for pattern in required_patterns if not list(model_dir.glob(pattern))
    ]
    if missing_patterns:
        raise FileNotFoundError(
            "InsightFace model bundle is incomplete. "
            f"Missing files matching {missing_patterns} under {model_dir}. "
            f"Copy the full '{model_name}' bundle to the local model root before running preprocessing."
        )
    return model_dir


def _probe_cuda_runtime_libraries() -> tuple[dict[str, dict[str, Any]], list[str]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    for library_name in CUDA_RUNTIME_LIBRARIES:
        loaded = False
        load_error = None
        try:
            ctypes.CDLL(library_name)
            loaded = True
        except OSError as exc:
            load_error = str(exc)

        diagnostics[library_name] = {
            "loaded": loaded,
            "find_library": ctypes.util.find_library(library_name.replace("lib", "").split(".so")[0]),
            "error": load_error,
        }
        if not loaded:
            missing.append(library_name)

    return diagnostics, missing


def collect_onnxruntime_runtime_diagnostics(
    *,
    preferred_execution_provider: str,
    fallback_execution_provider: str,
    device_id: int,
    strict_gpu_mode: bool,
) -> tuple[list[str], ProviderDiagnostics]:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnxruntime is not available in the active Python environment.") from exc

    preload_supported = hasattr(ort, "preload_dlls")
    preload_success = False
    preload_error = None
    if preload_supported:
        try:
            ort.preload_dlls()
            preload_success = True
        except Exception as exc:  # pragma: no cover - depends on ort build
            preload_error = str(exc)

    cuda_runtime_libraries, missing_cuda_runtime_libraries = _probe_cuda_runtime_libraries()
    available_providers = list(ort.get_available_providers())

    preferred = preferred_execution_provider.strip() if preferred_execution_provider else ""
    fallback = fallback_execution_provider.strip() if fallback_execution_provider else "CPUExecutionProvider"

    fallback_reason = None
    if preferred and preferred in available_providers:
        requested_providers = [preferred]
        if fallback not in requested_providers:
            requested_providers.append(fallback)
        selected_provider = preferred
        fallback_to_cpu = preferred == "CPUExecutionProvider"
    else:
        selected_provider = fallback if fallback in available_providers else "CPUExecutionProvider"
        requested_providers = [selected_provider]
        fallback_to_cpu = selected_provider == "CPUExecutionProvider"
        if preferred:
            fallback_reason = (
                f"Preferred execution provider '{preferred}' is unavailable. "
                f"Available providers: {available_providers}."
            )
        else:
            fallback_reason = "No preferred execution provider was configured."

    diagnostics = ProviderDiagnostics(
        onnxruntime_version=str(getattr(ort, "__version__", "unknown")),
        preload_dlls_supported=preload_supported,
        preload_dlls_success=preload_success,
        preload_dlls_error=preload_error,
        cuda_runtime_libraries=cuda_runtime_libraries,
        missing_cuda_runtime_libraries=missing_cuda_runtime_libraries,
        available_providers=available_providers,
        preferred_execution_provider=preferred or fallback,
        requested_providers=requested_providers,
        selected_execution_provider=selected_provider,
        fallback_to_cpu=fallback_to_cpu,
        fallback_reason=fallback_reason,
        device_id=int(device_id),
        actual_model_providers={},
        strict_gpu_mode=bool(strict_gpu_mode),
    )

    cuda_requested = preferred == "CUDAExecutionProvider"
    return requested_providers, diagnostics


def resolve_execution_providers(
    preferred_execution_provider: str,
    fallback_execution_provider: str,
) -> tuple[list[str], ProviderDiagnostics]:
    requested_providers, diagnostics = collect_onnxruntime_runtime_diagnostics(
        preferred_execution_provider=preferred_execution_provider,
        fallback_execution_provider=fallback_execution_provider,
        device_id=0,
        strict_gpu_mode=False,
    )
    return requested_providers, diagnostics


def _extract_model_provider_map(app: Any) -> dict[str, list[str]]:
    provider_map: dict[str, list[str]] = {}
    for name, model in getattr(app, "models", {}).items():
        providers: list[str] = []
        session = getattr(model, "session", None)
        if session is not None and hasattr(session, "get_providers"):
            providers = list(session.get_providers())
        provider_map[str(name)] = providers
    return provider_map


def _finalize_provider_diagnostics(diagnostics: ProviderDiagnostics) -> ProviderDiagnostics:
    actual_flat = [
        provider
        for providers in diagnostics.actual_model_providers.values()
        for provider in providers
    ]
    if (
        diagnostics.selected_execution_provider != "CPUExecutionProvider"
        and actual_flat
        and all(provider == "CPUExecutionProvider" for provider in actual_flat)
    ):
        diagnostics.fallback_to_cpu = True
        diagnostics.fallback_reason = (
            diagnostics.fallback_reason
            or "All loaded InsightFace models resolved to CPUExecutionProvider. "
            "This usually means the preferred GPU provider is installed but its CUDA runtime dependencies are unavailable."
        )
        diagnostics.selected_execution_provider = "CPUExecutionProvider"
    return diagnostics


def run_gpu_runtime_check(
    *,
    test_model_path: Path,
    preferred_execution_provider: str,
    fallback_execution_provider: str,
    device_id: int,
    strict_gpu_mode: bool,
) -> dict[str, Any]:
    requested_providers, diagnostics = collect_onnxruntime_runtime_diagnostics(
        preferred_execution_provider=preferred_execution_provider,
        fallback_execution_provider=fallback_execution_provider,
        device_id=device_id,
        strict_gpu_mode=strict_gpu_mode,
    )
    payload = asdict(diagnostics)
    payload["test_model_path"] = str(test_model_path)
    payload["session_init_ok"] = False
    payload["session_providers"] = []
    payload["gpu_provider_active"] = False
    payload["error"] = None

    if not test_model_path.exists():
        payload["error"] = f"Test model file not found: {test_model_path}"
        return payload

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(test_model_path), providers=requested_providers)
        session_providers = list(session.get_providers())
        payload["session_init_ok"] = True
        payload["session_providers"] = session_providers
        payload["gpu_provider_active"] = "CUDAExecutionProvider" in session_providers
        if session_providers:
            payload["selected_execution_provider"] = session_providers[0]
        if not payload["gpu_provider_active"] and preferred_execution_provider == "CUDAExecutionProvider":
            payload["fallback_to_cpu"] = True
            payload["fallback_reason"] = (
                payload.get("fallback_reason")
                or "CUDAExecutionProvider was requested but ONNX Runtime session used CPUExecutionProvider."
            )
    except Exception as exc:
        payload["error"] = str(exc)
        return payload

    return payload


class InsightFaceDetector:
    def __init__(
        self,
        model_name: str,
        model_root: Path,
        providers: list[str],
        ctx_id: int,
        det_size: list[int],
        device_id: int,
        preferred_execution_provider: str,
        fallback_execution_provider: str,
        strict_gpu_mode: bool,
        allowed_modules: list[str] | None = None,
        required_model_files: list[str] | None = None,
        fallback_detector_enabled: bool = False,
        fallback_detector_scale_factor: float = 1.1,
        fallback_detector_min_neighbors: int = 3,
        fallback_detector_min_size: int = 30,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
            from insightface.utils import face_align
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "InsightFace is not available in the active Python environment."
            ) from exc

        self.model_dir = _validate_insightface_model_dir(
            model_name=model_name,
            model_root=model_root,
            required_model_files=required_model_files,
        )
        self.requested_providers, self.provider_diagnostics = collect_onnxruntime_runtime_diagnostics(
            preferred_execution_provider=preferred_execution_provider,
            fallback_execution_provider=fallback_execution_provider,
            device_id=device_id,
            strict_gpu_mode=strict_gpu_mode,
        )
        self._face_align = face_align
        self._fallback_detector_enabled = fallback_detector_enabled
        self._fallback_scale_factor = float(fallback_detector_scale_factor)
        self._fallback_min_neighbors = int(fallback_detector_min_neighbors)
        self._fallback_min_size = int(fallback_detector_min_size)
        self._haar = (
            cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if fallback_detector_enabled
            else None
        )
        try:
            self._app = FaceAnalysis(
                name=model_name,
                root=str(model_root),
                providers=self.requested_providers or providers,
                allowed_modules=allowed_modules,
            )
            self._app.prepare(ctx_id=ctx_id, det_size=tuple(det_size))
            self.provider_diagnostics.actual_model_providers = _extract_model_provider_map(self._app)
            self.provider_diagnostics = _finalize_provider_diagnostics(self.provider_diagnostics)
            self._landmarker = self._app.models.get("landmark_2d_106")
            if (
                strict_gpu_mode
                and self.provider_diagnostics.preferred_execution_provider == "CUDAExecutionProvider"
                and self.provider_diagnostics.selected_execution_provider != "CUDAExecutionProvider"
            ):
                missing = self.provider_diagnostics.missing_cuda_runtime_libraries
                raise RuntimeError(
                    "Strict GPU mode is enabled, but CUDAExecutionProvider did not initialize for InsightFace models. "
                    f"Selected provider: {self.provider_diagnostics.selected_execution_provider}. "
                    f"Missing CUDA libraries: {missing}."
                )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize InsightFace FaceAnalysis with local models. "
                f"Model name: {model_name}. "
                f"Model root: {model_root}. "
                f"Resolved bundle dir: {self.model_dir}. "
                f"Requested providers: {self.requested_providers}. "
                f"Provider diagnostics: {self.provider_diagnostics}. "
                "Verify that the local model bundle is complete and compatible with the installed insightface package."
            ) from exc

    def diagnostics_payload(self) -> dict[str, Any]:
        return asdict(self.provider_diagnostics)

    def _fallback_detect(self, image_bgr: np.ndarray, target_image_size: list[int]) -> DetectedFace | None:
        if not self._fallback_detector_enabled or self._haar is None or self._landmarker is None:
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self._haar.detectMultiScale(
            gray,
            scaleFactor=self._fallback_scale_factor,
            minNeighbors=self._fallback_min_neighbors,
            minSize=(self._fallback_min_size, self._fallback_min_size),
        )
        bbox_array = _largest_bbox(boxes)
        if bbox_array is None:
            return None

        face = Face(bbox=bbox_array, det_score=1.0)
        dense = self._landmarker.get(image_bgr, face)
        dense_points = np.asarray(dense, dtype=np.float32)
        bbox = [float(value) for value in bbox_array.tolist()]
        landmarks_five = _derive_five_point_landmarks(dense_points.tolist(), bbox)

        aligned: np.ndarray | None = None
        try:
            aligned = self._face_align.norm_crop(
                image_bgr,
                landmark=np.asarray(landmarks_five, dtype=np.float32),
                image_size=int(target_image_size[0]),
            )
        except Exception:
            aligned = _crop_and_resize(image_bgr, bbox, int(target_image_size[0]))
        yaw, pitch, roll = estimate_pose_from_landmarks(image_bgr.shape, landmarks_five)

        return DetectedFace(
            bbox=bbox,
            landmarks=landmarks_five,
            landmarks_dense=[[float(x), float(y)] for x, y in dense_points.tolist()],
            det_score=1.0,
            aligned_image=aligned,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            occlusion_score=None,
            detection_backend="opencv_haar+insightface_lmk106",
        )

    def detect(self, image_bgr: np.ndarray, target_image_size: list[int]) -> list[DetectedFace]:
        faces = self._app.get(image_bgr)
        if not faces:
            fallback_face = self._fallback_detect(image_bgr, target_image_size)
            return [fallback_face] if fallback_face is not None else []

        detections: list[DetectedFace] = []

        for face in faces:
            bbox = [float(value) for value in np.asarray(face.bbox).tolist()]
            detector_landmarks = (
                np.asarray(face.kps, dtype=np.float32).tolist()
                if getattr(face, "kps", None) is not None
                else []
            )
            dense_landmarks_attr = getattr(face, "landmark_2d_106", None)
            dense_landmarks = (
                np.asarray(dense_landmarks_attr, dtype=np.float32).tolist()
                if dense_landmarks_attr is not None
                else detector_landmarks
            )
            landmarks = (
                _derive_five_point_landmarks(
                    dense_landmarks,
                    bbox,
                    detector_landmarks=detector_landmarks,
                )
                if dense_landmarks
                else []
            )
            aligned: np.ndarray | None = None
            if len(landmarks) >= 5:
                try:
                    aligned = self._face_align.norm_crop(
                        image_bgr,
                        landmark=np.asarray(landmarks, dtype=np.float32),
                        image_size=int(target_image_size[0]),
                    )
                except Exception:
                    aligned = None
            yaw, pitch, roll = estimate_pose_from_landmarks(image_bgr.shape, landmarks)

            detections.append(
                DetectedFace(
                    bbox=bbox,
                    landmarks=[[float(x), float(y)] for x, y in landmarks],
                    landmarks_dense=(
                        [[float(x), float(y)] for x, y in dense_landmarks]
                        if dense_landmarks
                        else None
                    ),
                    det_score=float(face.det_score) if getattr(face, "det_score", None) is not None else None,
                    aligned_image=aligned,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    occlusion_score=_extract_occlusion_score(face),
                    detection_backend="insightface",
                )
            )

        return detections


def select_detection_with_auto_alignment(
    *,
    detector: InsightFaceDetector,
    image_bgr: np.ndarray,
    target_image_size: list[int],
    quality_gate: dict[str, Any],
    auto_alignment_config: dict[str, Any],
    quality_evaluator: Any,
) -> tuple[list[DetectedFace], DetectedFace | None, list[str], AutoAlignmentOutcome, np.ndarray]:
    original_detections = detector.detect(image_bgr=image_bgr, target_image_size=target_image_size)
    original_num_faces = len(original_detections)
    original_primary = original_detections[0] if original_num_faces == 1 else None
    original_quality_reasons = quality_evaluator(original_primary, original_num_faces, quality_gate)
    original_pose = {
        "yaw": original_primary.yaw if original_primary is not None else None,
        "pitch": original_primary.pitch if original_primary is not None else None,
        "roll": original_primary.roll if original_primary is not None else None,
    }

    outcome = AutoAlignmentOutcome(
        attempted=False,
        applied=False,
        improved_quality=False,
        source=None,
        status="not_enabled" if not bool(auto_alignment_config.get("enabled", False)) else "not_needed",
        rotation_deg=None,
        transform=None,
        original_pose=original_pose,
        aligned_pose=original_pose,
        original_quality_pass=(original_num_faces == 1 and not original_quality_reasons) if original_primary is not None else None,
        original_quality_fail_reason=_quality_fail_reason(original_quality_reasons),
        aligned_quality_pass=(original_num_faces == 1 and not original_quality_reasons) if original_primary is not None else None,
        aligned_quality_fail_reason=_quality_fail_reason(original_quality_reasons),
    )

    if not bool(auto_alignment_config.get("enabled", False)):
        return original_detections, original_primary, original_quality_reasons, outcome, image_bgr
    if original_primary is None:
        outcome.status = "skipped_no_single_face"
        return original_detections, original_primary, original_quality_reasons, outcome, image_bgr

    rotation_deg, resolved_source = compute_auto_alignment_rotation(original_primary, auto_alignment_config)
    outcome.source = resolved_source
    if rotation_deg is None:
        outcome.status = "skipped_small_tilt" if resolved_source is not None else "skipped_missing_landmarks"
        return original_detections, original_primary, original_quality_reasons, outcome, image_bgr

    outcome.attempted = True
    outcome.rotation_deg = rotation_deg
    rotated_image, transform = _rotate_image_with_canvas(image_bgr, rotation_deg)
    outcome.transform = {
        "type": "affine_rotation",
        "matrix": [[float(value) for value in row] for row in transform.tolist()],
        "output_size": [int(rotated_image.shape[1]), int(rotated_image.shape[0])],
    }

    try:
        if bool(auto_alignment_config.get("reprocess_after_alignment", True)):
            aligned_detections = detector.detect(image_bgr=rotated_image, target_image_size=target_image_size)
        else:
            transformed_landmarks = _apply_affine_to_points(original_primary.landmarks, transform) or []
            transformed_dense = _apply_affine_to_points(original_primary.landmarks_dense, transform)
            transformed_bbox = _transform_bbox(original_primary.bbox, transform) or original_primary.bbox
            yaw, pitch, roll = estimate_pose_from_landmarks(rotated_image.shape, transformed_landmarks)
            aligned_detections = [
                DetectedFace(
                    bbox=transformed_bbox,
                    landmarks=transformed_landmarks,
                    landmarks_dense=transformed_dense,
                    det_score=original_primary.det_score,
                    aligned_image=_build_aligned_image(
                        image_bgr=rotated_image,
                        bbox=transformed_bbox,
                        landmarks=transformed_landmarks,
                        image_size=int(target_image_size[0]),
                        face_align=detector._face_align,
                    ),
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    occlusion_score=original_primary.occlusion_score,
                    detection_backend=f"{original_primary.detection_backend}+auto_aligned_transform",
                )
            ]
    except Exception as exc:
        outcome.status = f"failed:{exc}"
        return original_detections, original_primary, original_quality_reasons, outcome, image_bgr

    aligned_num_faces = len(aligned_detections)
    aligned_primary = aligned_detections[0] if aligned_num_faces == 1 else None
    aligned_quality_reasons = quality_evaluator(aligned_primary, aligned_num_faces, quality_gate)
    outcome.aligned_pose = {
        "yaw": aligned_primary.yaw if aligned_primary is not None else None,
        "pitch": aligned_primary.pitch if aligned_primary is not None else None,
        "roll": aligned_primary.roll if aligned_primary is not None else None,
    }
    outcome.aligned_quality_pass = (aligned_num_faces == 1 and not aligned_quality_reasons) if aligned_primary is not None else None
    outcome.aligned_quality_fail_reason = _quality_fail_reason(aligned_quality_reasons)
    outcome.improved_quality = _alignment_helped_quality(
        original_quality_reasons=original_quality_reasons,
        aligned_quality_reasons=aligned_quality_reasons,
        original_num_faces=original_num_faces,
        aligned_num_faces=aligned_num_faces,
    )

    original_pass = original_num_faces == 1 and not original_quality_reasons
    aligned_pass = aligned_num_faces == 1 and not aligned_quality_reasons
    choose_aligned = False

    if aligned_primary is not None:
        if aligned_pass and not original_pass:
            choose_aligned = True
        elif aligned_num_faces == 1 and original_num_faces != 1:
            choose_aligned = True
        elif len(aligned_quality_reasons) < len(original_quality_reasons):
            choose_aligned = True
        elif len(aligned_quality_reasons) == len(original_quality_reasons):
            original_roll = abs(float(original_primary.roll)) if original_primary.roll is not None else float("inf")
            aligned_roll = abs(float(aligned_primary.roll)) if aligned_primary.roll is not None else float("inf")
            original_score = float(original_primary.det_score) if original_primary.det_score is not None else 0.0
            aligned_score = float(aligned_primary.det_score) if aligned_primary.det_score is not None else 0.0
            if aligned_roll + 1.0 < original_roll and aligned_score + 0.05 >= original_score:
                choose_aligned = True

    if choose_aligned:
        outcome.applied = True
        outcome.status = "applied"
        return aligned_detections, aligned_primary, aligned_quality_reasons, outcome, rotated_image

    outcome.status = "not_improved"
    return original_detections, original_primary, original_quality_reasons, outcome, image_bgr
