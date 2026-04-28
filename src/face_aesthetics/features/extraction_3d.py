from __future__ import annotations

from typing import Any

import numpy as np


GEOMETRY_3D_COLUMNS = [
    "geom3d_eye_distance",
    "geom3d_mouth_width",
    "geom3d_eye_to_nose",
    "geom3d_nose_to_mouth",
    "geom3d_face_height_proxy",
    "geom3d_depth_range",
    "geom3d_nose_depth_offset",
    "geom3d_left_right_depth_asymmetry",
    "geom3d_eye_mouth_depth_delta",
    "geom3d_face_aspect_ratio",
    "geom3d_eye_to_nose_ratio",
    "geom3d_nose_to_mouth_ratio",
]


def empty_3d_feature_dict() -> dict[str, float | None]:
    return {column: None for column in GEOMETRY_3D_COLUMNS}


def _distance_3d(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def extract_geometry_3d_features(landmarks_3d: list[list[float]] | None) -> dict[str, float | None]:
    if not landmarks_3d or len(landmarks_3d) < 5:
        return empty_3d_feature_dict()

    points = np.asarray(landmarks_3d[:5], dtype=np.float64)
    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0

    eye_distance = _distance_3d(left_eye, right_eye)
    mouth_width = _distance_3d(left_mouth, right_mouth)
    eye_to_nose = _distance_3d(eye_center, nose)
    nose_to_mouth = _distance_3d(nose, mouth_center)
    face_height_proxy = _distance_3d(eye_center, mouth_center)
    z_values = points[:, 2]
    depth_range = float(z_values.max() - z_values.min())
    nose_depth_offset = float(nose[2] - np.mean([left_eye[2], right_eye[2], left_mouth[2], right_mouth[2]]))
    left_depth_mean = float(np.mean([left_eye[2], left_mouth[2]]))
    right_depth_mean = float(np.mean([right_eye[2], right_mouth[2]]))
    left_right_depth_asymmetry = float(abs(left_depth_mean - right_depth_mean))
    eye_mouth_depth_delta = float(abs(np.mean([left_eye[2], right_eye[2]]) - np.mean([left_mouth[2], right_mouth[2]])))

    return {
        "geom3d_eye_distance": eye_distance,
        "geom3d_mouth_width": mouth_width,
        "geom3d_eye_to_nose": eye_to_nose,
        "geom3d_nose_to_mouth": nose_to_mouth,
        "geom3d_face_height_proxy": face_height_proxy,
        "geom3d_depth_range": depth_range,
        "geom3d_nose_depth_offset": nose_depth_offset,
        "geom3d_left_right_depth_asymmetry": left_right_depth_asymmetry,
        "geom3d_eye_mouth_depth_delta": eye_mouth_depth_delta,
        "geom3d_face_aspect_ratio": _safe_divide(eye_distance, face_height_proxy),
        "geom3d_eye_to_nose_ratio": _safe_divide(eye_to_nose, face_height_proxy),
        "geom3d_nose_to_mouth_ratio": _safe_divide(nose_to_mouth, face_height_proxy),
    }


def validate_3d_feature_values(feature_values: dict[str, Any], *, image_id: str) -> None:
    issues: list[str] = []
    for key, value in feature_values.items():
        if value is None:
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            issues.append(f"{key}=non_finite")
        elif key not in {"geom3d_nose_depth_offset"} and numeric < 0.0:
            issues.append(f"{key}=negative")

    if issues:
        raise ValueError(f"Invalid 3D feature values for image_id={image_id}: {', '.join(sorted(issues))}")
