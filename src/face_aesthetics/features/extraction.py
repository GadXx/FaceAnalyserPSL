from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from src.face_aesthetics.landmarks.insightface_106 import (
    ALARE_LEFT_INDEX,
    ALARE_RIGHT_INDEX,
    CONTOUR_INDICES,
    LANDMARK_COUNT as INSIGHTFACE_106_LANDMARK_COUNT,
    LEFT_BROW_INDICES,
    LEFT_CONTOUR_INDICES,
    LEFT_EYE_INDICES,
    LOWER_LIP_ARC_INDICES,
    LOWER_LIP_CENTER_BOTTOM_INDEX,
    MOUTH_INDICES,
    MOUTH_LEFT_CORNER_INDEX,
    MOUTH_RIGHT_CORNER_INDEX,
    MOUTH_INNER_ZONE_INDICES,
    NOSE_BASE_LEFT_INDEX,
    NOSE_BASE_RIGHT_INDEX,
    NOSE_BRIDGE_INDICES,
    NOSE_INDICES,
    NOSE_TIP_INDEX,
    RIGHT_BROW_INDICES,
    RIGHT_CONTOUR_INDICES,
    RIGHT_EYE_INDICES,
    SUBNASALE_CENTER_INDEX,
    SUBNASALE_LEFT_INDEX,
    SUBNASALE_RIGHT_INDEX,
    UPPER_LIP_ARC_INDICES,
    UPPER_LIP_CENTER_TOP_INDEX,
    has_106_landmarks,
    point_for_index,
    points_for_indices,
    points_for_region,
    region_name_for_index,
)


RAW_MEASUREMENT_COLUMNS = [
    "raw_face_width_px",
    "raw_face_height_px",
    "raw_eye_distance_px",
    "raw_mouth_width_px",
    "raw_eye_to_nose_px",
    "raw_nose_to_mouth_px",
    "raw_eye_to_mouth_px",
    "raw_left_eye_to_nose_px",
    "raw_right_eye_to_nose_px",
    "raw_left_mouth_to_nose_px",
    "raw_right_mouth_to_nose_px",
    "raw_upper_face_height_px",
    "raw_lower_face_height_px",
    "raw_forehead_height_proxy_px",
    "raw_jaw_height_proxy_px",
    "raw_nose_center_offset_px",
    "raw_mouth_center_offset_px",
    "brow_to_eye_center_height",
    "eye_center_to_nose_base_height",
    "nose_base_to_mouth_center_height",
    "mouth_center_to_chin_height",
    "brow_to_nose_base_height",
    "nose_base_to_chin_height",
    "eye_center_to_chin_height",
    "forehead_proxy_to_brow_height",
    "upper_third_proxy",
    "middle_third_proxy",
    "lower_third_proxy",
    "intercanthal_distance",
    "outercanthal_distance",
    "left_eye_width",
    "right_eye_width",
    "mean_eye_width",
    "left_eye_height",
    "right_eye_height",
    "mean_eye_height",
    "nose_width",
    "nose_base_width",
    "philtrum_width_proxy",
    "chin_width_proxy",
    "jaw_width_proxy",
    "cheekbone_width_proxy",
    "left_right_eye_y_diff",
    "left_right_eye_width_diff",
    "left_right_eye_height_diff",
    "left_right_brow_height_diff",
    "left_right_mouth_corner_y_diff",
    "left_right_mouth_to_nose_diff",
    "left_right_eye_to_nose_diff",
    "nose_tip_midline_offset",
    "mouth_center_midline_offset",
    "chin_center_midline_offset",
    "philtrum_midline_offset_proxy",
    "left_brow_to_eye_distance",
    "right_brow_to_eye_distance",
    "mean_brow_to_eye_distance",
    "eye_area_proxy_left",
    "eye_area_proxy_right",
    "eye_area_proxy_mean",
    "nose_length",
    "nose_tip_to_upper_lip_distance",
    "philtrum_height",
    "upper_lip_height",
    "lower_lip_height",
    "chin_height",
]

SEMANTIC_UNRELIABLE_RAW_FEATURES = {
    "raw_mouth_width_px",
    "raw_nose_to_mouth_px",
    "raw_eye_to_mouth_px",
    "raw_left_mouth_to_nose_px",
    "raw_right_mouth_to_nose_px",
    "raw_lower_face_height_px",
    "raw_jaw_height_proxy_px",
    "raw_mouth_center_offset_px",
    "nose_base_to_mouth_center_height",
    "mouth_center_to_chin_height",
    "left_right_mouth_corner_y_diff",
    "left_right_mouth_to_nose_diff",
    "mouth_center_midline_offset",
    "philtrum_midline_offset_proxy",
    "nose_tip_to_upper_lip_distance",
    "philtrum_height",
    "upper_lip_height",
    "lower_lip_height",
    "chin_height",
    "chin_width_proxy",
    "jaw_width_proxy",
    "cheekbone_width_proxy",
}

SEMANTIC_UNRELIABLE_ANGLE_FEATURES = {
    "angle_mouth_line_deg",
    "angle_eye_center_to_mouth_center_deg",
    "angle_nose_to_mouth_center_deg",
    "nose_axis_angle_deg",
    "philtrum_axis_angle_deg",
    "chin_axis_angle_deg",
    "jaw_line_left_angle_proxy",
    "jaw_line_right_angle_proxy",
    "jaw_angle_asymmetry",
    "face_centerline_vs_nose_angle",
    "eye_line_vs_mouth_line_diff",
    "nose_to_chin_axis_diff",
    "lower_face_tilt_proxy",
}

NORMALIZED_MEASUREMENT_COLUMNS = [
    "ratio_face_aspect",
    "ratio_eye_distance_to_face_width",
    "ratio_mouth_width_to_face_width",
    "ratio_eye_to_nose_to_face_height",
    "ratio_nose_to_mouth_to_face_height",
    "ratio_eye_to_mouth_to_face_height",
    "ratio_eye_distance_to_mouth_width",
    "ratio_upper_face_height_to_face_height",
    "ratio_lower_face_height_to_face_height",
    "ratio_forehead_height_proxy_to_face_height",
    "ratio_jaw_height_proxy_to_face_height",
    "ratio_nose_center_offset_to_face_width",
    "ratio_mouth_center_offset_to_face_width",
    "ratio_upper_third_to_middle_third",
    "ratio_middle_third_to_lower_third",
    "ratio_nose_base_to_mouth_to_lower_third",
    "ratio_mouth_to_chin_to_lower_third",
    "thirds_balance_index",
    "ratio_intercanthal_to_eye_width",
    "ratio_outercanthal_to_face_width",
    "ratio_nose_width_to_face_width",
    "ratio_mouth_width_to_nose_width",
    "ratio_chin_width_to_jaw_width",
    "ratio_cheekbone_width_to_jaw_width",
    "ratio_eye_height_to_eye_width",
    "ratio_eye_spacing_to_eye_width",
    "ratio_eye_spacing_to_face_width",
    "ratio_nose_length_to_face_height",
    "ratio_philtrum_to_lower_face_height",
    "ratio_upper_lip_to_lower_lip",
    "ratio_mouth_to_chin",
    "ratio_nose_to_philtrum",
    "ratio_nose_width_to_mouth_width",
    "lip_thickness_ratio",
    "chin_taper_ratio",
]

SYMMETRY_FEATURE_COLUMNS = [
    "sym_eye_nose_delta_ratio",
    "sym_mouth_nose_delta_ratio",
    "sym_horizontal_center_offset_ratio",
    "symmetry_index",
    "brow_symmetry_ratio",
    "midline_alignment_index",
    "upper_face_symmetry_index",
    "lower_face_symmetry_index",
]

ANGLE_FEATURE_COLUMNS = [
    "angle_eye_line_deg",
    "angle_mouth_line_deg",
    "angle_eye_center_to_mouth_center_deg",
    "angle_nose_to_mouth_center_deg",
    "pose_yaw_deg",
    "pose_pitch_deg",
    "pose_roll_deg",
    "canthal_tilt_deg",
    "nose_axis_angle_deg",
    "philtrum_axis_angle_deg",
    "chin_axis_angle_deg",
    "brow_line_angle_deg",
    "jaw_line_left_angle_proxy",
    "jaw_line_right_angle_proxy",
    "jaw_angle_asymmetry",
    "face_centerline_vs_nose_angle",
    "eye_line_vs_mouth_line_diff",
    "nose_to_chin_axis_diff",
    "lower_face_tilt_proxy",
]


NONNEGATIVE_FEATURE_COLUMNS = set(RAW_MEASUREMENT_COLUMNS + NORMALIZED_MEASUREMENT_COLUMNS + SYMMETRY_FEATURE_COLUMNS)
UNIT_INTERVAL_COLUMNS = {
    "symmetry_index",
    "brow_symmetry_ratio",
    "midline_alignment_index",
    "upper_face_symmetry_index",
    "lower_face_symmetry_index",
    "thirds_balance_index",
}

FEATURE_FORMULAS: dict[str, str] = {
    "raw_face_width_px": "bbox_x2 - bbox_x1",
    "raw_face_height_px": "bbox_y2 - bbox_y1",
    "raw_eye_distance_px": "euclidean_distance(left_eye, right_eye)",
    "raw_mouth_width_px": "euclidean_distance(left_mouth, right_mouth)",
    "raw_eye_to_nose_px": "euclidean_distance(eye_center, nose_tip)",
    "raw_nose_to_mouth_px": "euclidean_distance(nose_tip, mouth_center)",
    "raw_eye_to_mouth_px": "euclidean_distance(eye_center, mouth_center)",
    "brow_to_eye_center_height": "abs(eye_center_y - brow_center_y)",
    "eye_center_to_nose_base_height": "abs(nose_base_center_y - eye_center_y)",
    "nose_base_to_mouth_center_height": "abs(mouth_center_y - nose_base_center_y)",
    "mouth_center_to_chin_height": "abs(chin_center_y - mouth_center_y)",
    "brow_to_nose_base_height": "abs(nose_base_center_y - brow_center_y)",
    "nose_base_to_chin_height": "abs(chin_center_y - nose_base_center_y)",
    "eye_center_to_chin_height": "abs(chin_center_y - eye_center_y)",
    "forehead_proxy_to_brow_height": "abs(brow_center_y - bbox_top)",
    "upper_third_proxy": "brow_to_nose_base_height",
    "middle_third_proxy": "eye_center_to_nose_base_height + abs(brow_center_y - eye_center_y)",
    "lower_third_proxy": "nose_base_to_chin_height",
    "intercanthal_distance": "distance(left_eye_inner_corner, right_eye_inner_corner)",
    "outercanthal_distance": "distance(left_eye_outer_corner, right_eye_outer_corner)",
    "left_eye_width": "distance(left_eye_outer_corner, left_eye_inner_corner)",
    "right_eye_width": "distance(right_eye_inner_corner, right_eye_outer_corner)",
    "left_eye_height": "distance(left_eye_top, left_eye_bottom)",
    "right_eye_height": "distance(right_eye_top, right_eye_bottom)",
    "nose_width": "distance(nose_mid_left, nose_mid_right)",
    "nose_base_width": "distance(nose_base_left, nose_base_right)",
    "philtrum_width_proxy": "x_span(upper_lip_band_points)",
    "chin_width_proxy": "x_span(chin_band_points)",
    "jaw_width_proxy": "distance(jaw_left, jaw_right)",
    "cheekbone_width_proxy": "distance(cheekbone_left, cheekbone_right)",
    "nose_tip_midline_offset": "abs(nose_tip_x - face_midline_x)",
    "mouth_center_midline_offset": "abs(mouth_center_x - face_midline_x)",
    "chin_center_midline_offset": "abs(chin_center_x - face_midline_x)",
    "philtrum_midline_offset_proxy": "abs(philtrum_top_x - face_midline_x)",
    "left_brow_to_eye_distance": "abs(left_eye_center_y - left_brow_y)",
    "right_brow_to_eye_distance": "abs(right_eye_center_y - right_brow_y)",
    "eye_area_proxy_left": "left_eye_width * left_eye_height",
    "eye_area_proxy_right": "right_eye_width * right_eye_height",
    "nose_length": "distance(nose_base_center, nose_tip)",
    "nose_tip_to_upper_lip_distance": "distance(nose_tip, upper_lip_top)",
    "philtrum_height": "abs(upper_lip_top_y - nose_base_center_y)",
    "upper_lip_height": "abs(mouth_center_y - upper_lip_top_y)",
    "lower_lip_height": "abs(lower_lip_bottom_y - mouth_center_y)",
    "chin_height": "abs(chin_center_y - lower_lip_bottom_y)",
    "ratio_face_aspect": "raw_face_width_px / raw_face_height_px",
    "ratio_upper_third_to_middle_third": "upper_third_proxy / middle_third_proxy",
    "ratio_middle_third_to_lower_third": "middle_third_proxy / lower_third_proxy",
    "ratio_nose_base_to_mouth_to_lower_third": "nose_base_to_mouth_center_height / lower_third_proxy",
    "ratio_mouth_to_chin_to_lower_third": "mouth_center_to_chin_height / lower_third_proxy",
    "thirds_balance_index": "1 - normalized dispersion of upper/middle/lower thirds",
    "ratio_intercanthal_to_eye_width": "intercanthal_distance / mean_eye_width",
    "ratio_outercanthal_to_face_width": "outercanthal_distance / raw_face_width_px",
    "ratio_nose_width_to_face_width": "nose_width / raw_face_width_px",
    "ratio_mouth_width_to_nose_width": "raw_mouth_width_px / nose_width",
    "ratio_chin_width_to_jaw_width": "chin_width_proxy / jaw_width_proxy",
    "ratio_cheekbone_width_to_jaw_width": "cheekbone_width_proxy / jaw_width_proxy",
    "ratio_eye_height_to_eye_width": "mean_eye_height / mean_eye_width",
    "ratio_eye_spacing_to_eye_width": "intercanthal_distance / mean_eye_width",
    "ratio_eye_spacing_to_face_width": "intercanthal_distance / raw_face_width_px",
    "ratio_nose_length_to_face_height": "nose_length / raw_face_height_px",
    "ratio_philtrum_to_lower_face_height": "philtrum_height / lower_third_proxy",
    "ratio_upper_lip_to_lower_lip": "upper_lip_height / lower_lip_height",
    "ratio_mouth_to_chin": "raw_mouth_width_px / mouth_center_to_chin_height",
    "ratio_nose_to_philtrum": "nose_length / philtrum_height",
    "ratio_nose_width_to_mouth_width": "nose_width / raw_mouth_width_px",
    "lip_thickness_ratio": "(upper_lip_height + lower_lip_height) / raw_mouth_width_px",
    "chin_taper_ratio": "chin_width_proxy / jaw_width_proxy",
    "sym_eye_nose_delta_ratio": "abs(raw_left_eye_to_nose_px - raw_right_eye_to_nose_px) / raw_eye_distance_px",
    "sym_mouth_nose_delta_ratio": "abs(raw_left_mouth_to_nose_px - raw_right_mouth_to_nose_px) / raw_mouth_width_px",
    "sym_horizontal_center_offset_ratio": "mean(nose_tip_midline_offset, mouth_center_midline_offset) / raw_face_width_px",
    "symmetry_index": "mean(1 - clipped asymmetry components)",
    "brow_symmetry_ratio": "1 - abs(left_brow_to_eye_distance - right_brow_to_eye_distance) / mean_brow_to_eye_distance",
    "midline_alignment_index": "1 - mean(midline offsets normalized by face width)",
    "upper_face_symmetry_index": "mean(1 - normalized upper-face asymmetry components)",
    "lower_face_symmetry_index": "mean(1 - normalized lower-face asymmetry components)",
    "angle_eye_line_deg": "atan2(dy, dx) between left_eye and right_eye",
    "angle_mouth_line_deg": "atan2(dy, dx) between left_mouth and right_mouth",
    "angle_eye_center_to_mouth_center_deg": "atan2(dy, dx) between eye_center and mouth_center",
    "angle_nose_to_mouth_center_deg": "atan2(dy, dx) between nose_tip and mouth_center",
    "canthal_tilt_deg": "angle_eye_line_deg",
    "nose_axis_angle_deg": "atan2(dy, dx) between nose_base_center and nose_tip",
    "philtrum_axis_angle_deg": "atan2(dy, dx) between nose_base_center and mouth_center",
    "chin_axis_angle_deg": "atan2(dy, dx) between mouth_center and chin_center",
    "brow_line_angle_deg": "atan2(dy, dx) between left_brow and right_brow",
    "jaw_line_left_angle_proxy": "atan2(dy, dx) between jaw_left and chin_center",
    "jaw_line_right_angle_proxy": "atan2(dy, dx) between chin_center and jaw_right",
    "jaw_angle_asymmetry": "abs(jaw_line_left_angle_proxy - jaw_line_right_angle_proxy)",
    "face_centerline_vs_nose_angle": "abs(90 - abs(nose_axis_angle_deg))",
    "eye_line_vs_mouth_line_diff": "abs(angle_eye_line_deg - angle_mouth_line_deg)",
    "nose_to_chin_axis_diff": "abs(nose_axis_angle_deg - chin_axis_angle_deg)",
    "lower_face_tilt_proxy": "chin_axis_angle_deg",
}


ALIASES = {
    "raw_lower_face_height_px": "mouth_center_to_chin_height",
    "raw_jaw_height_proxy_px": "mouth_center_to_chin_height",
    "raw_nose_center_offset_px": "nose_tip_midline_offset",
    "raw_mouth_center_offset_px": "mouth_center_midline_offset",
    "angle_eye_line_deg": "canthal_tilt_deg",
}

REDUNDANT_FEATURES = {
    "raw_jaw_height_proxy_px",
}


def parse_json_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return json.loads(stripped)
    return value


def _distance(point_a: np.ndarray | None, point_b: np.ndarray | None) -> float | None:
    if point_a is None or point_b is None:
        return None
    return float(np.linalg.norm(point_a - point_b))


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _vector_angle_degrees(point_a: np.ndarray | None, point_b: np.ndarray | None) -> float | None:
    if point_a is None or point_b is None:
        return None
    vector = point_b - point_a
    return float(math.degrees(math.atan2(float(vector[1]), float(vector[0]))))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not present:
        return None
    return float(sum(present) / len(present))


def _abs_diff(value_a: float | None, value_b: float | None) -> float | None:
    if value_a is None or value_b is None:
        return None
    return float(abs(value_a - value_b))


def _point_midpoint(point_a: np.ndarray | None, point_b: np.ndarray | None) -> np.ndarray | None:
    if point_a is None or point_b is None:
        return None
    return (point_a + point_b) / 2.0


def _point_from_xy(x: float | None, y: float | None) -> np.ndarray | None:
    if x is None or y is None:
        return None
    return np.asarray([float(x), float(y)], dtype=np.float64)


def _span(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.max(values) - np.min(values))


def empty_feature_dict(columns: list[str]) -> dict[str, float | None]:
    return {column: None for column in columns}


def _region_points(
    points: np.ndarray | None,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    mask = np.ones((points.shape[0],), dtype=bool)
    if x_min is not None:
        mask &= points[:, 0] >= x_min
    if x_max is not None:
        mask &= points[:, 0] <= x_max
    if y_min is not None:
        mask &= points[:, 1] >= y_min
    if y_max is not None:
        mask &= points[:, 1] <= y_max
    return points[mask]


def _indexed_region_points(
    points: np.ndarray | None,
    *,
    indices: np.ndarray | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if points is None or len(points) == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 2), dtype=np.float64)
    point_indices = np.arange(points.shape[0], dtype=np.int32) if indices is None else np.asarray(indices, dtype=np.int32)
    mask = np.ones((points.shape[0],), dtype=bool)
    if x_min is not None:
        mask &= points[:, 0] >= x_min
    if x_max is not None:
        mask &= points[:, 0] <= x_max
    if y_min is not None:
        mask &= points[:, 1] >= y_min
    if y_max is not None:
        mask &= points[:, 1] <= y_max
    return point_indices[mask], points[mask]


def _extreme_point(points: np.ndarray, *, axis: int, mode: str) -> np.ndarray | None:
    if len(points) == 0:
        return None
    index = int(np.argmin(points[:, axis])) if mode == "min" else int(np.argmax(points[:, axis]))
    return points[index]


def _cluster_center(points: np.ndarray) -> np.ndarray | None:
    if len(points) == 0:
        return None
    return points.mean(axis=0)


def _closest_point_to_x(points: np.ndarray, target_x: float, *, mode_y: str | None = None) -> np.ndarray | None:
    if len(points) == 0:
        return None
    x_scores = np.abs(points[:, 0] - target_x)
    if mode_y == "min":
        y_scores = points[:, 1] - np.min(points[:, 1])
        scores = x_scores + 0.1 * y_scores
    elif mode_y == "max":
        y_scores = np.max(points[:, 1]) - points[:, 1]
        scores = x_scores + 0.1 * y_scores
    else:
        scores = x_scores
    return points[int(np.argmin(scores))]


def _indexed_closest_point_to_x(
    points: np.ndarray,
    indices: np.ndarray,
    target_x: float,
    *,
    mode_y: str | None = None,
) -> tuple[np.ndarray | None, list[int]]:
    if len(points) == 0:
        return None, []
    x_scores = np.abs(points[:, 0] - target_x)
    if mode_y == "min":
        y_scores = points[:, 1] - np.min(points[:, 1])
        scores = x_scores + 0.1 * y_scores
    elif mode_y == "max":
        y_scores = np.max(points[:, 1]) - points[:, 1]
        scores = x_scores + 0.1 * y_scores
    else:
        scores = x_scores
    winner = int(np.argmin(scores))
    return points[winner], [int(indices[winner])]


def _semantic_point_debug_entry(
    *,
    point_name: str,
    point: np.ndarray | None,
    group: str,
    source_type: str,
    source_indices: list[int] | None,
    description: str,
    source_region: str | None = None,
    validation_status: str | None = None,
) -> dict[str, Any] | None:
    if point is None:
        return None
    normalized_indices = [int(value) for value in (source_indices or [])]
    source_index_106 = normalized_indices[0] if len(normalized_indices) == 1 else None
    return {
        "point_name": point_name,
        "group": group,
        "x": float(point[0]),
        "y": float(point[1]),
        "source_type": source_type,
        "source_indices": normalized_indices,
        "source_index_106": source_index_106,
        "source_region": source_region or (
            region_name_for_index(source_index_106) if source_index_106 is not None else None
        ),
        "validation_status": validation_status or ("fallback" if "fallback" in source_type else "validated"),
        "description": description,
    }


def _safe_index_point(points: np.ndarray | None, index: int) -> np.ndarray | None:
    if points is None:
        return None
    if index < 0 or index >= len(points):
        return None
    point = np.asarray(points[index], dtype=np.float64)
    if point.shape != (2,):
        return None
    return point


def _semantic_point_payload(
    *,
    point_name: str,
    point: np.ndarray | None,
    source_type: str,
    source_indices: list[int] | None,
    source_region: str | None,
    validation_status: str,
) -> dict[str, Any] | None:
    if point is None:
        return None
    normalized_indices = [int(value) for value in (source_indices or [])]
    source_index_106 = normalized_indices[0] if len(normalized_indices) == 1 else None
    return {
        "x": float(point[0]),
        "y": float(point[1]),
        "source_index_106": source_index_106,
        "source_indices": normalized_indices,
        "source_region": source_region,
        "source_type": source_type,
        "validation_status": validation_status,
        "point_name": point_name,
    }


def _region_for_source_indices(source_indices: list[int] | None) -> str | None:
    normalized_indices = [int(value) for value in (source_indices or [])]
    if not normalized_indices:
        return None
    regions = {region_name_for_index(index) for index in normalized_indices}
    if len(regions) == 1:
        return next(iter(regions))
    return "mixed"


def _indexed_extreme_point(
    points: np.ndarray,
    indices: np.ndarray,
    *,
    axis: int,
    mode: str,
) -> tuple[np.ndarray | None, list[int]]:
    if len(points) == 0 or len(indices) == 0:
        return None, []
    winner = int(np.argmin(points[:, axis])) if mode == "min" else int(np.argmax(points[:, axis]))
    return points[winner], [int(indices[winner])]


def _make_semantic_meta(
    *,
    point_name: str,
    point: np.ndarray | None,
    group: str,
    source_type: str,
    source_indices: list[int] | None,
    description: str,
    source_region: str | None = None,
    validation_status: str | None = None,
) -> dict[str, Any] | None:
    return _semantic_point_debug_entry(
        point_name=point_name,
        point=point,
        group=group,
        source_type=source_type,
        source_indices=source_indices,
        description=description,
        source_region=source_region or _region_for_source_indices(source_indices),
        validation_status=validation_status,
    )


def _is_valid_mouth_pair(
    *,
    left_mouth: np.ndarray | None,
    right_mouth: np.ndarray | None,
    nose_tip: np.ndarray,
    bbox: list[float],
) -> bool:
    if left_mouth is None or right_mouth is None:
        return False
    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width = max(x2 - x1, 1.0)
    face_height = max(y2 - y1, 1.0)
    face_center_x = (x1 + x2) / 2.0
    mouth_width = float(right_mouth[0] - left_mouth[0])
    if mouth_width <= 0.0:
        return False
    if mouth_width < 0.08 * face_width or mouth_width > 0.70 * face_width:
        return False
    if float(left_mouth[1]) <= float(nose_tip[1]) + 0.01 * face_height:
        return False
    if float(right_mouth[1]) <= float(nose_tip[1]) + 0.01 * face_height:
        return False
    lower_half_top = y1 + 0.52 * face_height
    if float(left_mouth[1]) < lower_half_top or float(right_mouth[1]) < lower_half_top:
        return False
    if (
        abs(float(left_mouth[0] - face_center_x)) > 0.40 * face_width
        or abs(float(right_mouth[0] - face_center_x)) > 0.40 * face_width
    ):
        return False
    return True


def _derive_mouth_geometry_from_106_indices(
    *,
    dense: np.ndarray | None,
    bbox: list[float],
    nose_tip: np.ndarray,
) -> dict[str, Any] | None:
    if dense is None or not has_106_landmarks(dense):
        return None
    left_idx = int(MOUTH_LEFT_CORNER_INDEX)
    right_idx = int(MOUTH_RIGHT_CORNER_INDEX)
    upper_idx = int(UPPER_LIP_CENTER_TOP_INDEX)
    lower_idx = int(LOWER_LIP_CENTER_BOTTOM_INDEX)
    left_mouth = _safe_index_point(dense, left_idx)
    right_mouth = _safe_index_point(dense, right_idx)
    if left_mouth is None or right_mouth is None:
        return None
    mouth_center = (left_mouth + right_mouth) / 2.0
    upper_lip_top = _safe_index_point(dense, upper_idx)
    lower_lip_bottom = _safe_index_point(dense, lower_idx)
    return {
        "left_mouth": left_mouth,
        "right_mouth": right_mouth,
        "mouth_center": mouth_center,
        "upper_lip_top": upper_lip_top,
        "lower_lip_bottom": lower_lip_bottom,
        "source_type": "insightface_106_indices",
        "debug": {
            "left_mouth": _semantic_point_debug_entry(
                point_name="mouth_left",
                point=left_mouth,
                group="mouth",
                source_type="insightface_106_index",
                source_indices=[left_idx],
                source_region="mouth",
                description="Left lip corner from the canonical InsightFace landmark_2d_106 index map.",
            ),
            "right_mouth": _semantic_point_debug_entry(
                point_name="mouth_right",
                point=right_mouth,
                group="mouth",
                source_type="insightface_106_index",
                source_indices=[right_idx],
                source_region="mouth",
                description="Right lip corner from the canonical InsightFace landmark_2d_106 index map.",
            ),
            "mouth_center": _semantic_point_debug_entry(
                point_name="mouth_center",
                point=mouth_center,
                group="mouth",
                source_type="derived",
                source_indices=[left_idx, right_idx],
                source_region="mouth",
                description="Midpoint between indexed mouth corners.",
            ),
            "upper_lip_top": _semantic_point_debug_entry(
                point_name="upper_lip_top",
                point=upper_lip_top,
                group="mouth",
                source_type="insightface_106_index",
                source_indices=[upper_idx],
                source_region="mouth",
                description="Upper-lip apex from the canonical InsightFace landmark_2d_106 index map.",
            ),
            "lower_lip_bottom": _semantic_point_debug_entry(
                point_name="lower_lip_bottom",
                point=lower_lip_bottom,
                group="mouth",
                source_type="insightface_106_index",
                source_indices=[lower_idx],
                source_region="mouth",
                description="Lower-lip nadir from the canonical InsightFace landmark_2d_106 index map.",
            ),
        },
    }


def _derive_mouth_geometry_from_dense(
    *,
    dense: np.ndarray | None,
    bbox: list[float] | None,
    nose_tip: np.ndarray,
    fallback_left: np.ndarray,
    fallback_right: np.ndarray,
) -> dict[str, Any]:
    fallback_center = (fallback_left + fallback_right) / 2.0
    fallback = {
        "left_mouth": fallback_left,
        "right_mouth": fallback_right,
        "mouth_center": fallback_center,
        "upper_lip_top": None,
        "lower_lip_bottom": None,
        "source_type": "anchor_fallback",
        "debug": {
            "left_mouth": _semantic_point_debug_entry(
                point_name="mouth_left",
                point=fallback_left,
                group="mouth",
                source_type="anchor_fallback",
                source_indices=[3],
                description="Fallback mouth corner from five-point anchor landmarks.",
            ),
            "right_mouth": _semantic_point_debug_entry(
                point_name="mouth_right",
                point=fallback_right,
                group="mouth",
                source_type="anchor_fallback",
                source_indices=[4],
                description="Fallback mouth corner from five-point anchor landmarks.",
            ),
            "mouth_center": _semantic_point_debug_entry(
                point_name="mouth_center",
                point=fallback_center,
                group="mouth",
                source_type="derived",
                source_indices=[3, 4],
                description="Midpoint between fallback mouth corner anchors.",
            ),
        },
    }
    if dense is None or bbox is None or len(dense) == 0 or len(bbox) != 4:
        return fallback

    if has_106_landmarks(dense):
        indexed_geometry = _derive_mouth_geometry_from_106_indices(
            dense=dense,
            bbox=bbox,
            nose_tip=nose_tip,
        )
        if indexed_geometry is not None:
            return indexed_geometry

    x1, y1, x2, _y2 = [float(value) for value in bbox]
    face_width = max(x2 - x1, 1.0)
    face_height = max(float(bbox[3]) - y1, 1.0)
    face_center_x = (x1 + x2) / 2.0
    candidate_dense = dense
    candidate_indices = np.arange(dense.shape[0], dtype=np.int32)
    if has_106_landmarks(dense):
        candidate_indices, candidate_dense = points_for_region(dense, "mouth")

    region_indices, mouth_region = _indexed_region_points(
        candidate_dense,
        indices=candidate_indices,
        x_min=float(face_center_x - 0.32 * face_width),
        x_max=float(face_center_x + 0.32 * face_width),
        y_min=float(max(nose_tip[1] + 0.06 * face_height, y1 + 0.58 * face_height)),
        y_max=float(y1 + 0.84 * face_height),
    )
    if len(mouth_region) < 6:
        region_indices, mouth_region = _indexed_region_points(
            candidate_dense,
            indices=candidate_indices,
            x_min=float(face_center_x - 0.40 * face_width),
            x_max=float(face_center_x + 0.40 * face_width),
            y_min=float(max(nose_tip[1] + 0.04 * face_height, y1 + 0.54 * face_height)),
            y_max=float(y1 + 0.88 * face_height),
        )
    if len(mouth_region) < 4:
        return fallback

    y_low = float(np.quantile(mouth_region[:, 1], 0.20))
    y_high = float(np.quantile(mouth_region[:, 1], 0.80))
    band_indices, lip_band = _indexed_region_points(
        mouth_region,
        indices=region_indices,
        y_min=y_low,
        y_max=y_high,
    )
    if len(lip_band) < 4:
        band_indices, lip_band = region_indices, mouth_region

    band_center_y = float(np.median(lip_band[:, 1]))
    left_indices, left_candidates = _indexed_region_points(
        lip_band,
        indices=band_indices,
        x_max=face_center_x,
    )
    right_indices, right_candidates = _indexed_region_points(
        lip_band,
        indices=band_indices,
        x_min=face_center_x,
    )
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return fallback

    target_left_x = float(np.quantile(lip_band[:, 0], 0.15))
    target_right_x = float(np.quantile(lip_band[:, 0], 0.85))
    x_scale = max(1.0, 0.25 * face_width)
    y_scale = max(1.0, 0.25 * face_height)
    left_scores = (
        np.abs(left_candidates[:, 0] - target_left_x) / x_scale
        + np.abs(left_candidates[:, 1] - band_center_y) / y_scale
    )
    right_scores = (
        np.abs(right_candidates[:, 0] - target_right_x) / x_scale
        + np.abs(right_candidates[:, 1] - band_center_y) / y_scale
    )
    left_pick = int(np.argmin(left_scores))
    right_pick = int(np.argmin(right_scores))
    left_mouth = left_candidates[left_pick]
    right_mouth = right_candidates[right_pick]
    left_source = [int(left_indices[left_pick])]
    right_source = [int(right_indices[right_pick])]

    if not _is_valid_mouth_pair(
        left_mouth=left_mouth,
        right_mouth=right_mouth,
        nose_tip=nose_tip,
        bbox=bbox,
    ):
        return fallback

    mouth_center = (left_mouth + right_mouth) / 2.0
    central_indices, central_lip_region = _indexed_region_points(
        mouth_region,
        indices=region_indices,
        x_min=float(mouth_center[0] - 0.10 * face_width),
        x_max=float(mouth_center[0] + 0.10 * face_width),
    )
    if len(central_lip_region) < 2:
        central_indices, central_lip_region = _indexed_region_points(
            mouth_region,
            indices=region_indices,
            x_min=float(face_center_x - 0.12 * face_width),
            x_max=float(face_center_x + 0.12 * face_width),
        )

    upper_lip_top, upper_source = _indexed_closest_point_to_x(
        central_lip_region,
        central_indices,
        float(mouth_center[0]),
        mode_y="min",
    )
    lower_lip_bottom, lower_source = _indexed_closest_point_to_x(
        central_lip_region,
        central_indices,
        float(mouth_center[0]),
        mode_y="max",
    )

    return {
        "left_mouth": left_mouth,
        "right_mouth": right_mouth,
        "mouth_center": mouth_center,
        "upper_lip_top": upper_lip_top,
        "lower_lip_bottom": lower_lip_bottom,
        "source_type": "heuristic_dense_lips",
        "debug": {
            "left_mouth": _semantic_point_debug_entry(
                point_name="mouth_left",
                point=left_mouth,
                group="mouth",
                source_type="heuristic_dense_lips",
                source_indices=left_source,
                description="Left lip corner selected from the central lip band of the dense landmark set.",
            ),
            "right_mouth": _semantic_point_debug_entry(
                point_name="mouth_right",
                point=right_mouth,
                group="mouth",
                source_type="heuristic_dense_lips",
                source_indices=right_source,
                description="Right lip corner selected from the central lip band of the dense landmark set.",
            ),
            "mouth_center": _semantic_point_debug_entry(
                point_name="mouth_center",
                point=mouth_center,
                group="mouth",
                source_type="derived",
                source_indices=left_source + right_source,
                description="Midpoint between the selected lip corners.",
            ),
            "upper_lip_top": _semantic_point_debug_entry(
                point_name="upper_lip_top",
                point=upper_lip_top,
                group="mouth",
                source_type="heuristic_dense_lips",
                source_indices=upper_source,
                description="Highest central lip point near the mouth midline.",
            ),
            "lower_lip_bottom": _semantic_point_debug_entry(
                point_name="lower_lip_bottom",
                point=lower_lip_bottom,
                group="mouth",
                source_type="heuristic_dense_lips",
                source_indices=lower_source,
                description="Lowest central lip point near the mouth midline.",
            ),
        },
    }


def _validate_semantic_geometry(
    *,
    bbox: list[float] | None,
    left_eye: np.ndarray | None,
    right_eye: np.ndarray | None,
    nose_tip: np.ndarray | None,
    nose_base_left: np.ndarray | None,
    nose_base_right: np.ndarray | None,
    mouth_left: np.ndarray | None,
    mouth_right: np.ndarray | None,
    mouth_center: np.ndarray | None,
    chin_center: np.ndarray | None,
    jaw_left: np.ndarray | None,
    jaw_right: np.ndarray | None,
    cheekbone_left: np.ndarray | None,
    cheekbone_right: np.ndarray | None,
    semantic_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    failed_checks: list[str] = []
    if bbox is None or len(bbox) != 4:
        warnings.append("semantic_validation_skipped_missing_bbox")
        return {
            "passed": False,
            "warnings": warnings,
            "failed_checks": ["missing_bbox"],
            "unreliable_features": sorted(SEMANTIC_UNRELIABLE_RAW_FEATURES),
        }

    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width = max(x2 - x1, 1.0)
    face_height = max(y2 - y1, 1.0)
    face_center_x = (x1 + x2) / 2.0
    lower_half_top = y1 + 0.50 * face_height

    def _meta(name: str) -> dict[str, Any]:
        if semantic_metadata is None:
            return {}
        value = semantic_metadata.get(name)
        return value if isinstance(value, dict) else {}

    mouth_left_meta = _meta("left_mouth")
    mouth_right_meta = _meta("right_mouth")
    nose_tip_meta = _meta("nose_tip")
    nose_base_left_meta = _meta("nose_base_left")
    nose_base_right_meta = _meta("nose_base_right")
    left_eye_meta = _meta("left_eye_center")
    right_eye_meta = _meta("right_eye_center")

    if mouth_left_meta.get("source_indices") and mouth_left_meta.get("source_indices") != [MOUTH_LEFT_CORNER_INDEX]:
        failed_checks.append("mouth_left_wrong_source_index")
    if mouth_right_meta.get("source_indices") and mouth_right_meta.get("source_indices") != [MOUTH_RIGHT_CORNER_INDEX]:
        failed_checks.append("mouth_right_wrong_source_index")
    if mouth_left_meta.get("source_region") not in {None, "mouth"}:
        failed_checks.append("mouth_left_wrong_source_region")
    if mouth_right_meta.get("source_region") not in {None, "mouth"}:
        failed_checks.append("mouth_right_wrong_source_region")

    if nose_tip_meta.get("source_indices") and nose_tip_meta.get("source_indices") != [NOSE_TIP_INDEX]:
        failed_checks.append("nose_tip_wrong_source_index")
    if nose_tip_meta.get("source_region") not in {None, "nose"}:
        failed_checks.append("nose_tip_wrong_source_region")

    for name, meta, expected in (
        ("nose_base_left", nose_base_left_meta, [NOSE_BASE_LEFT_INDEX]),
        ("nose_base_right", nose_base_right_meta, [NOSE_BASE_RIGHT_INDEX]),
    ):
        source_indices = [int(value) for value in meta.get("source_indices", [])]
        if source_indices and any(index in {MOUTH_LEFT_CORNER_INDEX, MOUTH_RIGHT_CORNER_INDEX} for index in source_indices):
            failed_checks.append(f"{name}_uses_mouth_indices")
        if source_indices and any(index < 77 or index > 85 for index in source_indices):
            failed_checks.append(f"{name}_outside_nose_base_region")
        if source_indices and source_indices != expected:
            failed_checks.append(f"{name}_wrong_source_index")
        if meta.get("source_region") not in {None, "nose"}:
            failed_checks.append(f"{name}_wrong_source_region")

    for name, meta, allowed_indices in (
        ("left_eye_center", left_eye_meta, set(LEFT_EYE_INDICES)),
        ("right_eye_center", right_eye_meta, set(RIGHT_EYE_INDICES)),
    ):
        source_indices = [int(value) for value in meta.get("source_indices", [])]
        if source_indices and any(index not in allowed_indices for index in source_indices):
            failed_checks.append(f"{name}_outside_eye_cluster")

    if mouth_left is None or mouth_right is None or mouth_center is None:
        failed_checks.append("missing_mouth_points")
    if nose_tip is None:
        failed_checks.append("missing_nose_tip")

    nose_base_center = _point_midpoint(nose_base_left, nose_base_right)

    if mouth_left is not None and mouth_right is not None:
        if not (float(mouth_left[0]) < float(mouth_right[0])):
            failed_checks.append("mouth_left_not_left_of_right")
        mouth_width = float(abs(mouth_right[0] - mouth_left[0]))
        if mouth_width < 0.08 * face_width or mouth_width > 0.70 * face_width:
            failed_checks.append("mouth_width_out_of_range")
        if float(mouth_left[1]) < lower_half_top or float(mouth_right[1]) < lower_half_top:
            failed_checks.append("mouth_corners_above_lower_half")
        if (
            abs(float(mouth_left[0] - face_center_x)) > 0.46 * face_width
            or abs(float(mouth_right[0] - face_center_x)) > 0.46 * face_width
        ):
            failed_checks.append("mouth_corners_far_from_center")

    if mouth_left is not None and mouth_right is not None and mouth_center is not None:
        if not (float(mouth_left[0]) <= float(mouth_center[0]) <= float(mouth_right[0])):
            failed_checks.append("mouth_center_not_between_corners")

    if nose_tip is not None and mouth_left is not None and mouth_right is not None:
        if float(mouth_left[1]) <= float(nose_tip[1]) or float(mouth_right[1]) <= float(nose_tip[1]):
            failed_checks.append("mouth_corners_above_nose_tip")
    if nose_base_center is not None and mouth_left is not None and mouth_right is not None:
        if float(mouth_left[1]) <= float(nose_base_center[1]) or float(mouth_right[1]) <= float(nose_base_center[1]):
            failed_checks.append("mouth_corners_above_nose_base")

    if left_eye is not None and right_eye is not None and nose_tip is not None and mouth_center is not None:
        eye_center_y = float((left_eye[1] + right_eye[1]) / 2.0)
        if not (eye_center_y < float(nose_tip[1]) < float(mouth_center[1])):
            failed_checks.append("nose_tip_not_between_eyes_and_mouth")

    if nose_base_center is not None and left_eye is not None and right_eye is not None:
        eye_center_y = float((left_eye[1] + right_eye[1]) / 2.0)
        if float(nose_base_center[1]) <= eye_center_y:
            failed_checks.append("nose_base_above_eyes")
        if mouth_center is not None and float(nose_base_center[1]) >= float(mouth_center[1]):
            failed_checks.append("nose_base_below_mouth")

    if chin_center is not None and mouth_center is not None and float(chin_center[1]) <= float(mouth_center[1]):
        failed_checks.append("chin_not_below_mouth")
    if jaw_left is not None and mouth_center is not None and float(jaw_left[1]) <= float(mouth_center[1]):
        failed_checks.append("jaw_left_not_below_mouth")
    if jaw_right is not None and mouth_center is not None and float(jaw_right[1]) <= float(mouth_center[1]):
        failed_checks.append("jaw_right_not_below_mouth")

    if mouth_left is not None and jaw_left is not None:
        if float(np.linalg.norm(mouth_left - jaw_left)) < 0.08 * face_width:
            failed_checks.append("mouth_left_too_close_to_jaw_left")
    if mouth_right is not None and jaw_right is not None:
        if float(np.linalg.norm(mouth_right - jaw_right)) < 0.08 * face_width:
            failed_checks.append("mouth_right_too_close_to_jaw_right")
    if mouth_left is not None and cheekbone_left is not None:
        if float(np.linalg.norm(mouth_left - cheekbone_left)) < 0.07 * face_width:
            failed_checks.append("mouth_left_too_close_to_cheekbone_left")
    if mouth_right is not None and cheekbone_right is not None:
        if float(np.linalg.norm(mouth_right - cheekbone_right)) < 0.07 * face_width:
            failed_checks.append("mouth_right_too_close_to_cheekbone_right")

    passed = len(failed_checks) == 0
    if not passed:
        warnings.append("semantic_validation_failed")
    return {
        "passed": passed,
        "warnings": warnings,
        "failed_checks": sorted(set(failed_checks)),
        "unreliable_features": (
            []
            if passed
            else sorted(SEMANTIC_UNRELIABLE_RAW_FEATURES.union(SEMANTIC_UNRELIABLE_ANGLE_FEATURES))
        ),
    }


def _derive_dense_context_from_insightface_106(
    *,
    landmarks: list[list[float]],
    landmarks_dense: list[list[float]],
    bbox: list[float],
) -> dict[str, np.ndarray | float | None]:
    context: dict[str, np.ndarray | float | None] = {}
    points5 = np.asarray(landmarks[:5], dtype=np.float64)
    left_eye_5, right_eye_5, nose_tip_5, left_mouth_5, right_mouth_5 = points5
    dense = np.asarray(landmarks_dense, dtype=np.float64)
    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width = max(x2 - x1, 1.0)
    face_height = max(y2 - y1, 1.0)
    face_center_x = (x1 + x2) / 2.0
    semantic_meta: dict[str, dict[str, Any]] = {}

    def register(
        name: str,
        point: np.ndarray | None,
        *,
        group: str,
        source_type: str,
        source_indices: list[int] | None,
        description: str,
    ) -> None:
        if point is None:
            return
        context[name] = point
        normalized_indices = [int(value) for value in (source_indices or [])]
        context[f"{name}_source_indices"] = normalized_indices
        meta = _make_semantic_meta(
            point_name=name,
            point=point,
            group=group,
            source_type=source_type,
            source_indices=normalized_indices,
            source_region=_region_for_source_indices(normalized_indices) or group,
            description=description,
            validation_status="validated",
        )
        if meta is not None:
            semantic_meta[name] = meta

    left_eye_indices, left_eye_points = points_for_region(dense, "left_eye")
    right_eye_indices, right_eye_points = points_for_region(dense, "right_eye")
    left_brow_indices, left_brow_points = points_for_region(dense, "left_brow")
    right_brow_indices, right_brow_points = points_for_region(dense, "right_brow")
    left_contour_indices, left_contour_points = points_for_indices(dense, LEFT_CONTOUR_INDICES)
    right_contour_indices, right_contour_points = points_for_indices(dense, RIGHT_CONTOUR_INDICES)
    contour_indices, contour_points = points_for_indices(dense, CONTOUR_INDICES)
    upper_lip_arc_indices, upper_lip_arc_points = points_for_indices(dense, UPPER_LIP_ARC_INDICES)

    left_eye_outer, left_eye_outer_indices = _indexed_extreme_point(left_eye_points, left_eye_indices, axis=0, mode="min")
    left_eye_inner, left_eye_inner_indices = _indexed_extreme_point(left_eye_points, left_eye_indices, axis=0, mode="max")
    right_eye_inner, right_eye_inner_indices = _indexed_extreme_point(right_eye_points, right_eye_indices, axis=0, mode="min")
    right_eye_outer, right_eye_outer_indices = _indexed_extreme_point(right_eye_points, right_eye_indices, axis=0, mode="max")
    left_eye_top, left_eye_top_indices = _indexed_extreme_point(left_eye_points, left_eye_indices, axis=1, mode="min")
    left_eye_bottom, left_eye_bottom_indices = _indexed_extreme_point(left_eye_points, left_eye_indices, axis=1, mode="max")
    right_eye_top, right_eye_top_indices = _indexed_extreme_point(right_eye_points, right_eye_indices, axis=1, mode="min")
    right_eye_bottom, right_eye_bottom_indices = _indexed_extreme_point(right_eye_points, right_eye_indices, axis=1, mode="max")

    left_eye_center = left_eye_points.mean(axis=0) if len(left_eye_points) > 0 else left_eye_5
    right_eye_center = right_eye_points.mean(axis=0) if len(right_eye_points) > 0 else right_eye_5
    eye_center = _point_midpoint(left_eye_center, right_eye_center)
    left_brow = left_brow_points.mean(axis=0) if len(left_brow_points) > 0 else None
    right_brow = right_brow_points.mean(axis=0) if len(right_brow_points) > 0 else None
    brow_center = _point_midpoint(left_brow, right_brow)

    nose_tip = point_for_index(dense, NOSE_TIP_INDEX)
    if nose_tip is None:
        nose_tip = nose_tip_5
    alare_left = point_for_index(dense, ALARE_LEFT_INDEX)
    nose_base_left = point_for_index(dense, NOSE_BASE_LEFT_INDEX)
    subnasale_left = point_for_index(dense, SUBNASALE_LEFT_INDEX)
    subnasale_center = point_for_index(dense, SUBNASALE_CENTER_INDEX)
    alare_right = point_for_index(dense, ALARE_RIGHT_INDEX)
    nose_base_right = point_for_index(dense, NOSE_BASE_RIGHT_INDEX)
    subnasale_right = point_for_index(dense, SUBNASALE_RIGHT_INDEX)
    nose_mid_left = alare_left
    nose_mid_right = alare_right
    nose_base_center = _point_midpoint(nose_base_left, nose_base_right)

    mouth_geometry = _derive_mouth_geometry_from_106_indices(
        dense=dense,
        bbox=bbox,
        nose_tip=nose_tip if nose_tip is not None else nose_tip_5,
    ) or {}
    left_mouth = mouth_geometry.get("left_mouth") if isinstance(mouth_geometry.get("left_mouth"), np.ndarray) else left_mouth_5
    right_mouth = (
        mouth_geometry.get("right_mouth") if isinstance(mouth_geometry.get("right_mouth"), np.ndarray) else right_mouth_5
    )
    mouth_center = (
        mouth_geometry.get("mouth_center") if isinstance(mouth_geometry.get("mouth_center"), np.ndarray) else _point_midpoint(left_mouth, right_mouth)
    )
    upper_lip_top = (
        mouth_geometry.get("upper_lip_top")
        if isinstance(mouth_geometry.get("upper_lip_top"), np.ndarray)
        else point_for_index(dense, UPPER_LIP_CENTER_TOP_INDEX)
    )
    lower_lip_bottom = (
        mouth_geometry.get("lower_lip_bottom")
        if isinstance(mouth_geometry.get("lower_lip_bottom"), np.ndarray)
        else point_for_index(dense, LOWER_LIP_CENTER_BOTTOM_INDEX)
    )
    philtrum_top = upper_lip_top

    contour_center_x = face_center_x if mouth_center is None else float(mouth_center[0])
    lower_left_band = left_contour_points[left_contour_points[:, 1] >= np.quantile(left_contour_points[:, 1], 0.55)] if len(left_contour_points) > 0 else np.zeros((0, 2), dtype=np.float64)
    lower_left_indices = left_contour_indices[left_contour_points[:, 1] >= np.quantile(left_contour_points[:, 1], 0.55)] if len(left_contour_points) > 0 else np.zeros((0,), dtype=np.int32)
    lower_right_band = right_contour_points[right_contour_points[:, 1] >= np.quantile(right_contour_points[:, 1], 0.55)] if len(right_contour_points) > 0 else np.zeros((0, 2), dtype=np.float64)
    lower_right_indices = right_contour_indices[right_contour_points[:, 1] >= np.quantile(right_contour_points[:, 1], 0.55)] if len(right_contour_points) > 0 else np.zeros((0,), dtype=np.int32)
    jaw_left, jaw_left_indices = _indexed_extreme_point(lower_left_band, lower_left_indices, axis=0, mode="min")
    jaw_right, jaw_right_indices = _indexed_extreme_point(lower_right_band, lower_right_indices, axis=0, mode="max")

    cheek_low = y1 + 0.32 * face_height
    cheek_high = y1 + 0.62 * face_height
    cheek_left_mask = (left_contour_points[:, 1] >= cheek_low) & (left_contour_points[:, 1] <= cheek_high) if len(left_contour_points) > 0 else np.zeros((0,), dtype=bool)
    cheek_right_mask = (right_contour_points[:, 1] >= cheek_low) & (right_contour_points[:, 1] <= cheek_high) if len(right_contour_points) > 0 else np.zeros((0,), dtype=bool)
    cheekbone_left, cheekbone_left_indices = _indexed_extreme_point(left_contour_points[cheek_left_mask], left_contour_indices[cheek_left_mask], axis=0, mode="min")
    cheekbone_right, cheekbone_right_indices = _indexed_extreme_point(right_contour_points[cheek_right_mask], right_contour_indices[cheek_right_mask], axis=0, mode="max")
    chin_center = point_for_index(dense, 0)

    chin_band_mask = (
        (contour_points[:, 1] >= y1 + 0.78 * face_height)
        & (contour_points[:, 1] <= y2)
        & (np.abs(contour_points[:, 0] - contour_center_x) <= 0.22 * face_width)
    ) if len(contour_points) > 0 else np.zeros((0,), dtype=bool)
    chin_band_points = contour_points[chin_band_mask]
    philtrum_width_proxy = _span(upper_lip_arc_points[:, 0])
    chin_width_proxy = _span(chin_band_points[:, 0])

    context.update(
        {
            "face_center_x": face_center_x,
            "bbox_top": y1,
            "bbox_bottom": y2,
            "left_eye_inner": left_eye_inner,
            "left_eye_outer": left_eye_outer,
            "right_eye_inner": right_eye_inner,
            "right_eye_outer": right_eye_outer,
            "left_eye_top": left_eye_top,
            "left_eye_bottom": left_eye_bottom,
            "right_eye_top": right_eye_top,
            "right_eye_bottom": right_eye_bottom,
            "eye_center": eye_center,
            "left_brow": left_brow,
            "right_brow": right_brow,
            "brow_center": brow_center,
            "nose_mid_left": nose_mid_left,
            "nose_mid_right": nose_mid_right,
            "nose_base_center": nose_base_center,
            "philtrum_top": philtrum_top,
            "chin_center": chin_center,
            "jaw_left": jaw_left,
            "jaw_right": jaw_right,
            "cheekbone_left": cheekbone_left,
            "cheekbone_right": cheekbone_right,
            "philtrum_width_proxy": philtrum_width_proxy,
            "chin_width_proxy": chin_width_proxy,
        }
    )

    register("left_eye_center", left_eye_center, group="left_eye", source_type="insightface_106_cluster", source_indices=left_eye_indices.tolist(), description="Mean of the left-eye cluster (33..42).")
    register("right_eye_center", right_eye_center, group="right_eye", source_type="insightface_106_cluster", source_indices=right_eye_indices.tolist(), description="Mean of the right-eye cluster (87..96).")
    register("nose_tip", nose_tip, group="nose", source_type="insightface_106_index", source_indices=[NOSE_TIP_INDEX], description="Nose tip from index 86.")
    register("left_mouth", left_mouth, group="mouth", source_type="insightface_106_index", source_indices=[MOUTH_LEFT_CORNER_INDEX], description="Left mouth corner from index 52.")
    register("right_mouth", right_mouth, group="mouth", source_type="insightface_106_index", source_indices=[MOUTH_RIGHT_CORNER_INDEX], description="Right mouth corner from index 61.")
    register("mouth_center", mouth_center, group="mouth", source_type="derived", source_indices=[MOUTH_LEFT_CORNER_INDEX, MOUTH_RIGHT_CORNER_INDEX], description="Midpoint between mouth corners 52 and 61.")
    register("eye_center", eye_center, group="eyes", source_type="derived", source_indices=left_eye_indices.tolist() + right_eye_indices.tolist(), description="Midpoint between left and right eye centers.")
    register("left_eye_outer", left_eye_outer, group="left_eye", source_type="insightface_106_cluster_extreme", source_indices=left_eye_outer_indices, description="Outer corner of the left eye from the left-eye cluster.")
    register("left_eye_inner", left_eye_inner, group="left_eye", source_type="insightface_106_cluster_extreme", source_indices=left_eye_inner_indices, description="Inner corner of the left eye from the left-eye cluster.")
    register("right_eye_inner", right_eye_inner, group="right_eye", source_type="insightface_106_cluster_extreme", source_indices=right_eye_inner_indices, description="Inner corner of the right eye from the right-eye cluster.")
    register("right_eye_outer", right_eye_outer, group="right_eye", source_type="insightface_106_cluster_extreme", source_indices=right_eye_outer_indices, description="Outer corner of the right eye from the right-eye cluster.")
    register("left_eye_top", left_eye_top, group="left_eye", source_type="insightface_106_cluster_extreme", source_indices=left_eye_top_indices, description="Top point of the left-eye cluster.")
    register("left_eye_bottom", left_eye_bottom, group="left_eye", source_type="insightface_106_cluster_extreme", source_indices=left_eye_bottom_indices, description="Bottom point of the left-eye cluster.")
    register("right_eye_top", right_eye_top, group="right_eye", source_type="insightface_106_cluster_extreme", source_indices=right_eye_top_indices, description="Top point of the right-eye cluster.")
    register("right_eye_bottom", right_eye_bottom, group="right_eye", source_type="insightface_106_cluster_extreme", source_indices=right_eye_bottom_indices, description="Bottom point of the right-eye cluster.")
    register("left_brow", left_brow, group="left_brow", source_type="insightface_106_cluster", source_indices=left_brow_indices.tolist(), description="Mean of the left brow cluster (43..51).")
    register("right_brow", right_brow, group="right_brow", source_type="insightface_106_cluster", source_indices=right_brow_indices.tolist(), description="Mean of the right brow cluster (97..105).")
    register("brow_center", brow_center, group="brows", source_type="derived", source_indices=left_brow_indices.tolist() + right_brow_indices.tolist(), description="Midpoint between left and right brow cluster centers.")
    register("alare_left", alare_left, group="nose", source_type="insightface_106_index", source_indices=[ALARE_LEFT_INDEX], description="Left alare from index 77.")
    register("nose_mid_left", nose_mid_left, group="nose", source_type="insightface_106_index", source_indices=[ALARE_LEFT_INDEX], description="Left nasal width point from alare index 77.")
    register("nose_base_left", nose_base_left, group="nose", source_type="insightface_106_index", source_indices=[NOSE_BASE_LEFT_INDEX], description="Left nose base from index 78.")
    register("subnasale_left", subnasale_left, group="nose", source_type="insightface_106_index", source_indices=[SUBNASALE_LEFT_INDEX], description="Left subnasale-adjacent point from index 79.")
    register("subnasale_center", subnasale_center, group="nose", source_type="insightface_106_index", source_indices=[SUBNASALE_CENTER_INDEX], description="Subnasale center from index 80.")
    register("alare_right", alare_right, group="nose", source_type="insightface_106_index", source_indices=[ALARE_RIGHT_INDEX], description="Right alare from index 83.")
    register("nose_mid_right", nose_mid_right, group="nose", source_type="insightface_106_index", source_indices=[ALARE_RIGHT_INDEX], description="Right nasal width point from alare index 83.")
    register("nose_base_right", nose_base_right, group="nose", source_type="insightface_106_index", source_indices=[NOSE_BASE_RIGHT_INDEX], description="Right nose base from index 84.")
    register("subnasale_right", subnasale_right, group="nose", source_type="insightface_106_index", source_indices=[SUBNASALE_RIGHT_INDEX], description="Right subnasale-adjacent point from index 85.")
    register("nose_base_center", nose_base_center, group="nose", source_type="derived", source_indices=[NOSE_BASE_LEFT_INDEX, NOSE_BASE_RIGHT_INDEX], description="Midpoint between nose-base points 78 and 84.")
    register("philtrum_top", philtrum_top, group="mouth", source_type="insightface_106_index", source_indices=[UPPER_LIP_CENTER_TOP_INDEX], description="Upper lip center-top used as philtrum-top proxy.")
    register("upper_lip_top", upper_lip_top, group="mouth", source_type="insightface_106_index", source_indices=[UPPER_LIP_CENTER_TOP_INDEX], description="Upper lip center top from index 71.")
    register("lower_lip_bottom", lower_lip_bottom, group="mouth", source_type="insightface_106_index", source_indices=[LOWER_LIP_CENTER_BOTTOM_INDEX], description="Lower lip center bottom from index 53.")
    register("chin_center", chin_center, group="contour", source_type="insightface_106_index", source_indices=[0], description="Chin center from contour index 0.")
    register("jaw_left", jaw_left, group="contour", source_type="contour_region_extreme", source_indices=jaw_left_indices, description="Left jaw proxy from lower left contour.")
    register("jaw_right", jaw_right, group="contour", source_type="contour_region_extreme", source_indices=jaw_right_indices, description="Right jaw proxy from lower right contour.")
    register("cheekbone_left", cheekbone_left, group="contour", source_type="contour_region_extreme", source_indices=cheekbone_left_indices, description="Left cheekbone proxy from mid-face left contour.")
    register("cheekbone_right", cheekbone_right, group="contour", source_type="contour_region_extreme", source_indices=cheekbone_right_indices, description="Right cheekbone proxy from mid-face right contour.")

    validation = _validate_semantic_geometry(
        bbox=bbox,
        left_eye=left_eye_center,
        right_eye=right_eye_center,
        nose_tip=nose_tip,
        nose_base_left=nose_base_left,
        nose_base_right=nose_base_right,
        mouth_left=left_mouth,
        mouth_right=right_mouth,
        mouth_center=mouth_center,
        chin_center=chin_center,
        jaw_left=jaw_left,
        jaw_right=jaw_right,
        cheekbone_left=cheekbone_left,
        cheekbone_right=cheekbone_right,
        semantic_metadata=semantic_meta,
    )
    status = "validated" if bool(validation.get("passed")) else "unreliable"
    for meta in semantic_meta.values():
        meta["validation_status"] = status
    context["semantic_metadata"] = semantic_meta
    context["semantic_validation"] = validation
    return context


def _derive_dense_context(
    *,
    landmarks: list[list[float]] | None,
    landmarks_dense: list[list[float]] | None,
    bbox: list[float] | None,
) -> dict[str, np.ndarray | float | None]:
    context: dict[str, np.ndarray | float | None] = {}
    if not landmarks or not bbox or len(landmarks) < 5 or len(bbox) != 4:
        return context

    points5 = np.asarray(landmarks[:5], dtype=np.float64)
    left_eye, right_eye, nose_tip, left_mouth, right_mouth = points5
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0
    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width = max(x2 - x1, 1.0)
    face_height = max(y2 - y1, 1.0)
    face_center_x = (x1 + x2) / 2.0

    dense = np.asarray(landmarks_dense, dtype=np.float64) if landmarks_dense else None
    context.update(
        {
            "left_eye_center": left_eye,
            "right_eye_center": right_eye,
            "nose_tip": nose_tip,
            "left_mouth": left_mouth,
            "right_mouth": right_mouth,
            "eye_center": eye_center,
            "mouth_center": mouth_center,
            "face_center_x": face_center_x,
            "bbox_top": y1,
            "bbox_bottom": y2,
        }
    )

    if dense is None or len(dense) == 0:
        return context

    if has_106_landmarks(dense):
        return _derive_dense_context_from_insightface_106(
            landmarks=landmarks,
            landmarks_dense=landmarks_dense or [],
            bbox=bbox,
        )

    mouth_geometry = _derive_mouth_geometry_from_dense(
        dense=dense,
        bbox=bbox,
        nose_tip=nose_tip,
        fallback_left=left_mouth,
        fallback_right=right_mouth,
    )
    left_mouth = mouth_geometry["left_mouth"]
    right_mouth = mouth_geometry["right_mouth"]
    mouth_center = mouth_geometry["mouth_center"]
    context.update(
        {
            "left_mouth": left_mouth,
            "right_mouth": right_mouth,
            "mouth_center": mouth_center,
        }
    )

    left_eye_region = _region_points(
        dense,
        x_min=float(x1),
        x_max=float(face_center_x),
        y_min=float(left_eye[1] - 0.12 * face_height),
        y_max=float(left_eye[1] + 0.08 * face_height),
    )
    right_eye_region = _region_points(
        dense,
        x_min=float(face_center_x),
        x_max=float(x2),
        y_min=float(right_eye[1] - 0.12 * face_height),
        y_max=float(right_eye[1] + 0.08 * face_height),
    )

    left_eye_outer = _extreme_point(left_eye_region, axis=0, mode="min")
    left_eye_inner = _extreme_point(left_eye_region, axis=0, mode="max")
    right_eye_inner = _extreme_point(right_eye_region, axis=0, mode="min")
    right_eye_outer = _extreme_point(right_eye_region, axis=0, mode="max")
    left_eye_top = _extreme_point(left_eye_region, axis=1, mode="min")
    left_eye_bottom = _extreme_point(left_eye_region, axis=1, mode="max")
    right_eye_top = _extreme_point(right_eye_region, axis=1, mode="min")
    right_eye_bottom = _extreme_point(right_eye_region, axis=1, mode="max")

    left_brow_region = _region_points(
        dense,
        x_min=float(left_eye[0] - 0.18 * face_width),
        x_max=float(left_eye[0] + 0.10 * face_width),
        y_min=float(left_eye[1] - 0.24 * face_height),
        y_max=float(left_eye[1] - 0.02 * face_height),
    )
    right_brow_region = _region_points(
        dense,
        x_min=float(right_eye[0] - 0.10 * face_width),
        x_max=float(right_eye[0] + 0.18 * face_width),
        y_min=float(right_eye[1] - 0.24 * face_height),
        y_max=float(right_eye[1] - 0.02 * face_height),
    )
    left_brow = _cluster_center(left_brow_region)
    right_brow = _cluster_center(right_brow_region)

    nose_mid_region = _region_points(
        dense,
        x_min=float(face_center_x - 0.16 * face_width),
        x_max=float(face_center_x + 0.16 * face_width),
        y_min=float(nose_tip[1] - 0.18 * face_height),
        y_max=float(nose_tip[1] - 0.02 * face_height),
    )
    nose_base_region = _region_points(
        dense,
        x_min=float(face_center_x - 0.18 * face_width),
        x_max=float(face_center_x + 0.18 * face_width),
        y_min=float(nose_tip[1] - 0.05 * face_height),
        y_max=float(nose_tip[1] + 0.14 * face_height),
    )
    nose_mid_left = _extreme_point(nose_mid_region, axis=0, mode="min")
    nose_mid_right = _extreme_point(nose_mid_region, axis=0, mode="max")
    nose_base_left = _extreme_point(nose_base_region, axis=0, mode="min")
    nose_base_right = _extreme_point(nose_base_region, axis=0, mode="max")
    nose_base_left_source_indices: list[int] = []
    nose_base_right_source_indices: list[int] = []
    if has_106_landmarks(dense):
        mapped_nose_base_left = _safe_index_point(dense, int(NOSE_BASE_LEFT_INDEX))
        mapped_nose_base_right = _safe_index_point(dense, int(NOSE_BASE_RIGHT_INDEX))
        if mapped_nose_base_left is not None:
            if (
                abs(float(mapped_nose_base_left[0] - face_center_x)) <= 0.28 * face_width
                and float(mapped_nose_base_left[1]) >= float(nose_tip[1]) - 0.10 * face_height
                and float(mapped_nose_base_left[1]) <= float(nose_tip[1]) + 0.22 * face_height
            ):
                nose_base_left = mapped_nose_base_left
                nose_base_left_source_indices = [int(NOSE_BASE_LEFT_INDEX)]
        if mapped_nose_base_right is not None:
            if (
                abs(float(mapped_nose_base_right[0] - face_center_x)) <= 0.28 * face_width
                and float(mapped_nose_base_right[1]) >= float(nose_tip[1]) - 0.10 * face_height
                and float(mapped_nose_base_right[1]) <= float(nose_tip[1]) + 0.22 * face_height
            ):
                nose_base_right = mapped_nose_base_right
                nose_base_right_source_indices = [int(NOSE_BASE_RIGHT_INDEX)]
    nose_base_center = _point_midpoint(nose_base_left, nose_base_right)

    mouth_region = _region_points(
        dense,
        x_min=float(face_center_x - 0.26 * face_width),
        x_max=float(face_center_x + 0.26 * face_width),
        y_min=float(max(nose_tip[1] + 0.05 * face_height, y1 + 0.56 * face_height)),
        y_max=float(y1 + 0.84 * face_height),
    )
    central_lip_region = _region_points(
        mouth_region,
        x_min=float(face_center_x - 0.14 * face_width),
        x_max=float(face_center_x + 0.14 * face_width),
    )
    upper_lip_top = mouth_geometry.get("upper_lip_top")
    if upper_lip_top is None:
        upper_lip_top = _closest_point_to_x(
            central_lip_region,
            face_center_x,
            mode_y="min",
        )
    lower_lip_bottom = mouth_geometry.get("lower_lip_bottom")
    if lower_lip_bottom is None:
        lower_lip_bottom = _closest_point_to_x(
            central_lip_region,
            face_center_x,
            mode_y="max",
        )
    philtrum_top_region = _region_points(
        dense,
        x_min=float(face_center_x - 0.10 * face_width),
        x_max=float(face_center_x + 0.10 * face_width),
        y_min=float((nose_base_center[1] if nose_base_center is not None else nose_tip[1]) - 0.02 * face_height),
        y_max=float(mouth_center[1]),
    )
    philtrum_top = _closest_point_to_x(philtrum_top_region, face_center_x, mode_y="min")

    chin_region = _region_points(
        dense,
        x_min=float(face_center_x - 0.15 * face_width),
        x_max=float(face_center_x + 0.15 * face_width),
        y_min=float(mouth_center[1] + 0.08 * face_height),
        y_max=float(y2),
    )
    chin_center = _closest_point_to_x(chin_region, face_center_x, mode_y="max")

    chin_band = _region_points(
        dense,
        x_min=float(face_center_x - 0.20 * face_width),
        x_max=float(face_center_x + 0.20 * face_width),
        y_min=float((chin_center[1] if chin_center is not None else mouth_center[1]) - 0.06 * face_height),
        y_max=float((chin_center[1] if chin_center is not None else y2) + 0.01 * face_height),
    )
    jaw_region_y_min = float(max(y1 + 0.74 * face_height, mouth_center[1] + 0.02 * face_height))
    jaw_source = dense[np.asarray(CONTOUR_INDICES, dtype=np.int32)] if has_106_landmarks(dense) else dense
    jaw_region = _region_points(
        jaw_source,
        y_min=jaw_region_y_min,
        y_max=float(y1 + 0.98 * face_height),
    )
    cheekbone_region = _region_points(
        dense,
        y_min=float(y1 + 0.35 * face_height),
        y_max=float(y1 + 0.65 * face_height),
    )

    context.update(
        {
            "left_eye_inner": left_eye_inner,
            "left_eye_outer": left_eye_outer,
            "right_eye_inner": right_eye_inner,
            "right_eye_outer": right_eye_outer,
            "left_eye_top": left_eye_top,
            "left_eye_bottom": left_eye_bottom,
            "right_eye_top": right_eye_top,
            "right_eye_bottom": right_eye_bottom,
            "left_brow": left_brow,
            "right_brow": right_brow,
            "nose_mid_left": nose_mid_left,
            "nose_mid_right": nose_mid_right,
            "nose_base_left": nose_base_left,
            "nose_base_right": nose_base_right,
            "nose_base_center": nose_base_center,
            "nose_base_left_source_indices": nose_base_left_source_indices,
            "nose_base_right_source_indices": nose_base_right_source_indices,
            "upper_lip_top": upper_lip_top,
            "lower_lip_bottom": lower_lip_bottom,
            "philtrum_top": philtrum_top,
            "chin_center": chin_center,
            "jaw_left": _extreme_point(jaw_region, axis=0, mode="min"),
            "jaw_right": _extreme_point(jaw_region, axis=0, mode="max"),
            "cheekbone_left": _extreme_point(cheekbone_region, axis=0, mode="min"),
            "cheekbone_right": _extreme_point(cheekbone_region, axis=0, mode="max"),
            "philtrum_width_proxy": _span(_region_points(mouth_region, y_min=float((upper_lip_top[1] if upper_lip_top is not None else mouth_center[1]) - 0.02 * face_height), y_max=float((upper_lip_top[1] if upper_lip_top is not None else mouth_center[1]) + 0.02 * face_height))[:, 0]),
            "chin_width_proxy": _span(chin_band[:, 0]),
        }
    )
    validation = _validate_semantic_geometry(
        bbox=bbox,
        left_eye=left_eye,
        right_eye=right_eye,
        nose_tip=nose_tip,
        nose_base_left=nose_base_left,
        nose_base_right=nose_base_right,
        mouth_left=left_mouth,
        mouth_right=right_mouth,
        mouth_center=mouth_center,
        chin_center=chin_center,
        jaw_left=context.get("jaw_left") if isinstance(context.get("jaw_left"), np.ndarray) else None,
        jaw_right=context.get("jaw_right") if isinstance(context.get("jaw_right"), np.ndarray) else None,
        cheekbone_left=context.get("cheekbone_left") if isinstance(context.get("cheekbone_left"), np.ndarray) else None,
        cheekbone_right=context.get("cheekbone_right") if isinstance(context.get("cheekbone_right"), np.ndarray) else None,
        semantic_metadata=context.get("semantic_metadata") if isinstance(context.get("semantic_metadata"), dict) else None,
    )
    context["semantic_validation"] = validation
    return context


def derive_dense_visual_context(
    *,
    landmarks: list[list[float]] | None,
    landmarks_dense: list[list[float]] | None,
    bbox: list[float] | None,
) -> dict[str, np.ndarray | float | None]:
    return _derive_dense_context(
        landmarks=landmarks,
        landmarks_dense=landmarks_dense,
        bbox=bbox,
    )


def derive_semantic_landmark_bundle(
    *,
    landmarks: list[list[float]] | None,
    landmarks_dense: list[list[float]] | None,
    bbox: list[float] | None,
) -> dict[str, Any]:
    if not landmarks or len(landmarks) < 5:
        return {
            "raw_landmarks": [],
            "raw_landmarks_debug": [],
            "semantic_points": {},
            "semantic_points_debug": [],
        }

    def _optional_point(name: str) -> np.ndarray | None:
        value = context.get(name)
        return value if isinstance(value, np.ndarray) else None

    def _optional_source_indices(name: str) -> list[int]:
        value = context.get(f"{name}_source_indices")
        if not isinstance(value, list):
            return []
        return [int(item) for item in value if isinstance(item, (int, np.integer))]

    context = _derive_dense_context(landmarks=landmarks, landmarks_dense=landmarks_dense, bbox=bbox)
    left_eye = _optional_point("left_eye_center")
    right_eye = _optional_point("right_eye_center")
    nose_tip = _optional_point("nose_tip")
    left_mouth = _optional_point("left_mouth")
    right_mouth = _optional_point("right_mouth")
    mouth_center = _optional_point("mouth_center")

    semantic_meta_by_name = (
        context.get("semantic_metadata")
        if isinstance(context.get("semantic_metadata"), dict)
        else {}
    )
    semantic_meta_aliases = {
        "mouth_left": "left_mouth",
        "mouth_right": "right_mouth",
        "brow_left": "left_brow",
        "brow_right": "right_brow",
    }

    semantic_specs: list[tuple[str, np.ndarray | None, str, str, list[int], str]] = [
        ("left_eye_center", left_eye, "eyes", "anchor", [0], "Left eye center used in morphometric measurements."),
        ("right_eye_center", right_eye, "eyes", "anchor", [1], "Right eye center used in morphometric measurements."),
        ("nose_tip", nose_tip, "nose", "anchor", [2], "Nasal tip used in morphometric measurements."),
        ("left_eye_outer", _optional_point("left_eye_outer"), "eyes", "derived", [], "Left eye outer corner constrained to the left-eye cluster."),
        ("left_eye_inner", _optional_point("left_eye_inner"), "eyes", "derived", [], "Left eye inner corner constrained to the left-eye cluster."),
        ("right_eye_inner", _optional_point("right_eye_inner"), "eyes", "derived", [], "Right eye inner corner constrained to the right-eye cluster."),
        ("right_eye_outer", _optional_point("right_eye_outer"), "eyes", "derived", [], "Right eye outer corner constrained to the right-eye cluster."),
        ("left_eye_top", _optional_point("left_eye_top"), "eyes", "derived", [], "Top point of the left-eye cluster."),
        ("left_eye_bottom", _optional_point("left_eye_bottom"), "eyes", "derived", [], "Bottom point of the left-eye cluster."),
        ("right_eye_top", _optional_point("right_eye_top"), "eyes", "derived", [], "Top point of the right-eye cluster."),
        ("right_eye_bottom", _optional_point("right_eye_bottom"), "eyes", "derived", [], "Bottom point of the right-eye cluster."),
        ("mouth_left", left_mouth, "mouth", "derived", [], "Left mouth corner used in morphometric measurements."),
        ("mouth_right", right_mouth, "mouth", "derived", [], "Right mouth corner used in morphometric measurements."),
        ("mouth_center", mouth_center, "mouth", "derived", [], "Midpoint between the left and right mouth corners."),
        ("eye_center", _optional_point("eye_center"), "eyes", "derived", [0, 1], "Midpoint between the eye centers."),
        ("brow_left", _optional_point("left_brow"), "brows", "derived", [], "Estimated left brow point from dense brow landmarks."),
        ("brow_right", _optional_point("right_brow"), "brows", "derived", [], "Estimated right brow point from dense brow landmarks."),
        ("brow_center", _optional_point("brow_center"), "brows", "derived", [], "Midpoint between left and right brow points."),
        ("nose_mid_left", _optional_point("nose_mid_left"), "nose", "derived", [], "Left point on the mid-nose width band."),
        ("nose_mid_right", _optional_point("nose_mid_right"), "nose", "derived", [], "Right point on the mid-nose width band."),
        (
            "nose_base_left",
            _optional_point("nose_base_left"),
            "nose",
            "derived",
            _optional_source_indices("nose_base_left"),
            "Left nose-base point from dense landmarks.",
        ),
        (
            "nose_base_right",
            _optional_point("nose_base_right"),
            "nose",
            "derived",
            _optional_source_indices("nose_base_right"),
            "Right nose-base point from dense landmarks.",
        ),
        ("alare_left", _optional_point("alare_left"), "nose", "derived", _optional_source_indices("alare_left"), "Left alare point from InsightFace 106 nose region."),
        ("subnasale_left", _optional_point("subnasale_left"), "nose", "derived", _optional_source_indices("subnasale_left"), "Left subnasale-adjacent point from InsightFace 106 nose region."),
        ("subnasale_center", _optional_point("subnasale_center"), "nose", "derived", _optional_source_indices("subnasale_center"), "Subnasale center from InsightFace 106 nose region."),
        ("subnasale_right", _optional_point("subnasale_right"), "nose", "derived", _optional_source_indices("subnasale_right"), "Right subnasale-adjacent point from InsightFace 106 nose region."),
        ("alare_right", _optional_point("alare_right"), "nose", "derived", _optional_source_indices("alare_right"), "Right alare point from InsightFace 106 nose region."),
        ("nose_base_center", _optional_point("nose_base_center"), "nose", "derived", [], "Midpoint of the nose base."),
        ("philtrum_top", _optional_point("philtrum_top"), "mouth", "derived", [], "Upper philtrum point near the face midline."),
        ("upper_lip_top", _optional_point("upper_lip_top"), "mouth", "derived", [], "Upper lip top near the mouth midline."),
        ("lower_lip_bottom", _optional_point("lower_lip_bottom"), "mouth", "derived", [], "Lower lip bottom near the mouth midline."),
        ("chin_center", _optional_point("chin_center"), "chin_jaw", "derived", [], "Lowest central chin point."),
        ("jaw_left", _optional_point("jaw_left"), "chin_jaw", "derived", [], "Left jaw proxy point from lower contour landmarks."),
        ("jaw_right", _optional_point("jaw_right"), "chin_jaw", "derived", [], "Right jaw proxy point from lower contour landmarks."),
        ("cheekbone_left", _optional_point("cheekbone_left"), "chin_jaw", "derived", [], "Left cheekbone proxy point from mid-face contour landmarks."),
        ("cheekbone_right", _optional_point("cheekbone_right"), "chin_jaw", "derived", [], "Right cheekbone proxy point from mid-face contour landmarks."),
    ]

    semantic_points: dict[str, dict[str, Any]] = {}
    semantic_points_debug: list[dict[str, Any]] = []
    for name, point, group, source_type, source_indices, description in semantic_specs:
        if point is None:
            continue
        debug_entry = semantic_meta_by_name.get(name)
        if debug_entry is None:
            debug_entry = semantic_meta_by_name.get(semantic_meta_aliases.get(name, ""))
        if debug_entry is None:
            debug_entry = _semantic_point_debug_entry(
                point_name=name,
                point=point,
                group=group,
                source_type=source_type,
                source_indices=source_indices,
                description=description,
                source_region=_region_for_source_indices(source_indices),
                validation_status=(
                    "validated"
                    if not isinstance(context.get("semantic_validation"), dict)
                    or bool(context.get("semantic_validation", {}).get("passed", False))
                    else "unreliable"
                ),
            )
        else:
            debug_entry = dict(debug_entry)
            debug_entry["point_name"] = name
        semantic_points[name] = _semantic_point_payload(
            point_name=name,
            point=point,
            source_type=str(debug_entry.get("source_type", source_type)),
            source_indices=list(debug_entry.get("source_indices", source_indices)),
            source_region=debug_entry.get("source_region"),
            validation_status=str(debug_entry.get("validation_status", "validated")),
        ) or {"x": float(point[0]), "y": float(point[1])}
        if debug_entry is not None:
            semantic_points_debug.append(debug_entry)

    dense = np.asarray(landmarks_dense, dtype=np.float64) if landmarks_dense else None
    raw_source = dense if dense is not None and len(dense) > 0 else np.asarray(landmarks[:5], dtype=np.float64)
    raw_group = "dense_landmarks" if dense is not None and len(dense) > 0 else "anchors"
    def _raw_group_for_index(index: int) -> str:
        if raw_group != "dense_landmarks":
            return raw_group
        return region_name_for_index(index)

    raw_landmarks = [
        {
            "index": int(index),
            "x": float(point[0]),
            "y": float(point[1]),
            "group": _raw_group_for_index(int(index)),
            "name": None,
            "source_type": "raw",
            "source_indices": [int(index)],
            "description": (
                "Raw landmark emitted by the dense InsightFace landmark_2d_106 model."
                if raw_group == "dense_landmarks"
                else "Raw five-point anchor landmark."
            ),
        }
        for index, point in enumerate(raw_source.tolist())
    ]
    raw_landmarks_debug = [
        {
            "index": int(index),
            "x": float(point[0]),
            "y": float(point[1]),
            "group": _raw_group_for_index(int(index)),
            "name": None,
            "source_type": "raw",
            "source_indices": [int(index)],
            "description": (
                "Raw landmark emitted by the dense InsightFace landmark_2d_106 model."
                if raw_group == "dense_landmarks"
                else "Raw five-point anchor landmark."
            ),
        }
        for index, point in enumerate(raw_source.tolist())
    ]
    semantic_validation = context.get("semantic_validation")
    return {
        "raw_landmarks": raw_landmarks,
        "raw_landmarks_debug": raw_landmarks_debug,
        "semantic_points": semantic_points,
        "semantic_points_debug": semantic_points_debug,
        "semantic_validation": semantic_validation,
    }


def extract_raw_measurements(
    landmarks: list[list[float]] | None,
    bbox: list[float] | None,
    landmarks_dense: list[list[float]] | None = None,
) -> dict[str, float | None]:
    if not landmarks or not bbox or len(landmarks) < 5 or len(bbox) != 4:
        return empty_feature_dict(RAW_MEASUREMENT_COLUMNS)

    points = np.asarray(landmarks, dtype=np.float64)
    left_eye, right_eye, nose = points[:3]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width = x2 - x1
    face_height = y2 - y1
    if face_width <= 0.0 or face_height <= 0.0:
        return empty_feature_dict(RAW_MEASUREMENT_COLUMNS)

    context = _derive_dense_context(landmarks=landmarks, landmarks_dense=landmarks_dense, bbox=bbox)
    left_eye = context.get("left_eye_center") if isinstance(context.get("left_eye_center"), np.ndarray) else left_eye
    right_eye = context.get("right_eye_center") if isinstance(context.get("right_eye_center"), np.ndarray) else right_eye
    nose = context.get("nose_tip") if isinstance(context.get("nose_tip"), np.ndarray) else nose
    left_mouth = context.get("left_mouth") if isinstance(context.get("left_mouth"), np.ndarray) else points[3]
    right_mouth = context.get("right_mouth") if isinstance(context.get("right_mouth"), np.ndarray) else points[4]
    eye_center = context.get("eye_center") if isinstance(context.get("eye_center"), np.ndarray) else (left_eye + right_eye) / 2.0
    mouth_center = context.get("mouth_center") if isinstance(context.get("mouth_center"), np.ndarray) else (left_mouth + right_mouth) / 2.0
    face_center_x = float(context.get("face_center_x")) if context.get("face_center_x") is not None else (x1 + x2) / 2.0

    left_brow = context.get("left_brow")
    right_brow = context.get("right_brow")
    brow_center = _point_midpoint(
        left_brow if isinstance(left_brow, np.ndarray) else None,
        right_brow if isinstance(right_brow, np.ndarray) else None,
    )
    nose_base_center = context.get("nose_base_center") if isinstance(context.get("nose_base_center"), np.ndarray) else None
    upper_lip_top = context.get("upper_lip_top") if isinstance(context.get("upper_lip_top"), np.ndarray) else None
    lower_lip_bottom = context.get("lower_lip_bottom") if isinstance(context.get("lower_lip_bottom"), np.ndarray) else None
    chin_center = context.get("chin_center") if isinstance(context.get("chin_center"), np.ndarray) else None
    left_eye_inner = context.get("left_eye_inner") if isinstance(context.get("left_eye_inner"), np.ndarray) else None
    left_eye_outer = context.get("left_eye_outer") if isinstance(context.get("left_eye_outer"), np.ndarray) else None
    right_eye_inner = context.get("right_eye_inner") if isinstance(context.get("right_eye_inner"), np.ndarray) else None
    right_eye_outer = context.get("right_eye_outer") if isinstance(context.get("right_eye_outer"), np.ndarray) else None
    left_eye_top = context.get("left_eye_top") if isinstance(context.get("left_eye_top"), np.ndarray) else None
    left_eye_bottom = context.get("left_eye_bottom") if isinstance(context.get("left_eye_bottom"), np.ndarray) else None
    right_eye_top = context.get("right_eye_top") if isinstance(context.get("right_eye_top"), np.ndarray) else None
    right_eye_bottom = context.get("right_eye_bottom") if isinstance(context.get("right_eye_bottom"), np.ndarray) else None
    nose_mid_left = context.get("nose_mid_left") if isinstance(context.get("nose_mid_left"), np.ndarray) else None
    nose_mid_right = context.get("nose_mid_right") if isinstance(context.get("nose_mid_right"), np.ndarray) else None
    nose_base_left = context.get("nose_base_left") if isinstance(context.get("nose_base_left"), np.ndarray) else None
    nose_base_right = context.get("nose_base_right") if isinstance(context.get("nose_base_right"), np.ndarray) else None
    philtrum_top = context.get("philtrum_top") if isinstance(context.get("philtrum_top"), np.ndarray) else None
    jaw_left = context.get("jaw_left") if isinstance(context.get("jaw_left"), np.ndarray) else None
    jaw_right = context.get("jaw_right") if isinstance(context.get("jaw_right"), np.ndarray) else None
    cheekbone_left = context.get("cheekbone_left") if isinstance(context.get("cheekbone_left"), np.ndarray) else None
    cheekbone_right = context.get("cheekbone_right") if isinstance(context.get("cheekbone_right"), np.ndarray) else None

    left_eye_width = _distance(left_eye_outer, left_eye_inner)
    right_eye_width = _distance(right_eye_inner, right_eye_outer)
    left_eye_height = _distance(left_eye_top, left_eye_bottom)
    right_eye_height = _distance(right_eye_top, right_eye_bottom)
    intercanthal_distance = _distance(left_eye_inner, right_eye_inner)
    outercanthal_distance = _distance(left_eye_outer, right_eye_outer)
    nose_width = _distance(nose_mid_left, nose_mid_right)
    nose_base_width = _distance(nose_base_left, nose_base_right)
    mouth_center_to_chin = abs(float(chin_center[1] - mouth_center[1])) if chin_center is not None else y2 - float(mouth_center[1])
    mouth_center_midline_offset = abs(float(mouth_center[0]) - face_center_x)
    nose_tip_midline_offset = abs(float(nose[0]) - face_center_x)
    chin_center_midline_offset = abs(float(chin_center[0]) - face_center_x) if chin_center is not None else None
    philtrum_midline_offset_proxy = abs(float(philtrum_top[0]) - face_center_x) if philtrum_top is not None else None

    result = {
        "raw_face_width_px": face_width,
        "raw_face_height_px": face_height,
        "raw_eye_distance_px": _distance(left_eye, right_eye),
        "raw_mouth_width_px": _distance(left_mouth, right_mouth),
        "raw_eye_to_nose_px": _distance(eye_center, nose),
        "raw_nose_to_mouth_px": _distance(nose, mouth_center),
        "raw_eye_to_mouth_px": _distance(eye_center, mouth_center),
        "raw_left_eye_to_nose_px": _distance(left_eye, nose),
        "raw_right_eye_to_nose_px": _distance(right_eye, nose),
        "raw_left_mouth_to_nose_px": _distance(left_mouth, nose),
        "raw_right_mouth_to_nose_px": _distance(right_mouth, nose),
        "raw_upper_face_height_px": float(nose[1]) - y1,
        "raw_lower_face_height_px": mouth_center_to_chin,
        "raw_forehead_height_proxy_px": float(eye_center[1]) - y1,
        "raw_jaw_height_proxy_px": mouth_center_to_chin,
        "raw_nose_center_offset_px": nose_tip_midline_offset,
        "raw_mouth_center_offset_px": mouth_center_midline_offset,
        "brow_to_eye_center_height": abs(float(eye_center[1] - brow_center[1])) if brow_center is not None else None,
        "eye_center_to_nose_base_height": abs(float(nose_base_center[1] - eye_center[1])) if nose_base_center is not None else None,
        "nose_base_to_mouth_center_height": abs(float(mouth_center[1] - nose_base_center[1])) if nose_base_center is not None else None,
        "mouth_center_to_chin_height": mouth_center_to_chin,
        "brow_to_nose_base_height": abs(float(nose_base_center[1] - brow_center[1])) if brow_center is not None and nose_base_center is not None else None,
        "nose_base_to_chin_height": abs(float(chin_center[1] - nose_base_center[1])) if chin_center is not None and nose_base_center is not None else None,
        "eye_center_to_chin_height": abs(float(chin_center[1] - eye_center[1])) if chin_center is not None else None,
        "forehead_proxy_to_brow_height": abs(float(brow_center[1] - y1)) if brow_center is not None else None,
        "upper_third_proxy": abs(float(nose_base_center[1] - brow_center[1])) if brow_center is not None and nose_base_center is not None else None,
        "middle_third_proxy": abs(float(nose_base_center[1] - eye_center[1])) if nose_base_center is not None else None,
        "lower_third_proxy": abs(float(chin_center[1] - nose_base_center[1])) if chin_center is not None and nose_base_center is not None else None,
        "intercanthal_distance": intercanthal_distance,
        "outercanthal_distance": outercanthal_distance,
        "left_eye_width": left_eye_width,
        "right_eye_width": right_eye_width,
        "mean_eye_width": _mean([left_eye_width, right_eye_width]),
        "left_eye_height": left_eye_height,
        "right_eye_height": right_eye_height,
        "mean_eye_height": _mean([left_eye_height, right_eye_height]),
        "nose_width": nose_width,
        "nose_base_width": nose_base_width,
        "philtrum_width_proxy": _to_float(context.get("philtrum_width_proxy")),
        "chin_width_proxy": _to_float(context.get("chin_width_proxy")),
        "jaw_width_proxy": _distance(jaw_left, jaw_right),
        "cheekbone_width_proxy": _distance(cheekbone_left, cheekbone_right),
        "left_right_eye_y_diff": abs(float(left_eye[1] - right_eye[1])),
        "left_right_eye_width_diff": _abs_diff(left_eye_width, right_eye_width),
        "left_right_eye_height_diff": _abs_diff(left_eye_height, right_eye_height),
        "left_right_brow_height_diff": _abs_diff(
            abs(float(left_eye[1] - left_brow[1])) if left_brow is not None else None,
            abs(float(right_eye[1] - right_brow[1])) if right_brow is not None else None,
        ),
        "left_right_mouth_corner_y_diff": abs(float(left_mouth[1] - right_mouth[1])),
        "left_right_mouth_to_nose_diff": _abs_diff(_distance(left_mouth, nose), _distance(right_mouth, nose)),
        "left_right_eye_to_nose_diff": _abs_diff(_distance(left_eye, nose), _distance(right_eye, nose)),
        "nose_tip_midline_offset": nose_tip_midline_offset,
        "mouth_center_midline_offset": mouth_center_midline_offset,
        "chin_center_midline_offset": chin_center_midline_offset,
        "philtrum_midline_offset_proxy": philtrum_midline_offset_proxy,
        "left_brow_to_eye_distance": abs(float(left_eye[1] - left_brow[1])) if left_brow is not None else None,
        "right_brow_to_eye_distance": abs(float(right_eye[1] - right_brow[1])) if right_brow is not None else None,
        "mean_brow_to_eye_distance": _mean([
            abs(float(left_eye[1] - left_brow[1])) if left_brow is not None else None,
            abs(float(right_eye[1] - right_brow[1])) if right_brow is not None else None,
        ]),
        "eye_area_proxy_left": (left_eye_width * left_eye_height) if left_eye_width is not None and left_eye_height is not None else None,
        "eye_area_proxy_right": (right_eye_width * right_eye_height) if right_eye_width is not None and right_eye_height is not None else None,
        "eye_area_proxy_mean": _mean([
            (left_eye_width * left_eye_height) if left_eye_width is not None and left_eye_height is not None else None,
            (right_eye_width * right_eye_height) if right_eye_width is not None and right_eye_height is not None else None,
        ]),
        "nose_length": _distance(nose_base_center, nose),
        "nose_tip_to_upper_lip_distance": _distance(nose, upper_lip_top),
        "philtrum_height": abs(float(upper_lip_top[1] - nose_base_center[1])) if upper_lip_top is not None and nose_base_center is not None else None,
        "upper_lip_height": abs(float(mouth_center[1] - upper_lip_top[1])) if upper_lip_top is not None else None,
        "lower_lip_height": abs(float(lower_lip_bottom[1] - mouth_center[1])) if lower_lip_bottom is not None else None,
        "chin_height": abs(float(chin_center[1] - lower_lip_bottom[1])) if chin_center is not None and lower_lip_bottom is not None else None,
    }
    validation = context.get("semantic_validation")
    if isinstance(validation, dict) and not bool(validation.get("passed", False)):
        for feature_name in SEMANTIC_UNRELIABLE_RAW_FEATURES:
            if feature_name in result:
                result[feature_name] = None
    return result


def extract_normalized_measurements(
    raw_measurements: dict[str, float | None],
) -> dict[str, float | None]:
    face_width = raw_measurements.get("raw_face_width_px")
    face_height = raw_measurements.get("raw_face_height_px")
    mouth_width = raw_measurements.get("raw_mouth_width_px")
    mean_eye_width = raw_measurements.get("mean_eye_width")
    mean_eye_height = raw_measurements.get("mean_eye_height")
    upper = raw_measurements.get("upper_third_proxy")
    middle = raw_measurements.get("middle_third_proxy")
    lower = raw_measurements.get("lower_third_proxy")
    thirds = [value for value in (upper, middle, lower) if value is not None]
    thirds_balance_index = None
    if len(thirds) == 3 and max(thirds) > 0:
        normalized = np.asarray(thirds, dtype=np.float64) / float(sum(thirds))
        thirds_balance_index = float(max(0.0, 1.0 - min(float(np.std(normalized, ddof=0)) * 3.0, 1.0)))

    return {
        "ratio_face_aspect": _safe_divide(face_width, face_height),
        "ratio_eye_distance_to_face_width": _safe_divide(raw_measurements.get("raw_eye_distance_px"), face_width),
        "ratio_mouth_width_to_face_width": _safe_divide(mouth_width, face_width),
        "ratio_eye_to_nose_to_face_height": _safe_divide(raw_measurements.get("raw_eye_to_nose_px"), face_height),
        "ratio_nose_to_mouth_to_face_height": _safe_divide(raw_measurements.get("raw_nose_to_mouth_px"), face_height),
        "ratio_eye_to_mouth_to_face_height": _safe_divide(raw_measurements.get("raw_eye_to_mouth_px"), face_height),
        "ratio_eye_distance_to_mouth_width": _safe_divide(raw_measurements.get("raw_eye_distance_px"), mouth_width),
        "ratio_upper_face_height_to_face_height": _safe_divide(raw_measurements.get("raw_upper_face_height_px"), face_height),
        "ratio_lower_face_height_to_face_height": _safe_divide(raw_measurements.get("raw_lower_face_height_px"), face_height),
        "ratio_forehead_height_proxy_to_face_height": _safe_divide(raw_measurements.get("raw_forehead_height_proxy_px"), face_height),
        "ratio_jaw_height_proxy_to_face_height": _safe_divide(raw_measurements.get("raw_jaw_height_proxy_px"), face_height),
        "ratio_nose_center_offset_to_face_width": _safe_divide(raw_measurements.get("raw_nose_center_offset_px"), face_width),
        "ratio_mouth_center_offset_to_face_width": _safe_divide(raw_measurements.get("raw_mouth_center_offset_px"), face_width),
        "ratio_upper_third_to_middle_third": _safe_divide(upper, middle),
        "ratio_middle_third_to_lower_third": _safe_divide(middle, lower),
        "ratio_nose_base_to_mouth_to_lower_third": _safe_divide(raw_measurements.get("nose_base_to_mouth_center_height"), lower),
        "ratio_mouth_to_chin_to_lower_third": _safe_divide(raw_measurements.get("mouth_center_to_chin_height"), lower),
        "thirds_balance_index": thirds_balance_index,
        "ratio_intercanthal_to_eye_width": _safe_divide(raw_measurements.get("intercanthal_distance"), mean_eye_width),
        "ratio_outercanthal_to_face_width": _safe_divide(raw_measurements.get("outercanthal_distance"), face_width),
        "ratio_nose_width_to_face_width": _safe_divide(raw_measurements.get("nose_width"), face_width),
        "ratio_mouth_width_to_nose_width": _safe_divide(mouth_width, raw_measurements.get("nose_width")),
        "ratio_chin_width_to_jaw_width": _safe_divide(raw_measurements.get("chin_width_proxy"), raw_measurements.get("jaw_width_proxy")),
        "ratio_cheekbone_width_to_jaw_width": _safe_divide(raw_measurements.get("cheekbone_width_proxy"), raw_measurements.get("jaw_width_proxy")),
        "ratio_eye_height_to_eye_width": _safe_divide(mean_eye_height, mean_eye_width),
        "ratio_eye_spacing_to_eye_width": _safe_divide(raw_measurements.get("intercanthal_distance"), mean_eye_width),
        "ratio_eye_spacing_to_face_width": _safe_divide(raw_measurements.get("intercanthal_distance"), face_width),
        "ratio_nose_length_to_face_height": _safe_divide(raw_measurements.get("nose_length"), face_height),
        "ratio_philtrum_to_lower_face_height": _safe_divide(raw_measurements.get("philtrum_height"), lower),
        "ratio_upper_lip_to_lower_lip": _safe_divide(raw_measurements.get("upper_lip_height"), raw_measurements.get("lower_lip_height")),
        "ratio_mouth_to_chin": _safe_divide(mouth_width, raw_measurements.get("mouth_center_to_chin_height")),
        "ratio_nose_to_philtrum": _safe_divide(raw_measurements.get("nose_length"), raw_measurements.get("philtrum_height")),
        "ratio_nose_width_to_mouth_width": _safe_divide(raw_measurements.get("nose_width"), mouth_width),
        "lip_thickness_ratio": _safe_divide(
            (raw_measurements.get("upper_lip_height") or 0.0) + (raw_measurements.get("lower_lip_height") or 0.0),
            mouth_width,
        ),
        "chin_taper_ratio": _safe_divide(raw_measurements.get("chin_width_proxy"), raw_measurements.get("jaw_width_proxy")),
    }


def extract_symmetry_features(
    raw_measurements: dict[str, float | None],
    normalized_measurements: dict[str, float | None],
) -> dict[str, float | None]:
    eye_distance = raw_measurements.get("raw_eye_distance_px")
    mouth_width = raw_measurements.get("raw_mouth_width_px")
    eye_delta = _safe_divide(raw_measurements.get("left_right_eye_to_nose_diff"), eye_distance)
    mouth_delta = _safe_divide(raw_measurements.get("left_right_mouth_to_nose_diff"), mouth_width)

    offsets = [
        normalized_measurements.get("ratio_nose_center_offset_to_face_width"),
        normalized_measurements.get("ratio_mouth_center_offset_to_face_width"),
        _safe_divide(raw_measurements.get("chin_center_midline_offset"), raw_measurements.get("raw_face_width_px")),
        _safe_divide(raw_measurements.get("philtrum_midline_offset_proxy"), raw_measurements.get("raw_face_width_px")),
    ]
    present_offsets = [float(value) for value in offsets if value is not None]
    center_offset_ratio = float(sum(present_offsets) / len(present_offsets)) if present_offsets else None

    brow_symmetry_ratio = None
    mean_brow_to_eye = raw_measurements.get("mean_brow_to_eye_distance")
    if mean_brow_to_eye is not None and mean_brow_to_eye > 0:
        brow_symmetry_ratio = max(
            0.0,
            1.0 - float(raw_measurements.get("left_right_brow_height_diff") or 0.0) / float(mean_brow_to_eye),
        )

    midline_alignment_index = None
    if present_offsets:
        clipped = [max(0.0, min(1.0, 1.0 - float(value))) for value in present_offsets]
        midline_alignment_index = float(sum(clipped) / len(clipped))

    upper_components = [
        eye_delta,
        _safe_divide(raw_measurements.get("left_right_eye_y_diff"), raw_measurements.get("raw_face_height_px")),
        _safe_divide(raw_measurements.get("left_right_eye_width_diff"), raw_measurements.get("mean_eye_width")),
        _safe_divide(raw_measurements.get("left_right_eye_height_diff"), raw_measurements.get("mean_eye_height")),
        (1.0 - brow_symmetry_ratio) if brow_symmetry_ratio is not None else None,
        _safe_divide(raw_measurements.get("nose_tip_midline_offset"), raw_measurements.get("raw_face_width_px")),
    ]
    lower_components = [
        mouth_delta,
        _safe_divide(raw_measurements.get("left_right_mouth_corner_y_diff"), raw_measurements.get("raw_face_height_px")),
        _safe_divide(raw_measurements.get("mouth_center_midline_offset"), raw_measurements.get("raw_face_width_px")),
        _safe_divide(raw_measurements.get("chin_center_midline_offset"), raw_measurements.get("raw_face_width_px")),
        _safe_divide(raw_measurements.get("philtrum_midline_offset_proxy"), raw_measurements.get("raw_face_width_px")),
    ]

    def _symmetry_index_from_components(values: list[float | None]) -> float | None:
        present = [float(value) for value in values if value is not None]
        if not present:
            return None
        clipped = [max(0.0, min(1.0, 1.0 - value)) for value in present]
        return float(sum(clipped) / len(clipped))

    symmetry_components = [value for value in (eye_delta, mouth_delta, center_offset_ratio) if value is not None]
    symmetry_index = _symmetry_index_from_components(symmetry_components)

    return {
        "sym_eye_nose_delta_ratio": eye_delta,
        "sym_mouth_nose_delta_ratio": mouth_delta,
        "sym_horizontal_center_offset_ratio": center_offset_ratio,
        "symmetry_index": symmetry_index,
        "brow_symmetry_ratio": brow_symmetry_ratio,
        "midline_alignment_index": midline_alignment_index,
        "upper_face_symmetry_index": _symmetry_index_from_components(upper_components),
        "lower_face_symmetry_index": _symmetry_index_from_components(lower_components),
    }


def extract_angle_features(
    landmarks: list[list[float]] | None,
    row: dict[str, Any],
    landmarks_dense: list[list[float]] | None = None,
    bbox: list[float] | None = None,
) -> dict[str, float | None]:
    result = empty_feature_dict(ANGLE_FEATURE_COLUMNS)
    result["pose_yaw_deg"] = _to_float(row.get("yaw"))
    result["pose_pitch_deg"] = _to_float(row.get("pitch"))
    result["pose_roll_deg"] = _to_float(row.get("roll"))

    if not landmarks or len(landmarks) < 5:
        return result

    context = _derive_dense_context(landmarks=landmarks, landmarks_dense=landmarks_dense, bbox=bbox)
    points = np.asarray(landmarks, dtype=np.float64)
    left_eye = context.get("left_eye_center") if isinstance(context.get("left_eye_center"), np.ndarray) else points[0]
    right_eye = context.get("right_eye_center") if isinstance(context.get("right_eye_center"), np.ndarray) else points[1]
    nose = context.get("nose_tip") if isinstance(context.get("nose_tip"), np.ndarray) else points[2]
    left_mouth = context.get("left_mouth") if isinstance(context.get("left_mouth"), np.ndarray) else points[3]
    right_mouth = context.get("right_mouth") if isinstance(context.get("right_mouth"), np.ndarray) else points[4]
    eye_center = context.get("eye_center") if isinstance(context.get("eye_center"), np.ndarray) else (left_eye + right_eye) / 2.0
    mouth_center = context.get("mouth_center") if isinstance(context.get("mouth_center"), np.ndarray) else (left_mouth + right_mouth) / 2.0
    left_brow = context.get("left_brow") if isinstance(context.get("left_brow"), np.ndarray) else None
    right_brow = context.get("right_brow") if isinstance(context.get("right_brow"), np.ndarray) else None
    nose_base_center = context.get("nose_base_center") if isinstance(context.get("nose_base_center"), np.ndarray) else None
    chin_center = context.get("chin_center") if isinstance(context.get("chin_center"), np.ndarray) else None
    jaw_left = context.get("jaw_left") if isinstance(context.get("jaw_left"), np.ndarray) else None
    jaw_right = context.get("jaw_right") if isinstance(context.get("jaw_right"), np.ndarray) else None

    eye_line = _vector_angle_degrees(left_eye, right_eye)
    mouth_line = _vector_angle_degrees(left_mouth, right_mouth)
    nose_axis = _vector_angle_degrees(nose_base_center, nose)
    philtrum_axis = _vector_angle_degrees(nose_base_center, mouth_center)
    chin_axis = _vector_angle_degrees(mouth_center, chin_center)
    brow_line = _vector_angle_degrees(left_brow, right_brow)
    jaw_left_angle = _vector_angle_degrees(jaw_left, chin_center)
    jaw_right_angle = _vector_angle_degrees(chin_center, jaw_right)

    result.update(
        {
            "angle_eye_line_deg": eye_line,
            "angle_mouth_line_deg": mouth_line,
            "angle_eye_center_to_mouth_center_deg": _vector_angle_degrees(eye_center, mouth_center),
            "angle_nose_to_mouth_center_deg": _vector_angle_degrees(nose, mouth_center),
            "canthal_tilt_deg": eye_line,
            "nose_axis_angle_deg": nose_axis,
            "philtrum_axis_angle_deg": philtrum_axis,
            "chin_axis_angle_deg": chin_axis,
            "brow_line_angle_deg": brow_line,
            "jaw_line_left_angle_proxy": jaw_left_angle,
            "jaw_line_right_angle_proxy": jaw_right_angle,
            "jaw_angle_asymmetry": _abs_diff(jaw_left_angle, jaw_right_angle),
            "face_centerline_vs_nose_angle": abs(90.0 - abs(nose_axis)) if nose_axis is not None else None,
            "eye_line_vs_mouth_line_diff": _abs_diff(eye_line, mouth_line),
            "nose_to_chin_axis_diff": _abs_diff(nose_axis, chin_axis),
            "lower_face_tilt_proxy": chin_axis,
        }
    )
    validation = context.get("semantic_validation")
    if isinstance(validation, dict) and not bool(validation.get("passed", False)):
        for feature_name in SEMANTIC_UNRELIABLE_ANGLE_FEATURES:
            if feature_name in result:
                result[feature_name] = None
    return result


def validate_feature_values(
    feature_values: dict[str, float | None],
    *,
    image_id: str,
) -> None:
    issues: list[str] = []

    for key, value in feature_values.items():
        if value is None:
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            issues.append(f"{key}=non_finite")
            continue
        if key in NONNEGATIVE_FEATURE_COLUMNS and numeric < 0.0:
            issues.append(f"{key}=negative")
        if key in UNIT_INTERVAL_COLUMNS and not (0.0 <= numeric <= 1.0):
            issues.append(f"{key}=outside_0_1")
        if key.startswith("angle_") or key.startswith("pose_") or key.endswith("_deg"):
            if abs(numeric) > 180.0:
                issues.append(f"{key}=outside_angle_range")

    if issues:
        raise ValueError(f"Invalid measurement values for image_id={image_id}: {', '.join(sorted(issues))}")


def geometry_feature_columns() -> list[str]:
    return RAW_MEASUREMENT_COLUMNS + NORMALIZED_MEASUREMENT_COLUMNS + SYMMETRY_FEATURE_COLUMNS + ANGLE_FEATURE_COLUMNS


def feature_registry_rows() -> list[dict[str, Any]]:
    implemented_features = geometry_feature_columns()
    rows: list[dict[str, Any]] = []
    for feature_name in implemented_features:
        if feature_name in RAW_MEASUREMENT_COLUMNS:
            family = "raw_measurements"
        elif feature_name in NORMALIZED_MEASUREMENT_COLUMNS:
            family = "normalized_ratios"
        elif feature_name in SYMMETRY_FEATURE_COLUMNS:
            family = "symmetry_metrics"
        else:
            family = "angle_metrics"

        if feature_name in REDUNDANT_FEATURES:
            status = "redundant"
        elif feature_name in ALIASES:
            status = "deprecated_but_kept_for_compatibility"
        elif feature_name.startswith(("raw_face_", "raw_eye_", "raw_mouth_", "raw_nose_", "ratio_face_", "ratio_eye_", "sym_")):
            status = "keep"
        else:
            status = "new"

        rows.append(
            {
                "feature_name": feature_name,
                "feature_family": family,
                "status": status,
                "alias_of": ALIASES.get(feature_name),
                "depends_on": [],
                "formula": FEATURE_FORMULAS.get(feature_name),
                "notes": (
                    "Compatibility field retained for downstream consumers."
                    if status == "deprecated_but_kept_for_compatibility"
                    else ("Exact duplicate retained only to avoid breaking old outputs." if status == "redundant" else None)
                ),
            }
        )

    # Requested but not yet reliable enough to emit as stable features.
    unsupported = [
        "philtrum_width_proxy",
    ]
    for row in rows:
        if row["feature_name"] in unsupported:
            row["status"] = "unstable"
            row["notes"] = "Proxy depends on heuristic dense landmark band selection; audit before promoting to stable downstream use."
    return rows


def feature_audit_payload() -> dict[str, Any]:
    rows = feature_registry_rows()
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    duplicates = [row for row in rows if row["status"] in {"redundant", "deprecated_but_kept_for_compatibility"}]
    unstable = [row for row in rows if row["status"] == "unstable"]

    return {
        "feature_count": len(rows),
        "status_counts": status_counts,
        "duplicates_and_aliases": [
            {
                "feature_name": row["feature_name"],
                "status": row["status"],
                "alias_of": row.get("alias_of"),
                "notes": row.get("notes"),
            }
            for row in duplicates
        ],
        "unstable_features": [
            {
                "feature_name": row["feature_name"],
                "notes": row.get("notes"),
            }
            for row in unstable
        ],
        "registry_version": "morphometric_feature_registry_v2",
    }
