from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


LANDMARK_COUNT = 106

REGIONS: dict[str, tuple[int, int]] = {
    "contour": (0, 32),
    "left_eye": (33, 42),
    "left_brow": (43, 51),
    "mouth": (52, 71),
    "nose": (72, 86),
    "right_eye": (87, 96),
    "right_brow": (97, 105),
}

CONTOUR_INDICES = tuple(range(0, 33))
LEFT_CONTOUR_INDICES = tuple(range(1, 17))
RIGHT_CONTOUR_INDICES = tuple(range(17, 33))

LEFT_EYE_INDICES = tuple(range(33, 43))
LEFT_BROW_INDICES = tuple(range(43, 52))
MOUTH_INDICES = tuple(range(52, 72))
NOSE_INDICES = tuple(range(72, 87))
RIGHT_EYE_INDICES = tuple(range(87, 97))
RIGHT_BROW_INDICES = tuple(range(97, 106))

MOUTH_LEFT_CORNER_INDEX = 52
LOWER_LIP_CENTER_BOTTOM_INDEX = 53
MOUTH_RIGHT_CORNER_INDEX = 61
UPPER_LIP_CENTER_TOP_INDEX = 71

UPPER_LIP_ARC_INDICES = (52, 63, 64, 71, 67, 68, 69, 61)
MOUTH_INNER_ZONE_INDICES = (65, 66, 62, 70)
LOWER_LIP_ARC_INDICES = (55, 56, 53, 59, 58)

NOSE_BRIDGE_INDICES = (72, 73, 74)
LEFT_NOSE_SIDE_INDICES = (75, 76, 77, 78, 79)
RIGHT_NOSE_SIDE_INDICES = (81, 82, 83, 84, 85)

ALARE_LEFT_INDEX = 77
NOSE_BASE_LEFT_INDEX = 78
SUBNASALE_LEFT_INDEX = 79
SUBNASALE_CENTER_INDEX = 80
ALARE_RIGHT_INDEX = 83
NOSE_BASE_RIGHT_INDEX = 84
SUBNASALE_RIGHT_INDEX = 85
NOSE_TIP_INDEX = 86


@dataclass(frozen=True)
class SemanticIndexSpec:
    name: str
    index: int
    region: str


SEMANTIC_INDEX_SPECS = {
    "mouth_left": SemanticIndexSpec("mouth_left", MOUTH_LEFT_CORNER_INDEX, "mouth"),
    "mouth_right": SemanticIndexSpec("mouth_right", MOUTH_RIGHT_CORNER_INDEX, "mouth"),
    "upper_lip_top": SemanticIndexSpec("upper_lip_top", UPPER_LIP_CENTER_TOP_INDEX, "mouth"),
    "lower_lip_bottom": SemanticIndexSpec("lower_lip_bottom", LOWER_LIP_CENTER_BOTTOM_INDEX, "mouth"),
    "nose_tip": SemanticIndexSpec("nose_tip", NOSE_TIP_INDEX, "nose"),
    "alare_left": SemanticIndexSpec("alare_left", ALARE_LEFT_INDEX, "nose"),
    "nose_base_left": SemanticIndexSpec("nose_base_left", NOSE_BASE_LEFT_INDEX, "nose"),
    "subnasale_left": SemanticIndexSpec("subnasale_left", SUBNASALE_LEFT_INDEX, "nose"),
    "subnasale_center": SemanticIndexSpec("subnasale_center", SUBNASALE_CENTER_INDEX, "nose"),
    "alare_right": SemanticIndexSpec("alare_right", ALARE_RIGHT_INDEX, "nose"),
    "nose_base_right": SemanticIndexSpec("nose_base_right", NOSE_BASE_RIGHT_INDEX, "nose"),
    "subnasale_right": SemanticIndexSpec("subnasale_right", SUBNASALE_RIGHT_INDEX, "nose"),
    "chin_center": SemanticIndexSpec("chin_center", 0, "contour"),
}


def has_106_landmarks(points: np.ndarray | None) -> bool:
    return points is not None and len(points) >= LANDMARK_COUNT


def region_indices(region_name: str) -> tuple[int, ...]:
    if region_name == "contour":
        return CONTOUR_INDICES
    if region_name == "left_contour":
        return LEFT_CONTOUR_INDICES
    if region_name == "right_contour":
        return RIGHT_CONTOUR_INDICES
    if region_name == "left_eye":
        return LEFT_EYE_INDICES
    if region_name == "right_eye":
        return RIGHT_EYE_INDICES
    if region_name == "left_brow":
        return LEFT_BROW_INDICES
    if region_name == "right_brow":
        return RIGHT_BROW_INDICES
    if region_name == "mouth":
        return MOUTH_INDICES
    if region_name == "nose":
        return NOSE_INDICES
    raise KeyError(f"Unknown InsightFace 106 region: {region_name}")


def region_name_for_index(index: int) -> str:
    for region_name, (start, end) in REGIONS.items():
        if start <= index <= end:
            return region_name
    return "unknown"


def point_for_index(points: np.ndarray | None, index: int) -> np.ndarray | None:
    if points is None or index < 0 or index >= len(points):
        return None
    point = np.asarray(points[index], dtype=np.float64)
    if point.shape != (2,):
        return None
    return point


def points_for_indices(
    points: np.ndarray | None,
    indices: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    if points is None:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 2), dtype=np.float64)
    collected_indices: list[int] = []
    collected_points: list[np.ndarray] = []
    for index in indices:
        point = point_for_index(points, int(index))
        if point is None:
            continue
        collected_indices.append(int(index))
        collected_points.append(point)
    if not collected_points:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 2), dtype=np.float64)
    return np.asarray(collected_indices, dtype=np.int32), np.asarray(collected_points, dtype=np.float64)


def points_for_region(points: np.ndarray | None, region_name: str) -> tuple[np.ndarray, np.ndarray]:
    return points_for_indices(points, region_indices(region_name))
