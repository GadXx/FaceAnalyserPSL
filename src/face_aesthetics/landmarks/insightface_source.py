from __future__ import annotations

from typing import Any

import numpy as np

from src.face_aesthetics.landmarks import LandmarkSourceResult
from src.face_aesthetics.landmarks.insightface_106 import has_106_landmarks


INSIGHTFACE_DIRECT_REGIONS = [
    "eyes",
    "brows",
    "nose",
    "lips",
]

INSIGHTFACE_PROXY_REGIONS = [
    "face_oval",
    "jaw",
    "cheekbones",
]

INSIGHTFACE_UNAVAILABLE_REGIONS = [
    "ears",
    "hairline",
    "skin_mask",
    "full_face_oval",
]


def build_insightface_source_result(
    *,
    landmarks_dense: list[list[float]] | np.ndarray | None,
    config: dict[str, Any],
) -> LandmarkSourceResult:
    enabled = bool(config.get("enabled", True))
    required = bool(config.get("required", False))
    dense = np.asarray(landmarks_dense, dtype=np.float64) if landmarks_dense is not None else None
    available = bool(enabled and has_106_landmarks(dense))
    landmark_count = int(len(dense)) if dense is not None else 0
    warnings: list[str] = []
    if enabled and not available:
        warnings.append("insightface_dense_landmarks_unavailable")
    return LandmarkSourceResult(
        source="insightface_106",
        enabled=enabled,
        available=available,
        required=required,
        optional=not required,
        coordinate_space="analysis_image",
        confidence=1.0 if available else None,
        landmark_count=landmark_count if available else 0,
        landmarks=dense.tolist() if available and dense is not None else [],
        available_regions=INSIGHTFACE_DIRECT_REGIONS + INSIGHTFACE_PROXY_REGIONS if available else [],
        direct_regions=INSIGHTFACE_DIRECT_REGIONS if available else [],
        proxy_regions=INSIGHTFACE_PROXY_REGIONS if available else [],
        unavailable_regions=INSIGHTFACE_UNAVAILABLE_REGIONS,
        warnings=warnings,
        dependency_message=None if available else "InsightFace 106 landmarks are required as the stable base geometry source.",
        metadata={
            "landmark_count_expected": 106,
            "backward_compatible_anchor_source": True,
        },
    )
