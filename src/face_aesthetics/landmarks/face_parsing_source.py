from __future__ import annotations

from typing import Any

import numpy as np

from src.face_aesthetics.landmarks import LandmarkSourceResult


PARSING_DIRECT_REGIONS = [
    "skin_mask",
    "hairline",
    "ears",
    "face_oval",
    "lips",
    "brows",
    "eyes",
    "nose",
]


def build_face_parsing_source_result(
    *,
    image_bgr: np.ndarray | None,
    config: dict[str, Any],
) -> LandmarkSourceResult:
    enabled = bool(config.get("enabled", False))
    classes = list(config.get("classes", []))
    if not enabled:
        return LandmarkSourceResult(
            source="face_parsing",
            enabled=False,
            available=False,
            optional=True,
            unavailable_regions=PARSING_DIRECT_REGIONS,
            dependency_message="disabled_in_config",
            metadata={"classes": classes},
        )
    if image_bgr is None:
        return LandmarkSourceResult(
            source="face_parsing",
            enabled=True,
            available=False,
            optional=True,
            unavailable_regions=PARSING_DIRECT_REGIONS,
            dependency_message="image_not_provided_for_runtime_extraction",
            metadata={"classes": classes},
        )
    return LandmarkSourceResult(
        source="face_parsing",
        enabled=True,
        available=False,
        optional=True,
        unavailable_regions=PARSING_DIRECT_REGIONS,
        warnings=["face_parsing_backend_not_configured"],
        dependency_message="face_parsing_backend_not_configured",
        metadata={"classes": classes},
    )
