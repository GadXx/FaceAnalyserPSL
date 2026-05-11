from __future__ import annotations

from typing import Any

from src.face_aesthetics.landmarks import LandmarkSourceResult


MESH3D_DIRECT_REGIONS = [
    "nose",
    "jaw",
    "chin",
    "face_oval",
]


def build_mesh3d_source_result(config: dict[str, Any]) -> LandmarkSourceResult:
    enabled = bool(config.get("enabled", False))
    backend = config.get("backend")
    if not enabled:
        return LandmarkSourceResult(
            source="3d_mesh",
            enabled=False,
            available=False,
            optional=True,
            unavailable_regions=MESH3D_DIRECT_REGIONS,
            dependency_message="disabled_in_config",
            metadata={"backend": backend},
        )
    return LandmarkSourceResult(
        source="3d_mesh",
        enabled=True,
        available=False,
        optional=True,
        unavailable_regions=MESH3D_DIRECT_REGIONS,
        warnings=["3d_mesh_backend_not_configured"],
        dependency_message="3d_mesh_backend_not_configured",
        metadata={"backend": backend},
    )
