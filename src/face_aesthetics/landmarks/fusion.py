from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.face_aesthetics.data.io import write_json
from src.face_aesthetics.landmarks import LandmarkSourceResult
from src.face_aesthetics.landmarks.face_parsing_source import build_face_parsing_source_result
from src.face_aesthetics.landmarks.mediapipe_source import (
    build_mediapipe_source_result,
    write_mediapipe_debug_artifacts,
)
from src.face_aesthetics.landmarks.mesh3d_source import build_mesh3d_source_result
from src.face_aesthetics.features.extraction import feature_registry_rows
from src.face_aesthetics.utils.config import load_named_configs


REGION_PRIORITY = {
    "face_oval": "face_parsing",
    "ears": "face_parsing",
    "hairline": "face_parsing",
    "skin_mask": "face_parsing",
    "eyes": "mediapipe_face_mesh",
    "brows": "mediapipe_face_mesh",
    "lips": "mediapipe_face_mesh",
    "nose": "mediapipe_face_mesh",
    "jaw": "3d_mesh",
    "chin": "3d_mesh",
    "cheekbones": "mediapipe_face_mesh",
}


OPTIONAL_ONLY_FEATURES = {
    "ears": [
        "visible_left_ear_height",
        "visible_right_ear_height",
        "visible_ear_width",
        "ear_y_alignment",
        "ear_to_face_height_ratio",
    ],
    "hairline": [
        "hairline_to_brow_height",
        "forehead_visible_height",
    ],
    "skin_mask": [
        "skin_mask_face_area",
        "full_face_oval_width",
        "full_face_oval_height",
        "face_oval_aspect_ratio",
    ],
}


def collect_landmark_sources(
    *,
    project_root: Path,
    image_bgr: np.ndarray | None,
    landmarks_dense: list[list[float]] | np.ndarray | None,
) -> list[LandmarkSourceResult]:
    configs = load_named_configs(project_root / "configs")
    landmark_config = configs.get("landmarks", {}).get("landmarks", {}).get("landmark_sources", {})
    return [
        build_mediapipe_source_result(
            image_bgr=image_bgr,
            config=dict(landmark_config.get("mediapipe_face_mesh", {})),
            project_root=project_root,
        ),
        build_face_parsing_source_result(
            image_bgr=image_bgr,
            config=dict(landmark_config.get("face_parsing", {})),
        ),
        build_mesh3d_source_result(dict(landmark_config.get("optional_3d_mesh", {}))),
    ]


def build_landmark_capability_report(
    *,
    project_root: Path,
    image_bgr: np.ndarray | None,
    landmarks_dense: list[list[float]] | np.ndarray | None,
) -> dict[str, Any]:
    sources = collect_landmark_sources(
        project_root=project_root,
        image_bgr=image_bgr,
        landmarks_dense=landmarks_dense,
    )
    source_reports = {result.source: result.to_report() for result in sources}
    region_reports: dict[str, dict[str, Any]] = {}
    all_regions = sorted(
        {
            *REGION_PRIORITY.keys(),
            *[region for result in sources for region in result.available_regions],
            *[region for result in sources for region in result.unavailable_regions],
        }
    )
    for region in all_regions:
        direct_sources = [result.source for result in sources if region in result.direct_regions and result.available]
        proxy_sources = [result.source for result in sources if region in result.proxy_regions and result.available]
        preferred = REGION_PRIORITY.get(region)
        status = "unavailable_without_extra_model"
        if direct_sources:
            status = "direct"
        elif proxy_sources:
            status = "proxy"
        region_reports[region] = {
            "status": status,
            "preferred_source": preferred,
            "available_sources": direct_sources + proxy_sources,
            "direct_sources": direct_sources,
            "proxy_sources": proxy_sources,
        }
    registry_rows = feature_registry_rows()
    runtime_available_sources = {result.source for result in sources if result.available}
    feature_counts_by_source: dict[str, int] = {}
    runtime_available_feature_counts: dict[str, int] = {}
    for row in registry_rows:
        source = str(row.get("source", "unknown"))
        feature_counts_by_source[source] = feature_counts_by_source.get(source, 0) + 1
        if source in runtime_available_sources:
            runtime_available_feature_counts[source] = runtime_available_feature_counts.get(source, 0) + 1
    report = {
        "sources": source_reports,
        "mediapipe_enabled": bool(source_reports.get("mediapipe_face_mesh", {}).get("enabled")),
        "mediapipe_available": bool(source_reports.get("mediapipe_face_mesh", {}).get("available")),
        "mediapipe_landmark_count": int(source_reports.get("mediapipe_face_mesh", {}).get("landmark_count", 0) or 0),
        "mediapipe_unavailable_reason": source_reports.get("mediapipe_face_mesh", {}).get("metadata", {}).get("mediapipe_unavailable_reason"),
        "available_regions": sorted([region for region, payload in region_reports.items() if payload["status"] != "unavailable_without_extra_model"]),
        "unavailable_regions": sorted([region for region, payload in region_reports.items() if payload["status"] == "unavailable_without_extra_model"]),
        "region_capabilities": region_reports,
        "feature_counts_by_source": feature_counts_by_source,
        "runtime_available_feature_counts": runtime_available_feature_counts,
        "ears": region_reports.get("ears", {"status": "unavailable_without_extra_model"}),
        "optional_only_features": OPTIONAL_ONLY_FEATURES,
    }
    return report


def write_landmark_capability_artifacts(
    *,
    project_root: Path,
    capability_report: dict[str, Any],
    image_bgr: np.ndarray | None = None,
) -> dict[str, str | None]:
    configs = load_named_configs(project_root / "configs")
    landmark_config = configs.get("landmarks", {}).get("landmarks", {})
    report_paths = landmark_config.get("report_paths", {})
    capability_path = (project_root / str(report_paths.get("capability_report_path", "reports/metrics/landmark_capability_report.json"))).resolve()
    mediapipe_runtime_diagnostics_path = (
        project_root / str(report_paths.get("mediapipe_runtime_diagnostics_path", "reports/metrics/mediapipe_runtime_diagnostics.json"))
    ).resolve()
    mediapipe_debug_path = (project_root / str(report_paths.get("mediapipe_debug_landmarks_path", "reports/debug/mediapipe_landmarks.json"))).resolve()
    write_json(capability_report, capability_path)
    mediapipe_payload = capability_report.get("sources", {}).get("mediapipe_face_mesh", {})
    mediapipe_metadata = dict(mediapipe_payload.get("metadata", {}))
    mediapipe_diagnostics = {
        "requested_delegate": mediapipe_metadata.get("requested_delegate"),
        "selected_delegate": mediapipe_metadata.get("selected_delegate"),
        "fallback_to_cpu": mediapipe_metadata.get("fallback_to_cpu"),
        "fallback_reason": mediapipe_metadata.get("fallback_reason"),
        "mediapipe_available": mediapipe_payload.get("available"),
        "model_asset_path": mediapipe_metadata.get("model_asset_path"),
        "landmark_count": mediapipe_payload.get("landmark_count"),
        "runtime_backend": mediapipe_metadata.get("runtime_backend"),
        "runtime_warnings": list(mediapipe_payload.get("warnings", [])),
    }
    write_json(mediapipe_diagnostics, mediapipe_runtime_diagnostics_path)
    mediapipe_landmarks = mediapipe_payload.get("landmarks")
    mediapipe_debug = None
    mediapipe_indices = None
    mediapipe_regions = None
    if mediapipe_landmarks and image_bgr is not None:
        written = write_mediapipe_debug_artifacts(
            project_root=project_root,
            report_paths=report_paths,
            image_bgr=image_bgr,
            landmarks_xyz=mediapipe_landmarks,
        )
        mediapipe_debug = written["mediapipe_debug_landmarks_path"]
        mediapipe_indices = written["mediapipe_debug_indices_image_path"]
        mediapipe_regions = written["mediapipe_region_report_path"]
    return {
        "capability_report_path": capability_path.relative_to(project_root).as_posix(),
        "mediapipe_runtime_diagnostics_path": mediapipe_runtime_diagnostics_path.relative_to(project_root).as_posix(),
        "mediapipe_debug_landmarks_path": mediapipe_debug,
        "mediapipe_debug_indices_image_path": mediapipe_indices,
        "mediapipe_region_report_path": mediapipe_regions,
    }
