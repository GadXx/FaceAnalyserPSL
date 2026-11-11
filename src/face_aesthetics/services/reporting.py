from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.face_aesthetics.data.io import write_json
from src.face_aesthetics.features.extraction import (
    derive_dense_visual_context,
    derive_semantic_landmark_bundle,
    feature_registry_rows,
)
from src.face_aesthetics.utils.filesystem import ensure_parent_directory


def _finite_values(values: dict[str, float | None]) -> list[float]:
    result: list[float] = []
    for value in values.values():
        if value is None:
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            result.append(numeric)
    return result


def compute_optional_composite_indices(
    *,
    normalized_measurements: dict[str, float | None],
    symmetry_features: dict[str, float | None],
    z_scores: dict[str, float | None],
    mahalanobis_distance: float | None,
) -> dict[str, float | None]:
    normalized_values = _finite_values(normalized_measurements)
    z_values = [abs(float(value)) for value in z_scores.values() if value is not None and np.isfinite(float(value))]

    symmetry_index = symmetry_features.get("symmetry_index")

    proportion_consistency_index = None
    if normalized_values:
        proportion_consistency_index = float(
            max(0.0, 1.0 - min(float(np.std(normalized_values, ddof=0)), 1.0))
        )

    reference_deviation_index = None
    if mahalanobis_distance is not None and np.isfinite(float(mahalanobis_distance)):
        reference_deviation_index = float(1.0 / (1.0 + max(float(mahalanobis_distance), 0.0)))

    morphometric_harmony_index = None
    harmony_components = [
        value
        for value in (
            float(symmetry_index) if symmetry_index is not None else None,
            proportion_consistency_index,
            reference_deviation_index,
        )
        if value is not None and np.isfinite(float(value))
    ]
    if harmony_components:
        morphometric_harmony_index = float(sum(harmony_components) / len(harmony_components))

    return {
        "symmetry_index": float(symmetry_index) if symmetry_index is not None else None,
        "proportion_consistency_index": proportion_consistency_index,
        "reference_deviation_index": reference_deviation_index,
        "morphometric_harmony_index": morphometric_harmony_index,
        "mean_absolute_z_score": (float(np.mean(z_values)) if z_values else None),
    }


def build_reporting_limitations(
    *,
    quality_pass: bool,
    quality_fail_reason: str | None,
    provider_diagnostics: dict[str, Any],
    reference_group_sample_count: int | None,
    reference_warning: str | None = None,
) -> list[str]:
    limitations = [
        "Outputs are descriptive morphometric statistics, not attractiveness or beauty scores.",
        "Percentile positions are estimated from stored reference percentiles and may be coarse between control points.",
        "Reference comparisons depend on the selected reference group and available sample counts.",
        "philtrum_width_proxy is a 2D proxy derived from lip-region landmarks and should be treated as lower-confidence than directly indexed nose-base or mouth-corner measurements.",
    ]
    if not quality_pass:
        limitations.append(
            "Quality gate did not fully pass; geometric measurements may be less reliable for this sample."
            + (f" quality_fail_reason={quality_fail_reason}." if quality_fail_reason else "")
        )
    selected_provider = provider_diagnostics.get("selected_execution_provider")
    if selected_provider == "CPUExecutionProvider":
        limitations.append("Preprocessing used CPUExecutionProvider; runtime path was valid but not GPU-accelerated.")
    if reference_group_sample_count is not None and reference_group_sample_count < 50:
        limitations.append(
            f"Selected reference group has limited sample support (sample_count={reference_group_sample_count})."
        )
    if reference_warning:
        limitations.append(reference_warning)
    return limitations


def build_formula_transparency() -> dict[str, Any]:
    grouped: dict[str, dict[str, str]] = {
        "raw_measurements": {},
        "normalized_ratios": {},
        "symmetry_metrics": {},
        "angle_metrics": {},
    }
    for row in feature_registry_rows():
        family = row.get("feature_family")
        feature_name = row.get("feature_name")
        formula = row.get("formula")
        if family in grouped and feature_name and formula:
            grouped[family][str(feature_name)] = str(formula)

    grouped["reference_comparison"] = {
        "z_score": "(feature_value - reference_mean) / max(reference_std, epsilon)",
        "mahalanobis_distance": "sqrt((x - mu)^T * inv(cov + lambda*I) * (x - mu))",
        "percentile_position": "piecewise-linear interpolation between stored reference percentiles",
    }
    grouped["optional_composite_indices"] = {
        "symmetry_index": "reported directly from symmetry metrics",
        "proportion_consistency_index": "1 - std(normalized ratios), clipped to [0, 1]",
        "reference_deviation_index": "1 / (1 + mahalanobis_distance)",
        "morphometric_harmony_index": "mean of available symmetry/proportion/reference components",
    }
    return grouped


def _feature_registry_lookup() -> dict[str, dict[str, Any]]:
    return {str(row["feature_name"]): row for row in feature_registry_rows()}


def _expanded_geometry_groups() -> dict[str, list[str]]:
    return {
        "face_oval": [
            "mediapipe_face_oval_width",
            "mediapipe_face_oval_height",
            "mediapipe_face_oval_aspect_ratio",
            "mediapipe_face_contour_symmetry_index",
        ],
        "eyes": [
            "mediapipe_left_eye_width",
            "mediapipe_right_eye_width",
            "mediapipe_left_eye_height",
            "mediapipe_right_eye_height",
            "mediapipe_left_eye_area_proxy",
            "mediapipe_right_eye_area_proxy",
            "mediapipe_eye_shape_symmetry_index",
            "mediapipe_left_upper_eyelid_curve_proxy",
            "mediapipe_right_upper_eyelid_curve_proxy",
        ],
        "brows": [
            "mediapipe_left_brow_width",
            "mediapipe_right_brow_width",
            "mediapipe_left_brow_angle_deg",
            "mediapipe_right_brow_angle_deg",
            "mediapipe_brow_symmetry_index",
        ],
        "nose": [
            "mediapipe_nose_width",
            "mediapipe_nose_base_width",
            "mediapipe_nose_length",
            "mediapipe_nose_bridge_width_proxy",
            "mediapipe_nose_axis_angle_deg",
            "mediapipe_nose_symmetry_index",
        ],
        "mouth_lips": [
            "mediapipe_mouth_width",
            "mediapipe_mouth_height",
            "mediapipe_upper_lip_height",
            "mediapipe_lower_lip_height",
            "mediapipe_total_lip_height",
            "mediapipe_lip_area_ratio",
            "mediapipe_mouth_line_angle_deg",
            "mediapipe_lip_symmetry_index",
        ],
        "cheekbones_midface": [
            "mediapipe_cheekbone_width_proxy",
            "mediapipe_midface_width_proxy",
            "mediapipe_midface_height_proxy",
            "mediapipe_cheekbone_width_to_jaw_ratio",
        ],
        "jaw_chin": [
            "mediapipe_jaw_width_proxy",
            "mediapipe_chin_width_to_jaw_width",
            "mediapipe_jaw_taper_ratio",
            "mediapipe_chin_taper_ratio",
        ],
    }


def build_expanded_geometry_payload(
    *,
    raw_measurements: dict[str, float | None],
    normalized_measurements: dict[str, float | None],
    symmetry_features: dict[str, float | None],
    angle_features: dict[str, float | None],
) -> dict[str, Any]:
    values = {
        **raw_measurements,
        **normalized_measurements,
        **symmetry_features,
        **angle_features,
    }
    registry = _feature_registry_lookup()
    groups: dict[str, list[dict[str, Any]]] = {}
    for group_name, feature_names in _expanded_geometry_groups().items():
        rows: list[dict[str, Any]] = []
        for feature_name in feature_names:
            metadata = registry.get(feature_name, {})
            rows.append(
                {
                    "feature_name": feature_name,
                    "value": values.get(feature_name),
                    "source": metadata.get("source"),
                    "reliability": metadata.get("reliability"),
                    "units": metadata.get("units"),
                    "component": metadata.get("component"),
                    "recommended_for_scorer": metadata.get("recommended_for_scorer"),
                    "exclusion_reason": metadata.get("exclusion_reason"),
                }
            )
        groups[group_name] = rows

    unavailable_without_face_parsing = []
    for feature_name in ("visible_left_ear_height", "visible_right_ear_height", "visible_ear_width", "hairline_to_brow_height", "skin_mask_face_area"):
        metadata = registry.get(feature_name, {})
        unavailable_without_face_parsing.append(
            {
                "feature_name": feature_name,
                "source": metadata.get("source", "face_parsing"),
                "reliability": metadata.get("reliability", "unavailable_without_extra_model"),
                "status": "unavailable_without_face_parsing",
            }
        )

    mediapipe_all = [name for name in values if str(name).startswith("mediapipe_")]
    mediapipe_non_null = [name for name in mediapipe_all if values.get(name) is not None and np.isfinite(float(values.get(name)))]

    reliability_counts: dict[str, int] = {}
    for feature_name in mediapipe_all:
        reliability = str(registry.get(feature_name, {}).get("reliability", "unknown"))
        reliability_counts[reliability] = reliability_counts.get(reliability, 0) + 1

    return {
        "available": bool(mediapipe_non_null),
        "mediapipe_feature_count": int(len(mediapipe_all)),
        "mediapipe_non_null_feature_count": int(len(mediapipe_non_null)),
        "reliability_counts": reliability_counts,
        "groups": groups,
        "unavailable_without_face_parsing": unavailable_without_face_parsing,
    }


def build_reference_selection_transparency(
    *,
    group_override: str | None,
    resolved_group_values: dict[str, str],
    selected_group_key: str,
    selected_group: dict[str, Any],
    reference_groupings: list[list[str]],
) -> dict[str, Any]:
    return {
        "group_override": group_override,
        "resolved_group_values": resolved_group_values,
        "selection_strategy": "Longest matching configured grouping with exact group_key match in reference artifact.",
        "configured_groupings": reference_groupings,
        "selected_group_key": selected_group_key,
        "selected_group_columns": selected_group.get("group_columns"),
        "selected_group_values": selected_group.get("group_values"),
        "selected_group_sample_count": selected_group.get("sample_count"),
        "selected_group_complete_case_count": selected_group.get("complete_case_count"),
    }


def _point_payload(point: np.ndarray | list[float] | tuple[float, float]) -> dict[str, float]:
    array = np.asarray(point, dtype=np.float64)
    return {
        "x": float(array[0]),
        "y": float(array[1]),
    }


def _context_point(context: dict[str, Any], key: str) -> np.ndarray | None:
    value = context.get(key)
    return value if isinstance(value, np.ndarray) else None


def _line_entry(
    *,
    key: str,
    start: str,
    end: str,
    group: str,
    kind: str,
    layers: list[str],
    metric_key: str | None = None,
    supported_features: list[str] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "start": start,
        "end": end,
        "group": group,
        "kind": kind,
        "layers": layers,
        "metric_key": metric_key,
        "supported_features": supported_features or ([] if metric_key is None else [metric_key]),
        "label": label,
    }


def build_visualization_payload(
    *,
    image_size: tuple[int, int] | None,
    bbox: list[float] | None,
    landmarks: list[list[float]] | None,
    landmarks_dense: list[list[float]] | None,
    mediapipe_landmarks: list[list[float]] | None = None,
    mediapipe_regions: dict[str, list[int]] | None = None,
    coordinate_space: str,
    image_path: Path | None = None,
) -> dict[str, Any]:
    notes: list[str] = []
    detector_bbox = bbox
    anchors: dict[str, dict[str, float]] | None = None
    derived_points: dict[str, dict[str, float]] | None = None
    named_points: dict[str, dict[str, float]] | None = None
    landmark_points: list[dict[str, Any]] = []
    raw_landmarks_debug: list[dict[str, Any]] = []
    semantic_points: dict[str, dict[str, float]] | None = None
    semantic_points_debug: list[dict[str, Any]] = []
    semantic_validation: dict[str, Any] | None = None
    lines: list[dict[str, Any]] = []
    axes: list[dict[str, Any]] = []
    guides: list[dict[str, Any]] = []
    groups: dict[str, list[str]] = {}
    semantic_face_extents: dict[str, Any] | None = None
    display_face_box: list[float] | None = None

    if bbox is None:
        notes.append("Bounding box was not available in preprocessing output.")
    if not landmarks or len(landmarks) < 5:
        notes.append("Five anchor landmarks were not available; overlay is limited.")
    else:
        semantic_bundle = derive_semantic_landmark_bundle(
            landmarks=landmarks,
            landmarks_dense=landmarks_dense,
            bbox=bbox,
        )
        landmark_points = list(semantic_bundle.get("raw_landmarks", []))
        raw_landmarks_debug = list(semantic_bundle.get("raw_landmarks_debug", []))
        semantic_points = dict(semantic_bundle.get("semantic_points", {}))
        semantic_points_debug = list(semantic_bundle.get("semantic_points_debug", []))
        semantic_validation = (
            dict(semantic_bundle.get("semantic_validation", {}))
            if isinstance(semantic_bundle.get("semantic_validation"), dict)
            else None
        )

        points = np.asarray(landmarks[:5], dtype=np.float64)
        dense_context = derive_dense_visual_context(
            landmarks=landmarks,
            landmarks_dense=landmarks_dense,
            bbox=bbox,
        )
        semantic_face_extents = (
            dict(dense_context.get("semantic_face_extents", {}))
            if isinstance(dense_context.get("semantic_face_extents"), dict)
            else None
        )
        display_face_box = (
            [float(value) for value in dense_context.get("display_face_box", [])]
            if isinstance(dense_context.get("display_face_box"), list) and len(dense_context.get("display_face_box", [])) == 4
            else bbox
        )
        left_eye = np.asarray(
            [semantic_points.get("left_eye_center", {}).get("x", points[0][0]), semantic_points.get("left_eye_center", {}).get("y", points[0][1])],
            dtype=np.float64,
        )
        right_eye = np.asarray(
            [semantic_points.get("right_eye_center", {}).get("x", points[1][0]), semantic_points.get("right_eye_center", {}).get("y", points[1][1])],
            dtype=np.float64,
        )
        nose = np.asarray(
            [semantic_points.get("nose_tip", {}).get("x", points[2][0]), semantic_points.get("nose_tip", {}).get("y", points[2][1])],
            dtype=np.float64,
        )
        left_mouth = np.asarray(
            [semantic_points.get("mouth_left", {}).get("x", points[3][0]), semantic_points.get("mouth_left", {}).get("y", points[3][1])],
            dtype=np.float64,
        )
        right_mouth = np.asarray(
            [semantic_points.get("mouth_right", {}).get("x", points[4][0]), semantic_points.get("mouth_right", {}).get("y", points[4][1])],
            dtype=np.float64,
        )
        eye_center = np.asarray(
            [semantic_points.get("eye_center", {}).get("x", ((left_eye + right_eye) / 2.0)[0]), semantic_points.get("eye_center", {}).get("y", ((left_eye + right_eye) / 2.0)[1])],
            dtype=np.float64,
        )
        mouth_center = np.asarray(
            [semantic_points.get("mouth_center", {}).get("x", ((left_mouth + right_mouth) / 2.0)[0]), semantic_points.get("mouth_center", {}).get("y", ((left_mouth + right_mouth) / 2.0)[1])],
            dtype=np.float64,
        )
        if display_face_box and len(display_face_box) == 4:
            x1, y1, x2, y2 = [float(value) for value in display_face_box]
            face_center_x = (x1 + x2) / 2.0
            axis_top = np.asarray([face_center_x, y1], dtype=np.float64)
            axis_bottom = np.asarray([face_center_x, y2], dtype=np.float64)
        else:
            face_center_x = float((eye_center[0] + mouth_center[0]) / 2.0)
            min_y = float(min(left_eye[1], right_eye[1], nose[1], left_mouth[1], right_mouth[1]))
            max_y = float(max(left_eye[1], right_eye[1], nose[1], left_mouth[1], right_mouth[1]))
            axis_top = np.asarray([face_center_x, min_y], dtype=np.float64)
            axis_bottom = np.asarray([face_center_x, max_y], dtype=np.float64)
        brow_left = _context_point(dense_context, "left_brow")
        brow_right = _context_point(dense_context, "right_brow")
        brow_center = (
            (brow_left + brow_right) / 2.0
            if brow_left is not None and brow_right is not None
            else None
        )
        nose_base_left = _context_point(dense_context, "nose_base_left")
        nose_base_right = _context_point(dense_context, "nose_base_right")
        nose_base_center = _context_point(dense_context, "nose_base_center")
        nose_mid_left = _context_point(dense_context, "nose_mid_left")
        nose_mid_right = _context_point(dense_context, "nose_mid_right")
        left_eye_inner = _context_point(dense_context, "left_eye_inner")
        left_eye_outer = _context_point(dense_context, "left_eye_outer")
        right_eye_inner = _context_point(dense_context, "right_eye_inner")
        right_eye_outer = _context_point(dense_context, "right_eye_outer")
        left_eye_top = _context_point(dense_context, "left_eye_top")
        left_eye_bottom = _context_point(dense_context, "left_eye_bottom")
        right_eye_top = _context_point(dense_context, "right_eye_top")
        right_eye_bottom = _context_point(dense_context, "right_eye_bottom")
        philtrum_top = _context_point(dense_context, "philtrum_top")
        upper_lip_top = _context_point(dense_context, "upper_lip_top")
        lower_lip_bottom = _context_point(dense_context, "lower_lip_bottom")
        chin_center = _context_point(dense_context, "chin_center")
        jaw_left = _context_point(dense_context, "jaw_left")
        jaw_right = _context_point(dense_context, "jaw_right")
        cheekbone_left = _context_point(dense_context, "cheekbone_left")
        cheekbone_right = _context_point(dense_context, "cheekbone_right")

        anchors = {
            "left_eye": _point_payload(left_eye),
            "right_eye": _point_payload(right_eye),
            "nose": _point_payload(nose),
            "left_mouth": _point_payload(left_mouth),
            "right_mouth": _point_payload(right_mouth),
        }
        derived_points = {
            "eye_center": _point_payload(eye_center),
            "mouth_center": _point_payload(mouth_center),
            "axis_top": _point_payload(axis_top),
            "axis_bottom": _point_payload(axis_bottom),
        }
        if brow_center is not None:
            derived_points["brow_center"] = _point_payload(brow_center)
        if nose_base_center is not None:
            derived_points["nose_base_center"] = _point_payload(nose_base_center)
        if chin_center is not None:
            derived_points["chin_center"] = _point_payload(chin_center)

        named_points = {
            "left_eye_center": anchors["left_eye"],
            "right_eye_center": anchors["right_eye"],
            "nose_tip": anchors["nose"],
            "mouth_left": anchors["left_mouth"],
            "mouth_right": anchors["right_mouth"],
            "mouth_center": derived_points["mouth_center"],
            "eye_center": derived_points["eye_center"],
            "facial_midline_top": derived_points["axis_top"],
            "facial_midline_bottom": derived_points["axis_bottom"],
        }
        optional_named_points = {
            "left_eye_inner": left_eye_inner,
            "left_eye_outer": left_eye_outer,
            "right_eye_inner": right_eye_inner,
            "right_eye_outer": right_eye_outer,
            "left_eye_top": left_eye_top,
            "left_eye_bottom": left_eye_bottom,
            "right_eye_top": right_eye_top,
            "right_eye_bottom": right_eye_bottom,
            "brow_left": brow_left,
            "brow_right": brow_right,
            "brow_center": brow_center,
            "nose_mid_left": nose_mid_left,
            "nose_mid_right": nose_mid_right,
            "nose_base_left": nose_base_left,
            "nose_base_right": nose_base_right,
            "nose_base_center": nose_base_center,
            "philtrum_top": philtrum_top,
            "upper_lip_top": upper_lip_top,
            "lower_lip_bottom": lower_lip_bottom,
            "chin_center": chin_center,
            "jaw_left": jaw_left,
            "jaw_right": jaw_right,
            "cheekbone_left": cheekbone_left,
            "cheekbone_right": cheekbone_right,
        }
        for name, point in optional_named_points.items():
            if point is not None:
                named_points[name] = _point_payload(point)
                if semantic_points is not None and name not in semantic_points:
                    semantic_points[name] = named_points[name]

        groups = {
            "eyes": ["left_eye_center", "right_eye_center", "eye_center"],
            "nose": ["nose_tip", "nose_base_center", "nose_base_left", "nose_base_right", "nose_mid_left", "nose_mid_right"],
            "mouth": ["mouth_left", "mouth_right", "mouth_center"],
            "brows": ["brow_left", "brow_right", "brow_center"],
            "chin_jaw": ["chin_center", "jaw_left", "jaw_right", "cheekbone_left", "cheekbone_right"],
            "midline": ["facial_midline_top", "facial_midline_bottom"],
        }
        if semantic_face_extents is not None:
            groups["face_extents"] = ["facial_midline_top", "facial_midline_bottom"]

        lines = [
            _line_entry(key="interpupillary_line", start="left_eye_center", end="right_eye_center", group="eyes", kind="measurement", layers=["measurements", "eyes"], metric_key="raw_eye_distance_px", supported_features=["raw_eye_distance_px", "ratio_eye_distance_to_face_width", "ratio_eye_distance_to_mouth_width"]),
            _line_entry(key="intercanthal_line", start="left_eye_inner", end="right_eye_inner", group="eyes", kind="measurement", layers=["measurements", "eyes"], metric_key="intercanthal_distance", supported_features=["intercanthal_distance", "ratio_intercanthal_to_eye_width", "ratio_eye_spacing_to_eye_width", "ratio_eye_spacing_to_face_width"]),
            _line_entry(key="outercanthal_line", start="left_eye_outer", end="right_eye_outer", group="eyes", kind="measurement", layers=["measurements", "eyes"], metric_key="outercanthal_distance", supported_features=["outercanthal_distance", "ratio_outercanthal_to_face_width"]),
            _line_entry(key="left_eye_width_line", start="left_eye_outer", end="left_eye_inner", group="eyes", kind="measurement", layers=["measurements", "eyes"], metric_key="left_eye_width", supported_features=["left_eye_width", "mean_eye_width"]),
            _line_entry(key="right_eye_width_line", start="right_eye_inner", end="right_eye_outer", group="eyes", kind="measurement", layers=["measurements", "eyes"], metric_key="right_eye_width", supported_features=["right_eye_width", "mean_eye_width"]),
            _line_entry(key="nose_width_line", start="nose_mid_left", end="nose_mid_right", group="nose", kind="measurement", layers=["measurements", "nose"], metric_key="nose_width", supported_features=["nose_width", "ratio_nose_width_to_face_width", "ratio_nose_width_to_mouth_width"]),
            _line_entry(key="nose_base_width_line", start="nose_base_left", end="nose_base_right", group="nose", kind="measurement", layers=["measurements", "nose"], metric_key="nose_base_width", supported_features=["nose_base_width"]),
            _line_entry(key="mouth_width_line", start="mouth_left", end="mouth_right", group="mouth", kind="measurement", layers=["measurements", "mouth"], metric_key="raw_mouth_width_px", supported_features=["raw_mouth_width_px", "ratio_mouth_width_to_face_width", "ratio_mouth_width_to_nose_width"]),
            _line_entry(key="jaw_width_proxy_line", start="jaw_left", end="jaw_right", group="chin_jaw", kind="measurement", layers=["measurements", "chin_jaw"], metric_key="jaw_width_proxy", supported_features=["jaw_width_proxy", "ratio_chin_width_to_jaw_width", "ratio_cheekbone_width_to_jaw_width"]),
            _line_entry(key="cheekbone_width_proxy_line", start="cheekbone_left", end="cheekbone_right", group="chin_jaw", kind="measurement", layers=["measurements", "chin_jaw"], metric_key="cheekbone_width_proxy", supported_features=["cheekbone_width_proxy", "ratio_cheekbone_width_to_jaw_width"]),
            _line_entry(key="eye_center_to_nose", start="eye_center", end="nose_tip", group="nose", kind="measurement", layers=["measurements", "nose"], metric_key="raw_eye_to_nose_px", supported_features=["raw_eye_to_nose_px"]),
            _line_entry(key="nose_to_mouth_line", start="nose_tip", end="mouth_center", group="mouth", kind="measurement", layers=["measurements", "mouth"], metric_key="raw_nose_to_mouth_px", supported_features=["raw_nose_to_mouth_px"]),
            _line_entry(key="mouth_center_to_chin_line", start="mouth_center", end="chin_center", group="chin_jaw", kind="measurement", layers=["measurements", "chin_jaw"], metric_key="mouth_center_to_chin_height", supported_features=["mouth_center_to_chin_height", "ratio_mouth_to_chin"]),
            _line_entry(key="brow_to_eye_center_line", start="brow_center", end="eye_center", group="brows", kind="measurement", layers=["measurements", "thirds"], metric_key="brow_to_eye_center_height", supported_features=["brow_to_eye_center_height"]),
            _line_entry(key="eye_center_to_nose_base_line", start="eye_center", end="nose_base_center", group="nose", kind="measurement", layers=["measurements", "thirds", "nose"], metric_key="eye_center_to_nose_base_height", supported_features=["eye_center_to_nose_base_height", "middle_third_proxy"]),
            _line_entry(key="nose_base_to_mouth_center_line", start="nose_base_center", end="mouth_center", group="mouth", kind="measurement", layers=["measurements", "thirds", "mouth"], metric_key="nose_base_to_mouth_center_height", supported_features=["nose_base_to_mouth_center_height", "ratio_nose_base_to_mouth_to_lower_third"]),
            _line_entry(key="brow_to_nose_base_line", start="brow_center", end="nose_base_center", group="brows", kind="measurement", layers=["measurements", "thirds"], metric_key="brow_to_nose_base_height", supported_features=["brow_to_nose_base_height", "upper_third_proxy"]),
            _line_entry(key="nose_base_to_chin_line", start="nose_base_center", end="chin_center", group="chin_jaw", kind="measurement", layers=["measurements", "thirds", "chin_jaw"], metric_key="nose_base_to_chin_height", supported_features=["nose_base_to_chin_height", "lower_third_proxy"]),
            _line_entry(key="eye_center_to_chin_line", start="eye_center", end="chin_center", group="chin_jaw", kind="measurement", layers=["measurements", "chin_jaw"], metric_key="eye_center_to_chin_height", supported_features=["eye_center_to_chin_height"]),
            _line_entry(key="eye_line_angle", start="left_eye_center", end="right_eye_center", group="eyes", kind="angle", layers=["angles", "eyes"], metric_key="canthal_tilt_deg", supported_features=["canthal_tilt_deg", "angle_eye_line_deg", "eye_line_vs_mouth_line_diff"]),
            _line_entry(key="mouth_line_angle", start="mouth_left", end="mouth_right", group="mouth", kind="angle", layers=["angles", "mouth"], metric_key="angle_mouth_line_deg", supported_features=["angle_mouth_line_deg", "eye_line_vs_mouth_line_diff"]),
            _line_entry(key="eye_center_to_mouth_line", start="eye_center", end="mouth_center", group="mouth", kind="angle", layers=["angles", "measurements"], metric_key="angle_eye_center_to_mouth_center_deg", supported_features=["angle_eye_center_to_mouth_center_deg"]),
            _line_entry(key="nose_to_mouth_direction", start="nose_tip", end="mouth_center", group="mouth", kind="angle", layers=["angles", "measurements"], metric_key="angle_nose_to_mouth_center_deg", supported_features=["angle_nose_to_mouth_center_deg"]),
            _line_entry(key="left_eye_to_nose", start="left_eye_center", end="nose_tip", group="eyes", kind="symmetry", layers=["symmetry", "eyes"], metric_key="raw_left_eye_to_nose_px", supported_features=["raw_left_eye_to_nose_px", "left_right_eye_to_nose_diff", "sym_eye_nose_delta_ratio"]),
            _line_entry(key="right_eye_to_nose", start="right_eye_center", end="nose_tip", group="eyes", kind="symmetry", layers=["symmetry", "eyes"], metric_key="raw_right_eye_to_nose_px", supported_features=["raw_right_eye_to_nose_px", "left_right_eye_to_nose_diff", "sym_eye_nose_delta_ratio"]),
            _line_entry(key="left_mouth_to_nose", start="mouth_left", end="nose_tip", group="mouth", kind="symmetry", layers=["symmetry", "mouth"], metric_key="raw_left_mouth_to_nose_px", supported_features=["raw_left_mouth_to_nose_px", "left_right_mouth_to_nose_diff", "sym_mouth_nose_delta_ratio"]),
            _line_entry(key="right_mouth_to_nose", start="mouth_right", end="nose_tip", group="mouth", kind="symmetry", layers=["symmetry", "mouth"], metric_key="raw_right_mouth_to_nose_px", supported_features=["raw_right_mouth_to_nose_px", "left_right_mouth_to_nose_diff", "sym_mouth_nose_delta_ratio"]),
        ]
        axes = [
            _line_entry(key="facial_midline", start="facial_midline_top", end="facial_midline_bottom", group="symmetry", kind="axis", layers=["symmetry", "thirds"], supported_features=["midline_alignment_index", "nose_tip_midline_offset", "mouth_center_midline_offset", "chin_center_midline_offset"]),
            _line_entry(key="nose_axis", start="nose_base_center", end="nose_tip", group="nose", kind="axis", layers=["angles", "nose"], metric_key="nose_axis_angle_deg", supported_features=["nose_axis_angle_deg", "face_centerline_vs_nose_angle"]),
            _line_entry(key="philtrum_axis", start="nose_base_center", end="mouth_center", group="mouth", kind="axis", layers=["angles", "mouth"], metric_key="philtrum_axis_angle_deg", supported_features=["philtrum_axis_angle_deg"]),
            _line_entry(key="chin_axis", start="mouth_center", end="chin_center", group="chin_jaw", kind="axis", layers=["angles", "chin_jaw"], metric_key="chin_axis_angle_deg", supported_features=["chin_axis_angle_deg", "nose_to_chin_axis_diff", "lower_face_tilt_proxy"]),
            _line_entry(key="brow_line", start="brow_left", end="brow_right", group="brows", kind="axis", layers=["angles", "thirds"], metric_key="brow_line_angle_deg", supported_features=["brow_line_angle_deg"]),
            _line_entry(key="jaw_left_axis", start="jaw_left", end="chin_center", group="chin_jaw", kind="axis", layers=["angles", "chin_jaw"], metric_key="jaw_line_left_angle_proxy", supported_features=["jaw_line_left_angle_proxy", "jaw_angle_asymmetry"]),
            _line_entry(key="jaw_right_axis", start="chin_center", end="jaw_right", group="chin_jaw", kind="axis", layers=["angles", "chin_jaw"], metric_key="jaw_line_right_angle_proxy", supported_features=["jaw_line_right_angle_proxy", "jaw_angle_asymmetry"]),
        ]
        guides = [
            _line_entry(key="thirds_brow_guide", start="brow_left", end="brow_right", group="thirds", kind="guide", layers=["thirds"], metric_key="upper_third_proxy", supported_features=["upper_third_proxy", "ratio_upper_third_to_middle_third", "thirds_balance_index"], label="brow"),
            _line_entry(key="thirds_eye_guide", start="left_eye_center", end="right_eye_center", group="thirds", kind="guide", metric_key="middle_third_proxy", layers=["thirds"], supported_features=["middle_third_proxy", "ratio_upper_third_to_middle_third", "ratio_middle_third_to_lower_third"], label="eye"),
            _line_entry(key="thirds_nose_base_guide", start="nose_base_left", end="nose_base_right", group="thirds", kind="guide", metric_key="lower_third_proxy", layers=["thirds", "nose"], supported_features=["lower_third_proxy", "ratio_middle_third_to_lower_third", "ratio_nose_base_to_mouth_to_lower_third"], label="nose base"),
            _line_entry(key="thirds_mouth_guide", start="mouth_left", end="mouth_right", group="thirds", kind="guide", metric_key="ratio_mouth_to_chin_to_lower_third", layers=["thirds", "mouth"], supported_features=["ratio_mouth_to_chin_to_lower_third"], label="mouth"),
            _line_entry(key="thirds_chin_guide", start="jaw_left", end="jaw_right", group="thirds", kind="guide", metric_key="chin_height", layers=["thirds", "chin_jaw"], supported_features=["chin_height"], label="chin"),
            _line_entry(key="nose_midline_offset", start="nose_tip", end="nose_base_center" if nose_base_center is not None else "nose_tip", group="nose", kind="guide", layers=["symmetry", "nose"], metric_key="nose_tip_midline_offset", supported_features=["nose_tip_midline_offset"]),
        ]
        if axis_top is not None:
            if nose_base_center is not None:
                named_points["nose_midline_projection"] = _point_payload(np.asarray([face_center_x, nose[1]], dtype=np.float64))
            named_points["mouth_midline_projection"] = _point_payload(np.asarray([face_center_x, mouth_center[1]], dtype=np.float64))
            if chin_center is not None:
                named_points["chin_midline_projection"] = _point_payload(np.asarray([face_center_x, chin_center[1]], dtype=np.float64))
            if philtrum_top is not None:
                named_points["philtrum_midline_projection"] = _point_payload(np.asarray([face_center_x, philtrum_top[1]], dtype=np.float64))
        for projection_key, target_key, metric_key in (
            ("nose_midline_projection", "nose_tip", "nose_tip_midline_offset"),
            ("mouth_midline_projection", "mouth_center", "mouth_center_midline_offset"),
            ("chin_midline_projection", "chin_center", "chin_center_midline_offset"),
            ("philtrum_midline_projection", "philtrum_top", "philtrum_midline_offset_proxy"),
        ):
            if projection_key in named_points and target_key in named_points:
                lines.append(
                    _line_entry(
                        key=f"{projection_key}_line",
                        start=projection_key,
                        end=target_key,
                        group="symmetry",
                        kind="symmetry",
                        layers=["symmetry"],
                        metric_key=metric_key,
                        supported_features=[metric_key],
                        )
                    )

        if semantic_points is not None and named_points is not None:
            for name, payload in named_points.items():
                if name not in semantic_points:
                    semantic_points[name] = payload

        if semantic_validation is not None and not bool(semantic_validation.get("passed", False)):
            for warning in semantic_validation.get("failed_checks", []):
                notes.append(f"semantic_validation:{warning}")
        nose_source_rules = {
            key: value.get("source_rule")
            for key, value in (semantic_points or {}).items()
            if "nose" in key or "alare" in key or "subnasale" in key or key == "philtrum_top"
            if isinstance(value, dict) and value.get("source_rule")
        }
        if nose_source_rules:
            notes.append(f"nose_semantic_rules:{json.dumps(nose_source_rules, ensure_ascii=False)}")

    if anchors and derived_points:
        status = "available"
    elif bbox is not None or image_size is not None:
        status = "partial"
    else:
        status = "unavailable"

    image_payload = None
    if image_size is not None:
        image_payload = {
            "width": int(image_size[0]),
            "height": int(image_size[1]),
        }

    return {
        "status": status,
        "image_role": coordinate_space,
        "coordinate_space": coordinate_space,
        "image_path": str(image_path.resolve()) if image_path is not None else None,
        "image_size": image_payload,
        "bbox": display_face_box,
        "detector_bbox": detector_bbox,
        "semantic_face_extents": semantic_face_extents,
        "display_face_box": display_face_box,
        "image_width": int(image_size[0]) if image_size is not None else None,
        "image_height": int(image_size[1]) if image_size is not None else None,
        "raw_landmarks": landmark_points,
        "raw_landmarks_debug": raw_landmarks_debug,
        "mediapipe_landmarks": (
            [
                {
                    "index": int(index),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "group": "mediapipe_dense",
                    "name": None,
                    "source_type": "mediapipe_face_mesh",
                    "source_indices": [int(index)],
                    "description": "MediaPipe dense landmark",
                }
                for index, point in enumerate(mediapipe_landmarks or [])
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
        ),
        "mediapipe_regions": mediapipe_regions or {},
        "semantic_points": semantic_points,
        "semantic_points_debug": semantic_points_debug,
        "measurement_overlays": {
            "lines": lines,
            "axes": axes,
            "guides": guides,
        },
        "measurement_overlays_debug": {
            "lines": lines,
            "axes": axes,
            "guides": guides,
        },
        "semantic_validation": semantic_validation,
        "landmarks": landmark_points,
        "anchors": anchors,
        "named_points": named_points,
        "derived_points": derived_points,
        "lines": lines,
        "axes": axes,
        "guides": guides,
        "groups": groups,
        "notes": notes,
    }


def build_report_payload(
    *,
    image_path: Path,
    analysis_image_path: Path,
    normalized_frame_image_path: Path | None,
    measurement_image_path: Path,
    measurement_image_role: str,
    output_dir: Path,
    group_override: str | None,
    provider_diagnostics: dict[str, Any],
    processing_row: dict[str, Any],
    image_size: tuple[int, int] | None,
    bbox: list[float] | None,
    landmarks: list[list[float]] | None,
    landmarks_dense: list[list[float]] | None,
    mediapipe_landmarks: list[list[float]] | None = None,
    mediapipe_regions: dict[str, list[int]] | None = None,
    raw_measurements: dict[str, float | None],
    normalized_measurements: dict[str, float | None],
    symmetry_features: dict[str, float | None],
    angle_features: dict[str, float | None],
    selected_group_key: str,
    selected_group: dict[str, Any],
    reference_comparison: dict[str, Any],
    z_scores: dict[str, float | None],
    mahalanobis_distance: float | None,
    percentile_positions: dict[str, float | None],
    feature_deviations: list[dict[str, Any]],
    top_k_features: int,
    aligned_image_path: Path | None,
    annotated_image_path: Path | None,
    reference_groupings: list[list[str]],
    resolved_group_values: dict[str, str],
) -> dict[str, Any]:
    composite_indices = compute_optional_composite_indices(
        normalized_measurements=normalized_measurements,
        symmetry_features=symmetry_features,
        z_scores=z_scores,
        mahalanobis_distance=mahalanobis_distance,
    )
    limitations = build_reporting_limitations(
        quality_pass=bool(processing_row.get("quality_pass")),
        quality_fail_reason=processing_row.get("quality_fail_reason"),
        provider_diagnostics=provider_diagnostics,
        reference_group_sample_count=selected_group.get("sample_count"),
        reference_warning=reference_comparison.get("warning"),
    )
    expanded_geometry = build_expanded_geometry_payload(
        raw_measurements=raw_measurements,
        normalized_measurements=normalized_measurements,
        symmetry_features=symmetry_features,
        angle_features=angle_features,
    )

    return {
        "input": {
            "image_path": str(image_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "group_override": group_override,
            "resolved_group_values": resolved_group_values,
            "analysis_image_role": measurement_image_role,
            "overlay_image_role": measurement_image_role,
            "provider_diagnostics": provider_diagnostics,
        },
        "input_quality": {
            "processing_status": processing_row.get("processing_status"),
            "quality_pass": bool(processing_row.get("quality_pass")),
            "quality_fail_reason": processing_row.get("quality_fail_reason"),
            "processing_error_message": processing_row.get("processing_error_message"),
            "alignment": {
                "attempted": bool(processing_row.get("alignment_attempted")),
                "applied": bool(processing_row.get("alignment_applied")),
                "improved_quality": bool(processing_row.get("alignment_improved_quality")),
                "source": processing_row.get("alignment_source"),
                "status": processing_row.get("alignment_status"),
                "rotation_deg": processing_row.get("alignment_rotation_deg"),
                "transform": json.loads(processing_row["alignment_transform"])
                if processing_row.get("alignment_transform")
                else None,
                "original_pose": {
                    "yaw": processing_row.get("original_yaw"),
                    "pitch": processing_row.get("original_pitch"),
                    "roll": processing_row.get("original_roll"),
                },
                "aligned_pose": {
                    "yaw": processing_row.get("yaw"),
                    "pitch": processing_row.get("pitch"),
                    "roll": processing_row.get("roll"),
                },
                "original_quality_pass": processing_row.get("original_quality_pass"),
                "original_quality_fail_reason": processing_row.get("original_quality_fail_reason"),
                "aligned_quality_pass": processing_row.get("aligned_quality_pass"),
                "aligned_quality_fail_reason": processing_row.get("aligned_quality_fail_reason"),
            },
            "detection": {
                "bbox": bbox,
                "det_score": processing_row.get("det_score"),
                "pose": {
                    "yaw": processing_row.get("yaw"),
                    "pitch": processing_row.get("pitch"),
                    "roll": processing_row.get("roll"),
                },
                "num_faces": processing_row.get("num_faces"),
                "detection_backend": processing_row.get("detection_backend"),
                "blur_variance": processing_row.get("blur_variance"),
                "occlusion_score": processing_row.get("occlusion_score"),
            },
        },
        "visualization": build_visualization_payload(
            image_size=image_size,
            bbox=bbox,
            landmarks=landmarks,
            landmarks_dense=landmarks_dense,
            mediapipe_landmarks=mediapipe_landmarks,
            mediapipe_regions=mediapipe_regions,
            coordinate_space=measurement_image_role,
            image_path=measurement_image_path,
        ),
        "raw_measurements": raw_measurements,
        "normalized_ratios": normalized_measurements,
        "symmetry_metrics": symmetry_features,
        "angle_metrics": angle_features,
        "expanded_geometry": expanded_geometry,
        "reference_comparison": {
            "selected_group": {
                "group_key": selected_group_key,
                "group_values": selected_group.get("group_values"),
                "sample_count": selected_group.get("sample_count"),
                "complete_case_count": selected_group.get("complete_case_count"),
            },
            **reference_comparison,
            "z_scores": z_scores,
            "mahalanobis_distance": (float(mahalanobis_distance) if mahalanobis_distance is not None else None),
            "percentile_positions": percentile_positions,
        },
        "feature_level_deviations": {
            "top_contributing_features": feature_deviations[:top_k_features],
            "all_feature_deviations": feature_deviations,
        },
        "optional_composite_indices": composite_indices,
        "transparency": {
            "formulas": build_formula_transparency(),
            "reference_group_selection": build_reference_selection_transparency(
                group_override=group_override,
                resolved_group_values=resolved_group_values,
                selected_group_key=selected_group_key,
                selected_group=selected_group,
                reference_groupings=reference_groupings,
            ),
        },
        "limitations": limitations,
        "artifacts": {
            "original_image_path": str(image_path.resolve()),
            "analysis_image_path": str(analysis_image_path.resolve()) if analysis_image_path else None,
            "normalized_frame_image_path": (
                str(normalized_frame_image_path.resolve()) if normalized_frame_image_path else None
            ),
            "aligned_image_path": str(aligned_image_path.resolve()) if aligned_image_path else None,
            "measurement_image_path": str(measurement_image_path.resolve()) if measurement_image_path else None,
            "overlay_image_path": str(measurement_image_path.resolve()) if measurement_image_path else None,
            "annotated_image_path": str(annotated_image_path.resolve()) if annotated_image_path else None,
        },
    }


def save_annotated_image(
    *,
    image_path: Path,
    visualization_payload: dict[str, Any] | None,
    output_path: Path,
) -> bool:
    if not visualization_payload:
        return False

    bbox = visualization_payload.get("bbox")
    anchors = visualization_payload.get("anchors") or {}
    derived_points = visualization_payload.get("derived_points") or {}

    try:
        image = cv2.imread(str(image_path))
    except Exception:
        image = None
    if image is None:
        return False

    def _point(name: str) -> tuple[int, int] | None:
        payload = anchors.get(name) or derived_points.get(name)
        if not isinstance(payload, dict):
            return None
        x = payload.get("x")
        y = payload.get("y")
        if x is None or y is None:
            return None
        return int(round(float(x))), int(round(float(y)))

    def _draw_line(start_name: str, end_name: str, color: tuple[int, int, int], thickness: int = 2) -> None:
        start = _point(start_name)
        end = _point(end_name)
        if start is None or end is None:
            return
        cv2.line(image, start, end, color, thickness, lineType=cv2.LINE_AA)

    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (240, 164, 70), 2, lineType=cv2.LINE_AA)

    for anchor_name in ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth"):
        point = _point(anchor_name)
        if point is None:
            continue
        cv2.circle(image, point, 4, (255, 245, 235), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(image, point, 6, (18, 96, 214), thickness=2, lineType=cv2.LINE_AA)

    _draw_line("left_eye", "right_eye", (28, 180, 112))
    _draw_line("left_mouth", "right_mouth", (28, 180, 112))
    _draw_line("eye_center", "nose", (74, 222, 128))
    _draw_line("nose", "mouth_center", (74, 222, 128))
    _draw_line("eye_center", "mouth_center", (242, 201, 76))
    _draw_line("axis_top", "axis_bottom", (240, 164, 70))
    _draw_line("left_eye", "nose", (111, 141, 255), 1)
    _draw_line("right_eye", "nose", (111, 141, 255), 1)
    _draw_line("left_mouth", "nose", (111, 141, 255), 1)
    _draw_line("right_mouth", "nose", (111, 141, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), image))


def render_markdown_report(report: dict[str, Any]) -> str:
    quality = report["input_quality"]
    reference = report["reference_comparison"]
    deviations = report["feature_level_deviations"]["top_contributing_features"]
    transparency = report["transparency"]
    optional_indices = report["optional_composite_indices"]

    def _emit_mapping(title: str, payload: dict[str, Any]) -> list[str]:
        lines = ["", f"## {title}"]
        for key, value in payload.items():
            lines.append(f"- {key}: `{value}`")
        return lines

    lines = [
        "# Morphometric Report",
        "",
        "## Input & Quality",
        f"- image_path: `{report['input']['image_path']}`",
        f"- analysis_image_path: `{report['artifacts']['analysis_image_path']}`",
        f"- normalized_frame_image_path: `{report['artifacts'].get('normalized_frame_image_path')}`",
        f"- measurement_image_path: `{report['artifacts']['measurement_image_path']}`",
        f"- analysis_image_role: `{report['input']['analysis_image_role']}`",
        f"- processing_status: `{quality['processing_status']}`",
        f"- quality_pass: `{quality['quality_pass']}`",
        f"- quality_fail_reason: `{quality['quality_fail_reason']}`",
        f"- bbox: `{quality['detection']['bbox']}`",
        f"- det_score: `{quality['detection']['det_score']}`",
        f"- pose: `yaw={quality['detection']['pose']['yaw']}, pitch={quality['detection']['pose']['pitch']}, roll={quality['detection']['pose']['roll']}`",
        f"- alignment_applied: `{quality['alignment']['applied']}`",
        f"- alignment_rotation_deg: `{quality['alignment']['rotation_deg']}`",
        f"- original_pose: `yaw={quality['alignment']['original_pose']['yaw']}, pitch={quality['alignment']['original_pose']['pitch']}, roll={quality['alignment']['original_pose']['roll']}`",
    ]

    lines.extend(_emit_mapping("Raw Measurements", report["raw_measurements"]))
    lines.extend(_emit_mapping("Normalized Ratios", report["normalized_ratios"]))
    lines.extend(_emit_mapping("Symmetry Metrics", report["symmetry_metrics"]))
    lines.extend(_emit_mapping("Angle Metrics", report["angle_metrics"]))
    expanded_geometry = report.get("expanded_geometry") or {}
    lines.extend(
        [
            "",
            "## Expanded Geometry",
            f"- available: `{expanded_geometry.get('available')}`",
            f"- mediapipe_feature_count: `{expanded_geometry.get('mediapipe_feature_count')}`",
            f"- mediapipe_non_null_feature_count: `{expanded_geometry.get('mediapipe_non_null_feature_count')}`",
        ]
    )
    reliability_counts = expanded_geometry.get("reliability_counts")
    if isinstance(reliability_counts, dict):
        for reliability_name, reliability_value in reliability_counts.items():
            lines.append(f"- reliability[{reliability_name}]: `{reliability_value}`")
    expanded_groups = expanded_geometry.get("groups")
    if isinstance(expanded_groups, dict):
        for group_name, items in expanded_groups.items():
            lines.append(f"- {group_name}:")
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "  - "
                    f"{item.get('feature_name')}: value=`{item.get('value')}`, "
                    f"source=`{item.get('source')}`, reliability=`{item.get('reliability')}`"
                )

    lines.extend(
        [
            "",
            "## Reference Comparison",
            f"- selected_group_key: `{reference['selected_group']['group_key']}`",
            f"- sample_count: `{reference['selected_group']['sample_count']}`",
            f"- complete_case_count: `{reference['selected_group']['complete_case_count']}`",
            f"- status: `{reference.get('status')}`",
            f"- warning: `{reference.get('warning')}`",
            f"- mahalanobis_distance: `{reference['mahalanobis_distance']}`",
        ]
    )
    for feature_name, value in reference["z_scores"].items():
        lines.append(f"- z_score[{feature_name}]: `{value}`")

    lines.extend(["", "## Feature-Level Deviations"])
    for item in deviations:
        lines.append(
            f"- {item['feature_name']}: z={item['z_score']}, direction={item['direction']}, abs_z={item['abs_z_score']}"
        )

    lines.extend(_emit_mapping("Optional Composite Indices", optional_indices))

    lines.extend(
        [
            "",
            "## Transparency",
            f"- reference_selection_strategy: `{transparency['reference_group_selection']['selection_strategy']}`",
            f"- resolved_group_values: `{transparency['reference_group_selection']['resolved_group_values']}`",
            "",
            "### Formula Notes",
        ]
    )
    for section_name, formulas in transparency["formulas"].items():
        lines.append(f"- {section_name}:")
        for formula_name, formula in formulas.items():
            lines.append(f"  - {formula_name}: `{formula}`")

    lines.extend(["", "## Limitations"])
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")

    return "\n".join(lines) + "\n"


def save_report_bundle(
    *,
    report: dict[str, Any],
    report_json_path: Path,
    report_markdown_path: Path,
    reporting_summary_path: Path,
) -> dict[str, Any]:
    ensure_parent_directory(report_json_path)
    ensure_parent_directory(report_markdown_path)
    ensure_parent_directory(reporting_summary_path)

    report["artifacts"].update(
        {
            "report_json_path": str(report_json_path.resolve()),
            "report_markdown_path": str(report_markdown_path.resolve()),
            "reporting_summary_path": str(reporting_summary_path.resolve()),
        }
    )
    write_json(report, report_json_path)
    report_markdown_path.write_text(render_markdown_report(report), encoding="utf-8")

    summary = {
        "status": "completed",
        "processing_status": report["input_quality"]["processing_status"],
        "quality_pass": report["input_quality"]["quality_pass"],
        "alignment_applied": report["input_quality"]["alignment"]["applied"],
        "alignment_rotation_deg": report["input_quality"]["alignment"]["rotation_deg"],
        "selected_group_key": report["reference_comparison"]["selected_group"]["group_key"],
        "selected_group_sample_count": report["reference_comparison"]["selected_group"]["sample_count"],
        "reference_comparison_status": report["reference_comparison"].get("status"),
        "reference_comparison_warning": report["reference_comparison"].get("warning"),
        "mahalanobis_distance": report["reference_comparison"]["mahalanobis_distance"],
        "top_feature_deviation_count": len(report["feature_level_deviations"]["top_contributing_features"]),
        "optional_composite_indices": report["optional_composite_indices"],
        "artifacts": {
            "report_json_path": str(report_json_path.resolve()),
            "report_markdown_path": str(report_markdown_path.resolve()),
            "analysis_image_path": report["artifacts"].get("analysis_image_path"),
            "normalized_frame_image_path": report["artifacts"].get("normalized_frame_image_path"),
            "aligned_image_path": report["artifacts"].get("aligned_image_path"),
            "measurement_image_path": report["artifacts"].get("measurement_image_path"),
            "overlay_image_path": report["artifacts"].get("overlay_image_path"),
            "annotated_image_path": report["artifacts"].get("annotated_image_path"),
        },
    }
    write_json(summary, reporting_summary_path)
    return summary
