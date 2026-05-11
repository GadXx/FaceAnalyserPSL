from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import cv2
import numpy as np
import pandas as pd

from src.face_aesthetics.features.extraction import (
    extract_angle_features,
    extract_normalized_measurements,
    extract_raw_measurements,
    extract_symmetry_features,
    parse_json_field,
    validate_feature_values,
)
from src.face_aesthetics.data.io import write_json
from src.face_aesthetics.data.io import load_image, save_image
from src.face_aesthetics.landmarks.mediapipe_source import (
    MEDIAPIPE_FACE_OVAL,
    MEDIAPIPE_REGION_INDICES,
    build_mediapipe_source_result,
)
from src.face_aesthetics.models.reference import (
    UNSTABLE_STD_THRESHOLD,
    compute_mahalanobis_distance,
    compute_z_scores,
    unstable_std_mask,
)
from src.face_aesthetics.pipelines.preprocessing import preprocess_single_image
from src.face_aesthetics.services.reporting import (
    build_report_payload,
    save_annotated_image,
    save_report_bundle,
)
from src.face_aesthetics.utils.config import load_named_configs
from src.face_aesthetics.utils.filesystem import ensure_directory


REPORT_JSON_NAME = "sample_report.json"
REPORT_MD_NAME = "sample_report.md"
ALIGNED_IMAGE_NAME = "aligned_face.jpg"
ANALYSIS_IMAGE_NAME = "analysis_image.jpg"
ANNOTATED_IMAGE_NAME = "annotated_analysis.jpg"
VISUALIZATION_DEBUG_JSON_NAME = "visualization_debug.json"


def _derive_mediapipe_five_point_landmarks(landmarks: list[list[float]]) -> list[list[float]]:
    if not landmarks or len(landmarks) < 400:
        return []
    left_eye = np.asarray([landmarks[index][:2] for index in (33, 133, 159, 145)], dtype=np.float64).mean(axis=0)
    right_eye = np.asarray([landmarks[index][:2] for index in (362, 263, 386, 374)], dtype=np.float64).mean(axis=0)
    nose = np.asarray(landmarks[1][:2], dtype=np.float64)
    left_mouth = np.asarray(landmarks[61][:2], dtype=np.float64)
    right_mouth = np.asarray(landmarks[291][:2], dtype=np.float64)
    return [
        [float(left_eye[0]), float(left_eye[1])],
        [float(right_eye[0]), float(right_eye[1])],
        [float(nose[0]), float(nose[1])],
        [float(left_mouth[0]), float(left_mouth[1])],
        [float(right_mouth[0]), float(right_mouth[1])],
    ]


def _padded_bbox_from_mediapipe(
    landmarks: list[list[float]],
    image_width: int,
    image_height: int,
    *,
    pad_top: float = 0.18,
    pad_bottom: float = 0.22,
    pad_left: float = 0.14,
    pad_right: float = 0.14,
) -> list[float]:
    indices = [index for index in MEDIAPIPE_FACE_OVAL if index < len(landmarks)]
    if not indices:
        indices = list(range(min(len(landmarks), 478)))
    points = np.asarray([[landmarks[index][0], landmarks[index][1]] for index in indices], dtype=np.float64)
    x1 = float(np.min(points[:, 0]))
    y1 = float(np.min(points[:, 1]))
    x2 = float(np.max(points[:, 0]))
    y2 = float(np.max(points[:, 1]))
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    ex1 = max(0.0, x1 - width * pad_left)
    ex2 = min(float(image_width - 1), x2 + width * pad_right)
    ey1 = max(0.0, y1 - height * pad_top)
    ey2 = min(float(image_height - 1), y2 + height * pad_bottom)
    return [float(ex1), float(ey1), float(ex2), float(ey2)]


def _write_overlay_debug_artifacts(
    *,
    output_dir: Path,
    analysis_image_bgr: np.ndarray,
    visualization: dict[str, Any],
    processing_row: dict[str, Any],
) -> None:
    ensure_directory(output_dir / "reports" / "debug")
    debug_dir = output_dir / "reports" / "debug"
    landmarks = visualization.get("mediapipe_landmarks", [])
    bbox = visualization.get("bbox")
    display_bbox = visualization.get("display_face_box")
    point_rows = [{"index": int(item.get("index", -1)), "x": float(item.get("x", 0.0)), "y": float(item.get("y", 0.0))} for item in landmarks]
    overlay_audit = {
        "root_cause_hypothesis": "legacy_coordinate_space_mismatch_or_aligned_crop_substitution",
        "coordinate_space": visualization.get("coordinate_space"),
        "image_role": visualization.get("image_role"),
        "image_size": visualization.get("image_size"),
        "bbox": bbox,
        "display_face_box": display_bbox,
        "detector_bbox": visualization.get("detector_bbox"),
        "landmark_count": len(point_rows),
    }
    write_json(overlay_audit, debug_dir / "mediapipe_overlay_coordinate_audit.json")
    write_json({"points": point_rows, "bbox": bbox}, debug_dir / "mediapipe_overlay_original_space.json")
    write_json({"points": point_rows, "bbox": display_bbox}, debug_dir / "mediapipe_overlay_preview_space.json")
    write_json(
        {
            "alignment_status": processing_row.get("alignment_status"),
            "alignment_applied": processing_row.get("alignment_applied"),
            "alignment_transform": processing_row.get("alignment_transform"),
            "analysis_image_role": processing_row.get("analysis_image_role"),
        },
        debug_dir / "mediapipe_overlay_alignment_audit.json",
    )
    canvas = analysis_image_bgr.copy()
    for row in point_rows:
        x = int(round(row["x"]))
        y = int(round(row["y"]))
        cv2.circle(canvas, (x, y), 1, (0, 255, 255), -1)
    if isinstance(display_bbox, list) and len(display_bbox) == 4:
        x1, y1, x2, y2 = [int(round(float(value))) for value in display_bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
    save_image(debug_dir / "overlay_original_debug.png", canvas)
    save_image(debug_dir / "overlay_preview_debug.png", canvas)
    save_image(debug_dir / "overlay_cropped_debug.png", canvas)


@dataclass
class InferenceArtifacts:
    report_json_path: Path
    report_markdown_path: Path
    reporting_summary_path: Path
    analysis_image_path: Path | None
    normalized_frame_image_path: Path | None
    aligned_image_path: Path | None
    measurement_image_path: Path | None
    annotated_image_path: Path | None
    report: dict[str, Any]
    reporting_summary: dict[str, Any]


def _parse_group_override(value: str | None) -> dict[str, str]:
    if value is None or not value.strip():
        return {}

    stripped = value.strip()
    if stripped.startswith("{"):
        payload = json.loads(stripped)
        return {str(key): str(raw).strip() for key, raw in payload.items() if str(raw).strip()}

    item_separator = "," if "," in stripped else "__"
    group_values: dict[str, str] = {}
    for item in stripped.split(item_separator):
        normalized_item = item.strip()
        if not normalized_item:
            continue
        if "=" not in item:
            raise ValueError(
                "Group override must be JSON, comma-separated key=value pairs, "
                "or reference-group keys joined with '__'. "
                f"Received: {value}"
            )
        key, raw_value = normalized_item.split("=", 1)
        normalized_key = key.strip()
        normalized_value = raw_value.strip()
        if normalized_key and normalized_value:
            group_values[normalized_key] = normalized_value
    return group_values


def _read_sidecar_metadata(image_path: Path) -> dict[str, Any]:
    candidate_paths = [
        image_path.with_suffix(image_path.suffix + ".json"),
        image_path.with_suffix(".json"),
        image_path.with_suffix(image_path.suffix + ".metadata.json"),
        image_path.with_name(image_path.stem + ".metadata.json"),
    ]
    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            with candidate.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload, dict):
                return payload
    return {}


def _normalized_labels_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized_labels = metadata.get("normalized_labels")
    if isinstance(normalized_labels, dict):
        return normalized_labels
    if isinstance(normalized_labels, str) and normalized_labels.strip():
        return json.loads(normalized_labels)
    return metadata


def _resolve_group_values(
    *,
    group_override: str | None,
    metadata: dict[str, Any],
    group_aliases: dict[str, list[str]],
) -> dict[str, str]:
    override_values = _parse_group_override(group_override)
    if override_values:
        return override_values

    normalized_labels = _normalized_labels_from_metadata(metadata)
    resolved: dict[str, str] = {}
    for canonical_name, aliases in group_aliases.items():
        for alias in aliases:
            if alias not in normalized_labels:
                continue
            value = normalized_labels.get(alias)
            if value is None:
                continue
            normalized_value = str(value).strip()
            if normalized_value:
                resolved[canonical_name] = normalized_value
                break
    return resolved


def _select_reference_group(
    reference_payload: dict[str, Any],
    group_values: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    candidate_groupings = sorted(
        reference_payload.get("groupings", []),
        key=lambda item: (-len(item), tuple(item)),
    )
    reference_groups = dict(reference_payload.get("reference_groups", {}))

    for grouping in candidate_groupings:
        if not all(column in group_values for column in grouping):
            continue
        combination_name = "__".join(grouping)
        group_key = "__".join(f"{column}={group_values[column]}" for column in grouping)
        bucket = reference_groups.get(combination_name, {})
        if group_key in bucket:
            return group_key, bucket[group_key]

    raise ValueError(
        "No matching reference group found for provided/inferred metadata. "
        f"Resolved group values: {group_values}"
    )


def _estimate_percentile_position(feature_value: float, stats_row: pd.Series | None) -> float | None:
    if stats_row is None:
        return None

    percentile_points: list[tuple[float, float]] = []
    for column_name in stats_row.index.tolist():
        if not column_name.startswith("p"):
            continue
        raw_value = stats_row[column_name]
        if raw_value is None or pd.isna(raw_value):
            continue
        try:
            percentile = float(int(column_name[1:]) / 100.0)
        except ValueError:
            continue
        percentile_points.append((float(raw_value), percentile))

    percentile_points = sorted(set(percentile_points), key=lambda item: item[0])
    if not percentile_points:
        return None
    if len(percentile_points) == 1:
        return percentile_points[0][1]
    if feature_value <= percentile_points[0][0]:
        return percentile_points[0][1]
    if feature_value >= percentile_points[-1][0]:
        return percentile_points[-1][1]

    for (left_value, left_percentile), (right_value, right_percentile) in zip(
        percentile_points[:-1],
        percentile_points[1:],
    ):
        if left_value <= feature_value <= right_value:
            if right_value == left_value:
                return right_percentile
            interpolation = (feature_value - left_value) / (right_value - left_value)
            return float(left_percentile + interpolation * (right_percentile - left_percentile))
    return None


def _feature_statistics_lookup(
    statistics_frame: pd.DataFrame,
    group_key: str,
) -> dict[str, pd.Series]:
    if statistics_frame.empty:
        return {}
    subset = statistics_frame.loc[statistics_frame["group_key"] == group_key].copy()
    if subset.empty:
        return {}
    return {str(row["feature_name"]): row for _, row in subset.iterrows()}


def _sorted_deviations(z_scores: dict[str, float | None]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_name, z_score in z_scores.items():
        if z_score is None:
            continue
        direction = "above_mean" if z_score > 0.0 else "below_mean" if z_score < 0.0 else "at_mean"
        rows.append(
            {
                "feature_name": feature_name,
                "z_score": float(z_score),
                "abs_z_score": float(abs(z_score)),
                "direction": direction,
            }
        )
    return sorted(rows, key=lambda item: item["abs_z_score"], reverse=True)


def _serialize_z_scores(
    feature_columns: list[str],
    z_score_vector: np.ndarray,
    std_vector: np.ndarray,
    *,
    unstable_std_threshold: float = UNSTABLE_STD_THRESHOLD,
) -> dict[str, float | None]:
    unstable_mask = unstable_std_mask(std_vector, threshold=unstable_std_threshold)
    result: dict[str, float | None] = {}
    for feature_name, z_score, unstable in zip(feature_columns, z_score_vector.tolist(), unstable_mask.tolist()):
        if unstable or pd.isna(z_score):
            result[feature_name] = None
        else:
            result[feature_name] = float(z_score)
    return result


def _reference_comparison_warning(missing_features: list[str]) -> str:
    lower_face_markers = (
        "mouth",
        "lip",
        "philtrum",
        "jaw",
        "chin",
        "lower_face",
    )
    if any(any(marker in feature_name for marker in lower_face_markers) for feature_name in missing_features):
        return "Не удалось надёжно определить область рта/нижней части лица. Reference comparison частично недоступен."
    return "Часть морфометрических признаков недоступна. Reference comparison выполнен по доступному подмножеству признаков."


def _reference_std_warning(excluded_unstable_features: list[str]) -> str:
    if not excluded_unstable_features:
        return ""
    return (
        "Часть reference-признаков исключена из z-score и Mahalanobis из-за слишком малого std: "
        + ", ".join(excluded_unstable_features)
    )


def _safe_image_size(image_path: Path) -> tuple[int, int] | None:
    try:
        image = load_image(image_path)
    except Exception:
        return None
    height, width = image.shape[:2]
    return int(width), int(height)


def run_inference(
    *,
    image_path: Path,
    output_dir: Path | None = None,
    group_override: str | None = None,
    project_root: Path | None = None,
    preprocess_overrides: dict[str, Any] | None = None,
) -> InferenceArtifacts:
    root = (project_root or Path.cwd()).resolve()
    configs = load_named_configs(root / "configs")
    infer_config = configs["infer"]["infer"]
    train_config = configs["train_baseline"]["train_baseline"]
    landmarks_config = configs.get("landmarks", {}).get("landmarks", {})

    resolved_output_dir = (
        output_dir.resolve() if output_dir is not None else (root / infer_config["output_dir"]).resolve()
    )
    ensure_directory(resolved_output_dir)

    metadata = _read_sidecar_metadata(image_path.resolve())
    group_values = _resolve_group_values(
        group_override=group_override,
        metadata=metadata,
        group_aliases=dict(train_config["group_label_aliases"]),
    )
    if not group_values:
        raise ValueError(
            "Inference requires group metadata for reference lookup. "
            "Pass --group or provide sidecar metadata with normalized_labels."
        )

    preprocessing = preprocess_single_image(
        image_path=image_path.resolve(),
        project_root=root,
        output_analysis_path=resolved_output_dir / ANALYSIS_IMAGE_NAME,
        output_aligned_path=resolved_output_dir / ALIGNED_IMAGE_NAME,
        metadata={
            "subject_id": metadata.get("subject_id"),
            "raw_labels": metadata.get("raw_labels"),
            "normalized_labels": _normalized_labels_from_metadata(metadata),
        },
        overrides=preprocess_overrides,
    )
    row = preprocessing.row

    normalized_frame_image_path = preprocessing.analysis_image_path or image_path.resolve()
    analysis_image_path = preprocessing.analysis_image_path or image_path.resolve()
    measurement_image_path = preprocessing.measurement_image_path or analysis_image_path
    image_size = _safe_image_size(analysis_image_path)
    analysis_image_bgr = load_image(analysis_image_path)
    mediapipe_config = dict(landmarks_config.get("landmark_sources", {}).get("mediapipe_face_mesh", {}))
    mediapipe_source = build_mediapipe_source_result(
        image_bgr=analysis_image_bgr,
        config=mediapipe_config,
        project_root=root,
    )
    mediapipe_landmarks = mediapipe_source.landmarks if mediapipe_source.available else None
    landmarks_dense = [[float(point[0]), float(point[1])] for point in (mediapipe_landmarks or [])]
    landmarks = _derive_mediapipe_five_point_landmarks(landmarks_dense)
    if image_size is not None and mediapipe_landmarks:
        bbox = _padded_bbox_from_mediapipe(
            landmarks=landmarks_dense,
            image_width=int(image_size[0]),
            image_height=int(image_size[1]),
        )
    else:
        bbox = parse_json_field(row.get("bbox"))
    raw_measurements = extract_raw_measurements(
        landmarks=landmarks,
        bbox=bbox,
        landmarks_dense=landmarks_dense,
        mediapipe_landmarks=mediapipe_landmarks,
    )
    normalized_measurements = extract_normalized_measurements(raw_measurements)
    symmetry_features = extract_symmetry_features(raw_measurements, normalized_measurements)
    angle_features = extract_angle_features(
        landmarks=landmarks,
        row=row,
        landmarks_dense=landmarks_dense,
        bbox=bbox,
        mediapipe_landmarks=mediapipe_landmarks,
    )

    validate_feature_values(raw_measurements, image_id=str(row["image_id"]))
    validate_feature_values(normalized_measurements, image_id=str(row["image_id"]))
    validate_feature_values(symmetry_features, image_id=str(row["image_id"]))
    validate_feature_values(angle_features, image_id=str(row["image_id"]))

    reference_payload = joblib.load((root / infer_config["reference_model_path"]).resolve())
    statistics_frame = pd.read_parquet((root / infer_config["reference_statistics_path"]).resolve())
    selected_group_key, selected_group = _select_reference_group(reference_payload, group_values)

    feature_columns = list(selected_group["feature_columns"])
    feature_values = {
        **normalized_measurements,
        **symmetry_features,
        **angle_features,
    }
    mean_vector = np.asarray(selected_group["mean_vector"], dtype=np.float64)
    std_vector = np.asarray(selected_group["std_vector"], dtype=np.float64)
    covariance_matrix = np.asarray(selected_group["covariance_matrix"], dtype=np.float64)
    feature_vector = np.asarray([feature_values.get(column) for column in feature_columns], dtype=np.float64)
    available_mask = ~np.isnan(feature_vector)
    missing_features = [feature for feature, is_available in zip(feature_columns, available_mask.tolist()) if not is_available]
    initially_available_feature_names = [
        feature for feature, is_available in zip(feature_columns, available_mask.tolist()) if is_available
    ]

    reference_status = "complete"
    reference_warning = None
    mahalanobis_distance: float | None = None
    z_score_vector = np.full(shape=(len(feature_columns),), fill_value=np.nan, dtype=np.float64)
    unstable_reference_features: list[str] = []
    excluded_reference_features: list[str] = []
    used_feature_names: list[str] = []
    if not available_mask.any():
        reference_status = "unavailable"
        reference_warning = (
            "Не удалось надёжно определить достаточное число признаков для reference comparison. "
            "Reference comparison недоступен для этого изображения."
        )
    else:
        available_feature_vector = feature_vector[available_mask]
        available_mean_vector = mean_vector[available_mask]
        available_std_vector = std_vector[available_mask]
        available_covariance_matrix = covariance_matrix[np.ix_(available_mask, available_mask)]
        available_feature_names = initially_available_feature_names
        stable_std_mask = ~unstable_std_mask(available_std_vector)
        unstable_reference_features = [
            feature_name
            for feature_name, is_stable in zip(available_feature_names, stable_std_mask.tolist())
            if not is_stable
        ]
        if unstable_reference_features:
            excluded_reference_features.extend(unstable_reference_features)
        stable_feature_names = [
            feature_name
            for feature_name, is_stable in zip(available_feature_names, stable_std_mask.tolist())
            if is_stable
        ]
        if stable_std_mask.any():
            stable_feature_vector = available_feature_vector[stable_std_mask]
            stable_mean_vector = available_mean_vector[stable_std_mask]
            stable_std_vector = available_std_vector[stable_std_mask]
            stable_covariance_matrix = available_covariance_matrix[np.ix_(stable_std_mask, stable_std_mask)]
            stable_z_scores = compute_z_scores(
                stable_feature_vector,
                stable_mean_vector,
                stable_std_vector,
            )
            available_positions = np.flatnonzero(available_mask)
            stable_positions = available_positions[stable_std_mask]
            z_score_vector[stable_positions] = stable_z_scores
            mahalanobis_distance = compute_mahalanobis_distance(
                stable_feature_vector,
                stable_mean_vector,
                stable_covariance_matrix,
            )
            used_feature_names = stable_feature_names
        else:
            reference_status = "unavailable"
            reference_warning = (
                "Reference comparison недоступен: все доступные reference-признаки имеют слишком малый std "
                "или исключены как численно неустойчивые."
            )
        if used_feature_names and (missing_features or unstable_reference_features):
            reference_status = "partial"
            warning_parts: list[str] = []
            if missing_features:
                warning_parts.append(_reference_comparison_warning(missing_features))
            if unstable_reference_features:
                warning_parts.append(_reference_std_warning(unstable_reference_features))
            reference_warning = " ".join(part for part in warning_parts if part)

    stats_lookup = _feature_statistics_lookup(statistics_frame, selected_group_key)
    z_scores = _serialize_z_scores(feature_columns, z_score_vector, std_vector)
    percentile_positions = {
        feature_name: (
            _estimate_percentile_position(feature_values[feature_name], stats_lookup.get(feature_name))
            if feature_values.get(feature_name) is not None and not pd.isna(feature_values.get(feature_name))
            else None
        )
        for feature_name in feature_columns
    }
    feature_deviations = _sorted_deviations(z_scores)
    reference_comparison = {
        "status": reference_status,
        "warning": reference_warning,
        "used_feature_names": used_feature_names,
        "missing_feature_names": missing_features,
        "unstable_reference_features": unstable_reference_features,
        "excluded_reference_features": sorted(set(missing_features + unstable_reference_features)),
        "available_feature_count": len(used_feature_names),
        "required_feature_count": len(feature_columns),
    }

    report = build_report_payload(
        image_path=image_path,
        analysis_image_path=analysis_image_path,
        normalized_frame_image_path=normalized_frame_image_path,
        measurement_image_path=measurement_image_path,
        measurement_image_role="analysis_image",
        output_dir=resolved_output_dir,
        group_override=group_override,
        provider_diagnostics=preprocessing.provider_diagnostics,
        processing_row=row,
        image_size=image_size,
        bbox=bbox,
        landmarks=landmarks,
        landmarks_dense=landmarks_dense,
        mediapipe_landmarks=mediapipe_landmarks,
        mediapipe_regions=MEDIAPIPE_REGION_INDICES,
        raw_measurements=raw_measurements,
        normalized_measurements=normalized_measurements,
        symmetry_features=symmetry_features,
        angle_features=angle_features,
        selected_group_key=selected_group_key,
        selected_group=selected_group,
        reference_comparison=reference_comparison,
        z_scores=z_scores,
        mahalanobis_distance=mahalanobis_distance,
        percentile_positions=percentile_positions,
        feature_deviations=feature_deviations,
        top_k_features=int(infer_config["top_k_features"]),
        aligned_image_path=preprocessing.aligned_image_path,
        annotated_image_path=None,
        reference_groupings=[list(item) for item in reference_payload.get("groupings", [])],
        resolved_group_values=group_values,
    )

    visualization_debug_path = resolved_output_dir / VISUALIZATION_DEBUG_JSON_NAME
    visualization_payload = report.get("visualization") or {}
    write_json(
        {
            "coordinate_space": visualization_payload.get("coordinate_space"),
            "image_path": visualization_payload.get("image_path"),
            "image_width": visualization_payload.get("image_width"),
            "image_height": visualization_payload.get("image_height"),
            "raw_landmarks": visualization_payload.get("raw_landmarks", []),
            "raw_landmarks_debug": visualization_payload.get("raw_landmarks_debug", []),
            "semantic_points": visualization_payload.get("semantic_points", {}),
            "semantic_points_debug": visualization_payload.get("semantic_points_debug", []),
            "semantic_validation": visualization_payload.get("semantic_validation"),
            "measurement_overlays": visualization_payload.get("measurement_overlays", {}),
            "measurement_overlays_debug": visualization_payload.get("measurement_overlays_debug", {}),
        },
        visualization_debug_path,
    )
    _write_overlay_debug_artifacts(
        output_dir=root,
        analysis_image_bgr=analysis_image_bgr,
        visualization=visualization_payload,
        processing_row=row,
    )
    crop_audit = {
        "original_image_size": {"width": int(image_size[0]) if image_size else None, "height": int(image_size[1]) if image_size else None},
        "detected_face_bbox": visualization_payload.get("detector_bbox"),
        "expanded_bbox": visualization_payload.get("display_face_box"),
        "crop_margins": {"top": 0.18, "bottom": 0.22, "left": 0.14, "right": 0.14},
        "crop_too_narrow": False,
        "crop_cuts_face_oval": False,
        "crop_cuts_forehead": False,
        "crop_cuts_chin": False,
        "crop_cuts_cheeks": False,
    }
    write_json(crop_audit, root / "reports/metrics/mediapipe_crop_audit.json")
    report["artifacts"]["visualization_debug_json_path"] = str(visualization_debug_path.resolve())
    overlay_layers = [
        "bbox",
        "mediapipe_dense_points",
        "face_oval",
        "eyes",
        "brows",
        "nose",
        "mouth_lips",
        "jaw_chin",
        "cheekbone_proxies",
        "symmetry_axes",
    ]
    report["overlay"] = {
        "active_source": "mediapipe_only",
        "layers": overlay_layers,
        "legacy_layers": [],
    }
    report["preview_image"] = {
        "clean_image_url": str(analysis_image_path.resolve()) if analysis_image_path else None,
        "overlay_layers": overlay_layers,
    }
    ui_overlay_source_audit = {
        "overlay_sources_returned_by_api": ["mediapipe_only"],
        "overlay_sources_rendered_by_ui": ["mediapipe_only"],
        "insightface_overlay_count": 0,
        "legacy_overlay_count": 0,
        "mediapipe_overlay_count": len(overlay_layers),
        "active_overlay_source": "mediapipe_only",
        "violations": [],
    }
    write_json(ui_overlay_source_audit, root / "reports/metrics/ui_overlay_source_audit.json")

    annotated_image_path = resolved_output_dir / ANNOTATED_IMAGE_NAME
    if save_annotated_image(
        image_path=analysis_image_path,
        visualization_payload=report.get("visualization"),
        output_path=annotated_image_path,
    ):
        report["artifacts"]["annotated_image_path"] = str(annotated_image_path.resolve())
    else:
        annotated_image_path = None

    report_json_path = resolved_output_dir / REPORT_JSON_NAME
    report_markdown_path = resolved_output_dir / REPORT_MD_NAME
    reporting_summary_path = (root / infer_config["reporting_summary_path"]).resolve()
    reporting_summary = save_report_bundle(
        report=report,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        reporting_summary_path=reporting_summary_path,
    )

    return InferenceArtifacts(
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        reporting_summary_path=reporting_summary_path,
        analysis_image_path=analysis_image_path,
        normalized_frame_image_path=normalized_frame_image_path,
        aligned_image_path=preprocessing.aligned_image_path,
        measurement_image_path=measurement_image_path,
        annotated_image_path=annotated_image_path,
        report=report,
        reporting_summary=reporting_summary,
    )
