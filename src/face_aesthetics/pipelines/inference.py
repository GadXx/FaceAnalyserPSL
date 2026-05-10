from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
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
from src.face_aesthetics.data.io import load_image
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
        result[feature_name] = None if unstable else float(z_score)
    return result


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

    landmarks = parse_json_field(row.get("landmarks"))
    landmarks_dense = parse_json_field(row.get("landmarks_dense"))
    bbox = parse_json_field(row.get("bbox"))
    normalized_frame_image_path = preprocessing.analysis_image_path or image_path.resolve()
    measurement_image_path = preprocessing.measurement_image_path or normalized_frame_image_path
    analysis_image_path = measurement_image_path
    image_size = _safe_image_size(analysis_image_path)
    raw_measurements = extract_raw_measurements(landmarks=landmarks, bbox=bbox, landmarks_dense=landmarks_dense)
    normalized_measurements = extract_normalized_measurements(raw_measurements)
    symmetry_features = extract_symmetry_features(raw_measurements, normalized_measurements)
    angle_features = extract_angle_features(landmarks=landmarks, row=row, landmarks_dense=landmarks_dense, bbox=bbox)

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
    feature_vector = np.asarray([feature_values.get(column) for column in feature_columns], dtype=np.float64)
    if np.isnan(feature_vector).any():
        missing = [feature for feature, value in zip(feature_columns, feature_vector.tolist()) if pd.isna(value)]
        raise ValueError(
            "Inference feature vector contains missing values for reference comparison. "
            f"Missing features: {missing}"
        )

    mean_vector = np.asarray(selected_group["mean_vector"], dtype=np.float64)
    std_vector = np.asarray(selected_group["std_vector"], dtype=np.float64)
    covariance_matrix = np.asarray(selected_group["covariance_matrix"], dtype=np.float64)
    z_score_vector = compute_z_scores(feature_vector, mean_vector, std_vector)
    mahalanobis_distance = compute_mahalanobis_distance(feature_vector, mean_vector, covariance_matrix)

    stats_lookup = _feature_statistics_lookup(statistics_frame, selected_group_key)
    z_scores = _serialize_z_scores(feature_columns, z_score_vector, std_vector)
    percentile_positions = {
        feature_name: _estimate_percentile_position(feature_values[feature_name], stats_lookup.get(feature_name))
        for feature_name in feature_columns
    }
    feature_deviations = _sorted_deviations(z_scores)

    report = build_report_payload(
        image_path=image_path,
        analysis_image_path=analysis_image_path,
        normalized_frame_image_path=normalized_frame_image_path,
        measurement_image_path=measurement_image_path,
        measurement_image_role=preprocessing.measurement_image_role,
        output_dir=resolved_output_dir,
        group_override=group_override,
        provider_diagnostics=preprocessing.provider_diagnostics,
        processing_row=row,
        image_size=image_size,
        bbox=bbox,
        landmarks=landmarks,
        landmarks_dense=landmarks_dense,
        raw_measurements=raw_measurements,
        normalized_measurements=normalized_measurements,
        symmetry_features=symmetry_features,
        angle_features=angle_features,
        selected_group_key=selected_group_key,
        selected_group=selected_group,
        z_scores=z_scores,
        mahalanobis_distance=float(mahalanobis_distance),
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
    report["artifacts"]["visualization_debug_json_path"] = str(visualization_debug_path.resolve())

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
