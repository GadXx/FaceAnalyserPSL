from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.face_aesthetics.features.extraction import (
    ANGLE_FEATURE_COLUMNS,
    NORMALIZED_MEASUREMENT_COLUMNS,
    RAW_MEASUREMENT_COLUMNS,
    SYMMETRY_FEATURE_COLUMNS,
    feature_registry_rows,
    geometry_feature_columns,
)
from src.face_aesthetics.utils.config import load_named_configs
from src.face_aesthetics.utils.filesystem import ensure_parent_directory


SCORING_CONFIG_FILES = (
    "base.yaml",
    "data.yaml",
    "features.yaml",
    "landmarks.yaml",
    "preprocess.yaml",
    "reference.yaml",
    "scoring.yaml",
)

SCORING_METADATA_COLUMNS = [
    "image_id",
    "image_path",
    "file_name",
    "source_score_folder",
    "target_score",
    "target_group_label",
    "split",
    "dataset_name",
    "sample_id",
    "subject_id",
    "quality_pass",
    "quality_fail_reason",
    "processing_status",
    "processing_error_message",
    "raw_labels",
    "normalized_labels",
    "analysis_image_path",
    "aligned_image_path",
    "reference_model",
    "reference_source",
    "reference_stats_path",
    "reference_filters",
]

SCORING_GROUP_COLUMNS = [
    "sex_group_normalized",
    "age_group_normalized",
    "ethnicity_group_normalized",
]

SCORING_EXCLUDED_METADATA_COLUMNS = {
    "image_id",
    "image_path",
    "file_name",
    "source_score_folder",
    "target_score",
    "target_group_label",
    "split",
    "dataset_name",
    "sample_id",
    "subject_id",
    "quality_pass",
    "quality_fail_reason",
    "processing_status",
    "processing_error_message",
    "raw_labels",
    "normalized_labels",
    "analysis_image_path",
    "aligned_image_path",
    "reference_model",
    "reference_source",
    "reference_stats_path",
    "reference_filters",
    "sex_group_normalized",
    "age_group_normalized",
    "ethnicity_group_normalized",
}

GEOMETRY_ONLY_PREFIXES = ("ratio_", "sym_", "angle_")
GEOMETRY_ONLY_EXPLICIT = {
    "symmetry_index",
    "brow_symmetry_ratio",
    "midline_alignment_index",
    "upper_face_symmetry_index",
    "lower_face_symmetry_index",
    "pose_yaw_deg",
    "pose_pitch_deg",
    "pose_roll_deg",
    "canthal_tilt_deg",
    "jaw_angle_asymmetry",
    "face_centerline_vs_nose_angle",
    "eye_line_vs_mouth_line_diff",
    "nose_to_chin_axis_diff",
    "lower_face_tilt_proxy",
}


def load_scoring_configs(project_root: Path) -> dict[str, dict[str, Any]]:
    return load_named_configs(project_root / "configs", file_names=SCORING_CONFIG_FILES)


def save_json(payload: dict[str, Any], path: Path) -> None:
    ensure_parent_directory(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True, sort_keys=True)


def scoring_feature_columns(config: dict[str, Any]) -> list[str]:
    registry_by_name = {
        str(row["feature_name"]): row
        for row in feature_registry_rows()
    }
    configured_families = config.get("include_feature_families", config.get("feature_include_families", []))
    families = set(str(value).strip().lower() for value in configured_families)
    columns: list[str] = []
    include_raw = bool(config.get("include_raw_pixel_features", False))
    if include_raw and (not families or "raw" in families):
        columns.extend(RAW_MEASUREMENT_COLUMNS)
    if not families or "normalized" in families:
        columns.extend(NORMALIZED_MEASUREMENT_COLUMNS)
    if not families or "symmetry" in families:
        columns.extend(SYMMETRY_FEATURE_COLUMNS)
    if not families or "angles" in families:
        columns.extend(ANGLE_FEATURE_COLUMNS)
    if not columns:
        columns = geometry_feature_columns()

    include_reference_relative = bool(config.get("include_reference_relative_features", True))
    reference_types = [str(value).strip().lower() for value in config.get("reference_relative_feature_types", ["z", "dir", "ratio", "absdev", "percentile"])]
    available_columns = [str(value) for value in config.get("available_columns", [])]
    if include_reference_relative and available_columns:
        for column in available_columns:
            if any(column.startswith(f"ref_{prefix}_") for prefix in reference_types):
                columns.append(column)

    excluded = {str(value) for value in config.get("feature_exclude_columns", [])}
    excluded_patterns = [str(value) for value in config.get("exclude_feature_patterns", [])]
    filtered: list[str] = []
    for column in columns:
        registry_row = registry_by_name.get(column, {})
        if column in excluded:
            continue
        if any(fnmatch(column, pattern) for pattern in excluded_patterns):
            continue
        if registry_row and not bool(registry_row.get("recommended_for_scorer", True)):
            continue
        filtered.append(column)
    return filtered


def is_geometry_only_feature(column: str) -> bool:
    reference_prefixes = ("ref_z_", "ref_dir_", "ref_ratio_", "ref_absdev_", "ref_percentile_")
    if column.startswith(reference_prefixes):
        base_name = column
        for prefix in reference_prefixes:
            if column.startswith(prefix):
                base_name = column[len(prefix):]
                break
        return is_geometry_only_feature(base_name)
    if column in GEOMETRY_ONLY_EXPLICIT:
        return True
    return column.startswith(GEOMETRY_ONLY_PREFIXES)


def feature_policy_report(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    selected = scoring_feature_columns(config)
    selected_set = set(selected)
    registry_by_name = {str(row["feature_name"]): row for row in feature_registry_rows()}
    report_rows: list[dict[str, Any]] = []
    for column in frame.columns:
        if column in selected_set:
            continue
        reason = "not_allowed_by_policy"
        registry_row = registry_by_name.get(column, {})
        if registry_row and not bool(registry_row.get("recommended_for_scorer", True)):
            reason = str(registry_row.get("exclusion_reason") or "not_recommended_for_scorer")
        if column in SCORING_EXCLUDED_METADATA_COLUMNS:
            reason = "metadata"
        elif column in {"target_score", "target_group_label"}:
            reason = "target_leakage"
        elif column.endswith("_path") or "path" in column:
            reason = "image_path"
        elif column.startswith("raw_") or column.endswith("_px"):
            reason = "raw_pixel_feature"
        elif column.startswith("emb") or "embedding" in column:
            reason = "embedding_disabled"
        elif not is_geometry_only_feature(column):
            reason = "not_in_geometry_allowlist"
        report_rows.append({"feature_name": column, "exclusion_reason": reason})
    return {
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "excluded_features": report_rows,
        "feature_policy": {
            "feature_mode": config.get("feature_mode", "geometry_only"),
            "allow_raw_pixel_features": bool(config.get("allow_raw_pixel_features", False)),
            "allow_embeddings": bool(config.get("allow_embeddings", False)),
            "allow_metadata_features": bool(config.get("allow_metadata_features", False)),
        },
    }


def numeric_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    available = [column for column in feature_columns if column in frame.columns]
    if not available:
        return pd.DataFrame(index=frame.index)
    numeric = frame.loc[:, available].apply(pd.to_numeric, errors="coerce")
    return numeric


def clip_prediction(value: float, *, lower: float | None, upper: float | None) -> float:
    clipped = float(value)
    if lower is not None:
        clipped = max(clipped, float(lower))
    if upper is not None:
        clipped = min(clipped, float(upper))
    return clipped


def nearest_group_payload(prediction: float, groups: list[float]) -> dict[str, Any]:
    if not groups:
        return {
            "nearest_score_group": None,
            "distance_to_nearest_group": None,
            "distances_to_groups": {},
            "soft_group_membership": {},
        }
    distances = {str(group): float(abs(prediction - float(group))) for group in groups}
    nearest_group = min(groups, key=lambda group: abs(prediction - float(group)))
    scale = max(np.std(np.asarray(groups, dtype=np.float64)), 1e-6)
    soft_membership = {}
    weights = {}
    for group in groups:
        weight = float(np.exp(-abs(prediction - float(group)) / scale))
        weights[str(group)] = weight
    total_weight = sum(weights.values()) or 1.0
    for group_label, weight in weights.items():
        soft_membership[group_label] = float(weight / total_weight)
    return {
        "nearest_score_group": float(nearest_group),
        "distance_to_nearest_group": float(abs(prediction - float(nearest_group))),
        "distances_to_groups": distances,
        "soft_group_membership": soft_membership,
    }


def inverse_frequency_weights(labels: pd.Series) -> pd.Series:
    normalized = labels.fillna("").astype(str)
    counts = normalized.value_counts()
    return normalized.map(lambda value: 1.0 / float(counts.get(value, 1)))
