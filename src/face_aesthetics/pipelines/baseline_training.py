from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.face_aesthetics.data.io import read_parquet
from src.face_aesthetics.models.reference import (
    build_group_combinations,
    compute_mahalanobis_distance,
    compute_z_scores,
    group_descriptor,
    percentile_column_name,
)
from src.face_aesthetics.utils.config import load_named_configs
from src.face_aesthetics.utils.filesystem import ensure_parent_directory


@dataclass
class BaselineTrainingArtifacts:
    summary_json_path: Path
    group_statistics_path: Path
    reference_model_path: Path
    config_snapshot_path: Path
    summary: dict[str, Any]


def _save_json(payload: dict[str, Any], path: Path) -> None:
    ensure_parent_directory(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True, sort_keys=True)


def _parse_json_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        return json.loads(stripped)
    return {}


def _select_feature_columns(frame: pd.DataFrame, prefixes: list[str]) -> list[str]:
    selected: list[str] = []
    for column in frame.columns:
        if any(column.startswith(prefix) for prefix in prefixes):
            selected.append(column)
    return selected


def _manifest_group_frame(
    manifest_frame: pd.DataFrame,
    *,
    image_id_column: str,
    normalized_labels_column: str,
    group_aliases: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, record in manifest_frame.iterrows():
        normalized_labels = _parse_json_payload(record.get(normalized_labels_column))
        row: dict[str, Any] = {image_id_column: record.get(image_id_column)}
        for canonical_name, aliases in group_aliases.items():
            value = None
            for alias in aliases:
                alias_value = normalized_labels.get(alias)
                if alias_value is None:
                    continue
                normalized_value = str(alias_value).strip()
                if normalized_value:
                    value = normalized_value
                    break
            row[canonical_name] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve_group_columns(config: dict[str, Any], available_group_columns: list[str]) -> list[list[str]]:
    requested = config.get("group_by")
    if not requested:
        return build_group_combinations(available_group_columns)

    groupings: list[list[str]] = []
    for item in requested:
        if isinstance(item, str):
            groupings.append([item])
        else:
            groupings.append([str(column) for column in item])
    return groupings


def _filter_by_split(frame: pd.DataFrame, split_column: str, allowed_splits: list[str]) -> pd.DataFrame:
    if split_column not in frame.columns or not allowed_splits:
        return frame
    normalized_splits = frame[split_column].fillna("").astype(str).str.strip().str.lower()
    allowed = {split.strip().lower() for split in allowed_splits}
    if not allowed:
        return frame
    return frame.loc[normalized_splits.isin(allowed)].copy()


def _feature_percentiles(frame: pd.DataFrame, feature_column: str, percentiles: list[float]) -> dict[str, float | None]:
    series = frame[feature_column].dropna()
    if series.empty:
        return {percentile_column_name(percentile): None for percentile in percentiles}
    return {
        percentile_column_name(percentile): float(series.quantile(percentile))
        for percentile in percentiles
    }


def _descriptive_rows(
    group_frame: pd.DataFrame,
    *,
    descriptor_key: str,
    descriptor_values: dict[str, str],
    feature_columns: list[str],
    percentiles: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_column in feature_columns:
        series = group_frame[feature_column].dropna()
        if series.empty:
            continue
        quantiles = _feature_percentiles(group_frame, feature_column, percentiles)
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        rows.append(
            {
                "feature_name": feature_column,
                "sample_count": int(len(series)),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "median": float(series.median()),
                "iqr": float(q3 - q1),
                **quantiles,
                **descriptor_values,
                "group_key": descriptor_key,
            }
        )
    return rows


def _covariance_frame(group_frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    complete_cases = group_frame[feature_columns].dropna()
    if complete_cases.empty:
        return complete_cases
    return complete_cases.astype(np.float64)


def _build_reference_entries(
    joined_frame: pd.DataFrame,
    *,
    groupings: list[list[str]],
    feature_columns: list[str],
    minimum_sample_count: int,
    minimum_complete_case_count: int,
    percentiles: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    descriptive_rows: list[dict[str, Any]] = []
    reference_entries: dict[str, Any] = {}
    skipped_groups: list[dict[str, Any]] = []

    for group_columns in groupings:
        combination_name = "__".join(group_columns)
        combination_bucket: dict[str, Any] = {}
        for group_values, group_frame in joined_frame.groupby(group_columns, dropna=True):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            row_payload = {column: value for column, value in zip(group_columns, group_values)}
            descriptor = group_descriptor(row_payload, group_columns)
            if descriptor is None:
                continue

            group_size = int(len(group_frame))
            if group_size < minimum_sample_count:
                skipped_groups.append(
                    {
                        "combination_name": combination_name,
                        "group_key": descriptor.group_key,
                        "reason": "below_minimum_sample_count",
                        "sample_count": group_size,
                    }
                )
                continue

            complete_cases = _covariance_frame(group_frame, feature_columns)
            complete_case_count = int(len(complete_cases))
            if complete_case_count < minimum_complete_case_count:
                skipped_groups.append(
                    {
                        "combination_name": combination_name,
                        "group_key": descriptor.group_key,
                        "reason": "below_minimum_complete_case_count",
                        "sample_count": group_size,
                        "complete_case_count": complete_case_count,
                    }
                )
                continue

            mean_vector = complete_cases.mean(axis=0).to_numpy(dtype=np.float64)
            std_vector = complete_cases.std(axis=0, ddof=1).fillna(0.0).to_numpy(dtype=np.float64)
            covariance_matrix = np.cov(complete_cases.to_numpy(dtype=np.float64), rowvar=False)
            if covariance_matrix.ndim == 0:
                covariance_matrix = np.asarray([[float(covariance_matrix)]], dtype=np.float64)

            z_score_matrix = compute_z_scores(
                complete_cases.to_numpy(dtype=np.float64),
                mean_vector=mean_vector,
                std_vector=std_vector,
            )
            z_score_l2 = np.linalg.norm(z_score_matrix, axis=1)
            mahalanobis_distances = [
                compute_mahalanobis_distance(
                    feature_vector=feature_vector,
                    mean_vector=mean_vector,
                    covariance_matrix=covariance_matrix,
                )
                for feature_vector in complete_cases.to_numpy(dtype=np.float64)
            ]

            descriptive_rows.extend(
                _descriptive_rows(
                    group_frame,
                    descriptor_key=descriptor.group_key,
                    descriptor_values={
                        "combination_name": descriptor.combination_name,
                        **descriptor.group_values,
                    },
                    feature_columns=feature_columns,
                    percentiles=percentiles,
                )
            )

            combination_bucket[descriptor.group_key] = {
                "group_columns": group_columns,
                "group_values": descriptor.group_values,
                "sample_count": group_size,
                "complete_case_count": complete_case_count,
                "feature_columns": feature_columns,
                "mean_vector": mean_vector.tolist(),
                "std_vector": std_vector.tolist(),
                "covariance_matrix": np.asarray(covariance_matrix, dtype=np.float64).tolist(),
                "z_score_l2_mean": float(np.mean(z_score_l2)),
                "z_score_l2_p95": float(np.quantile(z_score_l2, 0.95)),
                "mahalanobis_mean": float(np.mean(mahalanobis_distances)),
                "mahalanobis_p95": float(np.quantile(mahalanobis_distances, 0.95)),
            }

        reference_entries[combination_name] = combination_bucket

    return descriptive_rows, reference_entries, skipped_groups


def run_baseline_training(project_root: Path | None = None) -> BaselineTrainingArtifacts:
    root = (project_root or Path.cwd()).resolve()
    config = load_named_configs(root / "configs")["train_baseline"]["train_baseline"]

    input_features_path = (root / config["input_features_path"]).resolve()
    group_manifest_path = (root / config["group_manifest_path"]).resolve()
    summary_json_path = (root / config["artifacts"]["summary_json_path"]).resolve()
    group_statistics_path = (root / config["artifacts"]["group_statistics_path"]).resolve()
    reference_model_path = (root / config["artifacts"]["reference_model_path"]).resolve()
    config_snapshot_path = (root / config["artifacts"]["config_snapshot_path"]).resolve()

    ensure_parent_directory(summary_json_path)
    ensure_parent_directory(group_statistics_path)
    ensure_parent_directory(reference_model_path)
    ensure_parent_directory(config_snapshot_path)
    _save_json(config, config_snapshot_path)

    features_frame = pd.DataFrame(read_parquet(input_features_path))
    manifest_frame = pd.DataFrame(
        read_parquet(
            group_manifest_path,
            columns=[
                config["image_id_column"],
                config["split_column"],
                config["normalized_labels_column"],
            ],
        )
    )

    feature_columns = _select_feature_columns(features_frame, list(config["feature_prefixes"]))
    if features_frame.empty or manifest_frame.empty or not feature_columns:
        pd.DataFrame(columns=["group_key", "feature_name"]).to_parquet(group_statistics_path, index=False)
        joblib.dump({"reference_groups": {}}, reference_model_path)
        summary = {
            "status": "skipped_no_data",
            "reason": "Reference model construction requires feature rows, manifest rows, and configured normalized feature columns.",
            "selected_feature_count": int(len(feature_columns)),
            "built_group_count": 0,
            "skipped_group_count": 0,
            "artifacts": {
                "summary_json_path": summary_json_path.relative_to(root).as_posix(),
                "group_statistics_path": group_statistics_path.relative_to(root).as_posix(),
                "reference_model_path": reference_model_path.relative_to(root).as_posix(),
                "config_snapshot_path": config_snapshot_path.relative_to(root).as_posix(),
            },
        }
        _save_json(summary, summary_json_path)
        return BaselineTrainingArtifacts(
            summary_json_path=summary_json_path,
            group_statistics_path=group_statistics_path,
            reference_model_path=reference_model_path,
            config_snapshot_path=config_snapshot_path,
            summary=summary,
        )

    manifest_groups = _manifest_group_frame(
        manifest_frame,
        image_id_column=str(config["image_id_column"]),
        normalized_labels_column=str(config["normalized_labels_column"]),
        group_aliases=dict(config["group_label_aliases"]),
    )
    joined = features_frame.merge(
        manifest_groups,
        on=str(config["image_id_column"]),
        how="inner",
        suffixes=("", "_manifest"),
    )
    joined = _filter_by_split(
        joined,
        split_column=str(config["split_column"]),
        allowed_splits=[str(value) for value in config.get("allowed_reference_splits", [])],
    )

    available_group_columns = [
        column for column in dict(config["group_label_aliases"]).keys() if column in joined.columns
    ]
    groupings = _resolve_group_columns(config, available_group_columns)

    descriptive_rows, reference_entries, skipped_groups = _build_reference_entries(
        joined,
        groupings=groupings,
        feature_columns=feature_columns,
        minimum_sample_count=int(config["minimum_group_sample_count"]),
        minimum_complete_case_count=int(config["minimum_complete_case_count"]),
        percentiles=[float(value) for value in config["percentiles"]],
    )

    statistics_frame = pd.DataFrame(descriptive_rows)
    if statistics_frame.empty:
        statistics_frame = pd.DataFrame(columns=["combination_name", "group_key", "feature_name"])
    statistics_frame.to_parquet(group_statistics_path, index=False)

    reference_payload = {
        "feature_columns": feature_columns,
        "groupings": groupings,
        "reference_groups": reference_entries,
        "group_label_aliases": config["group_label_aliases"],
        "minimum_group_sample_count": int(config["minimum_group_sample_count"]),
        "minimum_complete_case_count": int(config["minimum_complete_case_count"]),
    }
    joblib.dump(reference_payload, reference_model_path)

    built_group_count = sum(len(bucket) for bucket in reference_entries.values())
    summary = {
        "status": "completed",
        "input_feature_rows": int(len(features_frame)),
        "joined_rows": int(len(joined)),
        "selected_feature_count": int(len(feature_columns)),
        "selected_feature_columns": feature_columns,
        "grouping_columns_available": available_group_columns,
        "groupings_requested": groupings,
        "allowed_reference_splits": [str(value) for value in config.get("allowed_reference_splits", [])],
        "minimum_group_sample_count": int(config["minimum_group_sample_count"]),
        "minimum_complete_case_count": int(config["minimum_complete_case_count"]),
        "built_group_count": built_group_count,
        "skipped_group_count": int(len(skipped_groups)),
        "built_groups_by_combination": {
            combination_name: int(len(bucket))
            for combination_name, bucket in reference_entries.items()
        },
        "skipped_groups": skipped_groups,
        "artifacts": {
            "summary_json_path": summary_json_path.relative_to(root).as_posix(),
            "group_statistics_path": group_statistics_path.relative_to(root).as_posix(),
            "reference_model_path": reference_model_path.relative_to(root).as_posix(),
            "config_snapshot_path": config_snapshot_path.relative_to(root).as_posix(),
        },
    }
    _save_json(summary, summary_json_path)

    return BaselineTrainingArtifacts(
        summary_json_path=summary_json_path,
        group_statistics_path=group_statistics_path,
        reference_model_path=reference_model_path,
        config_snapshot_path=config_snapshot_path,
        summary=summary,
    )

