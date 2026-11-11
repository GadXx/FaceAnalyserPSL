from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.face_aesthetics.data.io import read_parquet
from src.face_aesthetics.features.extraction import reference_feature_columns
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


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        stripped = value.strip()
        return bool(stripped and stripped.lower() not in {"null", "nan"})
    return not pd.isna(value)


def _select_feature_columns(frame: pd.DataFrame, prefixes: list[str]) -> list[str]:
    allowed_reference_features = set(reference_feature_columns())
    selected: list[str] = []
    for column in frame.columns:
        if any(column.startswith(prefix) for prefix in prefixes) and column in allowed_reference_features:
            selected.append(column)
    return selected


def _manifest_group_frame(
    manifest_frame: pd.DataFrame,
    *,
    join_columns: list[str],
    normalized_labels_column: str,
    group_aliases: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, record in manifest_frame.iterrows():
        normalized_labels = _parse_json_payload(record.get(normalized_labels_column))
        row: dict[str, Any] = {column: record.get(column) for column in join_columns}
        row["normalized_labels"] = record.get(normalized_labels_column)
        for canonical_name, aliases in group_aliases.items():
            value = record.get(canonical_name)
            if _is_nonempty(value):
                row[canonical_name] = str(value).strip()
                continue

            resolved = None
            for alias in [canonical_name, *aliases]:
                alias_value = record.get(alias)
                if alias_value is None:
                    alias_value = normalized_labels.get(alias)
                if _is_nonempty(alias_value):
                    normalized_value = str(alias_value).strip()
                    resolved = normalized_value
                    break
            row[canonical_name] = resolved
        rows.append(row)
    return pd.DataFrame(rows)


def _rows_with_group_labels(frame: pd.DataFrame, group_columns: list[str]) -> int:
    if frame.empty:
        return 0
    available_columns = [column for column in group_columns if column in frame.columns]
    if not available_columns:
        return 0

    def _row_has_labels(record: pd.Series) -> bool:
        return any(_is_nonempty(record.get(column)) for column in available_columns)

    return int(frame.apply(_row_has_labels, axis=1).sum())


def _join_group_labels(
    features_frame: pd.DataFrame,
    manifest_groups: pd.DataFrame,
    *,
    image_id_column: str,
    sample_id_column: str,
    dataset_name_column: str,
) -> pd.DataFrame:
    joined = features_frame.copy()
    joined["_join_row_index"] = np.arange(len(joined), dtype=np.int64)

    for column in manifest_groups.columns:
        if column not in joined.columns:
            joined[column] = None

    manifest_by_image = manifest_groups.drop_duplicates(subset=[image_id_column], keep="last")
    joined = joined.merge(
        manifest_by_image,
        on=image_id_column,
        how="left",
        suffixes=("", "_manifest"),
    )

    manifest_suffix_columns = [
        column for column in joined.columns if column.endswith("_manifest")
    ]
    for manifest_column in manifest_suffix_columns:
        base_column = manifest_column[: -len("_manifest")]
        if base_column not in joined.columns:
            joined[base_column] = joined[manifest_column]
        else:
            joined[base_column] = joined[base_column].where(
                joined[base_column].map(_is_nonempty),
                joined[manifest_column],
            )
        joined = joined.drop(columns=[manifest_column])

    join_columns = [column for column in [sample_id_column, dataset_name_column] if column in manifest_groups.columns]
    if join_columns:
        manifest_by_sample = manifest_groups.drop_duplicates(subset=join_columns, keep="last")
        relevant_columns = [
            column
            for column in (
                "sex_group_normalized",
                "age_group_normalized",
                "ethnicity_group_normalized",
                "normalized_labels",
            )
            if column in joined.columns
        ]
        missing_mask = pd.Series(False, index=joined.index)
        if relevant_columns:
            missing_mask = joined[relevant_columns].apply(
                lambda record: not any(_is_nonempty(value) for value in record.tolist()),
                axis=1,
            )
        fallback_candidates = joined.loc[missing_mask].copy()
        if not fallback_candidates.empty:
            fallback_joined = fallback_candidates.merge(
                manifest_by_sample,
                on=join_columns,
                how="left",
                suffixes=("", "_sample"),
            )
            suffix_columns = [
                column for column in fallback_joined.columns if column.endswith("_sample")
            ]
            for suffix_column in suffix_columns:
                base_column = suffix_column[: -len("_sample")]
                if base_column not in joined.columns:
                    joined[base_column] = None
                target_index = fallback_joined["_join_row_index"].to_numpy(dtype=np.int64)
                existing_values = joined.loc[target_index, base_column]
                fallback_values = fallback_joined[suffix_column].to_numpy()
                joined.loc[target_index, base_column] = existing_values.where(
                    existing_values.map(_is_nonempty),
                    fallback_values,
                )
    return joined.drop(columns=["_join_row_index"])


def _build_skip_summary(
    *,
    status: str,
    reason: str,
    root: Path,
    summary_json_path: Path,
    group_statistics_path: Path,
    reference_model_path: Path,
    config_snapshot_path: Path,
    feature_rows: int,
    rows_with_group_labels: int,
    rows_after_join: int,
    selected_feature_columns: list[str],
    group_columns: list[str],
    min_sample_count: int,
    min_complete_case_count: int,
    skip_reasons: dict[str, int] | None = None,
    available_label_columns: list[str] | None = None,
    missing_label_columns: list[str] | None = None,
    preserved_existing_artifacts: bool = False,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "feature_rows": int(feature_rows),
        "rows_with_group_labels": int(rows_with_group_labels),
        "rows_after_join": int(rows_after_join),
        "selected_feature_count": int(len(selected_feature_columns)),
        "selected_feature_columns": selected_feature_columns,
        "grouping_columns_available": group_columns,
        "available_label_columns": available_label_columns or [],
        "missing_label_columns": missing_label_columns or [],
        "min_sample_count": int(min_sample_count),
        "minimum_complete_case_count": int(min_complete_case_count),
        "built_group_count": 0,
        "skipped_group_count": int(sum((skip_reasons or {}).values())),
        "skip_reasons": skip_reasons or {},
        "artifacts_written": False,
        "preserved_existing_artifacts": preserved_existing_artifacts,
        "artifacts": {
            "summary_json_path": summary_json_path.relative_to(root).as_posix(),
            "group_statistics_path": group_statistics_path.relative_to(root).as_posix(),
            "reference_model_path": reference_model_path.relative_to(root).as_posix(),
            "config_snapshot_path": config_snapshot_path.relative_to(root).as_posix(),
        },
    }


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
    group_label_columns = list(dict(config["group_label_aliases"]).keys())
    manifest_frame = pd.DataFrame(
        read_parquet(
            group_manifest_path,
            columns=[
                config["image_id_column"],
                "dataset_name",
                "sample_id",
                config["split_column"],
                config["normalized_labels_column"],
                *group_label_columns,
            ],
        )
    )

    feature_columns = _select_feature_columns(features_frame, list(config["feature_prefixes"]))
    if features_frame.empty or manifest_frame.empty or not feature_columns:
        summary = _build_skip_summary(
            status="skipped_no_data",
            reason="Reference model construction requires feature rows, manifest rows, and configured normalized feature columns.",
            root=root,
            summary_json_path=summary_json_path,
            group_statistics_path=group_statistics_path,
            reference_model_path=reference_model_path,
            config_snapshot_path=config_snapshot_path,
            feature_rows=int(len(features_frame)),
            rows_with_group_labels=0,
            rows_after_join=0,
            selected_feature_columns=feature_columns,
            group_columns=[],
            min_sample_count=int(config["minimum_group_sample_count"]),
            min_complete_case_count=int(config["minimum_complete_case_count"]),
            available_label_columns=[column for column in group_label_columns if column in manifest_frame.columns],
            missing_label_columns=[column for column in group_label_columns if column not in manifest_frame.columns],
            preserved_existing_artifacts=group_statistics_path.exists() or reference_model_path.exists(),
        )
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
        join_columns=[
            str(config["image_id_column"]),
            "sample_id",
            "dataset_name",
            str(config["split_column"]),
        ],
        normalized_labels_column=str(config["normalized_labels_column"]),
        group_aliases=dict(config["group_label_aliases"]),
    )
    rows_with_group_labels_before_join = _rows_with_group_labels(features_frame, group_label_columns)
    joined = _join_group_labels(
        features_frame,
        manifest_groups,
        image_id_column=str(config["image_id_column"]),
        sample_id_column="sample_id",
        dataset_name_column="dataset_name",
    )
    joined = _filter_by_split(
        joined,
        split_column=str(config["split_column"]),
        allowed_splits=[str(value) for value in config.get("allowed_reference_splits", [])],
    )
    rows_with_group_labels_after_join = _rows_with_group_labels(joined, group_label_columns)

    available_group_columns = [
        column
        for column in group_label_columns
        if column in joined.columns and joined[column].map(_is_nonempty).any()
    ]
    if not available_group_columns:
        summary = _build_skip_summary(
            status="failed_missing_group_labels",
            reason="Reference model construction could not resolve any normalized subgroup labels after joining features with the preprocessing manifest.",
            root=root,
            summary_json_path=summary_json_path,
            group_statistics_path=group_statistics_path,
            reference_model_path=reference_model_path,
            config_snapshot_path=config_snapshot_path,
            feature_rows=int(len(features_frame)),
            rows_with_group_labels=rows_with_group_labels_after_join,
            rows_after_join=int(len(joined)),
            selected_feature_columns=feature_columns,
            group_columns=[],
            min_sample_count=int(config["minimum_group_sample_count"]),
            min_complete_case_count=int(config["minimum_complete_case_count"]),
            available_label_columns=[column for column in group_label_columns if column in joined.columns],
            missing_label_columns=[column for column in group_label_columns if column not in joined.columns],
            preserved_existing_artifacts=group_statistics_path.exists() or reference_model_path.exists(),
        )
        _save_json(summary, summary_json_path)
        return BaselineTrainingArtifacts(
            summary_json_path=summary_json_path,
            group_statistics_path=group_statistics_path,
            reference_model_path=reference_model_path,
            config_snapshot_path=config_snapshot_path,
            summary=summary,
        )
    groupings = _resolve_group_columns(config, available_group_columns)

    descriptive_rows, reference_entries, skipped_groups = _build_reference_entries(
        joined,
        groupings=groupings,
        feature_columns=feature_columns,
        minimum_sample_count=int(config["minimum_group_sample_count"]),
        minimum_complete_case_count=int(config["minimum_complete_case_count"]),
        percentiles=[float(value) for value in config["percentiles"]],
    )

    built_group_count = sum(len(bucket) for bucket in reference_entries.values())
    skip_reason_counts: dict[str, int] = {}
    for skipped_group in skipped_groups:
        reason = str(skipped_group.get("reason", "unknown"))
        skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1

    if built_group_count == 0:
        summary = _build_skip_summary(
            status="failed_no_reference_groups",
            reason="Reference model construction produced zero valid groups. Existing reference artifacts were preserved.",
            root=root,
            summary_json_path=summary_json_path,
            group_statistics_path=group_statistics_path,
            reference_model_path=reference_model_path,
            config_snapshot_path=config_snapshot_path,
            feature_rows=int(len(features_frame)),
            rows_with_group_labels=rows_with_group_labels_after_join,
            rows_after_join=int(len(joined)),
            selected_feature_columns=feature_columns,
            group_columns=available_group_columns,
            min_sample_count=int(config["minimum_group_sample_count"]),
            min_complete_case_count=int(config["minimum_complete_case_count"]),
            skip_reasons=skip_reason_counts,
            available_label_columns=[column for column in group_label_columns if column in joined.columns],
            missing_label_columns=[column for column in group_label_columns if column not in joined.columns],
            preserved_existing_artifacts=group_statistics_path.exists() or reference_model_path.exists(),
        )
        summary["groupings_requested"] = groupings
        summary["allowed_reference_splits"] = [str(value) for value in config.get("allowed_reference_splits", [])]
        summary["input_feature_rows"] = int(len(features_frame))
        summary["rows_with_group_labels_before_join"] = int(rows_with_group_labels_before_join)
        summary["skipped_groups"] = skipped_groups
        _save_json(summary, summary_json_path)
        return BaselineTrainingArtifacts(
            summary_json_path=summary_json_path,
            group_statistics_path=group_statistics_path,
            reference_model_path=reference_model_path,
            config_snapshot_path=config_snapshot_path,
            summary=summary,
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

    summary = {
        "status": "completed",
        "feature_rows": int(len(features_frame)),
        "input_feature_rows": int(len(features_frame)),
        "rows_with_group_labels": int(rows_with_group_labels_after_join),
        "rows_with_group_labels_before_join": int(rows_with_group_labels_before_join),
        "rows_after_join": int(len(joined)),
        "joined_rows": int(len(joined)),
        "selected_feature_count": int(len(feature_columns)),
        "selected_feature_columns": feature_columns,
        "grouping_columns_available": available_group_columns,
        "available_label_columns": [column for column in group_label_columns if column in joined.columns],
        "missing_label_columns": [column for column in group_label_columns if column not in joined.columns],
        "groupings_requested": groupings,
        "allowed_reference_splits": [str(value) for value in config.get("allowed_reference_splits", [])],
        "minimum_group_sample_count": int(config["minimum_group_sample_count"]),
        "min_sample_count": int(config["minimum_group_sample_count"]),
        "minimum_complete_case_count": int(config["minimum_complete_case_count"]),
        "built_group_count": built_group_count,
        "skipped_group_count": int(len(skipped_groups)),
        "skip_reasons": skip_reason_counts,
        "built_groups_by_combination": {
            combination_name: int(len(bucket))
            for combination_name, bucket in reference_entries.items()
        },
        "skipped_groups": skipped_groups,
        "artifacts_written": True,
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
