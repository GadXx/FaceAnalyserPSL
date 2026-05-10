from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.face_aesthetics.data.io import read_parquet, write_json
from src.face_aesthetics.utils.config import load_named_configs
from src.face_aesthetics.utils.filesystem import ensure_parent_directory


CONFIG_FILES = (
    "base.yaml",
    "data.yaml",
    "features.yaml",
    "train_baseline.yaml",
    "subgroup_validation.yaml",
)
GROUP_DIMENSIONS = (
    "dataset_name",
    "sex_group_normalized",
    "age_group_normalized",
    "ethnicity_group_normalized",
    "quality_bucket",
)
PREPROCESSING_COVERAGE = "preprocessing_coverage"
FEATURE_COVERAGE = "feature_coverage"
REFERENCE_COVERAGE = "reference_coverage"


@dataclass
class SubgroupValidationArtifacts:
    validation_json_path: Path
    reference_coverage_json_path: Path
    flat_table_path: Path
    validation_summary: dict[str, Any]
    reference_summary: dict[str, Any]


def _parse_json_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        stripped = value.strip()
        return bool(stripped and stripped.lower() not in {"null", "nan"})
    return not pd.isna(value)


def _normalized_group_columns(
    frame: pd.DataFrame,
    *,
    image_id_column: str,
    normalized_labels_column: str,
    group_aliases: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, record in frame.iterrows():
        normalized_labels = _parse_json_payload(record.get(normalized_labels_column))
        row: dict[str, Any] = {image_id_column: record.get(image_id_column)}
        for canonical_name, aliases in group_aliases.items():
            resolved = None
            for alias in aliases:
                raw_value = normalized_labels.get(alias)
                if raw_value is None:
                    continue
                normalized_value = str(raw_value).strip()
                if normalized_value:
                    resolved = normalized_value
                    break
            row[canonical_name] = resolved
        rows.append(row)
    return pd.DataFrame(rows)


def _quality_bucket(row: pd.Series, *, quality_pass_column: str, quality_fail_reason_column: str, quality_gate_status_column: str) -> str:
    if _is_truthy(row.get(quality_pass_column)):
        return "passed"
    fail_reason = row.get(quality_fail_reason_column)
    if _is_nonempty(fail_reason):
        return str(fail_reason).strip()
    gate_status = row.get(quality_gate_status_column)
    if _is_nonempty(gate_status):
        return str(gate_status).strip()
    return "unknown"


def _nonempty_unique_count(series: pd.Series) -> int | None:
    filtered = [str(value).strip() for value in series.tolist() if _is_nonempty(value)]
    if not filtered:
        return None
    return len(set(filtered))


def _warning_messages(
    *,
    coverage_type: str,
    row_count: int,
    unique_subject_count: int | None,
    thresholds: dict[str, Any],
    preprocessing_success_rate: float | None = None,
    feature_coverage_rate: float | None = None,
    mean_feature_completeness_rate: float | None = None,
    sample_count: int | None = None,
    complete_case_count: int | None = None,
) -> list[str]:
    warnings: list[str] = []
    min_rows = int(thresholds.get("min_rows", 0))
    min_unique_subjects = int(thresholds.get("min_unique_subjects", 0))

    if row_count < min_rows:
        warnings.append(f"low_row_count<{min_rows}")
    if unique_subject_count is not None and unique_subject_count < min_unique_subjects:
        warnings.append(f"low_unique_subject_count<{min_unique_subjects}")

    if coverage_type == PREPROCESSING_COVERAGE and preprocessing_success_rate is not None:
        threshold = float(thresholds.get("min_preprocessing_success_rate", 0.0))
        if preprocessing_success_rate < threshold:
            warnings.append(f"low_preprocessing_success_rate<{threshold}")

    if coverage_type == FEATURE_COVERAGE:
        if feature_coverage_rate is not None:
            threshold = float(thresholds.get("min_feature_coverage_rate", 0.0))
            if feature_coverage_rate < threshold:
                warnings.append(f"low_feature_coverage_rate<{threshold}")
        if mean_feature_completeness_rate is not None:
            threshold = float(thresholds.get("min_mean_feature_completeness_rate", 0.0))
            if mean_feature_completeness_rate < threshold:
                warnings.append(f"low_mean_feature_completeness_rate<{threshold}")

    if coverage_type == REFERENCE_COVERAGE:
        min_reference_sample_count = int(thresholds.get("min_reference_sample_count", 0))
        min_reference_complete_case_count = int(thresholds.get("min_reference_complete_case_count", 0))
        if sample_count is not None and sample_count < min_reference_sample_count:
            warnings.append(f"low_reference_sample_count<{min_reference_sample_count}")
        if complete_case_count is not None and complete_case_count < min_reference_complete_case_count:
            warnings.append(f"low_reference_complete_case_count<{min_reference_complete_case_count}")

    return warnings


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    ignored = {
        "dataset_name",
        "sample_id",
        "subject_id",
        "image_id",
        "split",
        "image_path",
        "aligned_image_path",
        "reference_model",
    }
    return [column for column in frame.columns if column not in ignored]


def _preprocessing_coverage_rows(
    frame: pd.DataFrame,
    *,
    dimension: str,
    dataset_name_column: str,
    subject_id_column: str,
    quality_pass_column: str,
    processing_status_column: str,
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subgroup_value, subset in frame.groupby(dimension, dropna=False):
        total_rows = int(len(subset))
        processed_rows = int(
            subset[processing_status_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"processed", "success"})
            .sum()
        )
        success_rows = int(subset[quality_pass_column].apply(_is_truthy).sum())
        unique_subject_count = _nonempty_unique_count(subset[subject_id_column]) if subject_id_column in subset.columns else None
        preprocessing_success_rate = _safe_rate(success_rows, total_rows)
        row = {
            "coverage_type": PREPROCESSING_COVERAGE,
            "group_dimension": dimension,
            "group_value": "unknown" if pd.isna(subgroup_value) or subgroup_value is None else str(subgroup_value),
            "row_count": total_rows,
            "unique_subject_count": unique_subject_count,
            "processed_row_count": processed_rows,
            "preprocessing_success_row_count": success_rows,
            "processing_status_coverage_rate": _safe_rate(processed_rows, total_rows),
            "preprocessing_success_rate": preprocessing_success_rate,
            "dataset_count": int(subset[dataset_name_column].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique())
            if dataset_name_column in subset.columns
            else None,
        }
        row["warnings"] = _warning_messages(
            coverage_type=PREPROCESSING_COVERAGE,
            row_count=total_rows,
            unique_subject_count=unique_subject_count,
            thresholds=thresholds,
            preprocessing_success_rate=preprocessing_success_rate,
        )
        rows.append(row)
    return sorted(rows, key=lambda item: item["group_value"])


def _feature_coverage_rows(
    preprocessed_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    dimension: str,
    image_id_column: str,
    subject_id_column: str,
    feature_columns: list[str],
    thresholds: dict[str, Any],
    top_k_incomplete_features: int,
) -> list[dict[str, Any]]:
    merged = preprocessed_frame[[image_id_column, subject_id_column, dimension]].merge(
        feature_frame[[image_id_column] + feature_columns],
        on=image_id_column,
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for subgroup_value, subset in merged.groupby(dimension, dropna=False):
        total_rows = int(len(subset))
        feature_present_mask = subset[feature_columns].notna().any(axis=1) if feature_columns else pd.Series(dtype=bool)
        feature_row_count = int(feature_present_mask.sum()) if feature_columns else 0
        unique_subject_count = _nonempty_unique_count(subset[subject_id_column]) if subject_id_column in subset.columns else None
        completeness_rates: dict[str, float] = {}
        for column in feature_columns:
            available = subset[column].notna().sum()
            completeness_rates[column] = float(available / total_rows) if total_rows else 0.0
        mean_completeness = float(sum(completeness_rates.values()) / len(completeness_rates)) if completeness_rates else None
        min_completeness = float(min(completeness_rates.values())) if completeness_rates else None
        lowest_features = sorted(completeness_rates.items(), key=lambda item: item[1])[:top_k_incomplete_features]
        feature_coverage_rate = _safe_rate(feature_row_count, total_rows)
        row = {
            "coverage_type": FEATURE_COVERAGE,
            "group_dimension": dimension,
            "group_value": "unknown" if pd.isna(subgroup_value) or subgroup_value is None else str(subgroup_value),
            "row_count": total_rows,
            "unique_subject_count": unique_subject_count,
            "feature_row_count": feature_row_count,
            "feature_coverage_rate": feature_coverage_rate,
            "mean_feature_completeness_rate": mean_completeness,
            "min_feature_completeness_rate": min_completeness,
            "lowest_feature_completeness": [
                {"feature_name": feature_name, "completeness_rate": completeness_rate}
                for feature_name, completeness_rate in lowest_features
            ],
        }
        row["warnings"] = _warning_messages(
            coverage_type=FEATURE_COVERAGE,
            row_count=total_rows,
            unique_subject_count=unique_subject_count,
            thresholds=thresholds,
            feature_coverage_rate=feature_coverage_rate,
            mean_feature_completeness_rate=mean_completeness,
        )
        rows.append(row)
    return sorted(rows, key=lambda item: item["group_value"])


def _reference_rows(
    *,
    reference_payload: dict[str, Any],
    statistics_frame: pd.DataFrame,
    thresholds: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    by_dimension: dict[str, list[dict[str, Any]]] = {
        "sex_group_normalized": [],
        "age_group_normalized": [],
        "ethnicity_group_normalized": [],
    }

    statistics_lookup = {}
    if not statistics_frame.empty:
        for (group_key, feature_name), subset in statistics_frame.groupby(["group_key", "feature_name"], dropna=False):
            statistics_lookup[(str(group_key), str(feature_name))] = subset.iloc[0]

    for combination_name, bucket in dict(reference_payload.get("reference_groups", {})).items():
        for group_key, payload in dict(bucket).items():
            group_values = dict(payload.get("group_values", {}))
            sample_count = int(payload.get("sample_count", 0))
            complete_case_count = int(payload.get("complete_case_count", 0))
            feature_columns = list(payload.get("feature_columns", []))
            stats_subset = statistics_frame.loc[statistics_frame["group_key"] == group_key].copy() if not statistics_frame.empty else pd.DataFrame()
            unique_subject_count = None
            row = {
                "coverage_type": REFERENCE_COVERAGE,
                "combination_name": combination_name,
                "group_key": group_key,
                "group_values": group_values,
                "row_count": sample_count,
                "unique_subject_count": unique_subject_count,
                "sample_count": sample_count,
                "complete_case_count": complete_case_count,
                "feature_count": len(feature_columns),
                "statistics_feature_rows": int(len(stats_subset)),
                "feature_names": feature_columns,
            }
            row["warnings"] = _warning_messages(
                coverage_type=REFERENCE_COVERAGE,
                row_count=sample_count,
                unique_subject_count=unique_subject_count,
                thresholds=thresholds,
                sample_count=sample_count,
                complete_case_count=complete_case_count,
            )
            rows.append(row)

            if len(group_values) == 1:
                dimension_name = next(iter(group_values.keys()))
                if dimension_name in by_dimension:
                    by_dimension[dimension_name].append(
                        {
                            "group_dimension": dimension_name,
                            "group_value": next(iter(group_values.values())),
                            "group_key": group_key,
                            "sample_count": sample_count,
                            "complete_case_count": complete_case_count,
                            "feature_count": len(feature_columns),
                            "warnings": row["warnings"],
                        }
                    )
    for key in by_dimension:
        by_dimension[key] = sorted(by_dimension[key], key=lambda item: item["group_value"])
    rows = sorted(rows, key=lambda item: (item["combination_name"], item["group_key"]))
    return rows, by_dimension


def _flat_rows_from_dimension_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload["warnings"] = json.dumps(payload.get("warnings", []), ensure_ascii=True, sort_keys=True)
        if "lowest_feature_completeness" in payload:
            payload["lowest_feature_completeness"] = json.dumps(
                payload["lowest_feature_completeness"], ensure_ascii=True, sort_keys=True
            )
        flattened.append(payload)
    return flattened


def _flat_rows_from_reference_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        payload = {
            "coverage_type": REFERENCE_COVERAGE,
            "group_dimension": row["combination_name"],
            "group_value": row["group_key"],
            "row_count": row["sample_count"],
            "unique_subject_count": row.get("unique_subject_count"),
            "sample_count": row["sample_count"],
            "complete_case_count": row["complete_case_count"],
            "feature_count": row["feature_count"],
            "warnings": json.dumps(row.get("warnings", []), ensure_ascii=True, sort_keys=True),
            "group_values": json.dumps(row.get("group_values", {}), ensure_ascii=True, sort_keys=True),
        }
        flattened.append(payload)
    return flattened


def run_subgroup_validation(project_root: Path | None = None) -> SubgroupValidationArtifacts:
    root = (project_root or Path.cwd()).resolve()
    configs = load_named_configs(root / "configs", file_names=CONFIG_FILES)
    validation_config = configs["subgroup_validation"]["subgroup_validation"]
    train_config = configs["train_baseline"].get("train_baseline", {})

    image_id_column = str(validation_config["image_id_column"])
    subject_id_column = str(validation_config["subject_id_column"])
    dataset_name_column = str(validation_config["dataset_name_column"])
    normalized_labels_column = str(validation_config["normalized_labels_column"])
    quality_pass_column = str(validation_config["quality_pass_column"])
    processing_status_column = str(validation_config["processing_status_column"])
    quality_gate_status_column = str(validation_config["quality_gate_status_column"])
    quality_fail_reason_column = str(validation_config["quality_fail_reason_column"])

    preprocessed_manifest_path = (root / validation_config["preprocessed_manifest_path"]).resolve()
    geometry_features_path = (root / validation_config["geometry_features_path"]).resolve()
    reference_statistics_path = (root / validation_config["reference_statistics_path"]).resolve()
    reference_model_path = (root / validation_config["reference_model_path"]).resolve()
    validation_json_path = (root / validation_config["output_validation_json_path"]).resolve()
    reference_coverage_json_path = (root / validation_config["output_reference_coverage_json_path"]).resolve()
    flat_table_path = (root / validation_config["output_flat_table_path"]).resolve()

    ensure_parent_directory(validation_json_path)
    ensure_parent_directory(reference_coverage_json_path)
    ensure_parent_directory(flat_table_path)

    thresholds = dict(validation_config.get("weak_support_thresholds", {}))
    top_k_incomplete_features = int(validation_config.get("top_k_incomplete_features", 10))

    group_aliases = dict(validation_config.get("group_label_aliases") or train_config.get("group_label_aliases") or {})

    preprocessed_frame = pd.DataFrame(read_parquet(preprocessed_manifest_path))
    feature_frame = pd.DataFrame(read_parquet(geometry_features_path))
    statistics_frame = pd.DataFrame(read_parquet(reference_statistics_path)) if reference_statistics_path.exists() else pd.DataFrame()
    reference_payload = joblib.load(reference_model_path) if reference_model_path.exists() else {"reference_groups": {}, "groupings": []}

    if preprocessed_frame.empty:
        preprocessed_frame = pd.DataFrame(columns=[
            image_id_column,
            subject_id_column,
            dataset_name_column,
            normalized_labels_column,
            quality_pass_column,
            processing_status_column,
            quality_gate_status_column,
            quality_fail_reason_column,
        ])

    normalized_groups = _normalized_group_columns(
        preprocessed_frame,
        image_id_column=image_id_column,
        normalized_labels_column=normalized_labels_column,
        group_aliases=group_aliases,
    )
    merged_preprocessed = preprocessed_frame.merge(normalized_groups, on=image_id_column, how="left")
    merged_preprocessed["quality_bucket"] = merged_preprocessed.apply(
        lambda row: _quality_bucket(
            row,
            quality_pass_column=quality_pass_column,
            quality_fail_reason_column=quality_fail_reason_column,
            quality_gate_status_column=quality_gate_status_column,
        ),
        axis=1,
    )

    preprocessing_by_dimension: dict[str, list[dict[str, Any]]] = {}
    feature_by_dimension: dict[str, list[dict[str, Any]]] = {}
    flat_rows: list[dict[str, Any]] = []
    feature_columns = _feature_columns(feature_frame) if not feature_frame.empty else []

    for dimension in GROUP_DIMENSIONS:
        preprocessing_rows = _preprocessing_coverage_rows(
            merged_preprocessed,
            dimension=dimension,
            dataset_name_column=dataset_name_column,
            subject_id_column=subject_id_column,
            quality_pass_column=quality_pass_column,
            processing_status_column=processing_status_column,
            thresholds=thresholds,
        )
        feature_rows = _feature_coverage_rows(
            merged_preprocessed,
            feature_frame,
            dimension=dimension,
            image_id_column=image_id_column,
            subject_id_column=subject_id_column,
            feature_columns=feature_columns,
            thresholds=thresholds,
            top_k_incomplete_features=top_k_incomplete_features,
        )
        preprocessing_by_dimension[dimension] = preprocessing_rows
        feature_by_dimension[dimension] = feature_rows
        flat_rows.extend(_flat_rows_from_dimension_rows(preprocessing_rows))
        flat_rows.extend(_flat_rows_from_dimension_rows(feature_rows))

    reference_rows, reference_by_dimension = _reference_rows(
        reference_payload=reference_payload,
        statistics_frame=statistics_frame,
        thresholds=thresholds,
    )
    flat_rows.extend(_flat_rows_from_reference_rows(reference_rows))

    if flat_rows:
        pd.DataFrame(flat_rows).to_parquet(flat_table_path, index=False)
    else:
        pd.DataFrame(columns=["coverage_type", "group_dimension", "group_value"]).to_parquet(flat_table_path, index=False)

    validation_summary = {
        "status": "completed",
        "analysis_type": "transparency_and_coverage_only",
        "note": "This report describes data/process coverage by subgroup. It is not a fairness marketing claim.",
        "weak_support_thresholds": thresholds,
        "quality_bucket_definition": "passed if quality_pass=true; otherwise quality_fail_reason if present; otherwise quality_gate_status; otherwise unknown.",
        "input_counts": {
            "preprocessed_manifest_rows": int(len(merged_preprocessed)),
            "geometry_feature_rows": int(len(feature_frame)),
            "reference_statistics_rows": int(len(statistics_frame)),
        },
        "preprocessing_coverage": preprocessing_by_dimension,
        "feature_coverage": feature_by_dimension,
        "reference_coverage_summary": {
            "reference_group_count": int(len(reference_rows)),
            "groupings": reference_payload.get("groupings", []),
            "by_dimension": reference_by_dimension,
        },
        "artifacts": {
            "validation_json_path": validation_json_path.relative_to(root).as_posix(),
            "reference_coverage_json_path": reference_coverage_json_path.relative_to(root).as_posix(),
            "flat_table_path": flat_table_path.relative_to(root).as_posix(),
        },
    }
    reference_summary = {
        "status": "completed",
        "analysis_type": "reference_coverage_only",
        "weak_support_thresholds": thresholds,
        "reference_group_count": int(len(reference_rows)),
        "groupings": reference_payload.get("groupings", []),
        "rows": reference_rows,
        "by_dimension": reference_by_dimension,
        "artifacts": {
            "reference_coverage_json_path": reference_coverage_json_path.relative_to(root).as_posix(),
            "flat_table_path": flat_table_path.relative_to(root).as_posix(),
        },
    }

    write_json(validation_summary, validation_json_path)
    write_json(reference_summary, reference_coverage_json_path)

    return SubgroupValidationArtifacts(
        validation_json_path=validation_json_path,
        reference_coverage_json_path=reference_coverage_json_path,
        flat_table_path=flat_table_path,
        validation_summary=validation_summary,
        reference_summary=reference_summary,
    )
