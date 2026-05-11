from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.face_aesthetics.features.extraction import geometry_feature_columns
from src.face_aesthetics.utils.config import load_named_configs
from src.face_aesthetics.utils.filesystem import ensure_directory


REFERENCE_FEATURE_PREFIXES = (
    "ref_z_",
    "ref_dir_",
    "ref_ratio_",
    "ref_absdev_",
    "ref_percentile_",
)

_REFERENCE_DEFAULT_FILES = ("base.yaml", "data.yaml", "reference.yaml", "scoring.yaml")
_REFERENCE_FILTER_KEYS = (
    "sex_group_normalized",
    "age_group_normalized",
    "ethnicity_group_normalized",
)
_REFERENCE_FILTER_ALIASES = {
    "gender": "sex_group_normalized",
    "sex": "sex_group_normalized",
    "gender_presentation": "sex_group_normalized",
    "age_group": "age_group_normalized",
    "age": "age_group_normalized",
    "race": "ethnicity_group_normalized",
    "ethnicity": "ethnicity_group_normalized",
    "ethnicity_group": "ethnicity_group_normalized",
}
_EPS = 1e-9


@dataclass(frozen=True)
class ReferenceStats:
    available: bool
    source: str
    stats_path: Path
    requested_filters: dict[str, str]
    filters: dict[str, str]
    fallback_chain: list[dict[str, Any]]
    sample_count: int
    feature_count: int
    mean: dict[str, float]
    std: dict[str, float]
    median: dict[str, float]
    iqr: dict[str, float]
    p05: dict[str, float]
    p10: dict[str, float]
    p25: dict[str, float]
    p50: dict[str, float]
    p75: dict[str, float]
    p90: dict[str, float]
    p95: dict[str, float]
    coverage: dict[str, Any]


def _load_reference_config(project_root: Path) -> dict[str, Any]:
    configs = load_named_configs(project_root / "configs", file_names=_REFERENCE_DEFAULT_FILES)
    reference_config = dict(configs.get("reference", {}).get("reference", {}))
    if not reference_config:
        reference_config = {
            "reference_source": "fairface_utkface",
            "reference_datasets": ["fairface", "utkface"],
            "reference_features_path": "data/reference/reference_features.parquet",
            "reference_manifest_path": "data/reference/reference_manifest.parquet",
            "reference_dataset_summary_path": "reports/metrics/reference_dataset_summary.json",
            "reference_group_statistics_path": "models/reference/reference_group_statistics.parquet",
            "reference_group_coverage_path": "reports/metrics/reference_group_coverage.json",
            "reference_feature_statistics_summary_path": "reports/metrics/reference_feature_statistics_summary.json",
            "reference_stats_report_dir": "reports/metrics",
            "use_fairface_reference": True,
            "min_subset_size": 25,
            "feature_types": ["z", "dir", "ratio", "absdev", "percentile"],
            "filters": {},
        }
    return reference_config


def _safe_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def _normalize_filter_map(filters: dict[str, Any] | None) -> dict[str, str]:
    if not filters:
        return {}
    normalized: dict[str, str] = {}
    for key, value in filters.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        canonical = _REFERENCE_FILTER_ALIASES.get(str(key).strip().lower(), str(key).strip())
        if canonical in _REFERENCE_FILTER_KEYS:
            normalized[canonical] = text
    return normalized


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _normalize_reference_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    sex_col = _find_column(result, ["sex_group_normalized", "gender", "gender_presentation", "sex"])
    age_col = _find_column(result, ["age_group_normalized", "age_group", "age"])
    ethnicity_col = _find_column(result, ["ethnicity_group_normalized", "race", "ethnicity", "ethnicity_group"])

    if "sex_group_normalized" not in result.columns:
        result["sex_group_normalized"] = result[sex_col] if sex_col else None
    if "age_group_normalized" not in result.columns:
        result["age_group_normalized"] = result[age_col] if age_col else None
    if "ethnicity_group_normalized" not in result.columns:
        result["ethnicity_group_normalized"] = result[ethnicity_col] if ethnicity_col else None

    for column in _REFERENCE_FILTER_KEYS:
        if column in result.columns:
            result[column] = result[column].apply(lambda value: None if pd.isna(value) else str(value).strip() or None)
        else:
            result[column] = None
    return result


def _reference_dataset_candidates(project_root: Path, reference_config: dict[str, Any]) -> list[Path]:
    candidates = [
        (project_root / str(reference_config.get("reference_features_path", "data/reference/reference_features.parquet"))).resolve(),
        (project_root / "data/reference/fairface_features.parquet").resolve(),
        (project_root / "data/features/geometry_features.parquet").resolve(),
    ]
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _bootstrap_reference_features(project_root: Path, reference_config: dict[str, Any]) -> pd.DataFrame:
    target_features_path = (project_root / str(reference_config.get("reference_features_path", "data/reference/reference_features.parquet"))).resolve()
    target_manifest_path = (project_root / str(reference_config.get("reference_manifest_path", "data/reference/reference_manifest.parquet"))).resolve()
    ensure_directory(target_features_path.parent)
    ensure_directory(target_manifest_path.parent)

    datasets = {str(value).strip().lower() for value in reference_config.get("reference_datasets", ["fairface", "utkface"])}
    frame: pd.DataFrame | None = None
    selected_source = None
    for candidate in _reference_dataset_candidates(project_root, reference_config):
        if not candidate.exists():
            continue
        loaded = pd.read_parquet(candidate)
        if "dataset_name" in loaded.columns:
            subset = loaded.loc[loaded["dataset_name"].astype(str).str.lower().isin(datasets)].copy()
            if subset.empty:
                continue
        else:
            subset = loaded.copy()
            subset["dataset_name"] = "unknown_reference"
        frame = subset
        selected_source = candidate
        break

    if frame is None or frame.empty:
        empty = pd.DataFrame(columns=["dataset_name", "sample_id", "image_path", "split", *_REFERENCE_FILTER_KEYS, *geometry_feature_columns()])
        empty.to_parquet(target_features_path, index=False)
        empty.loc[:, ["dataset_name", "sample_id", "image_path", "split", *_REFERENCE_FILTER_KEYS]].to_parquet(target_manifest_path, index=False)
        _safe_json(
            (project_root / str(reference_config.get("reference_dataset_summary_path", "reports/metrics/reference_dataset_summary.json"))).resolve(),
            {
                "status": "reference_data_unavailable",
                "reference_features_path": target_features_path.relative_to(project_root).as_posix(),
                "reference_manifest_path": target_manifest_path.relative_to(project_root).as_posix(),
                "datasets": sorted(datasets),
                "source_path": None,
                "rows": 0,
            },
        )
        return empty

    frame = _normalize_reference_metadata(frame)
    available_geometry = [column for column in geometry_feature_columns() if column in frame.columns]
    manifest_columns = [column for column in ["dataset_name", "sample_id", "image_path", "split", *_REFERENCE_FILTER_KEYS] if column in frame.columns]
    selected_columns = list(dict.fromkeys(manifest_columns + available_geometry))
    prepared = frame.loc[:, selected_columns].copy()
    prepared.to_parquet(target_features_path, index=False)
    prepared.loc[:, manifest_columns].to_parquet(target_manifest_path, index=False)

    dataset_counts = (
        prepared["dataset_name"].astype(str).value_counts().to_dict()
        if "dataset_name" in prepared.columns
        else {}
    )
    _safe_json(
        (project_root / str(reference_config.get("reference_dataset_summary_path", "reports/metrics/reference_dataset_summary.json"))).resolve(),
        {
            "status": "completed",
            "source_path": selected_source.relative_to(project_root).as_posix() if selected_source and selected_source.is_relative_to(project_root) else str(selected_source),
            "reference_features_path": target_features_path.relative_to(project_root).as_posix(),
            "reference_manifest_path": target_manifest_path.relative_to(project_root).as_posix(),
            "rows": int(len(prepared)),
            "dataset_counts": {str(key): int(value) for key, value in dataset_counts.items()},
            "feature_count": int(len(available_geometry)),
            "available_group_columns": [column for column in _REFERENCE_FILTER_KEYS if column in prepared.columns],
        },
    )
    return prepared


def _load_reference_features(project_root: Path, reference_config: dict[str, Any]) -> pd.DataFrame:
    reference_features_path = (project_root / str(reference_config.get("reference_features_path", "data/reference/reference_features.parquet"))).resolve()
    if reference_features_path.exists():
        frame = pd.read_parquet(reference_features_path)
        return _normalize_reference_metadata(frame)
    return _bootstrap_reference_features(project_root, reference_config)


def _fallback_chain(requested_filters: dict[str, str]) -> list[dict[str, str]]:
    if not requested_filters:
        return [{}]
    chain: list[dict[str, str]] = []
    ordered = {key: requested_filters[key] for key in _REFERENCE_FILTER_KEYS if key in requested_filters}
    if not ordered:
        return [{}]
    if len(ordered) == 3:
        sex = {"sex_group_normalized": ordered["sex_group_normalized"]}
        age = {"age_group_normalized": ordered["age_group_normalized"]}
        ethnicity = {"ethnicity_group_normalized": ordered["ethnicity_group_normalized"]}
        chain.extend(
            [
                dict(ordered),
                {**sex, **age},
                {**age, **ethnicity},
                dict(sex),
                dict(age),
                {},
            ]
        )
    else:
        chain.append(dict(ordered))
        for key in list(ordered.keys()):
            reduced = {column: value for column, value in ordered.items() if column != key}
            if reduced:
                chain.append(reduced)
        chain.append({})
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in chain:
        marker = json.dumps(item, ensure_ascii=True, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(item)
    return unique


def _apply_filters(frame: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    subset = frame
    for column, value in filters.items():
        if column not in subset.columns:
            return subset.iloc[0:0].copy()
        subset = subset.loc[subset[column].astype(str) == str(value)]
    return subset.copy()


def _stats_digest(source: str, filters: dict[str, str], features: list[str]) -> str:
    payload = {"source": source, "filters": filters, "feature_count": len(features)}
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def _feature_statistics(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float]]:
    numeric = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    quantiles = numeric.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], interpolation="linear")
    stats: dict[str, dict[str, float]] = {}
    for column in feature_columns:
        series = numeric[column]
        valid = series.dropna()
        if valid.empty:
            stats[column] = {
                "count": 0.0,
                "complete_count": 0.0,
                "mean": float("nan"),
                "std": float("nan"),
                "median": float("nan"),
                "iqr": float("nan"),
                "p05": float("nan"),
                "p10": float("nan"),
                "p25": float("nan"),
                "p50": float("nan"),
                "p75": float("nan"),
                "p90": float("nan"),
                "p95": float("nan"),
            }
            continue
        q = quantiles[column]
        p25 = float(q.loc[0.25])
        p75 = float(q.loc[0.75])
        stats[column] = {
            "count": float(len(series)),
            "complete_count": float(len(valid)),
            "mean": float(valid.mean()),
            "std": float(valid.std(ddof=0)),
            "median": float(valid.median()),
            "iqr": float(p75 - p25),
            "p05": float(q.loc[0.05]),
            "p10": float(q.loc[0.10]),
            "p25": p25,
            "p50": float(q.loc[0.50]),
            "p75": p75,
            "p90": float(q.loc[0.90]),
            "p95": float(q.loc[0.95]),
        }
    return stats


def _group_statistics_rows(reference_frame: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
    group_sets = [
        [],
        ["sex_group_normalized"],
        ["age_group_normalized"],
        ["ethnicity_group_normalized"],
        ["sex_group_normalized", "age_group_normalized"],
        ["sex_group_normalized", "ethnicity_group_normalized"],
        ["age_group_normalized", "ethnicity_group_normalized"],
        ["sex_group_normalized", "age_group_normalized", "ethnicity_group_normalized"],
    ]
    rows: list[dict[str, Any]] = []
    for group_columns in group_sets:
        if group_columns:
            grouped = reference_frame.groupby(group_columns, dropna=False)
            iterator = grouped
        else:
            iterator = [(("global",), reference_frame)]
        for group_key, subset in iterator:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = {}
            if group_columns:
                for idx, column in enumerate(group_columns):
                    raw = group_key[idx]
                    group_values[column] = None if pd.isna(raw) else str(raw)
            stats = _feature_statistics(subset, feature_columns)
            for feature_name, payload in stats.items():
                rows.append(
                    {
                        "grouping": "global" if not group_columns else "__".join(group_columns),
                        "group_key": "global"
                        if not group_columns
                        else "__".join(f"{column}={group_values.get(column)}" for column in group_columns),
                        "group_values": json.dumps(group_values, ensure_ascii=True, sort_keys=True),
                        "feature_name": feature_name,
                        **payload,
                    }
                )
    return rows


def _write_group_reports(
    project_root: Path,
    reference_config: dict[str, Any],
    reference_frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    group_stats_path = (project_root / str(reference_config.get("reference_group_statistics_path", "models/reference/reference_group_statistics.parquet"))).resolve()
    coverage_path = (project_root / str(reference_config.get("reference_group_coverage_path", "reports/metrics/reference_group_coverage.json"))).resolve()
    summary_path = (project_root / str(reference_config.get("reference_feature_statistics_summary_path", "reports/metrics/reference_feature_statistics_summary.json"))).resolve()

    rows = _group_statistics_rows(reference_frame, feature_columns) if not reference_frame.empty else []
    ensure_directory(group_stats_path.parent)
    if rows:
        pd.DataFrame(rows).to_parquet(group_stats_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "grouping",
                "group_key",
                "group_values",
                "feature_name",
                "count",
                "complete_count",
                "mean",
                "std",
                "median",
                "iqr",
                "p05",
                "p10",
                "p25",
                "p50",
                "p75",
                "p90",
                "p95",
            ]
        ).to_parquet(group_stats_path, index=False)

    if rows:
        frame = pd.DataFrame(rows)
        coverage_rows = (
            frame.groupby(["grouping", "group_key"], as_index=False)
            .agg(
                sample_count=("count", "max"),
                complete_case_count=("complete_count", "max"),
                feature_count=("feature_name", "nunique"),
            )
            .sort_values(["grouping", "group_key"])
        )
        coverage = [
            {
                "grouping": str(row["grouping"]),
                "group_key": str(row["group_key"]),
                "sample_count": int(row["sample_count"]),
                "complete_case_count": int(row["complete_case_count"]),
                "feature_count": int(row["feature_count"]),
            }
            for _, row in coverage_rows.iterrows()
        ]
    else:
        coverage = []
    _safe_json(
        coverage_path,
        {
            "status": "completed" if rows else "empty_reference",
            "reference_group_count": len(coverage),
            "groups": coverage,
            "reference_group_statistics_path": group_stats_path.relative_to(project_root).as_posix(),
        },
    )

    _safe_json(
        summary_path,
        {
            "status": "completed" if rows else "empty_reference",
            "rows": int(len(reference_frame)),
            "feature_count": int(len(feature_columns)),
            "reference_group_statistics_path": group_stats_path.relative_to(project_root).as_posix(),
            "reference_group_coverage_path": coverage_path.relative_to(project_root).as_posix(),
        },
    )
    return {
        "reference_group_statistics_path": group_stats_path,
        "reference_group_coverage_path": coverage_path,
        "reference_feature_statistics_summary_path": summary_path,
    }


def reference_relative_feature_names(
    feature_columns: list[str],
    include_types: list[str] | None = None,
) -> list[str]:
    include = [str(value).strip().lower() for value in (include_types or ["z", "dir", "ratio"])]
    prefixes = []
    if "z" in include:
        prefixes.append("ref_z_")
    if "dir" in include:
        prefixes.append("ref_dir_")
    if "ratio" in include:
        prefixes.append("ref_ratio_")
    if "absdev" in include:
        prefixes.append("ref_absdev_")
    if "percentile" in include:
        prefixes.append("ref_percentile_")
    return [f"{prefix}{feature}" for feature in feature_columns for prefix in prefixes]


def compute_reference_stats(
    *,
    project_root: Path,
    filters: dict[str, Any] | None,
    feature_columns: list[str] | None = None,
) -> ReferenceStats:
    root = project_root.resolve()
    reference_config = _load_reference_config(root)
    requested_filters = _normalize_filter_map(filters or reference_config.get("filters", {}))
    min_subset_size = int(reference_config.get("min_subset_size", 25))
    selected_features = [column for column in (feature_columns or geometry_feature_columns())]
    reference_frame = _load_reference_features(root, reference_config)
    selected_features = [column for column in selected_features if column in reference_frame.columns]
    _write_group_reports(root, reference_config, reference_frame, selected_features)

    fallback_chain = _fallback_chain(requested_filters)
    resolved_filters: dict[str, str] = {}
    resolved_subset = reference_frame
    fallback_used = False
    coverage_rows: list[dict[str, Any]] = []
    for candidate in fallback_chain:
        subset = _apply_filters(reference_frame, candidate)
        coverage_rows.append(
            {
                "requested_group": requested_filters,
                "resolved_group": candidate,
                "fallback_used": candidate != requested_filters,
                "reference_sample_count": int(len(subset)),
            }
        )
        if len(subset) >= min_subset_size or not candidate:
            resolved_subset = subset
            resolved_filters = candidate
            fallback_used = candidate != requested_filters
            break

    available = bool(not resolved_subset.empty and selected_features)
    stats_rows = _feature_statistics(resolved_subset, selected_features) if available else {}
    mean = {feature: float(payload["mean"]) for feature, payload in stats_rows.items() if np.isfinite(payload["mean"])}
    std = {feature: float(payload["std"]) for feature, payload in stats_rows.items() if np.isfinite(payload["std"])}
    median = {feature: float(payload["median"]) for feature, payload in stats_rows.items() if np.isfinite(payload["median"])}
    iqr = {feature: float(payload["iqr"]) for feature, payload in stats_rows.items() if np.isfinite(payload["iqr"])}
    p05 = {feature: float(payload["p05"]) for feature, payload in stats_rows.items() if np.isfinite(payload["p05"])}
    p10 = {feature: float(payload["p10"]) for feature, payload in stats_rows.items() if np.isfinite(payload["p10"])}
    p25 = {feature: float(payload["p25"]) for feature, payload in stats_rows.items() if np.isfinite(payload["p25"])}
    p50 = {feature: float(payload["p50"]) for feature, payload in stats_rows.items() if np.isfinite(payload["p50"])}
    p75 = {feature: float(payload["p75"]) for feature, payload in stats_rows.items() if np.isfinite(payload["p75"])}
    p90 = {feature: float(payload["p90"]) for feature, payload in stats_rows.items() if np.isfinite(payload["p90"])}
    p95 = {feature: float(payload["p95"]) for feature, payload in stats_rows.items() if np.isfinite(payload["p95"])}

    stats_report_dir = (root / str(reference_config.get("reference_stats_report_dir", "reports/metrics"))).resolve()
    ensure_directory(stats_report_dir)
    digest = _stats_digest(str(reference_config.get("reference_source", "fairface_utkface")), requested_filters, selected_features)
    stats_path = stats_report_dir / f"reference_stats_{digest}.json"
    report_payload = {
        "status": "completed" if available else "unavailable",
        "reference_source": reference_config.get("reference_source", "fairface_utkface"),
        "requested_filters": requested_filters,
        "resolved_filters": resolved_filters,
        "fallback_used": fallback_used,
        "fallback_chain": fallback_chain,
        "min_subset_size": min_subset_size,
        "reference_sample_count": int(len(resolved_subset)),
        "feature_count": int(len(selected_features)),
        "coverage_rows": coverage_rows,
        "reference_features_path": str(reference_config.get("reference_features_path", "data/reference/reference_features.parquet")),
        "datasets": (
            {
                str(key): int(value)
                for key, value in resolved_subset["dataset_name"].astype(str).value_counts().to_dict().items()
            }
            if not resolved_subset.empty and "dataset_name" in resolved_subset.columns
            else {}
        ),
        "statistics": stats_rows,
    }
    _safe_json(stats_path, report_payload)

    coverage = {
        "requested_group": requested_filters,
        "resolved_group": resolved_filters,
        "fallback_used": fallback_used,
        "reference_sample_count": int(len(resolved_subset)),
        "datasets": report_payload.get("datasets", {}),
    }

    return ReferenceStats(
        available=available,
        source=str(reference_config.get("reference_source", "fairface_utkface")),
        stats_path=stats_path,
        requested_filters=requested_filters,
        filters=resolved_filters,
        fallback_chain=coverage_rows,
        sample_count=int(len(resolved_subset)),
        feature_count=int(len(selected_features)),
        mean=mean,
        std=std,
        median=median,
        iqr=iqr,
        p05=p05,
        p10=p10,
        p25=p25,
        p50=p50,
        p75=p75,
        p90=p90,
        p95=p95,
        coverage=coverage,
    )


def append_reference_relative_features(
    frame: pd.DataFrame,
    *,
    stats: ReferenceStats,
    feature_columns: list[str],
    include_types: list[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    include = {str(value).strip().lower() for value in (include_types or ["z", "dir", "ratio"])}
    generated: dict[str, pd.Series] = {}
    for feature_name in feature_columns:
        if feature_name not in result.columns:
            continue
        values = pd.to_numeric(result[feature_name], errors="coerce")
        mean_value = stats.mean.get(feature_name)
        std_value = stats.std.get(feature_name)

        if "z" in include:
            if mean_value is None or std_value is None or not np.isfinite(mean_value) or not np.isfinite(std_value) or abs(std_value) <= _EPS:
                generated[f"ref_z_{feature_name}"] = pd.Series(np.nan, index=result.index, dtype=float)
            else:
                generated[f"ref_z_{feature_name}"] = (values - float(mean_value)) / float(std_value)

        if "dir" in include:
            if mean_value is None or not np.isfinite(mean_value):
                generated[f"ref_dir_{feature_name}"] = pd.Series(np.nan, index=result.index, dtype=float)
            else:
                generated[f"ref_dir_{feature_name}"] = values - float(mean_value)

        if "ratio" in include:
            if mean_value is None or not np.isfinite(mean_value) or abs(float(mean_value)) <= _EPS:
                generated[f"ref_ratio_{feature_name}"] = pd.Series(np.nan, index=result.index, dtype=float)
            else:
                generated[f"ref_ratio_{feature_name}"] = values / float(mean_value)

        if "absdev" in include:
            if mean_value is None or not np.isfinite(mean_value):
                generated[f"ref_absdev_{feature_name}"] = pd.Series(np.nan, index=result.index, dtype=float)
            else:
                generated[f"ref_absdev_{feature_name}"] = (values - float(mean_value)).abs()

        if "percentile" in include:
            if mean_value is None or std_value is None or not np.isfinite(mean_value) or not np.isfinite(std_value) or abs(std_value) <= _EPS:
                generated[f"ref_percentile_{feature_name}"] = pd.Series(np.nan, index=result.index, dtype=float)
            else:
                z = ((values - float(mean_value)) / float(std_value)).astype(float)
                generated[f"ref_percentile_{feature_name}"] = z.apply(
                    lambda value: float(0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))))
                    if pd.notna(value)
                    else np.nan
                )
    if generated:
        generated_frame = pd.DataFrame(generated, index=result.index)
        result = pd.concat([result, generated_frame], axis=1)
    return result
