from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from src.face_aesthetics.data.io import write_json, write_parquet
from src.face_aesthetics.pipelines.inference import run_inference
from src.face_aesthetics.services.perturbations import generate_perturbation_artifacts
from src.face_aesthetics.utils.config import load_named_configs
from src.face_aesthetics.utils.filesystem import ensure_directory


FEATURE_FAMILIES = {
    "raw_measurements": "raw_measurements",
    "normalized_ratios": "normalized_ratios",
    "symmetry_metrics": "symmetry_metrics",
    "angle_metrics": "angle_metrics",
}

INDEX_FAMILY = "optional_composite_indices"


@dataclass
class RobustnessArtifacts:
    summary_path: Path
    feature_table_path: Path
    summary: dict[str, Any]


def _parse_group_override(value: str | None) -> dict[str, str] | None:
    if value is None or not value.strip():
        return None
    stripped = value.strip()
    if stripped.startswith("{"):
        payload = json.loads(stripped)
        return {str(key): str(raw).strip() for key, raw in payload.items() if str(raw).strip()}
    item_separator = "," if "," in stripped else "__"
    result: dict[str, str] = {}
    for item in stripped.split(item_separator):
        normalized_item = item.strip()
        if not normalized_item:
            continue
        if "=" not in normalized_item:
            raise ValueError(f"Invalid group override entry: {item}")
        key, raw_value = normalized_item.split("=", 1)
        if key.strip() and raw_value.strip():
            result[key.strip()] = raw_value.strip()
    return result or None


def _flatten_feature_map(report: dict[str, Any]) -> dict[str, tuple[str, float | None]]:
    flattened: dict[str, tuple[str, float | None]] = {}
    for section_name, family_name in FEATURE_FAMILIES.items():
        for feature_name, value in report.get(section_name, {}).items():
            flattened[feature_name] = (family_name, float(value) if value is not None else None)
    return flattened


def _flatten_index_map(report: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for index_name, value in report.get(INDEX_FAMILY, {}).items():
        result[index_name] = float(value) if value is not None else None
    return result


def _relative_delta(original_value: float | None, perturbed_value: float | None) -> float | None:
    if original_value is None or perturbed_value is None:
        return None
    if original_value == 0.0:
        return None
    return float(abs(perturbed_value - original_value) / abs(original_value))


def _tolerance_for_family(tolerances: dict[str, Any], family_name: str) -> float | None:
    raw = tolerances.get(family_name)
    if raw is None:
        return None
    return float(raw)


def _feature_delta_rows(
    *,
    image_id: str,
    perturbation_name: str,
    original_report: dict[str, Any],
    perturbed_report: dict[str, Any],
    tolerances: dict[str, Any],
) -> list[dict[str, Any]]:
    original_features = _flatten_feature_map(original_report)
    perturbed_features = _flatten_feature_map(perturbed_report)
    original_quality = dict(original_report.get("input_quality", {}))
    perturbed_quality = dict(perturbed_report.get("input_quality", {}))
    rows: list[dict[str, Any]] = []

    for feature_name, (family_name, original_value) in original_features.items():
        perturbed_value = perturbed_features.get(feature_name, (family_name, None))[1]
        abs_delta = None
        if original_value is not None and perturbed_value is not None:
            abs_delta = float(abs(perturbed_value - original_value))
        tolerance = _tolerance_for_family(tolerances, family_name)
        rows.append(
            {
                "image_id": image_id,
                "perturbation_name": perturbation_name,
                "feature_family": family_name,
                "feature_name": feature_name,
                "original_value": original_value,
                "perturbed_value": perturbed_value,
                "absolute_delta": abs_delta,
                "relative_delta": _relative_delta(original_value, perturbed_value),
                "tolerance": tolerance,
                "tolerance_pass": (abs_delta <= tolerance) if abs_delta is not None and tolerance is not None else None,
                "original_processing_status": original_quality.get("processing_status"),
                "perturbed_processing_status": perturbed_quality.get("processing_status"),
                "original_quality_pass": original_quality.get("quality_pass"),
                "perturbed_quality_pass": perturbed_quality.get("quality_pass"),
            }
        )

    return rows


def _index_delta_rows(
    *,
    image_id: str,
    perturbation_name: str,
    original_report: dict[str, Any],
    perturbed_report: dict[str, Any],
) -> list[dict[str, Any]]:
    original_indices = _flatten_index_map(original_report)
    perturbed_indices = _flatten_index_map(perturbed_report)
    rows: list[dict[str, Any]] = []
    for index_name, original_value in original_indices.items():
        perturbed_value = perturbed_indices.get(index_name)
        abs_delta = None
        if original_value is not None and perturbed_value is not None:
            abs_delta = float(abs(perturbed_value - original_value))
        rows.append(
            {
                "image_id": image_id,
                "perturbation_name": perturbation_name,
                "index_name": index_name,
                "original_value": original_value,
                "perturbed_value": perturbed_value,
                "absolute_delta": abs_delta,
            }
        )
    return rows


def _aggregate_feature_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    tolerance_values: dict[str, list[bool]] = {}
    families: dict[str, str] = {}
    for row in rows:
        absolute_delta = row.get("absolute_delta")
        if absolute_delta is None:
            continue
        feature_name = str(row["feature_name"])
        grouped.setdefault(feature_name, []).append(float(absolute_delta))
        families.setdefault(feature_name, str(row["feature_family"]))
        tolerance_pass = row.get("tolerance_pass")
        if tolerance_pass is not None:
            tolerance_values.setdefault(feature_name, []).append(bool(tolerance_pass))

    aggregates: dict[str, dict[str, Any]] = {}
    for feature_name, deltas in grouped.items():
        aggregates[feature_name] = {
            "feature_family": families.get(feature_name),
            "mean_absolute_delta": float(sum(deltas) / len(deltas)),
            "median_absolute_delta": float(median(deltas)),
            "max_absolute_delta": float(max(deltas)),
            "tolerance_pass_rate": (
                float(sum(1 for item in tolerance_values.get(feature_name, []) if item) / len(tolerance_values[feature_name]))
                if tolerance_values.get(feature_name)
                else None
            ),
        }
    return aggregates


def _aggregate_index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        absolute_delta = row.get("absolute_delta")
        if absolute_delta is None:
            continue
        grouped.setdefault(str(row["index_name"]), []).append(float(absolute_delta))

    return {
        index_name: {
            "mean_absolute_delta": float(sum(deltas) / len(deltas)),
            "median_absolute_delta": float(median(deltas)),
            "max_absolute_delta": float(max(deltas)),
        }
        for index_name, deltas in grouped.items()
    }


def _status_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    input_quality = dict(report.get("input_quality", {}))
    return {
        "processing_status": input_quality.get("processing_status"),
        "quality_pass": input_quality.get("quality_pass"),
        "quality_fail_reason": input_quality.get("quality_fail_reason"),
    }


def run_robustness_evaluation(
    *,
    image_paths: list[Path],
    group_override: str | None = None,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> RobustnessArtifacts:
    root = (project_root or Path.cwd()).resolve()
    configs = load_named_configs(root / "configs")
    robustness_config = configs["robustness"]["robustness"]

    resolved_output_dir = (output_dir.resolve() if output_dir is not None else (root / robustness_config["work_dir"]).resolve())
    ensure_directory(resolved_output_dir)

    summary_path = (root / robustness_config["output_summary_path"]).resolve()
    feature_table_path = (root / robustness_config["output_feature_table_path"]).resolve()

    preprocess_overrides = {
        "preferred_execution_provider": robustness_config["preferred_execution_provider"],
        "fallback_execution_provider": robustness_config["fallback_execution_provider"],
        "device_id": int(robustness_config["device_id"]),
        "strict_gpu_mode": bool(robustness_config["strict_gpu_mode"]),
    }
    tolerances = dict(robustness_config.get("tolerances", {}))

    feature_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    image_summaries: list[dict[str, Any]] = []
    failed_image_count = 0
    successful_baseline_image_count = 0
    successful_perturbation_runs = 0
    failed_perturbation_runs = 0
    perturbation_names = [
        "crop_shift",
        "resize",
        "jpeg_compression",
        "mild_blur",
        "brightness_change",
        "small_rotation",
    ]

    for image_path in image_paths:
        resolved_image_path = image_path.resolve()
        original_output_dir = resolved_output_dir / resolved_image_path.stem / "original"
        try:
            original_artifacts = run_inference(
                image_path=resolved_image_path,
                output_dir=original_output_dir,
                group_override=group_override,
                project_root=root,
                preprocess_overrides=preprocess_overrides,
            )
        except Exception as exc:
            failed_image_count += 1
            image_summaries.append(
                {
                    "image_path": str(resolved_image_path),
                    "status": "failed_original_inference",
                    "error": str(exc),
                    "perturbation_count": 0,
                    "feature_delta_row_count": 0,
                    "index_delta_row_count": 0,
                    "successful_perturbation_runs": 0,
                    "failed_perturbation_runs": 0,
                }
            )
            continue
        successful_baseline_image_count += 1
        perturbation_artifacts = generate_perturbation_artifacts(
            image_path=resolved_image_path,
            output_dir=resolved_output_dir / resolved_image_path.stem / "perturbations",
            perturbation_config=dict(robustness_config["perturbations"]),
        )

        image_feature_rows: list[dict[str, Any]] = []
        image_index_rows: list[dict[str, Any]] = []
        image_perturbation_summaries: list[dict[str, Any]] = []
        image_successful_perturbations = 0
        image_failed_perturbations = 0
        for perturbation in perturbation_artifacts:
            perturbation_name = str(perturbation["perturbation_name"])
            try:
                perturbed_artifacts = run_inference(
                    image_path=Path(perturbation["image_path"]),
                    output_dir=resolved_output_dir / resolved_image_path.stem / perturbation_name,
                    group_override=group_override,
                    project_root=root,
                    preprocess_overrides=preprocess_overrides,
                )
            except Exception as exc:
                failed_perturbation_runs += 1
                image_failed_perturbations += 1
                image_perturbation_summaries.append(
                    {
                        "perturbation_name": perturbation_name,
                        "status": "failed_perturbed_inference",
                        "error": str(exc),
                    }
                )
                continue
            successful_perturbation_runs += 1
            image_successful_perturbations += 1

            image_feature_rows.extend(
                _feature_delta_rows(
                    image_id=str(original_artifacts.report["input"]["image_path"]),
                    perturbation_name=perturbation_name,
                    original_report=original_artifacts.report,
                    perturbed_report=perturbed_artifacts.report,
                    tolerances=tolerances,
                )
            )
            image_index_rows.extend(
                _index_delta_rows(
                    image_id=str(original_artifacts.report["input"]["image_path"]),
                    perturbation_name=perturbation_name,
                    original_report=original_artifacts.report,
                    perturbed_report=perturbed_artifacts.report,
                )
            )
            image_perturbation_summaries.append(
                {
                    "perturbation_name": perturbation_name,
                    "status": "completed",
                    "original": _status_snapshot(original_artifacts.report),
                    "perturbed": _status_snapshot(perturbed_artifacts.report),
                }
            )

        feature_rows.extend(image_feature_rows)
        index_rows.extend(image_index_rows)
        image_summaries.append(
            {
                "image_path": str(resolved_image_path),
                "status": "completed",
                "original": _status_snapshot(original_artifacts.report),
                "perturbation_count": len(perturbation_artifacts),
                "feature_delta_row_count": len(image_feature_rows),
                "index_delta_row_count": len(image_index_rows),
                "successful_perturbation_runs": image_successful_perturbations,
                "failed_perturbation_runs": image_failed_perturbations,
                "perturbations": image_perturbation_summaries,
            }
        )

    feature_columns = [
        "image_id",
        "perturbation_name",
        "feature_family",
        "feature_name",
        "original_value",
        "perturbed_value",
        "absolute_delta",
        "relative_delta",
        "tolerance",
        "tolerance_pass",
        "original_processing_status",
        "perturbed_processing_status",
        "original_quality_pass",
        "perturbed_quality_pass",
    ]
    write_parquet(feature_rows, feature_table_path, feature_columns)

    feature_aggregates = _aggregate_feature_rows(feature_rows)
    index_aggregates = _aggregate_index_rows(index_rows)
    most_unstable = sorted(
        (
            {"feature_name": feature_name, **stats}
            for feature_name, stats in feature_aggregates.items()
        ),
        key=lambda item: item["max_absolute_delta"],
        reverse=True,
    )[: int(robustness_config["top_k_unstable_features"])]

    summary = {
        "status": "completed",
        "image_count": len(image_paths),
        "successful_baseline_image_count": successful_baseline_image_count,
        "failed_image_count": failed_image_count,
        "successful_perturbation_runs": successful_perturbation_runs,
        "failed_perturbation_runs": failed_perturbation_runs,
        "perturbation_names": perturbation_names,
        "feature_delta_row_count": len(feature_rows),
        "index_delta_row_count": len(index_rows),
        "feature_aggregates": feature_aggregates,
        "index_aggregates": index_aggregates,
        "most_unstable_features": most_unstable,
        "tolerances": tolerances,
        "image_summaries": image_summaries,
        "artifacts": {
            "summary_path": summary_path.relative_to(root).as_posix(),
            "feature_table_path": feature_table_path.relative_to(root).as_posix(),
            "work_dir": resolved_output_dir.relative_to(root).as_posix(),
        },
    }
    write_json(summary, summary_path)
    return RobustnessArtifacts(
        summary_path=summary_path,
        feature_table_path=feature_table_path,
        summary=summary,
    )
