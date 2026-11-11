from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from src.face_aesthetics.data.io import load_image, read_parquet, save_image, write_parquet
from src.face_aesthetics.features.extraction import (
    extract_angle_features,
    extract_normalized_measurements,
    extract_raw_measurements,
    extract_symmetry_features,
    feature_registry_rows,
    geometry_feature_columns,
    parse_json_field,
)
from src.face_aesthetics.landmarks.mediapipe_source import build_mediapipe_source_result
from src.face_aesthetics.scoring.common import (
    SCORING_GROUP_COLUMNS,
    SCORING_METADATA_COLUMNS,
    load_scoring_configs,
    save_json,
    scoring_feature_columns,
)
from src.face_aesthetics.scoring.reference import append_reference_relative_features, compute_reference_stats, reference_relative_feature_names
from src.face_aesthetics.utils.filesystem import ensure_directory, to_relative_path
import pandas as pd


PROCESSED_STATUSES = {"processed", "success"}


@dataclass
class ScoringFeatureArtifacts:
    features_path: Path
    summary_path: Path
    summary: dict[str, Any]


def _json_or_none(payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        stripped = payload.strip()
        return stripped or None
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return str(payload)


def _bbox_from_landmarks(landmarks: list[list[float]] | None, width: int, height: int) -> list[float] | None:
    if not landmarks:
        return None
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return None
    x1 = float(np.clip(np.nanmin(pts[:, 0]), 0.0, max(width - 1, 0)))
    y1 = float(np.clip(np.nanmin(pts[:, 1]), 0.0, max(height - 1, 0)))
    x2 = float(np.clip(np.nanmax(pts[:, 0]), 0.0, max(width - 1, 0)))
    y2 = float(np.clip(np.nanmax(pts[:, 1]), 0.0, max(height - 1, 0)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def extract_single_image_scoring_features(
    *,
    image_path: Path,
    image_id: str,
    target_score: float,
    target_group_label: str,
    split: str,
    source_score_folder: str,
    file_name: str,
    project_root: Path,
    output_analysis_path: Path,
    output_aligned_path: Path,
    reference_filters: dict[str, Any] | None = None,
    use_fairface_reference: bool | None = None,
) -> dict[str, Any]:
    configs = load_scoring_configs(project_root)
    mediapipe_config = dict(
        configs.get("landmarks", {}).get("landmarks", {}).get("landmark_sources", {}).get("mediapipe_face_mesh", {})
    )
    reference_config = dict(configs.get("reference", {}).get("reference", {}))
    image = load_image(image_path)
    mediapipe_result = build_mediapipe_source_result(image_bgr=image, config=mediapipe_config, project_root=project_root)
    mediapipe_landmarks = mediapipe_result.landmarks if mediapipe_result.available else None
    save_image(output_analysis_path, image)
    save_image(output_aligned_path, image)
    num_faces = 1 if mediapipe_result.available else 0
    quality_pass = bool(mediapipe_result.available and num_faces == 1)
    quality_fail_reason = None
    if not mediapipe_result.available:
        quality_fail_reason = "mediapipe_no_face_detected"
    elif num_faces > 1:
        quality_fail_reason = "mediapipe_multiple_faces_detected"
    row = {
        "quality_pass": quality_pass,
        "quality_fail_reason": quality_fail_reason,
        "processing_status": "processed" if mediapipe_result.available else "detection_error",
        "processing_error_message": None if mediapipe_result.available else ";".join(mediapipe_result.warnings or []),
        "yaw": None,
        "pitch": None,
        "roll": None,
    }
    landmarks = None
    landmarks_dense = None
    bbox = _bbox_from_landmarks(mediapipe_landmarks, image.shape[1], image.shape[0])

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

    result = {
        "image_id": image_id,
        "image_path": str(image_path),
        "file_name": file_name,
        "source_score_folder": source_score_folder,
        "target_score": float(target_score),
        "target_group_label": target_group_label,
        "split": split,
        "dataset_name": "scoring_dataset",
        "sample_id": image_id,
        "subject_id": None,
        "quality_pass": bool(row.get("quality_pass")),
        "quality_fail_reason": row.get("quality_fail_reason"),
        "processing_status": row.get("processing_status"),
        "processing_error_message": row.get("processing_error_message"),
        "raw_labels": _json_or_none({"source_score_folder": source_score_folder}),
        "normalized_labels": _json_or_none({}),
        "analysis_image_path": (
            output_analysis_path.relative_to(project_root).as_posix()
            if output_analysis_path.exists() and output_analysis_path.is_relative_to(project_root)
            else (str(output_analysis_path) if output_analysis_path.exists() else None)
        ),
        "aligned_image_path": (
            output_aligned_path.relative_to(project_root).as_posix()
            if output_aligned_path.exists() and output_aligned_path.is_relative_to(project_root)
            else (str(output_aligned_path) if output_aligned_path.exists() else None)
        ),
        "reference_model": "scoring_experimental",
        **{column: None for column in SCORING_GROUP_COLUMNS},
        **raw_measurements,
        **normalized_measurements,
        **symmetry_features,
        **angle_features,
    }
    if bool(reference_config.get("use_fairface_reference", True) if use_fairface_reference is None else use_fairface_reference):
        reference_stats = compute_reference_stats(
            project_root=project_root,
            filters=reference_filters,
            feature_columns=geometry_feature_columns(),
        )
        result = append_reference_relative_features(
            pd.DataFrame([result]),
            stats=reference_stats,
            feature_columns=geometry_feature_columns(),
            include_types=list(reference_config.get("feature_types", ["z", "dir", "ratio", "absdev", "percentile"])),
        ).iloc[0].to_dict()
        result["reference_source"] = reference_config.get("reference_source", "fairface")
        result["reference_stats_path"] = reference_stats.stats_path.relative_to(project_root).as_posix()
        result["reference_filters"] = _json_or_none(reference_stats.filters)
        result["reference_requested_group"] = _json_or_none(reference_stats.requested_filters)
        result["reference_resolved_group"] = _json_or_none(reference_stats.filters)
        result["reference_fallback_used"] = bool(reference_stats.coverage.get("fallback_used", False))
        result["reference_sample_count"] = int(reference_stats.coverage.get("reference_sample_count", reference_stats.sample_count))
    return result


def run_scoring_feature_extraction(
    project_root: Path | None = None,
    *,
    mediapipe_delegate: str | None = None,
    mediapipe_fallback_to_cpu: bool | None = None,
    use_fairface_reference: bool | None = None,
) -> ScoringFeatureArtifacts:
    root = (project_root or Path.cwd()).resolve()
    configs = load_scoring_configs(root)
    scoring_config = configs["scoring"]["scoring"]
    reference_config = dict(configs.get("reference", {}).get("reference", {}))

    manifest_path = (root / str(scoring_config["manifest_path"])).resolve()
    features_path = (root / str(scoring_config["features_path"])).resolve()
    summary_path = (root / str(scoring_config["summary_paths"]["feature_summary_path"])).resolve()
    mediapipe_summary_path = (root / "reports/metrics/mediapipe_feature_runtime_summary.json").resolve()
    analysis_dir = (root / str(scoring_config["processed_analysis_dir"])).resolve()
    aligned_dir = (root / str(scoring_config["processed_aligned_dir"])).resolve()
    ensure_directory(analysis_dir)
    ensure_directory(aligned_dir)

    manifest_rows = read_parquet(manifest_path)
    feature_rows: list[dict[str, Any]] = []

    _default_reference_model = configs["data"]["dataset"].get("default_reference_model")
    mediapipe_config = dict(
        configs.get("landmarks", {}).get("landmarks", {}).get("landmark_sources", {}).get("mediapipe_face_mesh", {})
    )
    if mediapipe_delegate is not None:
        mediapipe_config["delegate"] = str(mediapipe_delegate)
    if mediapipe_fallback_to_cpu is not None:
        mediapipe_config["fallback_to_cpu"] = bool(mediapipe_fallback_to_cpu)
    fairface_reference_enabled = bool(reference_config.get("use_fairface_reference", True) if use_fairface_reference is None else use_fairface_reference)
    reference_stats = None
    if fairface_reference_enabled:
        reference_stats = compute_reference_stats(
            project_root=root,
            filters=dict(reference_config.get("filters", {})),
            feature_columns=geometry_feature_columns(),
        )

    for row in manifest_rows:
        image_path = Path(str(row["image_path"]))
        resolved_image_path = image_path if image_path.is_absolute() else (root / image_path).resolve()
        image_id = str(row["image_id"])
        output_analysis_path = analysis_dir / f"{image_id}.jpg"
        output_aligned_path = aligned_dir / f"{image_id}.jpg"
        processing_error_message = None
        image = None
        row_payload: dict[str, Any]
        try:
            image = load_image(resolved_image_path)
            save_image(output_analysis_path, image)
            save_image(output_aligned_path, image)
            mediapipe_result = build_mediapipe_source_result(image_bgr=image, config=mediapipe_config, project_root=root)
            quality_pass = bool(mediapipe_result.available)
            row_payload = {
                "quality_pass": quality_pass,
                "quality_fail_reason": None if quality_pass else "mediapipe_no_face_detected",
                "processing_status": "processed" if quality_pass else "detection_error",
                "processing_error_message": None if quality_pass else ";".join(mediapipe_result.warnings or []),
                "yaw": None,
                "pitch": None,
                "roll": None,
            }
        except Exception as exc:
            processing_error_message = str(exc)
            row_payload = {
                "quality_pass": False,
                "quality_fail_reason": "mediapipe_no_face_detected",
                "processing_status": "detection_error",
                "processing_error_message": processing_error_message,
                "yaw": None,
                "pitch": None,
                "roll": None,
            }
            mediapipe_result = None
        landmarks = None
        landmarks_dense = None
        mediapipe_landmarks = None
        if image is not None and mediapipe_result is not None and mediapipe_result.available:
            mediapipe_landmarks = mediapipe_result.landmarks
        bbox = _bbox_from_landmarks(mediapipe_landmarks, image.shape[1], image.shape[0]) if image is not None else None

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
            row=row_payload,
            landmarks_dense=landmarks_dense,
            bbox=bbox,
            mediapipe_landmarks=mediapipe_landmarks,
        )
        row_features = {
                "image_id": image_id,
                "image_path": str(resolved_image_path),
                "file_name": str(row["file_name"]),
                "source_score_folder": str(row["source_score_folder"]),
                "target_score": float(row["target_score"]),
                "target_group_label": str(row["target_group_label"]),
                "split": str(row["split"]),
                "dataset_name": "scoring_dataset",
                "sample_id": image_id,
                "subject_id": None,
                "quality_pass": bool(row_payload.get("quality_pass")),
                "quality_fail_reason": row_payload.get("quality_fail_reason"),
                "processing_status": row_payload.get("processing_status"),
                "processing_error_message": row_payload.get("processing_error_message"),
                "raw_labels": _json_or_none({"source_score_folder": str(row["source_score_folder"])}),
                "normalized_labels": _json_or_none({}),
                "analysis_image_path": (
                    to_relative_path(output_analysis_path, root) if output_analysis_path.exists() else None
                ),
                "aligned_image_path": (
                    to_relative_path(output_aligned_path, root) if output_aligned_path.exists() else None
                ),
                "reference_model": "scoring_experimental",
                **{column: None for column in SCORING_GROUP_COLUMNS},
                **raw_measurements,
                **normalized_measurements,
                **symmetry_features,
                **angle_features,
            }
        if reference_stats is not None:
            row_features = append_reference_relative_features(
                pd.DataFrame([row_features]),
                stats=reference_stats,
                feature_columns=geometry_feature_columns(),
                include_types=list(reference_config.get("feature_types", ["z", "dir", "ratio", "absdev", "percentile"])),
            ).iloc[0].to_dict()
            row_features["reference_source"] = reference_config.get("reference_source", "fairface")
            row_features["reference_stats_path"] = reference_stats.stats_path.relative_to(root).as_posix()
            row_features["reference_filters"] = _json_or_none(reference_stats.filters)
            row_features["reference_requested_group"] = _json_or_none(reference_stats.requested_filters)
            row_features["reference_resolved_group"] = _json_or_none(reference_stats.filters)
            row_features["reference_fallback_used"] = bool(reference_stats.coverage.get("fallback_used", False))
            row_features["reference_sample_count"] = int(reference_stats.coverage.get("reference_sample_count", reference_stats.sample_count))
        feature_rows.append(row_features)

    fieldnames = SCORING_METADATA_COLUMNS + SCORING_GROUP_COLUMNS + geometry_feature_columns()
    if reference_stats is not None:
        fieldnames += [
            "reference_source",
            "reference_stats_path",
            "reference_filters",
            "reference_requested_group",
            "reference_resolved_group",
            "reference_fallback_used",
            "reference_sample_count",
        ]
        fieldnames += reference_relative_feature_names(
            geometry_feature_columns(),
            include_types=list(reference_config.get("feature_types", ["z", "dir", "ratio", "absdev", "percentile"])),
        )
    fieldnames = list(dict.fromkeys(fieldnames))
    write_parquet(feature_rows, features_path, fieldnames)

    quality_pass_count = sum(1 for row in feature_rows if bool(row.get("quality_pass")))
    processed_count = sum(1 for row in feature_rows if str(row.get("processing_status")).strip().lower() in PROCESSED_STATUSES)
    mediapipe_columns = [column for column in geometry_feature_columns() if column.startswith("mediapipe_")]
    mediapipe_feature_rows = []
    if feature_rows and mediapipe_columns:
        mediapipe_frame = pd.DataFrame(feature_rows).loc[:, mediapipe_columns].apply(pd.to_numeric, errors="coerce")
        missing_rate = mediapipe_frame.isna().mean() if not mediapipe_frame.empty else pd.Series(dtype=float)
        variances = mediapipe_frame.var(axis=0, numeric_only=True).fillna(0.0) if not mediapipe_frame.empty else pd.Series(dtype=float)
        mediapipe_feature_rows = [
            {
                "feature_name": column,
                "missing_rate": float(missing_rate.get(column, 1.0)),
                "variance": float(variances.get(column, 0.0)),
                "all_null": bool(float(missing_rate.get(column, 1.0)) >= 1.0),
            }
            for column in mediapipe_columns
        ]
    mediapipe_summary = {
        "status": "completed",
        "features_path": features_path.relative_to(root).as_posix(),
        "total_rows": int(len(feature_rows)),
        "total_feature_columns": int(len(geometry_feature_columns())),
        "mediapipe_feature_count": int(len(mediapipe_columns)),
        "mediapipe_non_null_feature_count": int(sum(1 for row in mediapipe_feature_rows if not row["all_null"])),
        "mediapipe_lt_0_2_missing": int(sum(1 for row in mediapipe_feature_rows if row["missing_rate"] < 0.2)),
        "mediapipe_lt_0_5_missing": int(sum(1 for row in mediapipe_feature_rows if row["missing_rate"] < 0.5)),
        "mediapipe_all_null_feature_count": int(sum(1 for row in mediapipe_feature_rows if row["all_null"])),
        "rows": mediapipe_feature_rows,
    }
    save_json(mediapipe_summary, mediapipe_summary_path)
    mediapipe_gap_audit_path = (root / "reports/metrics/mediapipe_feature_gap_audit.json").resolve()
    mediapipe_expanded_distribution_path = (root / "reports/metrics/mediapipe_expanded_feature_distribution.json").resolve()
    selected_features_payload = None
    selected_features_path = (root / "reports/metrics/scorer_selected_features.json").resolve()
    if selected_features_path.exists():
        try:
            selected_features_payload = json.loads(selected_features_path.read_text(encoding="utf-8"))
        except Exception:
            selected_features_payload = None
    registry_rows = feature_registry_rows()
    registry_mediapipe = [row for row in registry_rows if str(row.get("feature_name", "")).startswith("mediapipe_")]
    registry_by_name = {str(row.get("feature_name")): row for row in registry_mediapipe}
    mediapipe_columns_set = set(mediapipe_columns)
    selected_features = [
        str(name)
        for name in (selected_features_payload or {}).get("selected_features", [])
        if isinstance(name, str)
    ]
    selected_mediapipe = [name for name in selected_features if name.startswith("mediapipe_")]
    selected_ref_mediapipe = [
        name
        for name in selected_features
        if name.startswith(("ref_z_mediapipe_", "ref_dir_mediapipe_", "ref_ratio_mediapipe_", "ref_absdev_mediapipe_", "ref_percentile_mediapipe_"))
    ]
    reliability_counts: dict[str, int] = {}
    component_counts: dict[str, dict[str, int]] = {}
    excluded_candidates: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    if feature_rows and mediapipe_columns:
        mediapipe_frame = pd.DataFrame(feature_rows).reindex(columns=mediapipe_columns).apply(pd.to_numeric, errors="coerce")
        for column in mediapipe_columns:
            series = mediapipe_frame[column]
            reliability = str(registry_by_name.get(column, {}).get("reliability", "unknown"))
            component = str(registry_by_name.get(column, {}).get("component", "unknown"))
            recommended = bool(registry_by_name.get(column, {}).get("recommended_for_scorer", False))
            missing_rate = float(series.isna().mean())
            nonfinite_count = int((~np.isfinite(series.to_numpy(dtype=np.float64, na_value=np.nan))).sum() - series.isna().sum())
            variance = float(series.dropna().var()) if len(series.dropna()) else 0.0
            distribution_rows.append(
                {
                    "feature_name": column,
                    "component": component,
                    "reliability": reliability,
                    "recommended_for_scorer": recommended,
                    "missing_rate": missing_rate,
                    "nonfinite_count": max(nonfinite_count, 0),
                    "mean": float(series.mean()) if len(series.dropna()) else None,
                    "std": float(series.std(ddof=0)) if len(series.dropna()) else None,
                    "min": float(series.min()) if len(series.dropna()) else None,
                    "max": float(series.max()) if len(series.dropna()) else None,
                    "near_zero_variance": bool(variance <= 1e-12),
                    "outlier_count": int((((series - series.mean()).abs() > 4.0 * (series.std(ddof=0) or 0.0)).fillna(False)).sum()) if len(series.dropna()) else 0,
                }
            )
            reliability_counts[reliability] = reliability_counts.get(reliability, 0) + 1
            comp = component_counts.setdefault(component, {"total": 0, "non_null": 0, "selected": 0})
            comp["total"] += 1
            if missing_rate < 1.0:
                comp["non_null"] += 1
            if column in selected_mediapipe:
                comp["selected"] += 1
            if missing_rate > 0.4:
                excluded_candidates.append({"feature_name": column, "reason": "missing_rate_gt_0.4", "missing_rate": missing_rate})
            elif variance <= 1e-12:
                excluded_candidates.append({"feature_name": column, "reason": "near_zero_variance", "variance": variance})
            elif not recommended:
                excluded_candidates.append({"feature_name": column, "reason": "not_recommended_for_scorer"})
    save_json(
        {
            "status": "completed",
            "rows": int(len(feature_rows)),
            "total_mediapipe_features_in_registry": int(len(registry_mediapipe)),
            "total_mediapipe_features_runtime": int(len(mediapipe_columns)),
            "selected_direct_mediapipe_features": int(len(selected_mediapipe)),
            "selected_mediapipe_reference_features": int(len(selected_ref_mediapipe)),
            "selected_total_mediapipe_influence": int(len(selected_mediapipe) + len(selected_ref_mediapipe)),
            "reliability_counts": reliability_counts,
            "component_coverage": component_counts,
            "excluded_candidates": excluded_candidates[:300],
            "potential_high_score_separation_features": [
                row["feature_name"]
                for row in distribution_rows
                if row["feature_name"].startswith("mediapipe_")
                and row.get("recommended_for_scorer")
                and float(row.get("missing_rate", 1.0)) < 0.2
                and not bool(row.get("near_zero_variance"))
                and (
                    "symmetry" in row["feature_name"]
                    or "taper" in row["feature_name"]
                    or "cheekbone" in row["feature_name"]
                    or "jaw" in row["feature_name"]
                    or "lip" in row["feature_name"]
                )
            ][:120],
        },
        mediapipe_gap_audit_path,
    )
    save_json(
        {
            "status": "completed",
            "rows": distribution_rows,
        },
        mediapipe_expanded_distribution_path,
    )
    reference_relative_summary_path = (root / "reports/metrics/reference_relative_feature_summary.json").resolve()
    lip_formula_audit_path = (root / "reports/metrics/mediapipe_lip_formula_audit.json").resolve()
    lip_mapping_audit_path = (root / "reports/metrics/mediapipe_lip_landmark_mapping_audit.json").resolve()
    lip_distribution_path = (root / "reports/metrics/mediapipe_lip_feature_distribution.json").resolve()
    brow_formula_audit_path = (root / "reports/metrics/mediapipe_brow_formula_audit.json").resolve()
    brow_distribution_path = (root / "reports/metrics/mediapipe_brow_feature_distribution.json").resolve()
    reference_types = list(reference_config.get("feature_types", ["z", "dir", "ratio", "absdev", "percentile"]))
    reference_relative_columns = reference_relative_feature_names(
        geometry_feature_columns(),
        include_types=reference_types,
    )
    excluded_relative_features = []
    for base_feature in geometry_feature_columns():
        mean_value = None if reference_stats is None else reference_stats.mean.get(base_feature)
        std_value = None if reference_stats is None else reference_stats.std.get(base_feature)
        if "z" in reference_types and (std_value is None or abs(float(std_value)) <= 1e-9):
            excluded_relative_features.append({"feature_name": f"ref_z_{base_feature}", "reason": "small_std_or_missing_reference"})
        if "dir" in reference_types and mean_value is None:
            excluded_relative_features.append({"feature_name": f"ref_dir_{base_feature}", "reason": "missing_reference_stats"})
        if "ratio" in reference_types and (mean_value is None or abs(float(mean_value)) <= 1e-9):
            excluded_relative_features.append({"feature_name": f"ref_ratio_{base_feature}", "reason": "missing_or_zero_reference_mean"})
        if "absdev" in reference_types and mean_value is None:
            excluded_relative_features.append({"feature_name": f"ref_absdev_{base_feature}", "reason": "missing_reference_stats"})
        if "percentile" in reference_types and (std_value is None or abs(float(std_value)) <= 1e-9):
            excluded_relative_features.append({"feature_name": f"ref_percentile_{base_feature}", "reason": "small_std_or_missing_reference"})
    reference_relative_summary = {
        "status": "completed",
        "absolute_feature_count": int(len(geometry_feature_columns())),
        "reference_relative_feature_count": int(len(reference_relative_columns)),
        "rows_with_reference_features": int(len(feature_rows)) if reference_stats is not None else 0,
        "rows_with_exact_reference_group": int(len(feature_rows))
        if reference_stats is not None and not bool(reference_stats.coverage.get("fallback_used"))
        else 0,
        "rows_with_fallback_reference_group": int(len(feature_rows))
        if reference_stats is not None and bool(reference_stats.coverage.get("fallback_used"))
        else 0,
        "reference_context": reference_stats.coverage if reference_stats is not None else {},
        "excluded_reference_relative_features": excluded_relative_features,
    }
    save_json(reference_relative_summary, reference_relative_summary_path)
    feature_frame = pd.DataFrame(feature_rows) if feature_rows else pd.DataFrame()
    if not feature_frame.empty:
        lip_columns = [
            "mediapipe_mouth_width",
            "mediapipe_mouth_height",
            "mediapipe_upper_lip_height",
            "mediapipe_lower_lip_height",
            "mediapipe_total_lip_height",
            "mediapipe_upper_lip_area_proxy",
            "mediapipe_lower_lip_area_proxy",
            "mediapipe_lip_area_ratio",
        ]
        lip_numeric = feature_frame.reindex(columns=lip_columns).apply(pd.to_numeric, errors="coerce")
        lip_distribution = []
        for column in lip_columns:
            series = lip_numeric[column]
            lip_distribution.append(
                {
                    "feature_name": column,
                    "missing_rate": float(series.isna().mean()),
                    "mean": float(series.mean()) if len(series.dropna()) else None,
                    "std": float(series.std(ddof=0)) if len(series.dropna()) else None,
                    "min": float(series.min()) if len(series.dropna()) else None,
                    "max": float(series.max()) if len(series.dropna()) else None,
                    "near_zero_variance": bool(float(series.dropna().var()) <= 1e-12) if len(series.dropna()) else True,
                }
            )
        upper = lip_numeric["mediapipe_upper_lip_height"]
        lower = lip_numeric["mediapipe_lower_lip_height"]
        up_area = lip_numeric["mediapipe_upper_lip_area_proxy"]
        low_area = lip_numeric["mediapipe_lower_lip_area_proxy"]
        lip_formula_audit = {
            "status": "completed",
            "n_rows": int(len(lip_numeric)),
            "landmark_mapping": {
                "mouth_corners": [61, 291],
                "inner_opening": [13, 14],
                "upper_outer_inner": [0, 13],
                "lower_inner_outer": [14, 17],
                "outer_lip_contour": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78],
                "inner_lip_contour": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],
            },
            "sanity_flags": {
                "lip_area_ratio_dimensionless": True,
                "mouth_height_is_opening_proxy": True,
                "upper_lower_height_equal_rate": float((upper.round(6) == lower.round(6)).mean()) if len(lip_numeric) else None,
                "upper_lower_area_equal_rate": float((up_area.round(6) == low_area.round(6)).mean()) if len(lip_numeric) else None,
            },
            "notes": [
                "mediapipe_mouth_height represents mouth opening height (inner lip distance).",
                "mediapipe_lip_area_ratio is upper_lip_area_proxy / lower_lip_area_proxy.",
            ],
        }
        save_json(lip_formula_audit, lip_formula_audit_path)
        save_json(lip_formula_audit, lip_mapping_audit_path)
        save_json(
            {
                "status": "completed",
                "rows": lip_distribution,
                "equality_rate_upper_lower_height": float((upper.round(6) == lower.round(6)).mean()) if len(lip_numeric) else None,
                "equality_rate_upper_lower_area": float((up_area.round(6) == low_area.round(6)).mean()) if len(lip_numeric) else None,
            },
            lip_distribution_path,
        )
        landmarks_cfg = dict(configs.get("landmarks", {}).get("landmarks", {}))
        report_paths_cfg = dict(landmarks_cfg.get("report_paths", {}))
        indices_source = (root / str(report_paths_cfg.get("mediapipe_debug_indices_image_path", "reports/debug/mediapipe_landmarks_indices.png"))).resolve()
        lips_indices_target = (root / "reports/debug/mediapipe_lips_indices.png").resolve()
        if indices_source.exists():
            ensure_directory(lips_indices_target.parent)
            try:
                shutil.copyfile(indices_source, lips_indices_target)
            except Exception:
                pass
        brow_columns = [
            "mediapipe_left_brow_angle_deg",
            "mediapipe_right_brow_angle_deg",
            "mediapipe_brow_symmetry_index",
            "mediapipe_left_brow_width",
            "mediapipe_right_brow_width",
            "mediapipe_left_brow_arch_height_proxy",
            "mediapipe_right_brow_arch_height_proxy",
        ]
        brow_numeric = feature_frame.reindex(columns=brow_columns).apply(pd.to_numeric, errors="coerce")
        left = brow_numeric["mediapipe_left_brow_angle_deg"]
        right = brow_numeric["mediapipe_right_brow_angle_deg"]
        symmetry = brow_numeric["mediapipe_brow_symmetry_index"]
        angle_diff = (left - right).abs()
        save_json(
            {
                "status": "completed",
                "landmark_mapping": {
                    "left_brow_indices": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
                    "right_brow_indices": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
                    "endpoint_rule": "inner/outer chosen relative to face center; angle normalized with axis wrapping",
                },
                "sanity": {
                    "left_right_angle_absdiff_mean": float(angle_diff.mean()) if len(angle_diff.dropna()) else None,
                    "left_right_angle_absdiff_p95": float(angle_diff.quantile(0.95)) if len(angle_diff.dropna()) else None,
                    "corr_angle_diff_vs_symmetry": float(angle_diff.corr(symmetry)) if len(pd.DataFrame({'a': angle_diff, 'b': symmetry}).dropna()) >= 5 else None,
                },
            },
            brow_formula_audit_path,
        )
        save_json(
            {
                "status": "completed",
                "rows": [
                    {
                        "feature_name": column,
                        "missing_rate": float(brow_numeric[column].isna().mean()),
                        "mean": float(brow_numeric[column].mean()) if len(brow_numeric[column].dropna()) else None,
                        "std": float(brow_numeric[column].std(ddof=0)) if len(brow_numeric[column].dropna()) else None,
                        "min": float(brow_numeric[column].min()) if len(brow_numeric[column].dropna()) else None,
                        "max": float(brow_numeric[column].max()) if len(brow_numeric[column].dropna()) else None,
                    }
                    for column in brow_columns
                ],
            },
            brow_distribution_path,
        )
    summary = {
        "status": "completed",
        "manifest_rows": len(manifest_rows),
        "feature_rows": len(feature_rows),
        "quality_pass_rows": quality_pass_count,
        "processed_rows": processed_count,
        "failed_rows": len(feature_rows) - processed_count,
        "features_path": features_path.relative_to(root).as_posix(),
        "quality_gate_policy": scoring_config["quality_gate_policy"],
        "mediapipe_delegate": mediapipe_config.get("delegate"),
        "mediapipe_fallback_to_cpu": bool(mediapipe_config.get("fallback_to_cpu", True)),
        "mediapipe_runtime_summary_path": mediapipe_summary_path.relative_to(root).as_posix(),
        "mediapipe_feature_gap_audit_path": mediapipe_gap_audit_path.relative_to(root).as_posix(),
        "mediapipe_expanded_feature_distribution_path": mediapipe_expanded_distribution_path.relative_to(root).as_posix(),
        "use_fairface_reference": fairface_reference_enabled,
        "reference_stats_path": reference_stats.stats_path.relative_to(root).as_posix() if reference_stats is not None else None,
        "reference_relative_feature_summary_path": reference_relative_summary_path.relative_to(root).as_posix(),
        "mediapipe_lip_formula_audit_path": lip_formula_audit_path.relative_to(root).as_posix(),
        "mediapipe_lip_landmark_mapping_audit_path": lip_mapping_audit_path.relative_to(root).as_posix(),
        "mediapipe_lip_feature_distribution_path": lip_distribution_path.relative_to(root).as_posix(),
        "mediapipe_brow_formula_audit_path": brow_formula_audit_path.relative_to(root).as_posix(),
        "mediapipe_brow_feature_distribution_path": brow_distribution_path.relative_to(root).as_posix(),
    }
    save_json(summary, summary_path)
    return ScoringFeatureArtifacts(
        features_path=features_path,
        summary_path=summary_path,
        summary=summary,
    )
