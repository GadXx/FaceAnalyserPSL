from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.face_aesthetics.data.io import load_image, read_parquet, write_json, write_parquet
from src.face_aesthetics.features.extraction_3d import (
    GEOMETRY_3D_COLUMNS,
    extract_geometry_3d_features,
    validate_3d_feature_values,
)
from src.face_aesthetics.features.extraction import (
    ANGLE_FEATURE_COLUMNS,
    NORMALIZED_MEASUREMENT_COLUMNS,
    RAW_MEASUREMENT_COLUMNS,
    SYMMETRY_FEATURE_COLUMNS,
    extract_angle_features,
    extract_normalized_measurements,
    extract_raw_measurements,
    extract_symmetry_features,
    feature_audit_payload,
    feature_registry_rows,
    geometry_feature_columns,
    parse_json_field,
    validate_feature_values,
)
from src.face_aesthetics.services.embedding import InsightFaceEmbedder
from src.face_aesthetics.utils.config import load_named_configs


BASE_COLUMNS = [
    "dataset_name",
    "sample_id",
    "subject_id",
    "image_id",
    "split",
    "image_path",
    "aligned_image_path",
    "reference_model",
]

AUXILIARY_EMBEDDING_BASE_COLUMNS = BASE_COLUMNS + ["encoder_name", "embedding_dim"]


@dataclass
class FeatureExtractionArtifacts:
    measurements_raw_path: Path
    measurements_normalized_path: Path
    symmetry_features_path: Path
    geometry_features_path: Path
    geometry_3d_features_path: Path | None
    summary_path: Path
    feature_registry_path: Path
    feature_audit_path: Path
    auxiliary_embeddings_path: Path | None
    summary: dict[str, Any]


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
        return bool(stripped and stripped.lower() != "null")
    return True


def _base_row(row: dict[str, Any], default_reference_model: str) -> dict[str, Any]:
    return {
        "dataset_name": row.get("dataset_name"),
        "sample_id": row.get("sample_id"),
        "subject_id": row.get("subject_id"),
        "image_id": row["image_id"],
        "split": row.get("split"),
        "image_path": row.get("image_path"),
        "aligned_image_path": row.get("aligned_image_path"),
        "reference_model": row.get("reference_model") or default_reference_model,
    }


def _eligible_rows(
    manifest_rows: list[dict[str, Any]],
    *,
    quality_pass_column: str,
    processing_status_column: str,
    required_processing_status: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counters = {
        "quality_gate_filtered_rows": 0,
        "processing_status_filtered_rows": 0,
        "missing_geometry_rows": 0,
    }
    rows: list[dict[str, Any]] = []
    accepted_processing_statuses = {
        required_processing_status.strip().lower(),
        "success",
    }

    for row in manifest_rows:
        if not _is_truthy(row.get(quality_pass_column)):
            counters["quality_gate_filtered_rows"] += 1
            continue

        processing_status = row.get(processing_status_column)
        if _is_nonempty(processing_status):
            if str(processing_status).strip().lower() not in accepted_processing_statuses:
                counters["processing_status_filtered_rows"] += 1
                continue
        else:
            # Backward compatibility: older preprocessing manifests may not have
            # processing_status. In that case accept rows only if geometry and
            # aligned crop fields are present together with quality_pass=true.
            if not (_is_nonempty(row.get("aligned_image_path")) and _is_nonempty(row.get("landmarks"))):
                counters["processing_status_filtered_rows"] += 1
                continue

        if not _is_nonempty(row.get("bbox")) or not _is_nonempty(row.get("landmarks")):
            counters["missing_geometry_rows"] += 1
            continue
        rows.append(row)

    return rows, counters


def _build_embedder_if_available(features_config: dict[str, Any], root: Path) -> tuple[InsightFaceEmbedder | None, str | None]:
    embedding_config = dict(features_config.get("auxiliary_embeddings", {}))
    if not embedding_config.get("enabled", False):
        return None, "auxiliary_embeddings_disabled"

    encoder_model_path = (root / str(embedding_config["encoder_model_path"])).resolve()
    if not encoder_model_path.exists():
        return None, f"auxiliary_encoder_missing:{encoder_model_path.relative_to(root).as_posix()}"

    try:
        return (
            InsightFaceEmbedder(
                encoder_name=str(embedding_config["encoder_name"]),
                encoder_model_path=encoder_model_path,
                providers=list(embedding_config["providers"]),
                ctx_id=int(embedding_config["ctx_id"]),
                expected_embedding_dim=int(embedding_config["embedding_dim"]),
            ),
            None,
        )
    except Exception as exc:
        return None, f"auxiliary_encoder_unavailable:{exc}"


def _load_3d_manifest_lookup(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    rows = read_parquet(manifest_path)
    return {str(row["image_id"]): row for row in rows if row.get("image_id")}


def run_feature_extraction(project_root: Path | None = None) -> FeatureExtractionArtifacts:
    root = (project_root or Path.cwd()).resolve()
    configs = load_named_configs(root / "configs")
    features_config = configs["features"]["features"]

    input_manifest_path = (root / features_config["input_manifest_path"]).resolve()
    measurements_raw_path = (root / features_config["measurements_raw_path"]).resolve()
    measurements_normalized_path = (root / features_config["measurements_normalized_path"]).resolve()
    symmetry_features_path = (root / features_config["symmetry_features_path"]).resolve()
    geometry_features_path = (root / features_config["geometry_features_path"]).resolve()
    geometry_3d_features_path = (root / features_config["geometry_3d_features_path"]).resolve()
    summary_path = (root / features_config["summary_path"]).resolve()
    feature_registry_path = (
        root / str(features_config.get("feature_registry_path", "reports/metrics/feature_registry.json"))
    ).resolve()
    feature_audit_path = (
        root / str(features_config.get("feature_audit_path", "reports/metrics/feature_audit.json"))
    ).resolve()
    three_d_config = dict(features_config.get("three_d", {}))
    three_d_enabled = bool(three_d_config.get("enabled", False))
    three_d_input_manifest_path = (
        (root / str(three_d_config.get("input_artifacts_manifest_path"))).resolve()
        if three_d_enabled and three_d_config.get("input_artifacts_manifest_path")
        else None
    )
    auxiliary_embeddings_config = dict(features_config.get("auxiliary_embeddings", {}))
    auxiliary_embeddings_path = (
        (root / auxiliary_embeddings_config["output_path"]).resolve()
        if auxiliary_embeddings_config.get("enabled", False)
        else None
    )

    default_reference_model = str(features_config["default_reference_model"])
    manifest_rows = read_parquet(input_manifest_path)
    data_config = configs.get("data", {})
    upstream_manifest_relpath = (
        data_config.get("dataset", {}).get("interim_manifest_path")
        if isinstance(data_config, dict)
        else None
    )
    upstream_manifest_rows = None
    if isinstance(upstream_manifest_relpath, str):
        upstream_manifest_path = (root / upstream_manifest_relpath).resolve()
        if upstream_manifest_path.exists():
            upstream_manifest_rows = len(read_parquet(upstream_manifest_path))

    processed_rows = 0
    failed_rows = 0
    if manifest_rows:
        for row in manifest_rows:
            status = row.get(str(features_config["processing_status_column"]))
            if _is_nonempty(status):
                if str(status).strip().lower() in {"processed", "success"}:
                    processed_rows += 1
                else:
                    failed_rows += 1
            elif _is_truthy(row.get(str(features_config["quality_pass_column"]))):
                processed_rows += 1
            else:
                failed_rows += 1

    eligible_rows, filter_counts = _eligible_rows(
        manifest_rows,
        quality_pass_column=str(features_config["quality_pass_column"]),
        processing_status_column=str(features_config["processing_status_column"]),
        required_processing_status=str(features_config["required_processing_status"]),
    )

    raw_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    symmetry_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    geometry_3d_rows: list[dict[str, Any]] = []

    embedder, auxiliary_embedding_reason = _build_embedder_if_available(features_config, root)
    auxiliary_rows: list[dict[str, Any]] = []
    three_d_manifest_lookup = (
        _load_3d_manifest_lookup(three_d_input_manifest_path)
        if three_d_input_manifest_path is not None
        else {}
    )
    three_d_processed_rows = 0
    three_d_success_rows = 0
    three_d_reason = "disabled"
    if three_d_enabled:
        three_d_reason = "active" if three_d_manifest_lookup else "missing_or_empty_3d_artifacts_manifest"

    for row in eligible_rows:
        base = _base_row(row, default_reference_model)
        landmarks = parse_json_field(row.get("landmarks"))
        landmarks_dense = parse_json_field(row.get("landmarks_dense"))
        bbox = parse_json_field(row.get("bbox"))

        raw_measurements = extract_raw_measurements(landmarks=landmarks, bbox=bbox, landmarks_dense=landmarks_dense)
        normalized_measurements = extract_normalized_measurements(raw_measurements)
        symmetry_features = extract_symmetry_features(raw_measurements, normalized_measurements)
        angle_features = extract_angle_features(landmarks=landmarks, row=row, landmarks_dense=landmarks_dense, bbox=bbox)

        validate_feature_values(raw_measurements, image_id=row["image_id"])
        validate_feature_values(normalized_measurements, image_id=row["image_id"])
        validate_feature_values(symmetry_features, image_id=row["image_id"])
        validate_feature_values(angle_features, image_id=row["image_id"])

        raw_rows.append({**base, **raw_measurements})
        normalized_rows.append({**base, **normalized_measurements})
        symmetry_rows.append({**base, **symmetry_features})
        geometry_rows.append(
            {
                **base,
                **raw_measurements,
                **normalized_measurements,
                **symmetry_features,
                **angle_features,
            }
        )

        if three_d_enabled:
            three_d_processed_rows += 1
            three_d_manifest_row = three_d_manifest_lookup.get(str(row["image_id"]))
            if (
                three_d_manifest_row is not None
                and str(three_d_manifest_row.get("status")).strip().lower() == "success"
                and three_d_manifest_row.get("artifact_path")
            ):
                artifact_path = (root / str(three_d_manifest_row["artifact_path"])).resolve()
                artifact_payload = parse_json_field(artifact_path.read_text(encoding="utf-8"))
                geometry_3d = extract_geometry_3d_features(artifact_payload.get("landmarks_3d"))
                validate_3d_feature_values(geometry_3d, image_id=row["image_id"])
                geometry_3d_rows.append({**base, **geometry_3d})
                three_d_success_rows += 1

        if embedder is not None and row.get("aligned_image_path"):
            aligned_image = load_image((root / str(row["aligned_image_path"])).resolve())
            embedding = embedder.encode(aligned_image)
            auxiliary_rows.append(
                {
                    **base,
                    "encoder_name": auxiliary_embeddings_config["encoder_name"],
                    "embedding_dim": embedder.embedding_dim,
                    **{
                        f"emb_{index:04d}": float(value)
                        for index, value in enumerate(embedding.tolist())
                    },
                }
            )

    write_parquet(raw_rows, measurements_raw_path, BASE_COLUMNS + RAW_MEASUREMENT_COLUMNS)
    write_parquet(
        normalized_rows,
        measurements_normalized_path,
        BASE_COLUMNS + NORMALIZED_MEASUREMENT_COLUMNS,
    )
    write_parquet(
        symmetry_rows,
        symmetry_features_path,
        BASE_COLUMNS + SYMMETRY_FEATURE_COLUMNS,
    )
    write_parquet(
        geometry_rows,
        geometry_features_path,
        BASE_COLUMNS + geometry_feature_columns(),
    )
    write_parquet(
        geometry_3d_rows,
        geometry_3d_features_path,
        BASE_COLUMNS + GEOMETRY_3D_COLUMNS,
    )

    if auxiliary_embeddings_path is not None:
        auxiliary_columns = []
        if embedder is not None:
            auxiliary_columns = [f"emb_{index:04d}" for index in range(embedder.embedding_dim)]
        elif auxiliary_embeddings_config.get("embedding_dim") is not None:
            auxiliary_columns = [
                f"emb_{index:04d}"
                for index in range(int(auxiliary_embeddings_config["embedding_dim"]))
            ]
        write_parquet(
            auxiliary_rows,
            auxiliary_embeddings_path,
            AUXILIARY_EMBEDDING_BASE_COLUMNS + auxiliary_columns,
        )

    registry_rows = feature_registry_rows()
    feature_audit = feature_audit_payload()
    write_json(registry_rows, feature_registry_path)
    write_json(feature_audit, feature_audit_path)

    summary = {
        "input_rows": len(manifest_rows),
        "total_rows": len(manifest_rows),
        "eligible_rows": len(eligible_rows),
        "processed_rows": processed_rows,
        "failed_rows": failed_rows,
        "measurements_raw_rows": len(raw_rows),
        "measurements_normalized_rows": len(normalized_rows),
        "symmetry_rows": len(symmetry_rows),
        "geometry_rows": len(geometry_rows),
        "geometry_3d_rows": len(geometry_3d_rows),
        "raw_measurement_count": len(RAW_MEASUREMENT_COLUMNS),
        "normalized_measurement_count": len(NORMALIZED_MEASUREMENT_COLUMNS),
        "symmetry_feature_count": len(SYMMETRY_FEATURE_COLUMNS),
        "angle_feature_count": len(ANGLE_FEATURE_COLUMNS),
        "geometry_feature_count": len(geometry_feature_columns()),
        "geometry_3d_feature_count": len(GEOMETRY_3D_COLUMNS),
        "feature_registry_path": str(feature_registry_path),
        "feature_audit_path": str(feature_audit_path),
        "feature_registry_count": len(registry_rows),
        "three_d_enabled": three_d_enabled,
        "three_d_processed_rows": three_d_processed_rows,
        "three_d_success_rows": three_d_success_rows,
        "three_d_reason": three_d_reason,
        "filter_counts": filter_counts,
        "auxiliary_embeddings_enabled": auxiliary_embeddings_config.get("enabled", False),
        "auxiliary_embeddings_written": auxiliary_embeddings_path is not None,
        "auxiliary_embedding_rows": len(auxiliary_rows),
        "auxiliary_embedding_dim": embedder.embedding_dim if embedder is not None else 0,
        "auxiliary_embedding_reason": auxiliary_embedding_reason,
        "upstream_dataset_manifest_rows": upstream_manifest_rows,
        "input_manifest_empty": len(manifest_rows) == 0,
        "input_manifest_stale_against_dataset_manifest": bool(
            len(manifest_rows) == 0 and upstream_manifest_rows not in (None, 0)
        ),
        "paths": {
            "measurements_raw_path": measurements_raw_path.relative_to(root).as_posix(),
            "measurements_normalized_path": measurements_normalized_path.relative_to(root).as_posix(),
            "symmetry_features_path": symmetry_features_path.relative_to(root).as_posix(),
            "geometry_features_path": geometry_features_path.relative_to(root).as_posix(),
            "geometry_3d_features_path": geometry_3d_features_path.relative_to(root).as_posix(),
            "summary_path": summary_path.relative_to(root).as_posix(),
            "auxiliary_embeddings_path": (
                auxiliary_embeddings_path.relative_to(root).as_posix()
                if auxiliary_embeddings_path is not None
                else None
            ),
        },
    }
    write_json(summary, summary_path)

    return FeatureExtractionArtifacts(
        measurements_raw_path=measurements_raw_path,
        measurements_normalized_path=measurements_normalized_path,
        symmetry_features_path=symmetry_features_path,
        geometry_features_path=geometry_features_path,
        geometry_3d_features_path=geometry_3d_features_path,
        summary_path=summary_path,
        feature_registry_path=feature_registry_path,
        feature_audit_path=feature_audit_path,
        auxiliary_embeddings_path=auxiliary_embeddings_path,
        summary=summary,
    )
