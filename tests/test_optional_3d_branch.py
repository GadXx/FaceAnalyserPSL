from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.face_aesthetics.features.extraction_3d import extract_geometry_3d_features
from src.face_aesthetics.pipelines.feature_extraction import run_feature_extraction
from src.face_aesthetics.services.three_d import (
    DEFAULT_CANONICAL_TEMPLATE_5PT,
    OptionalFace3DProcessor,
    ensure_default_canonical_face_template,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_base_configs(project_root: Path, *, three_d_enabled: bool, artifacts_manifest_path: str) -> None:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "base.yaml", {"project": {"name": "optional-3d-test"}})
    _write_json(config_dir / "data.yaml", {"dataset": {"default_reference_model": "default_reference"}})
    _write_json(
        config_dir / "features.yaml",
        {
            "features": {
                "input_manifest_path": "data/processed/preprocessed_manifest.parquet",
                "measurements_raw_path": "data/features/measurements_raw.parquet",
                "measurements_normalized_path": "data/features/measurements_normalized.parquet",
                "symmetry_features_path": "data/features/symmetry_features.parquet",
                "geometry_features_path": "data/features/geometry_features.parquet",
                "geometry_3d_features_path": "data/features/geometry_3d_features.parquet",
                "summary_path": "reports/metrics/feature_extraction_summary.json",
                "image_key_column": "image_id",
                "quality_pass_column": "quality_pass",
                "processing_status_column": "processing_status",
                "required_processing_status": "processed",
                "default_reference_model": "default_reference",
                "three_d": {
                    "enabled": three_d_enabled,
                    "input_artifacts_manifest_path": artifacts_manifest_path,
                },
                "auxiliary_embeddings": {
                    "enabled": False,
                    "output_path": "data/features/deep_embeddings.parquet",
                    "encoder_name": "arcface",
                    "encoder_model_path": "missing.onnx",
                    "embedding_dim": 512,
                    "providers": ["CPUExecutionProvider"],
                    "ctx_id": -1,
                },
            }
        },
    )
    for name in (
        "preprocess.yaml",
        "train_baseline.yaml",
        "train_geometry.yaml",
        "train_fusion.yaml",
        "infer.yaml",
        "robustness.yaml",
    ):
        _write_json(config_dir / name, {})


class Optional3DBranchTests(unittest.TestCase):
    def test_default_template_is_created_deterministically_for_canonical_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / "canonical_face_5pt.npy"
            resolved = ensure_default_canonical_face_template(
                template_path=template_path,
                auto_create=True,
            )
            self.assertEqual(resolved, template_path)
            self.assertTrue(template_path.exists())
            loaded = np.load(template_path)
            np.testing.assert_allclose(loaded, DEFAULT_CANONICAL_TEMPLATE_5PT)

    def test_optional_3d_processor_skips_when_template_missing(self) -> None:
        processor = OptionalFace3DProcessor(
            enabled=True,
            template_path=Path("/tmp/definitely_missing_canonical_face_5pt.npy"),
        )
        payload = processor.availability_payload()
        self.assertEqual(payload["status"], "skipped_missing_models")
        self.assertFalse(payload["active"])

    def test_extract_geometry_3d_features_returns_expected_columns(self) -> None:
        features = extract_geometry_3d_features(
            [
                [30.0, 40.0, -2.0],
                [70.0, 40.0, 2.0],
                [50.0, 60.0, 6.0],
                [35.0, 80.0, -1.0],
                [65.0, 80.0, 1.0],
            ]
        )
        self.assertGreater(float(features["geom3d_eye_distance"]), 0.0)
        self.assertGreaterEqual(float(features["geom3d_depth_range"]), 0.0)

    def test_optional_3d_processor_is_active_with_generated_default_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / "canonical_face_5pt.npy"
            ensure_default_canonical_face_template(template_path=template_path, auto_create=True)
            processor = OptionalFace3DProcessor(
                enabled=True,
                template_path=template_path,
            )
            self.assertTrue(processor.availability.active)
            artifact = processor.build_artifact(
                image_id="sample-1",
                landmarks_2d=[
                    [30.0, 40.0],
                    [70.0, 40.0],
                    [50.0, 60.0],
                    [35.0, 80.0],
                    [65.0, 80.0],
                ],
                yaw=1.0,
                pitch=-1.0,
                roll=0.5,
            )
            self.assertEqual(artifact["status"], "success")
            self.assertEqual(len(artifact["landmarks_3d"]), 5)

    def test_feature_extraction_writes_empty_geometry_3d_when_branch_enabled_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_base_configs(
                project_root,
                three_d_enabled=True,
                artifacts_manifest_path="data/processed/3d_artifacts/manifest.parquet",
            )

            preprocessed = pd.DataFrame(
                [
                    {
                        "dataset_name": "test",
                        "sample_id": "sample-1",
                        "subject_id": "sample-1",
                        "image_id": "sample-1",
                        "split": "train",
                        "image_path": "data/raw/sample-1.jpg",
                        "aligned_image_path": "data/processed/aligned_faces/sample-1.jpg",
                        "reference_model": "default_reference",
                        "quality_pass": True,
                        "processing_status": "processed",
                        "bbox": json.dumps([20.0, 20.0, 80.0, 100.0]),
                        "landmarks": json.dumps(
                            [
                                [30.0, 40.0],
                                [70.0, 40.0],
                                [50.0, 60.0],
                                [35.0, 80.0],
                                [65.0, 80.0],
                            ]
                        ),
                        "yaw": 1.0,
                        "pitch": -1.0,
                        "roll": 0.5,
                    }
                ]
            )
            input_path = project_root / "data" / "processed" / "preprocessed_manifest.parquet"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            preprocessed.to_parquet(input_path, index=False)

            artifacts = run_feature_extraction(project_root=project_root)
            geometry_3d_frame = pd.read_parquet(artifacts.geometry_3d_features_path)

            self.assertTrue(artifacts.geometry_3d_features_path.exists())
            self.assertEqual(len(geometry_3d_frame), 0)
            self.assertEqual(artifacts.summary["three_d_reason"], "missing_or_empty_3d_artifacts_manifest")

    def test_feature_extraction_writes_geometry_3d_when_artifact_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_base_configs(
                project_root,
                three_d_enabled=True,
                artifacts_manifest_path="data/processed/3d_artifacts/manifest.parquet",
            )

            preprocessed = pd.DataFrame(
                [
                    {
                        "dataset_name": "test",
                        "sample_id": "sample-1",
                        "subject_id": "sample-1",
                        "image_id": "sample-1",
                        "split": "train",
                        "image_path": "data/raw/sample-1.jpg",
                        "aligned_image_path": "data/processed/aligned_faces/sample-1.jpg",
                        "reference_model": "default_reference",
                        "quality_pass": True,
                        "processing_status": "processed",
                        "bbox": json.dumps([20.0, 20.0, 80.0, 100.0]),
                        "landmarks": json.dumps(
                            [
                                [30.0, 40.0],
                                [70.0, 40.0],
                                [50.0, 60.0],
                                [35.0, 80.0],
                                [65.0, 80.0],
                            ]
                        ),
                        "yaw": 1.0,
                        "pitch": -1.0,
                        "roll": 0.5,
                    }
                ]
            )
            input_path = project_root / "data" / "processed" / "preprocessed_manifest.parquet"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            preprocessed.to_parquet(input_path, index=False)

            artifacts_dir = project_root / "data" / "processed" / "3d_artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifacts_dir / "sample-1.json"
            artifact_path.write_text(
                json.dumps(
                    {
                        "image_id": "sample-1",
                        "status": "success",
                        "artifact_version": "template_pose_lift@1",
                        "landmarks_3d": [
                            [30.0, 40.0, -2.0],
                            [70.0, 40.0, 2.0],
                            [50.0, 60.0, 6.0],
                            [35.0, 80.0, -1.0],
                            [65.0, 80.0, 1.0],
                        ],
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "image_id": "sample-1",
                        "artifact_path": "data/processed/3d_artifacts/sample-1.json",
                        "status": "success",
                        "reason": None,
                        "artifact_version": "template_pose_lift@1",
                        "template_path": "data/interim/3d_models/canonical_face_5pt.npy",
                    }
                ]
            ).to_parquet(artifacts_dir / "manifest.parquet", index=False)

            artifacts = run_feature_extraction(project_root=project_root)
            geometry_3d_frame = pd.read_parquet(artifacts.geometry_3d_features_path)

            self.assertEqual(len(geometry_3d_frame), 1)
            self.assertEqual(artifacts.summary["geometry_3d_rows"], 1)
            self.assertEqual(artifacts.summary["three_d_success_rows"], 1)


if __name__ == "__main__":
    unittest.main()
