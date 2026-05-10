from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.face_aesthetics.models.reference import (
    UNSTABLE_STD_THRESHOLD,
    compute_mahalanobis_distance,
    compute_z_scores,
    unstable_std_mask,
)
from src.face_aesthetics.pipelines.baseline_training import run_baseline_training


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_minimal_config_tree(project_root: Path) -> None:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    _write_json(config_dir / "base.yaml", {"project": {"name": "reference-model-test"}})
    _write_json(config_dir / "data.yaml", {"dataset": {"default_reference_model": "default_reference"}})
    _write_json(config_dir / "features.yaml", {})
    _write_json(config_dir / "preprocess.yaml", {})
    _write_json(config_dir / "train_geometry.yaml", {})
    _write_json(config_dir / "train_fusion.yaml", {})
    _write_json(config_dir / "infer.yaml", {})
    _write_json(config_dir / "robustness.yaml", {})
    _write_json(
        config_dir / "train_baseline.yaml",
        {
            "train_baseline": {
                "input_features_path": "data/features/geometry_features.parquet",
                "group_manifest_path": "data/processed/preprocessed_manifest.parquet",
                "image_id_column": "image_id",
                "split_column": "split",
                "normalized_labels_column": "normalized_labels",
                "feature_prefixes": ["ratio_", "sym_", "angle_"],
                "group_label_aliases": {
                    "sex_group_normalized": ["gender_presentation"],
                    "age_group_normalized": ["age_group"],
                    "ethnicity_group_normalized": ["ethnicity_group"],
                },
                "group_by": [
                    ["sex_group_normalized"],
                    ["sex_group_normalized", "age_group_normalized"],
                ],
                "allowed_reference_splits": ["train"],
                "minimum_group_sample_count": 2,
                "minimum_complete_case_count": 2,
                "percentiles": [0.05, 0.5, 0.95],
                "artifacts": {
                    "summary_json_path": "reports/metrics/reference_model_summary.json",
                    "group_statistics_path": "models/reference/reference_group_statistics.parquet",
                    "reference_model_path": "models/reference/reference_model.joblib",
                    "config_snapshot_path": "models/reference/reference_model_config_snapshot.json",
                },
            }
        },
    )


class ReferenceModelTests(unittest.TestCase):
    def test_compute_z_scores_stabilizes_zero_std(self) -> None:
        values = np.asarray([2.0, 1.0], dtype=np.float64)
        mean = np.asarray([1.0, 1.0], dtype=np.float64)
        std = np.asarray([0.0, 2.0], dtype=np.float64)

        z_scores = compute_z_scores(values, mean, std)

        self.assertTrue(np.isfinite(z_scores).all())
        self.assertAlmostEqual(float(z_scores[1]), 0.0)
        self.assertLess(abs(float(z_scores[0])), 2000.0)

    def test_unstable_std_mask_flags_poorly_conditioned_features(self) -> None:
        std = np.asarray([1e-4, 2e-2], dtype=np.float64)

        unstable = unstable_std_mask(std)

        self.assertTrue(bool(unstable[0]))
        self.assertFalse(bool(unstable[1]))
        self.assertGreater(UNSTABLE_STD_THRESHOLD, 1e-4)

    def test_compute_mahalanobis_distance_matches_identity_covariance(self) -> None:
        value = np.asarray([3.0, 4.0], dtype=np.float64)
        mean = np.asarray([0.0, 0.0], dtype=np.float64)
        covariance = np.eye(2, dtype=np.float64)

        distance = compute_mahalanobis_distance(value, mean, covariance)

        self.assertAlmostEqual(distance, 5.0, places=5)

    def test_reference_model_builder_uses_group_aliases_and_minimum_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_config_tree(project_root)

            features_frame = pd.DataFrame(
                [
                    {
                        "image_id": "img-1",
                        "split": "train",
                        "ratio_face_aspect": 0.70,
                        "symmetry_index": 0.95,
                        "angle_eye_line_deg": 1.0,
                    },
                    {
                        "image_id": "img-2",
                        "split": "train",
                        "ratio_face_aspect": 0.72,
                        "symmetry_index": 0.94,
                        "angle_eye_line_deg": 1.5,
                    },
                    {
                        "image_id": "img-3",
                        "split": "train",
                        "ratio_face_aspect": 0.68,
                        "symmetry_index": 0.90,
                        "angle_eye_line_deg": 2.0,
                    },
                    {
                        "image_id": "img-4",
                        "split": "val",
                        "ratio_face_aspect": 0.69,
                        "symmetry_index": 0.91,
                        "angle_eye_line_deg": 1.8,
                    },
                ]
            )
            manifest_frame = pd.DataFrame(
                [
                    {
                        "image_id": "img-1",
                        "split": "train",
                        "normalized_labels": json.dumps(
                            {
                                "gender_presentation": "Female",
                                "age_group": "20-29",
                                "ethnicity_group": "GroupA",
                            }
                        ),
                    },
                    {
                        "image_id": "img-2",
                        "split": "train",
                        "normalized_labels": json.dumps(
                            {
                                "gender_presentation": "Female",
                                "age_group": "20-29",
                                "ethnicity_group": "GroupA",
                            }
                        ),
                    },
                    {
                        "image_id": "img-3",
                        "split": "train",
                        "normalized_labels": json.dumps(
                            {
                                "gender_presentation": "Male",
                                "age_group": "20-29",
                                "ethnicity_group": "GroupA",
                            }
                        ),
                    },
                    {
                        "image_id": "img-4",
                        "split": "val",
                        "normalized_labels": json.dumps(
                            {
                                "gender_presentation": "Female",
                                "age_group": "30-39",
                                "ethnicity_group": "GroupA",
                            }
                        ),
                    },
                ]
            )

            features_path = project_root / "data" / "features" / "geometry_features.parquet"
            manifest_path = project_root / "data" / "processed" / "preprocessed_manifest.parquet"
            features_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            features_frame.to_parquet(features_path, index=False)
            manifest_frame.to_parquet(manifest_path, index=False)

            artifacts = run_baseline_training(project_root=project_root)

            self.assertEqual(artifacts.summary["status"], "completed")
            self.assertEqual(artifacts.summary["built_group_count"], 2)
            self.assertGreaterEqual(artifacts.summary["skipped_group_count"], 1)

            statistics_frame = pd.read_parquet(artifacts.group_statistics_path)
            self.assertIn("sex_group_normalized", statistics_frame.columns)
            self.assertIn("age_group_normalized", statistics_frame.columns)

            reference_payload = joblib.load(artifacts.reference_model_path)
            self.assertIn("sex_group_normalized", reference_payload["groupings"][0])
            self.assertIn(
                "sex_group_normalized=Female",
                reference_payload["reference_groups"]["sex_group_normalized"],
            )
            self.assertNotIn(
                "sex_group_normalized=Male",
                reference_payload["reference_groups"]["sex_group_normalized"],
            )


if __name__ == "__main__":
    unittest.main()
