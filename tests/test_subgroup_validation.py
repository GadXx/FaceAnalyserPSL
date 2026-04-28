from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.face_aesthetics.pipelines.subgroup_validation import run_subgroup_validation


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_configs(project_root: Path) -> None:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "base.yaml", {"project": {"name": "subgroup-validation-test"}})
    _write_json(config_dir / "data.yaml", {"dataset": {"default_reference_model": "default_reference"}})
    _write_json(config_dir / "features.yaml", {})
    _write_json(
        config_dir / "train_baseline.yaml",
        {
            "train_baseline": {
                "group_label_aliases": {
                    "sex_group_normalized": ["gender_presentation"],
                    "age_group_normalized": ["age_group"],
                    "ethnicity_group_normalized": ["ethnicity_group"],
                }
            }
        },
    )
    _write_json(
        config_dir / "subgroup_validation.yaml",
        {
            "subgroup_validation": {
                "preprocessed_manifest_path": "data/processed/preprocessed_manifest.parquet",
                "geometry_features_path": "data/features/geometry_features.parquet",
                "reference_statistics_path": "models/reference/reference_group_statistics.parquet",
                "reference_model_path": "models/reference/reference_model.joblib",
                "output_validation_json_path": "reports/metrics/subgroup_validation.json",
                "output_reference_coverage_json_path": "reports/metrics/reference_group_coverage.json",
                "output_flat_table_path": "reports/metrics/subgroup_coverage.parquet",
                "image_id_column": "image_id",
                "subject_id_column": "subject_id",
                "dataset_name_column": "dataset_name",
                "normalized_labels_column": "normalized_labels",
                "quality_pass_column": "quality_pass",
                "processing_status_column": "processing_status",
                "quality_gate_status_column": "quality_gate_status",
                "quality_fail_reason_column": "quality_fail_reason",
                "group_label_aliases": {
                    "sex_group_normalized": ["gender_presentation"],
                    "age_group_normalized": ["age_group"],
                    "ethnicity_group_normalized": ["ethnicity_group"],
                },
                "weak_support_thresholds": {
                    "min_rows": 2,
                    "min_unique_subjects": 2,
                    "min_preprocessing_success_rate": 0.6,
                    "min_feature_coverage_rate": 0.6,
                    "min_mean_feature_completeness_rate": 0.9,
                    "min_reference_sample_count": 3,
                    "min_reference_complete_case_count": 2,
                },
                "top_k_incomplete_features": 3,
            }
        },
    )


class SubgroupValidationTests(unittest.TestCase):
    def test_subgroup_validation_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_configs(project_root)

            preprocessed = pd.DataFrame(
                [
                    {
                        "image_id": "img-1",
                        "subject_id": "subj-1",
                        "dataset_name": "fairface",
                        "normalized_labels": json.dumps(
                            {"gender_presentation": "Female", "age_group": "20-29", "ethnicity_group": "GroupA"}
                        ),
                        "quality_pass": True,
                        "processing_status": "processed",
                        "quality_gate_status": "passed",
                        "quality_fail_reason": None,
                    },
                    {
                        "image_id": "img-2",
                        "subject_id": "subj-2",
                        "dataset_name": "fairface",
                        "normalized_labels": json.dumps(
                            {"gender_presentation": "Female", "age_group": "20-29", "ethnicity_group": "GroupA"}
                        ),
                        "quality_pass": False,
                        "processing_status": "processed",
                        "quality_gate_status": "failed",
                        "quality_fail_reason": "no_face",
                    },
                    {
                        "image_id": "img-3",
                        "subject_id": "subj-3",
                        "dataset_name": "utkface",
                        "normalized_labels": json.dumps(
                            {"gender_presentation": "Male", "age_group": "30-39", "ethnicity_group": "GroupB"}
                        ),
                        "quality_pass": True,
                        "processing_status": "processed",
                        "quality_gate_status": "passed",
                        "quality_fail_reason": None,
                    },
                ]
            )
            features = pd.DataFrame(
                [
                    {
                        "image_id": "img-1",
                        "subject_id": "subj-1",
                        "dataset_name": "fairface",
                        "ratio_face_aspect": 0.8,
                        "symmetry_index": 0.9,
                        "angle_eye_line_deg": 1.0,
                    },
                    {
                        "image_id": "img-3",
                        "subject_id": "subj-3",
                        "dataset_name": "utkface",
                        "ratio_face_aspect": 0.7,
                        "symmetry_index": None,
                        "angle_eye_line_deg": 2.0,
                    },
                ]
            )
            reference_stats = pd.DataFrame(
                [
                    {
                        "combination_name": "sex_group_normalized",
                        "group_key": "sex_group_normalized=Female",
                        "feature_name": "ratio_face_aspect",
                        "sample_count": 4,
                        "mean": 0.8,
                        "std": 0.1,
                        "median": 0.8,
                        "iqr": 0.1,
                        "p05": 0.6,
                        "p25": 0.7,
                        "p50": 0.8,
                        "p75": 0.9,
                        "p95": 1.0,
                        "sex_group_normalized": "Female",
                        "age_group_normalized": None,
                        "ethnicity_group_normalized": None,
                    }
                ]
            )
            reference_payload = {
                "groupings": [["sex_group_normalized"]],
                "reference_groups": {
                    "sex_group_normalized": {
                        "sex_group_normalized=Female": {
                            "group_columns": ["sex_group_normalized"],
                            "group_values": {"sex_group_normalized": "Female"},
                            "sample_count": 4,
                            "complete_case_count": 3,
                            "feature_columns": ["ratio_face_aspect", "symmetry_index", "angle_eye_line_deg"],
                            "mean_vector": [0.8, 0.9, 1.0],
                            "std_vector": [0.1, 0.1, 0.5],
                            "covariance_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        },
                        "sex_group_normalized=Male": {
                            "group_columns": ["sex_group_normalized"],
                            "group_values": {"sex_group_normalized": "Male"},
                            "sample_count": 1,
                            "complete_case_count": 1,
                            "feature_columns": ["ratio_face_aspect", "symmetry_index", "angle_eye_line_deg"],
                            "mean_vector": [0.7, 0.8, 2.0],
                            "std_vector": [0.1, 0.1, 0.5],
                            "covariance_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        },
                    }
                },
            }

            preprocessed_path = project_root / "data" / "processed" / "preprocessed_manifest.parquet"
            features_path = project_root / "data" / "features" / "geometry_features.parquet"
            reference_stats_path = project_root / "models" / "reference" / "reference_group_statistics.parquet"
            reference_model_path = project_root / "models" / "reference" / "reference_model.joblib"
            preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
            features_path.parent.mkdir(parents=True, exist_ok=True)
            reference_stats_path.parent.mkdir(parents=True, exist_ok=True)
            preprocessed.to_parquet(preprocessed_path, index=False)
            features.to_parquet(features_path, index=False)
            reference_stats.to_parquet(reference_stats_path, index=False)
            joblib.dump(reference_payload, reference_model_path)

            artifacts = run_subgroup_validation(project_root=project_root)

            self.assertTrue(artifacts.validation_json_path.exists())
            self.assertTrue(artifacts.reference_coverage_json_path.exists())
            self.assertTrue(artifacts.flat_table_path.exists())

            summary = json.loads(artifacts.validation_json_path.read_text(encoding="utf-8"))
            reference_summary = json.loads(artifacts.reference_coverage_json_path.read_text(encoding="utf-8"))
            flat_frame = pd.read_parquet(artifacts.flat_table_path)

            self.assertEqual(summary["status"], "completed")
            self.assertIn("preprocessing_coverage", summary)
            self.assertIn("feature_coverage", summary)
            self.assertIn("reference_coverage_summary", summary)
            female_preprocess = next(
                row for row in summary["preprocessing_coverage"]["sex_group_normalized"] if row["group_value"] == "Female"
            )
            self.assertEqual(female_preprocess["row_count"], 2)
            self.assertAlmostEqual(female_preprocess["preprocessing_success_rate"], 0.5)
            self.assertTrue(any("low_preprocessing_success_rate" in item for item in female_preprocess["warnings"]))

            female_feature = next(
                row for row in summary["feature_coverage"]["sex_group_normalized"] if row["group_value"] == "Female"
            )
            self.assertEqual(female_feature["feature_row_count"], 1)
            self.assertAlmostEqual(female_feature["feature_coverage_rate"], 0.5)

            self.assertEqual(reference_summary["reference_group_count"], 2)
            male_reference = next(
                row for row in reference_summary["by_dimension"]["sex_group_normalized"] if row["group_value"] == "Male"
            )
            self.assertTrue(any("low_reference_sample_count" in item for item in male_reference["warnings"]))
            self.assertIn("coverage_type", flat_frame.columns)
            self.assertGreater(len(flat_frame), 0)


if __name__ == "__main__":
    unittest.main()
