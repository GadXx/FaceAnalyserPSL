from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import joblib
import numpy as np
import pandas as pd

from src.face_aesthetics.pipelines.inference import (
    _parse_group_override,
    _select_reference_group,
    run_inference,
)
from src.face_aesthetics.pipelines.preprocessing import SingleImagePreprocessingArtifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_minimal_configs(project_root: Path) -> None:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "base.yaml", {"project": {"name": "inference-test"}})
    _write_json(config_dir / "data.yaml", {"dataset": {"default_reference_model": "default_reference"}})
    _write_json(config_dir / "features.yaml", {})
    _write_json(config_dir / "preprocess.yaml", {"preprocess": {}})
    _write_json(config_dir / "train_geometry.yaml", {})
    _write_json(config_dir / "train_fusion.yaml", {})
    _write_json(
        config_dir / "train_baseline.yaml",
        {
            "train_baseline": {
                "group_label_aliases": {
                    "sex_group_normalized": ["sex_group_normalized", "gender_presentation"],
                    "age_group_normalized": ["age_group_normalized", "age_group"],
                    "ethnicity_group_normalized": ["ethnicity_group_normalized", "ethnicity_group"],
                }
            }
        },
    )
    _write_json(
        config_dir / "infer.yaml",
        {
            "infer": {
                "enabled": True,
                "output_dir": "reports/examples",
                "reference_model_path": "models/reference/reference_model.joblib",
                "reference_statistics_path": "models/reference/reference_group_statistics.parquet",
                "reporting_summary_path": "reports/metrics/reporting_summary.json",
                "top_k_features": 3,
            }
        },
    )
    _write_json(config_dir / "robustness.yaml", {})


def _reference_payload() -> dict:
    return {
        "feature_columns": [
            "ratio_face_aspect",
            "symmetry_index",
            "angle_eye_line_deg",
        ],
        "groupings": [
            ["sex_group_normalized", "age_group_normalized", "ethnicity_group_normalized"],
            ["sex_group_normalized", "age_group_normalized"],
            ["age_group_normalized"],
            ["sex_group_normalized"],
        ],
        "reference_groups": {
            "sex_group_normalized__age_group_normalized__ethnicity_group_normalized": {
                "sex_group_normalized=Female__age_group_normalized=20-29__ethnicity_group_normalized=GroupA": {
                    "group_columns": [
                        "sex_group_normalized",
                        "age_group_normalized",
                        "ethnicity_group_normalized",
                    ],
                    "group_values": {
                        "sex_group_normalized": "Female",
                        "age_group_normalized": "20-29",
                        "ethnicity_group_normalized": "GroupA",
                    },
                    "sample_count": 20,
                    "complete_case_count": 20,
                    "feature_columns": [
                        "ratio_face_aspect",
                        "symmetry_index",
                        "angle_eye_line_deg",
                    ],
                    "mean_vector": [0.75, 0.95, 0.0],
                    "std_vector": [0.05, 0.02, 1.0],
                    "covariance_matrix": [
                        [0.01, 0.0, 0.0],
                        [0.0, 0.01, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            },
            "sex_group_normalized__age_group_normalized": {
                "sex_group_normalized=Female__age_group_normalized=20-29": {
                    "group_columns": [
                        "sex_group_normalized",
                        "age_group_normalized",
                    ],
                    "group_values": {
                        "sex_group_normalized": "Female",
                        "age_group_normalized": "20-29",
                    },
                    "sample_count": 32,
                    "complete_case_count": 32,
                    "feature_columns": [
                        "ratio_face_aspect",
                        "symmetry_index",
                        "angle_eye_line_deg",
                    ],
                    "mean_vector": [0.76, 0.94, 0.1],
                    "std_vector": [0.05, 0.02, 1.0],
                    "covariance_matrix": [
                        [0.01, 0.0, 0.0],
                        [0.0, 0.01, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            },
            "age_group_normalized": {
                "age_group_normalized=20-29": {
                    "group_columns": ["age_group_normalized"],
                    "group_values": {"age_group_normalized": "20-29"},
                    "sample_count": 48,
                    "complete_case_count": 48,
                    "feature_columns": [
                        "ratio_face_aspect",
                        "symmetry_index",
                        "angle_eye_line_deg",
                    ],
                    "mean_vector": [0.78, 0.93, 0.2],
                    "std_vector": [0.05, 0.02, 1.0],
                    "covariance_matrix": [
                        [0.01, 0.0, 0.0],
                        [0.0, 0.01, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            }
        },
    }


def _statistics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group_key": "sex_group_normalized=Female__age_group_normalized=20-29__ethnicity_group_normalized=GroupA",
                "feature_name": "ratio_face_aspect",
                "p05": 0.65,
                "p25": 0.70,
                "p50": 0.75,
                "p75": 0.80,
                "p95": 0.85,
            },
            {
                "group_key": "sex_group_normalized=Female__age_group_normalized=20-29__ethnicity_group_normalized=GroupA",
                "feature_name": "symmetry_index",
                "p05": 0.85,
                "p25": 0.90,
                "p50": 0.95,
                "p75": 0.98,
                "p95": 1.00,
            },
            {
                "group_key": "sex_group_normalized=Female__age_group_normalized=20-29__ethnicity_group_normalized=GroupA",
                "feature_name": "angle_eye_line_deg",
                "p05": -2.0,
                "p25": -1.0,
                "p50": 0.0,
                "p75": 1.0,
                "p95": 2.0,
            },
        ]
    )


def _fake_preprocessing_artifacts(
    *,
    analysis_image_path: Path | None = None,
    aligned_image_path: Path | None = None,
    measurement_image_path: Path | None = None,
    measurement_image_role: str = "analysis_image",
) -> SingleImagePreprocessingArtifacts:
    row = {
        "image_id": "single_inference::sample",
        "processing_status": "processed",
        "quality_pass": True,
        "quality_fail_reason": None,
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
        "landmarks_dense": json.dumps(
            [
                [26.0, 30.0], [32.0, 28.0], [38.0, 27.0], [62.0, 27.0], [68.0, 28.0], [74.0, 30.0],
                [20.0, 40.0], [30.0, 40.0], [38.0, 38.0], [46.0, 39.0], [54.0, 39.0], [62.0, 38.0], [70.0, 40.0], [80.0, 40.0],
                [49.0, 50.0], [51.0, 50.0], [45.0, 58.0], [55.0, 58.0],
                [36.0, 74.0], [42.0, 72.0], [50.0, 71.0], [58.0, 72.0], [64.0, 74.0],
                [40.0, 80.0], [50.0, 82.0], [60.0, 80.0],
                [46.0, 88.0], [54.0, 88.0], [50.0, 96.0],
            ]
        ),
        "det_score": 0.99,
        "yaw": 1.0,
        "pitch": -1.0,
        "roll": 0.5,
        "num_faces": 1,
        "detection_backend": "stub",
    }
    return SingleImagePreprocessingArtifacts(
        row=row,
        aligned_image_path=aligned_image_path,
        analysis_image_path=analysis_image_path,
        measurement_image_path=measurement_image_path,
        measurement_image_role=measurement_image_role,
        provider_diagnostics={"selected_execution_provider": "CPUExecutionProvider"},
    )


class InferencePipelineTests(unittest.TestCase):
    def test_parse_group_override_age_only(self) -> None:
        self.assertEqual(_parse_group_override("age_group_normalized=20-29"), {"age_group_normalized": "20-29"})

    def test_parse_group_override_sex_and_age(self) -> None:
        self.assertEqual(
            _parse_group_override("sex_group_normalized=Female,age_group_normalized=20-29"),
            {
                "sex_group_normalized": "Female",
                "age_group_normalized": "20-29",
            },
        )

    def test_parse_group_override_full_three_field_string(self) -> None:
        self.assertEqual(
            _parse_group_override("sex_group_normalized=Female,age_group_normalized=20-29,ethnicity_group_normalized=GroupA"),
            {
                "sex_group_normalized": "Female",
                "age_group_normalized": "20-29",
                "ethnicity_group_normalized": "GroupA",
            },
        )

    def test_parse_group_override_accepts_reference_group_key_format(self) -> None:
        self.assertEqual(
            _parse_group_override("sex_group_normalized=Female__age_group_normalized=20-29__ethnicity_group_normalized=GroupA"),
            {
                "sex_group_normalized": "Female",
                "age_group_normalized": "20-29",
                "ethnicity_group_normalized": "GroupA",
            },
        )

    def test_reference_group_selection_falls_back_to_more_general_group(self) -> None:
        group_key, payload = _select_reference_group(
            _reference_payload(),
            {
                "sex_group_normalized": "Female",
                "age_group_normalized": "20-29",
                "ethnicity_group_normalized": "White",
            },
        )
        self.assertEqual(group_key, "sex_group_normalized=Female__age_group_normalized=20-29")
        self.assertEqual(payload["group_values"]["sex_group_normalized"], "Female")

    def test_inference_generates_reports_with_group_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)

            image_path = project_root / "sample.jpg"
            image_path.write_bytes(b"fake")
            reference_dir = project_root / "models" / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(_reference_payload(), reference_dir / "reference_model.joblib")
            _statistics_frame().to_parquet(reference_dir / "reference_group_statistics.parquet", index=False)

            with patch(
                "src.face_aesthetics.pipelines.inference.preprocess_single_image",
                return_value=_fake_preprocessing_artifacts(),
            ):
                artifacts = run_inference(
                    image_path=image_path,
                    group_override="sex_group_normalized=Female,age_group_normalized=20-29,ethnicity_group_normalized=GroupA",
                    project_root=project_root,
                )

            self.assertTrue(artifacts.report_json_path.exists())
            self.assertTrue(artifacts.report_markdown_path.exists())
            self.assertTrue(artifacts.reporting_summary_path.exists())
            self.assertIn("input_quality", artifacts.report)
            self.assertIn("reference_comparison", artifacts.report)
            self.assertIn("feature_level_deviations", artifacts.report)
            self.assertIn("limitations", artifacts.report)
            self.assertIn("transparency", artifacts.report)
            self.assertIn("optional_composite_indices", artifacts.report)
            self.assertIn("visualization", artifacts.report)
            self.assertIn("alignment", artifacts.report["input_quality"])
            self.assertIn("analysis_image_path", artifacts.report["artifacts"])
            self.assertIn("measurement_image_path", artifacts.report["artifacts"])
            self.assertIn("annotated_image_path", artifacts.report["artifacts"])
            self.assertEqual(
                artifacts.report["reference_comparison"]["selected_group"]["group_key"],
                "sex_group_normalized=Female__age_group_normalized=20-29__ethnicity_group_normalized=GroupA",
            )

    def test_inference_exposes_visualization_payload_with_anchor_points(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)

            image_path = project_root / "sample_visualization.jpg"
            image_path.write_bytes(b"fake")
            reference_dir = project_root / "models" / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(_reference_payload(), reference_dir / "reference_model.joblib")
            _statistics_frame().to_parquet(reference_dir / "reference_group_statistics.parquet", index=False)

            with patch(
                "src.face_aesthetics.pipelines.inference.preprocess_single_image",
                return_value=_fake_preprocessing_artifacts(),
            ), patch(
                "src.face_aesthetics.pipelines.inference.load_image",
                return_value=np.zeros((120, 100, 3), dtype=np.uint8),
            ):
                artifacts = run_inference(
                    image_path=image_path,
                    group_override="sex_group_normalized=Female,age_group_normalized=20-29,ethnicity_group_normalized=GroupA",
                    project_root=project_root,
                )

            visualization = artifacts.report["visualization"]
            self.assertEqual(visualization["status"], "available")
            self.assertEqual(visualization["image_size"], {"width": 100, "height": 120})
            self.assertEqual(visualization["image_role"], "analysis_image")
            self.assertEqual(visualization["coordinate_space"], "analysis_image")
            self.assertEqual(visualization["anchors"]["left_eye"], {"x": 30.0, "y": 40.0})
            self.assertEqual(visualization["derived_points"]["mouth_center"], {"x": 50.0, "y": 74.0})
            self.assertEqual(visualization["named_points"]["mouth_left"], {"x": 36.0, "y": 74.0})
            self.assertEqual(visualization["semantic_points"]["mouth_left"]["x"], 36.0)
            self.assertEqual(visualization["semantic_points"]["mouth_left"]["y"], 74.0)
            self.assertIn("validation_status", visualization["semantic_points"]["mouth_left"])
            self.assertIn("nose_base_center", visualization["named_points"])
            self.assertTrue(visualization["axes"])
            self.assertTrue(visualization["guides"])
            self.assertTrue(visualization["lines"])
            self.assertTrue(visualization["measurement_overlays"]["lines"])
            self.assertTrue(visualization["semantic_points_debug"])
            self.assertEqual(visualization["landmarks"][0]["index"], 0)
            self.assertIsNone(visualization["landmarks"][0]["name"])
            self.assertEqual(visualization["raw_landmarks"][0]["source_type"], "raw")
            self.assertTrue(artifacts.report["artifacts"]["visualization_debug_json_path"])

    def test_inference_visualization_gracefully_degrades_without_landmarks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)

            image_path = project_root / "sample_visualization_partial.jpg"
            image_path.write_bytes(b"fake")
            reference_dir = project_root / "models" / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(_reference_payload(), reference_dir / "reference_model.joblib")
            _statistics_frame().to_parquet(reference_dir / "reference_group_statistics.parquet", index=False)

            partial = _fake_preprocessing_artifacts()
            partial.row["landmarks"] = None
            with patch(
                "src.face_aesthetics.pipelines.inference.preprocess_single_image",
                return_value=partial,
            ), patch(
                "src.face_aesthetics.pipelines.inference.load_image",
                return_value=np.zeros((120, 100, 3), dtype=np.uint8),
            ), patch(
                "src.face_aesthetics.pipelines.inference.extract_raw_measurements",
                return_value={"raw_face_width_px": None},
            ), patch(
                "src.face_aesthetics.pipelines.inference.extract_normalized_measurements",
                return_value={"ratio_face_aspect": 0.75},
            ), patch(
                "src.face_aesthetics.pipelines.inference.extract_symmetry_features",
                return_value={"symmetry_index": 0.95},
            ), patch(
                "src.face_aesthetics.pipelines.inference.extract_angle_features",
                return_value={"angle_eye_line_deg": 0.0},
            ), patch(
                "src.face_aesthetics.pipelines.inference.validate_feature_values",
                return_value=None,
            ):
                artifacts = run_inference(
                    image_path=image_path,
                    group_override="sex_group_normalized=Female,age_group_normalized=20-29,ethnicity_group_normalized=GroupA",
                    project_root=project_root,
                )

            visualization = artifacts.report["visualization"]
            self.assertEqual(visualization["status"], "partial")
            self.assertIsNone(visualization["anchors"])
            self.assertTrue(visualization["notes"])

    def test_inference_accepts_reference_group_key_string_from_api_or_ui(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)

            image_path = project_root / "sample_group_key.jpg"
            image_path.write_bytes(b"fake")
            reference_dir = project_root / "models" / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(_reference_payload(), reference_dir / "reference_model.joblib")
            _statistics_frame().to_parquet(reference_dir / "reference_group_statistics.parquet", index=False)

            with patch(
                "src.face_aesthetics.pipelines.inference.preprocess_single_image",
                return_value=_fake_preprocessing_artifacts(),
            ):
                artifacts = run_inference(
                    image_path=image_path,
                    group_override="sex_group_normalized=Female__age_group_normalized=20-29__ethnicity_group_normalized=GroupA",
                    project_root=project_root,
                )

            self.assertEqual(
                artifacts.report["reference_comparison"]["selected_group"]["group_key"],
                "sex_group_normalized=Female__age_group_normalized=20-29__ethnicity_group_normalized=GroupA",
            )

    def test_inference_masks_unstable_reference_std_in_z_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)

            image_path = project_root / "sample_unstable_std.jpg"
            image_path.write_bytes(b"fake")
            reference_dir = project_root / "models" / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            payload = _reference_payload()
            payload["reference_groups"]["sex_group_normalized__age_group_normalized__ethnicity_group_normalized"][
                "sex_group_normalized=Female__age_group_normalized=20-29__ethnicity_group_normalized=GroupA"
            ]["std_vector"] = [1e-5, 0.02, 1.0]
            joblib.dump(payload, reference_dir / "reference_model.joblib")
            _statistics_frame().to_parquet(reference_dir / "reference_group_statistics.parquet", index=False)

            with patch(
                "src.face_aesthetics.pipelines.inference.preprocess_single_image",
                return_value=_fake_preprocessing_artifacts(),
            ):
                artifacts = run_inference(
                    image_path=image_path,
                    group_override="sex_group_normalized=Female,age_group_normalized=20-29,ethnicity_group_normalized=GroupA",
                    project_root=project_root,
                )

            self.assertIsNone(artifacts.report["reference_comparison"]["z_scores"]["ratio_face_aspect"])
            self.assertTrue(
                isinstance(artifacts.report["reference_comparison"]["mahalanobis_distance"], float)
            )

    def test_inference_uses_analysis_image_as_real_overlay_coordinate_space(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)

            image_path = project_root / "sample_analysis_source.jpg"
            analysis_image_path = project_root / "reports" / "examples" / "analysis_image.jpg"
            analysis_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), np.zeros((120, 80, 3), dtype=np.uint8))
            cv2.imwrite(str(analysis_image_path), np.zeros((300, 200, 3), dtype=np.uint8))

            reference_dir = project_root / "models" / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(_reference_payload(), reference_dir / "reference_model.joblib")
            _statistics_frame().to_parquet(reference_dir / "reference_group_statistics.parquet", index=False)

            with patch(
                "src.face_aesthetics.pipelines.inference.preprocess_single_image",
                return_value=_fake_preprocessing_artifacts(analysis_image_path=analysis_image_path),
            ):
                artifacts = run_inference(
                    image_path=image_path,
                    group_override="sex_group_normalized=Female,age_group_normalized=20-29,ethnicity_group_normalized=GroupA",
                    project_root=project_root,
                )

            self.assertEqual(artifacts.report["visualization"]["image_size"], {"width": 200, "height": 300})
            self.assertEqual(artifacts.report["visualization"]["image_path"], str(analysis_image_path.resolve()))
            self.assertEqual(artifacts.report["artifacts"]["original_image_path"], str(image_path.resolve()))
            self.assertEqual(artifacts.report["artifacts"]["analysis_image_path"], str(analysis_image_path.resolve()))
            self.assertEqual(
                artifacts.report["artifacts"]["normalized_frame_image_path"],
                str(analysis_image_path.resolve()),
            )
            self.assertIsNotNone(artifacts.annotated_image_path)
            self.assertTrue(artifacts.annotated_image_path.exists())

    def test_inference_prefers_aligned_face_as_measurement_and_overlay_image_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)

            image_path = project_root / "sample_measurement_source.jpg"
            analysis_image_path = project_root / "reports" / "examples" / "analysis_image.jpg"
            aligned_image_path = project_root / "reports" / "examples" / "aligned_face.jpg"
            analysis_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), np.zeros((200, 140, 3), dtype=np.uint8))
            cv2.imwrite(str(analysis_image_path), np.zeros((260, 220, 3), dtype=np.uint8))
            cv2.imwrite(str(aligned_image_path), np.zeros((224, 224, 3), dtype=np.uint8))

            reference_dir = project_root / "models" / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(_reference_payload(), reference_dir / "reference_model.joblib")
            _statistics_frame().to_parquet(reference_dir / "reference_group_statistics.parquet", index=False)

            with patch(
                "src.face_aesthetics.pipelines.inference.preprocess_single_image",
                return_value=_fake_preprocessing_artifacts(
                    analysis_image_path=analysis_image_path,
                    aligned_image_path=aligned_image_path,
                    measurement_image_path=aligned_image_path,
                    measurement_image_role="aligned_face",
                ),
            ):
                artifacts = run_inference(
                    image_path=image_path,
                    group_override="sex_group_normalized=Female,age_group_normalized=20-29,ethnicity_group_normalized=GroupA",
                    project_root=project_root,
                )

            self.assertEqual(artifacts.report["input"]["analysis_image_role"], "aligned_face")
            self.assertEqual(artifacts.report["visualization"]["image_role"], "aligned_face")
            self.assertEqual(artifacts.report["visualization"]["coordinate_space"], "aligned_face")
            self.assertEqual(artifacts.report["visualization"]["image_path"], str(aligned_image_path.resolve()))
            self.assertEqual(artifacts.report["visualization"]["image_size"], {"width": 224, "height": 224})
            self.assertEqual(artifacts.report["artifacts"]["analysis_image_path"], str(aligned_image_path.resolve()))
            self.assertEqual(
                artifacts.report["artifacts"]["normalized_frame_image_path"],
                str(analysis_image_path.resolve()),
            )
            self.assertEqual(artifacts.report["artifacts"]["aligned_image_path"], str(aligned_image_path.resolve()))
            self.assertEqual(artifacts.report["artifacts"]["measurement_image_path"], str(aligned_image_path.resolve()))
            self.assertEqual(artifacts.report["artifacts"]["overlay_image_path"], str(aligned_image_path.resolve()))

    def test_visualization_anchors_do_not_use_first_dense_landmarks_as_named_mouth_points(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)

            image_path = project_root / "sample_dense_landmarks.jpg"
            image_path.write_bytes(b"fake")
            reference_dir = project_root / "models" / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(_reference_payload(), reference_dir / "reference_model.joblib")
            _statistics_frame().to_parquet(reference_dir / "reference_group_statistics.parquet", index=False)

            with patch(
                "src.face_aesthetics.pipelines.inference.preprocess_single_image",
                return_value=_fake_preprocessing_artifacts(),
            ):
                artifacts = run_inference(
                    image_path=image_path,
                    group_override="sex_group_normalized=Female,age_group_normalized=20-29,ethnicity_group_normalized=GroupA",
                    project_root=project_root,
                )

            visualization = artifacts.report["visualization"]
            self.assertEqual(visualization["anchors"]["left_mouth"], {"x": 36.0, "y": 74.0})
            self.assertEqual(visualization["anchors"]["right_mouth"], {"x": 64.0, "y": 74.0})
            self.assertEqual(visualization["semantic_points"]["mouth_left"]["x"], 36.0)
            self.assertEqual(visualization["semantic_points"]["mouth_left"]["y"], 74.0)
            self.assertEqual(visualization["semantic_points"]["mouth_right"]["x"], 64.0)
            self.assertEqual(visualization["semantic_points"]["mouth_right"]["y"], 74.0)
            self.assertNotEqual(
                visualization["landmarks"][0],
                {
                    "index": 0,
                    "x": 35.0,
                    "y": 80.0,
                    "group": "mouth",
                    "name": "mouth_left",
                },
            )

    def test_inference_reads_sidecar_metadata_when_group_override_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)

            image_path = project_root / "sample2.jpg"
            image_path.write_bytes(b"fake")
            _write_json(
                image_path.with_suffix(image_path.suffix + ".json"),
                {
                    "normalized_labels": {
                        "gender_presentation": "Female",
                        "age_group": "20-29",
                        "ethnicity_group": "GroupA",
                    }
                },
            )

            reference_dir = project_root / "models" / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(_reference_payload(), reference_dir / "reference_model.joblib")
            _statistics_frame().to_parquet(reference_dir / "reference_group_statistics.parquet", index=False)

            with patch(
                "src.face_aesthetics.pipelines.inference.preprocess_single_image",
                return_value=_fake_preprocessing_artifacts(),
            ):
                artifacts = run_inference(
                    image_path=image_path,
                    project_root=project_root,
                )

            payload = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
            self.assertIn("raw_measurements", payload)
            self.assertIn("normalized_ratios", payload)
            self.assertIn("symmetry_metrics", payload)
            self.assertIn("angle_metrics", payload)
            self.assertIn("mahalanobis_distance", payload["reference_comparison"])
            self.assertIn("reference_group_selection", payload["transparency"])
            summary = json.loads(artifacts.reporting_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertIn("optional_composite_indices", summary)


if __name__ == "__main__":
    unittest.main()
