from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
from fastapi import HTTPException

from src.face_aesthetics.api.app import create_app
from src.face_aesthetics.api.schemas import AnalyzeRequest
from src.face_aesthetics.pipelines.inference import InferenceArtifacts
from src.face_aesthetics.scoring.inference import ScorerInferenceArtifacts
from src.face_aesthetics.services.inference_service import ServiceDependencyError


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_minimal_configs(project_root: Path) -> None:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "data.yaml", {"dataset": {"default_reference_model": "default_reference"}})
    _write_json(
        config_dir / "preprocess.yaml",
        {
            "preprocess": {
                "insightface": {
                    "model_name": "buffalo_l",
                    "model_root": "data/interim/insightface_models",
                    "required_model_files": ["det_10g.onnx", "2d106det.onnx"],
                },
                "three_d": {
                    "template_path": "data/interim/3d_models/canonical_face_5pt.npy",
                },
            }
        },
    )
    _write_json(
        config_dir / "train_baseline.yaml",
        {"train_baseline": {"group_label_aliases": {"sex_group_normalized": ["gender_presentation"]}}},
    )
    _write_json(
        config_dir / "infer.yaml",
        {
            "infer": {
                "reference_model_path": "models/reference/reference_model.joblib",
                "reference_statistics_path": "models/reference/reference_group_statistics.parquet",
            }
        },
    )


def _fake_inference_artifacts(project_root: Path) -> InferenceArtifacts:
    report_json = project_root / "reports" / "examples" / "sample_report.json"
    report_md = project_root / "reports" / "examples" / "sample_report.md"
    summary = project_root / "reports" / "metrics" / "reporting_summary.json"
    normalized_frame_image = project_root / "reports" / "examples" / "analysis_image.jpg"
    measurement_image = project_root / "reports" / "examples" / "aligned_face.jpg"
    annotated_image = project_root / "reports" / "examples" / "annotated_analysis.jpg"
    report = {
        "input": {
            "image_path": str(project_root / "sample.jpg"),
            "output_dir": str(project_root / "reports" / "examples"),
            "analysis_image_role": "aligned_face",
            "overlay_image_role": "aligned_face",
        },
        "input_quality": {
            "processing_status": "processed",
            "quality_pass": True,
            "quality_fail_reason": None,
            "processing_error_message": None,
            "alignment": {
                "attempted": True,
                "applied": True,
                "improved_quality": True,
                "source": "eye_line",
                "status": "applied",
                "rotation_deg": -8.5,
                "transform": {"type": "affine_rotation"},
                "original_pose": {"yaw": 1.0, "pitch": 0.0, "roll": 8.5},
                "aligned_pose": {"yaw": 1.0, "pitch": 0.0, "roll": 0.7},
                "original_quality_pass": False,
                "original_quality_fail_reason": "roll_extreme",
                "aligned_quality_pass": True,
                "aligned_quality_fail_reason": None,
            },
            "detection": {
                "bbox": [10.0, 20.0, 80.0, 100.0],
                "det_score": 0.99,
                "pose": {"yaw": 1.0, "pitch": -1.0, "roll": 0.5},
                "num_faces": 1,
                "detection_backend": "stub",
                "blur_variance": 120.0,
                "occlusion_score": 0.1,
            },
        },
        "visualization": {
            "status": "available",
            "coordinate_space": "aligned_face",
            "image_role": "aligned_face",
            "image_path": str(measurement_image),
            "image_width": 100,
            "image_height": 120,
            "image_size": {"width": 100, "height": 120},
            "bbox": [18.0, 24.0, 76.0, 102.0],
            "detector_bbox": [10.0, 20.0, 80.0, 100.0],
            "display_face_box": [18.0, 24.0, 76.0, 102.0],
            "semantic_face_extents": {
                "left": 18.0,
                "right": 76.0,
                "top": 24.0,
                "bottom": 102.0,
                "width": 58.0,
                "height": 78.0,
                "center_x": 47.0,
                "center_y": 63.0,
                "source_type": "semantic_face_extents",
                "source_rule": "test_fixture",
                "source_indices": list(range(33)),
            },
            "raw_landmarks": [
                {"index": 0, "x": 10.0, "y": 18.0, "group": "dense_landmarks", "name": None, "source_type": "raw", "source_indices": [0], "description": "raw"},
                {"index": 5, "x": 30.0, "y": 40.0, "group": "dense_landmarks", "name": None, "source_type": "raw", "source_indices": [5], "description": "raw"},
            ],
            "raw_landmarks_debug": [
                {"index": 0, "x": 10.0, "y": 18.0, "group": "dense_landmarks", "name": None, "source_type": "raw", "source_indices": [0], "description": "raw"},
            ],
            "mediapipe_landmarks": [
                {"index": 1, "x": 12.0, "y": 20.0, "group": "dense_face_oval", "name": None, "source_type": "mediapipe_face_mesh", "source_indices": [1], "description": "mediapipe"},
                {"index": 2, "x": 13.0, "y": 21.0, "group": "dense_face_oval", "name": None, "source_type": "mediapipe_face_mesh", "source_indices": [2], "description": "mediapipe"},
            ],
            "mediapipe_regions": {
                "dense_face_oval": [1, 2],
                "dense_eyes": [33, 263],
            },
            "semantic_points": {
                "left_eye_center": {"x": 30.0, "y": 40.0, "source_type": "cluster", "source_indices": [33, 34], "validation_status": "validated"},
                "right_eye_center": {"x": 70.0, "y": 40.0, "source_type": "cluster", "source_indices": [87, 88], "validation_status": "validated"},
                "nose_tip": {"x": 50.0, "y": 60.0, "source_type": "index", "source_index_106": 86, "source_indices": [86], "source_region": "nose", "validation_status": "validated", "source_rule": "canonical_index_86"},
                "mouth_left": {"x": 35.0, "y": 80.0, "source_type": "index", "source_index_106": 52, "source_indices": [52], "source_region": "mouth", "validation_status": "validated"},
                "mouth_right": {"x": 65.0, "y": 80.0, "source_type": "index", "source_index_106": 61, "source_indices": [61], "source_region": "mouth", "validation_status": "validated"},
                "mouth_center": {"x": 50.0, "y": 80.0, "source_type": "derived", "source_indices": [52, 61], "validation_status": "validated"},
            },
            "semantic_points_debug": [
                {"point_name": "mouth_left", "group": "mouth", "x": 35.0, "y": 80.0, "source_type": "heuristic_dense_lips", "source_indices": [21], "description": "debug"},
            ],
            "measurement_overlays": {
                "lines": [
                    {
                        "key": "interpupillary_line",
                        "start": "left_eye_center",
                        "end": "right_eye_center",
                        "group": "measurements",
                        "kind": "measurement",
                        "layers": ["measurements", "eyes"],
                        "metric_key": "raw_eye_distance_px",
                        "supported_features": ["raw_eye_distance_px"],
                    }
                ],
                "axes": [
                    {
                        "key": "facial_midline",
                        "start": "facial_midline_top",
                        "end": "facial_midline_bottom",
                        "group": "symmetry",
                        "kind": "axis",
                        "layers": ["symmetry", "thirds"],
                        "supported_features": ["midline_alignment_index"],
                    }
                ],
                "guides": [
                    {
                        "key": "thirds_eye_guide",
                        "start": "left_eye_center",
                        "end": "right_eye_center",
                        "group": "thirds",
                        "kind": "guide",
                        "layers": ["thirds"],
                        "metric_key": "middle_third_proxy",
                        "supported_features": ["middle_third_proxy"],
                    }
                ],
            },
            "measurement_overlays_debug": {"lines": [], "axes": [], "guides": []},
            "landmarks": [
                {"index": 0, "x": 10.0, "y": 18.0, "group": "dense_landmarks", "name": None},
                {"index": 5, "x": 30.0, "y": 40.0, "group": "dense_landmarks", "name": None},
            ],
            "anchors": {
                "left_eye": {"x": 30.0, "y": 40.0},
                "right_eye": {"x": 70.0, "y": 40.0},
                "nose": {"x": 50.0, "y": 60.0},
                "left_mouth": {"x": 35.0, "y": 80.0},
                "right_mouth": {"x": 65.0, "y": 80.0},
            },
            "named_points": {
                "left_eye_center": {"x": 30.0, "y": 40.0},
                "right_eye_center": {"x": 70.0, "y": 40.0},
                "nose_tip": {"x": 50.0, "y": 60.0},
                "mouth_left": {"x": 35.0, "y": 80.0},
                "mouth_right": {"x": 65.0, "y": 80.0},
                "mouth_center": {"x": 50.0, "y": 80.0},
            },
            "derived_points": {
                "eye_center": {"x": 50.0, "y": 40.0},
                "mouth_center": {"x": 50.0, "y": 80.0},
                "axis_top": {"x": 45.0, "y": 20.0},
                "axis_bottom": {"x": 45.0, "y": 100.0},
            },
            "lines": [
                {
                    "key": "interpupillary_line",
                    "start": "left_eye_center",
                    "end": "right_eye_center",
                    "group": "measurements",
                    "kind": "measurement",
                    "layers": ["measurements", "eyes"],
                    "metric_key": "raw_eye_distance_px",
                    "supported_features": ["raw_eye_distance_px"],
                }
            ],
            "axes": [
                {
                    "key": "facial_midline",
                    "start": "facial_midline_top",
                    "end": "facial_midline_bottom",
                    "group": "symmetry",
                    "kind": "axis",
                    "layers": ["symmetry", "thirds"],
                    "supported_features": ["midline_alignment_index"],
                }
            ],
            "guides": [
                {
                    "key": "thirds_eye_guide",
                    "start": "left_eye_center",
                    "end": "right_eye_center",
                    "group": "thirds",
                    "kind": "guide",
                    "layers": ["thirds"],
                    "metric_key": "middle_third_proxy",
                    "supported_features": ["middle_third_proxy"],
                }
            ],
            "groups": {"eyes": ["left_eye_center", "right_eye_center"]},
            "notes": [],
        },
        "raw_measurements": {"raw_face_width_px": 70.0},
        "normalized_ratios": {"ratio_face_aspect": 0.8},
        "symmetry_metrics": {"symmetry_index": 0.95},
        "angle_metrics": {"angle_eye_line_deg": 1.2},
        "expanded_geometry": {
            "available": True,
            "mediapipe_feature_count": 8,
            "mediapipe_non_null_feature_count": 8,
            "reliability_counts": {"reliable": 5, "proxy": 3},
            "groups": {
                "face_oval": [
                    {
                        "feature_name": "mediapipe_face_oval_aspect_ratio",
                        "value": 1.22,
                        "source": "mediapipe_face_mesh",
                        "reliability": "reliable",
                        "units": "ratio",
                    }
                ]
            },
            "unavailable_without_face_parsing": [{"feature_name": "visible_left_ear_height"}],
        },
        "reference_comparison": {
            "selected_group": {
                "group_key": "sex_group_normalized=Female",
                "group_values": {"sex_group_normalized": "Female"},
                "sample_count": 80,
                "complete_case_count": 80,
            },
            "z_scores": {"ratio_face_aspect": 0.2},
            "mahalanobis_distance": 1.4,
            "percentile_positions": {"ratio_face_aspect": 0.61},
        },
        "feature_level_deviations": {
            "top_contributing_features": [
                {"feature_name": "ratio_face_aspect", "z_score": 0.2, "abs_z_score": 0.2, "direction": "above_mean"}
            ],
            "all_feature_deviations": [
                {"feature_name": "ratio_face_aspect", "z_score": 0.2, "abs_z_score": 0.2, "direction": "above_mean"}
            ],
        },
        "optional_composite_indices": {"symmetry_index": 0.95},
        "transparency": {"formulas": {}, "reference_group_selection": {}},
        "limitations": ["objective morphometric output"],
            "artifacts": {
                "original_image_path": str(project_root / "sample.jpg"),
                "analysis_image_path": str(measurement_image),
                "normalized_frame_image_path": str(normalized_frame_image),
                "aligned_image_path": str(measurement_image),
                "measurement_image_path": str(measurement_image),
                "overlay_image_path": str(measurement_image),
                "annotated_image_path": str(annotated_image),
                "visualization_debug_json_path": str(project_root / "reports" / "examples" / "visualization_debug.json"),
            },
    }
    return InferenceArtifacts(
        report_json_path=report_json,
        report_markdown_path=report_md,
        reporting_summary_path=summary,
        analysis_image_path=measurement_image,
        normalized_frame_image_path=normalized_frame_image,
        aligned_image_path=measurement_image,
        measurement_image_path=measurement_image,
        annotated_image_path=annotated_image,
        report=report,
        reporting_summary={},
    )


def _fake_scorer_artifacts(project_root: Path, *, quality_pass: bool = True) -> ScorerInferenceArtifacts:
    report_json = project_root / "reports" / "examples" / "scorer_report.json"
    report_md = project_root / "reports" / "examples" / "scorer_report.md"
    report = {
        "primary_score": {
            "predicted_score": 8.6079,
            "score_source": "raw_continuous",
            "uses_calibration": True,
            "calibration_used_as_primary": False,
            "calibration_used_for_diagnostics": True,
            "display_score": 8.61,
            "predicted_score_raw": 8.6079,
            "raw_predicted_score": 8.6079,
            "calibrated_predicted_score": 8.5,
            "calibration_method": "isotonic_regression",
            "model": "ridge",
            "model_family": "regression",
            "nearest_score_group": 8.5,
            "distance_to_nearest_group": 0.1079,
            "score_range": [0.0, 10.0],
            "observed_training_label_range": [1.0, 9.5],
        },
        "auxiliary_scores": {
            "pairwise": {"approx_score": 5.49, "nearest_score_group": 5.5, "use_for_absolute_scoring": False},
            "ordinal": {"predicted_score": 8.31, "nearest_score_group": 8.5},
            "classifier": {
                "predicted_group": "9.5",
                "group_probabilities": {"8.5": 0.19, "9.5": 0.41},
                "top_k_groups": [{"group": "9.5", "probability": 0.41}],
            },
            "ensemble": {"predicted_score": 8.44, "nearest_score_group": 8.5, "std": 0.11},
        },
        "regression": {
            "predicted_score": 8.6079,
            "score_source": "raw_continuous",
            "uses_calibration": True,
            "calibration_used_as_primary": False,
            "calibration_used_for_diagnostics": True,
            "display_score": 8.61,
            "predicted_score_raw": 8.6079,
            "raw_predicted_score": 8.6079,
            "calibrated_predicted_score": 8.5,
            "calibration_method": "isotonic_regression",
            "nearest_score_group": 8.5,
            "distance_to_nearest_group": 0.1079,
            "distances_to_groups": {"8.5": 0.1079, "9.0": 0.3921},
            "soft_group_membership": {"8.5": 0.64, "9.0": 0.36},
            "score_range": [0.0, 10.0],
            "observed_training_label_range": [1.0, 9.5],
        },
        "classification": {
            "predicted_group": "9.5",
            "group_probabilities": {"8.5": 0.19, "9.5": 0.41},
            "top_k_groups": [{"group": "9.5", "probability": 0.41}],
        },
        "pairwise": {"approx_score": 5.49, "nearest_score_group": 5.5, "use_for_absolute_scoring": False},
        "ordinal": {"predicted_score": 8.31, "nearest_score_group": 8.5},
        "final_score": {"predicted_score": 8.44, "nearest_score_group": 8.5, "confidence": "medium"},
        "local_explanation": {
            "baseline_score": 6.54,
            "predicted_score": 8.6079,
            "explanation_method": "linear_coefficients",
            "feature_global_weights": [{"feature_name": "symmetry_index", "normalized_weight": 0.22}],
            "top_positive_features": [{"feature": "symmetry_index", "contribution": 0.22, "global_weight": 0.22, "component": "symmetry"}],
            "top_negative_features": [{"feature": "ratio_face_aspect", "contribution": -0.11, "global_weight": 0.13, "component": "face_shape"}],
            "component_contributions": {
                "nose": {"contribution": 0.19, "direction": "positive", "top_features": [{"feature": "ratio_nose_width_to_face_width", "contribution": 0.08}]}
            },
        },
        "learned_weights": {
            "component_weights": [{"component": "cheekbones_midface", "normalized_weight": 0.22, "feature_count": 8}],
            "anatomical_component_weights": [{"component": "cheekbones_midface", "normalized_weight": 0.22, "feature_count": 8}],
            "transform_family_weights": [{"transform_family": "reference_relative", "normalized_weight": 0.55, "feature_count": 50}],
            "source_weights": [{"source_family": "mediapipe_reference_relative", "normalized_weight": 0.55, "feature_count": 50}],
            "top_feature_weights": [{"feature_name": "mediapipe_cheekbone_width_to_jaw_ratio", "normalized_weight": 0.11}],
        },
        "scoring_model_info": {
            "primary_model": "ridge",
            "primary_feature_set": "absolute_plus_best_reference_50",
            "uses_mediapipe": False,
            "uses_reference_relative": True,
            "weight_method": "standardized_coefficients",
            "selected_feature_count": 106,
            "selected_component_counts": {"face_shape": 12, "jaw_chin": 10},
            "selected_transform_family_counts": {"reference_relative": 50, "absolute_geometry": 56},
            "selected_source_counts": {"mediapipe_reference_relative": 50, "mediapipe_direct": 56},
        },
        "component_contributions": {
            "nose": {"contribution": 0.19, "direction": "positive", "top_features": [{"feature": "ratio_nose_width_to_face_width", "contribution": 0.08}]}
        },
        "explanation_method": "linear_coefficients",
        "confidence": {
            "level": "low" if not quality_pass else "medium",
            "reasons": ["quality_gate_failed"] if not quality_pass else [],
        },
        "quality": {
            "quality_pass": quality_pass,
            "processing_status": "processed",
            "processing_error_message": None,
        },
        "evaluation_context": {"test_metrics_available": True},
        "reference_context": {
            "source": "fairface",
            "stats_path": "reports/metrics/reference_stats_x.json",
            "filters": "{\"sex_group_normalized\":\"male\"}",
            "requested_group": "{\"sex_group_normalized\":\"male\",\"age_group_normalized\":\"20-29\"}",
            "resolved_group": "{\"sex_group_normalized\":\"male\"}",
            "fallback_used": True,
            "reference_sample_count": 74,
        },
        "top_contributing_features": [{"feature_name": "ratio_face_aspect", "importance": 0.2}],
        "warnings": ["quality warning"] if not quality_pass else [],
    }
    return ScorerInferenceArtifacts(report_json_path=report_json, report_markdown_path=report_md, report=report)


class ApiTests(unittest.TestCase):
    class _FakeUploadFile:
        def __init__(self, *, filename: str, content_type: str, payload: bytes) -> None:
            self.filename = filename
            self.content_type = content_type
            self._payload = payload

        async def read(self) -> bytes:
            return self._payload

    class _FakeRequest:
        def __init__(self, form_payload: dict) -> None:
            self._form_payload = form_payload

        async def form(self) -> dict:
            return self._form_payload

    def _endpoint(self, app, path: str, method: str):
        for route in app.routes:
            if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
                return route.endpoint
        raise AssertionError(f"Route not found: {method} {path}")

    def test_health_and_reference_groups_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)

            model_dir = project_root / "data" / "interim" / "insightface_models" / "models" / "buffalo_l"
            model_dir.mkdir(parents=True, exist_ok=True)
            for name in ("det_10g.onnx", "2d106det.onnx"):
                (model_dir / name).write_bytes(b"stub")
            template_dir = project_root / "data" / "interim" / "3d_models"
            template_dir.mkdir(parents=True, exist_ok=True)
            (template_dir / "canonical_face_5pt.npy").write_bytes(b"stub")

            reference_dir = project_root / "models" / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "groupings": [["sex_group_normalized"]],
                    "reference_groups": {
                        "sex_group_normalized": {
                            "sex_group_normalized=Female": {
                                "group_values": {"sex_group_normalized": "Female"},
                                "sample_count": 50,
                                "complete_case_count": 50,
                                "feature_columns": ["ratio_face_aspect"],
                            }
                        }
                    },
                },
                reference_dir / "reference_model.joblib",
            )
            pd.DataFrame(
                [
                    {
                        "group_key": "sex_group_normalized=Female",
                        "feature_name": "ratio_face_aspect",
                    }
                ]
            ).to_parquet(reference_dir / "reference_group_statistics.parquet", index=False)

            app = create_app(project_root=project_root)
            health_endpoint = self._endpoint(app, "/health", "GET")
            groups_endpoint = self._endpoint(app, "/reference-groups", "GET")

            health_payload = health_endpoint()
            self.assertTrue(health_payload["can_analyze"])

            payload = groups_endpoint()
            self.assertEqual(payload["reference_group_count"], 1)
            self.assertEqual(payload["groups"][0]["group_key"], "sex_group_normalized=Female")

    def test_analyze_endpoint_uses_shared_service_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)
            app = create_app(project_root=project_root)
            analyze_endpoint = self._endpoint(app, "/analyze", "POST")

            with patch("src.face_aesthetics.api.app.analyze_image", return_value=_fake_inference_artifacts(project_root)):
                payload = analyze_endpoint(
                    AnalyzeRequest(
                        image_path=str(project_root / "sample.jpg"),
                        group="sex_group_normalized=Female",
                        preferred_execution_provider="CPUExecutionProvider",
                    )
                )

            self.assertIn("generated_artifacts", payload)
            self.assertIn("visualization", payload)
            self.assertIn("preview_image", payload)
            self.assertIn("overlay", payload)
            self.assertIn("alignment", payload["input_quality"])
            self.assertIn("analysis_image_path", payload["generated_artifacts"])
            self.assertIn("raw_landmarks", payload["visualization"])
            self.assertIn("semantic_points", payload["visualization"])
            self.assertIn("measurement_overlays", payload["visualization"])
            self.assertIn("mediapipe_landmarks", payload["visualization"])
            self.assertIn("mediapipe_regions", payload["visualization"])
            self.assertIn("detector_bbox", payload["visualization"])
            self.assertIn("display_face_box", payload["visualization"])
            self.assertIn("semantic_face_extents", payload["visualization"])
            self.assertIn("expanded_geometry", payload)
            self.assertTrue(payload["expanded_geometry"]["available"])
            self.assertNotEqual(payload["visualization"]["detector_bbox"], payload["visualization"]["display_face_box"])
            self.assertEqual(
                payload["generated_artifacts"]["analysis_image_path"],
                str(project_root / "reports" / "examples" / "aligned_face.jpg"),
            )
            self.assertEqual(payload["reference_comparison"]["selected_group"]["group_key"], "sex_group_normalized=Female")
            self.assertEqual(payload["overlay"]["active_source"], "mediapipe_only")
            self.assertEqual(payload["overlay"]["legacy_layers"], [])
            self.assertEqual(
                payload["preview_image"]["clean_image_url"],
                payload["generated_artifacts"]["analysis_image_path"],
            )
            overlay_audit_path = project_root / "reports" / "metrics" / "ui_overlay_source_audit.json"
            self.assertTrue(overlay_audit_path.exists())
            overlay_audit = json.loads(overlay_audit_path.read_text())
            self.assertEqual(int(overlay_audit.get("insightface_overlay_count", -1)), 0)
            self.assertEqual(int(overlay_audit.get("legacy_overlay_count", -1)), 0)

    def test_artifact_file_endpoint_serves_project_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)
            app = create_app(project_root=project_root)
            endpoint = self._endpoint(app, "/artifact-file", "GET")

            artifact_path = project_root / "reports" / "examples" / "analysis_image.jpg"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(b"artifact")

            response = endpoint(path=str(artifact_path))
            self.assertEqual(Path(response.path), artifact_path)

    def test_analyze_endpoint_accepts_reference_group_key_style_group_string(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)
            app = create_app(project_root=project_root)
            analyze_endpoint = self._endpoint(app, "/analyze", "POST")

            with patch("src.face_aesthetics.api.app.analyze_image", return_value=_fake_inference_artifacts(project_root)) as analyze_mock:
                analyze_endpoint(
                    AnalyzeRequest(
                        image_path=str(project_root / "sample.jpg"),
                        group="sex_group_normalized=Female__age_group_normalized=20-29__ethnicity_group_normalized=White",
                        preferred_execution_provider="CPUExecutionProvider",
                    )
                )

            self.assertEqual(analyze_mock.call_args.kwargs["group_override"], "sex_group_normalized=Female__age_group_normalized=20-29__ethnicity_group_normalized=White")

    def test_analyze_endpoint_returns_dependency_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)
            app = create_app(project_root=project_root)
            analyze_endpoint = self._endpoint(app, "/analyze", "POST")

            with self.assertRaises(HTTPException) as context:
                analyze_endpoint(AnalyzeRequest(image_path=str(project_root / "missing.jpg")))

            self.assertEqual(context.exception.status_code, 422)
            self.assertEqual(context.exception.detail["error_type"], "validation_error")

    def test_analyze_upload_endpoint_uses_shared_service_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)
            app = create_app(project_root=project_root)
            analyze_upload_endpoint = self._endpoint(app, "/analyze-upload", "POST")

            request = self._FakeRequest(
                {
                    "file": self._FakeUploadFile(
                        filename="sample.jpg",
                        content_type="image/jpeg",
                        payload=b"fake-image-bytes",
                    ),
                    "group": "sex_group_normalized=Female",
                    "preferred_execution_provider": "CPUExecutionProvider",
                    "strict_gpu": "false",
                }
            )
            stored_path = project_root / "data" / "interim" / "runtime_uploads" / "sample.jpg"
            with (
                patch("src.face_aesthetics.api.app._multipart_runtime_available", return_value=True),
                patch("src.face_aesthetics.api.app.persist_uploaded_image", return_value=stored_path),
                patch("src.face_aesthetics.api.app.analyze_image", return_value=_fake_inference_artifacts(project_root)),
            ):
                payload = asyncio.run(analyze_upload_endpoint(request))

            self.assertEqual(payload["input"]["upload_original_filename"], "sample.jpg")
            self.assertEqual(payload["reference_comparison"]["selected_group"]["group_key"], "sex_group_normalized=Female")

    def test_analyze_upload_endpoint_returns_dependency_error_when_multipart_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)
            app = create_app(project_root=project_root)
            analyze_upload_endpoint = self._endpoint(app, "/analyze-upload", "POST")

            with (
                patch("src.face_aesthetics.api.app._multipart_runtime_available", return_value=False),
                self.assertRaises(HTTPException) as context,
            ):
                asyncio.run(analyze_upload_endpoint(self._FakeRequest({})))

            self.assertEqual(context.exception.status_code, 424)
            self.assertEqual(context.exception.detail["error_type"], "dependency_error")

    def test_score_upload_endpoint_returns_score_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)
            app = create_app(project_root=project_root)
            score_upload_endpoint = self._endpoint(app, "/score-upload", "POST")
            request = self._FakeRequest(
                {
                    "file": self._FakeUploadFile(
                        filename="sample.jpg",
                        content_type="image/jpeg",
                        payload=b"fake-image-bytes",
                    )
                }
            )
            stored_path = project_root / "data" / "interim" / "runtime_uploads" / "sample.jpg"
            with (
                patch("src.face_aesthetics.api.app._multipart_runtime_available", return_value=True),
                patch("src.face_aesthetics.api.app.persist_uploaded_image", return_value=stored_path),
                patch("src.face_aesthetics.api.app.score_image", return_value=_fake_scorer_artifacts(project_root)),
            ):
                payload = asyncio.run(score_upload_endpoint(request))
            self.assertEqual(payload["primary_score"]["predicted_score"], 8.6079)
            self.assertEqual(payload["primary_score"]["display_score"], 8.61)
            self.assertEqual(payload["primary_score"]["score_source"], "raw_continuous")
            self.assertTrue(payload["primary_score"]["uses_calibration"])
            self.assertFalse(payload["primary_score"]["calibration_used_as_primary"])
            self.assertEqual(payload["primary_score"]["model"], "ridge")
            self.assertEqual(payload["regression"]["nearest_score_group"], 8.5)
            self.assertEqual(payload["classification"]["predicted_group"], "9.5")
            self.assertIn("local_explanation", payload)
            self.assertIn("component_contributions", payload)
            self.assertIn("scoring_model_info", payload)
            self.assertIn("learned_weights", payload)
            self.assertIn("anatomical_component_weights", payload["learned_weights"])
            self.assertIn("transform_family_weights", payload["learned_weights"])
            self.assertIn("source_weights", payload["learned_weights"])
            self.assertTrue(
                all(
                    str(item.get("component")) != "reference_deviation"
                    for item in payload["learned_weights"].get("anatomical_component_weights", [])
                )
            )
            self.assertIn("selected_transform_family_counts", payload["scoring_model_info"])
            self.assertIn("selected_source_counts", payload["scoring_model_info"])
            self.assertIn("reference_context", payload)
            self.assertTrue(payload["reference_context"]["fallback_used"])
            self.assertIn("generated_artifacts", payload)

    def test_score_upload_endpoint_returns_dependency_error_when_model_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)
            app = create_app(project_root=project_root)
            score_upload_endpoint = self._endpoint(app, "/score-upload", "POST")
            request = self._FakeRequest(
                {
                    "file": self._FakeUploadFile(
                        filename="sample.jpg",
                        content_type="image/jpeg",
                        payload=b"fake-image-bytes",
                    )
                }
            )
            stored_path = project_root / "data" / "interim" / "runtime_uploads" / "sample.jpg"
            with (
                patch("src.face_aesthetics.api.app._multipart_runtime_available", return_value=True),
                patch("src.face_aesthetics.api.app.persist_uploaded_image", return_value=stored_path),
                patch("src.face_aesthetics.api.app.score_image", side_effect=ServiceDependencyError("scorer model missing")),
                self.assertRaises(HTTPException) as context,
            ):
                asyncio.run(score_upload_endpoint(request))
            self.assertEqual(context.exception.status_code, 424)
            self.assertEqual(context.exception.detail["error_type"], "dependency_error")

    def test_score_upload_endpoint_keeps_low_confidence_when_quality_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)
            app = create_app(project_root=project_root)
            score_upload_endpoint = self._endpoint(app, "/score-upload", "POST")
            request = self._FakeRequest(
                {
                    "file": self._FakeUploadFile(
                        filename="sample.jpg",
                        content_type="image/jpeg",
                        payload=b"fake-image-bytes",
                    )
                }
            )
            stored_path = project_root / "data" / "interim" / "runtime_uploads" / "sample.jpg"
            with (
                patch("src.face_aesthetics.api.app._multipart_runtime_available", return_value=True),
                patch("src.face_aesthetics.api.app.persist_uploaded_image", return_value=stored_path),
                patch("src.face_aesthetics.api.app.score_image", return_value=_fake_scorer_artifacts(project_root, quality_pass=False)),
            ):
                payload = asyncio.run(score_upload_endpoint(request))
            self.assertEqual(payload["confidence"]["level"], "low")
            self.assertIn("quality_gate_failed", payload["confidence"]["reasons"])

    def test_score_upload_endpoint_allows_missing_local_explanation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)
            app = create_app(project_root=project_root)
            score_upload_endpoint = self._endpoint(app, "/score-upload", "POST")
            request = self._FakeRequest(
                {
                    "file": self._FakeUploadFile(
                        filename="sample.jpg",
                        content_type="image/jpeg",
                        payload=b"fake-image-bytes",
                    )
                }
            )
            stored_path = project_root / "data" / "interim" / "runtime_uploads" / "sample.jpg"
            payload_no_explanation = _fake_scorer_artifacts(project_root).report
            payload_no_explanation["local_explanation"] = None
            payload_no_explanation["component_contributions"] = {}
            payload_no_explanation["warnings"] = ["explanation_unavailable"]
            with (
                patch("src.face_aesthetics.api.app._multipart_runtime_available", return_value=True),
                patch("src.face_aesthetics.api.app.persist_uploaded_image", return_value=stored_path),
                patch(
                    "src.face_aesthetics.api.app.score_image",
                    return_value=ScorerInferenceArtifacts(
                        report_json_path=project_root / "reports" / "examples" / "scorer_report.json",
                        report_markdown_path=project_root / "reports" / "examples" / "scorer_report.md",
                        report=payload_no_explanation,
                    ),
                ),
            ):
                payload = asyncio.run(score_upload_endpoint(request))
            self.assertIsNone(payload["local_explanation"])
            self.assertIn("explanation_unavailable", payload["warnings"])


if __name__ == "__main__":
    unittest.main()
