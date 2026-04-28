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
            "bbox": [10.0, 20.0, 80.0, 100.0],
            "raw_landmarks": [
                {"index": 0, "x": 10.0, "y": 18.0, "group": "dense_landmarks", "name": None, "source_type": "raw", "source_indices": [0], "description": "raw"},
                {"index": 5, "x": 30.0, "y": 40.0, "group": "dense_landmarks", "name": None, "source_type": "raw", "source_indices": [5], "description": "raw"},
            ],
            "raw_landmarks_debug": [
                {"index": 0, "x": 10.0, "y": 18.0, "group": "dense_landmarks", "name": None, "source_type": "raw", "source_indices": [0], "description": "raw"},
            ],
            "semantic_points": {
                "left_eye_center": {"x": 30.0, "y": 40.0},
                "right_eye_center": {"x": 70.0, "y": 40.0},
                "nose_tip": {"x": 50.0, "y": 60.0},
                "mouth_left": {"x": 35.0, "y": 80.0},
                "mouth_right": {"x": 65.0, "y": 80.0},
                "mouth_center": {"x": 50.0, "y": 80.0},
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
            self.assertIn("alignment", payload["input_quality"])
            self.assertIn("analysis_image_path", payload["generated_artifacts"])
            self.assertIn("raw_landmarks", payload["visualization"])
            self.assertIn("semantic_points", payload["visualization"])
            self.assertIn("measurement_overlays", payload["visualization"])
            self.assertEqual(
                payload["generated_artifacts"]["analysis_image_path"],
                str(project_root / "reports" / "examples" / "aligned_face.jpg"),
            )
            self.assertEqual(payload["reference_comparison"]["selected_group"]["group_key"], "sex_group_normalized=Female")

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


if __name__ == "__main__":
    unittest.main()
