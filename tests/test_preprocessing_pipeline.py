from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.face_aesthetics.data.io import write_parquet
from unittest.mock import patch

from src.face_aesthetics.pipelines.preprocessing import _evaluate_quality, preprocess_single_image, run_preprocessing
from src.face_aesthetics.services.preprocessing import (
    DetectedFace,
    InsightFaceDetector,
    _derive_five_point_landmarks,
    compute_auto_alignment_rotation,
    estimate_eye_line_angle_degrees,
    _normalize_pose_angles,
    estimate_pose_from_landmarks,
    select_detection_with_auto_alignment,
)


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc`\x00\x00\x00\x02\x00\x01"
    b"\xe2!\xbc3"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

MANIFEST_COLUMNS = [
    "dataset_name",
    "sample_id",
    "subject_id",
    "image_path",
    "split",
    "raw_labels",
    "normalized_labels",
    "quality_gate_status",
    "adapter_version",
    "image_id",
]


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_BYTES)


def _write_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_minimal_configs(project_root: Path, *, model_root: str) -> None:
    configs_dir = project_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    _write_config(
        configs_dir / "base.yaml",
        {
            "project": {"name": "test-morphometric-analysis"},
            "paths": {
                "project_root": ".",
                "data_dir": "data",
                "raw_data_dir": "data/raw",
                "interim_data_dir": "data/interim",
                "processed_data_dir": "data/processed",
                "features_data_dir": "data/features",
                "reports_dir": "reports",
                "source_dir": "src",
                "tests_dir": "tests",
            },
        },
    )
    _write_config(
        configs_dir / "data.yaml",
        {
            "dataset": {
                "default_reference_model": "default_morphometric_reference",
            }
        },
    )
    _write_config(
        configs_dir / "preprocess.yaml",
        {
            "preprocess": {
                "enabled": True,
                "input_manifest_path": "data/interim/dataset_manifest.parquet",
                "output_dir": "data/processed",
                "aligned_dir": "data/processed/aligned_faces",
                "output_manifest_path": "data/processed/preprocessed_manifest.parquet",
                "summary_path": "reports/metrics/preprocessing_summary.json",
                "progress_path": "data/processed/preprocessing_progress.json",
                "face_backend": "insightface",
                "target_image_size": [224, 224],
                "color_mode": "rgb",
                "save_debug_images": False,
                "runtime": {
                    "batch_size": 8,
                    "limit": None,
                    "offset": 0,
                    "resume": True,
                    "overwrite": False,
                    "num_workers": 1,
                    "preferred_execution_provider": "CUDAExecutionProvider",
                    "fallback_execution_provider": "CPUExecutionProvider",
                    "device_id": 0,
                    "smoke_mode": False,
                    "smoke_limit": 4,
                },
                "insightface": {
                    "model_name": "buffalo_l",
                    "model_root": model_root,
                    "allowed_modules": ["detection", "landmark_2d_106"],
                    "required_model_files": ["det_10g.onnx", "2d106det.onnx"],
                    "providers": [],
                    "preferred_execution_provider": "CUDAExecutionProvider",
                    "fallback_execution_provider": "CPUExecutionProvider",
                    "device_id": 0,
                    "ctx_id": -1,
                    "det_size": [640, 640],
                },
                "quality_gate": {
                    "require_single_face": True,
                    "min_bbox_size": 64,
                    "min_detector_score": 0.5,
                    "min_blur_variance": 80.0,
                    "max_abs_yaw": 35.0,
                    "max_abs_pitch": 25.0,
                    "max_abs_roll": 35.0,
                    "max_occlusion_score": 0.6,
                },
            }
        },
    )
    for name in (
        "features.yaml",
        "train_baseline.yaml",
        "train_geometry.yaml",
        "train_fusion.yaml",
        "infer.yaml",
        "robustness.yaml",
    ):
        _write_config(configs_dir / name, {})


class PreprocessingPipelineTests(unittest.TestCase):
    class _FaceAlignStub:
        def norm_crop(self, image, landmark, image_size):
            return np.zeros((image_size, image_size, 3), dtype=np.uint8)

    class _DetectorStub:
        def __init__(self, responses):
            self._responses = list(responses)
            self._face_align = PreprocessingPipelineTests._FaceAlignStub()

        def detect(self, image_bgr, target_image_size):
            if not self._responses:
                return []
            response = self._responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    def _detection_with_eye_tilt(self, tilt_deg: float, *, roll: float | None = None) -> DetectedFace:
        eye_center = np.asarray([100.0, 80.0], dtype=np.float64)
        mouth_center = np.asarray([100.0, 150.0], dtype=np.float64)
        eye_distance = 60.0
        mouth_width = 50.0
        radians = np.deg2rad(tilt_deg)
        eye_offset = np.asarray(
            [
                math.cos(radians) * (eye_distance / 2.0),
                math.sin(radians) * (eye_distance / 2.0),
            ],
            dtype=np.float64,
        )
        mouth_offset = np.asarray(
            [
                math.cos(radians) * (mouth_width / 2.0),
                math.sin(radians) * (mouth_width / 2.0),
            ],
            dtype=np.float64,
        )
        left_eye = eye_center - eye_offset
        right_eye = eye_center + eye_offset
        left_mouth = mouth_center - mouth_offset
        right_mouth = mouth_center + mouth_offset
        return DetectedFace(
            bbox=[50.0, 40.0, 150.0, 190.0],
            landmarks=[
                [float(left_eye[0]), float(left_eye[1])],
                [float(right_eye[0]), float(right_eye[1])],
                [100.0, 115.0],
                [float(left_mouth[0]), float(left_mouth[1])],
                [float(right_mouth[0]), float(right_mouth[1])],
            ],
            landmarks_dense=None,
            det_score=0.99,
            aligned_image=np.zeros((224, 224, 3), dtype=np.uint8),
            yaw=0.0,
            pitch=0.0,
            roll=tilt_deg if roll is None else roll,
            occlusion_score=0.1,
            detection_backend="stub",
        )

    def test_normalize_pose_angles_folds_wraparound_pitch_into_signed_range(self) -> None:
        yaw, pitch, roll = _normalize_pose_angles(1.5, 168.0, -2.0)

        self.assertAlmostEqual(yaw, 1.5)
        self.assertAlmostEqual(pitch, -12.0)
        self.assertAlmostEqual(roll, -2.0)

    def test_quality_gate_accepts_frontal_and_moderate_pitch(self) -> None:
        frontal_detection = DetectedFace(
            bbox=[10.0, 10.0, 120.0, 150.0],
            landmarks=[[20.0, 20.0]] * 5,
            landmarks_dense=None,
            det_score=0.99,
            aligned_image=None,
            yaw=1.0,
            pitch=-12.0,
            roll=2.0,
            occlusion_score=0.1,
            detection_backend="stub",
        )
        moderate_detection = DetectedFace(
            bbox=[10.0, 10.0, 120.0, 150.0],
            landmarks=[[20.0, 20.0]] * 5,
            landmarks_dense=None,
            det_score=0.99,
            aligned_image=None,
            yaw=3.0,
            pitch=22.0,
            roll=4.0,
            occlusion_score=0.1,
            detection_backend="stub",
        )
        quality_gate = {
            "require_single_face": True,
            "min_bbox_size": 64,
            "min_detector_score": 0.5,
            "min_blur_variance": 0.0,
            "max_abs_yaw": 35.0,
            "max_abs_pitch": 25.0,
            "max_abs_roll": 35.0,
            "max_occlusion_score": 0.6,
        }

        frontal_reasons = _evaluate_quality(frontal_detection, 1, quality_gate)
        moderate_reasons = _evaluate_quality(moderate_detection, 1, quality_gate)

        self.assertNotIn("pitch_extreme", frontal_reasons)
        self.assertNotIn("pitch_extreme", moderate_reasons)

    def test_quality_gate_rejects_real_extreme_pitch(self) -> None:
        detection = DetectedFace(
            bbox=[10.0, 10.0, 120.0, 150.0],
            landmarks=[[20.0, 20.0]] * 5,
            landmarks_dense=None,
            det_score=0.99,
            aligned_image=None,
            yaw=2.0,
            pitch=41.0,
            roll=1.0,
            occlusion_score=0.1,
            detection_backend="stub",
        )
        quality_gate = {
            "require_single_face": True,
            "min_bbox_size": 64,
            "min_detector_score": 0.5,
            "min_blur_variance": 0.0,
            "max_abs_yaw": 35.0,
            "max_abs_pitch": 25.0,
            "max_abs_roll": 35.0,
            "max_occlusion_score": 0.6,
        }

        reasons = _evaluate_quality(detection, 1, quality_gate)

        self.assertIn("pitch_extreme", reasons)

    def test_pose_estimation_returns_interpretable_pitch_for_frontal_landmarks(self) -> None:
        landmarks = [
            [60.0, 70.0],
            [140.0, 70.0],
            [100.0, 104.0],
            [72.0, 138.0],
            [128.0, 138.0],
        ]

        yaw, pitch, roll = estimate_pose_from_landmarks((224, 224, 3), landmarks)

        self.assertIsNotNone(yaw)
        self.assertIsNotNone(pitch)
        self.assertIsNotNone(roll)
        self.assertLess(abs(float(pitch)), 25.0)

    def test_dense_landmark_mouth_extraction_avoids_jawline_extremes(self) -> None:
        dense_points = [
            [40.0, 55.0], [55.0, 52.0], [70.0, 51.0], [130.0, 51.0], [145.0, 52.0], [160.0, 55.0],
            [100.0, 95.0], [95.0, 108.0], [105.0, 108.0],
            [28.0, 150.0], [38.0, 168.0], [52.0, 182.0], [148.0, 182.0], [162.0, 168.0], [172.0, 150.0],
            [72.0, 142.0], [84.0, 138.0], [96.0, 136.0], [104.0, 136.0], [116.0, 138.0], [128.0, 142.0],
            [74.0, 150.0], [86.0, 154.0], [100.0, 156.0], [114.0, 154.0], [126.0, 150.0],
        ]
        detector_kps = [
            [68.0, 60.0],
            [132.0, 60.0],
            [100.0, 104.0],
            [78.0, 146.0],
            [122.0, 146.0],
        ]

        landmarks = _derive_five_point_landmarks(
            dense_points,
            [20.0, 30.0, 180.0, 210.0],
            detector_landmarks=detector_kps,
        )

        left_mouth = np.asarray(landmarks[3], dtype=np.float64)
        right_mouth = np.asarray(landmarks[4], dtype=np.float64)
        mouth_center = (left_mouth + right_mouth) / 2.0

        self.assertGreater(left_mouth[0], 60.0)
        self.assertLess(right_mouth[0], 140.0)
        self.assertLess(left_mouth[0], mouth_center[0])
        self.assertGreater(right_mouth[0], mouth_center[0])
        self.assertLess(abs(float(mouth_center[0] - 100.0)), 12.0)
        self.assertLess(abs(float(left_mouth[1] - right_mouth[1])), 12.0)

    def test_derive_five_point_prefers_canonical_106_mouth_indices_when_available(self) -> None:
        dense_points = np.tile(np.asarray([[100.0, 100.0]], dtype=np.float32), (106, 1))
        dense_points[33:43] = np.asarray([[88.0 + i, 82.0 + (i % 2)] for i in range(10)], dtype=np.float32)
        dense_points[87:97] = np.asarray([[102.0 + i, 82.0 + (i % 2)] for i in range(10)], dtype=np.float32)
        dense_points[52] = np.asarray([80.0, 150.0], dtype=np.float32)
        dense_points[53] = np.asarray([100.0, 156.0], dtype=np.float32)
        dense_points[61] = np.asarray([120.0, 150.0], dtype=np.float32)
        dense_points[71] = np.asarray([100.0, 146.0], dtype=np.float32)
        dense_points[78] = np.asarray([92.0, 132.0], dtype=np.float32)
        dense_points[84] = np.asarray([108.0, 132.0], dtype=np.float32)
        dense_points[86] = np.asarray([100.0, 112.0], dtype=np.float32)
        detector_kps = [
            [90.0, 82.0],
            [110.0, 82.0],
            [100.0, 112.0],
            [24.0, 164.0],
            [176.0, 164.0],
        ]

        landmarks = _derive_five_point_landmarks(
            dense_points.tolist(),
            [20.0, 30.0, 180.0, 210.0],
            detector_landmarks=detector_kps,
        )

        self.assertAlmostEqual(float(landmarks[3][0]), 80.0, places=4)
        self.assertAlmostEqual(float(landmarks[4][0]), 120.0, places=4)
        self.assertAlmostEqual(float(landmarks[3][1]), 150.0, places=4)
        self.assertAlmostEqual(float(landmarks[4][1]), 150.0, places=4)

    def test_insightface_detect_prefers_landmark_2d_106_over_detector_kps_for_mouth(self) -> None:
        class _FaceAlignStub:
            def norm_crop(self, image_bgr, landmark, image_size):
                return np.zeros((image_size, image_size, 3), dtype=np.uint8)

        class _FaceStub:
            def __init__(self) -> None:
                self.bbox = np.asarray([20.0, 30.0, 180.0, 210.0], dtype=np.float32)
                self.kps = np.asarray(
                    [
                        [68.0, 60.0],
                        [132.0, 60.0],
                        [100.0, 104.0],
                        [28.0, 150.0],
                        [172.0, 150.0],
                    ],
                    dtype=np.float32,
                )
                self.landmark_2d_106 = np.asarray(
                    [
                        [40.0, 55.0], [55.0, 52.0], [70.0, 51.0], [130.0, 51.0], [145.0, 52.0], [160.0, 55.0],
                        [100.0, 95.0], [95.0, 108.0], [105.0, 108.0],
                        [28.0, 150.0], [38.0, 168.0], [52.0, 182.0], [148.0, 182.0], [162.0, 168.0], [172.0, 150.0],
                        [72.0, 142.0], [84.0, 138.0], [96.0, 136.0], [104.0, 136.0], [116.0, 138.0], [128.0, 142.0],
                        [74.0, 150.0], [86.0, 154.0], [100.0, 156.0], [114.0, 154.0], [126.0, 150.0],
                    ],
                    dtype=np.float32,
                )
                self.det_score = 0.99

        class _AppStub:
            def get(self, image_bgr):
                return [_FaceStub()]

        detector = InsightFaceDetector.__new__(InsightFaceDetector)
        detector._app = _AppStub()
        detector._face_align = _FaceAlignStub()
        detector._fallback_detector_enabled = False

        detections = detector.detect(np.zeros((256, 256, 3), dtype=np.uint8), [224, 224])

        self.assertEqual(len(detections), 1)
        left_mouth = detections[0].landmarks[3]
        right_mouth = detections[0].landmarks[4]
        self.assertGreater(left_mouth[0], 60.0)
        self.assertLess(right_mouth[0], 140.0)
        self.assertEqual(len(detections[0].landmarks_dense or []), 26)

    def test_auto_alignment_skips_small_tilt(self) -> None:
        detection = self._detection_with_eye_tilt(1.5)

        rotation_deg, source = compute_auto_alignment_rotation(
            detection,
            {
                "enabled": True,
                "min_auto_rotation_deg": 3.0,
                "max_auto_rotation_deg": 20.0,
                "alignment_source": "auto",
            },
        )

        self.assertIsNone(rotation_deg)
        self.assertEqual(source, "eye_line")

    def test_auto_alignment_reduces_eye_line_tilt_and_improves_quality(self) -> None:
        original = self._detection_with_eye_tilt(12.0, roll=12.0)
        corrected = self._detection_with_eye_tilt(1.0, roll=1.0)
        detector = self._DetectorStub([[original], [corrected]])
        image = np.zeros((220, 220, 3), dtype=np.uint8)
        quality_gate = {
            "require_single_face": True,
            "min_bbox_size": 64,
            "min_detector_score": 0.5,
            "min_blur_variance": 0.0,
            "max_abs_yaw": 35.0,
            "max_abs_pitch": 25.0,
            "max_abs_roll": 10.0,
            "max_occlusion_score": 0.6,
        }

        detections, primary, quality_reasons, outcome, _analysis_image = select_detection_with_auto_alignment(
            detector=detector,
            image_bgr=image,
            target_image_size=[224, 224],
            quality_gate=quality_gate,
            auto_alignment_config={
                "enabled": True,
                "min_auto_rotation_deg": 3.0,
                "max_auto_rotation_deg": 20.0,
                "alignment_source": "eye_line",
                "reprocess_after_alignment": True,
            },
            quality_evaluator=_evaluate_quality,
        )

        self.assertEqual(len(detections), 1)
        self.assertIsNotNone(primary)
        assert primary is not None
        self.assertTrue(outcome.applied)
        self.assertTrue(outcome.improved_quality)
        self.assertNotIn("roll_extreme", quality_reasons)
        self.assertLess(
            abs(float(estimate_eye_line_angle_degrees(primary.landmarks))),
            abs(float(estimate_eye_line_angle_degrees(original.landmarks))),
        )

    def test_auto_alignment_gracefully_falls_back_when_realignment_fails(self) -> None:
        original = self._detection_with_eye_tilt(10.0, roll=10.0)
        detector = self._DetectorStub([[original], RuntimeError("realignment failed")])

        detections, primary, quality_reasons, outcome, _analysis_image = select_detection_with_auto_alignment(
            detector=detector,
            image_bgr=np.zeros((220, 220, 3), dtype=np.uint8),
            target_image_size=[224, 224],
            quality_gate={
                "require_single_face": True,
                "min_bbox_size": 64,
                "min_detector_score": 0.5,
                "min_blur_variance": 0.0,
                "max_abs_yaw": 35.0,
                "max_abs_pitch": 25.0,
                "max_abs_roll": 35.0,
                "max_occlusion_score": 0.6,
            },
            auto_alignment_config={
                "enabled": True,
                "min_auto_rotation_deg": 3.0,
                "max_auto_rotation_deg": 20.0,
                "alignment_source": "eye_line",
                "reprocess_after_alignment": True,
            },
            quality_evaluator=_evaluate_quality,
        )

        self.assertEqual(len(detections), 1)
        self.assertIs(primary, original)
        self.assertEqual(quality_reasons, [])
        self.assertTrue(outcome.attempted)
        self.assertFalse(outcome.applied)
        self.assertTrue(outcome.status.startswith("failed:"))

    def test_single_image_preprocessing_uses_aligned_face_as_measurement_space_when_redetection_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            image_path = project_root / "input.png"
            _write_image(image_path)

            configs = {
                "preprocess": {
                    "preprocess": {
                        "insightface": {
                            "model_name": "buffalo_l",
                            "model_root": "data/interim/insightface_models",
                        },
                        "gpu_runtime_check": {
                            "test_model_file": "det_10g.onnx",
                        },
                        "quality_gate": {
                            "require_single_face": True,
                            "min_bbox_size": 32,
                            "min_detector_score": 0.5,
                            "min_blur_variance": 0.0,
                            "max_abs_yaw": 35.0,
                            "max_abs_pitch": 25.0,
                            "max_abs_roll": 35.0,
                            "max_occlusion_score": 0.6,
                        },
                        "runtime": {
                            "preferred_execution_provider": "CPUExecutionProvider",
                            "fallback_execution_provider": "CPUExecutionProvider",
                            "device_id": 0,
                            "strict_gpu_mode": False,
                        },
                        "auto_alignment": {
                            "enabled": False,
                        },
                        "target_image_size": [224, 224],
                    }
                },
                "data": {"dataset": {"default_reference_model": "default_morphometric_reference"}},
            }

            original_detection = DetectedFace(
                bbox=[30.0, 40.0, 180.0, 210.0],
                landmarks=[
                    [70.0, 90.0],
                    [140.0, 90.0],
                    [105.0, 130.0],
                    [78.0, 170.0],
                    [132.0, 170.0],
                ],
                landmarks_dense=None,
                det_score=0.98,
                aligned_image=np.zeros((224, 224, 3), dtype=np.uint8),
                yaw=2.0,
                pitch=1.0,
                roll=3.0,
                occlusion_score=0.1,
                detection_backend="stub",
            )
            aligned_detection = DetectedFace(
                bbox=[12.0, 16.0, 212.0, 216.0],
                landmarks=[
                    [70.0, 82.0],
                    [154.0, 82.0],
                    [112.0, 120.0],
                    [82.0, 160.0],
                    [142.0, 160.0],
                ],
                landmarks_dense=None,
                det_score=0.99,
                aligned_image=np.zeros((224, 224, 3), dtype=np.uint8),
                yaw=0.5,
                pitch=0.0,
                roll=0.8,
                occlusion_score=0.05,
                detection_backend="stub_aligned",
            )

            class _SingleImageDetectorStub:
                def __init__(self) -> None:
                    self._responses = [[[original_detection]], [[aligned_detection]]]

                def diagnostics_payload(self) -> dict:
                    return {"selected_execution_provider": "CPUExecutionProvider"}

                def detect(self, image_bgr, target_image_size):
                    if not self._responses:
                        return []
                    return self._responses.pop(0)[0]

            with patch("src.face_aesthetics.pipelines.preprocessing.load_named_configs", return_value=configs), patch(
                "src.face_aesthetics.pipelines.preprocessing._build_detector",
                return_value=_SingleImageDetectorStub(),
            ), patch(
                "src.face_aesthetics.pipelines.preprocessing.run_gpu_runtime_check",
                return_value={
                    "available_providers": ["CPUExecutionProvider"],
                    "selected_execution_provider": "CPUExecutionProvider",
                    "fallback_to_cpu": False,
                    "device_id": 0,
                    "actual_model_providers": {"detection": ["CPUExecutionProvider"]},
                },
            ), patch(
                "src.face_aesthetics.pipelines.preprocessing.load_image",
                return_value=np.zeros((256, 256, 3), dtype=np.uint8),
            ):
                artifacts = preprocess_single_image(
                    image_path=image_path,
                    project_root=project_root,
                    output_analysis_path=project_root / "analysis_image.jpg",
                    output_aligned_path=project_root / "aligned_face.jpg",
                )

            self.assertEqual(artifacts.measurement_image_role, "aligned_face")
            self.assertEqual(json.loads(artifacts.row["bbox"]), aligned_detection.bbox)
            self.assertEqual(json.loads(artifacts.row["landmarks"]), aligned_detection.landmarks)
            self.assertEqual(artifacts.measurement_image_path, project_root / "aligned_face.jpg")

    def test_empty_manifest_writes_empty_outputs_without_model_init(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root, model_root="data/interim/missing_models")

            manifest_path = project_root / "data" / "interim" / "dataset_manifest.parquet"
            write_parquet([], manifest_path, MANIFEST_COLUMNS)

            artifacts = run_preprocessing(project_root=project_root)

            self.assertTrue(artifacts.output_manifest_path.exists())
            self.assertTrue(artifacts.output_manifest_path.is_dir())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertEqual(artifacts.summary["input_total_rows"], 0)
            self.assertEqual(artifacts.summary["processed_rows"], 0)
            self.assertEqual(artifacts.summary["error_rows"], 0)

            self.assertEqual(list(artifacts.output_manifest_path.glob("*.parquet")), [])

    def test_missing_insightface_bundle_fails_clearly_for_non_empty_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root, model_root="data/interim/missing_models")

            image_path = project_root / "data" / "raw" / "generic" / "sample_001.png"
            _write_image(image_path)

            manifest_path = project_root / "data" / "interim" / "dataset_manifest.parquet"
            write_parquet(
                [
                    {
                        "dataset_name": "generic",
                        "sample_id": "sample_001",
                        "subject_id": "sample_001",
                        "image_path": "data/raw/generic/sample_001.png",
                        "split": "train",
                        "raw_labels": "{}",
                        "normalized_labels": "{}",
                        "quality_gate_status": "pending_preprocessing",
                        "adapter_version": "generic_image_folder@1",
                        "image_id": "generic::sample_001",
                    }
                ],
                manifest_path,
                MANIFEST_COLUMNS,
            )

            with self.assertRaises(FileNotFoundError) as context:
                run_preprocessing(project_root=project_root)

            self.assertIn("InsightFace model bundle was not found", str(context.exception))
            self.assertIn("data/interim/missing_models/models/buffalo_l", str(context.exception))


if __name__ == "__main__":
    unittest.main()
