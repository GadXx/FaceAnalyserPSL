from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.face_aesthetics.features.extraction import feature_registry_rows
from src.face_aesthetics.landmarks.fusion import build_landmark_capability_report, write_landmark_capability_artifacts
from src.face_aesthetics.landmarks import mediapipe_source
from src.face_aesthetics.landmarks.mediapipe_source import build_mediapipe_source_result
from src.face_aesthetics.scoring.common import scoring_feature_columns


def _write_minimal_config_tree(project_root: Path) -> None:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "base.yaml").write_text("{}", encoding="utf-8")
    (config_dir / "data.yaml").write_text("{}", encoding="utf-8")
    (config_dir / "preprocess.yaml").write_text("{}", encoding="utf-8")
    (config_dir / "features.yaml").write_text(
        json.dumps({"features": {}}),
        encoding="utf-8",
    )
    (config_dir / "landmarks.yaml").write_text(
        json.dumps(
            {
                "landmarks": {
                    "report_paths": {
                        "capability_report_path": "reports/metrics/landmark_capability_report.json",
                        "mediapipe_runtime_diagnostics_path": "reports/metrics/mediapipe_runtime_diagnostics.json",
                        "mediapipe_debug_landmarks_path": "reports/debug/mediapipe_landmarks.json",
                    },
                    "landmark_sources": {
                        "insightface_106": {"enabled": True, "required": True},
                        "mediapipe_face_mesh": {
                            "enabled": False,
                            "optional": True,
                            "delegate": "auto",
                            "fallback_to_cpu": True,
                            "model_asset_path": "data/interim/mediapipe/face_landmarker.task",
                        },
                        "face_parsing": {"enabled": False, "optional": True},
                        "optional_3d_mesh": {"enabled": False, "optional": True},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "scoring.yaml").write_text(
        json.dumps(
            {
                "scoring": {
                    "include_feature_families": ["normalized", "symmetry", "angles"],
                    "exclude_feature_patterns": ["raw_*", "*_px"],
                    "feature_exclude_columns": ["philtrum_width_proxy"],
                    "allow_raw_pixel_features": False,
                    "allow_embeddings": False,
                    "allow_metadata_features": False,
                }
            }
        ),
        encoding="utf-8",
    )


class LandmarkSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        mediapipe_source._TASKS_LANDMARKER_CACHE.clear()

    def _fake_runtime_mp(self):
        class _FakeLandmark:
            def __init__(self, x: float, y: float, z: float = 0.0) -> None:
                self.x = x
                self.y = y
                self.z = z

        class _FakeFaceLandmarker:
            def __init__(self, count: int = 478) -> None:
                self._count = count

            def detect(self, _image: object):
                points = [_FakeLandmark(0.5, 0.5, 0.0) for _ in range(self._count)]
                return type("Result", (), {"face_landmarks": [points]})()

            def close(self) -> None:
                return None

        fake_mp = type(
            "FakeMediaPipe",
            (),
            {
                "ImageFormat": type("ImageFormat", (), {"SRGB": "SRGB"})(),
                "Image": lambda *args, **kwargs: object(),
                "tasks": type("Tasks", (), {"BaseOptions": type("BaseOptions", (), {"Delegate": type("Delegate", (), {"CPU": "CPU", "GPU": "GPU"})})})(),
            },
        )()
        return fake_mp, _FakeFaceLandmarker

    def test_capability_report_created_without_optional_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_config_tree(project_root)
            dense = np.zeros((106, 2), dtype=np.float64)
            report = build_landmark_capability_report(
                project_root=project_root,
                image_bgr=None,
                landmarks_dense=dense,
            )
            self.assertIn("sources", report)
            self.assertTrue(report["sources"]["insightface_106"]["available"])
            self.assertFalse(report["sources"]["mediapipe_face_mesh"]["available"])
            self.assertEqual(report["ears"]["status"], "unavailable_without_extra_model")
            written = write_landmark_capability_artifacts(project_root=project_root, capability_report=report)
            self.assertTrue((project_root / written["capability_report_path"]).exists())

    def test_unavailable_ear_features_are_not_invented(self) -> None:
        rows = feature_registry_rows()
        ear_rows = {row["feature_name"]: row for row in rows if "ear" in str(row["feature_name"])}
        self.assertIn("visible_left_ear_height", ear_rows)
        self.assertEqual(ear_rows["visible_left_ear_height"]["reliability"], "unavailable_without_extra_model")
        self.assertFalse(bool(ear_rows["visible_left_ear_height"]["recommended_for_scorer"]))

    def test_mediapipe_feature_registry_metadata_present(self) -> None:
        rows = {row["feature_name"]: row for row in feature_registry_rows()}
        self.assertEqual(rows["mediapipe_face_oval_aspect_ratio"]["source"], "mediapipe_face_mesh")
        self.assertEqual(rows["mediapipe_face_oval_aspect_ratio"]["reliability"], "reliable")
        self.assertEqual(rows["mediapipe_cheekbone_width_proxy"]["reliability"], "proxy")
        self.assertFalse(bool(rows["mediapipe_cheekbone_width_proxy"]["recommended_for_scorer"]))

    def test_scorer_excludes_unstable_and_optional_multi_source_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_config_tree(project_root)
            selected = scoring_feature_columns(
                {
                    "include_feature_families": ["normalized", "symmetry", "angles"],
                    "exclude_feature_patterns": ["raw_*", "*_px"],
                    "feature_exclude_columns": ["philtrum_width_proxy"],
                    "allow_raw_pixel_features": False,
                    "allow_embeddings": False,
                    "allow_metadata_features": False,
                }
            )
            self.assertNotIn("philtrum_width_proxy", selected)
            self.assertNotIn("visible_left_ear_height", selected)
            self.assertNotIn("mediapipe_cheekbone_width_proxy", selected)
            self.assertIn("ratio_face_aspect", selected)

    def test_mediapipe_source_returns_landmarks_when_dependency_available(self) -> None:
        class _FakeLandmark:
            def __init__(self, x: float, y: float, z: float = 0.0) -> None:
                self.x = x
                self.y = y
                self.z = z

        class _FakeFaceLandmarks:
            def __init__(self) -> None:
                self.landmark = [_FakeLandmark(0.5, 0.5, 0.0) for _ in range(478)]

        class _FakeFaceMesh:
            def __init__(self, **_: object) -> None:
                pass

            def process(self, _image: np.ndarray):
                return type("Result", (), {"multi_face_landmarks": [_FakeFaceLandmarks()]})()

            def close(self) -> None:
                return None

        fake_mp = type(
            "FakeMediaPipe",
            (),
            {
                "solutions": type(
                    "Solutions",
                    (),
                    {"face_mesh": type("FaceMeshModule", (), {"FaceMesh": _FakeFaceMesh})()},
                )()
            },
        )()

        with mock.patch("src.face_aesthetics.landmarks.mediapipe_source._import_mediapipe", return_value=(fake_mp, None)):
            result = build_mediapipe_source_result(
                image_bgr=np.zeros((64, 64, 3), dtype=np.uint8),
                config={
                    "enabled": True,
                    "static_image_mode": True,
                    "max_num_faces": 1,
                    "refine_landmarks": True,
                    "min_detection_confidence": 0.5,
                    "min_tracking_confidence": 0.5,
                },
            )
        self.assertTrue(result.available)
        self.assertEqual(result.landmark_count, 478)
        self.assertIn("dense_face_oval", result.available_regions)

    def test_mediapipe_gpu_delegate_falls_back_to_cpu(self) -> None:
        fake_mp, fake_landmarker_class = self._fake_runtime_mp()
        diagnostics = {
            "requested_delegate": "gpu",
            "selected_delegate": "cpu",
            "fallback_to_cpu": True,
            "fallback_reason": "gpu_init_failed",
            "model_asset_path": "/tmp/fake.task",
            "warnings": ["mediapipe_delegate_fallback_to_cpu"],
        }
        with mock.patch("src.face_aesthetics.landmarks.mediapipe_source._import_mediapipe", return_value=(fake_mp, None)), \
             mock.patch("src.face_aesthetics.landmarks.mediapipe_source._build_solutions_face_mesh", return_value=None), \
             mock.patch(
                 "src.face_aesthetics.landmarks.mediapipe_source._build_tasks_face_landmarker",
                 return_value=({"landmarker": fake_landmarker_class(), "lock": mock.MagicMock()}, diagnostics),
             ):
            result = build_mediapipe_source_result(
                image_bgr=np.zeros((64, 64, 3), dtype=np.uint8),
                config={"enabled": True, "delegate": "gpu", "fallback_to_cpu": True, "model_asset_path": "/tmp/fake.task"},
            )
        self.assertTrue(result.available)
        self.assertEqual(result.metadata["requested_delegate"], "gpu")
        self.assertEqual(result.metadata["selected_delegate"], "cpu")
        self.assertTrue(result.metadata["fallback_to_cpu"])
        self.assertEqual(result.metadata["fallback_reason"], "gpu_init_failed")

    def test_mediapipe_cpu_delegate_keeps_cpu_selection(self) -> None:
        fake_mp, fake_landmarker_class = self._fake_runtime_mp()
        diagnostics = {
            "requested_delegate": "cpu",
            "selected_delegate": "cpu",
            "fallback_to_cpu": False,
            "fallback_reason": None,
            "model_asset_path": "/tmp/fake.task",
            "warnings": [],
        }
        with mock.patch("src.face_aesthetics.landmarks.mediapipe_source._import_mediapipe", return_value=(fake_mp, None)), \
             mock.patch("src.face_aesthetics.landmarks.mediapipe_source._build_solutions_face_mesh", return_value=None), \
             mock.patch(
                 "src.face_aesthetics.landmarks.mediapipe_source._build_tasks_face_landmarker",
                 return_value=({"landmarker": fake_landmarker_class(), "lock": mock.MagicMock()}, diagnostics),
             ):
            result = build_mediapipe_source_result(
                image_bgr=np.zeros((64, 64, 3), dtype=np.uint8),
                config={"enabled": True, "delegate": "cpu", "fallback_to_cpu": True, "model_asset_path": "/tmp/fake.task"},
            )
        self.assertTrue(result.available)
        self.assertEqual(result.metadata["requested_delegate"], "cpu")
        self.assertEqual(result.metadata["selected_delegate"], "cpu")
        self.assertFalse(result.metadata["fallback_to_cpu"])

    def test_mediapipe_auto_delegate_does_not_break_when_cpu_selected(self) -> None:
        fake_mp, fake_landmarker_class = self._fake_runtime_mp()
        diagnostics = {
            "requested_delegate": "auto",
            "selected_delegate": "cpu",
            "fallback_to_cpu": True,
            "fallback_reason": "gpu_init_failed",
            "model_asset_path": "/tmp/fake.task",
            "warnings": ["mediapipe_delegate_fallback_to_cpu"],
        }
        with mock.patch("src.face_aesthetics.landmarks.mediapipe_source._import_mediapipe", return_value=(fake_mp, None)), \
             mock.patch("src.face_aesthetics.landmarks.mediapipe_source._build_solutions_face_mesh", return_value=None), \
             mock.patch(
                 "src.face_aesthetics.landmarks.mediapipe_source._build_tasks_face_landmarker",
                 return_value=({"landmarker": fake_landmarker_class(), "lock": mock.MagicMock()}, diagnostics),
             ):
            result = build_mediapipe_source_result(
                image_bgr=np.zeros((64, 64, 3), dtype=np.uint8),
                config={"enabled": True, "delegate": "auto", "fallback_to_cpu": True, "model_asset_path": "/tmp/fake.task"},
            )
        self.assertTrue(result.available)
        self.assertEqual(result.metadata["requested_delegate"], "auto")
        self.assertEqual(result.metadata["selected_delegate"], "cpu")

    def test_mediapipe_gpu_is_forced_to_cpu_when_env_disables_gpu(self) -> None:
        fake_mp, fake_landmarker_class = self._fake_runtime_mp()
        diagnostics = {
            "requested_delegate": "gpu",
            "selected_delegate": "cpu",
            "fallback_to_cpu": False,
            "fallback_reason": "gpu_disabled_by_env",
            "gpu_disabled": True,
            "model_asset_path": "/tmp/fake.task",
            "warnings": ["mediapipe_gpu_disabled_by_env"],
        }
        with mock.patch.dict("os.environ", {"MEDIAPIPE_DISABLE_GPU": "1"}), \
             mock.patch("src.face_aesthetics.landmarks.mediapipe_source._import_mediapipe", return_value=(fake_mp, None)), \
             mock.patch("src.face_aesthetics.landmarks.mediapipe_source._build_solutions_face_mesh", return_value=None), \
             mock.patch("src.face_aesthetics.landmarks.mediapipe_source._build_tasks_face_landmarker", return_value=({"landmarker": fake_landmarker_class(), "lock": mock.MagicMock()}, diagnostics)):
            result = build_mediapipe_source_result(
                image_bgr=np.zeros((64, 64, 3), dtype=np.uint8),
                config={"enabled": True, "delegate": "gpu", "fallback_to_cpu": True, "model_asset_path": "/tmp/fake.task"},
            )
        self.assertTrue(result.available)
        self.assertTrue(result.metadata["gpu_disabled"])
        self.assertEqual(result.metadata["selected_delegate"], "cpu")
        self.assertIn("mediapipe_gpu_disabled_by_env", result.warnings)

    def test_tasks_landmarker_reuses_cached_instance(self) -> None:
        fake_mp = type(
            "FakeMediaPipe",
            (),
            {
                "tasks": type(
                    "Tasks",
                    (),
                    {
                        "vision": object(),
                        "BaseOptions": type(
                            "BaseOptions",
                            (),
                            {"Delegate": type("Delegate", (), {"CPU": "CPU", "GPU": "GPU"})},
                        ),
                    },
                )()
            },
        )()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "face_landmarker.task"
            model_path.write_bytes(b"fake")
            cache_key = (str(model_path.resolve()), "cpu")
            cached = {"landmarker": object(), "lock": mock.MagicMock()}
            mediapipe_source._TASKS_LANDMARKER_CACHE[cache_key] = cached
            with mock.patch.dict("os.environ", {"MEDIAPIPE_DISABLE_GPU": "1"}):
                landmarker, diagnostics = mediapipe_source._build_tasks_face_landmarker(
                    fake_mp,
                    {"enabled": True, "delegate": "cpu", "fallback_to_cpu": True, "model_asset_path": str(model_path)},
                    None,
                )
        self.assertIs(landmarker, cached)
        self.assertEqual(diagnostics["selected_delegate"], "cpu")

    def test_capability_writer_emits_mediapipe_runtime_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_config_tree(project_root)
            report = {
                "sources": {
                    "mediapipe_face_mesh": {
                        "available": False,
                        "landmark_count": 0,
                        "warnings": ["mediapipe_runtime_backend_unavailable"],
                        "metadata": {
                            "requested_delegate": "auto",
                            "selected_delegate": "cpu",
                            "fallback_to_cpu": True,
                            "fallback_reason": "gpu_init_failed",
                            "model_asset_path": "data/interim/mediapipe/face_landmarker.task",
                            "runtime_backend": "tasks_face_landmarker",
                        },
                    }
                }
            }
            written = write_landmark_capability_artifacts(project_root=project_root, capability_report=report)
            diagnostics_path = project_root / str(written["mediapipe_runtime_diagnostics_path"])
            self.assertTrue(diagnostics_path.exists())
            payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["requested_delegate"], "auto")
            self.assertEqual(payload["selected_delegate"], "cpu")
            self.assertTrue(payload["fallback_to_cpu"])
            self.assertEqual(payload["fallback_reason"], "gpu_init_failed")


if __name__ == "__main__":
    unittest.main()
