from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.face_aesthetics.data.io import save_image
from src.face_aesthetics.pipelines.inference import InferenceArtifacts
from src.face_aesthetics.pipelines.robustness import run_robustness_evaluation


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_minimal_configs(project_root: Path) -> None:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "base.yaml", {"project": {"name": "robustness-test"}})
    _write_json(config_dir / "data.yaml", {"dataset": {"default_reference_model": "default_reference"}})
    _write_json(config_dir / "features.yaml", {})
    _write_json(config_dir / "preprocess.yaml", {})
    _write_json(config_dir / "train_geometry.yaml", {})
    _write_json(config_dir / "train_fusion.yaml", {})
    _write_json(config_dir / "train_baseline.yaml", {"train_baseline": {"group_label_aliases": {}}})
    _write_json(config_dir / "infer.yaml", {})
    _write_json(
        config_dir / "robustness.yaml",
        {
            "robustness": {
                "output_summary_path": "reports/metrics/robustness_summary.json",
                "output_feature_table_path": "reports/metrics/robustness_feature_deltas.parquet",
                "work_dir": "reports/robustness_eval",
                "preferred_execution_provider": "CPUExecutionProvider",
                "fallback_execution_provider": "CPUExecutionProvider",
                "device_id": 0,
                "strict_gpu_mode": False,
                "top_k_unstable_features": 5,
                "perturbations": {
                    "crop_shift_x": 0.04,
                    "crop_shift_y": 0.04,
                    "resize_scale": 0.9,
                    "jpeg_quality": 60,
                    "blur_kernel_size": 3,
                    "brightness_delta": 18.0,
                    "rotation_deg": 4.0,
                },
                "tolerances": {
                    "raw_measurements": 5.0,
                    "normalized_ratios": 0.1,
                    "symmetry_metrics": 0.1,
                    "angle_metrics": 4.0,
                    "optional_composite_indices": 0.1,
                },
            }
        },
    )


def _fake_inference_artifacts(image_path: Path, delta: float) -> InferenceArtifacts:
    report = {
        "input": {"image_path": str(image_path)},
        "input_quality": {
            "processing_status": "processed",
            "quality_pass": True,
            "quality_fail_reason": None,
        },
        "raw_measurements": {
            "raw_eye_distance_px": 40.0 + delta,
            "raw_face_width_px": 80.0 + delta,
        },
        "normalized_ratios": {
            "ratio_face_aspect": 0.75 + delta / 100.0,
        },
        "symmetry_metrics": {
            "symmetry_index": 0.95 - delta / 100.0,
        },
        "angle_metrics": {
            "angle_eye_line_deg": 1.0 + delta,
        },
        "optional_composite_indices": {
            "morphometric_harmony_index": 0.8 - delta / 100.0,
        },
    }
    return InferenceArtifacts(
        report_json_path=image_path.parent / "sample_report.json",
        report_markdown_path=image_path.parent / "sample_report.md",
        reporting_summary_path=image_path.parent / "reporting_summary.json",
        analysis_image_path=image_path,
        normalized_frame_image_path=image_path,
        aligned_image_path=None,
        measurement_image_path=image_path,
        annotated_image_path=None,
        report=report,
        reporting_summary={},
    )


class RobustnessPipelineTests(unittest.TestCase):
    def test_robustness_evaluation_writes_summary_and_feature_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)

            image_path = project_root / "sample.jpg"
            import numpy as np

            save_image(image_path, (np.ones((32, 32, 3), dtype=np.uint8) * 127))

            def _patched_run_inference(*, image_path: Path, **_: object) -> InferenceArtifacts:
                perturbation_key = image_path.stem.split("__")[-1] if "__" in image_path.stem else "original"
                deltas = {
                    "original": 0.0,
                    "crop_shift": 1.0,
                    "resize": 2.0,
                    "jpeg_compression": 3.0,
                    "mild_blur": 4.0,
                    "brightness_change": 5.0,
                    "small_rotation": 6.0,
                }
                return _fake_inference_artifacts(image_path, deltas[perturbation_key])

            with patch("src.face_aesthetics.pipelines.robustness.run_inference", side_effect=_patched_run_inference):
                artifacts = run_robustness_evaluation(
                    image_paths=[image_path],
                    group_override="sex_group_normalized=Female",
                    project_root=project_root,
                )

            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.feature_table_path.exists())

            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            frame = pd.read_parquet(artifacts.feature_table_path)

            self.assertEqual(summary["image_count"], 1)
            self.assertEqual(summary["successful_baseline_image_count"], 1)
            self.assertEqual(summary["failed_perturbation_runs"], 0)
            self.assertEqual(summary["feature_delta_row_count"], len(frame))
            self.assertIn("raw_eye_distance_px", summary["feature_aggregates"])
            self.assertIn("morphometric_harmony_index", summary["index_aggregates"])
            self.assertEqual(frame["perturbation_name"].nunique(), 6)
            self.assertIn("absolute_delta", frame.columns)
            self.assertIn("relative_delta", frame.columns)
            self.assertIn("tolerance_pass", frame.columns)
            self.assertTrue(frame["tolerance"].notna().any())

    def test_robustness_evaluation_records_failed_perturbation_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_minimal_configs(project_root)

            image_path = project_root / "sample.jpg"
            import numpy as np

            save_image(image_path, (np.ones((32, 32, 3), dtype=np.uint8) * 127))

            def _patched_run_inference(*, image_path: Path, **_: object) -> InferenceArtifacts:
                if image_path.stem.endswith("small_rotation"):
                    raise ValueError("rotation preprocessing failed")
                perturbation_key = image_path.stem.split("__")[-1] if "__" in image_path.stem else "original"
                deltas = {
                    "original": 0.0,
                    "crop_shift": 1.0,
                    "resize": 2.0,
                    "jpeg_compression": 3.0,
                    "mild_blur": 4.0,
                    "brightness_change": 5.0,
                }
                return _fake_inference_artifacts(image_path, deltas[perturbation_key])

            with patch("src.face_aesthetics.pipelines.robustness.run_inference", side_effect=_patched_run_inference):
                artifacts = run_robustness_evaluation(
                    image_paths=[image_path],
                    group_override="sex_group_normalized=Female",
                    project_root=project_root,
                )

            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["failed_perturbation_runs"], 1)
            self.assertEqual(summary["successful_perturbation_runs"], 5)
            self.assertEqual(summary["image_summaries"][0]["failed_perturbation_runs"], 1)
            self.assertEqual(
                summary["image_summaries"][0]["perturbations"][-1]["status"],
                "failed_perturbed_inference",
            )


if __name__ == "__main__":
    unittest.main()
