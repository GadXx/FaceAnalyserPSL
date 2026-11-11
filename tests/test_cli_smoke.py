from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CliSmokeTests(unittest.TestCase):
    def test_run_inference_cli_prints_expected_summary(self) -> None:
        script_path = Path("scripts/run_inference.py").resolve()
        module = _load_script_module("run_inference_cli_test", script_path)
        fake_artifacts = SimpleNamespace(
            report_json_path=Path("/tmp/sample_report.json"),
            report_markdown_path=Path("/tmp/sample_report.md"),
            reporting_summary_path=Path("/tmp/reporting_summary.json"),
            analysis_image_path=Path("/tmp/analysis_image.jpg"),
            normalized_frame_image_path=Path("/tmp/normalized_frame.jpg"),
            aligned_image_path=Path("/tmp/aligned_face.jpg"),
            measurement_image_path=Path("/tmp/aligned_face.jpg"),
            annotated_image_path=Path("/tmp/annotated_analysis.jpg"),
            report={
                "input": {
                    "analysis_image_role": "aligned_face",
                },
                "input_quality": {
                    "processing_status": "processed",
                    "quality_pass": True,
                    "alignment": {
                        "applied": True,
                        "rotation_deg": -7.5,
                    },
                },
                "reference_comparison": {
                    "selected_group": {"group_key": "sex_group_normalized=Female"},
                    "mahalanobis_distance": 1.25,
                },
            },
        )

        with patch.object(module, "analyze_image", return_value=fake_artifacts):
            with patch.object(sys, "argv", ["run_inference.py", "--image-path", "sample.jpg", "--group", "sex_group_normalized=Female"]):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    module.main()

        output = buffer.getvalue()
        self.assertIn("report_json=/tmp/sample_report.json", output)
        self.assertIn("analysis_image=/tmp/analysis_image.jpg", output)
        self.assertIn("processing_status=processed", output)
        self.assertIn("quality_pass=True", output)
        self.assertIn("analysis_image_role=aligned_face", output)
        self.assertIn("alignment_applied=True", output)
        self.assertIn("selected_group=sex_group_normalized=Female", output)

    def test_extract_preprocessing_artifacts_cli_prints_smoke_summary(self) -> None:
        script_path = Path("scripts/extract_preprocessing_artifacts.py").resolve()
        module = _load_script_module("run_preprocessing_cli_test", script_path)
        fake_artifacts = SimpleNamespace(
            output_manifest_path=Path("/tmp/preprocessed_manifest.parquet"),
            summary_path=Path("/tmp/preprocessing_summary.json"),
            summary={
                "input_total_rows": 16,
                "processed_rows": 16,
                "successful_rows": 7,
                "rejected_rows": 9,
                "error_rows": 0,
                "quality_pass_true_count": 7,
                "processing_status_counts": {"processed": 16},
                "rejection_counts": {"no_face": 8},
                "aligned_nonempty_count": 8,
                "landmarks_nonempty_count": 8,
                "shards_written": 2,
                "aligned_examples": ["data/processed/aligned_faces/sample.jpg"],
                "provider_diagnostics": {
                    "available_providers": ["CPUExecutionProvider"],
                    "selected_execution_provider": "CPUExecutionProvider",
                    "fallback_to_cpu": False,
                    "device_id": 0,
                    "actual_model_providers": {"detection": ["CPUExecutionProvider"]},
                },
            },
        )

        with patch.object(module, "run_preprocessing", return_value=fake_artifacts):
            with patch.object(sys, "argv", ["extract_preprocessing_artifacts.py", "--smoke", "--limit", "16"]):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    module.main()

        output = buffer.getvalue()
        self.assertIn("preprocessed_manifest=/tmp/preprocessed_manifest.parquet", output)
        self.assertIn("input_total_rows=16", output)
        self.assertIn("successful_rows=7", output)
        self.assertIn("selected_execution_provider=CPUExecutionProvider", output)

    def test_run_api_cli_delegates_to_uvicorn(self) -> None:
        script_path = Path("scripts/run_api.py").resolve()
        module = _load_script_module("run_api_cli_test", script_path)

        self.assertEqual(os.environ.get("MEDIAPIPE_DISABLE_GPU"), "1")

        with patch.object(module.uvicorn, "run") as mocked_run:
            with patch.object(sys, "argv", ["run_api.py", "--host", "127.0.0.1", "--port", "9000"]):
                module.main()

        self.assertTrue(mocked_run.called)
        _, kwargs = mocked_run.call_args
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 9000)


if __name__ == "__main__":
    unittest.main()
