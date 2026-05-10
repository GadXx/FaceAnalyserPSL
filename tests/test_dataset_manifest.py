from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.face_aesthetics.data.manifest import build_dataset_manifest, collect_dataset_manifest_rows


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc`\x00\x00\x00\x02\x00\x01"
    b"\xe2!\xbc3"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_BYTES)


def _dataset_config() -> dict:
    return {
        "raw_input_dir": "data/raw",
        "manifest_csv_path": "data/interim/dataset_manifest.csv",
        "interim_manifest_path": "data/interim/dataset_manifest.parquet",
        "summary_path": "reports/metrics/dataset_manifest_summary.json",
        "adapter_order": [
            "fairface",
            "utkface",
            "chicago_face_database",
            "scut_compat",
            "generic_image_folder",
        ],
        "ignore_names": [".gitkeep"],
        "adapters": {
            "fairface": {
                "enabled": True,
                "dataset_name": "fairface",
                "root_name_patterns": ["fairface", "fairface*"],
                "label_globs": [
                    "**/train_labels.csv",
                    "**/val_labels.csv",
                    "**/fairface_label*.csv",
                    "**/*fairface*label*.csv",
                ],
            },
            "utkface": {
                "enabled": True,
                "dataset_name": "utkface",
                "root_name_patterns": ["utkface", "utkface*"],
            },
            "chicago_face_database": {
                "enabled": True,
                "dataset_name": "chicago_face_database",
                "root_name_patterns": ["cfd", "cfd*", "chicago*face*database*"],
                "metadata_globs": ["**/*.csv", "**/*.tsv", "**/*.json"],
            },
            "scut_compat": {
                "enabled": True,
                "dataset_name": "scut_compat",
                "root_name_patterns": ["scut*"],
                "metadata_globs": ["**/*.csv", "**/*.tsv", "**/*.txt", "**/*.json"],
            },
            "generic_image_folder": {
                "enabled": True,
                "dataset_name_map": {},
            },
        },
        "split_aliases": {
            "train": "train",
            "training": "train",
            "val": "val",
            "valid": "val",
            "validation": "val",
            "dev": "val",
            "test": "test",
            "testing": "test",
        },
        "deterministic_split": {
            "enabled": True,
            "seed": "manifest_v1",
            "default_train_fraction": 0.8,
            "default_val_fraction": 0.1,
            "default_test_fraction": 0.1,
            "missing_test_from_train_fraction": 0.1,
        },
    }


def _io_config() -> dict:
    return {
        "allowed_image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
    }


class DatasetManifestTests(unittest.TestCase):
    def test_collect_rows_from_fairface_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            raw_dir = project_root / "data" / "raw"
            image_path = raw_dir / "FairFace" / "images" / "001.png"
            _write_image(image_path)

            labels_path = raw_dir / "FairFace" / "fairface_label_train.csv"
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            labels_path.write_text(
                "file,age,gender,race,service_test\n"
                "images/001.png,20-29,Female,East Asian,False\n",
                encoding="utf-8",
            )

            rows, adapter_counts = collect_dataset_manifest_rows(
                project_root=project_root,
                raw_dir=raw_dir,
                dataset_config=_dataset_config(),
                io_config=_io_config(),
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(adapter_counts.get("fairface"), 1)

            row = rows[0]
            self.assertEqual(row["dataset_name"], "fairface")
            self.assertEqual(row["sample_id"], "images__001")
            self.assertEqual(row["image_id"], "fairface::images__001")
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["quality_gate_status"], "pending_preprocessing")
            self.assertEqual(row["adapter_version"], "fairface@1")

            raw_labels = json.loads(row["raw_labels"])
            normalized_labels = json.loads(row["normalized_labels"])
            self.assertEqual(raw_labels["gender"], "Female")
            self.assertEqual(normalized_labels["age_group"], "20-29")
            self.assertEqual(normalized_labels["ethnicity_group"], "East Asian")

    def test_collect_rows_from_utkface_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            raw_dir = project_root / "data" / "raw"
            image_path = raw_dir / "UTKFace" / "21_1_3_20170109150557335.jpg"
            _write_image(image_path)

            rows, adapter_counts = collect_dataset_manifest_rows(
                project_root=project_root,
                raw_dir=raw_dir,
                dataset_config=_dataset_config(),
                io_config=_io_config(),
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(adapter_counts.get("utkface"), 1)
            row = rows[0]
            self.assertEqual(row["dataset_name"], "utkface")
            self.assertEqual(row["image_id"], "utkface::21_1_3_20170109150557335")
            normalized_labels = json.loads(row["normalized_labels"])
            self.assertEqual(normalized_labels["age_years"], 21)
            self.assertEqual(normalized_labels["sex_code"], 1)
            self.assertEqual(normalized_labels["ethnicity_code"], 3)

    def test_build_dataset_manifest_for_generic_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            configs_dir = project_root / "configs"
            configs_dir.mkdir(parents=True, exist_ok=True)

            (configs_dir / "base.yaml").write_text(
                json.dumps(
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
                    }
                ),
                encoding="utf-8",
            )
            (configs_dir / "data.yaml").write_text(
                json.dumps(
                    {
                        "dataset": _dataset_config(),
                        "io": _io_config(),
                    }
                ),
                encoding="utf-8",
            )
            for name in (
                "preprocess.yaml",
                "features.yaml",
                "train_baseline.yaml",
                "train_geometry.yaml",
                "train_fusion.yaml",
                "infer.yaml",
                "robustness.yaml",
            ):
                (configs_dir / name).write_text("{}", encoding="utf-8")

            image_path = project_root / "data" / "raw" / "measurements" / "train" / "subject_001.png"
            _write_image(image_path)

            artifacts = build_dataset_manifest(project_root=project_root)
            self.assertTrue(artifacts.manifest_parquet_path.exists())
            self.assertEqual(artifacts.summary["total_rows"], 1)
            self.assertEqual(artifacts.summary["dataset_counts"]["measurements"], 1)

            import pandas as pd

            frame = pd.read_parquet(artifacts.manifest_parquet_path)
            self.assertEqual(frame.loc[0, "dataset_name"], "measurements")
            self.assertEqual(frame.loc[0, "split"], "train")
            self.assertEqual(frame.loc[0, "quality_gate_status"], "pending_preprocessing")

    def test_empty_chicago_root_does_not_capture_fairface_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            raw_dir = project_root / "data" / "raw"
            _write_image(raw_dir / "fairface" / "train" / "1.png")
            (raw_dir / "fairface" / "train_labels.csv").write_text(
                "file,age,gender,race,service_test\n"
                "train/1.png,20-29,Female,East Asian,False\n",
                encoding="utf-8",
            )
            (raw_dir / "chicago_face_database").mkdir(parents=True, exist_ok=True)

            rows, adapter_counts = collect_dataset_manifest_rows(
                project_root=project_root,
                raw_dir=raw_dir,
                dataset_config=_dataset_config(),
                io_config=_io_config(),
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(adapter_counts.get("fairface"), 1)
            self.assertIsNone(adapter_counts.get("chicago_face_database"))
            self.assertEqual(rows[0]["dataset_name"], "fairface")

    def test_irrelevant_hidden_metadata_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            raw_dir = project_root / "data" / "raw"
            _write_image(raw_dir / "fairface" / "train" / "1.png")
            (raw_dir / "fairface" / "train_labels.csv").write_text(
                "file,age,gender,race,service_test\n"
                "train/1.png,20-29,Female,East Asian,False\n",
                encoding="utf-8",
            )
            hidden_dir = raw_dir / "fairface" / ".ipynb_checkpoints"
            hidden_dir.mkdir(parents=True, exist_ok=True)
            (hidden_dir / "train_labels.csv").write_text(
                "file,age,gender,race,service_test\n"
                ".ipynb_checkpoints/ghost.png,30-39,Male,White,False\n",
                encoding="utf-8",
            )

            rows, adapter_counts = collect_dataset_manifest_rows(
                project_root=project_root,
                raw_dir=raw_dir,
                dataset_config=_dataset_config(),
                io_config=_io_config(),
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(adapter_counts.get("fairface"), 1)
            self.assertEqual(rows[0]["sample_id"], "train__1")


if __name__ == "__main__":
    unittest.main()
