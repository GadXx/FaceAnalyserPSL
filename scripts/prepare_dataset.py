from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.data.manifest import build_dataset_manifest


def main() -> None:
    artifacts = build_dataset_manifest(project_root=PROJECT_ROOT)
    split_counts = artifacts.summary["split_counts"]
    dataset_counts = artifacts.summary.get("dataset_counts", {})
    validation = artifacts.summary["validation"]

    print(f"manifest_csv={artifacts.manifest_csv_path}")
    print(f"manifest_parquet={artifacts.manifest_parquet_path}")
    print(f"summary_json={artifacts.summary_path}")
    print(
        "split_sizes="
        f"train:{split_counts['train']} "
        f"val:{split_counts['val']} "
        f"test:{split_counts['test']}"
    )
    print(
        "validation="
        f"broken_files:{validation['broken_files']} "
        f"empty_paths:{validation['empty_paths']} "
        f"duplicate_image_ids:{validation['duplicate_image_ids']}"
    )
    print(f"dataset_counts={dataset_counts}")


if __name__ == "__main__":
    main()
