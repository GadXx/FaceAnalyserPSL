from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.pipelines.baseline_training import run_baseline_training


def main() -> None:
    artifacts = run_baseline_training(project_root=PROJECT_ROOT)
    summary = artifacts.summary

    print(f"reference_model_summary={artifacts.summary_json_path}")
    print(f"group_statistics={artifacts.group_statistics_path}")
    print(f"reference_model={artifacts.reference_model_path}")
    print(f"config_snapshot={artifacts.config_snapshot_path}")
    print(f"status={summary['status']}")
    print(f"built_group_count={summary['built_group_count']}")
    print(f"skipped_group_count={summary['skipped_group_count']}")


if __name__ == "__main__":
    main()
