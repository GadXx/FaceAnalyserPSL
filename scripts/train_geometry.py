from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.pipelines.geometry_training import run_geometry_training


def main() -> None:
    artifacts = run_geometry_training(project_root=PROJECT_ROOT)
    geometry_summary = artifacts.summary["geometry"]
    comparison_summary = artifacts.summary["comparison"]

    print(f"metrics_json={artifacts.metrics_json_path}")
    print(f"comparison_report={artifacts.comparison_report_path}")
    print(f"val_predictions={artifacts.val_predictions_path}")
    print(f"test_predictions={artifacts.test_predictions_path}")
    print(f"feature_importance={artifacts.feature_importance_path}")
    print(f"config_snapshot={artifacts.config_snapshot_path}")
    print(f"geometry_status={geometry_summary['status']}")
    print(f"best_model={geometry_summary['best_model']}")
    print(f"comparison_conclusion={comparison_summary['conclusion']}")


if __name__ == "__main__":
    main()
