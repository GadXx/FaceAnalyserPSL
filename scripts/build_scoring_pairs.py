from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.scoring.training import run_scorer_training


def main() -> None:
    artifacts = run_scorer_training(project_root=PROJECT_ROOT)
    summary = artifacts.summary
    print(f"pairwise_dataset={summary['artifacts']['pairwise_dataset_path']}")
    print(f"pairwise_summary={summary['artifacts']['pairwise_dataset_summary_path']}")


if __name__ == "__main__":
    main()
