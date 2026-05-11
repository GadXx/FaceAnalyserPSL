from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.scoring.dataset import build_scoring_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare manifest for folder-scored experimental dataset.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifacts = build_scoring_manifest(project_root=PROJECT_ROOT, dataset_root=args.dataset_root)
    summary = artifacts.summary
    print(f"scoring_manifest={artifacts.manifest_path}")
    print(f"summary_json={artifacts.summary_path}")
    print(f"rows={summary['rows']}")
    print(f"group_count={summary['group_count']}")
    print(f"split_counts={summary['split_counts']}")


if __name__ == "__main__":
    main()
