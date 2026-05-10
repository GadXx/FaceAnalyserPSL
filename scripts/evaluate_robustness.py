from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.pipelines.robustness import run_robustness_evaluation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run robustness evaluation for the objective morphometric pipeline."
    )
    parser.add_argument(
        "--image-path",
        dest="image_paths",
        type=Path,
        action="append",
        required=True,
        help="Input image path. Pass multiple times to evaluate multiple images.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--group", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifacts = run_robustness_evaluation(
        image_paths=args.image_paths,
        group_override=args.group,
        output_dir=args.output_dir,
        project_root=PROJECT_ROOT,
    )

    print(f"robustness_summary={artifacts.summary_path}")
    print(f"feature_table={artifacts.feature_table_path}")
    print(f"image_count={artifacts.summary['image_count']}")
    print(f"successful_baseline_image_count={artifacts.summary['successful_baseline_image_count']}")
    print(f"failed_image_count={artifacts.summary['failed_image_count']}")
    print(f"successful_perturbation_runs={artifacts.summary['successful_perturbation_runs']}")
    print(f"failed_perturbation_runs={artifacts.summary['failed_perturbation_runs']}")
    print(f"feature_delta_row_count={artifacts.summary['feature_delta_row_count']}")
    print(f"index_delta_row_count={artifacts.summary['index_delta_row_count']}")


if __name__ == "__main__":
    main()
