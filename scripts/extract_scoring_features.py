from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.scoring.features import run_scoring_feature_extraction


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract scoring features from the experimental dataset.")
    parser.add_argument("--mediapipe-delegate", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--mediapipe-fallback-to-cpu", type=_parse_bool, default=None)
    parser.add_argument("--use-fairface-reference", type=_parse_bool, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifacts = run_scoring_feature_extraction(
        project_root=PROJECT_ROOT,
        mediapipe_delegate=args.mediapipe_delegate,
        mediapipe_fallback_to_cpu=args.mediapipe_fallback_to_cpu,
        use_fairface_reference=args.use_fairface_reference,
    )
    summary = artifacts.summary
    print(f"scoring_features={artifacts.features_path}")
    print(f"summary_json={artifacts.summary_path}")
    print(f"feature_rows={summary['feature_rows']}")
    print(f"quality_pass_rows={summary['quality_pass_rows']}")
    print(f"processed_rows={summary['processed_rows']}")


if __name__ == "__main__":
    main()
