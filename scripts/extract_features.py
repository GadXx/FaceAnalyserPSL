from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.pipelines.feature_extraction import run_feature_extraction


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run morphometric feature extraction.")
    parser.add_argument("--mediapipe-delegate", choices=("auto", "cpu", "gpu"), default=None)
    parser.add_argument("--mediapipe-fallback-to-cpu", type=_parse_bool, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifacts = run_feature_extraction(
        project_root=PROJECT_ROOT,
        mediapipe_delegate=args.mediapipe_delegate,
        mediapipe_fallback_to_cpu=args.mediapipe_fallback_to_cpu,
    )
    summary = artifacts.summary

    print(f"measurements_raw_path={artifacts.measurements_raw_path}")
    print(f"measurements_normalized_path={artifacts.measurements_normalized_path}")
    print(f"symmetry_features_path={artifacts.symmetry_features_path}")
    print(f"geometry_features_path={artifacts.geometry_features_path}")
    print(f"geometry_3d_features_path={artifacts.geometry_3d_features_path}")
    print(f"auxiliary_embeddings_path={artifacts.auxiliary_embeddings_path}")
    print(f"summary_json={artifacts.summary_path}")
    print(f"mediapipe_runtime_diagnostics={summary.get('mediapipe_runtime_diagnostics_path')}")
    print(f"total_rows={summary['total_rows']}")
    print(f"eligible_rows={summary['eligible_rows']}")
    print(f"processed_rows={summary['processed_rows']}")
    print(f"failed_rows={summary['failed_rows']}")
    print(f"raw_measurement_count={summary['raw_measurement_count']}")
    print(f"normalized_measurement_count={summary['normalized_measurement_count']}")
    print(f"symmetry_feature_count={summary['symmetry_feature_count']}")
    print(f"angle_feature_count={summary['angle_feature_count']}")
    print(f"geometry_feature_count={summary['geometry_feature_count']}")
    print(f"geometry_3d_feature_count={summary['geometry_3d_feature_count']}")
    print(f"geometry_3d_rows={summary['geometry_3d_rows']}")
    print(f"three_d_enabled={summary['three_d_enabled']}")
    print(f"three_d_reason={summary['three_d_reason']}")
    print(f"auxiliary_embedding_rows={summary['auxiliary_embedding_rows']}")


if __name__ == "__main__":
    main()
