from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.services.inference_service import analyze_image, build_preprocess_overrides


def _str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-sample 2D morphometric inference.")
    parser.add_argument("--image-path", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--preferred-execution-provider", type=str, default=None)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--strict-gpu", type=_str_to_bool, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    preprocess_overrides = build_preprocess_overrides(
        preferred_execution_provider=args.preferred_execution_provider,
        device_id=args.device_id,
        strict_gpu=args.strict_gpu,
    )

    artifacts = analyze_image(
        image_path=args.image_path,
        output_dir=args.output_dir,
        group_override=args.group,
        project_root=PROJECT_ROOT,
        preprocess_overrides=preprocess_overrides,
    )

    print(f"report_json={artifacts.report_json_path}")
    print(f"report_markdown={artifacts.report_markdown_path}")
    print(f"reporting_summary={artifacts.reporting_summary_path}")
    print(f"analysis_image={artifacts.analysis_image_path}")
    print(f"aligned_image={artifacts.aligned_image_path}")
    print(f"measurement_image={artifacts.measurement_image_path}")
    print(f"annotated_image={artifacts.annotated_image_path}")
    print(f"processing_status={artifacts.report['input_quality']['processing_status']}")
    print(f"quality_pass={artifacts.report['input_quality']['quality_pass']}")
    print(f"analysis_image_role={artifacts.report['input']['analysis_image_role']}")
    print(f"alignment_applied={artifacts.report['input_quality']['alignment']['applied']}")
    print(f"alignment_rotation_deg={artifacts.report['input_quality']['alignment']['rotation_deg']}")
    print(f"selected_group={artifacts.report['reference_comparison']['selected_group']['group_key']}")
    print(f"mahalanobis_distance={artifacts.report['reference_comparison']['mahalanobis_distance']}")


if __name__ == "__main__":
    main()
