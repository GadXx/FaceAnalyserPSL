from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.scoring.inference import run_scorer_inference


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experimental continuous scorer on a single image.")
    parser.add_argument("--image-path", required=True, type=Path)
    parser.add_argument("--gender", type=str, default=None)
    parser.add_argument("--age-group", type=str, default=None)
    parser.add_argument("--race", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reference_filters = {k: v for k, v in {"gender": args.gender, "age_group": args.age_group, "race": args.race}.items() if v}
    artifacts = run_scorer_inference(
        image_path=args.image_path,
        project_root=PROJECT_ROOT,
        reference_filters=reference_filters or None,
    )
    report = artifacts.report
    primary = report["primary_score"]
    regression = report["regression"]
    classification = report["classification"]
    final_score = report.get("final_score", {})
    print(f"report_json={artifacts.report_json_path}")
    print(f"report_markdown={artifacts.report_markdown_path}")
    print(f"primary_model={primary['model']}")
    print(f"primary_score={primary['predicted_score']}")
    print(f"score_source={primary.get('score_source')}")
    print(f"uses_calibration={primary.get('uses_calibration')}")
    print(f"display_score={primary.get('display_score')}")
    print(f"raw_primary_score={primary.get('raw_predicted_score')}")
    print(f"calibrated_primary_score={primary.get('calibrated_predicted_score')}")
    print(f"nearest_score_group={primary['nearest_score_group']}")
    print(f"distance_to_nearest_group={primary['distance_to_nearest_group']}")
    print(f"predicted_score={regression['predicted_score']}")
    if report.get("pairwise"):
        print(f"pairwise_score={report['pairwise'].get('approx_score')}")
    if report.get("ordinal"):
        print(f"ordinal_score={report['ordinal'].get('predicted_score')}")
    print(f"final_score={final_score.get('predicted_score')}")
    print(f"predicted_group={classification['predicted_group']}")
    print(f"confidence_level={report['confidence']['level']}")
    print(f"quality_pass={report['quality']['quality_pass']}")
    if report.get("reference_context"):
        print(f"reference_context={report['reference_context']}")


if __name__ == "__main__":
    main()
