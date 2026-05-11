from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.scoring.inference import run_scorer_inference


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain experimental scorer output for a single image.")
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
    explanation = report.get("local_explanation", {})
    print(f"report_json={artifacts.report_json_path}")
    print(f"primary_model={report['primary_score']['model']}")
    print(f"primary_score={report['primary_score']['predicted_score']}")
    print(f"explanation_method={explanation.get('explanation_method')}")
    print(f"baseline_score={explanation.get('baseline_score')}")
    if report.get("reference_context"):
        print(f"reference_context={report['reference_context']}")
    for row in explanation.get("top_positive_features", [])[:3]:
        print(f"positive_feature={row['feature']} contribution={row['contribution']}")
    for row in explanation.get("top_negative_features", [])[:3]:
        print(f"negative_feature={row['feature']} contribution={row['contribution']}")


if __name__ == "__main__":
    main()
