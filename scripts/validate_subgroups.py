from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.pipelines.subgroup_validation import run_subgroup_validation


def main() -> None:
    artifacts = run_subgroup_validation(project_root=PROJECT_ROOT)
    print(f"subgroup_validation={artifacts.validation_json_path}")
    print(f"reference_group_coverage={artifacts.reference_coverage_json_path}")
    print(f"flat_coverage_table={artifacts.flat_table_path}")
    print(
        "preprocessed_manifest_rows="
        f"{artifacts.validation_summary['input_counts']['preprocessed_manifest_rows']}"
    )
    print(
        "geometry_feature_rows="
        f"{artifacts.validation_summary['input_counts']['geometry_feature_rows']}"
    )
    print(
        "reference_group_count="
        f"{artifacts.reference_summary['reference_group_count']}"
    )


if __name__ == "__main__":
    main()
