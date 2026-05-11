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
    print(f"training_summary={artifacts.summary_path}")
    print(f"cv_results={summary['artifacts']['cv_results_path']}")
    print(f"model_selection={summary['artifacts']['model_selection_path']}")
    print(f"selected_features={summary['artifacts']['selected_features_path']}")
    print(f"feature_set_comparison={summary['artifacts']['feature_set_comparison_path']}")
    print(f"pairwise_summary={summary['artifacts']['pairwise_summary_path']}")
    print(f"ordinal_summary={summary['artifacts']['ordinal_summary_path']}")
    print(f"ensemble_summary={summary['artifacts']['ensemble_summary_path']}")
    print(f"global_feature_importance={summary['artifacts']['global_feature_importance_path']}")
    print(f"component_importance={summary['artifacts']['component_importance_path']}")
    print(f"feature_effects={summary['artifacts']['feature_effects_path']}")
    print(f"feature_importance={artifacts.feature_importance_path}")
    print(f"feature_correlations={artifacts.feature_correlations_path}")
    print(f"group_confusion={artifacts.group_confusion_path}")
    print(f"regressor_artifact={artifacts.regressor_artifact_path}")
    print(f"classifier_artifact={artifacts.classifier_artifact_path}")
    print(f"status={summary['status']}")
    print(f"feature_set_name={summary['data']['feature_set_name']}")
    print(f"primary_score_model={summary['primary_score_model']['model_name']}")
    print(f"best_regressor_model={summary['best_regressor_model']}")
    print(f"best_classifier_model={summary['best_classifier_model']}")


if __name__ == "__main__":
    main()
