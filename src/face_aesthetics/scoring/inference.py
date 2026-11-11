from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.face_aesthetics.scoring.common import clip_prediction, load_scoring_configs, nearest_group_payload, save_json
from src.face_aesthetics.scoring.explainability import build_local_explanation
from src.face_aesthetics.scoring.features import extract_single_image_scoring_features
from src.face_aesthetics.scoring.training import _ensemble_prediction, _ordinal_predict


@dataclass
class ScorerInferenceArtifacts:
    report_json_path: Path
    report_markdown_path: Path
    report: dict[str, Any]


def _markdown_report(report: dict[str, Any]) -> str:
    primary = report["primary_score"]
    explanation = report.get("local_explanation", {})
    lines = [
        "# Experimental Scoring Report",
        "",
        "This report comes from an experimental subjective scoring layer trained on labeled examples.",
        "The score is learned from combinations of morphometric features rather than from hand-written metric ideals.",
        "",
        f"- primary_model: `{primary['model']}`",
        f"- predicted_score: `{primary['predicted_score']}`",
        f"- raw_predicted_score: `{primary.get('raw_predicted_score')}`",
        f"- calibrated_predicted_score: `{primary.get('calibrated_predicted_score')}`",
        f"- nearest_score_group: `{primary['nearest_score_group']}`",
        f"- quality_pass: `{report['quality'].get('quality_pass')}`",
        f"- processing_status: `{report['quality'].get('processing_status')}`",
        f"- confidence_level: `{report['confidence'].get('level')}`",
    ]
    if explanation:
        lines.extend(
            [
                "",
                "## Local Explanation",
                f"- explanation_method: `{explanation.get('explanation_method')}`",
                f"- baseline_score: `{explanation.get('baseline_score')}`",
            ]
        )
    if report.get("warnings"):
        lines.append("")
        lines.append("## Warnings")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _readable_feature_label(feature_name: str) -> str:
    if not isinstance(feature_name, str):
        return str(feature_name)
    mappings = [
        ("ref_z_", "z-score vs reference: "),
        ("ref_dir_", "deviation vs reference: "),
        ("ref_ratio_", "ratio vs reference mean: "),
        ("ref_absdev_", "absolute deviation vs reference: "),
        ("ref_percentile_", "percentile vs reference: "),
    ]
    for prefix, title in mappings:
        if feature_name.startswith(prefix):
            return f"{title}{feature_name[len(prefix):]}"
    return feature_name


def _confidence_payload(
    *,
    primary_score: float,
    raw_primary_score: float,
    pairwise_score: float | None,
    ordinal_score: float | None,
    nearest_group: float | None,
    predicted_group: str | None,
    classifier_top_probability: float | None,
    quality_pass: bool,
    dense_training_range: list[float | None],
    observed_training_label_range: list[float | None],
    score_range: list[float | None],
    group_support: dict[str, int],
    imputed_fraction: float,
    disagreement_threshold: float,
    primary_aux_threshold: float,
    min_top_prob_high: float,
    min_top_prob_medium: float,
    max_imputed_fraction_for_high: float,
    max_imputed_fraction_for_medium: float,
    elite_threshold: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    soft_reasons: list[str] = []
    level = "high"
    if not quality_pass:
        reasons.append("quality_gate_failed")
    lower, upper = dense_training_range
    if lower is not None and upper is not None and (primary_score < float(lower) or primary_score > float(upper)):
        reasons.append("outside_dense_training_range")
    observed_lower, observed_upper = observed_training_label_range
    if observed_lower is not None and primary_score < float(observed_lower):
        reasons.append("unsupported_low_extreme")
    if observed_upper is not None and primary_score > float(observed_upper):
        reasons.append("unsupported_high_extreme")
    score_lower, score_upper = score_range
    if score_lower is not None and score_upper is not None:
        if primary_score <= float(score_lower) + 0.5 or primary_score >= float(score_upper) - 0.5:
            soft_reasons.append("near_scale_extreme")
    if nearest_group is not None:
        nearest_key = str(int(nearest_group)) if float(nearest_group).is_integer() else str(nearest_group)
        if int(group_support.get(nearest_key, 0)) < 5:
            reasons.append("nearest_group_under_supported")
        elif int(group_support.get(nearest_key, 0)) < 8:
            soft_reasons.append("nearest_group_weak_support")
    if predicted_group is not None and nearest_group is not None:
        try:
            if abs(float(predicted_group) - float(nearest_group)) > disagreement_threshold:
                reasons.append("regression_classifier_disagreement")
        except Exception:
            reasons.append("regression_classifier_disagreement")
    if classifier_top_probability is not None:
        if float(classifier_top_probability) < min_top_prob_medium:
            reasons.append("classifier_low_confidence")
        elif float(classifier_top_probability) < min_top_prob_high:
            soft_reasons.append("classifier_moderate_confidence")
    aux_scores = [float(value) for value in (pairwise_score, ordinal_score) if value is not None]
    if aux_scores:
        max_delta = max(abs(primary_score - value) for value in aux_scores)
        if max_delta > primary_aux_threshold:
            reasons.append("primary_auxiliary_disagreement")
        elif max_delta > (primary_aux_threshold * 0.6):
            soft_reasons.append("mild_model_disagreement")
        if ordinal_score is not None and primary_score < elite_threshold - 0.5 and ordinal_score >= elite_threshold:
            reasons.append("possible_high_score_underprediction")
        if pairwise_score is not None and primary_score < elite_threshold - 0.5 and pairwise_score >= elite_threshold:
            reasons.append("possible_high_score_underprediction")
    if abs(primary_score - raw_primary_score) > 0.75:
        soft_reasons.append("raw_calibrated_gap")
    if imputed_fraction > max_imputed_fraction_for_medium:
        reasons.append("many_features_imputed")
    elif imputed_fraction > max_imputed_fraction_for_high:
        soft_reasons.append("some_features_imputed")
    if reasons:
        level = "low"
    elif soft_reasons:
        level = "medium"
    return {"level": level, "reasons": reasons + soft_reasons}


def run_scorer_inference(
    *,
    image_path: Path,
    project_root: Path | None = None,
    reference_filters: dict[str, Any] | None = None,
) -> ScorerInferenceArtifacts:
    root = (project_root or Path.cwd()).resolve()
    configs = load_scoring_configs(root)
    config = configs["scoring"]["scoring"]

    regressor_artifact_path = (root / str(config["regressor_artifact_path"])).resolve()
    classifier_artifact_path = (root / str(config["classifier_artifact_path"])).resolve()
    pairwise_artifact_path = (root / "models/scoring/pairwise_ranker.joblib").resolve()
    ordinal_artifact_path = (root / "models/scoring/ordinal_scorer.joblib").resolve()
    report_json_path = (root / str(config["report_paths"]["json_path"])).resolve()
    report_markdown_path = (root / str(config["report_paths"]["markdown_path"])).resolve()
    disagreement_summary_path = (root / "reports/metrics/scorer_diagnostic_disagreement_summary.json").resolve()
    analysis_dir = (root / str(config["processed_analysis_dir"])).resolve()
    aligned_dir = (root / str(config["processed_aligned_dir"])).resolve()

    regressor_payload = joblib.load(regressor_artifact_path)
    classifier_payload = joblib.load(classifier_artifact_path) if classifier_artifact_path.exists() else None
    pairwise_payload = joblib.load(pairwise_artifact_path) if pairwise_artifact_path.exists() else None
    ordinal_payload = joblib.load(ordinal_artifact_path) if ordinal_artifact_path.exists() else None

    feature_row = extract_single_image_scoring_features(
        image_path=image_path.resolve(),
        image_id=f"scoring_inference::{image_path.stem}",
        target_score=0.0,
        target_group_label="inference_only",
        split="inference",
        source_score_folder="inference_only",
        file_name=image_path.name,
        project_root=root,
        output_analysis_path=analysis_dir / f"scoring_inference::{image_path.stem}.jpg",
        output_aligned_path=aligned_dir / f"scoring_inference::{image_path.stem}.jpg",
        reference_filters=reference_filters,
    )

    feature_columns = list(regressor_payload["selected_features"])
    feature_frame = pd.DataFrame([feature_row]).reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    missing_features = [column for column in feature_columns if pd.isna(feature_frame.loc[0, column])]
    imputed_fraction = float(len(missing_features) / len(feature_columns)) if feature_columns else 0.0
    medians = dict(regressor_payload.get("feature_medians", {}))
    for column in feature_columns:
        if pd.isna(feature_frame.loc[0, column]):
            feature_frame.loc[0, column] = medians.get(column)
    X = feature_frame.to_numpy(dtype=np.float64)

    raw_prediction = float(regressor_payload["estimator"].predict(X)[0])
    calibrator = regressor_payload.get("score_calibrator")
    if calibrator is not None:
        calibrated_prediction = float(calibrator.predict(np.asarray([raw_prediction], dtype=np.float64))[0])
    else:
        calibrated_prediction = raw_prediction
    artifact_uses_calibration = bool(regressor_payload.get("primary_uses_calibration", False))
    raw_score = clip_prediction(
        raw_prediction,
        lower=regressor_payload.get("prediction_clip_lower"),
        upper=regressor_payload.get("prediction_clip_upper"),
    )
    calibrated_score = clip_prediction(
        calibrated_prediction,
        lower=regressor_payload.get("prediction_clip_lower"),
        upper=regressor_payload.get("prediction_clip_upper"),
    )
    predicted_score = float(raw_score)
    score_source = "raw_continuous"
    calibration_used_as_primary = False
    calibration_used_for_diagnostics = calibrator is not None
    display_score = round(float(predicted_score), 2)
    group_payload = nearest_group_payload(predicted_score, [float(value) for value in regressor_payload.get("score_groups", [])])

    group_probabilities: dict[str, float] = {}
    predicted_group = None
    top_k_groups: list[dict[str, Any]] = []
    classifier_top_probability = None
    if classifier_payload is not None:
        estimator = classifier_payload["estimator"]
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(X)[0]
            classes = [str(value) for value in classifier_payload["classes"]]
            group_probabilities = {label: float(prob) for label, prob in zip(classes, probabilities.tolist())}
            top_k_groups = [{"group": label, "probability": probability} for label, probability in sorted(group_probabilities.items(), key=lambda item: item[1], reverse=True)[:3]]
            classifier_top_probability = float(max(probabilities.tolist())) if len(probabilities) else None
        predicted_group = str(estimator.predict(X)[0])

    warnings: list[str] = []

    pairwise_block: dict[str, Any] = {}
    pairwise_score = None
    if pairwise_payload is not None:
        reference_examples = pd.DataFrame(pairwise_payload.get("reference_examples", []))
        if not reference_examples.empty:
            repeated = pd.DataFrame([feature_frame.loc[0, feature_columns].to_dict()] * len(reference_examples))
            diff = repeated.to_numpy(dtype=np.float64) - reference_examples.loc[:, feature_columns].to_numpy(dtype=np.float64)
            estimator = pairwise_payload["estimator"]
            if hasattr(estimator, "predict_proba"):
                probs = estimator.predict_proba(diff)[:, 1]
            else:
                decision = estimator.decision_function(diff)
                probs = 1.0 / (1.0 + np.exp(-decision))
            ref_scores = reference_examples["target_score"].astype(float).to_numpy(dtype=np.float64)
            pairwise_score = float(np.average(ref_scores, weights=probs + 1e-6))
            pairwise_group = nearest_group_payload(pairwise_score, [float(value) for value in pairwise_payload.get("score_groups", [])])
            pairwise_block = {
                "ranking_score": float(np.mean(probs)),
                "approx_score": pairwise_score,
                "nearest_score_group": pairwise_group["nearest_score_group"],
                "comparison_count": int(len(reference_examples)),
                "confidence": "medium",
                "use_for_absolute_scoring": bool(pairwise_payload.get("use_for_absolute_scoring", False)),
            }
            if not bool(pairwise_payload.get("use_for_absolute_scoring", False)):
                warnings.append("pairwise_score_poorly_calibrated")

    ordinal_block: dict[str, Any] = {}
    ordinal_score = None
    if ordinal_payload is not None:
        ordinal_pred, threshold_probabilities = _ordinal_predict(ordinal_payload, X)
        if len(ordinal_pred):
            ordinal_score = float(ordinal_pred[0])
            ordinal_group = nearest_group_payload(ordinal_score, [float(value) for value in ordinal_payload.get("score_groups", [])])
            ordinal_block = {
                "predicted_score": ordinal_score,
                "nearest_score_group": ordinal_group["nearest_score_group"],
                "threshold_probabilities": threshold_probabilities[0] if threshold_probabilities else {},
            }

    if missing_features:
        warnings.append("Some morphometric features were unavailable at inference time and were imputed from training medians: " + ", ".join(sorted(missing_features)))
    if not bool(feature_row.get("quality_pass")):
        warnings.append("Unable to reliably pass the quality gate for this image. Experimental score inference may be unstable.")
    if calibrator is None:
        warnings.append("primary_score_calibration_unavailable")
    warnings.extend(str(value) for value in regressor_payload.get("warnings", []))
    if bool(regressor_payload.get("angles_pose_weight_cap_exceeded", False)):
        warnings.append("score_may_be_pose_sensitive")

    confidence_config = dict(config.get("confidence", {}))
    pairwise_for_confidence = pairwise_score if pairwise_block.get("use_for_absolute_scoring", False) else None
    confidence = _confidence_payload(
        primary_score=predicted_score,
        raw_primary_score=float(raw_prediction),
        pairwise_score=pairwise_for_confidence,
        ordinal_score=ordinal_score,
        nearest_group=group_payload["nearest_score_group"],
        predicted_group=predicted_group,
        classifier_top_probability=classifier_top_probability,
        quality_pass=bool(feature_row.get("quality_pass")),
        dense_training_range=list(regressor_payload.get("dense_training_range", [None, None])),
        observed_training_label_range=list(regressor_payload.get("observed_training_label_range", [None, None])),
        score_range=[regressor_payload.get("prediction_clip_lower"), regressor_payload.get("prediction_clip_upper")],
        group_support=dict(regressor_payload.get("group_support", {})),
        imputed_fraction=imputed_fraction,
        disagreement_threshold=float(confidence_config.get("max_regression_classifier_disagreement", 1.5)),
        primary_aux_threshold=float(confidence_config.get("max_primary_aux_disagreement", 1.25)),
        min_top_prob_high=float(confidence_config.get("min_classifier_top_probability_for_high", 0.45)),
        min_top_prob_medium=float(confidence_config.get("min_classifier_top_probability_for_medium", 0.25)),
        max_imputed_fraction_for_high=float(confidence_config.get("max_imputed_fraction_for_high", 0.05)),
        max_imputed_fraction_for_medium=float(confidence_config.get("max_imputed_fraction_for_medium", 0.2)),
        elite_threshold=float(config.get("strict_scoring", {}).get("elite_score_threshold", 9.0)),
    )
    if pairwise_score is not None and pairwise_for_confidence is None:
        pairwise_delta = abs(float(predicted_score) - float(pairwise_score))
        if pairwise_delta > float(confidence_config.get("max_primary_aux_disagreement", 1.25)):
            if confidence["level"] == "high":
                confidence["level"] = "medium"
            if "pairwise_ranking_disagreement" not in confidence["reasons"]:
                confidence["reasons"].append("pairwise_ranking_disagreement")
    for reason in confidence["reasons"]:
        if reason == "outside_dense_training_range":
            warnings.append("Prediction falls outside the dense training range and may be weakly supported.")
        elif reason == "nearest_group_under_supported":
            warnings.append("Nearest score group has weak support in the training data.")
        elif reason == "regression_classifier_disagreement":
            warnings.append("Regression score and classifier group disagree strongly.")
        elif reason == "classifier_low_confidence":
            warnings.append("Classifier diagnostic has low confidence for the nearest score group.")
        elif reason == "many_features_imputed":
            warnings.append("Many selected morphometric features were imputed from training medians.")
        elif reason == "primary_auxiliary_disagreement":
            warnings.append("Primary regression and auxiliary scorers disagree strongly.")
        elif reason == "possible_high_score_underprediction":
            warnings.append("Auxiliary scorers suggest a higher-end pattern than the primary score currently reflects.")
        elif reason == "unsupported_low_extreme":
            warnings.append("Prediction is near an unsupported low extreme outside observed training labels.")
        elif reason == "unsupported_high_extreme":
            warnings.append("Prediction is near an unsupported high extreme outside observed training labels.")
        elif reason == "model_disagreement":
            warnings.append("Primary regression and auxiliary scorers disagree strongly.")
        elif reason == "pairwise_ranking_disagreement":
            warnings.append("Pairwise ranking diagnostic diverges from primary score.")

    inferred_target_group = None
    try:
        inferred_target_group = float(str(image_path.parent.name).strip())
    except Exception:
        inferred_target_group = None
    if bool(feature_row.get("quality_pass")) and inferred_target_group is not None and inferred_target_group >= 8.5:
        if float(predicted_score) < 7.0:
            warnings.append("high_score_collapse_risk")
            confidence["level"] = "low"
        if float(predicted_score) < 6.0:
            warnings.append("severe_high_score_collapse_detected")
            confidence["level"] = "low"

    ensemble_weights = {
        "regression": float(config.get("ensemble", {}).get("regression_weight", 0.45)),
        "pairwise": float(config.get("ensemble", {}).get("pairwise_weight", 0.35)),
        "ordinal": float(config.get("ensemble", {}).get("ordinal_weight", 0.20)),
    }
    final_score, final_std = _ensemble_prediction(
        {
            "regression": predicted_score,
            "pairwise": pairwise_score if pairwise_block.get("use_for_absolute_scoring", False) else None,
            "ordinal": ordinal_score,
        },
        ensemble_weights,
    )
    if final_std is not None and final_std > float(config.get("ensemble", {}).get("disagreement_std_threshold", 0.75)):
        warnings.append("Model disagreement across regression/pairwise/ordinal scorers is high.")
        if "model_disagreement" not in confidence["reasons"]:
            confidence["reasons"].append("model_disagreement")
            confidence["level"] = "low"

    local_explanation: dict[str, Any] | None = None
    try:
        local_explanation = build_local_explanation(
            estimator_payload=regressor_payload,
            feature_frame=feature_frame,
            predicted_score=float(predicted_score),
            scoring_config=config,
        )
        if local_explanation is not None:
            local_explanation["explained_score_kind"] = "raw_continuous_model_score"
            for row in local_explanation.get("top_positive_features", []):
                row["readable_label"] = _readable_feature_label(str(row.get("feature")))
            for row in local_explanation.get("top_negative_features", []):
                row["readable_label"] = _readable_feature_label(str(row.get("feature")))
            for row in local_explanation.get("feature_contributions", []):
                row["readable_label"] = _readable_feature_label(str(row.get("feature")))
            local_explanation["feature_global_weights"] = [
                {
                    "feature_name": row.get("feature_name"),
                    "component": row.get("anatomical_component") or row.get("component"),
                    "anatomical_component": row.get("anatomical_component") or row.get("component"),
                    "transform_family": row.get("transform_family"),
                    "reference_transform": row.get("reference_transform"),
                    "base_feature_name": row.get("base_feature_name"),
                    "source": row.get("source"),
                    "original_source": row.get("original_source"),
                    "normalized_weight": row.get("normalized_abs_weight"),
                    "signed_weight": row.get("signed_weight"),
                }
                for row in list(regressor_payload.get("learned_feature_weights", []))[:40]
            ]
    except Exception:
        warnings.append("explanation_unavailable")
    top_contributing_features = (
        (local_explanation.get("top_positive_features", []) + local_explanation.get("top_negative_features", [])[:5])
        if local_explanation
        else []
    )
    learned_feature_weights = list(regressor_payload.get("learned_feature_weights", []))
    learned_component_weights = list(
        regressor_payload.get("learned_anatomical_component_weights", regressor_payload.get("learned_component_weights", []))
    )
    learned_transform_family_weights = list(regressor_payload.get("learned_transform_family_weights", []))
    learned_source_weights = list(regressor_payload.get("learned_source_weights", []))
    learned_weights_payload = {
        "component_weights": [
            {
                "component": row.get("component"),
                "normalized_weight": row.get("normalized_component_weight"),
                "signed_sum": row.get("signed_sum"),
                "feature_count": row.get("feature_count"),
            }
            for row in learned_component_weights
        ],
        "anatomical_component_weights": [
            {
                "component": row.get("component"),
                "normalized_weight": row.get("normalized_component_weight"),
                "signed_sum": row.get("signed_sum"),
                "feature_count": row.get("feature_count"),
            }
            for row in learned_component_weights
        ],
        "transform_family_weights": [
            {
                "transform_family": row.get("transform_family"),
                "normalized_weight": row.get("normalized_weight"),
                "signed_sum": row.get("signed_sum"),
                "feature_count": row.get("feature_count"),
            }
            for row in learned_transform_family_weights
        ],
        "source_weights": [
            {
                "source_family": row.get("source_family"),
                "normalized_weight": row.get("normalized_weight"),
                "signed_sum": row.get("signed_sum"),
                "feature_count": row.get("feature_count"),
            }
            for row in learned_source_weights
        ],
        "top_feature_weights": [
            {
                "feature_name": row.get("feature_name"),
                "component": row.get("anatomical_component") or row.get("component"),
                "anatomical_component": row.get("anatomical_component") or row.get("component"),
                "transform_family": row.get("transform_family"),
                "reference_transform": row.get("reference_transform"),
                "source": row.get("source"),
                "original_source": row.get("original_source"),
                "base_feature_name": row.get("base_feature_name"),
                "normalized_weight": row.get("normalized_abs_weight"),
                "signed_weight": row.get("signed_weight"),
            }
            for row in learned_feature_weights[:20]
        ],
    }
    top_mediapipe_contributors = [
        item for item in top_contributing_features if str(item.get("feature", "")).startswith("mediapipe_")
    ][:10]
    primary_feature_set = str(regressor_payload.get("feature_set_name") or "")
    expanded_geometry_used_in_primary = bool(
        primary_feature_set in {"mediapipe_stable_geometry", "combined_stable_geometry", "combined_no_duplicate_geometry", "expanded_mediapipe_geometry_forced"}
        or int(regressor_payload.get("mediapipe_features_used", 0) or 0) > 0
    )
    direct_mediapipe_features_used = int(regressor_payload.get("direct_mediapipe_features_used", regressor_payload.get("mediapipe_features_used", 0)) or 0)
    mediapipe_derived_reference_features_used = int(regressor_payload.get("mediapipe_derived_reference_features_used", 0) or 0)
    total_mediapipe_influence_count = int(
        regressor_payload.get(
            "total_mediapipe_influence_count",
            direct_mediapipe_features_used + mediapipe_derived_reference_features_used,
        )
        or 0
    )
    primary_score = {
        "predicted_score": float(predicted_score),
        "score_source": score_source,
        "uses_calibration": bool(artifact_uses_calibration),
        "calibration_used_as_primary": bool(calibration_used_as_primary),
        "calibration_used_for_diagnostics": bool(calibration_used_for_diagnostics),
        "display_score": float(display_score),
        "predicted_score_raw": float(raw_score),
        "raw_predicted_score": float(raw_score),
        "calibrated_predicted_score": float(calibrated_score),
        "calibration_method": regressor_payload.get("calibration_method"),
        "model": str(regressor_payload.get("model_name")),
        "model_family": "regression",
        "nearest_score_group": group_payload["nearest_score_group"],
        "distance_to_nearest_group": group_payload["distance_to_nearest_group"],
        "score_range": [regressor_payload.get("prediction_clip_lower"), regressor_payload.get("prediction_clip_upper")],
        "observed_training_label_range": list(regressor_payload.get("observed_training_label_range", [None, None])),
        "feature_set_name": regressor_payload.get("feature_set_name"),
        "primary_feature_set": primary_feature_set or None,
        "selected_feature_source_counts": dict(regressor_payload.get("selected_feature_source_counts", {})),
        "mediapipe_features_used": direct_mediapipe_features_used,
        "direct_mediapipe_features_used": direct_mediapipe_features_used,
        "mediapipe_derived_reference_features_used": mediapipe_derived_reference_features_used,
        "total_mediapipe_influence_count": total_mediapipe_influence_count,
        "reference_relative_features_used": int(regressor_payload.get("reference_relative_features_used", 0) or 0),
        "expanded_geometry_available": True,
        "expanded_geometry_used_in_primary": expanded_geometry_used_in_primary,
        "top_mediapipe_contributors": top_mediapipe_contributors,
        "active_feature_source_policy": dict(regressor_payload.get("active_feature_source_policy", {})),
    }
    scoring_model_info = {
        "primary_model": str(regressor_payload.get("model_name")),
        "primary_feature_set": primary_feature_set or regressor_payload.get("feature_set_name"),
        "uses_mediapipe": bool(int(regressor_payload.get("mediapipe_features_used", 0) or 0) > 0),
        "uses_reference_relative": bool(int(regressor_payload.get("reference_relative_features_used", 0) or 0) > 0),
        "weight_method": regressor_payload.get("learned_weight_method"),
        "selected_feature_count": len(feature_columns),
        "selected_component_counts": dict(
            regressor_payload.get(
                "selected_anatomical_component_counts",
                regressor_payload.get("selected_component_counts", {}),
            )
        ),
        "selected_transform_family_counts": dict(regressor_payload.get("selected_transform_family_counts", {})),
        "selected_source_counts": dict(regressor_payload.get("selected_source_counts", {})),
        "active_feature_source_policy": dict(regressor_payload.get("active_feature_source_policy", {})),
    }
    auxiliary_scores = {
        "regression": {
            "predicted_score": float(predicted_score),
            "nearest_score_group": group_payload["nearest_score_group"],
            "distance_to_nearest_group": group_payload["distance_to_nearest_group"],
        },
        "pairwise": pairwise_block,
        "ordinal": ordinal_block,
        "classifier": {
            "predicted_group": predicted_group,
            "group_probabilities": group_probabilities,
            "top_k_groups": top_k_groups,
        },
        "ensemble": {
            "predicted_score": float(final_score) if final_score is not None else None,
            "nearest_score_group": nearest_group_payload(float(final_score), [float(value) for value in regressor_payload.get("score_groups", [])])["nearest_score_group"] if final_score is not None else None,
            "std": float(final_std) if final_std is not None else None,
        },
    }
    report = {
        "primary_score": primary_score,
        "auxiliary_scores": auxiliary_scores,
        "regression": {
            "predicted_score": float(predicted_score),
            "score_source": score_source,
            "uses_calibration": bool(artifact_uses_calibration),
            "calibration_used_as_primary": bool(calibration_used_as_primary),
            "calibration_used_for_diagnostics": bool(calibration_used_for_diagnostics),
            "display_score": float(display_score),
            "predicted_score_raw": float(raw_score),
            "raw_predicted_score": float(raw_score),
            "calibrated_predicted_score": float(calibrated_score),
            "calibration_method": regressor_payload.get("calibration_method"),
            "nearest_score_group": group_payload["nearest_score_group"],
            "distance_to_nearest_group": group_payload["distance_to_nearest_group"],
            "distances_to_groups": group_payload["distances_to_groups"],
            "soft_group_membership": group_payload["soft_group_membership"],
            "score_range": [regressor_payload.get("prediction_clip_lower"), regressor_payload.get("prediction_clip_upper")],
            "observed_training_label_range": list(regressor_payload.get("observed_training_label_range", [None, None])),
        },
        "pairwise": pairwise_block,
        "ordinal": ordinal_block,
        "final_score": {
            "predicted_score": float(final_score) if final_score is not None else None,
            "nearest_score_group": nearest_group_payload(float(final_score), [float(value) for value in regressor_payload.get("score_groups", [])])["nearest_score_group"] if final_score is not None else None,
            "confidence": confidence["level"],
            "model_agreement": {
                "regression": float(predicted_score),
                "pairwise": float(pairwise_score) if pairwise_score is not None else None,
                "ordinal": float(ordinal_score) if ordinal_score is not None else None,
                "std": float(final_std) if final_std is not None else None,
            },
        },
        "classification": {
            "predicted_group": predicted_group,
            "group_probabilities": group_probabilities,
            "top_k_groups": top_k_groups,
        },
        "local_explanation": local_explanation,
        "learned_weighting": {
            "feature_weights": learned_feature_weights,
            "component_weights": learned_component_weights,
            "top_weighted_components": learned_component_weights[:10],
            "top_weighted_features": learned_feature_weights[:20],
            "transform_family_weights": learned_transform_family_weights,
            "source_weights": learned_source_weights,
        },
        "learned_weights": learned_weights_payload,
        "scoring_model_info": scoring_model_info,
        "component_contributions": local_explanation.get("component_contributions", {}) if local_explanation else {},
        "explanation_method": local_explanation.get("explanation_method") if local_explanation else None,
        "feature_set_name": regressor_payload.get("feature_set_name"),
        "selected_feature_source_counts": dict(regressor_payload.get("selected_feature_source_counts", {})),
        "mediapipe_features_used": direct_mediapipe_features_used,
        "direct_mediapipe_features_used": direct_mediapipe_features_used,
        "mediapipe_derived_reference_features_used": mediapipe_derived_reference_features_used,
        "total_mediapipe_influence_count": total_mediapipe_influence_count,
        "reference_relative_features_used": int(regressor_payload.get("reference_relative_features_used", 0) or 0),
        "primary_feature_set": primary_feature_set or None,
        "expanded_geometry_available": True,
        "expanded_geometry_used_in_primary": expanded_geometry_used_in_primary,
        "top_mediapipe_contributors": top_mediapipe_contributors,
        "active_feature_source_policy": dict(regressor_payload.get("active_feature_source_policy", {})),
        "reference_context": {
            "source": feature_row.get("reference_source"),
            "stats_path": feature_row.get("reference_stats_path"),
            "filters": feature_row.get("reference_filters"),
            "requested_group": feature_row.get("reference_requested_group"),
            "resolved_group": feature_row.get("reference_resolved_group"),
            "fallback_used": feature_row.get("reference_fallback_used"),
            "reference_sample_count": feature_row.get("reference_sample_count"),
        },
        "quality": {
            "quality_pass": bool(feature_row.get("quality_pass")),
            "processing_status": feature_row.get("processing_status"),
            "processing_error_message": feature_row.get("processing_error_message"),
        },
        "confidence": confidence,
        "evaluation_context": {
            "model_selected_by": regressor_payload.get("model_selected_by"),
            "test_metrics_available": bool(regressor_payload.get("test_metrics_available", False)),
        },
        "top_contributing_features": top_contributing_features,
        "warnings": warnings,
    }
    disagreement_payload = {
        "primary_vs_pairwise_delta": float(abs(predicted_score - pairwise_score)) if pairwise_score is not None else None,
        "primary_vs_ordinal_delta": float(abs(predicted_score - ordinal_score)) if ordinal_score is not None else None,
        "primary_vs_ensemble_delta": float(abs(predicted_score - final_score)) if final_score is not None else None,
        "primary_vs_classifier_group_delta": float(abs(predicted_score - float(predicted_group))) if predicted_group is not None else None,
        "pairwise_calibration_reliability": "low" if pairwise_block and not pairwise_block.get("use_for_absolute_scoring", False) else "medium_or_high",
        "pairwise_used_for_confidence": bool(pairwise_for_confidence is not None),
    }
    save_json({"status": "completed", "disagreement": disagreement_payload}, disagreement_summary_path)
    report["diagnostic_disagreement"] = disagreement_payload
    concept_alignment_path = (root / "reports/metrics/scorer_concept_alignment_audit.json").resolve()
    selected_feature_names = [str(name) for name in regressor_payload.get("selected_features", [])]
    non_mediapipe_selected = [
        name
        for name in selected_feature_names
        if not (name.startswith("mediapipe_") or name.startswith("ref_") and "_mediapipe_" in name)
    ]
    direct_mediapipe = [name for name in selected_feature_names if name.startswith("mediapipe_")]
    ref_mediapipe = [name for name in selected_feature_names if name.startswith("ref_") and "_mediapipe_" in name]
    anatomical_weights = learned_weights_payload.get("anatomical_component_weights", [])
    transform_weights = learned_weights_payload.get("transform_family_weights", [])
    concept_alignment_payload = {
        "status": "completed",
        "active_feature_source_policy": dict(regressor_payload.get("active_feature_source_policy", {})),
        "selected_feature_count": len(selected_feature_names),
        "direct_mediapipe_selected_count": len(direct_mediapipe),
        "mediapipe_reference_relative_selected_count": len(ref_mediapipe),
        "absolute_mediapipe_selected_count": len(direct_mediapipe),
        "non_mediapipe_selected_count": len(non_mediapipe_selected),
        "non_mediapipe_selected_features": non_mediapipe_selected,
        "reference_relative_ratio": float(len(ref_mediapipe) / len(selected_feature_names)) if selected_feature_names else 0.0,
        "top_anatomical_component_weights": anatomical_weights[:10],
        "top_transform_family_weights": transform_weights[:10],
    }
    save_json(concept_alignment_payload, concept_alignment_path)
    policy_consistency_path = (root / "reports/metrics/scorer_primary_policy_consistency.json").resolve()
    policy_consistency_payload = {
        "status": "completed",
        "artifact_primary_uses_calibration": bool(artifact_uses_calibration),
        "runtime_score_source": score_source,
        "api_score_source": score_source,
        "ui_score_source": score_source,
        "consistency_passed": bool(score_source == "raw_continuous"),
    }
    save_json(policy_consistency_payload, policy_consistency_path)
    save_json(report, report_json_path)
    report_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return ScorerInferenceArtifacts(report_json_path=report_json_path, report_markdown_path=report_markdown_path, report=report)
