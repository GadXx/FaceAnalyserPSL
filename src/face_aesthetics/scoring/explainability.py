from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.face_aesthetics.scoring.common import clip_prediction


def _component_map(config: dict[str, Any]) -> dict[str, list[str]]:
    return {str(name): [str(feature) for feature in values] for name, values in dict(config.get("components", {})).items()}


def aggregate_component_contributions(
    feature_contributions: list[dict[str, Any]],
    component_map: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    feature_to_component: dict[str, str] = {}
    for component_name, features in component_map.items():
        for feature_name in features:
            feature_to_component[str(feature_name)] = str(component_name)

    by_component: dict[str, list[dict[str, Any]]] = {}
    for row in feature_contributions:
        feature_name = str(row.get("feature") or "")
        anatomical_component = str(row.get("anatomical_component") or row.get("component") or "").strip()
        if not anatomical_component:
            anatomical_component = feature_to_component.get(feature_name, "unknown")
        by_component.setdefault(anatomical_component, []).append(row)

    rows: dict[str, dict[str, Any]] = {}
    for component_name, members in by_component.items():
        contribution = float(sum(float(item.get("contribution", 0.0) or 0.0) for item in members))
        rows[component_name] = {
            "contribution": contribution,
            "direction": "positive" if contribution > 0 else "negative" if contribution < 0 else "neutral",
            "feature_count": len(members),
            "top_features": [
                {
                    "feature": str(item.get("feature")),
                    "contribution": float(item.get("contribution", 0.0) or 0.0),
                    "anatomical_component": str(item.get("anatomical_component") or component_name),
                    "transform_family": item.get("transform_family"),
                    "reference_transform": item.get("reference_transform"),
                }
                for item in sorted(members, key=lambda item: abs(float(item.get("contribution", 0.0) or 0.0)), reverse=True)[:5]
            ],
        }
    return rows


def _linear_explanation(
    *,
    estimator_payload: dict[str, Any],
    feature_frame: pd.DataFrame,
    predicted_score: float,
) -> dict[str, Any] | None:
    estimator = estimator_payload.get("estimator")
    if estimator is None:
        return None
    pipeline_model = estimator.named_steps["model"] if isinstance(estimator, Pipeline) and "model" in estimator.named_steps else estimator
    scaler = estimator.named_steps.get("scaler") if isinstance(estimator, Pipeline) else None
    if not hasattr(pipeline_model, "coef_"):
        return None
    coefficients = np.asarray(pipeline_model.coef_, dtype=np.float64)
    coefficients = np.mean(coefficients, axis=0) if coefficients.ndim > 1 else coefficients
    feature_columns = list(feature_frame.columns)
    values = feature_frame.iloc[0].to_numpy(dtype=np.float64)
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        scale = np.where(scale == 0, 1.0, scale)
        centered = (values - np.asarray(scaler.mean_, dtype=np.float64)) / scale
    else:
        centered = values
    contributions = centered * coefficients
    intercept = float(np.ravel(np.asarray(getattr(pipeline_model, "intercept_", [0.0]), dtype=np.float64))[0])
    baseline_score = clip_prediction(
        intercept,
        lower=estimator_payload.get("prediction_clip_lower"),
        upper=estimator_payload.get("prediction_clip_upper"),
    )
    rows = [
        {
            "feature": feature_name,
            "value": float(value),
            "contribution": float(contribution),
        }
        for feature_name, value, contribution in zip(feature_columns, values.tolist(), contributions.tolist())
    ]
    rows.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)
    return {
        "baseline_score": float(baseline_score),
        "predicted_score": float(predicted_score),
        "explanation_method": "linear_coefficients",
        "feature_contributions": rows,
    }


def _fallback_explanation(
    *,
    estimator_payload: dict[str, Any],
    feature_frame: pd.DataFrame,
    predicted_score: float,
) -> dict[str, Any]:
    feature_columns = list(feature_frame.columns)
    values = feature_frame.iloc[0].to_numpy(dtype=np.float64)
    means = np.asarray([float(estimator_payload.get("feature_means", {}).get(column, 0.0)) for column in feature_columns], dtype=np.float64)
    stds = np.asarray([float(estimator_payload.get("feature_stds", {}).get(column, 1.0) or 1.0) for column in feature_columns], dtype=np.float64)
    stds = np.where(stds == 0, 1.0, stds)
    centered = (values - means) / stds

    sign_map = {str(row["feature_name"]): float(row.get("signed_effect", 0.0) or 0.0) for row in estimator_payload.get("global_feature_effects", [])}
    magnitude_map = {str(row["feature_name"]): float(row.get("importance", row.get("abs_effect", 0.0)) or 0.0) for row in estimator_payload.get("global_feature_importance", [])}
    raw_scores = np.asarray(
        [
            centered[idx] * (sign_map.get(feature_name, 0.0) or magnitude_map.get(feature_name, 0.0))
            for idx, feature_name in enumerate(feature_columns)
        ],
        dtype=np.float64,
    )
    baseline_score = float(estimator_payload.get("training_target_mean", predicted_score))
    delta = float(predicted_score - baseline_score)
    total = float(raw_scores.sum())
    if abs(total) > 1e-8:
        scale = delta / total
        contributions = raw_scores * scale
    elif len(raw_scores):
        contributions = np.full_like(raw_scores, fill_value=delta / len(raw_scores), dtype=np.float64)
    else:
        contributions = raw_scores
    rows = [
        {
            "feature": feature_name,
            "value": float(value),
            "contribution": float(contribution),
        }
        for feature_name, value, contribution in zip(feature_columns, values.tolist(), contributions.tolist())
    ]
    rows.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)
    return {
        "baseline_score": baseline_score,
        "predicted_score": float(predicted_score),
        "explanation_method": "importance_fallback",
        "feature_contributions": rows,
    }


def build_local_explanation(
    *,
    estimator_payload: dict[str, Any],
    feature_frame: pd.DataFrame,
    predicted_score: float,
    scoring_config: dict[str, Any],
) -> dict[str, Any]:
    explanation = _linear_explanation(
        estimator_payload=estimator_payload,
        feature_frame=feature_frame,
        predicted_score=predicted_score,
    )
    if explanation is None:
        try:
            import shap  # noqa: F401
        except Exception:
            explanation = _fallback_explanation(
                estimator_payload=estimator_payload,
                feature_frame=feature_frame,
                predicted_score=predicted_score,
            )
        else:
            explanation = _fallback_explanation(
                estimator_payload=estimator_payload,
                feature_frame=feature_frame,
                predicted_score=predicted_score,
            )
            explanation["explanation_method"] = "shap_fallback"

    feature_contributions = explanation.get("feature_contributions", [])
    learned_weight_rows = list(estimator_payload.get("learned_feature_weights", []))
    global_weight_map = {
        str(row.get("feature_name")): {
            "global_weight": float(row.get("normalized_abs_weight", row.get("abs_weight", 0.0)) or 0.0),
            "component": str(row.get("anatomical_component") or row.get("component") or ""),
            "anatomical_component": str(row.get("anatomical_component") or row.get("component") or ""),
            "transform_family": str(row.get("transform_family") or ""),
            "reference_transform": row.get("reference_transform"),
            "base_feature_name": row.get("base_feature_name"),
            "source": row.get("source"),
            "original_source": row.get("original_source"),
            "weight_method": row.get("weight_method"),
        }
        for row in learned_weight_rows
    }
    for row in feature_contributions:
        feature_name = str(row.get("feature") or "")
        weight_info = global_weight_map.get(feature_name)
        if weight_info is None:
            continue
        row["global_weight"] = float(weight_info.get("global_weight", 0.0) or 0.0)
        row["component"] = str(weight_info.get("component") or "")
        row["anatomical_component"] = str(weight_info.get("anatomical_component") or "")
        row["transform_family"] = str(weight_info.get("transform_family") or "")
        row["reference_transform"] = weight_info.get("reference_transform")
        row["base_feature_name"] = weight_info.get("base_feature_name")
        row["source"] = weight_info.get("source")
        row["original_source"] = weight_info.get("original_source")
        row["local_contribution"] = float(row.get("contribution", 0.0) or 0.0)
    component_rows = aggregate_component_contributions(feature_contributions, _component_map(scoring_config))
    positive = [row for row in feature_contributions if float(row["contribution"]) > 0]
    negative = [row for row in feature_contributions if float(row["contribution"]) < 0]
    explanation["top_positive_features"] = [
        {
            "feature": str(row["feature"]),
            "contribution": float(row["contribution"]),
            "local_contribution": float(row["contribution"]),
            "global_weight": float(row.get("global_weight", 0.0) or 0.0),
            "component": row.get("component"),
            "anatomical_component": row.get("anatomical_component"),
            "transform_family": row.get("transform_family"),
            "reference_transform": row.get("reference_transform"),
            "base_feature_name": row.get("base_feature_name"),
            "source": row.get("source"),
            "original_source": row.get("original_source"),
        }
        for row in sorted(positive, key=lambda item: float(item["contribution"]), reverse=True)[:5]
    ]
    explanation["top_negative_features"] = [
        {
            "feature": str(row["feature"]),
            "contribution": float(row["contribution"]),
            "local_contribution": float(row["contribution"]),
            "global_weight": float(row.get("global_weight", 0.0) or 0.0),
            "component": row.get("component"),
            "anatomical_component": row.get("anatomical_component"),
            "transform_family": row.get("transform_family"),
            "reference_transform": row.get("reference_transform"),
            "base_feature_name": row.get("base_feature_name"),
            "source": row.get("source"),
            "original_source": row.get("original_source"),
        }
        for row in sorted(negative, key=lambda item: float(item["contribution"]))[:5]
    ]
    explanation["component_contributions"] = component_rows
    transform_family_contributions: dict[str, float] = {}
    for row in feature_contributions:
        family = str(row.get("transform_family") or "unknown")
        transform_family_contributions[family] = transform_family_contributions.get(family, 0.0) + float(
            row.get("contribution", 0.0) or 0.0
        )
    explanation["transform_family_contributions"] = {
        key: float(value) for key, value in sorted(transform_family_contributions.items(), key=lambda item: item[0])
    }
    contribution_sum = float(sum(float(item.get("contribution", 0.0) or 0.0) for item in component_rows.values()))
    prediction_minus_baseline = float(predicted_score - float(explanation.get("baseline_score") or 0.0))
    residual = float(prediction_minus_baseline - contribution_sum)
    explanation["additive_check"] = {
        "contribution_sum": contribution_sum,
        "prediction_minus_baseline": prediction_minus_baseline,
        "contribution_residual": residual,
        "additive_check_passed": bool(abs(residual) <= 1e-3 if explanation.get("explanation_method") == "linear_coefficients" else False),
    }
    return explanation


def global_component_rows(
    feature_rows: list[dict[str, Any]],
    component_map: dict[str, list[str]],
    *,
    value_key: str,
) -> list[dict[str, Any]]:
    by_feature = {str(row["feature_name"]): row for row in feature_rows}
    rows: list[dict[str, Any]] = []
    for component_name, features in component_map.items():
        matched = [by_feature[feature] for feature in features if feature in by_feature]
        if not matched:
            continue
        signed = float(sum(float(row.get(value_key, 0.0) or 0.0) for row in matched))
        magnitude = float(sum(abs(float(row.get(value_key, 0.0) or 0.0)) for row in matched))
        rows.append(
            {
                "component_name": component_name,
                "signed_effect": signed,
                "importance": magnitude,
                "feature_count": len(matched),
                "top_features": [
                    {
                        "feature_name": str(row["feature_name"]),
                        value_key: float(row.get(value_key, 0.0) or 0.0),
                    }
                    for row in sorted(matched, key=lambda item: abs(float(item.get(value_key, 0.0) or 0.0)), reverse=True)[:5]
                ],
            }
        )
    return sorted(rows, key=lambda item: item["importance"], reverse=True)
