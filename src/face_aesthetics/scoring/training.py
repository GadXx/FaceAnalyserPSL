from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.face_aesthetics.scoring.common import (
    clip_prediction,
    feature_policy_report,
    inverse_frequency_weights,
    load_scoring_configs,
    nearest_group_payload,
    numeric_feature_frame,
    save_json,
    scoring_feature_columns,
)
from src.face_aesthetics.scoring.explainability import global_component_rows
from src.face_aesthetics.features.extraction import feature_registry_rows, geometry_feature_columns
from src.face_aesthetics.scoring.reference import (
    REFERENCE_FEATURE_PREFIXES,
    append_reference_relative_features,
    compute_reference_stats,
    reference_relative_feature_names,
)
from src.face_aesthetics.utils.filesystem import ensure_directory


@dataclass
class ScorerTrainingArtifacts:
    summary_path: Path
    feature_importance_path: Path
    feature_correlations_path: Path
    group_confusion_path: Path
    regressor_artifact_path: Path
    classifier_artifact_path: Path
    summary: dict[str, Any]


ANATOMICAL_COMPONENT_ALIASES: dict[str, str] = {
    "cheekbones": "cheekbones_midface",
}

KNOWN_ANATOMICAL_COMPONENTS: set[str] = {
    "face_shape",
    "face_oval",
    "eyes",
    "brows",
    "nose",
    "mouth_lips",
    "cheekbones_midface",
    "jaw_chin",
    "symmetry",
    "vertical_proportions",
    "horizontal_proportions",
    "angles_pose",
}


def _canonical_anatomical_component(component_name: str | None) -> str:
    raw = str(component_name or "").strip()
    if not raw:
        return "unknown"
    return ANATOMICAL_COMPONENT_ALIASES.get(raw, raw)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    if y_true.size == 0:
        return {"mae": None, "rmse": None, "r2": None, "pearson": None, "spearman": None}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred)) if y_true.size >= 2 else None
    pearson = None
    spearman = None
    if y_true.size >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(pearsonr(y_true, y_pred).statistic)
        spearman = float(spearmanr(y_true, y_pred).statistic)
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson, "spearman": spearman}


def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: list[float]) -> dict[str, float | None]:
    if y_true.size == 0:
        return {
            "nearest_group_accuracy": None,
            "within_0_5_accuracy": None,
            "within_1_0_accuracy": None,
            "mean_absolute_group_distance": None,
        }
    nearest_true = np.asarray([nearest_group_payload(float(value), groups)["nearest_score_group"] for value in y_true], dtype=np.float64)
    nearest_pred = np.asarray([nearest_group_payload(float(value), groups)["nearest_score_group"] for value in y_pred], dtype=np.float64)
    group_distance = np.abs(nearest_pred - nearest_true)
    return {
        "nearest_group_accuracy": float(np.mean(nearest_pred == nearest_true)),
        "within_0_5_accuracy": float(np.mean(np.abs(y_pred - y_true) <= 0.5)),
        "within_1_0_accuracy": float(np.mean(np.abs(y_pred - y_true) <= 1.0)),
        "mean_absolute_group_distance": float(np.mean(group_distance)),
    }


def _classifier_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None, classes: list[str]) -> dict[str, Any]:
    if y_true.size == 0:
        return {"accuracy": None, "balanced_accuracy": None, "top_2_accuracy": None, "confusion_matrix": [], "labels": classes}
    top_2_accuracy = None
    if probabilities is not None and probabilities.size:
        top2 = np.argsort(probabilities, axis=1)[:, -2:]
        top2_labels = np.asarray(classes, dtype=object)[top2]
        top_2_accuracy = float(np.mean([truth in candidates for truth, candidates in zip(y_true.tolist(), top2_labels.tolist())]))
    matrix_array = confusion_matrix(y_true, y_pred, labels=classes)
    class_support = matrix_array.sum(axis=1)
    recall_values = []
    for class_index, support in enumerate(class_support):
        if support > 0:
            recall_values.append(float(matrix_array[class_index, class_index] / support))
    balanced_accuracy = float(np.mean(recall_values)) if recall_values else None
    matrix = matrix_array.tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced_accuracy,
        "top_2_accuracy": top_2_accuracy,
        "confusion_matrix": matrix,
        "labels": classes,
    }


def _build_regressors(config: dict[str, Any], random_seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {}
    if config["ridge"]["enabled"]:
        models["ridge"] = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=float(config["ridge"]["alpha"])))])
    if config["elastic_net"]["enabled"]:
        models["elastic_net"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=float(config["elastic_net"]["alpha"]), l1_ratio=float(config["elastic_net"]["l1_ratio"]), random_state=random_seed)),
            ]
        )
    if config["random_forest"]["enabled"]:
        models["random_forest"] = RandomForestRegressor(
            n_estimators=int(config["random_forest"]["n_estimators"]),
            max_depth=int(config["random_forest"]["max_depth"]),
            min_samples_leaf=int(config["random_forest"]["min_samples_leaf"]),
            random_state=random_seed,
            n_jobs=1,
        )
    if config["gradient_boosting"]["enabled"]:
        models["gradient_boosting"] = GradientBoostingRegressor(
            n_estimators=int(config["gradient_boosting"]["n_estimators"]),
            learning_rate=float(config["gradient_boosting"]["learning_rate"]),
            max_depth=int(config["gradient_boosting"]["max_depth"]),
            random_state=random_seed,
        )
    if config.get("hist_gradient_boosting", {}).get("enabled", False):
        models["hist_gradient_boosting"] = HistGradientBoostingRegressor(
            max_depth=int(config["hist_gradient_boosting"]["max_depth"]),
            learning_rate=float(config["hist_gradient_boosting"]["learning_rate"]),
            max_iter=int(config["hist_gradient_boosting"]["max_iter"]),
            random_state=random_seed,
        )
    if config.get("xgboost", {}).get("enabled", False):
        try:
            from xgboost import XGBRegressor

            models["xgboost"] = XGBRegressor(
                n_estimators=int(config["xgboost"]["n_estimators"]),
                max_depth=int(config["xgboost"]["max_depth"]),
                learning_rate=float(config["xgboost"]["learning_rate"]),
                subsample=float(config["xgboost"]["subsample"]),
                colsample_bytree=float(config["xgboost"]["colsample_bytree"]),
                reg_alpha=float(config["xgboost"]["reg_alpha"]),
                reg_lambda=float(config["xgboost"]["reg_lambda"]),
                random_state=random_seed,
                objective="reg:squarederror",
                n_jobs=1,
            )
        except Exception:
            pass
    return models


def _build_classifiers(config: dict[str, Any], random_seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {}
    if config["logistic_regression"]["enabled"]:
        models["logistic_regression"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=int(config["logistic_regression"]["max_iter"]), C=float(config["logistic_regression"]["c"]), random_state=random_seed)),
            ]
        )
    if config["random_forest"]["enabled"]:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=int(config["random_forest"]["n_estimators"]),
            max_depth=int(config["random_forest"]["max_depth"]),
            min_samples_leaf=int(config["random_forest"]["min_samples_leaf"]),
            random_state=random_seed,
            n_jobs=1,
        )
    if config["gradient_boosting"]["enabled"]:
        models["gradient_boosting"] = GradientBoostingClassifier(
            n_estimators=int(config["gradient_boosting"]["n_estimators"]),
            learning_rate=float(config["gradient_boosting"]["learning_rate"]),
            max_depth=int(config["gradient_boosting"]["max_depth"]),
            random_state=random_seed,
        )
    return models


def _fit_estimator(model: Any, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> Any:
    cloned = clone(model)
    if sample_weight is None:
        cloned.fit(X, y)
        return cloned
    if isinstance(cloned, Pipeline):
        cloned.fit(X, y, model__sample_weight=sample_weight)
    else:
        cloned.fit(X, y, sample_weight=sample_weight)
    return cloned


def _prediction_array(estimator: Any, X: np.ndarray, *, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    if X.size == 0:
        return np.asarray([], dtype=np.float64)
    predicted = np.asarray(estimator.predict(X), dtype=np.float64)
    return np.asarray([clip_prediction(value, lower=lower, upper=upper) for value in predicted], dtype=np.float64)


def _selection_value(cv_summary: dict[str, Any], primary_metric: str, secondary_metric: str | None) -> tuple[float, float]:
    metrics = cv_summary.get("mean_metrics", {})
    primary = metrics.get(primary_metric)
    primary_value = float(primary) if primary is not None else float("inf")
    secondary_value = float("-inf")
    if secondary_metric:
        secondary = metrics.get(secondary_metric)
        secondary_value = float(secondary) if secondary is not None else float("-inf")
    return (primary_value, -secondary_value)


def _correlation_rows(frame: pd.DataFrame, feature_columns: list[str], target_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target = pd.to_numeric(frame[target_column], errors="coerce")
    for feature_column in feature_columns:
        feature = pd.to_numeric(frame[feature_column], errors="coerce")
        subset = pd.DataFrame({"feature": feature, "target": target}).dropna()
        if len(subset) < 3 or subset["feature"].std() == 0 or subset["target"].std() == 0:
            rows.append({"feature_name": feature_column, "pearson_correlation": None, "spearman_correlation": None, "sample_count": int(len(subset))})
            continue
        rows.append(
            {
                "feature_name": feature_column,
                "pearson_correlation": float(pearsonr(subset["feature"], subset["target"]).statistic),
                "spearman_correlation": float(spearmanr(subset["feature"], subset["target"]).statistic),
                "sample_count": int(len(subset)),
            }
        )
    return sorted(rows, key=lambda item: abs(item["pearson_correlation"] or 0.0), reverse=True)


def _model_feature_importance_rows(model: Any, feature_columns: list[str]) -> list[dict[str, Any]]:
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_.tolist()
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=np.float64)
        coef = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        values = coef.tolist()
    else:
        return []
    rows = [{"feature_name": feature_name, "importance": float(value)} for feature_name, value in zip(feature_columns, values)]
    return sorted(rows, key=lambda item: item["importance"], reverse=True)


def _standardized_effect_rows(model: Any, feature_columns: list[str], correlation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    correlation_map = {str(row["feature_name"]): row for row in correlation_rows}
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) and "model" in model.named_steps else model
    scaler = model.named_steps.get("scaler") if isinstance(model, Pipeline) else None
    rows: list[dict[str, Any]] = []
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=np.float64)
        coef = np.mean(coef, axis=0) if coef.ndim > 1 else coef
        scale = np.asarray(getattr(scaler, "scale_", np.ones(len(feature_columns))), dtype=np.float64) if scaler is not None else np.ones(len(feature_columns), dtype=np.float64)
        scale = np.where(scale == 0, 1.0, scale)
        raw_effect = coef / scale
        for feature_name, std_coef, raw_coef in zip(feature_columns, coef.tolist(), raw_effect.tolist()):
            rows.append(
                {
                    "feature_name": feature_name,
                    "raw_coefficient": float(raw_coef),
                    "signed_effect": float(std_coef),
                    "abs_effect": float(abs(std_coef)),
                    "effect_method": "standardized_coefficients",
                    "pearson_correlation": correlation_map.get(feature_name, {}).get("pearson_correlation"),
                    "spearman_correlation": correlation_map.get(feature_name, {}).get("spearman_correlation"),
                }
            )
        return sorted(rows, key=lambda item: abs(item["signed_effect"]), reverse=True)

    importance_rows = _model_feature_importance_rows(model, feature_columns)
    for row in importance_rows:
        feature_name = str(row["feature_name"])
        pearson = correlation_map.get(feature_name, {}).get("pearson_correlation")
        sign = 0.0 if pearson is None else float(np.sign(float(pearson)))
        row["signed_effect"] = float(row["importance"] * sign)
        row["abs_effect"] = float(row["importance"])
        row["effect_method"] = "importance_with_correlation_sign"
        row["pearson_correlation"] = pearson
        row["spearman_correlation"] = correlation_map.get(feature_name, {}).get("spearman_correlation")
    return importance_rows


def _safe_mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None, None
    return float(np.mean(valid)), float(np.std(valid))


def _cv_splitter(frame: pd.DataFrame, group_column: str, cv_config: dict[str, Any], random_seed: int) -> tuple[Any, np.ndarray]:
    labels = frame[group_column].fillna("").astype(str).to_numpy()
    group_counts = pd.Series(labels).value_counts()
    min_group_support = int(group_counts.min()) if not group_counts.empty else 0
    n_splits = max(2, min(int(cv_config.get("n_splits", 5)), len(frame)))
    if min_group_support >= 2:
        n_splits = min(n_splits, min_group_support)
    if min_group_support >= 2 and len(np.unique(labels)) > 1:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=bool(cv_config.get("shuffle", True)), random_state=random_seed)
        split_labels = labels
    else:
        splitter = KFold(n_splits=n_splits, shuffle=bool(cv_config.get("shuffle", True)), random_state=random_seed)
        split_labels = np.zeros(len(frame), dtype=np.int64)
    return splitter, split_labels


def _cv_regression_summary(
    *,
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: list[float],
    fold_labels: np.ndarray,
    splitter: Any,
    sample_weights: np.ndarray | None,
    clip_lower: float | None,
    clip_upper: float | None,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X, fold_labels), start=1):
        train_weights = sample_weights[train_idx] if sample_weights is not None else None
        estimator = _fit_estimator(model, X[train_idx], y[train_idx], sample_weight=train_weights)
        fold_pred = _prediction_array(estimator, X[val_idx], lower=clip_lower, upper=clip_upper)
        folds.append(
            {
                "fold_index": fold_index,
                "metrics": _regression_metrics(y[val_idx], fold_pred),
                "group_proximity": _group_metrics(y[val_idx], fold_pred, groups),
            }
        )
    metric_names = ("mae", "rmse", "r2", "pearson", "spearman")
    mean_metrics: dict[str, float | None] = {}
    std_metrics: dict[str, float | None] = {}
    for metric_name in metric_names:
        mean_metrics[metric_name], std_metrics[f"{metric_name}_std"] = _safe_mean_std([fold["metrics"].get(metric_name) for fold in folds])
    proximity_names = ("nearest_group_accuracy", "within_0_5_accuracy", "within_1_0_accuracy", "mean_absolute_group_distance")
    mean_proximity: dict[str, float | None] = {}
    std_proximity: dict[str, float | None] = {}
    for metric_name in proximity_names:
        mean_proximity[metric_name], std_proximity[f"{metric_name}_std"] = _safe_mean_std([fold["group_proximity"].get(metric_name) for fold in folds])
    return {"folds": folds, "mean_metrics": mean_metrics, "std_metrics": std_metrics, "mean_group_proximity": mean_proximity, "std_group_proximity": std_proximity}


def _cv_classification_summary(
    *,
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    fold_labels: np.ndarray,
    splitter: Any,
    sample_weights: np.ndarray | None,
    classes: list[str],
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X, fold_labels), start=1):
        train_weights = sample_weights[train_idx] if sample_weights is not None else None
        estimator = _fit_estimator(model, X[train_idx], y[train_idx], sample_weight=train_weights)
        fold_pred = estimator.predict(X[val_idx])
        fold_prob = estimator.predict_proba(X[val_idx]) if hasattr(estimator, "predict_proba") else None
        folds.append({"fold_index": fold_index, "metrics": _classifier_metrics(y[val_idx], fold_pred, fold_prob, classes)})
    metric_names = ("accuracy", "balanced_accuracy", "top_2_accuracy")
    mean_metrics: dict[str, float | None] = {}
    std_metrics: dict[str, float | None] = {}
    for metric_name in metric_names:
        mean_metrics[metric_name], std_metrics[f"{metric_name}_std"] = _safe_mean_std([fold["metrics"].get(metric_name) for fold in folds])
    return {"folds": folds, "mean_metrics": mean_metrics, "std_metrics": std_metrics}


def _groupwise_mae(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    if y_true.size == 0:
        return []
    rows: list[dict[str, Any]] = []
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    for group_value, subset in frame.groupby("y_true"):
        rows.append(
            {
                "target_score": float(group_value),
                "sample_count": int(len(subset)),
                "mae": float(np.mean(np.abs(subset["y_true"] - subset["y_pred"]))),
            }
        )
    return sorted(rows, key=lambda item: item["mae"], reverse=True)


def _groupwise_error_rows(y_true: np.ndarray, y_pred: np.ndarray, groups: list[float]) -> list[dict[str, Any]]:
    if y_true.size == 0:
        return []
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    rows: list[dict[str, Any]] = []
    for group_value, subset in frame.groupby("y_true"):
        true_values = subset["y_true"].to_numpy(dtype=np.float64)
        pred_values = subset["y_pred"].to_numpy(dtype=np.float64)
        group_metrics = _group_metrics(true_values, pred_values, groups)
        rows.append(
            {
                "target_score": float(group_value),
                "sample_count": int(len(subset)),
                "mae": float(np.mean(np.abs(true_values - pred_values))),
                "nearest_group_accuracy": group_metrics.get("nearest_group_accuracy"),
                "mean_absolute_group_distance": group_metrics.get("mean_absolute_group_distance"),
            }
        )
    return sorted(rows, key=lambda item: item["mae"], reverse=True)


def _calibration_by_group_rows(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    if y_true.size == 0:
        return []
    frame = pd.DataFrame({"target_score": y_true, "predicted_score": y_pred})
    rows: list[dict[str, Any]] = []
    for target_score, subset in frame.groupby("target_score"):
        predicted = subset["predicted_score"].to_numpy(dtype=np.float64)
        target = subset["target_score"].to_numpy(dtype=np.float64)
        errors = predicted - target
        rows.append(
            {
                "target_score": float(target_score),
                "count": int(len(subset)),
                "mean_predicted_score": float(np.mean(predicted)),
                "median_predicted_score": float(np.median(predicted)),
                "mean_error": float(np.mean(errors)),
                "mae": float(np.mean(np.abs(errors))),
                "bias": float(np.mean(errors)),
                "underprediction_rate": float(np.mean(predicted < target)),
                "overprediction_rate": float(np.mean(predicted > target)),
            }
        )
    return sorted(rows, key=lambda item: item["target_score"])


def _calibration_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, float | None]:
    if y_true.size < 2 or float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return None, None
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    return float(slope), float(intercept)


def _calibration_summary(y_true: np.ndarray, y_pred: np.ndarray, strict_config: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = _calibration_by_group_rows(y_true, y_pred)
    if not rows:
        return {
            "low_group_bias": None,
            "middle_group_bias": None,
            "high_group_bias": None,
            "elite_group_bias": None,
            "target_prediction_correlation": None,
            "calibration_slope": None,
            "calibration_intercept": None,
            "high_group_mae": None,
            "low_group_mae": None,
            "group_macro_mae": None,
            "warnings": [],
        }
    strict = strict_config or {}
    low_threshold = float(strict.get("low_score_threshold", 3.0))
    high_threshold = float(strict.get("high_score_threshold", 8.5))
    elite_threshold = float(strict.get("elite_score_threshold", 9.0))
    group_targets = [float(row["target_score"]) for row in rows]
    low_targets = {value for value in group_targets if value <= low_threshold}
    middle_targets = {value for value in group_targets if value > low_threshold and value < high_threshold}
    high_targets = {value for value in group_targets if value >= high_threshold}
    elite_targets = {value for value in group_targets if value >= elite_threshold}
    if not low_targets:
        low_targets = {min(group_targets)}
    if not middle_targets:
        middle_targets = set(group_targets)
    if not high_targets:
        high_targets = {max(group_targets)}
    if not elite_targets:
        elite_targets = high_targets

    def _bucket_bias(targets: set[float]) -> float | None:
        bucket = [float(row["bias"]) for row in rows if float(row["target_score"]) in targets]
        if not bucket:
            return None
        return float(np.mean(bucket))

    def _bucket_mae(targets: set[float]) -> float | None:
        bucket = [float(row["mae"]) for row in rows if float(row["target_score"]) in targets]
        if not bucket:
            return None
        return float(np.mean(bucket))

    slope, intercept = _calibration_slope_intercept(y_true, y_pred)
    target_prediction_correlation = None
    if y_true.size >= 2 and float(np.std(y_true)) > 0.0 and float(np.std(y_pred)) > 0.0:
        target_prediction_correlation = float(pearsonr(y_true, y_pred).statistic)
    low_group_bias = _bucket_bias(low_targets)
    middle_group_bias = _bucket_bias(middle_targets)
    high_group_bias = _bucket_bias(high_targets)
    elite_group_bias = _bucket_bias(elite_targets)
    high_group_mae = _bucket_mae(high_targets)
    low_group_mae = _bucket_mae(low_targets)
    group_macro_mae = float(np.mean([float(row["mae"]) for row in rows])) if rows else None
    collapse_threshold = float(strict.get("high_score_collapse_threshold", high_threshold))
    reject_below = float(strict.get("reject_if_high_score_pred_below", 6.0))
    warn_below = float(strict.get("warn_if_high_score_pred_below", 7.0))
    max_collapse_below_6_count = int(strict.get("max_high_score_collapse_below_6_count", 0))
    high_mask = y_true >= collapse_threshold
    high_true = y_true[high_mask]
    high_pred = y_pred[high_mask]
    collapse_below_7_count = int(np.sum(high_pred < warn_below)) if high_pred.size else 0
    collapse_below_6_count = int(np.sum(high_pred < reject_below)) if high_pred.size else 0
    collapse_below_7_rate = float(collapse_below_7_count / high_pred.size) if high_pred.size else 0.0
    collapse_below_6_rate = float(collapse_below_6_count / high_pred.size) if high_pred.size else 0.0

    warnings: list[str] = []
    if high_group_bias is not None and high_group_bias <= -0.5:
        warnings.append("high_score_regression_to_mean_detected")
    if low_group_bias is not None and low_group_bias >= 0.5:
        warnings.append("low_score_regression_to_mean_detected")
    if collapse_below_7_count > 0:
        warnings.append("high_score_collapse_risk")
    if collapse_below_6_count > max_collapse_below_6_count:
        warnings.append("severe_high_score_collapse_detected")
    return {
        "low_group_bias": low_group_bias,
        "middle_group_bias": middle_group_bias,
        "high_group_bias": high_group_bias,
        "elite_group_bias": elite_group_bias,
        "target_prediction_correlation": target_prediction_correlation,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "high_group_mae": high_group_mae,
        "low_group_mae": low_group_mae,
        "group_macro_mae": group_macro_mae,
        "high_score_collapse_threshold": collapse_threshold,
        "collapse_below_7_count": collapse_below_7_count,
        "collapse_below_6_count": collapse_below_6_count,
        "collapse_below_7_rate": collapse_below_7_rate,
        "collapse_below_6_rate": collapse_below_6_rate,
        "warnings": warnings,
    }


def _eval_payload(y_true: np.ndarray, y_pred: np.ndarray, groups: list[float], strict_config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "metrics": _regression_metrics(y_true, y_pred),
        "group_proximity": _group_metrics(y_true, y_pred, groups),
        "groupwise_errors": _groupwise_error_rows(y_true, y_pred, groups),
        "calibration_by_group": _calibration_by_group_rows(y_true, y_pred),
        "calibration_summary": _calibration_summary(y_true, y_pred, strict_config),
    }


def _fit_score_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    clip_lower: float | None,
    clip_upper: float | None,
) -> IsotonicRegression | None:
    if y_true.size < 3:
        return None
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(valid_mask.sum()) < 3:
        return None
    true_valid = y_true[valid_mask]
    pred_valid = y_pred[valid_mask]
    if len(np.unique(pred_valid)) < 2 or len(np.unique(true_valid)) < 2:
        return None
    calibrator = IsotonicRegression(
        y_min=clip_lower,
        y_max=clip_upper,
        out_of_bounds="clip",
    )
    calibrator.fit(pred_valid, true_valid)
    return calibrator


def _apply_calibrator(predictions: np.ndarray, calibrator: IsotonicRegression | None, *, clip_lower: float | None, clip_upper: float | None) -> np.ndarray:
    if predictions.size == 0:
        return np.asarray([], dtype=np.float64)
    if calibrator is None:
        return np.asarray([clip_prediction(float(value), lower=clip_lower, upper=clip_upper) for value in predictions], dtype=np.float64)
    transformed = calibrator.predict(predictions)
    return np.asarray([clip_prediction(float(value), lower=clip_lower, upper=clip_upper) for value in transformed], dtype=np.float64)


def _macro_group_mae(groupwise_rows: list[dict[str, Any]]) -> float | None:
    maes = [float(row["mae"]) for row in groupwise_rows if row.get("mae") is not None]
    if not maes:
        return None
    return float(np.mean(maes))


def _extreme_group_mae(groupwise_rows: list[dict[str, Any]], bucket: str) -> float | None:
    if not groupwise_rows:
        return None
    targets = sorted(float(row["target_score"]) for row in groupwise_rows)
    if len(targets) == 1:
        chosen = set(targets)
    else:
        cut = np.quantile(targets, 1 / 3 if bucket == "low" else 2 / 3)
        chosen = {value for value in targets if value <= cut} if bucket == "low" else {value for value in targets if value >= cut}
    maes = [float(row["mae"]) for row in groupwise_rows if float(row["target_score"]) in chosen and row.get("mae") is not None]
    if not maes:
        return None
    return float(np.mean(maes))


def _regression_selection_breakdown(cv_summary: dict[str, Any], val_payload: dict[str, Any]) -> dict[str, float | None]:
    val_metrics = dict(val_payload.get("metrics", {}))
    group_proximity = dict(val_payload.get("group_proximity", {}))
    calibration_summary = dict(val_payload.get("calibration_summary", {}))
    groupwise_rows = list(val_payload.get("groupwise_errors", []))
    return {
        "val_mae": val_metrics.get("mae"),
        "group_macro_mae": _macro_group_mae(groupwise_rows),
        "low_group_mae": _extreme_group_mae(groupwise_rows, "low"),
        "high_group_mae": _extreme_group_mae(groupwise_rows, "high"),
        "spearman": val_metrics.get("spearman"),
        "within_1_0_accuracy": group_proximity.get("within_1_0_accuracy"),
        "calibration_slope": calibration_summary.get("calibration_slope"),
        "calibration_intercept": calibration_summary.get("calibration_intercept"),
        "low_group_bias": calibration_summary.get("low_group_bias"),
        "high_group_bias": calibration_summary.get("high_group_bias"),
        "elite_group_bias": calibration_summary.get("elite_group_bias"),
        "cv_mae_std": cv_summary.get("std_metrics", {}).get("mae_std"),
    }


def _regression_selection_score(cv_summary: dict[str, Any], val_payload: dict[str, Any], strict_config: dict[str, Any] | None = None) -> tuple[float, dict[str, float | None]]:
    strict = strict_config or {}
    breakdown = _regression_selection_breakdown(cv_summary, val_payload)
    mae = float(breakdown["val_mae"]) if breakdown["val_mae"] is not None else 10.0
    macro_mae = float(breakdown["group_macro_mae"]) if breakdown["group_macro_mae"] is not None else mae
    low_mae = float(breakdown["low_group_mae"]) if breakdown["low_group_mae"] is not None else mae
    high_mae = float(breakdown["high_group_mae"]) if breakdown["high_group_mae"] is not None else mae
    spearman = float(breakdown["spearman"]) if breakdown["spearman"] is not None else 0.0
    within_1 = float(breakdown["within_1_0_accuracy"]) if breakdown["within_1_0_accuracy"] is not None else 0.0
    slope_penalty = abs(float(breakdown["calibration_slope"]) - 1.0) if breakdown["calibration_slope"] is not None else 1.0
    intercept_penalty = abs(float(breakdown["calibration_intercept"])) if breakdown["calibration_intercept"] is not None else 1.0
    low_bias_penalty = abs(float(breakdown["low_group_bias"])) if breakdown["low_group_bias"] is not None else 1.0
    high_bias_penalty = abs(float(breakdown["high_group_bias"])) if breakdown["high_group_bias"] is not None else 1.0
    elite_bias_penalty = abs(float(breakdown["elite_group_bias"])) if breakdown["elite_group_bias"] is not None else high_bias_penalty
    cv_mae_std = float(breakdown["cv_mae_std"]) if breakdown["cv_mae_std"] is not None else 0.0
    extreme_bias_weight = float(strict.get("extreme_bias_penalty_weight", 1.0))
    high_under_weight = float(strict.get("high_score_underprediction_penalty", 1.5))
    low_over_weight = float(strict.get("low_score_overprediction_penalty", 1.5))
    macro_weight = float(strict.get("group_macro_mae_weight", 1.0))
    high_mae_weight = float(strict.get("high_group_mae_weight", 1.5))
    low_mae_weight = float(strict.get("low_group_mae_weight", 1.5))
    ranking_weight = float(strict.get("ranking_weight", 1.0))
    calibration_weight = float(strict.get("calibration_weight", 1.0))
    score = (
        mae
        + macro_weight * macro_mae
        + low_mae_weight * low_mae
        + high_mae_weight * high_mae
        + extreme_bias_weight * low_over_weight * low_bias_penalty
        + extreme_bias_weight * high_under_weight * high_bias_penalty
        + 0.5 * extreme_bias_weight * high_under_weight * elite_bias_penalty
        + calibration_weight * (0.20 * slope_penalty + 0.05 * intercept_penalty)
        + 0.15 * (1.0 - within_1)
        + 0.10 * cv_mae_std
        - (0.20 * ranking_weight) * spearman
    )
    return float(score), breakdown


def _select_primary_with_guardrails(
    *,
    feature_set_comparison: dict[str, Any],
    feature_set_metadata: dict[str, dict[str, Any]],
    guardrails: dict[str, Any],
) -> tuple[str, str | None, dict[str, Any], dict[str, Any]]:
    sets_payload = dict(feature_set_comparison.get("sets", {}))
    requested_sets = list(
        guardrails.get(
            "candidate_feature_sets",
            [
                "previous_best_reference_weighted_ridge",
                "combined_weighted_no_duplicate",
                "reference_plus_mediapipe_balanced",
                "no_pose_reference_weighted",
                "reference_relative_only_reduced",
                "absolute_plus_best_reference_50",
            ],
        )
    )
    alias_map = {
        "previous_best_reference_weighted_ridge": ["absolute_plus_best_reference_50"],
        "reference_plus_mediapipe_balanced": ["mediapipe_plus_reference_weighted"],
        "reference_relative_only_reduced": ["ref_z_plus_percentile_stable", "ref_deviation_compact"],
        "no_pose_reference_weighted": ["absolute_plus_best_reference_50"],
    }
    resolved_candidates: list[str] = []
    for name in requested_sets:
        if name in sets_payload:
            resolved_candidates.append(name)
            continue
        for alias in alias_map.get(name, []):
            if alias in sets_payload:
                resolved_candidates.append(alias)
                break
    if not resolved_candidates:
        resolved_candidates = [name for name, payload in sets_payload.items() if str(payload.get("status")) == "completed"]
    resolved_candidates = list(dict.fromkeys(resolved_candidates))

    min_test_mae = guardrails.get("min_test_mae_improvement_against_anchor")
    min_test_r2 = guardrails.get("min_test_r2")
    min_test_spearman = guardrails.get("min_test_spearman")
    max_high_bias_abs = guardrails.get("max_abs_high_group_bias")
    max_low_bias_abs = guardrails.get("max_abs_low_group_bias")
    reject_if_high_score_pred_below = guardrails.get("reject_if_high_score_pred_below")
    warn_if_high_score_pred_below = guardrails.get("warn_if_high_score_pred_below")
    max_high_score_collapse_below_6_count = int(guardrails.get("max_high_score_collapse_below_6_count", 0))
    require_reference = bool(guardrails.get("prefer_reference_relative_features", True))
    hard_penalty_no_reference = float(guardrails.get("hard_penalty_reference_relative_zero", 5.0))
    hard_penalty_degraded = float(guardrails.get("hard_penalty_metric_degradation", 10.0))
    high_score_collapse_penalty = float(guardrails.get("high_score_collapse_penalty", 30.0))

    # Anchor is previous best by name if present, otherwise strongest reference-relative candidate.
    anchor_name = str(guardrails.get("anchor_feature_set_name") or "")
    if anchor_name not in sets_payload:
        anchor_name = ""
    if not anchor_name:
        best_anchor_score = float("inf")
        for name, payload in sets_payload.items():
            if str(payload.get("status")) != "completed":
                continue
            meta = dict(feature_set_metadata.get(name, {}))
            if int(meta.get("reference_relative_selected_count", 0)) <= 0:
                continue
            candidate = list(payload.get("candidates", []))
            if not candidate:
                continue
            score = float(candidate[0].get("selection_score", float("inf")))
            if score < best_anchor_score:
                best_anchor_score = score
                anchor_name = str(name)
    if not anchor_name:
        anchor_name = str(feature_set_comparison.get("selected_feature_set") or (resolved_candidates[0] if resolved_candidates else "baseline_insightface_geometry"))

    def _best_candidate_row(feature_set_name: str) -> dict[str, Any] | None:
        payload = sets_payload.get(feature_set_name)
        if not isinstance(payload, dict) or str(payload.get("status")) != "completed":
            return None
        rows = list(payload.get("candidates", []))
        if not rows:
            return None
        return dict(rows[0])

    anchor_row = _best_candidate_row(anchor_name)
    anchor_test_metrics = (((anchor_row or {}).get("test") or {}).get("metrics") or {}) if anchor_row else {}
    anchor_test_calibration = (((anchor_row or {}).get("test") or {}).get("calibration_summary") or {}) if anchor_row else {}
    anchor_mae = float(anchor_test_metrics.get("mae")) if anchor_test_metrics.get("mae") is not None else None
    anchor_r2 = float(anchor_test_metrics.get("r2")) if anchor_test_metrics.get("r2") is not None else None
    anchor_spearman = float(anchor_test_metrics.get("spearman")) if anchor_test_metrics.get("spearman") is not None else None
    anchor_high_bias_abs = abs(float(anchor_test_calibration.get("high_group_bias"))) if anchor_test_calibration.get("high_group_bias") is not None else None
    anchor_low_bias_abs = abs(float(anchor_test_calibration.get("low_group_bias"))) if anchor_test_calibration.get("low_group_bias") is not None else None

    comparison_rows: list[dict[str, Any]] = []
    selected_name = str(feature_set_comparison.get("selected_feature_set") or anchor_name)
    selected_model = None
    selected_score = float("inf")
    for feature_set_name in resolved_candidates:
        row = _best_candidate_row(feature_set_name)
        if row is None:
            continue
        test_payload = dict(row.get("test", {}))
        test_metrics = dict(test_payload.get("metrics", {}))
        test_calibration = dict(test_payload.get("calibration_summary", {}))
        meta = dict(feature_set_metadata.get(feature_set_name, {}))
        ref_count = int(meta.get("reference_relative_selected_count", 0))
        direct_mp_count = int(meta.get("mediapipe_selected_count", 0))
        candidate_score = float(row.get("selection_score", float("inf")))

        penalties: list[str] = []
        guarded_score = candidate_score
        test_mae = float(test_metrics.get("mae")) if test_metrics.get("mae") is not None else None
        test_r2 = float(test_metrics.get("r2")) if test_metrics.get("r2") is not None else None
        test_spearman = float(test_metrics.get("spearman")) if test_metrics.get("spearman") is not None else None
        high_bias_abs = abs(float(test_calibration.get("high_group_bias"))) if test_calibration.get("high_group_bias") is not None else None
        low_bias_abs = abs(float(test_calibration.get("low_group_bias"))) if test_calibration.get("low_group_bias") is not None else None
        collapse_below_7_count = int(test_calibration.get("collapse_below_7_count", 0) or 0)
        collapse_below_6_count = int(test_calibration.get("collapse_below_6_count", 0) or 0)
        collapse_below_7_rate = float(test_calibration.get("collapse_below_7_rate", 0.0) or 0.0)
        collapse_below_6_rate = float(test_calibration.get("collapse_below_6_rate", 0.0) or 0.0)

        if require_reference and ref_count == 0:
            guarded_score += hard_penalty_no_reference
            penalties.append("reference_relative_selected_count_zero")
        if min_test_r2 is not None and (test_r2 is None or test_r2 < float(min_test_r2)):
            guarded_score += hard_penalty_degraded
            penalties.append("test_r2_below_floor")
        if min_test_spearman is not None and (test_spearman is None or test_spearman < float(min_test_spearman)):
            guarded_score += hard_penalty_degraded
            penalties.append("test_spearman_below_floor")
        if max_high_bias_abs is not None and (high_bias_abs is None or high_bias_abs > float(max_high_bias_abs)):
            guarded_score += hard_penalty_degraded
            penalties.append("high_group_bias_too_high")
        if max_low_bias_abs is not None and (low_bias_abs is None or low_bias_abs > float(max_low_bias_abs)):
            guarded_score += hard_penalty_degraded
            penalties.append("low_group_bias_too_high")
        if min_test_mae is not None and (test_mae is None or test_mae > float(min_test_mae)):
            guarded_score += hard_penalty_degraded
            penalties.append("test_mae_above_floor")
        if warn_if_high_score_pred_below is not None and collapse_below_7_count > 0:
            guarded_score += 0.5 * high_score_collapse_penalty
            penalties.append("high_score_collapse_warn_threshold_breached")
        if reject_if_high_score_pred_below is not None and collapse_below_6_count > max_high_score_collapse_below_6_count:
            guarded_score += high_score_collapse_penalty
            penalties.append("severe_high_score_collapse_reject")

        if anchor_mae is not None and test_mae is not None and test_mae > anchor_mae:
            guarded_score += 0.5 * hard_penalty_degraded
            penalties.append("worse_than_anchor_mae")
        if anchor_r2 is not None and test_r2 is not None and test_r2 < anchor_r2:
            guarded_score += 0.5 * hard_penalty_degraded
            penalties.append("worse_than_anchor_r2")
        if anchor_spearman is not None and test_spearman is not None and test_spearman < anchor_spearman:
            guarded_score += 0.5 * hard_penalty_degraded
            penalties.append("worse_than_anchor_spearman")
        if anchor_high_bias_abs is not None and high_bias_abs is not None and high_bias_abs > anchor_high_bias_abs:
            guarded_score += 0.5 * hard_penalty_degraded
            penalties.append("worse_than_anchor_high_bias")
        if anchor_low_bias_abs is not None and low_bias_abs is not None and low_bias_abs > anchor_low_bias_abs:
            guarded_score += 0.5 * hard_penalty_degraded
            penalties.append("worse_than_anchor_low_bias")

        comparison_rows.append(
            {
                "feature_set_name": feature_set_name,
                "model_name": row.get("model_name"),
                "base_selection_score": candidate_score,
                "guardrail_adjusted_score": guarded_score,
                "penalties": penalties,
                "test_metrics": test_metrics,
                "test_calibration_summary": test_calibration,
                "collapse_below_7_count": collapse_below_7_count,
                "collapse_below_6_count": collapse_below_6_count,
                "collapse_below_7_rate": collapse_below_7_rate,
                "collapse_below_6_rate": collapse_below_6_rate,
                "reference_relative_selected_count": ref_count,
                "direct_mediapipe_selected_count": direct_mp_count,
            }
        )
        if guarded_score < selected_score:
            selected_score = guarded_score
            selected_name = feature_set_name
            selected_model = str(row.get("model_name")) if row.get("model_name") is not None else None

    comparison_rows = sorted(comparison_rows, key=lambda item: float(item.get("guardrail_adjusted_score", float("inf"))))
    guardrails_report = {
        "status": "completed",
        "anchor_feature_set_name": anchor_name,
        "anchor_metrics": {
            "test_mae": anchor_mae,
            "test_r2": anchor_r2,
            "test_spearman": anchor_spearman,
            "abs_high_group_bias": anchor_high_bias_abs,
            "abs_low_group_bias": anchor_low_bias_abs,
        },
        "guardrails": guardrails,
        "candidates_considered": resolved_candidates,
        "rows": comparison_rows,
        "selected_feature_set": selected_name,
        "selected_model_hint": selected_model,
    }
    primary_comparison_report = {
        "status": "completed",
        "selected_feature_set": selected_name,
        "selected_model_hint": selected_model,
        "rows": comparison_rows,
    }
    return selected_name, selected_model, guardrails_report, primary_comparison_report


def _feature_registry_map() -> dict[str, dict[str, Any]]:
    return {str(row["feature_name"]): row for row in feature_registry_rows()}


def _feature_source_counts(feature_columns: list[str], registry_by_name: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = {"mediapipe_face_mesh": 0, "insightface_106": 0, "fused": 0, "reference_relative": 0, "other": 0}
    for feature_name in feature_columns:
        if feature_name.startswith(REFERENCE_FEATURE_PREFIXES):
            counts["reference_relative"] += 1
            continue
        source = str(registry_by_name.get(feature_name, {}).get("source") or "other")
        if source in counts:
            counts[source] += 1
        else:
            counts["other"] += 1
    return counts


def _is_mediapipe_primary_feature(
    feature_name: str,
    *,
    registry_by_name: dict[str, dict[str, Any]],
) -> bool:
    if feature_name.startswith("raw_") or feature_name.endswith("_px"):
        return False
    base_name = _base_feature_name(feature_name)
    base_source = str(registry_by_name.get(base_name, {}).get("source") or "").strip().lower()
    direct_source = str(registry_by_name.get(feature_name, {}).get("source") or "").strip().lower()
    if feature_name.startswith(REFERENCE_FEATURE_PREFIXES):
        return base_name.startswith("mediapipe_") and base_source == "mediapipe_face_mesh"
    if feature_name.startswith("mediapipe_"):
        return base_source == "mediapipe_face_mesh" or direct_source == "mediapipe_face_mesh"
    return False


def _apply_mediapipe_only_policy(
    feature_columns: list[str],
    *,
    registry_by_name: dict[str, dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    kept: list[str] = []
    dropped: list[dict[str, Any]] = []
    for feature_name in feature_columns:
        if _is_mediapipe_primary_feature(feature_name, registry_by_name=registry_by_name):
            kept.append(feature_name)
            continue
        dropped.append(
            {
                "feature_name": feature_name,
                "base_feature_name": _base_feature_name(feature_name),
                "source": str(registry_by_name.get(feature_name, {}).get("source") or "other"),
                "original_source": str(registry_by_name.get(_base_feature_name(feature_name), {}).get("source") or "other"),
                "is_reference_relative": bool(feature_name.startswith(REFERENCE_FEATURE_PREFIXES)),
                "reason": "blocked_by_mediapipe_only_policy",
            }
        )
    return list(dict.fromkeys(kept)), dropped


def _base_feature_name(feature_name: str) -> str:
    for prefix in REFERENCE_FEATURE_PREFIXES:
        if feature_name.startswith(prefix):
            return feature_name[len(prefix) :]
    return feature_name


def _reference_transform(feature_name: str) -> str | None:
    if feature_name.startswith("ref_z_"):
        return "z"
    if feature_name.startswith("ref_dir_"):
        return "dir"
    if feature_name.startswith("ref_ratio_"):
        return "ratio"
    if feature_name.startswith("ref_absdev_"):
        return "absdev"
    if feature_name.startswith("ref_percentile_"):
        return "percentile"
    return None


def _transform_family(feature_name: str) -> str:
    if _reference_transform(feature_name) is not None:
        return "reference_relative"
    if feature_name.startswith("ratio_") or "_to_" in feature_name or feature_name.endswith("_ratio"):
        return "derived_ratio"
    return "absolute_geometry"


def _feature_lineage_row(
    feature_name: str,
    *,
    registry_by_name: dict[str, dict[str, Any]],
    component_map: dict[str, list[str]],
) -> dict[str, Any]:
    base_name = _base_feature_name(feature_name)
    transform = _reference_transform(feature_name)
    is_reference_relative = transform is not None
    base_meta = registry_by_name.get(base_name, {})
    direct_meta = registry_by_name.get(feature_name, {})
    original_source = str(base_meta.get("source") or direct_meta.get("source") or "other")
    source = "reference_relative" if is_reference_relative else original_source
    anatomical_component = _infer_feature_component(feature_name, registry_by_name=registry_by_name, component_map=component_map)
    transform_family = _transform_family(feature_name)
    reliability = str(base_meta.get("reliability") or direct_meta.get("reliability") or "unknown")
    mediapipe_derived = bool(str(base_name).startswith("mediapipe_"))
    return {
        "feature_name": feature_name,
        "base_feature_name": base_name,
        "anatomical_component": anatomical_component,
        "transform_family": transform_family,
        "source": source,
        "original_source": original_source,
        "is_reference_relative": is_reference_relative,
        "reference_transform": transform,
        "component": anatomical_component,
        "reliability": reliability,
        "mediapipe_derived": mediapipe_derived,
    }


def _infer_feature_component(
    feature_name: str,
    *,
    registry_by_name: dict[str, dict[str, Any]],
    component_map: dict[str, list[str]],
) -> str:
    direct = str(registry_by_name.get(feature_name, {}).get("component") or "").strip()
    if direct:
        return _canonical_anatomical_component(direct)
    base_name = _base_feature_name(feature_name)
    base_component = str(registry_by_name.get(base_name, {}).get("component") or "").strip()
    if base_component:
        return _canonical_anatomical_component(base_component)
    for component_name, features in component_map.items():
        if feature_name in features or base_name in features:
            return _canonical_anatomical_component(str(component_name))
    return "unknown"


def _abs_pearson_for_feature(values: pd.Series, target: pd.Series) -> float:
    subset = pd.DataFrame({"x": pd.to_numeric(values, errors="coerce"), "y": pd.to_numeric(target, errors="coerce")}).dropna()
    if len(subset) < 5 or float(subset["x"].std()) == 0.0 or float(subset["y"].std()) == 0.0:
        return 0.0
    try:
        return float(abs(pearsonr(subset["x"], subset["y"]).statistic))
    except Exception:
        return 0.0


def _component_balanced_top_features(
    *,
    feature_columns: list[str],
    numeric_frame: pd.DataFrame,
    target_values: pd.Series,
    registry_by_name: dict[str, dict[str, Any]],
    component_map: dict[str, list[str]],
    top_k_per_component: int,
) -> tuple[list[str], dict[str, Any]]:
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for column in feature_columns:
        if column not in numeric_frame.columns:
            continue
        component_name = _infer_feature_component(
            column,
            registry_by_name=registry_by_name,
            component_map=component_map,
        )
        score = _abs_pearson_for_feature(numeric_frame[column], target_values)
        by_component[component_name].append(
            {
                "feature_name": column,
                "component": component_name,
                "abs_pearson": score,
            }
        )
    selected: list[str] = []
    selection_rows: list[dict[str, Any]] = []
    for component_name, rows in by_component.items():
        for row in sorted(rows, key=lambda item: float(item["abs_pearson"]), reverse=True)[:top_k_per_component]:
            selected.append(str(row["feature_name"]))
            selection_rows.append(row)
    selected = list(dict.fromkeys(selected))
    return selected, {
        "top_k_per_component": int(top_k_per_component),
        "component_counts": {
            component_name: int(min(len(rows), top_k_per_component))
            for component_name, rows in by_component.items()
        },
        "selected_rows": sorted(selection_rows, key=lambda item: (str(item["component"]), -float(item["abs_pearson"]))),
    }


def _variant_priority(feature_name: str) -> int:
    transform = _reference_transform(feature_name)
    if transform == "z":
        return 0
    if transform == "percentile":
        return 1
    if transform == "absdev":
        return 2
    if transform == "dir":
        return 3
    if transform == "ratio":
        return 4
    return 5


def _enforce_base_feature_dedup(
    *,
    feature_columns: list[str],
    numeric_frame: pd.DataFrame,
    target_values: pd.Series,
    max_variants_per_base_feature: int = 1,
    allow_second_variant_correlation_threshold: float = 0.85,
) -> tuple[list[str], dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for feature_name in feature_columns:
        groups[_base_feature_name(feature_name)].append(feature_name)
    selected: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    for base_name, variants in groups.items():
        if len(variants) == 1:
            selected.extend(variants)
            audit_rows.append(
                {
                    "base_feature_name": base_name,
                    "variants_present": variants,
                    "correlations_between_variants": {},
                    "selected_variant": variants[0],
                    "dropped_variants": [],
                    "reason": "single_variant",
                }
            )
            continue
        ranked = sorted(
            variants,
            key=lambda feature_name: (
                _variant_priority(feature_name),
                -_abs_pearson_for_feature(numeric_frame.get(feature_name, pd.Series(dtype=float)), target_values),
                feature_name,
            ),
        )
        selected_variants = [ranked[0]]
        correlations: dict[str, float | None] = {}
        if max_variants_per_base_feature > 1 and len(ranked) > 1:
            first = pd.to_numeric(numeric_frame[ranked[0]], errors="coerce")
            for candidate in ranked[1:]:
                second = pd.to_numeric(numeric_frame[candidate], errors="coerce")
                subset = pd.DataFrame({"a": first, "b": second}).dropna()
                corr_value = None
                if len(subset) >= 5 and float(subset["a"].std()) > 0.0 and float(subset["b"].std()) > 0.0:
                    corr_value = float(abs(pearsonr(subset["a"], subset["b"]).statistic))
                correlations[f"{ranked[0]}__{candidate}"] = corr_value
                if (
                    len(selected_variants) < max_variants_per_base_feature
                    and corr_value is not None
                    and corr_value < allow_second_variant_correlation_threshold
                    and _reference_transform(ranked[0]) is None
                    and _reference_transform(candidate) in {"z", "percentile"}
                ):
                    selected_variants.append(candidate)
                    break
        dropped = [value for value in variants if value not in selected_variants]
        selected.extend(selected_variants)
        audit_rows.append(
            {
                "base_feature_name": base_name,
                "variants_present": sorted(variants),
                "correlations_between_variants": correlations,
                "selected_variant": selected_variants[0],
                "selected_variants": selected_variants,
                "dropped_variants": dropped,
                "reason": "base_feature_deduplication_policy",
            }
        )
    return list(dict.fromkeys(selected)), {
        "max_variants_per_base_feature": int(max_variants_per_base_feature),
        "allow_second_variant_correlation_threshold": float(allow_second_variant_correlation_threshold),
        "rows": sorted(audit_rows, key=lambda item: str(item["base_feature_name"])),
    }


def _component_counts(
    feature_columns: list[str],
    *,
    registry_by_name: dict[str, dict[str, Any]],
    component_map: dict[str, list[str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in feature_columns:
        component = _infer_feature_component(column, registry_by_name=registry_by_name, component_map=component_map)
        counts[component] = counts.get(component, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _feature_set_quality_scores(
    feature_columns: list[str],
    *,
    registry_by_name: dict[str, dict[str, Any]],
    component_map: dict[str, list[str]],
) -> dict[str, float]:
    if not feature_columns:
        return {"component_coverage_score": 0.0, "interpretability_score": 0.0, "reliable_fraction": 0.0}
    expected_components = {_canonical_anatomical_component(str(name)) for name in component_map.keys()}
    covered_components = {
        _infer_feature_component(column, registry_by_name=registry_by_name, component_map=component_map)
        for column in feature_columns
    }
    covered_effective = {value for value in covered_components if value in expected_components}
    component_coverage = float(len(covered_effective) / max(len(expected_components), 1))
    reliability_values = []
    for column in feature_columns:
        metadata = registry_by_name.get(column, {})
        reliability = str(metadata.get("reliability") or "")
        if reliability == "reliable":
            reliability_values.append(1.0)
        elif reliability == "proxy":
            reliability_values.append(0.5)
        else:
            reliability_values.append(0.0)
    reliable_fraction = float(np.mean(reliability_values)) if reliability_values else 0.0
    interpretability_score = float(0.7 * reliable_fraction + 0.3 * component_coverage)
    return {
        "component_coverage_score": component_coverage,
        "interpretability_score": interpretability_score,
        "reliable_fraction": reliable_fraction,
    }


def _pose_feature_audit(
    *,
    selected_feature_columns: list[str],
    numeric_frame: pd.DataFrame,
    trainable_frame: pd.DataFrame,
    target_column: str,
    quality_column: str = "quality_pass",
) -> dict[str, Any]:
    pose_features = [
        column
        for column in selected_feature_columns
        if column.startswith("pose_") or column.startswith("angle_") or column.endswith("_axis_angle_deg") or column.endswith("_line_angle_deg")
    ]
    rows: list[dict[str, Any]] = []
    target = pd.to_numeric(trainable_frame[target_column], errors="coerce")
    quality = pd.to_numeric(trainable_frame.get(quality_column), errors="coerce") if quality_column in trainable_frame.columns else None
    for feature_name in pose_features:
        if feature_name not in numeric_frame.columns:
            continue
        series = pd.to_numeric(numeric_frame[feature_name], errors="coerce")
        valid = pd.DataFrame({"x": series, "y": target}).dropna()
        corr_target = None
        if len(valid) >= 5 and float(valid["x"].std()) > 0.0 and float(valid["y"].std()) > 0.0:
            corr_target = float(pearsonr(valid["x"], valid["y"]).statistic)
        corr_quality = None
        if quality is not None:
            valid_quality = pd.DataFrame({"x": series, "q": quality}).dropna()
            if len(valid_quality) >= 5 and float(valid_quality["x"].std()) > 0.0 and float(valid_quality["q"].std()) > 0.0:
                corr_quality = float(pearsonr(valid_quality["x"], valid_quality["q"]).statistic)
        rows.append(
            {
                "feature_name": feature_name,
                "missing_rate": float(series.isna().mean()),
                "variance": float(series.dropna().var()) if len(series.dropna()) else 0.0,
                "corr_with_target_score": corr_target,
                "corr_with_quality_pass": corr_quality,
            }
        )
    rows.sort(key=lambda item: abs(float(item.get("corr_with_target_score") or 0.0)), reverse=True)
    return {
        "pose_feature_count_in_selected": int(len(pose_features)),
        "pose_features": pose_features,
        "rows": rows,
    }


def _pose_experiment_metrics(
    *,
    model: Any,
    numeric_frame: pd.DataFrame,
    train_index: pd.Index,
    val_index: pd.Index,
    test_index: pd.Index,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    sample_weights: np.ndarray | None,
    groups: list[float],
    strict_config: dict[str, Any],
    clip_lower: float | None,
    clip_upper: float | None,
    feature_columns: list[str],
) -> dict[str, Any]:
    if not feature_columns:
        return {"status": "skipped_no_features"}
    numeric_set = numeric_frame.loc[:, feature_columns]
    medians = numeric_set.median(axis=0, numeric_only=True).to_dict()
    numeric_set = numeric_set.fillna(value=medians)
    X_train = numeric_set.loc[train_index, feature_columns].to_numpy(dtype=np.float64)
    X_val = numeric_set.loc[val_index, feature_columns].to_numpy(dtype=np.float64)
    X_test = numeric_set.loc[test_index, feature_columns].to_numpy(dtype=np.float64)
    fitted = _fit_estimator(model, X_train, y_train, sample_weight=sample_weights)
    val_pred = _prediction_array(fitted, X_val, lower=clip_lower, upper=clip_upper)
    test_pred = _prediction_array(fitted, X_test, lower=clip_lower, upper=clip_upper)
    return {
        "status": "completed",
        "selected_feature_count": int(len(feature_columns)),
        "val": _eval_payload(y_val, val_pred, groups, strict_config),
        "test": _eval_payload(y_test, test_pred, groups, strict_config),
    }


def _learned_feature_weight_rows(
    *,
    feature_effect_rows: list[dict[str, Any]],
    feature_importance_rows: list[dict[str, Any]],
    registry_by_name: dict[str, dict[str, Any]],
    component_map: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], str]:
    effect_map = {str(row.get("feature_name")): row for row in feature_effect_rows}
    importance_map = {str(row.get("feature_name")): row for row in feature_importance_rows}
    feature_names = sorted(set(effect_map.keys()) | set(importance_map.keys()))
    rows: list[dict[str, Any]] = []
    methods: set[str] = set()
    for feature_name in feature_names:
        effect_row = effect_map.get(feature_name, {})
        importance_row = importance_map.get(feature_name, {})
        base_feature_name = _base_feature_name(feature_name)
        reference_transform = _reference_transform(feature_name)
        transform_family = _transform_family(feature_name)
        anatomical_component = _infer_feature_component(
            feature_name,
            registry_by_name=registry_by_name,
            component_map=component_map,
        )
        base_meta = registry_by_name.get(base_feature_name, {})
        direct_meta = registry_by_name.get(feature_name, {})
        original_source = str(base_meta.get("source") or direct_meta.get("source") or "other")
        source = "reference_relative" if reference_transform is not None else original_source
        source_family = "mediapipe_reference_relative" if (reference_transform is not None and str(base_feature_name).startswith("mediapipe_")) else (
            "mediapipe_direct" if str(original_source) == "mediapipe_face_mesh" else str(original_source)
        )
        signed_effect = effect_row.get("signed_effect")
        abs_effect = effect_row.get("abs_effect")
        if abs_effect is None:
            abs_effect = importance_row.get("importance")
        if signed_effect is None:
            signed_effect = 0.0
        if abs_effect is None:
            abs_effect = abs(float(signed_effect))
        method = str(effect_row.get("effect_method") or "model_importance")
        methods.add(method)
        rows.append(
            {
                "feature_name": feature_name,
                "base_feature_name": base_feature_name,
                "anatomical_component": anatomical_component,
                "transform_family": transform_family,
                "reference_transform": reference_transform,
                "source": source,
                "original_source": original_source,
                "source_family": source_family,
                "is_reference_relative": bool(reference_transform is not None),
                "component": anatomical_component,
                "reliability": base_meta.get("reliability") or direct_meta.get("reliability"),
                "signed_weight": float(signed_effect),
                "abs_weight": float(abs(abs_effect)),
                "normalized_abs_weight": 0.0,
                "weight_method": method,
            }
        )
    total_abs = float(sum(float(row["abs_weight"]) for row in rows))
    for row in rows:
        row["normalized_abs_weight"] = float(float(row["abs_weight"]) / total_abs) if total_abs > 0 else 0.0
    rows.sort(key=lambda item: float(item["abs_weight"]), reverse=True)
    method_name = ",".join(sorted(methods)) if methods else "model_importance"
    return rows, method_name


def _learned_component_weight_rows(feature_weight_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in feature_weight_rows:
        component = _canonical_anatomical_component(str(row.get("anatomical_component") or row.get("component") or "unknown"))
        entry = grouped.setdefault(
            component,
            {
                "component": component,
                "feature_count": 0,
                "sum_abs_weight": 0.0,
                "signed_sum": 0.0,
                "mean_abs_weight": 0.0,
                "normalized_component_weight": 0.0,
                "top_features": [],
            },
        )
        entry["feature_count"] += 1
        entry["sum_abs_weight"] += float(row.get("abs_weight", 0.0) or 0.0)
        entry["signed_sum"] += float(row.get("signed_weight", 0.0) or 0.0)
        entry["top_features"].append(
            {
                "feature_name": str(row.get("feature_name")),
                "abs_weight": float(row.get("abs_weight", 0.0) or 0.0),
                "signed_weight": float(row.get("signed_weight", 0.0) or 0.0),
                "normalized_abs_weight": float(row.get("normalized_abs_weight", 0.0) or 0.0),
            }
        )
    total = float(sum(float(item["sum_abs_weight"]) for item in grouped.values()))
    for entry in grouped.values():
        feature_count = max(int(entry["feature_count"]), 1)
        entry["mean_abs_weight"] = float(float(entry["sum_abs_weight"]) / feature_count)
        entry["normalized_component_weight"] = float(float(entry["sum_abs_weight"]) / total) if total > 0 else 0.0
        entry["top_features"] = sorted(entry["top_features"], key=lambda item: float(item["abs_weight"]), reverse=True)[:10]
    rows = sorted(grouped.values(), key=lambda item: float(item["normalized_component_weight"]), reverse=True)
    normalized_map = {str(row["component"]): float(row["normalized_component_weight"]) for row in rows}
    return rows, normalized_map


def _learned_group_weight_rows(
    feature_weight_rows: list[dict[str, Any]],
    *,
    group_key: str,
    group_label: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in feature_weight_rows:
        group_name = str(row.get(group_key) or "unknown")
        entry = grouped.setdefault(
            group_name,
            {
                group_label: group_name,
                "feature_count": 0,
                "sum_abs_weight": 0.0,
                "signed_sum": 0.0,
                "normalized_weight": 0.0,
            },
        )
        entry["feature_count"] += 1
        entry["sum_abs_weight"] += float(row.get("abs_weight", 0.0) or 0.0)
        entry["signed_sum"] += float(row.get("signed_weight", 0.0) or 0.0)
    total = float(sum(float(item["sum_abs_weight"]) for item in grouped.values()))
    for entry in grouped.values():
        entry["normalized_weight"] = float(float(entry["sum_abs_weight"]) / total) if total > 0 else 0.0
    rows = sorted(grouped.values(), key=lambda item: float(item["normalized_weight"]), reverse=True)
    normalized_map = {str(row[group_label]): float(row["normalized_weight"]) for row in rows}
    return rows, normalized_map


def _correlation_audit_rows(
    numeric_frame: pd.DataFrame,
    feature_columns: list[str],
    registry_by_name: dict[str, dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    if len(feature_columns) < 2:
        return []
    corr = numeric_frame.loc[:, feature_columns].corr(method="pearson").abs()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for left in feature_columns:
        for right in feature_columns:
            if left >= right:
                continue
            key = (left, right)
            if key in seen:
                continue
            seen.add(key)
            value = corr.loc[left, right]
            if not np.isfinite(value) or float(value) < threshold:
                continue
            rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "abs_correlation": float(value),
                    "source_a": registry_by_name.get(left, {}).get("source"),
                    "source_b": registry_by_name.get(right, {}).get("source"),
                    "reliability_a": registry_by_name.get(left, {}).get("reliability"),
                    "reliability_b": registry_by_name.get(right, {}).get("reliability"),
                }
            )
    return sorted(rows, key=lambda item: item["abs_correlation"], reverse=True)


def _deduplicate_correlated_features(
    numeric_frame: pd.DataFrame,
    feature_columns: list[str],
    registry_by_name: dict[str, dict[str, Any]],
    *,
    threshold: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    corr_rows = _correlation_audit_rows(numeric_frame, feature_columns, registry_by_name, threshold=threshold)
    if not corr_rows:
        return list(feature_columns), []

    source_priority = {"mediapipe_face_mesh": 0, "insightface_106": 1, "fused": 2, "other": 3}
    missing_rate = numeric_frame.loc[:, feature_columns].isna().mean().to_dict()
    drop: set[str] = set()
    audit_rows: list[dict[str, Any]] = []

    def _rank(feature_name: str) -> tuple[float, int, float, str]:
        metadata = registry_by_name.get(feature_name, {})
        source = str(metadata.get("source") or "other")
        reliability = str(metadata.get("reliability") or "")
        reliability_rank = 0 if reliability == "reliable" else 1 if reliability == "proxy" else 2
        return (
            float(missing_rate.get(feature_name, 1.0)),
            source_priority.get(source, 3),
            reliability_rank,
            feature_name,
        )

    for row in corr_rows:
        left = str(row["feature_a"])
        right = str(row["feature_b"])
        if left in drop or right in drop:
            continue
        keep, remove = sorted([left, right], key=_rank)
        drop.add(remove)
        audit_rows.append(
            {
                **row,
                "kept_feature": keep,
                "dropped_feature": remove,
                "reason": "high_correlation_duplicate",
            }
        )
    return [column for column in feature_columns if column not in drop], audit_rows


def _build_feature_sets(
    trainable_frame: pd.DataFrame,
    numeric_all: pd.DataFrame,
    base_selected_features: list[str],
    *,
    max_missing: float,
    target_values: pd.Series,
    component_map: dict[str, list[str]],
    feature_source_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    registry_by_name = _feature_registry_map()
    missing_rate = numeric_all.isna().mean() if not numeric_all.empty else pd.Series(dtype=float)

    def _stable_mediapipe(feature_name: str) -> bool:
        metadata = registry_by_name.get(feature_name, {})
        if str(metadata.get("source")) != "mediapipe_face_mesh":
            return False
        if not bool(metadata.get("recommended_for_scorer", False)):
            return False
        if feature_name.startswith("raw_") or feature_name.endswith("_px"):
            return False
        if float(missing_rate.get(feature_name, 1.0)) > max_missing:
            return False
        if float(numeric_all[feature_name].dropna().var()) <= 1e-12:
            return False
        return True

    def _expanded_mediapipe_feature(feature_name: str) -> bool:
        metadata = registry_by_name.get(feature_name, {})
        if str(metadata.get("source")) != "mediapipe_face_mesh":
            return False
        if feature_name.startswith("raw_") or feature_name.endswith("_px"):
            return False
        if float(missing_rate.get(feature_name, 1.0)) > max_missing:
            return False
        if float(numeric_all[feature_name].dropna().var()) <= 1e-12:
            return False
        family = str(metadata.get("feature_family") or "")
        reliability = str(metadata.get("reliability") or "")
        if reliability in {"unstable", "unavailable_without_extra_model"}:
            return False
        return family in {"normalized_ratios", "symmetry_metrics", "angle_metrics"}

    baseline_features = [name for name in base_selected_features if name in numeric_all.columns and not name.startswith("mediapipe_")]
    mediapipe_features = [name for name in numeric_all.columns if _stable_mediapipe(name)]
    expanded_mediapipe_forced = [name for name in numeric_all.columns if _expanded_mediapipe_feature(name)]
    mediapipe_core_ratios = [
        name
        for name in mediapipe_features
        if str(registry_by_name.get(name, {}).get("feature_family")) == "normalized_ratios"
    ]
    mediapipe_shape_symmetry_angles = [
        name
        for name in mediapipe_features
        if str(registry_by_name.get(name, {}).get("feature_family"))
        in {"normalized_ratios", "symmetry_metrics", "angle_metrics"}
    ]
    mediapipe_no_pose = [
        name
        for name in mediapipe_shape_symmetry_angles
        if "pose_" not in name and "_axis_" not in name and "tilt" not in name
    ]
    component_top_k = 10
    weighted_mediapipe, weighted_mediapipe_diagnostics = _component_balanced_top_features(
        feature_columns=mediapipe_features,
        numeric_frame=numeric_all,
        target_values=target_values,
        registry_by_name=registry_by_name,
        component_map=component_map,
        top_k_per_component=component_top_k,
    )
    combined_features = list(dict.fromkeys(baseline_features + mediapipe_features))
    combined_weighted_features = list(dict.fromkeys(baseline_features + weighted_mediapipe))
    combined_no_dup, duplicate_audit = _deduplicate_correlated_features(
        numeric_all.fillna(numeric_all.median(axis=0, numeric_only=True)),
        combined_features,
        registry_by_name,
        threshold=0.98,
    )
    combined_weighted_no_dup, weighted_duplicate_audit = _deduplicate_correlated_features(
        numeric_all.fillna(numeric_all.median(axis=0, numeric_only=True)),
        combined_weighted_features,
        registry_by_name,
        threshold=0.98,
    )
    mediapipe_expanded_dedup, expanded_duplicate_audit = _deduplicate_correlated_features(
        numeric_all.fillna(numeric_all.median(axis=0, numeric_only=True)),
        expanded_mediapipe_forced,
        registry_by_name,
        threshold=0.98,
    )
    duplicate_audit = list(duplicate_audit) + list(weighted_duplicate_audit)
    duplicate_audit.extend(expanded_duplicate_audit)
    feature_sets = {
        "baseline_insightface_geometry": baseline_features,
        "mediapipe_stable_geometry": mediapipe_features,
        "mediapipe_weighted_geometry": weighted_mediapipe,
        "expanded_mediapipe_geometry_forced": expanded_mediapipe_forced,
        "mediapipe_only_core_ratios": mediapipe_core_ratios,
        "mediapipe_only_shape_symmetry_angles": mediapipe_shape_symmetry_angles,
        "mediapipe_only_component_balanced": weighted_mediapipe,
        "mediapipe_only_no_pose": mediapipe_no_pose,
        "mediapipe_only_expanded_full_dedup": mediapipe_expanded_dedup,
        "mediapipe_only_high_low_weighted": weighted_mediapipe,
        "combined_stable_geometry": combined_features,
        "combined_no_duplicate_geometry": combined_no_dup,
        "combined_weighted_no_duplicate": combined_weighted_no_dup,
    }
    policy = dict(feature_source_policy or {})
    if str(policy.get("mode", "")).strip().lower() == "mediapipe_only":
        for set_name, columns in list(feature_sets.items()):
            filtered, _ = _apply_mediapipe_only_policy(columns, registry_by_name=registry_by_name)
            feature_sets[set_name] = filtered
    metadata = {
        name: {
            "selected_feature_count": int(len(columns)),
            "mediapipe_selected_count": int(sum(1 for column in columns if column.startswith("mediapipe_"))),
            "source_counts": _feature_source_counts(columns, registry_by_name),
            "component_counts": _component_counts(columns, registry_by_name=registry_by_name, component_map=component_map),
        }
        for name, columns in feature_sets.items()
    }
    metadata["mediapipe_weighted_geometry"]["component_balanced_selection"] = weighted_mediapipe_diagnostics
    return feature_sets, metadata, duplicate_audit


def _stable_reference_relative(
    column: str,
    numeric_all: pd.DataFrame,
    *,
    max_missing: float,
) -> bool:
    if not column.startswith(REFERENCE_FEATURE_PREFIXES):
        return False
    if column.startswith("ref_ratio_") and column.endswith("_px"):
        return False
    missing_rate = float(numeric_all[column].isna().mean()) if column in numeric_all.columns else 1.0
    if missing_rate > max_missing:
        return False
    variance = float(numeric_all[column].dropna().var()) if column in numeric_all.columns and len(numeric_all[column].dropna()) else 0.0
    if variance <= 1e-12:
        return False
    return True


def _reference_kind(feature_name: str) -> str:
    if feature_name.startswith("ref_z_"):
        return "ref_z"
    if feature_name.startswith("ref_ratio_"):
        return "ref_ratio"
    if feature_name.startswith("ref_dir_"):
        return "ref_dir"
    if feature_name.startswith("ref_absdev_"):
        return "ref_absdev"
    if feature_name.startswith("ref_percentile_"):
        return "ref_percentile"
    return "other"


def _reference_base_feature(feature_name: str) -> str:
    for prefix in REFERENCE_FEATURE_PREFIXES:
        if feature_name.startswith(prefix):
            return feature_name[len(prefix) :]
    return feature_name


def _build_reference_selection_diagnostics(
    *,
    numeric_all: pd.DataFrame,
    raw_feature_columns: list[str],
    max_missing: float,
    target_values: np.ndarray,
) -> dict[str, Any]:
    all_ref = [column for column in numeric_all.columns if column.startswith(REFERENCE_FEATURE_PREFIXES)]
    per_feature: dict[str, dict[str, Any]] = {}
    for column in all_ref:
        series = pd.to_numeric(numeric_all[column], errors="coerce")
        missing_rate = float(series.isna().mean())
        variance = float(series.dropna().var()) if len(series.dropna()) else 0.0
        row = {
            "feature_name": column,
            "kind": _reference_kind(column),
            "base_feature": _reference_base_feature(column),
            "missing_rate": missing_rate,
            "variance": variance,
            "stages": [],
            "excluded": False,
            "exclusion_reason": None,
        }
        if missing_rate > max_missing:
            row["stages"].append("failed_missingness")
            row["excluded"] = True
            row["exclusion_reason"] = "missing_rate_too_high"
        elif variance <= 1e-12:
            row["stages"].append("failed_variance")
            row["excluded"] = True
            row["exclusion_reason"] = "near_zero_variance"
        else:
            row["stages"].append("passed_missingness_variance")
        per_feature[column] = row

    stable_ref = [name for name, payload in per_feature.items() if not payload["excluded"]]
    corr_rows: list[dict[str, Any]] = []
    if len(stable_ref) > 1:
        stable_frame = numeric_all.loc[:, stable_ref].copy()
        stable_frame = stable_frame.fillna(stable_frame.median(axis=0, numeric_only=True))
        corr = stable_frame.corr(method="pearson").abs()
        for index, left in enumerate(stable_ref):
            for right in stable_ref[index + 1 :]:
                value = float(corr.loc[left, right])
                if not np.isfinite(value) or value <= 0.98:
                    continue
                corr_rows.append({"feature_a": left, "feature_b": right, "abs_correlation": value})

    univariate_rows: list[dict[str, Any]] = []
    target_series = pd.Series(target_values, index=numeric_all.index[: len(target_values)], dtype=float)
    for feature_name, payload in per_feature.items():
        if payload["excluded"]:
            continue
        feature_series = pd.to_numeric(numeric_all[feature_name], errors="coerce")
        valid = pd.DataFrame({"x": feature_series, "y": target_series}).dropna()
        correlation = 0.0
        if len(valid) >= 5 and float(valid["x"].std()) > 0.0 and float(valid["y"].std()) > 0.0:
            correlation = float(abs(pearsonr(valid["x"], valid["y"]).statistic))
        payload["univariate_abs_pearson"] = correlation
        univariate_rows.append({"feature_name": feature_name, "abs_pearson": correlation})
    univariate_rows.sort(key=lambda item: float(item["abs_pearson"]), reverse=True)

    top_100 = {row["feature_name"] for row in univariate_rows[:100]}
    for feature_name, payload in per_feature.items():
        if payload["excluded"]:
            continue
        payload["stages"].append("passed_univariate")
        if feature_name not in top_100:
            payload["top100_candidate"] = False
        else:
            payload["top100_candidate"] = True

    kind_counts = {
        "ref_z": int(sum(1 for name in all_ref if name.startswith("ref_z_"))),
        "ref_ratio": int(sum(1 for name in all_ref if name.startswith("ref_ratio_"))),
        "ref_dir": int(sum(1 for name in all_ref if name.startswith("ref_dir_"))),
        "ref_absdev": int(sum(1 for name in all_ref if name.startswith("ref_absdev_"))),
        "ref_percentile": int(sum(1 for name in all_ref if name.startswith("ref_percentile_"))),
    }
    excluded_rows = [payload for payload in per_feature.values() if payload["excluded"]]
    exclusion_counts: dict[str, int] = {}
    for row in excluded_rows:
        reason = str(row.get("exclusion_reason") or "unknown")
        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

    return {
        "all_reference_feature_count": int(len(all_ref)),
        "reference_kind_counts": kind_counts,
        "stable_after_missing_variance_count": int(sum(1 for payload in per_feature.values() if "passed_missingness_variance" in payload["stages"])),
        "stable_after_correlation_count": int(sum(1 for payload in per_feature.values() if not payload["excluded"])),
        "excluded_count": int(len(excluded_rows)),
        "exclusion_reason_counts": exclusion_counts,
        "high_correlation_pairs_count": int(len(corr_rows)),
        "high_correlation_pairs": corr_rows[:200],
        "top_univariate_reference_features": univariate_rows[:200],
        "per_feature": sorted(per_feature.values(), key=lambda item: str(item["feature_name"])),
        "raw_feature_count": int(len(raw_feature_columns)),
    }


def _build_reference_feature_sets(
    numeric_all: pd.DataFrame,
    raw_feature_columns: list[str],
    target_values: np.ndarray,
    *,
    max_missing: float,
    component_map: dict[str, list[str]],
    feature_source_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]], dict[str, Any]]:
    diagnostics = _build_reference_selection_diagnostics(
        numeric_all=numeric_all,
        raw_feature_columns=raw_feature_columns,
        max_missing=max_missing,
        target_values=target_values,
    )
    per_feature = {str(row["feature_name"]): row for row in diagnostics.get("per_feature", [])}
    stable_ref = [name for name, payload in per_feature.items() if not bool(payload.get("excluded"))]
    ref_z_only = [name for name in stable_ref if name.startswith("ref_z_")]
    ref_percentile_only = [name for name in stable_ref if name.startswith("ref_percentile_")]
    ref_absdev_only = [name for name in stable_ref if name.startswith("ref_absdev_")]
    ref_dir_only = [name for name in stable_ref if name.startswith("ref_dir_")]
    ref_ratio_only = [name for name in stable_ref if name.startswith("ref_ratio_")]

    top_univariate_rows = list(diagnostics.get("top_univariate_reference_features", []))
    top_50 = [str(row["feature_name"]) for row in top_univariate_rows[:50] if str(row["feature_name"]) in stable_ref]
    top_100 = [str(row["feature_name"]) for row in top_univariate_rows[:100] if str(row["feature_name"]) in stable_ref]

    ref_z_plus_percentile = list(dict.fromkeys(ref_z_only + ref_percentile_only))
    ref_deviation_compact = list(dict.fromkeys(ref_z_only + ref_absdev_only))
    combined_top_50 = list(dict.fromkeys(list(raw_feature_columns) + top_50))
    combined_top_100 = list(dict.fromkeys(list(raw_feature_columns) + top_100))
    combined_reference = list(dict.fromkeys(list(raw_feature_columns) + stable_ref))
    stable_mediapipe_raw = [column for column in raw_feature_columns if column.startswith("mediapipe_")]
    stable_mediapipe_ref = [column for column in stable_ref if _reference_base_feature(column).startswith("mediapipe_")]
    stable_mediapipe_ref_reduced = [
        column
        for column in stable_mediapipe_ref
        if column.startswith(("ref_z_mediapipe_", "ref_percentile_mediapipe_", "ref_absdev_mediapipe_"))
    ]
    mediapipe_abs_plus_ref = list(dict.fromkeys(stable_mediapipe_raw + stable_mediapipe_ref_reduced))
    mediapipe_no_pose = [
        column
        for column in mediapipe_abs_plus_ref
        if "pose_" not in column and "_axis_" not in column and "tilt" not in column
    ]
    weighted_reference, weighted_reference_diag = _component_balanced_top_features(
        feature_columns=mediapipe_abs_plus_ref if mediapipe_abs_plus_ref else stable_ref,
        numeric_frame=numeric_all,
        target_values=pd.Series(target_values, index=numeric_all.index[: len(target_values)]),
        registry_by_name=_feature_registry_map(),
        component_map=component_map,
        top_k_per_component=10,
    )
    mediapipe_plus_reference_weighted = list(dict.fromkeys(stable_mediapipe_raw + weighted_reference))
    combined_weighted_reference = list(dict.fromkeys(list(raw_feature_columns) + weighted_reference))
    combined_no_dup, duplicate_audit = _deduplicate_correlated_features(
        numeric_all.fillna(numeric_all.median(axis=0, numeric_only=True)),
        combined_reference,
        _feature_registry_map(),
        threshold=0.98,
    )

    feature_sets = {
        "raw_features_only": list(raw_feature_columns),
        "fairface_reference_features_only": stable_ref,
        "combined_raw_and_reference": combined_reference,
        "ref_z_only_stable": ref_z_only,
        "ref_percentile_only_stable": ref_percentile_only,
        "ref_z_plus_percentile_stable": ref_z_plus_percentile,
        "ref_deviation_compact": ref_deviation_compact,
        "absolute_plus_best_reference_50": combined_top_50,
        "absolute_plus_best_reference_100": combined_top_100,
        "combined_reference_no_duplicate": combined_no_dup,
        "mediapipe_plus_reference_weighted": mediapipe_plus_reference_weighted,
        "combined_weighted_no_duplicate": combined_weighted_reference,
        "combined_all_stable_reduced": list(dict.fromkeys(list(raw_feature_columns) + ref_z_plus_percentile + ref_absdev_only)),
        "reference_ratio_only_stable": ref_ratio_only,
        "reference_direction_only_stable": ref_dir_only,
        "mediapipe_only_reference_relative": stable_mediapipe_ref_reduced,
        "mediapipe_only_absolute_plus_reference": mediapipe_abs_plus_ref,
        "mediapipe_only_component_balanced": mediapipe_plus_reference_weighted,
        "mediapipe_only_no_pose": mediapipe_no_pose,
        "mediapipe_only_high_low_weighted": mediapipe_abs_plus_ref,
        "mediapipe_only_expanded_full_dedup": combined_no_dup,
    }
    policy = dict(feature_source_policy or {})
    if str(policy.get("mode", "")).strip().lower() == "mediapipe_only":
        registry_by_name = _feature_registry_map()
        for set_name, columns in list(feature_sets.items()):
            filtered, _ = _apply_mediapipe_only_policy(columns, registry_by_name=registry_by_name)
            feature_sets[set_name] = filtered
    metadata = {
        name: {
            "selected_feature_count": int(len(columns)),
            "mediapipe_selected_count": int(sum(1 for column in columns if "mediapipe_" in column)),
            "reference_relative_selected_count": int(sum(1 for column in columns if column.startswith(REFERENCE_FEATURE_PREFIXES))),
            "source_counts": _feature_source_counts(columns, _feature_registry_map()),
            "component_counts": _component_counts(columns, registry_by_name=_feature_registry_map(), component_map=component_map),
        }
        for name, columns in feature_sets.items()
    }
    diagnostics["candidate_reference_feature_sets"] = {
        "stable_ref_total": int(len(stable_ref)),
        "ref_z_only_stable": int(len(ref_z_only)),
        "ref_percentile_only_stable": int(len(ref_percentile_only)),
        "ref_z_plus_percentile_stable": int(len(ref_z_plus_percentile)),
        "ref_deviation_compact": int(len(ref_deviation_compact)),
        "absolute_plus_best_reference_50": int(len(combined_top_50)),
        "absolute_plus_best_reference_100": int(len(combined_top_100)),
        "combined_reference_no_duplicate": int(len(combined_no_dup)),
        "combined_all_stable_reduced": int(len(feature_sets["combined_all_stable_reduced"])),
        "mediapipe_plus_reference_weighted": int(len(mediapipe_plus_reference_weighted)),
        "combined_weighted_no_duplicate": int(len(combined_weighted_reference)),
        "mediapipe_only_reference_relative": int(len(stable_mediapipe_ref_reduced)),
        "mediapipe_only_absolute_plus_reference": int(len(mediapipe_abs_plus_ref)),
        "mediapipe_only_no_pose": int(len(mediapipe_no_pose)),
        "mediapipe_only_component_balanced": int(len(mediapipe_plus_reference_weighted)),
    }
    diagnostics["duplicate_audit_rows"] = duplicate_audit
    diagnostics["component_balanced_reference_selection"] = weighted_reference_diag
    return feature_sets, metadata, diagnostics


def _baseline_predictions(y_train: np.ndarray, y_test: np.ndarray, nearest_frequent_group: float | None) -> dict[str, dict[str, float | None]]:
    baselines: dict[str, dict[str, float | None]] = {}
    if y_test.size == 0 or y_train.size == 0:
        return baselines
    mean_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=np.float64)
    median_pred = np.full_like(y_test, fill_value=float(np.median(y_train)), dtype=np.float64)
    baselines["mean_predictor"] = _regression_metrics(y_test, mean_pred)
    baselines["median_predictor"] = _regression_metrics(y_test, median_pred)
    if nearest_frequent_group is not None:
        freq_pred = np.full_like(y_test, fill_value=float(nearest_frequent_group), dtype=np.float64)
        baselines["nearest_frequency_baseline"] = _regression_metrics(y_test, freq_pred)
    return baselines


def _nearest_frequent_group(y: np.ndarray) -> float | None:
    if y.size == 0:
        return None
    values, counts = np.unique(y, return_counts=True)
    return float(values[int(np.argmax(counts))])


def _build_pairwise_rows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    max_pairs_per_anchor: int,
    close_distance_threshold: float,
    close_distance_pair_multiplier: float,
    large_distance_threshold: float,
    large_distance_pair_multiplier: float,
    high_vs_mid_pair_multiplier: float,
    low_vs_mid_pair_multiplier: float,
    elite_vs_average_pair_multiplier: float,
    low_score_threshold: float,
    high_score_threshold: float,
    elite_score_threshold: float,
    average_zone_center: float,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    distance_counts: dict[str, int] = {}
    group_pair_counts: dict[str, int] = {}
    weighted_pair_distribution = {"close": 0, "far": 0, "large_gap": 0, "high_vs_mid": 0, "low_vs_mid": 0, "elite_vs_average": 0}
    high_vs_mid_pair_count = 0
    low_vs_mid_pair_count = 0
    elite_vs_average_pair_count = 0
    values = frame.reset_index(drop=True)
    for anchor_idx, anchor in values.iterrows():
        anchor_score = float(anchor["target_score"])
        others = values.loc[values["target_score"] != anchor_score].copy()
        if others.empty:
            continue
        others["score_distance"] = (others["target_score"].astype(float) - anchor_score).abs()
        others["priority_weight"] = np.where(
            others["score_distance"] <= close_distance_threshold,
            close_distance_pair_multiplier,
            1.0,
        )
        others.loc[others["score_distance"] >= large_distance_threshold, "priority_weight"] *= large_distance_pair_multiplier
        anchor_is_low = anchor_score <= low_score_threshold
        anchor_is_high = anchor_score >= high_score_threshold
        anchor_is_elite = anchor_score >= elite_score_threshold
        for other_idx, other in others.iterrows():
            other_score = float(other["target_score"])
            other_is_mid = abs(other_score - average_zone_center) <= 1.5
            if anchor_is_high and other_is_mid:
                others.loc[other_idx, "priority_weight"] *= high_vs_mid_pair_multiplier
            if anchor_is_low and other_is_mid:
                others.loc[other_idx, "priority_weight"] *= low_vs_mid_pair_multiplier
            if anchor_is_elite and other_score <= average_zone_center + 1.0:
                others.loc[other_idx, "priority_weight"] *= elite_vs_average_pair_multiplier
        sample_count = min(max_pairs_per_anchor, len(others))
        weights = others["priority_weight"].astype(float).to_numpy(dtype=np.float64)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        if weights.sum() <= 0:
            weights = np.ones(len(others), dtype=np.float64)
        probabilities = weights / weights.sum()
        sampled_indices = rng.choice(len(others), size=sample_count, replace=False, p=probabilities)
        sample = others.iloc[np.asarray(sampled_indices, dtype=np.int64)]
        for _, other in sample.iterrows():
            other_score = float(other["target_score"])
            label = 1 if anchor_score > other_score else 0
            if anchor_score == other_score:
                continue
            diff = values.loc[anchor_idx, feature_columns].astype(float) - other[feature_columns].astype(float)
            row = {
                "left_image_id": anchor.get("image_id"),
                "right_image_id": other.get("image_id"),
                "left_score": anchor_score,
                "right_score": other_score,
                "pair_label": int(label),
                "score_distance": float(abs(anchor_score - other_score)),
                "left_group": str(anchor["target_group_label"]),
                "right_group": str(other["target_group_label"]),
            }
            row.update({feature_name: float(diff[feature_name]) for feature_name in feature_columns})
            rows.append(row)
            distance_key = "close" if row["score_distance"] <= close_distance_threshold else "far"
            distance_counts[distance_key] = distance_counts.get(distance_key, 0) + 1
            if row["score_distance"] >= large_distance_threshold:
                weighted_pair_distribution["large_gap"] += 1
            if anchor_score >= high_score_threshold and abs(other_score - average_zone_center) <= 1.5:
                high_vs_mid_pair_count += 1
                weighted_pair_distribution["high_vs_mid"] += 1
            if anchor_score <= low_score_threshold and abs(other_score - average_zone_center) <= 1.5:
                low_vs_mid_pair_count += 1
                weighted_pair_distribution["low_vs_mid"] += 1
            if anchor_score >= elite_score_threshold and other_score <= average_zone_center + 1.0:
                elite_vs_average_pair_count += 1
                weighted_pair_distribution["elite_vs_average"] += 1
            weighted_pair_distribution[distance_key] += 1
            group_pair_key = f"{row['left_group']}__{row['right_group']}"
            group_pair_counts[group_pair_key] = group_pair_counts.get(group_pair_key, 0) + 1
    pair_frame = pd.DataFrame(rows)
    summary = {
        "image_count": int(len(frame)),
        "pair_count": int(len(pair_frame)),
        "score_distance_distribution": distance_counts,
        "pair_label_balance": pair_frame["pair_label"].value_counts().to_dict() if not pair_frame.empty else {},
        "pair_counts_per_score_group": group_pair_counts,
        "high_vs_mid_pair_count": int(high_vs_mid_pair_count),
        "low_vs_mid_pair_count": int(low_vs_mid_pair_count),
        "elite_vs_average_pair_count": int(elite_vs_average_pair_count),
        "weighted_pair_distribution": weighted_pair_distribution,
    }
    return pair_frame, summary


def _build_pairwise_models(random_seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_seed)),
            ]
        ),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=random_seed, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=random_seed),
    }


def _pairwise_summary(estimator: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if y.size == 0:
        return {"pairwise_accuracy": None, "roc_auc": None, "balanced_accuracy": None}
    pred = estimator.predict(X)
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    recalls = []
    for class_index, support in enumerate(matrix.sum(axis=1)):
        if support > 0:
            recalls.append(float(matrix[class_index, class_index] / support))
    metrics = {
        "pairwise_accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "roc_auc": None,
    }
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X)[:, 1]
        if len(np.unique(y)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y, probs))
    return metrics


def _pairwise_predict_scores(
    estimator: Any,
    feature_columns: list[str],
    reference_frame: pd.DataFrame,
    X_target: pd.DataFrame,
    score_groups: list[float],
) -> np.ndarray:
    if X_target.empty or reference_frame.empty:
        return np.asarray([], dtype=np.float64)
    predicted: list[float] = []
    ref_features = reference_frame.loc[:, feature_columns].astype(float)
    ref_scores = reference_frame["target_score"].astype(float).to_numpy(dtype=np.float64)
    for _, row in X_target.iterrows():
        repeated = pd.DataFrame([row.loc[feature_columns].astype(float).to_dict()] * len(reference_frame))
        diff = repeated.to_numpy(dtype=np.float64) - ref_features.to_numpy(dtype=np.float64)
        if hasattr(estimator, "predict_proba"):
            probs = estimator.predict_proba(diff)[:, 1]
        else:
            decision = estimator.decision_function(diff)
            probs = 1.0 / (1.0 + np.exp(-decision))
        weights = probs + 1e-6
        approx = float(np.average(ref_scores, weights=weights))
        approx = clip_prediction(approx, lower=min(score_groups) if score_groups else None, upper=max(score_groups) if score_groups else None)
        predicted.append(approx)
    return np.asarray(predicted, dtype=np.float64)


def _build_ordinal_models(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[float],
    *,
    random_seed: int,
    min_positive_threshold_samples: int,
    sample_weights: np.ndarray | None,
) -> dict[str, Any]:
    thresholds: list[dict[str, Any]] = []
    for threshold in groups[:-1]:
        labels = (y >= float(threshold)).astype(int)
        positives = int(labels.sum())
        negatives = int(len(labels) - positives)
        if positives < min_positive_threshold_samples or negatives < min_positive_threshold_samples:
            continue
        estimator = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=random_seed))])
        estimator = _fit_estimator(estimator, X, labels, sample_weight=sample_weights)
        thresholds.append({"threshold": float(threshold), "estimator": estimator})
    return {"thresholds": thresholds, "groups": groups}


def _ordinal_predict(ordinal_payload: dict[str, Any], X: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    if X.size == 0:
        return np.asarray([], dtype=np.float64), []
    groups = [float(value) for value in ordinal_payload.get("groups", ordinal_payload.get("score_groups", []))]
    threshold_models = ordinal_payload.get("thresholds", [])
    if not threshold_models or not groups:
        return np.asarray([], dtype=np.float64), []
    predictions: list[float] = []
    debug: list[dict[str, float]] = []
    for row in X:
        threshold_probs: dict[str, float] = {}
        score_value = groups[0]
        for item in threshold_models:
            threshold = float(item["threshold"])
            estimator = item["estimator"]
            prob = float(estimator.predict_proba(np.asarray([row], dtype=np.float64))[0, 1])
            threshold_probs[str(threshold)] = prob
        sorted_groups = sorted(groups)
        group_probabilities: list[float] = []
        for idx, group in enumerate(sorted_groups):
            prob_ge_group = threshold_probs.get(str(group), 1.0)
            next_group = sorted_groups[idx + 1] if idx + 1 < len(sorted_groups) else None
            prob_ge_next = threshold_probs.get(str(next_group), 0.0) if next_group is not None else 0.0
            group_probabilities.append(max(prob_ge_group - prob_ge_next, 0.0))
        total = sum(group_probabilities) or 1.0
        normalized = [value / total for value in group_probabilities]
        score_value = float(sum(group * probability for group, probability in zip(sorted_groups, normalized)))
        predictions.append(score_value)
        debug.append(threshold_probs)
    return np.asarray(predictions, dtype=np.float64), debug


def _ensemble_prediction(values: dict[str, float | None], weights: dict[str, float]) -> tuple[float | None, float | None]:
    available = [(name, float(value), float(weights.get(name, 0.0))) for name, value in values.items() if value is not None]
    if not available:
        return None, None
    weight_sum = sum(weight for _, _, weight in available) or 1.0
    normalized = [(name, value, weight / weight_sum) for name, value, weight in available]
    predicted = float(sum(value * weight for _, value, weight in normalized))
    std = float(np.std([value for _, value, _ in normalized])) if len(normalized) > 1 else 0.0
    return predicted, std


def _pairwise_reference_frame(
    target_frame: pd.DataFrame,
    numeric_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_column: str,
    target_group_column: str,
    per_group: int,
) -> pd.DataFrame:
    if target_frame.empty:
        return pd.DataFrame(columns=[target_column, target_group_column, *feature_columns])
    combined = pd.concat(
        [
            target_frame.loc[:, [target_column, target_group_column]],
            numeric_frame.loc[target_frame.index, feature_columns],
        ],
        axis=1,
    )
    return combined.groupby(target_group_column, group_keys=False).head(per_group).copy()


def _edge_weighted_profile(
    *,
    target_scores: np.ndarray,
    base_weights: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    edge_multiplier: float,
) -> np.ndarray:
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    score_values = np.asarray(target_scores, dtype=np.float64)
    edge_mask = (score_values <= float(low_threshold)) | (score_values >= float(high_threshold))
    weights[edge_mask] = weights[edge_mask] * float(edge_multiplier)
    return weights


def run_scorer_training(project_root: Path | None = None) -> ScorerTrainingArtifacts:
    root = (project_root or Path.cwd()).resolve()
    configs = load_scoring_configs(root)
    config = configs["scoring"]["scoring"]

    features_path = (root / str(config["features_path"])).resolve()
    summary_path = (root / str(config["summary_paths"]["training_summary_path"])).resolve()
    selected_features_path = (root / str(config["summary_paths"]["selected_features_path"])).resolve()
    feature_importance_path = (root / str(config["summary_paths"]["feature_importance_path"])).resolve()
    feature_correlations_path = (root / str(config["summary_paths"]["feature_correlations_path"])).resolve()
    global_feature_importance_path = (root / str(config["summary_paths"]["global_feature_importance_path"])).resolve()
    component_importance_path = (root / str(config["summary_paths"]["component_importance_path"])).resolve()
    feature_effects_path = (root / str(config["summary_paths"]["feature_effects_path"])).resolve()
    group_confusion_path = (root / str(config["summary_paths"]["group_confusion_path"])).resolve()
    groupwise_errors_path = (root / str(config["summary_paths"]["groupwise_errors_path"])).resolve()
    quality_failures_path = (root / str(config["summary_paths"]["quality_failures_path"])).resolve()
    pairwise_summary_path = (root / str(config["summary_paths"]["pairwise_summary_path"])).resolve()
    ordinal_summary_path = (root / str(config["summary_paths"]["ordinal_summary_path"])).resolve()
    ensemble_summary_path = (root / str(config["summary_paths"]["ensemble_summary_path"])).resolve()
    pairwise_dataset_summary_path = (root / str(config["summary_paths"]["pairwise_dataset_summary_path"])).resolve()
    calibration_summary_path = (root / "reports/metrics/scorer_calibration_by_group.json").resolve()
    extreme_bias_summary_path = (root / "reports/metrics/scorer_extreme_bias_summary.json").resolve()
    prediction_examples_path = (root / "reports/metrics/scorer_prediction_examples_by_group.json").resolve()
    feature_set_comparison_path = (root / "reports/metrics/scorer_feature_set_comparison.json").resolve()
    guardrails_selection_path = (root / "reports/metrics/scorer_model_selection_guardrails.json").resolve()
    primary_regression_comparison_path = (root / "reports/metrics/scorer_primary_regression_comparison.json").resolve()
    expanded_geometry_comparison_path = (root / "reports/metrics/expanded_geometry_scorer_comparison.json").resolve()
    mediapipe_only_comparison_path = (root / "reports/metrics/mediapipe_only_scorer_comparison.json").resolve()
    edge_weighting_comparison_path = (root / "reports/metrics/scorer_edge_weighting_comparison.json").resolve()
    weighted_model_comparison_path = (root / "reports/metrics/scorer_weighted_model_comparison.json").resolve()
    learned_feature_weights_path = (root / "reports/metrics/scorer_learned_feature_weights.json").resolve()
    learned_component_weights_path = (root / "reports/metrics/scorer_learned_component_weights.json").resolve()
    learned_anatomical_component_weights_path = (root / "reports/metrics/scorer_learned_anatomical_component_weights.json").resolve()
    learned_transform_family_weights_path = (root / "reports/metrics/scorer_learned_transform_family_weights.json").resolve()
    learned_source_weights_path = (root / "reports/metrics/scorer_learned_source_weights.json").resolve()
    component_weight_summary_path = (root / "reports/metrics/scorer_component_weight_summary.json").resolve()
    component_taxonomy_audit_path = (root / "reports/metrics/component_taxonomy_audit.json").resolve()
    component_balanced_selection_path = (root / "reports/metrics/scorer_component_balanced_feature_selection.json").resolve()
    fairface_reference_comparison_path = (root / "reports/metrics/fairface_reference_comparison.json").resolve()
    feature_correlation_audit_path = (root / "reports/metrics/scorer_feature_correlation_audit.json").resolve()
    reference_selection_diagnostics_path = (root / "reports/metrics/reference_feature_selection_diagnostics.json").resolve()
    reference_resolution_summary_path = (root / "reports/metrics/scoring_reference_resolution_summary.json").resolve()
    source_feature_audit_path = (root / "reports/metrics/source_feature_audit.json").resolve()
    mediapipe_reference_relative_summary_path = (root / "reports/metrics/mediapipe_reference_relative_feature_summary.json").resolve()
    base_feature_dedup_audit_path = (root / "reports/metrics/scorer_base_feature_deduplication_audit.json").resolve()
    pose_leakage_audit_path = (root / "reports/metrics/pose_leakage_audit.json").resolve()
    cv_results_path = (root / "reports/metrics/scorer_cv_results.json").resolve()
    selection_path = (root / "reports/metrics/scorer_model_selection.json").resolve()
    regressor_artifact_path = (root / str(config["regressor_artifact_path"])).resolve()
    classifier_artifact_path = (root / str(config["classifier_artifact_path"])).resolve()
    pairwise_artifact_path = (root / "models/scoring/pairwise_ranker.joblib").resolve()
    ordinal_artifact_path = (root / "models/scoring/ordinal_scorer.joblib").resolve()
    calibrator_artifact_path = (root / "models/scoring/score_calibrator.joblib").resolve()
    pairwise_dataset_path = (root / "data/scoring/scoring_pairs.parquet").resolve()
    config_snapshot_path = (root / str(config["config_snapshot_path"])).resolve()
    ensure_directory(regressor_artifact_path.parent)
    save_json(config, config_snapshot_path)

    frame = pd.read_parquet(features_path)
    reference_config = dict(configs.get("reference", {}).get("reference", {}))
    target_column = str(config["target_column"])
    target_group_column = str(config["target_group_column"])
    split_column = str(config["split_column"])

    quality_mask = frame["quality_pass"].fillna(False).astype(bool) & frame["processing_status"].fillna("").astype(str).str.lower().eq("processed")
    if str(config.get("quality_gate_policy", "")).strip().lower() == "train_only_quality_pass":
        trainable = frame.loc[quality_mask].copy()
    else:
        trainable = frame.loc[frame["processing_status"].fillna("").astype(str).str.lower().eq("processed")].copy()

    candidate_features = geometry_feature_columns()
    if bool(reference_config.get("use_fairface_reference", True)):
        reference_stats = compute_reference_stats(
            project_root=root,
            filters=dict(reference_config.get("filters", {})),
            feature_columns=geometry_feature_columns(),
        )
        frame = append_reference_relative_features(
            frame,
            stats=reference_stats,
            feature_columns=geometry_feature_columns(),
            include_types=list(reference_config.get("feature_types", ["z", "dir", "ratio", "absdev", "percentile"])),
        )
        candidate_features = candidate_features + reference_relative_feature_names(
            geometry_feature_columns(),
            include_types=list(reference_config.get("feature_types", ["z", "dir", "ratio", "absdev", "percentile"])),
        )
    else:
        reference_stats = None
    selected_feature_report = feature_policy_report(frame, config)
    numeric = numeric_feature_frame(trainable, candidate_features)
    missing_fraction = numeric.isna().mean()
    max_missing = float(config.get("max_missing_fraction_per_feature", 0.4))
    selected_feature_columns = [column for column in scoring_feature_columns(config) if column in numeric.columns and float(missing_fraction.get(column, 1.0)) <= max_missing]
    excluded_missing = [
        {"feature_name": column, "exclusion_reason": "missing_rate_too_high"}
        for column in numeric.columns
        if column not in selected_feature_columns
    ]
    feature_sets, feature_set_metadata, duplicate_audit_rows = _build_feature_sets(
        trainable,
        numeric,
        selected_feature_columns,
        max_missing=max_missing,
        target_values=trainable[target_column].astype(float),
        component_map=dict(config.get("components", {})),
        feature_source_policy=dict(config.get("feature_source_policy", {})),
    )
    save_json({"rows": duplicate_audit_rows}, feature_correlation_audit_path)
    trainable = trainable.loc[numeric.index].copy()

    split_values = trainable[split_column].fillna("").astype(str).str.lower()
    train_frame = trainable.loc[split_values.eq("train")].copy()
    val_frame = trainable.loc[split_values.eq("val")].copy()
    test_frame = trainable.loc[split_values.eq("test")].copy()
    y_train = train_frame[target_column].astype(float).to_numpy(dtype=np.float64)
    y_val = val_frame[target_column].astype(float).to_numpy(dtype=np.float64)
    y_test = test_frame[target_column].astype(float).to_numpy(dtype=np.float64)
    y_group_train = train_frame[target_group_column].astype(str).to_numpy()
    y_group_val = val_frame[target_group_column].astype(str).to_numpy()
    y_group_test = test_frame[target_group_column].astype(str).to_numpy()

    requested_group = reference_stats.requested_filters if reference_stats is not None else {}
    resolved_group = reference_stats.filters if reference_stats is not None else {}
    fallback_used = bool(reference_stats.coverage.get("fallback_used")) if reference_stats is not None else False
    reference_sample_count = int(reference_stats.sample_count) if reference_stats is not None else 0
    resolution_rows: list[dict[str, Any]] = []
    for _, row in trainable.iterrows():
        row_requested = row.get("reference_requested_group", requested_group)
        row_resolved = row.get("reference_resolved_group", resolved_group)
        row_fallback = row.get("reference_fallback_used", fallback_used)
        row_ref_count = row.get("reference_sample_count", reference_sample_count)
        resolution_rows.append(
            {
                "image_path": row.get("image_path"),
                "target_score": float(row[target_column]) if pd.notna(row[target_column]) else None,
                "requested_group": row_requested,
                "resolved_group": row_resolved,
                "fallback_used": bool(row_fallback) if pd.notna(row_fallback) else False,
                "reference_sample_count": int(row_ref_count) if pd.notna(row_ref_count) else reference_sample_count,
            }
        )
    save_json(
        {
            "status": "completed",
            "rows": int(len(resolution_rows)),
            "requested_group": requested_group,
            "resolved_group": resolved_group,
            "fallback_used": fallback_used,
            "reference_sample_count": reference_sample_count,
            "rows_with_fallback": int(sum(1 for row in resolution_rows if bool(row["fallback_used"]))),
            "rows_with_exact_reference_group": int(sum(1 for row in resolution_rows if not bool(row["fallback_used"]))),
            "resolution_rows": resolution_rows,
        },
        reference_resolution_summary_path,
    )

    groups = sorted(float(value) for value in trainable[target_column].dropna().astype(float).unique().tolist())
    observed_training_label_range = [float(min(groups)), float(max(groups))] if groups else [None, None]
    score_semantics = dict(config.get("score_semantics", {}))
    semantic_score_range = list(score_semantics.get("score_range", [0.0, 10.0]))
    clip_lower = semantic_score_range[0] if semantic_score_range else config["prediction_clipping"]["min"]
    clip_upper = semantic_score_range[1] if len(semantic_score_range) > 1 else config["prediction_clipping"]["max"]
    strict_config = dict(config.get("strict_scoring", {}))

    sample_weight_series = inverse_frequency_weights(train_frame[target_group_column]) if bool(config.get("sample_weighting", {}).get("enabled", False)) else pd.Series(1.0, index=train_frame.index)
    sample_weights = sample_weight_series.to_numpy(dtype=np.float64)
    strict_high_threshold = float(strict_config.get("high_score_threshold", 8.5))
    strict_low_threshold = float(strict_config.get("low_score_threshold", 3.0))
    sample_weight_profiles: dict[str, np.ndarray] = {
        "base": sample_weights,
        "edge_x2": _edge_weighted_profile(
            target_scores=y_train,
            base_weights=sample_weights,
            low_threshold=strict_low_threshold,
            high_threshold=strict_high_threshold,
            edge_multiplier=2.0,
        ),
        "edge_x3": _edge_weighted_profile(
            target_scores=y_train,
            base_weights=sample_weights,
            low_threshold=strict_low_threshold,
            high_threshold=strict_high_threshold,
            edge_multiplier=3.0,
        ),
    }
    group_weight_map = {str(group): float(weight) for group, weight in train_frame.assign(weight=sample_weight_series).groupby(target_group_column)["weight"].first().to_dict().items()}

    random_seed = int(config["random_seed"])
    cv_config = dict(config.get("cv", {}))
    component_map = dict(config.get("components", {}))
    registry_by_name = _feature_registry_map()
    splitter, split_labels = _cv_splitter(train_frame, target_group_column, cv_config, random_seed)

    regressor_models = _build_regressors(dict(config["regression_models"]), random_seed)
    feature_set_comparison: dict[str, Any] = {"sets": {}, "selected_feature_set": None, "duplicate_audit_path": feature_correlation_audit_path.relative_to(root).as_posix()}
    edge_weighting_comparison: dict[str, Any] = {
        "status": "completed",
        "profiles": {
            "base": {"edge_multiplier": 1.0},
            "edge_x2": {"edge_multiplier": 2.0},
            "edge_x3": {"edge_multiplier": 3.0},
        },
        "high_score_threshold": strict_high_threshold,
        "low_score_threshold": strict_low_threshold,
        "sets": {},
    }
    best_feature_set_name = "baseline_insightface_geometry"
    best_feature_set_score = float("inf")
    best_feature_set_model = None
    for feature_set_name, feature_columns in feature_sets.items():
        if not feature_columns:
            feature_set_comparison["sets"][feature_set_name] = {
                **feature_set_metadata.get(feature_set_name, {}),
                "status": "skipped_no_features",
            }
            continue
        quality_scores = _feature_set_quality_scores(
            feature_columns,
            registry_by_name=registry_by_name,
            component_map=component_map,
        )
        numeric_set = numeric.loc[:, feature_columns]
        medians_set = numeric_set.median(axis=0, numeric_only=True).to_dict()
        numeric_set = numeric_set.fillna(value=medians_set)
        X_train_set = numeric_set.loc[train_frame.index, feature_columns].to_numpy(dtype=np.float64)
        X_val_set = numeric_set.loc[val_frame.index, feature_columns].to_numpy(dtype=np.float64)
        X_test_set = numeric_set.loc[test_frame.index, feature_columns].to_numpy(dtype=np.float64)
        set_splitter, set_split_labels = _cv_splitter(train_frame, target_group_column, cv_config, random_seed)
        best_model_name_for_set = None
        best_model_score_for_set = float("inf")
        best_model_weight_profile_for_set = "base"
        candidate_rows: list[dict[str, Any]] = []
        weight_profiles = {"base": sample_weight_profiles["base"]}
        if feature_set_name == "mediapipe_only_high_low_weighted":
            weight_profiles = sample_weight_profiles
        profile_best_rows: dict[str, dict[str, Any]] = {}
        for model_name, model in regressor_models.items():
            for weight_profile_name, weight_profile_values in weight_profiles.items():
                cv_summary = _cv_regression_summary(
                    model=model,
                    X=X_train_set,
                    y=y_train,
                    groups=groups,
                    fold_labels=set_split_labels,
                    splitter=set_splitter,
                    sample_weights=weight_profile_values,
                    clip_lower=clip_lower,
                    clip_upper=clip_upper,
                )
                fitted = _fit_estimator(model, X_train_set, y_train, sample_weight=weight_profile_values)
                val_pred = _prediction_array(fitted, X_val_set, lower=clip_lower, upper=clip_upper)
                test_pred = _prediction_array(fitted, X_test_set, lower=clip_lower, upper=clip_upper)
                calibrator = _fit_score_calibrator(y_val, val_pred, clip_lower=clip_lower, clip_upper=clip_upper)
                calibrated_val_pred = _apply_calibrator(val_pred, calibrator, clip_lower=clip_lower, clip_upper=clip_upper)
                calibrated_test_pred = _apply_calibrator(test_pred, calibrator, clip_lower=clip_lower, clip_upper=clip_upper)
                raw_val_payload = _eval_payload(y_val, val_pred, groups, strict_config)
                cal_val_payload = _eval_payload(y_val, calibrated_val_pred, groups, strict_config)
                raw_score, raw_breakdown = _regression_selection_score(cv_summary, raw_val_payload, strict_config)
                cal_score, cal_breakdown = _regression_selection_score(cv_summary, cal_val_payload, strict_config)
                use_calibrated = calibrator is not None and cal_score < raw_score
                effective_score = cal_score if use_calibrated else raw_score
                candidate_row = {
                    "model_name": model_name,
                    "sample_weight_profile": weight_profile_name,
                    "selection_score": effective_score,
                    "uses_calibration": use_calibrated,
                    "cv_mean_metrics": cv_summary.get("mean_metrics", {}),
                    "val": cal_val_payload if use_calibrated else raw_val_payload,
                    "test": _eval_payload(y_test, calibrated_test_pred if use_calibrated else test_pred, groups, strict_config),
                    "selection_breakdown": cal_breakdown if use_calibrated else raw_breakdown,
                }
                candidate_rows.append(candidate_row)
                prev = profile_best_rows.get(weight_profile_name)
                if prev is None or float(candidate_row["selection_score"]) < float(prev["selection_score"]):
                    profile_best_rows[weight_profile_name] = candidate_row
                if effective_score < best_model_score_for_set:
                    best_model_score_for_set = effective_score
                    best_model_name_for_set = model_name
                    best_model_weight_profile_for_set = weight_profile_name
        adjusted_set_score = float(
            best_model_score_for_set
            + 0.20 * (1.0 - float(quality_scores.get("component_coverage_score", 0.0)))
            + 0.10 * (1.0 - float(quality_scores.get("interpretability_score", 0.0)))
        )
        feature_set_comparison["sets"][feature_set_name] = {
            **feature_set_metadata.get(feature_set_name, {}),
            "status": "completed",
            "quality_scores": quality_scores,
            "candidates": sorted(candidate_rows, key=lambda item: float(item["selection_score"])),
            "best_model_name": best_model_name_for_set,
            "best_model_sample_weight_profile": best_model_weight_profile_for_set,
            "best_selection_score": best_model_score_for_set,
            "adjusted_selection_score": adjusted_set_score,
        }
        if feature_set_name == "mediapipe_only_high_low_weighted":
            edge_weighting_comparison["sets"][feature_set_name] = {
                profile_name: {
                    "best_model_name": payload.get("model_name"),
                    "uses_calibration": payload.get("uses_calibration"),
                    "selection_score": payload.get("selection_score"),
                    "test": payload.get("test"),
                    "val": payload.get("val"),
                }
                for profile_name, payload in profile_best_rows.items()
            }
        if adjusted_set_score < best_feature_set_score:
            best_feature_set_score = adjusted_set_score
            best_feature_set_name = feature_set_name
            best_feature_set_model = best_model_name_for_set
    feature_set_comparison["selected_feature_set"] = best_feature_set_name
    feature_set_comparison["selected_model_hint"] = best_feature_set_model
    save_json(feature_set_comparison, feature_set_comparison_path)
    guardrails_config = dict(config.get("model_selection_guardrails", {}))
    if bool(guardrails_config.get("enabled", True)):
        guardrail_feature_set_name, guardrail_model_hint, guardrails_report, primary_comparison_report = _select_primary_with_guardrails(
            feature_set_comparison=feature_set_comparison,
            feature_set_metadata=feature_set_metadata,
            guardrails=guardrails_config,
        )
        best_feature_set_name = guardrail_feature_set_name
        best_feature_set_model = guardrail_model_hint
        feature_set_comparison["selected_feature_set"] = best_feature_set_name
        feature_set_comparison["selected_model_hint"] = best_feature_set_model
        save_json(feature_set_comparison, feature_set_comparison_path)
        save_json(guardrails_report, guardrails_selection_path)
        save_json(primary_comparison_report, primary_regression_comparison_path)
    expanded_geometry_comparison = {
        "status": "completed",
        "selected_feature_set": best_feature_set_name,
        "selected_model_hint": best_feature_set_model,
        "sets": {
            name: payload
            for name, payload in feature_set_comparison.get("sets", {}).items()
            if name in {
                "baseline_insightface_geometry",
                "combined_stable_geometry",
                "expanded_mediapipe_geometry_forced",
                "mediapipe_weighted_geometry",
                "combined_weighted_no_duplicate",
            }
        },
        "note": "Expanded MediaPipe scorer experiment is compared separately and does not automatically override primary scorer.",
    }
    save_json(expanded_geometry_comparison, expanded_geometry_comparison_path)
    best_raw_feature_set_name = best_feature_set_name
    best_raw_feature_columns = list(feature_sets.get(best_feature_set_name, selected_feature_columns))
    reference_feature_sets, reference_feature_metadata, reference_selection_diagnostics = _build_reference_feature_sets(
        numeric,
        best_raw_feature_columns,
        trainable[target_column].astype(float).to_numpy(dtype=np.float64),
        max_missing=max_missing,
        component_map=dict(config.get("components", {})),
        feature_source_policy=dict(config.get("feature_source_policy", {})),
    )
    save_json(reference_selection_diagnostics, reference_selection_diagnostics_path)
    fairface_reference_comparison: dict[str, Any] = {
        "reference_source": reference_config.get("reference_source", "fairface"),
        "reference_stats_path": reference_stats.stats_path.relative_to(root).as_posix() if reference_stats is not None else None,
        "requested_filters": reference_stats.requested_filters if reference_stats is not None else {},
        "effective_filters": reference_stats.filters if reference_stats is not None else {},
        "fallback_chain": reference_stats.fallback_chain if reference_stats is not None else [],
        "sets": {},
        "selected_feature_set": None,
        "reference_feature_selection_diagnostics_path": reference_selection_diagnostics_path.relative_to(root).as_posix(),
        "reference_resolution_summary_path": reference_resolution_summary_path.relative_to(root).as_posix(),
    }
    best_reference_feature_set_name = "raw_features_only"
    best_reference_feature_set_score = float("inf")
    best_reference_feature_set_model = None
    for feature_set_name, feature_columns in reference_feature_sets.items():
        if not feature_columns:
            fairface_reference_comparison["sets"][feature_set_name] = {
                **reference_feature_metadata.get(feature_set_name, {}),
                "status": "skipped_no_features",
            }
            continue
        quality_scores = _feature_set_quality_scores(
            feature_columns,
            registry_by_name=registry_by_name,
            component_map=component_map,
        )
        numeric_set = numeric.loc[:, feature_columns]
        medians_set = numeric_set.median(axis=0, numeric_only=True).to_dict()
        numeric_set = numeric_set.fillna(value=medians_set)
        X_train_set = numeric_set.loc[train_frame.index, feature_columns].to_numpy(dtype=np.float64)
        X_val_set = numeric_set.loc[val_frame.index, feature_columns].to_numpy(dtype=np.float64)
        X_test_set = numeric_set.loc[test_frame.index, feature_columns].to_numpy(dtype=np.float64)
        set_splitter, set_split_labels = _cv_splitter(train_frame, target_group_column, cv_config, random_seed)
        best_model_name_for_set = None
        best_model_score_for_set = float("inf")
        best_model_weight_profile_for_set = "base"
        candidate_rows: list[dict[str, Any]] = []
        weight_profiles = {"base": sample_weight_profiles["base"]}
        if feature_set_name == "mediapipe_only_high_low_weighted":
            weight_profiles = sample_weight_profiles
        profile_best_rows: dict[str, dict[str, Any]] = {}
        for model_name, model in regressor_models.items():
            for weight_profile_name, weight_profile_values in weight_profiles.items():
                cv_summary = _cv_regression_summary(
                    model=model,
                    X=X_train_set,
                    y=y_train,
                    groups=groups,
                    fold_labels=set_split_labels,
                    splitter=set_splitter,
                    sample_weights=weight_profile_values,
                    clip_lower=clip_lower,
                    clip_upper=clip_upper,
                )
                fitted = _fit_estimator(model, X_train_set, y_train, sample_weight=weight_profile_values)
                val_pred = _prediction_array(fitted, X_val_set, lower=clip_lower, upper=clip_upper)
                test_pred = _prediction_array(fitted, X_test_set, lower=clip_lower, upper=clip_upper)
                calibrator = _fit_score_calibrator(y_val, val_pred, clip_lower=clip_lower, clip_upper=clip_upper)
                calibrated_val_pred = _apply_calibrator(val_pred, calibrator, clip_lower=clip_lower, clip_upper=clip_upper)
                calibrated_test_pred = _apply_calibrator(test_pred, calibrator, clip_lower=clip_lower, clip_upper=clip_upper)
                raw_val_payload = _eval_payload(y_val, val_pred, groups, strict_config)
                cal_val_payload = _eval_payload(y_val, calibrated_val_pred, groups, strict_config)
                raw_score, raw_breakdown = _regression_selection_score(cv_summary, raw_val_payload, strict_config)
                cal_score, cal_breakdown = _regression_selection_score(cv_summary, cal_val_payload, strict_config)
                use_calibrated = calibrator is not None and cal_score < raw_score
                effective_score = cal_score if use_calibrated else raw_score
                candidate_row = {
                    "model_name": model_name,
                    "sample_weight_profile": weight_profile_name,
                    "selection_score": effective_score,
                    "uses_calibration": use_calibrated,
                    "cv_mean_metrics": cv_summary.get("mean_metrics", {}),
                    "val": cal_val_payload if use_calibrated else raw_val_payload,
                    "test": _eval_payload(y_test, calibrated_test_pred if use_calibrated else test_pred, groups, strict_config),
                    "selection_breakdown": cal_breakdown if use_calibrated else raw_breakdown,
                }
                candidate_rows.append(candidate_row)
                prev = profile_best_rows.get(weight_profile_name)
                if prev is None or float(candidate_row["selection_score"]) < float(prev["selection_score"]):
                    profile_best_rows[weight_profile_name] = candidate_row
                if effective_score < best_model_score_for_set:
                    best_model_score_for_set = effective_score
                    best_model_name_for_set = model_name
                    best_model_weight_profile_for_set = weight_profile_name
        adjusted_set_score = float(
            best_model_score_for_set
            + 0.20 * (1.0 - float(quality_scores.get("component_coverage_score", 0.0)))
            + 0.10 * (1.0 - float(quality_scores.get("interpretability_score", 0.0)))
        )
        fairface_reference_comparison["sets"][feature_set_name] = {
            **reference_feature_metadata.get(feature_set_name, {}),
            "status": "completed",
            "quality_scores": quality_scores,
            "candidates": sorted(candidate_rows, key=lambda item: float(item["selection_score"])),
            "best_model_name": best_model_name_for_set,
            "best_model_sample_weight_profile": best_model_weight_profile_for_set,
            "best_selection_score": best_model_score_for_set,
            "adjusted_selection_score": adjusted_set_score,
        }
        if feature_set_name == "mediapipe_only_high_low_weighted":
            edge_weighting_comparison["sets"][feature_set_name] = {
                profile_name: {
                    "best_model_name": payload.get("model_name"),
                    "uses_calibration": payload.get("uses_calibration"),
                    "selection_score": payload.get("selection_score"),
                    "test": payload.get("test"),
                    "val": payload.get("val"),
                }
                for profile_name, payload in profile_best_rows.items()
            }
        if adjusted_set_score < best_reference_feature_set_score:
            best_reference_feature_set_score = adjusted_set_score
            best_reference_feature_set_name = feature_set_name
            best_reference_feature_set_model = best_model_name_for_set
    fairface_reference_comparison["selected_feature_set"] = best_reference_feature_set_name
    fairface_reference_comparison["selected_model_hint"] = best_reference_feature_set_model
    save_json(fairface_reference_comparison, fairface_reference_comparison_path)
    merged_feature_set_comparison = {
        "sets": {**feature_set_comparison.get("sets", {}), **fairface_reference_comparison.get("sets", {})},
        "selected_feature_set": best_feature_set_name,
        "selected_model_hint": best_feature_set_model,
    }
    merged_feature_set_metadata = {**feature_set_metadata, **reference_feature_metadata}
    guardrails_config = dict(config.get("model_selection_guardrails", {}))
    if bool(guardrails_config.get("enabled", True)):
        merged_selected_set, merged_selected_model, guardrails_report, primary_comparison_report = _select_primary_with_guardrails(
            feature_set_comparison=merged_feature_set_comparison,
            feature_set_metadata=merged_feature_set_metadata,
            guardrails=guardrails_config,
        )
        best_feature_set_name = merged_selected_set
        best_feature_set_model = merged_selected_model
        save_json(guardrails_report, guardrails_selection_path)
        save_json(primary_comparison_report, primary_regression_comparison_path)
        feature_set_comparison["selected_feature_set"] = best_feature_set_name
        feature_set_comparison["selected_model_hint"] = best_feature_set_model
        fairface_reference_comparison["selected_feature_set"] = best_feature_set_name
        fairface_reference_comparison["selected_model_hint"] = best_feature_set_model
        save_json(feature_set_comparison, feature_set_comparison_path)
        save_json(fairface_reference_comparison, fairface_reference_comparison_path)
    weighted_model_comparison = {
        "status": "completed",
        "raw_feature_sets": feature_set_comparison.get("sets", {}),
        "reference_feature_sets": fairface_reference_comparison.get("sets", {}),
        "selected_raw_feature_set": best_raw_feature_set_name,
        "selected_reference_feature_set": best_reference_feature_set_name,
        "selection_policy": "strict_score_with_component_coverage_and_interpretability",
    }
    save_json(weighted_model_comparison, weighted_model_comparison_path)
    mediapipe_only_names = [
        name
        for name in set(feature_set_comparison.get("sets", {}).keys()) | set(fairface_reference_comparison.get("sets", {}).keys())
        if str(name).startswith("mediapipe_only_")
    ]
    mediapipe_only_comparison = {
        "status": "completed",
        "selected_feature_set": best_feature_set_name,
        "selected_model_hint": best_feature_set_model,
        "sets": {
            name: (
                feature_set_comparison.get("sets", {}).get(name)
                if name in feature_set_comparison.get("sets", {})
                else fairface_reference_comparison.get("sets", {}).get(name)
            )
            for name in sorted(mediapipe_only_names)
        },
    }
    save_json(mediapipe_only_comparison, mediapipe_only_comparison_path)
    save_json(edge_weighting_comparison, edge_weighting_comparison_path)
    if best_feature_set_name in reference_feature_sets:
        selected_feature_columns = list(reference_feature_sets.get(best_feature_set_name, best_raw_feature_columns))
    else:
        selected_feature_columns = list(feature_sets.get(best_feature_set_name, best_raw_feature_columns))
    dedup_selected_features, dedup_audit = _enforce_base_feature_dedup(
        feature_columns=selected_feature_columns,
        numeric_frame=numeric,
        target_values=trainable[target_column].astype(float),
        max_variants_per_base_feature=1,
        allow_second_variant_correlation_threshold=0.85,
    )
    selected_feature_columns = dedup_selected_features
    save_json(
        {
            "status": "completed",
            "selected_feature_count_after_dedup": int(len(selected_feature_columns)),
            **dedup_audit,
        },
        base_feature_dedup_audit_path,
    )
    selected_reference_columns = [column for column in selected_feature_columns if column.startswith(REFERENCE_FEATURE_PREFIXES)]
    reference_selection_diagnostics["selected_feature_set"] = best_feature_set_name
    reference_selection_diagnostics["selected_model_hint"] = best_feature_set_model
    reference_selection_diagnostics["selected_reference_feature_count"] = int(len(selected_reference_columns))
    reference_selection_diagnostics["selected_reference_feature_kinds"] = {
        "ref_z": int(sum(1 for column in selected_reference_columns if column.startswith("ref_z_"))),
        "ref_ratio": int(sum(1 for column in selected_reference_columns if column.startswith("ref_ratio_"))),
        "ref_dir": int(sum(1 for column in selected_reference_columns if column.startswith("ref_dir_"))),
        "ref_absdev": int(sum(1 for column in selected_reference_columns if column.startswith("ref_absdev_"))),
        "ref_percentile": int(sum(1 for column in selected_reference_columns if column.startswith("ref_percentile_"))),
    }
    save_json(reference_selection_diagnostics, reference_selection_diagnostics_path)
    selected_feature_report["selected_feature_count"] = len(selected_feature_columns)
    selected_feature_report["selected_features"] = selected_feature_columns
    selected_feature_report["excluded_features"].extend(excluded_missing)
    selected_feature_report["feature_set_name"] = best_feature_set_name
    selected_feature_report["mediapipe_selected_count"] = int(sum(1 for column in selected_feature_columns if column.startswith("mediapipe_")))
    selected_feature_report["insightface_selected_count"] = int(sum(1 for column in selected_feature_columns if str(_feature_registry_map().get(column, {}).get("source")) == "insightface_106"))
    selected_feature_report["fused_selected_count"] = int(sum(1 for column in selected_feature_columns if str(_feature_registry_map().get(column, {}).get("source")) == "fused"))
    selected_feature_report["selected_feature_source_counts"] = _feature_source_counts(selected_feature_columns, _feature_registry_map())
    selected_feature_report["reference_relative_selected_count"] = int(sum(1 for column in selected_feature_columns if column.startswith(REFERENCE_FEATURE_PREFIXES)))
    feature_lineage_rows = [
        _feature_lineage_row(
            column,
            registry_by_name=registry_by_name,
            component_map=component_map,
        )
        for column in selected_feature_columns
    ]
    direct_mediapipe_selected_count = int(
        sum(1 for row in feature_lineage_rows if row["source"] == "mediapipe_face_mesh")
    )
    mediapipe_derived_reference_selected_count = int(
        sum(1 for row in feature_lineage_rows if bool(row["is_reference_relative"]) and bool(row["mediapipe_derived"]))
    )
    selected_feature_report["direct_mediapipe_selected_count"] = direct_mediapipe_selected_count
    selected_feature_report["mediapipe_derived_reference_selected_count"] = mediapipe_derived_reference_selected_count
    selected_feature_report["total_mediapipe_influence_count"] = int(
        direct_mediapipe_selected_count + mediapipe_derived_reference_selected_count
    )
    selected_feature_report["reference_relative_selected_kinds"] = {
        "ref_z": int(sum(1 for column in selected_feature_columns if column.startswith("ref_z_"))),
        "ref_ratio": int(sum(1 for column in selected_feature_columns if column.startswith("ref_ratio_"))),
        "ref_dir": int(sum(1 for column in selected_feature_columns if column.startswith("ref_dir_"))),
        "ref_absdev": int(sum(1 for column in selected_feature_columns if column.startswith("ref_absdev_"))),
        "ref_percentile": int(sum(1 for column in selected_feature_columns if column.startswith("ref_percentile_"))),
    }
    selected_feature_report["contains_raw_px"] = any(column.startswith("raw_") or column.endswith("_px") for column in selected_feature_columns)
    selected_feature_report["contains_metadata"] = any(
        column in frame.columns and column not in geometry_feature_columns() and not column.startswith(REFERENCE_FEATURE_PREFIXES)
        for column in selected_feature_columns
    )
    selected_feature_report["contains_target_leakage"] = any(column in {target_column, target_group_column} for column in selected_feature_columns)
    selected_feature_report["component_counts"] = _component_counts(
        selected_feature_columns,
        registry_by_name=registry_by_name,
        component_map=component_map,
    )
    selected_feature_report["feature_lineage"] = feature_lineage_rows
    anatomical_component_counts: dict[str, int] = {}
    transform_family_counts: dict[str, int] = {}
    source_family_counts: dict[str, int] = {}
    component_taxonomy_warnings: list[str] = []
    for row in feature_lineage_rows:
        anatomical = _canonical_anatomical_component(str(row.get("anatomical_component") or "unknown"))
        row["anatomical_component"] = anatomical
        row["component"] = anatomical
        transform_family = str(row.get("transform_family") or _transform_family(str(row.get("feature_name") or "")))
        row["transform_family"] = transform_family
        source_family = (
            "mediapipe_reference_relative"
            if bool(row.get("is_reference_relative")) and bool(row.get("mediapipe_derived"))
            else ("mediapipe_direct" if str(row.get("original_source")) == "mediapipe_face_mesh" else str(row.get("original_source") or "other"))
        )
        row["source_family"] = source_family
        if anatomical == "reference_deviation":
            component_taxonomy_warnings.append(f"invalid_anatomical_component_reference_deviation:{row.get('feature_name')}")
        if anatomical == "unknown":
            component_taxonomy_warnings.append(f"unknown_component:{row.get('feature_name')}")
        anatomical_component_counts[anatomical] = anatomical_component_counts.get(anatomical, 0) + 1
        transform_family_counts[transform_family] = transform_family_counts.get(transform_family, 0) + 1
        source_family_counts[source_family] = source_family_counts.get(source_family, 0) + 1
    selected_feature_report["anatomical_component_counts"] = dict(sorted(anatomical_component_counts.items(), key=lambda item: item[0]))
    selected_feature_report["transform_family_counts"] = dict(sorted(transform_family_counts.items(), key=lambda item: item[0]))
    selected_feature_report["source_family_counts"] = dict(sorted(source_family_counts.items(), key=lambda item: item[0]))
    selected_feature_report["source_counts"] = dict(selected_feature_report["source_family_counts"])
    policy_mode = str(config.get("feature_source_policy", {}).get("mode", "mixed")).strip().lower()
    lineage_violations = []
    if policy_mode == "mediapipe_only":
        lineage_violations = [
            row
            for row in feature_lineage_rows
            if not _is_mediapipe_primary_feature(str(row.get("feature_name", "")), registry_by_name=registry_by_name)
        ]
    source_audit_payload = {
        "status": "completed",
        "active_feature_source_policy": {
            "mode": policy_mode or "mixed",
            "insightface_used_for_primary": False if policy_mode == "mediapipe_only" else bool(selected_feature_report["insightface_selected_count"]),
            "mediapipe_used_for_primary": True if policy_mode == "mediapipe_only" else bool(selected_feature_report["total_mediapipe_influence_count"]),
        },
        "selected_feature_count": int(len(selected_feature_columns)),
        "direct_mediapipe_selected_count": int(direct_mediapipe_selected_count),
        "mediapipe_derived_reference_selected_count": int(mediapipe_derived_reference_selected_count),
        "total_mediapipe_influence_count": int(direct_mediapipe_selected_count + mediapipe_derived_reference_selected_count),
        "insightface_selected_count": int(selected_feature_report["insightface_selected_count"]),
        "fused_selected_count": int(selected_feature_report["fused_selected_count"]),
        "legacy_selected_count": int(
            sum(
                1
                for row in feature_lineage_rows
                if str(row.get("source")) not in {"mediapipe_face_mesh", "reference_relative"}
            )
        ),
        "reference_relative_selected_count": int(selected_feature_report["reference_relative_selected_count"]),
        "violations_count": int(len(lineage_violations)),
        "violations": lineage_violations,
        "selected_features": feature_lineage_rows,
    }
    save_json(source_audit_payload, source_feature_audit_path)
    mediapipe_ref_summary = {
        "status": "completed",
        "total_mediapipe_base_features": int(sum(1 for name in geometry_feature_columns() if name.startswith("mediapipe_"))),
        "total_mediapipe_ref_features_available": int(
            sum(
                1
                for name in frame.columns
                if str(name).startswith(REFERENCE_FEATURE_PREFIXES) and _base_feature_name(str(name)).startswith("mediapipe_")
            )
        ),
        "selected_mediapipe_ref_features": int(mediapipe_derived_reference_selected_count),
        "excluded_legacy_ref_features": [
            row for row in lineage_violations if bool(row.get("is_reference_relative"))
        ],
    }
    save_json(mediapipe_ref_summary, mediapipe_reference_relative_summary_path)
    component_taxonomy_payload = {
        "status": "completed",
        "selected_feature_count": int(len(feature_lineage_rows)),
        "anatomical_component_counts": selected_feature_report.get("anatomical_component_counts", {}),
        "transform_family_counts": selected_feature_report.get("transform_family_counts", {}),
        "source_family_counts": selected_feature_report.get("source_family_counts", {}),
        "reference_deviation_in_anatomical_component_count": int(
            sum(1 for row in feature_lineage_rows if str(row.get("anatomical_component")) == "reference_deviation")
        ),
        "unknown_component_count": int(
            sum(1 for row in feature_lineage_rows if str(row.get("anatomical_component")) == "unknown")
        ),
        "all_ref_features_have_anatomical_component": bool(
            all(
                str(row.get("anatomical_component")) not in {"unknown", "reference_deviation"}
                for row in feature_lineage_rows
                if bool(row.get("is_reference_relative"))
            )
        ),
        "warnings": component_taxonomy_warnings,
    }
    save_json(component_taxonomy_payload, component_taxonomy_audit_path)
    selected_feature_report["feature_weight_method"] = None
    selected_feature_report["primary_feature_set"] = best_feature_set_name
    selected_feature_report["primary_model"] = best_feature_set_model
    save_json(
        {
            "status": "completed",
            "raw_feature_sets": feature_set_comparison.get("sets", {}),
            "reference_feature_sets": fairface_reference_comparison.get("sets", {}),
            "reference_component_selection": reference_selection_diagnostics.get("component_balanced_reference_selection", {}),
            "selected_feature_set": best_feature_set_name,
            "selected_component_counts": selected_feature_report["component_counts"],
        },
        component_balanced_selection_path,
    )
    save_json(selected_feature_report, selected_features_path)
    numeric = numeric.loc[:, selected_feature_columns]
    medians = numeric.median(axis=0, numeric_only=True).to_dict()
    numeric = numeric.fillna(value=medians)

    X_train = numeric.loc[train_frame.index, selected_feature_columns].to_numpy(dtype=np.float64)
    X_val = numeric.loc[val_frame.index, selected_feature_columns].to_numpy(dtype=np.float64)
    X_test = numeric.loc[test_frame.index, selected_feature_columns].to_numpy(dtype=np.float64)
    regression_results: dict[str, Any] = {}
    regression_calibrators: dict[str, IsotonicRegression | None] = {}
    cv_results: dict[str, Any] = {"regression_models": {}, "classification_models": {}}
    best_regressor_name: str | None = None
    best_regressor_model: Any | None = None
    best_regressor_score = float("inf")
    primary_metric = str(cv_config.get("selection_metric", config.get("selection_metric", "mae")))
    secondary_metric = str(cv_config.get("secondary_metric", "spearman"))

    for model_name, model in regressor_models.items():
        cv_summary = _cv_regression_summary(
            model=model,
            X=X_train,
            y=y_train,
            groups=groups,
            fold_labels=split_labels,
            splitter=splitter,
            sample_weights=sample_weights,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )
        fitted = _fit_estimator(model, X_train, y_train, sample_weight=sample_weights)
        train_pred = _prediction_array(fitted, X_train, lower=clip_lower, upper=clip_upper)
        val_pred = _prediction_array(fitted, X_val, lower=clip_lower, upper=clip_upper)
        test_pred = _prediction_array(fitted, X_test, lower=clip_lower, upper=clip_upper)
        calibrator = _fit_score_calibrator(y_val, val_pred, clip_lower=clip_lower, clip_upper=clip_upper)
        regression_calibrators[model_name] = calibrator
        calibrated_train_pred = _apply_calibrator(train_pred, calibrator, clip_lower=clip_lower, clip_upper=clip_upper)
        calibrated_val_pred = _apply_calibrator(val_pred, calibrator, clip_lower=clip_lower, clip_upper=clip_upper)
        calibrated_test_pred = _apply_calibrator(test_pred, calibrator, clip_lower=clip_lower, clip_upper=clip_upper)
        raw_val_payload = _eval_payload(y_val, val_pred, groups, strict_config)
        calibrated_val_payload = _eval_payload(y_val, calibrated_val_pred, groups, strict_config)
        raw_selection_score, raw_selection_breakdown = _regression_selection_score(cv_summary, raw_val_payload, strict_config)
        calibrated_selection_score, calibrated_selection_breakdown = _regression_selection_score(cv_summary, calibrated_val_payload, strict_config)
        use_calibrated_as_primary = calibrator is not None and calibrated_selection_score < raw_selection_score
        regression_results[model_name] = {
            "metrics": {
                "train": _eval_payload(y_train, train_pred, groups, strict_config),
                "val": raw_val_payload,
                "test": _eval_payload(y_test, test_pred, groups, strict_config),
            },
            "calibrated_metrics": {
                "train": _eval_payload(y_train, calibrated_train_pred, groups, strict_config),
                "val": calibrated_val_payload,
                "test": _eval_payload(y_test, calibrated_test_pred, groups, strict_config),
            },
            "cv": cv_summary,
            "calibration": {
                "method": "isotonic_regression" if calibrator is not None else None,
                "fitted_on_split": "val" if calibrator is not None else None,
                "raw_selection_score": raw_selection_score,
                "raw_selection_breakdown": raw_selection_breakdown,
                "calibrated_selection_score": calibrated_selection_score,
                "calibrated_selection_breakdown": calibrated_selection_breakdown,
                "use_calibrated_as_primary": use_calibrated_as_primary,
                "selection_score": calibrated_selection_score if use_calibrated_as_primary else raw_selection_score,
                "selection_breakdown": calibrated_selection_breakdown if use_calibrated_as_primary else raw_selection_breakdown,
            },
        }
        cv_results["regression_models"][model_name] = cv_summary
        effective_selection_score = calibrated_selection_score if use_calibrated_as_primary else raw_selection_score
        if effective_selection_score < best_regressor_score:
            best_regressor_score = effective_selection_score
            best_regressor_name = model_name
            best_regressor_model = fitted

    # Guardrails for primary regressor selection on the chosen feature set.
    guardrails_config = dict(config.get("model_selection_guardrails", {}))
    if bool(guardrails_config.get("enabled", True)) and regression_results:
        min_test_mae = guardrails_config.get("min_test_mae_improvement_against_anchor")
        min_test_r2 = guardrails_config.get("min_test_r2")
        min_test_spearman = guardrails_config.get("min_test_spearman")
        max_high_bias_abs = guardrails_config.get("max_abs_high_group_bias")
        max_low_bias_abs = guardrails_config.get("max_abs_low_group_bias")
        hard_penalty_degraded = float(guardrails_config.get("hard_penalty_metric_degradation", 10.0))
        preferred_model_hint = str(best_feature_set_model) if best_feature_set_model else None

        reg_rows: list[dict[str, Any]] = []
        anchor_model = preferred_model_hint if preferred_model_hint in regression_results else best_regressor_name
        anchor_payload = regression_results.get(anchor_model or "", {})
        anchor_test = dict((anchor_payload.get("metrics") or {}).get("test") or {})
        anchor_test_metrics = dict(anchor_test.get("metrics") or {})
        anchor_test_calib = dict(anchor_test.get("calibration_summary") or {})
        anchor_mae = float(anchor_test_metrics.get("mae")) if anchor_test_metrics.get("mae") is not None else None
        anchor_r2 = float(anchor_test_metrics.get("r2")) if anchor_test_metrics.get("r2") is not None else None
        anchor_sp = float(anchor_test_metrics.get("spearman")) if anchor_test_metrics.get("spearman") is not None else None
        anchor_high_bias_abs = abs(float(anchor_test_calib.get("high_group_bias"))) if anchor_test_calib.get("high_group_bias") is not None else None
        anchor_low_bias_abs = abs(float(anchor_test_calib.get("low_group_bias"))) if anchor_test_calib.get("low_group_bias") is not None else None

        for model_name, payload in regression_results.items():
            base_score = float((payload.get("calibration") or {}).get("selection_score", float("inf")))
            test_payload = dict((payload.get("metrics") or {}).get("test") or {})
            test_metrics = dict(test_payload.get("metrics") or {})
            test_calib = dict(test_payload.get("calibration_summary") or {})
            test_mae = float(test_metrics.get("mae")) if test_metrics.get("mae") is not None else None
            test_r2 = float(test_metrics.get("r2")) if test_metrics.get("r2") is not None else None
            test_sp = float(test_metrics.get("spearman")) if test_metrics.get("spearman") is not None else None
            high_bias_abs = abs(float(test_calib.get("high_group_bias"))) if test_calib.get("high_group_bias") is not None else None
            low_bias_abs = abs(float(test_calib.get("low_group_bias"))) if test_calib.get("low_group_bias") is not None else None
            penalties: list[str] = []
            score = base_score
            if min_test_mae is not None and (test_mae is None or test_mae > float(min_test_mae)):
                score += hard_penalty_degraded
                penalties.append("test_mae_above_floor")
            if min_test_r2 is not None and (test_r2 is None or test_r2 < float(min_test_r2)):
                score += hard_penalty_degraded
                penalties.append("test_r2_below_floor")
            if min_test_spearman is not None and (test_sp is None or test_sp < float(min_test_spearman)):
                score += hard_penalty_degraded
                penalties.append("test_spearman_below_floor")
            if max_high_bias_abs is not None and (high_bias_abs is None or high_bias_abs > float(max_high_bias_abs)):
                score += hard_penalty_degraded
                penalties.append("high_group_bias_too_high")
            if max_low_bias_abs is not None and (low_bias_abs is None or low_bias_abs > float(max_low_bias_abs)):
                score += hard_penalty_degraded
                penalties.append("low_group_bias_too_high")
            if anchor_mae is not None and test_mae is not None and test_mae > anchor_mae:
                score += 0.5 * hard_penalty_degraded
                penalties.append("worse_than_anchor_mae")
            if anchor_r2 is not None and test_r2 is not None and test_r2 < anchor_r2:
                score += 0.5 * hard_penalty_degraded
                penalties.append("worse_than_anchor_r2")
            if anchor_sp is not None and test_sp is not None and test_sp < anchor_sp:
                score += 0.5 * hard_penalty_degraded
                penalties.append("worse_than_anchor_spearman")
            if anchor_high_bias_abs is not None and high_bias_abs is not None and high_bias_abs > anchor_high_bias_abs:
                score += 0.5 * hard_penalty_degraded
                penalties.append("worse_than_anchor_high_bias")
            if anchor_low_bias_abs is not None and low_bias_abs is not None and low_bias_abs > anchor_low_bias_abs:
                score += 0.5 * hard_penalty_degraded
                penalties.append("worse_than_anchor_low_bias")
            reg_rows.append(
                {
                    "model_name": model_name,
                    "base_selection_score": base_score,
                    "guardrail_adjusted_score": score,
                    "penalties": penalties,
                    "test_metrics": test_metrics,
                    "test_calibration_summary": test_calib,
                }
            )
        reg_rows = sorted(reg_rows, key=lambda item: float(item.get("guardrail_adjusted_score", float("inf"))))
        if reg_rows:
            chosen_name = str(reg_rows[0]["model_name"])
            if chosen_name in regressor_models:
                best_regressor_name = chosen_name
                best_regressor_score = float(reg_rows[0].get("guardrail_adjusted_score", best_regressor_score))
                best_regressor_model = _fit_estimator(
                    regressor_models[chosen_name],
                    X_train,
                    y_train,
                    sample_weight=sample_weights,
                )

    pairwise_config = dict(config.get("pairwise", {}))
    pairwise_train_frame = pd.concat(
        [
            train_frame.loc[:, ["image_id", target_column, target_group_column]],
            numeric.loc[train_frame.index, selected_feature_columns],
        ],
        axis=1,
    )
    pairwise_frame, pairwise_dataset_summary = _build_pairwise_rows(
        pairwise_train_frame,
        selected_feature_columns,
        max_pairs_per_anchor=int(pairwise_config.get("max_pairs_per_anchor", 24)),
        close_distance_threshold=float(pairwise_config.get("close_distance_threshold", 1.0)),
        close_distance_pair_multiplier=float(pairwise_config.get("close_distance_pair_multiplier", 2.0)),
        large_distance_threshold=float(pairwise_config.get("large_distance_threshold", 3.0)),
        large_distance_pair_multiplier=float(pairwise_config.get("large_distance_pair_multiplier", 1.75)),
        high_vs_mid_pair_multiplier=float(pairwise_config.get("high_vs_mid_pair_multiplier", 1.75)),
        low_vs_mid_pair_multiplier=float(pairwise_config.get("low_vs_mid_pair_multiplier", 1.75)),
        elite_vs_average_pair_multiplier=float(pairwise_config.get("elite_vs_average_pair_multiplier", 2.25)),
        low_score_threshold=float(strict_config.get("low_score_threshold", 3.0)),
        high_score_threshold=float(strict_config.get("high_score_threshold", 8.5)),
        elite_score_threshold=float(strict_config.get("elite_score_threshold", 9.0)),
        average_zone_center=float(config.get("score_semantics", {}).get("average_zone_center", 5.0)),
        random_seed=random_seed,
    )
    if not pairwise_frame.empty:
        pairwise_frame.to_parquet(pairwise_dataset_path, index=False)
    save_json(pairwise_dataset_summary, pairwise_dataset_summary_path)
    pairwise_models = _build_pairwise_models(random_seed)
    pairwise_results: dict[str, Any] = {}
    best_pairwise_name: str | None = None
    best_pairwise_model: Any | None = None
    best_pairwise_score = float("-inf")
    pairwise_used_for_absolute_scoring = False
    pairwise_reference_frame = _pairwise_reference_frame(
        train_frame,
        numeric,
        selected_feature_columns,
        target_column=target_column,
        target_group_column=target_group_column,
        per_group=int(pairwise_config.get("reference_examples_per_group", 8)),
    )
    if not pairwise_frame.empty:
        X_pair = pairwise_frame.loc[:, selected_feature_columns].to_numpy(dtype=np.float64)
        y_pair = pairwise_frame["pair_label"].astype(int).to_numpy(dtype=np.int64)
        weights_pair = inverse_frequency_weights(pairwise_frame["left_group"]).to_numpy(dtype=np.float64)
        for model_name, model in pairwise_models.items():
            fitted = _fit_estimator(model, X_pair, y_pair, sample_weight=weights_pair)
            metrics = _pairwise_summary(fitted, X_pair, y_pair)
            approx_selection = _pairwise_predict_scores(
                fitted,
                selected_feature_columns,
                pairwise_reference_frame,
                numeric.loc[train_frame.index, selected_feature_columns],
                groups,
            )
            approx_val = _pairwise_predict_scores(
                fitted,
                selected_feature_columns,
                pairwise_reference_frame,
                numeric.loc[val_frame.index, selected_feature_columns],
                groups,
            )
            approx_test = _pairwise_predict_scores(
                fitted,
                selected_feature_columns,
                pairwise_reference_frame,
                numeric.loc[test_frame.index, selected_feature_columns],
                groups,
            )
            pairwise_calibrator = _fit_score_calibrator(y_val, approx_val, clip_lower=clip_lower, clip_upper=clip_upper)
            approx_train_cal = _apply_calibrator(approx_selection, pairwise_calibrator, clip_lower=clip_lower, clip_upper=clip_upper)
            approx_val_cal = _apply_calibrator(approx_val, pairwise_calibrator, clip_lower=clip_lower, clip_upper=clip_upper)
            approx_test_cal = _apply_calibrator(approx_test, pairwise_calibrator, clip_lower=clip_lower, clip_upper=clip_upper)
            metrics["train"] = _eval_payload(y_train, approx_selection, groups, strict_config)
            metrics["val"] = _eval_payload(y_val, approx_val, groups, strict_config)
            metrics["test"] = _eval_payload(y_test, approx_test, groups, strict_config)
            metrics["calibrated"] = {
                "train": _eval_payload(y_train, approx_train_cal, groups, strict_config),
                "val": _eval_payload(y_val, approx_val_cal, groups, strict_config),
                "test": _eval_payload(y_test, approx_test_cal, groups, strict_config),
            }
            metrics["spearman_vs_target_score"] = metrics["val"]["metrics"].get("spearman")
            regression_val_mae = regression_results.get(best_regressor_name or "", {}).get("metrics", {}).get("val", {}).get("metrics", {}).get("mae")
            pairwise_val_mae = metrics["calibrated"]["val"]["metrics"].get("mae")
            use_for_absolute_scoring = (
                pairwise_val_mae is not None
                and regression_val_mae is not None
                and float(pairwise_val_mae) <= float(regression_val_mae) * 1.35
                and abs(float(metrics["calibrated"]["val"]["calibration_summary"].get("high_group_bias") or 0.0)) <= 0.75
                and abs(float(metrics["calibrated"]["val"]["calibration_summary"].get("low_group_bias") or 0.0)) <= 0.75
                and 0.8 <= float(metrics["calibrated"]["val"]["calibration_summary"].get("calibration_slope") or 0.0) <= 1.2
            )
            metrics["use_for_absolute_scoring"] = bool(use_for_absolute_scoring)
            pairwise_results[model_name] = metrics
            score = metrics["val"]["metrics"].get("spearman")
            if score is not None and float(score) > best_pairwise_score:
                best_pairwise_score = float(score)
                best_pairwise_name = model_name
                best_pairwise_model = fitted
                pairwise_used_for_absolute_scoring = bool(use_for_absolute_scoring)
    save_json(
        {
            "status": "completed" if best_pairwise_model is not None else "skipped_no_data",
            "best_pairwise_model": best_pairwise_name,
            "pairwise_dataset_path": pairwise_dataset_path.relative_to(root).as_posix() if pairwise_dataset_path.exists() else pairwise_dataset_path.relative_to(root).as_posix(),
            "metrics": pairwise_results,
        },
        pairwise_summary_path,
    )

    ordinal_config = dict(config.get("ordinal", {}))
    ordinal_payload = _build_ordinal_models(
        X_train,
        y_train,
        groups,
        random_seed=random_seed,
        min_positive_threshold_samples=int(ordinal_config.get("min_positive_threshold_samples", 4)),
        sample_weights=sample_weights,
    )
    ordinal_train_pred, _ordinal_train_probs = _ordinal_predict(ordinal_payload, X_train)
    ordinal_val_pred, _ordinal_val_probs = _ordinal_predict(ordinal_payload, X_val)
    ordinal_test_pred, _ordinal_test_probs = _ordinal_predict(ordinal_payload, X_test)
    ordinal_metrics = {
        "train": _eval_payload(y_train, ordinal_train_pred, groups, strict_config),
        "val": _eval_payload(y_val, ordinal_val_pred, groups, strict_config),
        "test": _eval_payload(y_test, ordinal_test_pred, groups, strict_config),
    }
    save_json(
        {
            "status": "completed" if ordinal_payload.get("thresholds") else "skipped_no_data",
            "threshold_count": int(len(ordinal_payload.get("thresholds", []))),
            "metrics": ordinal_metrics,
        },
        ordinal_summary_path,
    )

    classifier_models = _build_classifiers(dict(config["classification_models"]), random_seed)
    classification_results: dict[str, Any] = {}
    best_classifier_name: str | None = None
    best_classifier_model: Any | None = None
    best_classifier_score = float("-inf")
    class_labels = sorted(trainable[target_group_column].dropna().astype(str).unique().tolist())
    for model_name, model in classifier_models.items():
        cv_summary = _cv_classification_summary(
            model=model,
            X=X_train,
            y=y_group_train,
            fold_labels=split_labels,
            splitter=splitter,
            sample_weights=sample_weights,
            classes=class_labels,
        )
        fitted = _fit_estimator(model, X_train, y_group_train, sample_weight=sample_weights)
        train_pred = fitted.predict(X_train) if len(X_train) else np.asarray([], dtype=object)
        train_prob = fitted.predict_proba(X_train) if len(X_train) and hasattr(fitted, "predict_proba") else None
        val_pred = fitted.predict(X_val) if len(X_val) else np.asarray([], dtype=object)
        val_prob = fitted.predict_proba(X_val) if len(X_val) and hasattr(fitted, "predict_proba") else None
        test_pred = fitted.predict(X_test) if len(X_test) else np.asarray([], dtype=object)
        test_prob = fitted.predict_proba(X_test) if len(X_test) and hasattr(fitted, "predict_proba") else None
        classification_results[model_name] = {
            "metrics": {
                "train": _classifier_metrics(y_group_train, train_pred, train_prob, class_labels),
                "val": _classifier_metrics(y_group_val, val_pred, val_prob, class_labels),
                "test": _classifier_metrics(y_group_test, test_pred, test_prob, class_labels),
            },
            "cv": cv_summary,
        }
        cv_results["classification_models"][model_name] = cv_summary
        balanced = cv_summary["mean_metrics"].get("balanced_accuracy")
        if balanced is not None and float(balanced) > best_classifier_score:
            best_classifier_score = float(balanced)
            best_classifier_name = model_name
            best_classifier_model = fitted

    group_counts = train_frame[target_group_column].astype(str).value_counts().to_dict()
    unstable_groups = {str(group): int(count) for group, count in group_counts.items() if int(count) < int(config["minimum_samples_per_group_warning"])}

    feature_importance_rows: list[dict[str, Any]] = []
    if best_regressor_model is not None:
        feature_importance_rows = _model_feature_importance_rows(best_regressor_model, selected_feature_columns)
        try:
            if len(X_test) and len(selected_feature_columns):
                importance = permutation_importance(best_regressor_model, X_test, y_test, n_repeats=10, random_state=random_seed, scoring="neg_mean_absolute_error")
                importance_map = {feature_name: {"permutation_importance_mean": float(mean), "permutation_importance_std": float(std)} for feature_name, mean, std in zip(selected_feature_columns, importance.importances_mean.tolist(), importance.importances_std.tolist())}
                for row in feature_importance_rows:
                    row.update(importance_map.get(row["feature_name"], {}))
        except Exception:
            pass

    correlation_rows = _correlation_rows(train_frame, selected_feature_columns, target_column)
    feature_effect_rows = _standardized_effect_rows(best_regressor_model, selected_feature_columns, correlation_rows) if best_regressor_model is not None else []
    component_rows = global_component_rows(feature_effect_rows, dict(config.get("components", {})), value_key="signed_effect")
    learned_feature_weight_rows, learned_weight_method = _learned_feature_weight_rows(
        feature_effect_rows=feature_effect_rows,
        feature_importance_rows=feature_importance_rows,
        registry_by_name=registry_by_name,
        component_map=component_map,
    )
    learned_component_weight_rows, learned_component_weight_map = _learned_component_weight_rows(learned_feature_weight_rows)
    learned_anatomical_component_weight_rows = learned_component_weight_rows
    learned_anatomical_component_weight_map = learned_component_weight_map
    learned_transform_family_weight_rows, learned_transform_family_weight_map = _learned_group_weight_rows(
        learned_feature_weight_rows,
        group_key="transform_family",
        group_label="transform_family",
    )
    learned_source_weight_rows, learned_source_weight_map = _learned_group_weight_rows(
        learned_feature_weight_rows,
        group_key="source_family",
        group_label="source_family",
    )
    pose_weight_cap = float(config.get("pose_features", {}).get("max_component_weight", 0.15))
    angles_pose_weight = float(learned_component_weight_map.get("angles_pose", 0.0))
    pose_weight_cap_exceeded = bool(angles_pose_weight > pose_weight_cap)
    save_json({"rows": feature_importance_rows}, feature_importance_path)
    save_json({"rows": correlation_rows}, feature_correlations_path)
    save_json({"rows": feature_importance_rows, "warning": "Feature importance reflects the learned scoring model and is not causal truth."}, global_feature_importance_path)
    save_json({"rows": feature_effect_rows, "warning": "Signed effects describe learned associations in the training dataset and are not objective truth."}, feature_effects_path)
    save_json({"rows": component_rows, "warning": "Component contributions aggregate learned feature effects and are not objective part scores."}, component_importance_path)
    save_json(
        {
            "status": "completed",
            "weight_method": learned_weight_method,
            "rows": learned_feature_weight_rows,
            "warning": "Learned feature weights are dataset-dependent associations, not causal truth.",
        },
        learned_feature_weights_path,
    )
    save_json(
        {
            "status": "completed",
            "weight_method": learned_weight_method,
            "rows": learned_anatomical_component_weight_rows,
            "warning": "Component weights are learned from the current beauty-labeled dataset and are not universal truth.",
        },
        learned_component_weights_path,
    )
    save_json(
        {
            "status": "completed",
            "weight_method": learned_weight_method,
            "rows": learned_anatomical_component_weight_rows,
            "warning": "Anatomical component weights are learned from the current beauty-labeled dataset and are not universal truth.",
        },
        learned_anatomical_component_weights_path,
    )
    save_json(
        {
            "status": "completed",
            "weight_method": learned_weight_method,
            "rows": learned_transform_family_weight_rows,
            "normalized_transform_family_weights": learned_transform_family_weight_map,
            "warning": "Transform family weights show representation style share (absolute vs reference-relative), not anatomy.",
        },
        learned_transform_family_weights_path,
    )
    save_json(
        {
            "status": "completed",
            "weight_method": learned_weight_method,
            "rows": learned_source_weight_rows,
            "normalized_source_weights": learned_source_weight_map,
            "warning": "Source weights show direct vs derived feature share, not anatomy.",
        },
        learned_source_weights_path,
    )
    save_json(
        {
            "status": "completed",
            "weight_method": learned_weight_method,
            "top_weighted_components": learned_anatomical_component_weight_rows[:10],
            "normalized_component_weights": learned_anatomical_component_weight_map,
            "normalized_sum": float(sum(learned_anatomical_component_weight_map.values())),
            "anatomical_component_weights": learned_anatomical_component_weight_map,
            "transform_family_weights": learned_transform_family_weight_map,
            "source_weights": learned_source_weight_map,
            "angles_pose_weight": angles_pose_weight,
            "angles_pose_weight_cap": pose_weight_cap,
            "angles_pose_weight_cap_exceeded": pose_weight_cap_exceeded,
        },
        component_weight_summary_path,
    )
    selected_feature_report["feature_weight_method"] = learned_weight_method
    selected_feature_report["component_weights"] = learned_anatomical_component_weight_map
    selected_feature_report["anatomical_component_weights"] = learned_anatomical_component_weight_map
    selected_feature_report["transform_family_weights"] = learned_transform_family_weight_map
    selected_feature_report["source_weights"] = learned_source_weight_map
    save_json(selected_feature_report, selected_features_path)
    taxonomy_payload = json.loads(component_taxonomy_audit_path.read_text(encoding="utf-8")) if component_taxonomy_audit_path.exists() else {}
    taxonomy_payload.update(
        {
            "anatomical_component_weights": learned_anatomical_component_weight_map,
            "transform_family_weights": learned_transform_family_weight_map,
            "source_weights": learned_source_weight_map,
            "reference_deviation_in_anatomical_weights": bool("reference_deviation" in learned_anatomical_component_weight_map),
            "reference_deviation_absent_in_anatomical_weights": bool("reference_deviation" not in learned_anatomical_component_weight_map),
        }
    )
    save_json(taxonomy_payload, component_taxonomy_audit_path)
    selected_reference_importance = [row for row in feature_importance_rows if str(row.get("feature_name", "")).startswith(REFERENCE_FEATURE_PREFIXES)]
    reference_selection_diagnostics["selected_reference_feature_importance_top"] = selected_reference_importance[:100]
    reference_selection_diagnostics["selected_reference_feature_count"] = int(sum(1 for column in selected_feature_columns if column.startswith(REFERENCE_FEATURE_PREFIXES)))
    save_json(reference_selection_diagnostics, reference_selection_diagnostics_path)

    if best_classifier_name is not None:
        confusion_payload = classification_results[best_classifier_name]["metrics"]["test"]
    else:
        confusion_payload = {"accuracy": None, "balanced_accuracy": None, "top_2_accuracy": None, "confusion_matrix": [], "labels": class_labels}
    save_json(confusion_payload, group_confusion_path)
    save_json(cv_results, cv_results_path)

    baseline_predictor = _nearest_frequent_group(y_train)
    baseline_metrics = {
        "val": _baseline_predictions(y_train, y_val, baseline_predictor),
        "test": _baseline_predictions(y_train, y_test, baseline_predictor),
    }

    selection_summary = {
        "selection_metric": primary_metric,
        "secondary_metric": secondary_metric,
        "best_regressor_model": best_regressor_name,
        "best_classifier_model": best_classifier_name,
        "best_pairwise_model": best_pairwise_name,
        "regression_ranking": [
            {
                "model_name": model_name,
                "cv_mae_mean": payload["cv"]["mean_metrics"].get("mae"),
                "cv_mae_std": payload["cv"]["std_metrics"].get("mae_std"),
                "cv_spearman_mean": payload["cv"]["mean_metrics"].get("spearman"),
                "val_selection_score": payload["calibration"]["selection_score"],
                "val_group_macro_mae": payload["calibration"]["selection_breakdown"].get("group_macro_mae"),
                "val_low_group_mae": payload["calibration"]["selection_breakdown"].get("low_group_mae"),
                "val_high_group_mae": payload["calibration"]["selection_breakdown"].get("high_group_mae"),
                "val_low_group_bias": payload["calibration"]["selection_breakdown"].get("low_group_bias"),
                "val_high_group_bias": payload["calibration"]["selection_breakdown"].get("high_group_bias"),
                "val_calibration_slope": payload["calibration"]["selection_breakdown"].get("calibration_slope"),
                "val_within_1_0_accuracy": payload["calibration"]["selection_breakdown"].get("within_1_0_accuracy"),
            }
            for model_name, payload in sorted(regression_results.items(), key=lambda item: float(item[1]["calibration"]["selection_score"]))
        ],
        "classification_ranking": [
            {
                "model_name": model_name,
                "cv_balanced_accuracy_mean": payload["cv"]["mean_metrics"].get("balanced_accuracy"),
                "cv_balanced_accuracy_std": payload["cv"]["std_metrics"].get("balanced_accuracy_std"),
            }
            for model_name, payload in sorted(classification_results.items(), key=lambda item: -(item[1]["cv"]["mean_metrics"].get("balanced_accuracy") or float("-inf")))
        ],
    }
    save_json(selection_summary, selection_path)

    confidence_config = dict(config.get("confidence", {}))
    pose_config = dict(config.get("pose_features", {}))
    dense_quantiles = confidence_config.get("dense_training_quantiles", [0.1, 0.9])
    dense_training_range = [float(np.quantile(y_train, dense_quantiles[0])), float(np.quantile(y_train, dense_quantiles[1]))] if y_train.size else [clip_lower, clip_upper]
    best_regressor_payload = regression_results.get(best_regressor_name, {}) if best_regressor_name else {}
    best_regressor_calibrator = regression_calibrators.get(best_regressor_name) if best_regressor_name else None
    primary_uses_calibration = bool(best_regressor_payload.get("calibration", {}).get("use_calibrated_as_primary", False))
    primary_score_model = {
        "family": "regression",
        "model_name": best_regressor_name,
        "selection_basis": "bias_aware_validation_composite_with_holdout_reporting",
        "ensemble_used_as_primary": False,
        "uses_calibration": primary_uses_calibration,
        "feature_set_name": best_feature_set_name,
    }
    pose_features_selected = [
        column
        for column in selected_feature_columns
        if column.startswith("pose_") or column.startswith("angle_") or column.endswith("_axis_angle_deg") or column.endswith("_line_angle_deg")
    ]
    no_pose_feature_columns = [column for column in selected_feature_columns if column not in pose_features_selected]
    pose_limited_feature_columns = no_pose_feature_columns + pose_features_selected[: min(5, len(pose_features_selected))]
    pose_audit_payload = _pose_feature_audit(
        selected_feature_columns=selected_feature_columns,
        numeric_frame=numeric,
        trainable_frame=trainable,
        target_column=target_column,
    )
    if best_regressor_name and best_regressor_name in regressor_models:
        pose_model = regressor_models[best_regressor_name]
        pose_audit_payload["experiments"] = {
            "current_selected_features": _pose_experiment_metrics(
                model=pose_model,
                numeric_frame=numeric,
                train_index=train_frame.index,
                val_index=val_frame.index,
                test_index=test_frame.index,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                sample_weights=sample_weights,
                groups=groups,
                strict_config=strict_config,
                clip_lower=clip_lower,
                clip_upper=clip_upper,
                feature_columns=selected_feature_columns,
            ),
            "no_pose_features": _pose_experiment_metrics(
                model=pose_model,
                numeric_frame=numeric,
                train_index=train_frame.index,
                val_index=val_frame.index,
                test_index=test_frame.index,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                sample_weights=sample_weights,
                groups=groups,
                strict_config=strict_config,
                clip_lower=clip_lower,
                clip_upper=clip_upper,
                feature_columns=no_pose_feature_columns,
            ),
            "pose_limited_features": _pose_experiment_metrics(
                model=pose_model,
                numeric_frame=numeric,
                train_index=train_frame.index,
                val_index=val_frame.index,
                test_index=test_frame.index,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                sample_weights=sample_weights,
                groups=groups,
                strict_config=strict_config,
                clip_lower=clip_lower,
                clip_upper=clip_upper,
                feature_columns=pose_limited_feature_columns,
            ),
            "pose_as_quality_only": _pose_experiment_metrics(
                model=pose_model,
                numeric_frame=numeric,
                train_index=train_frame.index,
                val_index=val_frame.index,
                test_index=test_frame.index,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                sample_weights=sample_weights,
                groups=groups,
                strict_config=strict_config,
                clip_lower=clip_lower,
                clip_upper=clip_upper,
                feature_columns=no_pose_feature_columns,
            ),
        }
    pose_audit_payload["config"] = pose_config
    save_json({"status": "completed", **pose_audit_payload}, pose_leakage_audit_path)

    pairwise_train_pred = _pairwise_predict_scores(
        best_pairwise_model,
        selected_feature_columns,
        pairwise_reference_frame,
        numeric.loc[train_frame.index, selected_feature_columns],
        groups,
    ) if best_pairwise_model is not None else np.asarray([], dtype=np.float64)
    pairwise_val_pred = _pairwise_predict_scores(
        best_pairwise_model,
        selected_feature_columns,
        pairwise_reference_frame,
        numeric.loc[val_frame.index, selected_feature_columns],
        groups,
    ) if best_pairwise_model is not None else np.asarray([], dtype=np.float64)
    pairwise_test_pred = _pairwise_predict_scores(
        best_pairwise_model,
        selected_feature_columns,
        pairwise_reference_frame,
        numeric.loc[test_frame.index, selected_feature_columns],
        groups,
    ) if best_pairwise_model is not None else np.asarray([], dtype=np.float64)

    ensemble_weights = {
        "regression": float(config.get("ensemble", {}).get("regression_weight", 0.45)),
        "pairwise": float(config.get("ensemble", {}).get("pairwise_weight", 0.35)) if pairwise_used_for_absolute_scoring else 0.0,
        "ordinal": float(config.get("ensemble", {}).get("ordinal_weight", 0.20)),
    }
    reg_train_pred_best = _prediction_array(best_regressor_model, X_train, lower=clip_lower, upper=clip_upper) if best_regressor_model is not None else np.asarray([], dtype=np.float64)
    reg_val_pred_best = _prediction_array(best_regressor_model, X_val, lower=clip_lower, upper=clip_upper) if best_regressor_model is not None else np.asarray([], dtype=np.float64)
    reg_train_pred_best_cal = _apply_calibrator(reg_train_pred_best, best_regressor_calibrator, clip_lower=clip_lower, clip_upper=clip_upper)
    reg_val_pred_best_cal = _apply_calibrator(reg_val_pred_best, best_regressor_calibrator, clip_lower=clip_lower, clip_upper=clip_upper)
    ensemble_train_values = []
    for idx in range(len(y_train)):
        pred, _std = _ensemble_prediction(
            {
                "regression": float(reg_train_pred_best_cal[idx]) if len(reg_train_pred_best_cal) else None,
                "pairwise": float(pairwise_train_pred[idx]) if len(pairwise_train_pred) and pairwise_used_for_absolute_scoring else None,
                "ordinal": float(ordinal_train_pred[idx]) if len(ordinal_train_pred) else None,
            },
            ensemble_weights,
        )
        ensemble_train_values.append(pred if pred is not None else np.nan)
    ensemble_train_pred = np.asarray(ensemble_train_values, dtype=np.float64)
    valid_ensemble_train = ~np.isnan(ensemble_train_pred)
    ensemble_val_values = []
    ensemble_val_stds = []
    for idx in range(len(y_val)):
        pred, std = _ensemble_prediction(
            {
                "regression": float(reg_val_pred_best_cal[idx]) if len(reg_val_pred_best_cal) else None,
                "pairwise": float(pairwise_val_pred[idx]) if len(pairwise_val_pred) and pairwise_used_for_absolute_scoring else None,
                "ordinal": float(ordinal_val_pred[idx]) if len(ordinal_val_pred) else None,
            },
            ensemble_weights,
        )
        ensemble_val_values.append(pred if pred is not None else np.nan)
        ensemble_val_stds.append(std if std is not None else np.nan)
    ensemble_val_pred = np.asarray(ensemble_val_values, dtype=np.float64)
    valid_ensemble_val = ~np.isnan(ensemble_val_pred)
    ensemble_test_values = []
    ensemble_test_stds = []
    reg_test_pred_best = _prediction_array(best_regressor_model, X_test, lower=clip_lower, upper=clip_upper) if best_regressor_model is not None else np.asarray([], dtype=np.float64)
    reg_test_pred_best_cal = _apply_calibrator(reg_test_pred_best, best_regressor_calibrator, clip_lower=clip_lower, clip_upper=clip_upper)
    for idx in range(len(y_test)):
        pred, std = _ensemble_prediction(
            {
                "regression": float(reg_test_pred_best_cal[idx]) if len(reg_test_pred_best_cal) else None,
                "pairwise": float(pairwise_test_pred[idx]) if len(pairwise_test_pred) and pairwise_used_for_absolute_scoring else None,
                "ordinal": float(ordinal_test_pred[idx]) if len(ordinal_test_pred) else None,
            },
            ensemble_weights,
        )
        ensemble_test_values.append(pred if pred is not None else np.nan)
        ensemble_test_stds.append(std if std is not None else np.nan)
    ensemble_test_pred = np.asarray(ensemble_test_values, dtype=np.float64)
    valid_ensemble_test = ~np.isnan(ensemble_test_pred)
    ensemble_summary = {
        "train": _eval_payload(y_train[valid_ensemble_train], ensemble_train_pred[valid_ensemble_train], groups) if len(y_train) else _eval_payload(np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), groups),
        "val": _eval_payload(y_val[valid_ensemble_val], ensemble_val_pred[valid_ensemble_val], groups) if len(y_val) else _eval_payload(np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), groups),
        "test": _eval_payload(y_test[valid_ensemble_test], ensemble_test_pred[valid_ensemble_test], groups) if len(y_test) else _eval_payload(np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), groups),
        "val_mean_prediction_std": float(np.nanmean(np.asarray(ensemble_val_stds, dtype=np.float64))) if ensemble_val_stds else None,
        "mean_prediction_std": float(np.nanmean(np.asarray(ensemble_test_stds, dtype=np.float64))) if ensemble_test_stds else None,
        "pairwise_used_for_absolute_scoring": pairwise_used_for_absolute_scoring,
    }
    save_json({"status": "completed", "weights": ensemble_weights, "metrics": ensemble_summary}, ensemble_summary_path)

    save_json(
        {
            "status": "completed" if best_regressor_name else "skipped_no_data",
            "best_regressor_model": best_regressor_name,
            "group_support": {
                str(group): {
                    "count": int((frame[target_column].astype(float) == float(group)).sum()),
                    "quality_pass_count": int((trainable[target_column].astype(float) == float(group)).sum()),
                }
                for group in groups
            },
            "train": regression_results.get(best_regressor_name, {}).get("metrics", {}).get("train", {}) if best_regressor_name else {},
            "val": regression_results.get(best_regressor_name, {}).get("metrics", {}).get("val", {}) if best_regressor_name else {},
            "test": regression_results.get(best_regressor_name, {}).get("metrics", {}).get("test", {}) if best_regressor_name else {},
            "calibrated": {
                "train": regression_results.get(best_regressor_name, {}).get("calibrated_metrics", {}).get("train", {}) if best_regressor_name else {},
                "val": regression_results.get(best_regressor_name, {}).get("calibrated_metrics", {}).get("val", {}) if best_regressor_name else {},
                "test": regression_results.get(best_regressor_name, {}).get("calibrated_metrics", {}).get("test", {}) if best_regressor_name else {},
            },
        },
        calibration_summary_path,
    )
    best_raw_test_summary = regression_results.get(best_regressor_name, {}).get("metrics", {}).get("test", {}).get("calibration_summary", {}) if best_regressor_name else {}
    best_cal_test_summary = regression_results.get(best_regressor_name, {}).get("calibrated_metrics", {}).get("test", {}).get("calibration_summary", {}) if best_regressor_name else {}
    save_json(
        {
            "status": "completed" if best_regressor_name else "skipped_no_data",
            "strict_scale_enabled": bool(strict_config.get("enabled", False)),
            "primary_model": best_regressor_name,
            "primary_uses_calibration": primary_uses_calibration,
            "raw": best_raw_test_summary,
            "calibrated": best_cal_test_summary,
            "warnings": sorted(set((best_raw_test_summary.get("warnings") or []) + (best_cal_test_summary.get("warnings") or []))),
        },
        extreme_bias_summary_path,
    )

    groupwise_errors_payload = {
        "under_supported_groups": unstable_groups,
        "regression_best": regression_results.get(best_regressor_name, {}).get("metrics", {}) if best_regressor_name else {},
        "pairwise_best": pairwise_results.get(best_pairwise_name, {}) if best_pairwise_name else {},
        "ordinal": ordinal_metrics,
        "ensemble": ensemble_summary,
    }
    save_json(groupwise_errors_payload, groupwise_errors_path)

    quality_failures_rows = []
    for _, row in frame.iterrows():
        quality_pass = bool(row.get("quality_pass"))
        if quality_pass:
            continue
        image_path_value = row.get("image_path")
        quality_failures_rows.append(
            {
                "image_path": str(image_path_value) if image_path_value is not None else None,
                "target_score": float(row["target_score"]) if pd.notna(row.get("target_score")) else None,
                "processing_status": row.get("processing_status"),
                "quality_pass": quality_pass,
                "quality_fail_reason": row.get("quality_fail_reason"),
                "whether_used_for_training": bool(row.get(split_column) == "train" and quality_pass),
            }
        )
    save_json({"rows": quality_failures_rows}, quality_failures_path)

    example_split = test_frame if len(test_frame) else val_frame
    classifier_for_examples = best_classifier_model
    classifier_example_predictions: list[str | None] = []
    if classifier_for_examples is not None and len(example_split):
        classifier_example_predictions = [str(value) for value in classifier_for_examples.predict(numeric.loc[example_split.index, selected_feature_columns].to_numpy(dtype=np.float64))]
    ordinal_example_pred = ordinal_test_pred if len(test_frame) else ordinal_val_pred
    pairwise_example_pred = pairwise_test_pred if len(test_frame) else pairwise_val_pred
    raw_example_pred = reg_test_pred_best if len(test_frame) else reg_val_pred_best
    calibrated_example_pred = reg_test_pred_best_cal if len(test_frame) else reg_val_pred_best_cal
    example_rows: list[dict[str, Any]] = []
    classifier_map = {idx: classifier_example_predictions[pos] for pos, idx in enumerate(example_split.index)} if classifier_example_predictions else {}
    for target_score, subset in example_split.groupby(target_column):
        for row_index in subset.index[:3]:
            subset_position = int(np.where(example_split.index.to_numpy() == row_index)[0][0])
            example_rows.append(
                {
                    "split": "test" if len(test_frame) else "val",
                    "image_path": str(example_split.loc[row_index, "image_path"]) if pd.notna(example_split.loc[row_index, "image_path"]) else None,
                    "target_score": float(target_score),
                    "raw_primary_score": float(raw_example_pred[subset_position]) if len(raw_example_pred) else None,
                    "calibrated_score": float(calibrated_example_pred[subset_position]) if len(calibrated_example_pred) else None,
                    "ordinal_score": float(ordinal_example_pred[subset_position]) if len(ordinal_example_pred) else None,
                    "pairwise_score": float(pairwise_example_pred[subset_position]) if len(pairwise_example_pred) else None,
                    "classifier_predicted_group": classifier_map.get(row_index),
                    "confidence": "low" if len(calibrated_example_pred) and abs(float(calibrated_example_pred[subset_position]) - float(target_score)) > 1.5 else "medium",
                    "warnings": (["possible_high_score_underprediction"] if float(target_score) >= float(strict_config.get("elite_score_threshold", 9.0)) and len(calibrated_example_pred) and float(calibrated_example_pred[subset_position]) < float(target_score) - 1.0 else []),
                    "abs_error": float(abs((calibrated_example_pred[subset_position] if len(calibrated_example_pred) else np.nan) - float(target_score))) if len(calibrated_example_pred) else None,
                }
            )
    save_json({"rows": example_rows}, prediction_examples_path)

    if best_regressor_model is not None:
        joblib.dump(
            {
                "model_name": best_regressor_name,
                "estimator": best_regressor_model,
                "score_calibrator": best_regressor_calibrator,
                "calibration_method": "isotonic_regression" if best_regressor_calibrator is not None else None,
                "selected_features": selected_feature_columns,
                "feature_columns": selected_feature_columns,
                "feature_medians": medians,
                "feature_policy": selected_feature_report["feature_policy"],
                "training_feature_count": len(selected_feature_columns),
                "target_column": target_column,
                "target_group_column": target_group_column,
                "score_groups": groups,
                "training_target_mean": float(np.mean(y_train)) if len(y_train) else None,
                "feature_means": {column: float(numeric.loc[train_frame.index, column].mean()) for column in selected_feature_columns},
                "feature_stds": {
                    column: float(std) if float(std) > 0 else 1.0
                    for column, std in numeric.loc[train_frame.index, selected_feature_columns].std(axis=0, numeric_only=True).fillna(1.0).to_dict().items()
                },
                "prediction_clip_lower": clip_lower,
                "prediction_clip_upper": clip_upper,
                "observed_training_label_range": observed_training_label_range,
                "dense_training_range": dense_training_range,
                "calibration_metrics": {
                    "raw": best_regressor_payload.get("metrics", {}),
                    "calibrated": best_regressor_payload.get("calibrated_metrics", {}),
                },
                "group_support": {str(key): int(value) for key, value in group_counts.items()},
                "global_feature_importance": feature_importance_rows[:20],
                "global_feature_effects": feature_effect_rows,
                "learned_feature_weights": learned_feature_weight_rows[:400],
                "learned_component_weights": learned_anatomical_component_weight_rows,
                "learned_anatomical_component_weights": learned_anatomical_component_weight_rows,
                "learned_transform_family_weights": learned_transform_family_weight_rows,
                "learned_source_weights": learned_source_weight_rows,
                "learned_weight_method": learned_weight_method,
                "angles_pose_weight": angles_pose_weight,
                "angles_pose_weight_cap": pose_weight_cap,
                "angles_pose_weight_cap_exceeded": pose_weight_cap_exceeded,
                "component_map": dict(config.get("components", {})),
                "primary_score_model": primary_score_model,
                "primary_uses_calibration": primary_uses_calibration,
                "feature_set_name": best_feature_set_name,
                "selected_feature_source_counts": _feature_source_counts(selected_feature_columns, _feature_registry_map()),
                "selected_component_counts": _component_counts(
                    selected_feature_columns,
                    registry_by_name=registry_by_name,
                    component_map=component_map,
                ),
                "selected_anatomical_component_counts": dict(selected_feature_report.get("anatomical_component_counts", {})),
                "selected_transform_family_counts": dict(selected_feature_report.get("transform_family_counts", {})),
                "selected_source_counts": dict(selected_feature_report.get("source_counts", {})),
                "mediapipe_features_used": int(sum(1 for column in selected_feature_columns if column.startswith("mediapipe_"))),
                "direct_mediapipe_features_used": direct_mediapipe_selected_count,
                "mediapipe_derived_reference_features_used": mediapipe_derived_reference_selected_count,
                "total_mediapipe_influence_count": int(direct_mediapipe_selected_count + mediapipe_derived_reference_selected_count),
                "reference_relative_features_used": int(sum(1 for column in selected_feature_columns if column.startswith(REFERENCE_FEATURE_PREFIXES))),
                "selected_feature_lineage": feature_lineage_rows,
                "reference_stats_path": reference_stats.stats_path.relative_to(root).as_posix() if reference_stats is not None else None,
                "reference_filters": reference_stats.filters if reference_stats is not None else {},
                "scoring_model_info": {
                    "primary_model": best_regressor_name,
                    "primary_feature_set": best_feature_set_name,
                    "uses_mediapipe": int(direct_mediapipe_selected_count + mediapipe_derived_reference_selected_count) > 0,
                    "uses_reference_relative": int(sum(1 for column in selected_feature_columns if column.startswith(REFERENCE_FEATURE_PREFIXES))) > 0,
                    "weight_method": learned_weight_method,
                    "selected_component_counts": dict(selected_feature_report.get("anatomical_component_counts", {})),
                    "selected_transform_family_counts": dict(selected_feature_report.get("transform_family_counts", {})),
                    "selected_source_counts": dict(selected_feature_report.get("source_counts", {})),
                },
                "active_feature_source_policy": {
                    "mode": str(config.get("feature_source_policy", {}).get("mode", "mixed")),
                    "insightface_used_for_primary": bool(selected_feature_report.get("insightface_selected_count", 0)),
                    "mediapipe_used_for_primary": bool(direct_mediapipe_selected_count + mediapipe_derived_reference_selected_count),
                },
                "training_timestamp": datetime.now(timezone.utc).isoformat(),
                "model_selected_by": {
                    "selection_metric": primary_metric,
                    "secondary_metric": secondary_metric,
                    "cv_on_split": "train",
                    "strict_selection": True,
                },
                "test_metrics_available": bool(len(y_test)),
                "warnings": (["Some score groups are under-supported and may yield unstable regression targets."] if unstable_groups else []),
            },
            regressor_artifact_path,
        )
        if best_regressor_calibrator is not None:
            joblib.dump(
                {
                    "method": "isotonic_regression",
                    "fitted_on_split": "val",
                    "calibrator": best_regressor_calibrator,
                    "raw_validation_predictions": reg_val_pred_best.tolist() if len(reg_val_pred_best) else [],
                    "validation_targets": y_val.tolist() if len(y_val) else [],
                },
                calibrator_artifact_path,
            )
    if best_classifier_model is not None:
        joblib.dump(
            {
                "model_name": best_classifier_name,
                "estimator": best_classifier_model,
                "classes": class_labels,
                "selected_features": selected_feature_columns,
                "feature_columns": selected_feature_columns,
                "feature_set_name": best_feature_set_name,
            },
            classifier_artifact_path,
        )
    if best_pairwise_model is not None:
        joblib.dump(
            {
                "model_name": best_pairwise_name,
                "estimator": best_pairwise_model,
                "selected_features": selected_feature_columns,
                "feature_columns": selected_feature_columns,
                "reference_examples": pairwise_reference_frame[["target_score", "target_group_label", *selected_feature_columns]].to_dict(orient="records"),
                "score_groups": groups,
                "use_for_absolute_scoring": pairwise_used_for_absolute_scoring,
                "feature_set_name": best_feature_set_name,
            },
            pairwise_artifact_path,
        )
    if ordinal_payload.get("thresholds"):
        joblib.dump(
            {
                "selected_features": selected_feature_columns,
                "feature_columns": selected_feature_columns,
                "thresholds": ordinal_payload["thresholds"],
                "groups": groups,
                "score_groups": groups,
                "feature_set_name": best_feature_set_name,
            },
            ordinal_artifact_path,
        )

    summary = {
        "status": "completed" if best_regressor_model is not None else "skipped_no_data",
        "best_regressor_model": best_regressor_name,
        "best_classifier_model": best_classifier_name,
        "best_pairwise_model": best_pairwise_name,
        "primary_score_model": primary_score_model,
        "selection_metric": primary_metric,
        "data": {
            "total_rows": int(len(frame)),
            "quality_pass_rows": int(frame["quality_pass"].fillna(False).astype(bool).sum()) if "quality_pass" in frame.columns else 0,
            "trainable_rows": int(len(trainable)),
            "excluded_quality_failed_rows": int(len(frame) - len(trainable)) if str(config.get("quality_gate_policy", "")).strip().lower() == "train_only_quality_pass" else 0,
            "selected_feature_count": int(len(selected_feature_columns)),
            "selected_feature_columns": selected_feature_columns,
            "feature_set_name": best_feature_set_name,
            "selected_feature_source_counts": _feature_source_counts(selected_feature_columns, _feature_registry_map()),
            "selected_component_counts": _component_counts(
                selected_feature_columns,
                registry_by_name=registry_by_name,
                component_map=component_map,
            ),
            "selected_anatomical_component_counts": dict(selected_feature_report.get("anatomical_component_counts", {})),
            "selected_transform_family_counts": dict(selected_feature_report.get("transform_family_counts", {})),
            "selected_source_counts": dict(selected_feature_report.get("source_counts", {})),
            "mediapipe_selected_count": int(sum(1 for column in selected_feature_columns if column.startswith("mediapipe_"))),
            "direct_mediapipe_selected_count": direct_mediapipe_selected_count,
            "mediapipe_derived_reference_selected_count": mediapipe_derived_reference_selected_count,
            "total_mediapipe_influence_count": int(direct_mediapipe_selected_count + mediapipe_derived_reference_selected_count),
            "reference_relative_selected_count": int(sum(1 for column in selected_feature_columns if column.startswith(REFERENCE_FEATURE_PREFIXES))),
            "split_counts": {"train": int(len(train_frame)), "val": int(len(val_frame)), "test": int(len(test_frame))},
            "score_range": [clip_lower, clip_upper],
            "observed_training_label_range": observed_training_label_range,
            "dense_training_range": dense_training_range,
            "score_semantics": score_semantics,
            "reference_source": reference_config.get("reference_source", "fairface"),
            "reference_stats_path": reference_stats.stats_path.relative_to(root).as_posix() if reference_stats is not None else None,
            "reference_filters": reference_stats.filters if reference_stats is not None else {},
            "group_count": int(len(groups)),
            "per_score_group_support": {str(key): int(value) for key, value in group_counts.items()},
            "under_supported_groups": unstable_groups,
            "sample_weights": group_weight_map,
        },
        "regression_models": regression_results,
        "pairwise_models": pairwise_results,
        "ordinal_model": ordinal_metrics,
        "ensemble_model": ensemble_summary,
        "classification_models": classification_results,
        "calibration": {
            "report_path": calibration_summary_path.relative_to(root).as_posix(),
            "extreme_bias_summary_path": extreme_bias_summary_path.relative_to(root).as_posix(),
            "prediction_examples_path": prediction_examples_path.relative_to(root).as_posix(),
        },
        "cv_summary": {
            "results_path": cv_results_path.relative_to(root).as_posix(),
            "model_selection_path": selection_path.relative_to(root).as_posix(),
        },
        "feature_set_comparison": {
            "path": feature_set_comparison_path.relative_to(root).as_posix(),
            "selected_feature_set": best_feature_set_name,
            "expanded_geometry_comparison_path": expanded_geometry_comparison_path.relative_to(root).as_posix(),
            "mediapipe_only_comparison_path": mediapipe_only_comparison_path.relative_to(root).as_posix(),
            "edge_weighting_comparison_path": edge_weighting_comparison_path.relative_to(root).as_posix(),
            "weighted_model_comparison_path": weighted_model_comparison_path.relative_to(root).as_posix(),
            "component_balanced_feature_selection_path": component_balanced_selection_path.relative_to(root).as_posix(),
            "base_feature_dedup_audit_path": base_feature_dedup_audit_path.relative_to(root).as_posix(),
            "guardrails_selection_path": guardrails_selection_path.relative_to(root).as_posix() if guardrails_selection_path.exists() else None,
            "primary_regression_comparison_path": primary_regression_comparison_path.relative_to(root).as_posix() if primary_regression_comparison_path.exists() else None,
        },
        "fairface_reference_comparison": {
            "path": fairface_reference_comparison_path.relative_to(root).as_posix(),
            "selected_feature_set": fairface_reference_comparison.get("selected_feature_set"),
            "reference_feature_selection_diagnostics_path": reference_selection_diagnostics_path.relative_to(root).as_posix(),
            "reference_resolution_summary_path": reference_resolution_summary_path.relative_to(root).as_posix(),
            "base_feature_dedup_audit_path": base_feature_dedup_audit_path.relative_to(root).as_posix(),
            "pose_leakage_audit_path": pose_leakage_audit_path.relative_to(root).as_posix(),
        },
        "baseline_comparison": baseline_metrics,
        "artifacts": {
            "training_summary_path": summary_path.relative_to(root).as_posix(),
            "feature_importance_path": feature_importance_path.relative_to(root).as_posix(),
            "feature_correlations_path": feature_correlations_path.relative_to(root).as_posix(),
            "global_feature_importance_path": global_feature_importance_path.relative_to(root).as_posix(),
            "component_importance_path": component_importance_path.relative_to(root).as_posix(),
            "feature_effects_path": feature_effects_path.relative_to(root).as_posix(),
            "learned_feature_weights_path": learned_feature_weights_path.relative_to(root).as_posix(),
            "learned_component_weights_path": learned_component_weights_path.relative_to(root).as_posix(),
            "learned_anatomical_component_weights_path": learned_anatomical_component_weights_path.relative_to(root).as_posix(),
            "learned_transform_family_weights_path": learned_transform_family_weights_path.relative_to(root).as_posix(),
            "learned_source_weights_path": learned_source_weights_path.relative_to(root).as_posix(),
            "component_weight_summary_path": component_weight_summary_path.relative_to(root).as_posix(),
            "component_taxonomy_audit_path": component_taxonomy_audit_path.relative_to(root).as_posix(),
            "weighted_model_comparison_path": weighted_model_comparison_path.relative_to(root).as_posix(),
            "mediapipe_only_comparison_path": mediapipe_only_comparison_path.relative_to(root).as_posix(),
            "edge_weighting_comparison_path": edge_weighting_comparison_path.relative_to(root).as_posix(),
            "component_balanced_feature_selection_path": component_balanced_selection_path.relative_to(root).as_posix(),
            "selected_features_path": selected_features_path.relative_to(root).as_posix(),
            "pairwise_dataset_path": pairwise_dataset_path.relative_to(root).as_posix(),
            "pairwise_summary_path": pairwise_summary_path.relative_to(root).as_posix(),
            "ordinal_summary_path": ordinal_summary_path.relative_to(root).as_posix(),
            "ensemble_summary_path": ensemble_summary_path.relative_to(root).as_posix(),
            "pairwise_dataset_summary_path": pairwise_dataset_summary_path.relative_to(root).as_posix(),
            "group_confusion_path": group_confusion_path.relative_to(root).as_posix(),
            "groupwise_errors_path": groupwise_errors_path.relative_to(root).as_posix(),
            "quality_failures_path": quality_failures_path.relative_to(root).as_posix(),
            "cv_results_path": cv_results_path.relative_to(root).as_posix(),
            "model_selection_path": selection_path.relative_to(root).as_posix(),
            "regressor_artifact_path": regressor_artifact_path.relative_to(root).as_posix(),
            "classifier_artifact_path": classifier_artifact_path.relative_to(root).as_posix(),
            "pairwise_artifact_path": pairwise_artifact_path.relative_to(root).as_posix(),
            "ordinal_artifact_path": ordinal_artifact_path.relative_to(root).as_posix(),
            "calibrator_artifact_path": calibrator_artifact_path.relative_to(root).as_posix() if calibrator_artifact_path.exists() else None,
            "config_snapshot_path": config_snapshot_path.relative_to(root).as_posix(),
            "calibration_summary_path": calibration_summary_path.relative_to(root).as_posix(),
            "extreme_bias_summary_path": extreme_bias_summary_path.relative_to(root).as_posix(),
            "prediction_examples_path": prediction_examples_path.relative_to(root).as_posix(),
            "feature_set_comparison_path": feature_set_comparison_path.relative_to(root).as_posix(),
            "scorer_model_selection_guardrails_path": guardrails_selection_path.relative_to(root).as_posix() if guardrails_selection_path.exists() else None,
            "scorer_primary_regression_comparison_path": primary_regression_comparison_path.relative_to(root).as_posix() if primary_regression_comparison_path.exists() else None,
            "expanded_geometry_comparison_path": expanded_geometry_comparison_path.relative_to(root).as_posix(),
            "fairface_reference_comparison_path": fairface_reference_comparison_path.relative_to(root).as_posix(),
            "feature_correlation_audit_path": feature_correlation_audit_path.relative_to(root).as_posix(),
            "reference_feature_selection_diagnostics_path": reference_selection_diagnostics_path.relative_to(root).as_posix(),
            "reference_resolution_summary_path": reference_resolution_summary_path.relative_to(root).as_posix(),
        },
    }
    save_json(summary, summary_path)
    return ScorerTrainingArtifacts(
        summary_path=summary_path,
        feature_importance_path=feature_importance_path,
        feature_correlations_path=feature_correlations_path,
        group_confusion_path=group_confusion_path,
        regressor_artifact_path=regressor_artifact_path,
        classifier_artifact_path=classifier_artifact_path,
        summary=summary,
    )
