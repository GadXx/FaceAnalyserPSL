from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src.face_aesthetics.models.baselines import build_xgboost_regressor, compute_regression_metrics
from src.face_aesthetics.utils.config import load_named_configs
from src.face_aesthetics.utils.filesystem import ensure_parent_directory


@dataclass
class GeometryTrainingArtifacts:
    metrics_json_path: Path
    comparison_report_path: Path
    val_predictions_path: Path
    test_predictions_path: Path
    feature_importance_path: Path
    config_snapshot_path: Path
    summary: dict[str, Any]


def _save_json(payload: dict[str, Any], path: Path) -> None:
    ensure_parent_directory(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True, sort_keys=True)


def _save_predictions(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_parent_directory(path)
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_feature_importance(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_parent_directory(path)
    pd.DataFrame(rows).to_csv(path, index=False)


def _select_feature_columns(frame: pd.DataFrame, prefixes: list[str]) -> list[str]:
    return [
        column
        for column in frame.columns
        if any(column.startswith(prefix) for prefix in prefixes)
    ]


def _coerce_splits(frame: pd.DataFrame, split_column: str, random_seed: int, derived_split: dict[str, Any]) -> pd.DataFrame:
    working = frame.copy()
    if split_column in working.columns:
        normalized = working[split_column].fillna("").astype(str).str.lower().str.strip()
        if normalized.isin(["train", "val", "test"]).any():
            working[split_column] = normalized.replace({"validation": "val", "valid": "val"})
            return working

    train_size = float(derived_split["train_size"])
    val_size = float(derived_split["val_size"])
    test_size = float(derived_split["test_size"])
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Derived split fractions must sum to 1.0")

    indices = np.arange(len(working))
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_size,
        random_state=random_seed,
        shuffle=True,
    )
    relative_val_size = val_size / max(val_size + test_size, 1e-12)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=relative_val_size,
        random_state=random_seed,
        shuffle=True,
    )

    split_values = np.full(len(working), "train", dtype=object)
    split_values[val_idx] = "val"
    split_values[test_idx] = "test"
    working[split_column] = split_values
    return working


def _prediction_rows(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    image_id_column: str,
    split_name: str,
    model_name: str,
    target_column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_id, target, prediction in zip(frame[image_id_column].tolist(), y_true.tolist(), y_pred.tolist()):
        rows.append(
            {
                "image_id": image_id,
                "split": split_name,
                target_column: float(target),
                f"predicted_{target_column}": float(prediction),
                "model_name": model_name,
            }
        )
    return rows


def _feature_importance_rows(model: Any, feature_columns: list[str]) -> list[dict[str, Any]]:
    if not hasattr(model, "feature_importances_"):
        return []
    importances = getattr(model, "feature_importances_")
    rows = [
        {"feature_name": feature_name, "importance": float(importance)}
        for feature_name, importance in zip(feature_columns, importances.tolist())
    ]
    return sorted(rows, key=lambda item: item["importance"], reverse=True)


def _build_ridge_regressor(config: dict[str, Any]) -> Ridge:
    return Ridge(alpha=float(config["alpha"]))


def _extract_best_metrics(summary: dict[str, Any]) -> dict[str, Any] | None:
    best_model = summary.get("best_model")
    if not best_model:
        return None
    models = summary.get("models", {})
    if best_model not in models:
        return None
    return models[best_model].get("metrics")


def _build_comparison_report(
    geometry_summary: dict[str, Any],
    deep_summary: dict[str, Any],
) -> dict[str, Any]:
    geometry_best = geometry_summary.get("best_model")
    deep_best = deep_summary.get("best_model")
    geometry_best_metrics = _extract_best_metrics(geometry_summary)
    deep_best_metrics = _extract_best_metrics(deep_summary)

    comparison = {
        "geometry_status": geometry_summary.get("status"),
        "deep_status": deep_summary.get("status"),
        "geometry_best_model": geometry_best,
        "deep_best_model": deep_best,
        "geometry_best_metrics": geometry_best_metrics,
        "deep_best_metrics": deep_best_metrics,
        "delta_vs_deep": None,
        "conclusion": None,
    }

    if geometry_best_metrics and deep_best_metrics:
        geometry_val = geometry_best_metrics.get("val", {})
        deep_val = deep_best_metrics.get("val", {})
        delta = {}
        for metric_name in ("mae", "rmse", "spearman", "pearson"):
            g_value = geometry_val.get(metric_name)
            d_value = deep_val.get(metric_name)
            if g_value is None or d_value is None:
                delta[metric_name] = None
            else:
                delta[metric_name] = float(g_value) - float(d_value)
        comparison["delta_vs_deep"] = delta

        rmse_delta = delta.get("rmse")
        if rmse_delta is None:
            comparison["conclusion"] = "comparison_inconclusive"
        elif rmse_delta < -1e-9:
            comparison["conclusion"] = "geometry_branch_beats_deep_baseline_on_val"
        elif rmse_delta > 1e-9:
            comparison["conclusion"] = "geometry_branch_is_weaker_than_deep_baseline_on_val"
        else:
            comparison["conclusion"] = "geometry_branch_matches_deep_baseline_on_val"
    else:
        comparison["conclusion"] = "comparison_inconclusive_no_metrics"

    return comparison


def run_geometry_training(project_root: Path | None = None) -> GeometryTrainingArtifacts:
    root = (project_root or Path.cwd()).resolve()
    config = load_named_configs(root / "configs")["train_geometry"]["train_geometry"]

    input_features_path = (root / config["input_features_path"]).resolve()
    deep_baseline_metrics_path = (root / config["deep_baseline_metrics_path"]).resolve()
    metrics_json_path = (root / config["artifacts"]["metrics_json_path"]).resolve()
    comparison_report_path = (root / config["artifacts"]["comparison_report_path"]).resolve()
    val_predictions_path = (root / config["artifacts"]["val_predictions_path"]).resolve()
    test_predictions_path = (root / config["artifacts"]["test_predictions_path"]).resolve()
    feature_importance_path = (root / config["artifacts"]["feature_importance_path"]).resolve()
    config_snapshot_path = (root / config["artifacts"]["config_snapshot_path"]).resolve()
    models_dir = (root / config["artifacts"]["models_dir"]).resolve()

    for path in (
        metrics_json_path,
        comparison_report_path,
        val_predictions_path,
        test_predictions_path,
        feature_importance_path,
        config_snapshot_path,
    ):
        ensure_parent_directory(path)
    models_dir.mkdir(parents=True, exist_ok=True)

    _save_json(config, config_snapshot_path)

    frame = pd.read_parquet(input_features_path)
    target_column = config["target_column"]
    split_column = config["split_column"]
    image_id_column = config["image_id_column"]
    feature_columns = _select_feature_columns(frame, list(config["feature_prefixes"]))

    if deep_baseline_metrics_path.exists():
        deep_summary = json.loads(deep_baseline_metrics_path.read_text(encoding="utf-8"))
    else:
        deep_summary = {"status": "missing_deep_baseline_metrics", "best_model": None, "models": {}}

    if frame.empty or not feature_columns:
        empty_predictions = pd.DataFrame(
            columns=["image_id", "split", target_column, f"predicted_{target_column}", "model_name"]
        )
        empty_predictions.to_csv(val_predictions_path, index=False)
        empty_predictions.to_csv(test_predictions_path, index=False)
        _save_feature_importance([], feature_importance_path)

        summary = {
            "status": "skipped_no_data",
            "reason": "No geometry morphometric feature rows available for training.",
            "best_model": None,
            "selection_metric": config["selection_metric"],
            "models": {},
            "data": {
                "input_rows": int(len(frame)),
                "feature_count": int(len(feature_columns)),
                "split_counts": {},
            },
            "artifacts": {
                "metrics_json_path": metrics_json_path.relative_to(root).as_posix(),
                "comparison_report_path": comparison_report_path.relative_to(root).as_posix(),
                "val_predictions_path": val_predictions_path.relative_to(root).as_posix(),
                "test_predictions_path": test_predictions_path.relative_to(root).as_posix(),
                "feature_importance_path": feature_importance_path.relative_to(root).as_posix(),
                "config_snapshot_path": config_snapshot_path.relative_to(root).as_posix(),
                "models_dir": models_dir.relative_to(root).as_posix(),
            },
        }
        comparison_report = _build_comparison_report(summary, deep_summary)
        _save_json(summary, metrics_json_path)
        _save_json(comparison_report, comparison_report_path)
        return GeometryTrainingArtifacts(
            metrics_json_path=metrics_json_path,
            comparison_report_path=comparison_report_path,
            val_predictions_path=val_predictions_path,
            test_predictions_path=test_predictions_path,
            feature_importance_path=feature_importance_path,
            config_snapshot_path=config_snapshot_path,
            summary={"geometry": summary, "comparison": comparison_report},
        )

    working = frame.dropna(subset=[target_column]).copy()
    working = _coerce_splits(
        frame=working,
        split_column=split_column,
        random_seed=int(config["random_seed"]),
        derived_split=config["derived_split"],
    )

    train_frame = working[working[split_column] == "train"].copy()
    val_frame = working[working[split_column] == "val"].copy()
    test_frame = working[working[split_column] == "test"].copy()

    if working.empty or train_frame.empty:
        empty_predictions = pd.DataFrame(
            columns=["image_id", "split", target_column, f"predicted_{target_column}", "model_name"]
        )
        empty_predictions.to_csv(val_predictions_path, index=False)
        empty_predictions.to_csv(test_predictions_path, index=False)
        _save_feature_importance([], feature_importance_path)

        summary = {
            "status": "skipped_no_train_rows",
            "reason": "Rows exist, but there is no usable training split after filtering.",
            "best_model": None,
            "selection_metric": config["selection_metric"],
            "models": {},
            "data": {
                "input_rows": int(len(frame)),
                "usable_rows": int(len(working)),
                "train_rows": int(len(train_frame)),
                "val_rows": int(len(val_frame)),
                "test_rows": int(len(test_frame)),
                "feature_count": int(len(feature_columns)),
            },
            "artifacts": {
                "metrics_json_path": metrics_json_path.relative_to(root).as_posix(),
                "comparison_report_path": comparison_report_path.relative_to(root).as_posix(),
                "val_predictions_path": val_predictions_path.relative_to(root).as_posix(),
                "test_predictions_path": test_predictions_path.relative_to(root).as_posix(),
                "feature_importance_path": feature_importance_path.relative_to(root).as_posix(),
                "config_snapshot_path": config_snapshot_path.relative_to(root).as_posix(),
                "models_dir": models_dir.relative_to(root).as_posix(),
            },
        }
        comparison_report = _build_comparison_report(summary, deep_summary)
        _save_json(summary, metrics_json_path)
        _save_json(comparison_report, comparison_report_path)
        return GeometryTrainingArtifacts(
            metrics_json_path=metrics_json_path,
            comparison_report_path=comparison_report_path,
            val_predictions_path=val_predictions_path,
            test_predictions_path=test_predictions_path,
            feature_importance_path=feature_importance_path,
            config_snapshot_path=config_snapshot_path,
            summary={"geometry": summary, "comparison": comparison_report},
        )

    x_train = train_frame[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train_frame[target_column].to_numpy(dtype=np.float32)
    x_val = val_frame[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    y_val = val_frame[target_column].to_numpy(dtype=np.float32)
    x_test = test_frame[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    y_test = test_frame[target_column].to_numpy(dtype=np.float32)

    model_specs: list[tuple[str, Any, Path]] = []
    if config["models"]["ridge"]["enabled"]:
        model_specs.append(
            (
                "ridge",
                _build_ridge_regressor(config["models"]["ridge"]),
                (root / config["models"]["ridge"]["artifact_path"]).resolve(),
            )
        )
    if config["models"]["xgboost"]["enabled"]:
        model_specs.append(
            (
                "xgboost",
                build_xgboost_regressor(config["models"]["xgboost"], int(config["random_seed"])),
                (root / config["models"]["xgboost"]["artifact_path"]).resolve(),
            )
        )

    val_prediction_rows: list[dict[str, Any]] = []
    test_prediction_rows: list[dict[str, Any]] = []
    feature_importance_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}

    for model_name, model, artifact_path in model_specs:
        model.fit(x_train, y_train)
        ensure_parent_directory(artifact_path)
        joblib.dump(model, artifact_path)

        val_pred = model.predict(x_val) if len(val_frame) else np.asarray([], dtype=np.float32)
        test_pred = model.predict(x_test) if len(test_frame) else np.asarray([], dtype=np.float32)

        model_summaries[model_name] = {
            "artifact_path": artifact_path.relative_to(root).as_posix(),
            "metrics": {
                "val": compute_regression_metrics(y_val, val_pred),
                "test": compute_regression_metrics(y_test, test_pred),
            },
        }

        val_prediction_rows.extend(
            _prediction_rows(val_frame, y_val, val_pred, image_id_column, "val", model_name, target_column)
        )
        test_prediction_rows.extend(
            _prediction_rows(test_frame, y_test, test_pred, image_id_column, "test", model_name, target_column)
        )

        if model_name == "xgboost":
            feature_importance_rows = _feature_importance_rows(model, feature_columns)

    selection_metric = config["selection_metric"]
    ranked = []
    for model_name, payload in model_summaries.items():
        metric_value = payload["metrics"]["val"].get(selection_metric)
        if metric_value is not None:
            ranked.append((model_name, float(metric_value)))
    best_model = min(ranked, key=lambda item: item[1])[0] if ranked else None

    _save_predictions(val_prediction_rows, val_predictions_path)
    _save_predictions(test_prediction_rows, test_predictions_path)
    _save_feature_importance(feature_importance_rows, feature_importance_path)

    summary = {
        "status": "completed",
        "best_model": best_model,
        "selection_metric": selection_metric,
        "models": model_summaries,
        "data": {
            "input_rows": int(len(frame)),
            "train_rows": int(len(train_frame)),
            "val_rows": int(len(val_frame)),
            "test_rows": int(len(test_frame)),
            "feature_count": int(len(feature_columns)),
        },
        "artifacts": {
            "metrics_json_path": metrics_json_path.relative_to(root).as_posix(),
            "comparison_report_path": comparison_report_path.relative_to(root).as_posix(),
            "val_predictions_path": val_predictions_path.relative_to(root).as_posix(),
            "test_predictions_path": test_predictions_path.relative_to(root).as_posix(),
            "feature_importance_path": feature_importance_path.relative_to(root).as_posix(),
            "config_snapshot_path": config_snapshot_path.relative_to(root).as_posix(),
            "models_dir": models_dir.relative_to(root).as_posix(),
        },
    }
    comparison_report = _build_comparison_report(summary, deep_summary)
    _save_json(summary, metrics_json_path)
    _save_json(comparison_report, comparison_report_path)

    return GeometryTrainingArtifacts(
        metrics_json_path=metrics_json_path,
        comparison_report_path=comparison_report_path,
        val_predictions_path=val_predictions_path,
        test_predictions_path=test_predictions_path,
        feature_importance_path=feature_importance_path,
        config_snapshot_path=config_snapshot_path,
        summary={"geometry": summary, "comparison": comparison_report},
    )
