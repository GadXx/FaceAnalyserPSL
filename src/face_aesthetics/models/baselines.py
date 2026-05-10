from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "mae": None,
            "rmse": None,
            "spearman": None,
            "pearson": None,
        }

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    spearman = None
    pearson = None
    if y_true.size >= 2 and np.std(y_true) > 0.0 and np.std(y_pred) > 0.0:
        spearman = float(spearmanr(y_true, y_pred).statistic)
        pearson = float(pearsonr(y_true, y_pred).statistic)

    return {
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "pearson": pearson,
    }


def build_xgboost_regressor(config: dict[str, Any], random_seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        reg_alpha=float(config["reg_alpha"]),
        reg_lambda=float(config["reg_lambda"]),
        random_state=random_seed,
        objective="reg:squarederror",
        n_jobs=1,
    )


def build_mlp_regressor(config: dict[str, Any], random_seed: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=tuple(int(value) for value in config["hidden_layer_sizes"]),
        activation=str(config["activation"]),
        alpha=float(config["alpha"]),
        batch_size=int(config["batch_size"]),
        learning_rate_init=float(config["learning_rate_init"]),
        max_iter=int(config["max_iter"]),
        early_stopping=bool(config["early_stopping"]),
        validation_fraction=float(config["validation_fraction"]),
        random_state=random_seed,
    )
