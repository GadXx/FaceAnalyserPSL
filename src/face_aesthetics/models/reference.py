from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


MIN_STD_EPSILON = 1e-3
UNSTABLE_STD_THRESHOLD = 1e-2


@dataclass(frozen=True)
class GroupDescriptor:
    combination_name: str
    group_key: str
    group_values: dict[str, str]


def build_group_combinations(available_group_columns: list[str]) -> list[list[str]]:
    combinations_list: list[list[str]] = []
    for size in range(1, len(available_group_columns) + 1):
        for columns in combinations(available_group_columns, size):
            combinations_list.append(list(columns))
    return combinations_list


def group_descriptor(row: dict[str, Any], group_columns: list[str]) -> GroupDescriptor | None:
    group_values: dict[str, str] = {}
    for column in group_columns:
        value = row.get(column)
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            return None
        group_values[column] = normalized

    combination_name = "__".join(group_columns)
    group_key = "__".join(f"{column}={group_values[column]}" for column in group_columns)
    return GroupDescriptor(
        combination_name=combination_name,
        group_key=group_key,
        group_values=group_values,
    )


def compute_z_scores(
    feature_vector: np.ndarray,
    mean_vector: np.ndarray,
    std_vector: np.ndarray,
    *,
    min_std_epsilon: float = MIN_STD_EPSILON,
) -> np.ndarray:
    stabilized_std = np.where(np.abs(std_vector) < min_std_epsilon, min_std_epsilon, std_vector)
    return (feature_vector - mean_vector) / stabilized_std


def unstable_std_mask(std_vector: np.ndarray, *, threshold: float = UNSTABLE_STD_THRESHOLD) -> np.ndarray:
    return np.abs(np.asarray(std_vector, dtype=np.float64)) < float(threshold)


def compute_mahalanobis_distance(
    feature_vector: np.ndarray,
    mean_vector: np.ndarray,
    covariance_matrix: np.ndarray,
    *,
    regularization: float = 1e-6,
) -> float:
    if covariance_matrix.ndim != 2:
        raise ValueError("Covariance matrix must be 2-dimensional.")
    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    centered = feature_vector - mean_vector
    regularized_covariance = covariance_matrix + np.eye(covariance_matrix.shape[0], dtype=np.float64) * regularization
    inverse_covariance = np.linalg.pinv(regularized_covariance)
    distance_squared = float(centered.T @ inverse_covariance @ centered)
    return float(np.sqrt(max(distance_squared, 0.0)))


def percentile_column_name(percentile: float) -> str:
    return f"p{int(round(percentile * 100)):02d}"
