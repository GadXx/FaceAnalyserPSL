from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_FILES = (
    "base.yaml",
    "data.yaml",
    "features.yaml",
    "preprocess.yaml",
    "train_baseline.yaml",
    "train_geometry.yaml",
    "train_fusion.yaml",
    "infer.yaml",
    "robustness.yaml",
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.loads(file.read() or "{}")

    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")

    return payload


def load_named_configs(
    config_dir: Path, file_names: tuple[str, ...] = DEFAULT_CONFIG_FILES
) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}

    for file_name in file_names:
        config_path = config_dir / file_name
        configs[config_path.stem] = load_yaml(config_path)

    return configs


def resolve_project_paths(
    project_root: Path, configured_paths: dict[str, Any]
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}

    for key, value in configured_paths.items():
        if isinstance(value, str):
            resolved[key] = (project_root / value).resolve()

    return resolved
