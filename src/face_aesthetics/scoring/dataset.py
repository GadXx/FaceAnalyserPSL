from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.face_aesthetics.data.io import write_parquet
from src.face_aesthetics.scoring.common import load_scoring_configs, save_json
from src.face_aesthetics.utils.filesystem import iter_files


@dataclass
class ScoringManifestArtifacts:
    manifest_path: Path
    summary_path: Path
    summary: dict[str, Any]


def _parse_score_folder_name(name: str) -> float:
    try:
        return float(name)
    except ValueError as exc:
        raise ValueError(f"Scoring dataset folder name must be numeric: {name}") from exc


def _distribute_group_splits(group_rows: list[dict[str, Any]], ratios: dict[str, float], seed: int) -> list[dict[str, Any]]:
    import random

    rng = random.Random(seed)
    rows = list(group_rows)
    rng.shuffle(rows)
    count = len(rows)
    train_ratio = float(ratios.get("train", 0.7))
    val_ratio = float(ratios.get("val", 0.15))
    use_test_split = "test" in ratios and float(ratios.get("test", 0.0)) > 0.0
    test_ratio = float(ratios.get("test", 0.0))

    train_count = int(round(count * train_ratio))
    val_count = int(round(count * val_ratio))
    test_count = int(round(count * test_ratio)) if use_test_split else 0

    if count == 1:
        train_count, val_count, test_count = 1, 0, 0
    elif count == 2:
        train_count, val_count, test_count = 1, 1, 0
    else:
        if val_count == 0:
            val_count = 1
        if use_test_split and test_count == 0:
            test_count = 1
        train_count = count - val_count - test_count
        if train_count <= 0:
            train_count = 1
            if use_test_split and test_count > 1:
                test_count -= 1
            elif val_count > 1:
                val_count -= 1

    while train_count + val_count + test_count > count:
        if use_test_split and test_count > 1:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        elif train_count > 1:
            train_count -= 1
        else:
            break

    while train_count + val_count + test_count < count:
        train_count += 1

    splits = ["train"] * train_count + ["val"] * val_count
    if use_test_split:
        splits.extend(["test"] * test_count)
    if len(splits) < count:
        splits.extend(["train"] * (count - len(splits)))

    for row, split in zip(rows, splits):
        row["split"] = split
    return rows


def build_scoring_manifest(
    *,
    project_root: Path | None = None,
    dataset_root: Path | None = None,
) -> ScoringManifestArtifacts:
    root = (project_root or Path.cwd()).resolve()
    configs = load_scoring_configs(root)
    scoring_config = configs["scoring"]["scoring"]

    dataset_dir = (dataset_root or (root / str(scoring_config["dataset_root"]))).resolve()
    manifest_path = (root / str(scoring_config["manifest_path"])).resolve()
    summary_path = (root / str(scoring_config["summary_paths"]["dataset_summary_path"])).resolve()

    allowed_extensions = tuple(str(value).lower() for value in scoring_config["allowed_image_extensions"])
    ratios = dict(scoring_config["split_ratios"])
    seed = int(scoring_config["random_seed"])

    rows: list[dict[str, Any]] = []
    per_group_rows: dict[str, list[dict[str, Any]]] = {}
    group_counts: dict[str, int] = {}

    for score_dir in sorted([path for path in dataset_dir.iterdir() if path.is_dir()], key=lambda path: float(path.name)):
        group_label = score_dir.name
        target_score = _parse_score_folder_name(group_label)
        files = [path for path in iter_files(score_dir, ["*"]) if path.suffix.lower() in allowed_extensions]
        group_counts[group_label] = len(files)
        group_rows: list[dict[str, Any]] = []
        for index, image_path in enumerate(files, start=1):
            relative_path = image_path.relative_to(root).as_posix() if image_path.is_relative_to(root) else str(image_path)
            file_stem = image_path.stem
            image_id = f"scoring::{group_label}::{file_stem}::{index:04d}"
            group_rows.append(
                {
                    "image_id": image_id,
                    "image_path": relative_path,
                    "file_name": image_path.name,
                    "source_score_folder": group_label,
                    "target_score": float(target_score),
                    "target_group_label": group_label,
                    "split": None,
                }
            )
        per_group_rows[group_label] = _distribute_group_splits(group_rows, ratios, seed + int(round(target_score * 100)))
        rows.extend(per_group_rows[group_label])

    fieldnames = [
        "image_id",
        "image_path",
        "file_name",
        "source_score_folder",
        "target_score",
        "target_group_label",
        "split",
    ]
    write_parquet(rows, manifest_path, fieldnames)

    split_counts = {"train": 0, "val": 0}
    if "test" in ratios:
        split_counts["test"] = 0
    for row in rows:
        split_counts[str(row["split"])] = split_counts.get(str(row["split"]), 0) + 1
    low_support_groups = {
        group: count
        for group, count in group_counts.items()
        if count < int(scoring_config["minimum_samples_per_group_warning"])
    }
    summary = {
        "status": "completed",
        "dataset_root": dataset_dir.relative_to(root).as_posix() if dataset_dir.is_relative_to(root) else str(dataset_dir),
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "rows": len(rows),
        "group_count": len(group_counts),
        "group_counts": group_counts,
        "split_counts": split_counts,
        "low_support_groups": low_support_groups,
        "warnings": (
            ["Some score groups have low sample counts; regression and group probabilities may be unstable."]
            if low_support_groups
            else []
        ),
    }
    save_json(summary, summary_path)
    return ScoringManifestArtifacts(
        manifest_path=manifest_path,
        summary_path=summary_path,
        summary=summary,
    )
