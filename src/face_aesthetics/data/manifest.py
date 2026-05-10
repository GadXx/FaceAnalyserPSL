from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.face_aesthetics.data.io import is_image_readable, write_csv, write_json, write_parquet
from src.face_aesthetics.utils.config import load_named_configs, resolve_project_paths
from src.face_aesthetics.utils.filesystem import ensure_directory, iter_files, to_relative_path


MANIFEST_COLUMNS = [
    "dataset_name",
    "sample_id",
    "subject_id",
    "image_path",
    "split",
    "raw_labels",
    "normalized_labels",
    "quality_gate_status",
    "adapter_version",
    "image_id",
]

IMAGE_REF_KEYS = {
    "image",
    "image_id",
    "image_name",
    "image_path",
    "img",
    "img_name",
    "img_path",
    "path",
    "file",
    "file_name",
    "filename",
}
SUBJECT_KEYS = {"subject_id", "subject", "person_id", "identity", "subjectid", "model_id"}
SPLIT_KEYS = {"split", "subset", "partition", "fold", "set"}
UTKFACE_PATTERN = re.compile(
    r"^(?P<age>\d+)_(?P<gender>\d+)_(?P<race>\d+)_(?P<timestamp>[^.]+)"
)
_IMAGE_NAME_INDEX_CACHE: dict[tuple[str, tuple[str, ...]], dict[str, list[Path]]] = {}
DEFAULT_IGNORED_PARTS = {"__macosx", ".ipynb_checkpoints"}


@dataclass
class ManifestArtifacts:
    manifest_csv_path: Path
    manifest_parquet_path: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass
class AdapterContext:
    project_root: Path
    raw_dir: Path
    dataset_root: Path
    adapter_name: str
    adapter_config: dict[str, Any]
    split_aliases: dict[str, str]
    image_extensions: tuple[str, ...]
    ignore_names: set[str]


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(" ", "_")


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _canonical_split(value: Any, aliases: dict[str, str]) -> str | None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None

    normalized = str(cleaned).strip().lower()
    return aliases.get(normalized, normalized)


def _infer_split_from_path(path: Path, split_aliases: dict[str, str]) -> str | None:
    for part in path.parts:
        split = _canonical_split(part, split_aliases)
        if split in {"train", "val", "test"}:
            return split
        stem_tokens = re.split(r"[^a-zA-Z0-9]+", part)
        for token in stem_tokens:
            split = _canonical_split(token, split_aliases)
            if split in {"train", "val", "test"}:
                return split
    return None


def _iter_image_paths(root: Path, image_extensions: tuple[str, ...], ignore_names: set[str]) -> list[Path]:
    return [
        path
        for path in iter_files(root, ["**/*"])
        if _is_relevant_path(path, root, ignore_names) and path.suffix.lower() in image_extensions
    ]


def _is_relevant_path(path: Path, root: Path, ignore_names: set[str]) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        relative_parts = path.parts

    for part in relative_parts:
        lowered = part.lower()
        if part in ignore_names or lowered in DEFAULT_IGNORED_PARTS:
            return False
        if part.startswith(".") and part not in {".", ".."}:
            return False
    return True


def _iter_metadata_paths(root: Path, patterns: list[str], ignore_names: set[str]) -> list[Path]:
    return [
        path
        for path in iter_files(root, patterns)
        if _is_relevant_path(path, root, ignore_names)
    ]


def _parse_delimited_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            reader = csv.DictReader(file, dialect=dialect)
            if reader.fieldnames:
                return [dict(row) for row in reader]
        except csv.Error:
            file.seek(0)

        rows: list[dict[str, Any]] = []
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if not parts:
                continue
            rows.append({"image_path": parts[0]})
        return rows


def _parse_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("items"), list):
            return [item for item in payload["items"] if isinstance(item, dict)]
        if isinstance(payload.get("images"), list):
            return [item for item in payload["images"] if isinstance(item, dict)]
    return []


def _load_structured_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return _parse_json_rows(path)
    return _parse_delimited_rows(path)


def _extract_image_ref(row: dict[str, Any]) -> str | None:
    for key, value in row.items():
        if _normalize_key(key) in IMAGE_REF_KEYS:
            cleaned = _clean_value(value)
            if cleaned is not None:
                return str(cleaned)
    return None


def _extract_subject_id(row: dict[str, Any]) -> str | None:
    for key, value in row.items():
        if _normalize_key(key) in SUBJECT_KEYS:
            cleaned = _clean_value(value)
            if cleaned is not None:
                return str(cleaned)
    return None


def _extract_split(row: dict[str, Any], split_aliases: dict[str, str]) -> str | None:
    for key, value in row.items():
        if _normalize_key(key) in SPLIT_KEYS:
            split = _canonical_split(value, split_aliases)
            if split is not None:
                return split
    return None


def _matches_root_name(root: Path, patterns: list[str]) -> bool:
    normalized = root.name.lower()
    return any(fnmatch.fnmatch(normalized, pattern.lower()) for pattern in patterns)


def _get_image_name_index(
    root: Path,
    image_extensions: tuple[str, ...],
    ignore_names: set[str],
) -> dict[str, list[Path]]:
    cache_key = (str(root.resolve()), image_extensions)
    if cache_key not in _IMAGE_NAME_INDEX_CACHE:
        index: dict[str, list[Path]] = defaultdict(list)
        for image_path in _iter_image_paths(root, image_extensions, ignore_names):
            index[image_path.name].append(image_path.resolve())
        _IMAGE_NAME_INDEX_CACHE[cache_key] = dict(index)
    return _IMAGE_NAME_INDEX_CACHE[cache_key]


def _resolve_image_path(
    image_ref: str,
    root: Path,
    relative_to: Path | None = None,
    *,
    image_name_index: dict[str, list[Path]] | None = None,
) -> Path | None:
    candidates: list[Path] = []
    ref_path = Path(image_ref)
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        if relative_to is not None:
            candidates.append((relative_to / ref_path).resolve())
        candidates.append((root / ref_path).resolve())
        if len(ref_path.parts) == 1 and image_name_index is not None:
            candidates.extend(image_name_index.get(ref_path.name, []))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _build_manifest_row(
    *,
    project_root: Path,
    dataset_name: str,
    sample_id: str,
    image_path: Path,
    split: str | None,
    subject_id: str | None,
    raw_labels: dict[str, Any],
    normalized_labels: dict[str, Any],
    adapter_version: str,
    quality_gate_status: str = "pending_preprocessing",
) -> dict[str, Any]:
    composite_id = f"{dataset_name}::{sample_id}"
    return {
        "dataset_name": dataset_name,
        "sample_id": sample_id,
        "subject_id": subject_id,
        "image_path": to_relative_path(image_path, project_root),
        "split": split,
        "raw_labels": _json_dumps(raw_labels),
        "normalized_labels": _json_dumps(normalized_labels),
        "quality_gate_status": quality_gate_status,
        "adapter_version": adapter_version,
        "image_id": composite_id,
    }


def _derive_sample_id(image_path: Path, dataset_root: Path) -> str:
    return image_path.relative_to(dataset_root).with_suffix("").as_posix().replace("/", "__")


class BaseDatasetAdapter:
    adapter_name = "base"
    adapter_version = "v1"

    def matches(self, context: AdapterContext) -> bool:
        raise NotImplementedError

    def build_rows(self, context: AdapterContext) -> list[dict[str, Any]]:
        raise NotImplementedError


class FairFaceAdapter(BaseDatasetAdapter):
    adapter_name = "fairface"
    adapter_version = "fairface@1"

    def matches(self, context: AdapterContext) -> bool:
        patterns = list(context.adapter_config.get("root_name_patterns", []))
        label_globs = list(context.adapter_config.get("label_globs", []))
        return _matches_root_name(context.dataset_root, patterns) or bool(
            _iter_metadata_paths(context.dataset_root, label_globs, context.ignore_names)
        )

    def build_rows(self, context: AdapterContext) -> list[dict[str, Any]]:
        dataset_name = str(context.adapter_config.get("dataset_name", self.adapter_name))
        label_paths = _iter_metadata_paths(
            context.dataset_root,
            list(context.adapter_config.get("label_globs", [])),
            context.ignore_names,
        )
        rows: list[dict[str, Any]] = []
        image_name_index = _get_image_name_index(
            context.dataset_root,
            context.image_extensions,
            context.ignore_names,
        )

        for label_path in label_paths:
            source_rows = _load_structured_rows(label_path)
            source_split = _infer_split_from_path(label_path, context.split_aliases)
            for raw_row in source_rows:
                image_ref = _extract_image_ref(raw_row)
                if image_ref is None:
                    continue
                image_path = _resolve_image_path(
                    image_ref,
                    context.dataset_root,
                    label_path.parent,
                    image_name_index=image_name_index,
                )
                if image_path is None:
                    continue

                sample_id = _derive_sample_id(image_path, context.dataset_root)
                normalized_labels = {
                    "age_group": _clean_value(raw_row.get("age")),
                    "gender_presentation": _clean_value(raw_row.get("gender")),
                    "ethnicity_group": _clean_value(raw_row.get("race")),
                    "service_test_flag": _clean_value(raw_row.get("service_test")),
                }
                normalized_labels = {
                    key: value for key, value in normalized_labels.items() if value is not None
                }
                split = _extract_split(raw_row, context.split_aliases) or source_split
                rows.append(
                    _build_manifest_row(
                        project_root=context.project_root,
                        dataset_name=dataset_name,
                        sample_id=sample_id,
                        image_path=image_path,
                        split=split,
                        subject_id=_extract_subject_id(raw_row) or image_path.stem,
                        raw_labels={k: v for k, v in raw_row.items() if _clean_value(v) is not None},
                        normalized_labels=normalized_labels,
                        adapter_version=self.adapter_version,
                    )
                )

        return rows


class UTKFaceAdapter(BaseDatasetAdapter):
    adapter_name = "utkface"
    adapter_version = "utkface@1"

    def matches(self, context: AdapterContext) -> bool:
        patterns = list(context.adapter_config.get("root_name_patterns", []))
        if _matches_root_name(context.dataset_root, patterns):
            return True
        return any(UTKFACE_PATTERN.match(path.stem) for path in _iter_image_paths(context.dataset_root, context.image_extensions, context.ignore_names))

    def build_rows(self, context: AdapterContext) -> list[dict[str, Any]]:
        dataset_name = str(context.adapter_config.get("dataset_name", self.adapter_name))
        rows: list[dict[str, Any]] = []

        for image_path in _iter_image_paths(context.dataset_root, context.image_extensions, context.ignore_names):
            match = UTKFACE_PATTERN.match(image_path.stem)
            if match is None:
                continue

            raw_labels = match.groupdict()
            normalized_labels = {
                "age_years": int(match.group("age")),
                "sex_code": int(match.group("gender")),
                "ethnicity_code": int(match.group("race")),
                "capture_token": match.group("timestamp"),
            }
            rows.append(
                    _build_manifest_row(
                        project_root=context.project_root,
                        dataset_name=dataset_name,
                        sample_id=_derive_sample_id(image_path, context.dataset_root),
                        image_path=image_path,
                        split=_infer_split_from_path(image_path, context.split_aliases),
                        subject_id=image_path.stem,
                    raw_labels=raw_labels,
                    normalized_labels=normalized_labels,
                    adapter_version=self.adapter_version,
                )
            )

        return rows


class ChicagoFaceDatabaseAdapter(BaseDatasetAdapter):
    adapter_name = "chicago_face_database"
    adapter_version = "cfd@1"

    def matches(self, context: AdapterContext) -> bool:
        patterns = list(context.adapter_config.get("root_name_patterns", []))
        return _matches_root_name(context.dataset_root, patterns)

    def build_rows(self, context: AdapterContext) -> list[dict[str, Any]]:
        dataset_name = str(context.adapter_config.get("dataset_name", self.adapter_name))
        metadata_paths = _iter_metadata_paths(
            context.dataset_root,
            list(context.adapter_config.get("metadata_globs", [])),
            context.ignore_names,
        )
        rows: list[dict[str, Any]] = []
        image_name_index = _get_image_name_index(
            context.dataset_root,
            context.image_extensions,
            context.ignore_names,
        )

        for metadata_path in metadata_paths:
            for raw_row in _load_structured_rows(metadata_path):
                image_ref = _extract_image_ref(raw_row)
                if image_ref is None:
                    continue
                image_path = _resolve_image_path(
                    image_ref,
                    context.dataset_root,
                    metadata_path.parent,
                    image_name_index=image_name_index,
                )
                if image_path is None:
                    continue

                normalized_labels = {
                    _normalize_key(key): _clean_value(value)
                    for key, value in raw_row.items()
                    if _normalize_key(key) not in IMAGE_REF_KEYS and _clean_value(value) is not None
                }
                rows.append(
                    _build_manifest_row(
                        project_root=context.project_root,
                        dataset_name=dataset_name,
                        sample_id=_derive_sample_id(image_path, context.dataset_root),
                        image_path=image_path,
                        split=_extract_split(raw_row, context.split_aliases) or _infer_split_from_path(image_path, context.split_aliases),
                        subject_id=_extract_subject_id(raw_row) or image_path.stem,
                        raw_labels={k: v for k, v in raw_row.items() if _clean_value(v) is not None},
                        normalized_labels=normalized_labels,
                        adapter_version=self.adapter_version,
                    )
                )

        return rows


class SCUTCompatibilityAdapter(BaseDatasetAdapter):
    adapter_name = "scut_compat"
    adapter_version = "scut_compat@1"

    def matches(self, context: AdapterContext) -> bool:
        patterns = list(context.adapter_config.get("root_name_patterns", []))
        return _matches_root_name(context.dataset_root, patterns)

    def build_rows(self, context: AdapterContext) -> list[dict[str, Any]]:
        dataset_name = str(context.adapter_config.get("dataset_name", self.adapter_name))
        metadata_paths = _iter_metadata_paths(
            context.dataset_root,
            list(context.adapter_config.get("metadata_globs", [])),
            context.ignore_names,
        )
        rows: list[dict[str, Any]] = []
        image_name_index = _get_image_name_index(
            context.dataset_root,
            context.image_extensions,
            context.ignore_names,
        )

        for metadata_path in metadata_paths:
            for raw_row in _load_structured_rows(metadata_path):
                image_ref = _extract_image_ref(raw_row)
                if image_ref is None:
                    continue
                image_path = _resolve_image_path(
                    image_ref,
                    context.dataset_root,
                    metadata_path.parent,
                    image_name_index=image_name_index,
                )
                if image_path is None:
                    continue

                normalized_labels = {
                    "legacy_dataset": dataset_name,
                    "legacy_subjective_annotations_preserved_raw_only": True,
                }
                split = _extract_split(raw_row, context.split_aliases) or _infer_split_from_path(image_path, context.split_aliases)
                rows.append(
                    _build_manifest_row(
                        project_root=context.project_root,
                        dataset_name=dataset_name,
                        sample_id=_derive_sample_id(image_path, context.dataset_root),
                        image_path=image_path,
                        split=split,
                        subject_id=_extract_subject_id(raw_row) or image_path.stem,
                        raw_labels={k: v for k, v in raw_row.items() if _clean_value(v) is not None},
                        normalized_labels=normalized_labels,
                        adapter_version=self.adapter_version,
                    )
                )

        return rows


class GenericImageFolderAdapter(BaseDatasetAdapter):
    adapter_name = "generic_image_folder"
    adapter_version = "generic_image_folder@1"

    def matches(self, context: AdapterContext) -> bool:
        return bool(_iter_image_paths(context.dataset_root, context.image_extensions, context.ignore_names))

    def build_rows(self, context: AdapterContext) -> list[dict[str, Any]]:
        dataset_name = str(context.adapter_config.get("dataset_name", context.dataset_root.name or self.adapter_name))
        rows: list[dict[str, Any]] = []

        for image_path in _iter_image_paths(context.dataset_root, context.image_extensions, context.ignore_names):
            rows.append(
                _build_manifest_row(
                    project_root=context.project_root,
                    dataset_name=dataset_name,
                    sample_id=_derive_sample_id(image_path, context.dataset_root),
                    image_path=image_path,
                    split=_infer_split_from_path(image_path, context.split_aliases),
                    subject_id=None,
                    raw_labels={},
                    normalized_labels={},
                    adapter_version=self.adapter_version,
                )
            )

        return rows


ADAPTER_REGISTRY: dict[str, BaseDatasetAdapter] = {
    "fairface": FairFaceAdapter(),
    "utkface": UTKFaceAdapter(),
    "chicago_face_database": ChicagoFaceDatabaseAdapter(),
    "scut_compat": SCUTCompatibilityAdapter(),
    "generic_image_folder": GenericImageFolderAdapter(),
}


def _candidate_dataset_roots(raw_dir: Path, ignore_names: set[str]) -> list[Path]:
    roots = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_dir() and _is_relevant_path(path, raw_dir, ignore_names)
    )
    has_top_level_files = any(
        path.is_file() and _is_relevant_path(path, raw_dir, ignore_names) for path in raw_dir.iterdir()
    )
    if has_top_level_files or not roots:
        roots.append(raw_dir)
    return roots


def _collect_adapter_rows(
    *,
    project_root: Path,
    raw_dir: Path,
    dataset_config: dict[str, Any],
    io_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    split_aliases = dataset_config.get("split_aliases", {})
    ignore_names = set(dataset_config.get("ignore_names", []))
    image_extensions = tuple(
        extension.lower() for extension in io_config.get("allowed_image_extensions", [])
    )
    adapter_order = list(dataset_config.get("adapter_order", []))
    adapter_configs = dict(dataset_config.get("adapters", {}))
    roots = _candidate_dataset_roots(raw_dir, ignore_names)

    rows: list[dict[str, Any]] = []
    claimed_images: set[str] = set()
    adapter_counts: dict[str, int] = defaultdict(int)
    used_roots: set[Path] = set()

    for adapter_name in adapter_order:
        if adapter_name == "generic_image_folder":
            continue
        adapter = ADAPTER_REGISTRY[adapter_name]
        adapter_config = adapter_configs.get(adapter_name, {})
        if not adapter_config.get("enabled", True):
            continue

        for dataset_root in roots:
            context = AdapterContext(
                project_root=project_root,
                raw_dir=raw_dir,
                dataset_root=dataset_root,
                adapter_name=adapter_name,
                adapter_config=adapter_config,
                split_aliases=split_aliases,
                image_extensions=image_extensions,
                ignore_names=ignore_names,
            )
            if not adapter.matches(context):
                continue

            adapter_rows = adapter.build_rows(context)
            if not adapter_rows:
                continue

            filtered_rows = []
            for row in adapter_rows:
                image_key = row["image_path"]
                if image_key in claimed_images:
                    continue
                claimed_images.add(image_key)
                filtered_rows.append(row)

            if filtered_rows:
                rows.extend(filtered_rows)
                adapter_counts[adapter_name] += len(filtered_rows)
                used_roots.add(dataset_root)

    generic_config = adapter_configs.get("generic_image_folder", {})
    if generic_config.get("enabled", True):
        adapter = ADAPTER_REGISTRY["generic_image_folder"]
        for dataset_root in roots:
            if dataset_root == raw_dir and any(path.is_dir() for path in raw_dir.iterdir()):
                continue
            if dataset_root in used_roots and dataset_root != raw_dir:
                continue

            dataset_name = (
                generic_config.get("dataset_name_map", {}).get(dataset_root.name)
                or dataset_root.name
                or "generic_image_folder"
            )
            context = AdapterContext(
                project_root=project_root,
                raw_dir=raw_dir,
                dataset_root=dataset_root,
                adapter_name="generic_image_folder",
                adapter_config={**generic_config, "dataset_name": dataset_name},
                split_aliases=split_aliases,
                image_extensions=image_extensions,
                ignore_names=ignore_names,
            )
            if not adapter.matches(context):
                continue

            for row in adapter.build_rows(context):
                if row["image_path"] in claimed_images:
                    continue
                claimed_images.add(row["image_path"])
                rows.append(row)
                adapter_counts["generic_image_folder"] += 1

    return rows, dict(sorted(adapter_counts.items()))


def collect_dataset_manifest_rows(
    *,
    project_root: Path,
    raw_dir: Path,
    dataset_config: dict[str, Any],
    io_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    return _collect_adapter_rows(
        project_root=project_root,
        raw_dir=raw_dir,
        dataset_config=dataset_config,
        io_config=io_config,
    )


def _stable_fraction(sample_key: str, seed: str) -> float:
    digest = hashlib.md5(f"{seed}:{sample_key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _assign_split_by_fraction(
    fraction: float,
    *,
    train_fraction: float,
    val_fraction: float,
) -> str:
    if fraction < train_fraction:
        return "train"
    if fraction < train_fraction + val_fraction:
        return "val"
    return "test"


def _apply_deterministic_splits(
    rows: list[dict[str, Any]],
    split_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not split_config.get("enabled", True):
        return rows

    seed = str(split_config.get("seed", "manifest_v1"))
    default_train_fraction = float(split_config.get("default_train_fraction", 0.8))
    default_val_fraction = float(split_config.get("default_val_fraction", 0.1))
    missing_test_from_train_fraction = float(
        split_config.get("missing_test_from_train_fraction", 0.1)
    )

    rows_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_dataset[str(row["dataset_name"])].append(row)

    for dataset_rows in rows_by_dataset.values():
        split_counts = Counter(
            str(row.get("split"))
            for row in dataset_rows
            if row.get("split") in {"train", "val", "test"}
        )

        if split_counts.get("train", 0) == 0 and split_counts.get("val", 0) == 0 and split_counts.get("test", 0) == 0:
            for row in dataset_rows:
                fraction = _stable_fraction(str(row["image_id"]), seed)
                row["split"] = _assign_split_by_fraction(
                    fraction,
                    train_fraction=default_train_fraction,
                    val_fraction=default_val_fraction,
                )
            continue

        if split_counts.get("test", 0) == 0 and split_counts.get("train", 0) > 0:
            for row in dataset_rows:
                if row.get("split") != "train":
                    continue
                fraction = _stable_fraction(str(row["image_id"]), seed)
                if fraction >= 1.0 - missing_test_from_train_fraction:
                    row["split"] = "test"

    return rows


def _summarize(
    rows: list[dict[str, Any]],
    *,
    broken_files: list[str],
    duplicate_image_ids: list[str],
    empty_paths: int,
    adapter_counts: dict[str, int],
) -> dict[str, Any]:
    split_counts = Counter()
    dataset_counts = Counter()
    missing_counts = Counter()

    for row in rows:
        split = row.get("split")
        if split in {"train", "val", "test"}:
            split_counts[split] += 1
        dataset_counts[str(row.get("dataset_name"))] += 1

        for field in MANIFEST_COLUMNS:
            if row.get(field) in (None, ""):
                missing_counts[field] += 1

    return {
        "total_rows": len(rows),
        "split_counts": {
            "train": split_counts.get("train", 0),
            "val": split_counts.get("val", 0),
            "test": split_counts.get("test", 0),
        },
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "adapter_counts": adapter_counts,
        "missing_counts": dict(sorted(missing_counts.items())),
        "validation": {
            "empty_paths": empty_paths,
            "broken_files": len(broken_files),
            "broken_file_paths": broken_files,
            "duplicate_image_ids": len(duplicate_image_ids),
            "duplicate_image_id_values": duplicate_image_ids,
        },
    }


def build_dataset_manifest(project_root: Path | None = None) -> ManifestArtifacts:
    root = (project_root or Path.cwd()).resolve()
    configs = load_named_configs(root / "configs")
    base_paths = resolve_project_paths(root, configs["base"].get("paths", {}))
    data_config = configs["data"]

    raw_dir = (root / data_config["dataset"]["raw_input_dir"]).resolve()
    manifest_csv_path = (root / data_config["dataset"]["manifest_csv_path"]).resolve()
    manifest_parquet_path = (root / data_config["dataset"]["interim_manifest_path"]).resolve()
    summary_path = (root / data_config["dataset"]["summary_path"]).resolve()

    ensure_directory(raw_dir)

    rows, adapter_counts = collect_dataset_manifest_rows(
        project_root=root,
        raw_dir=raw_dir,
        dataset_config=data_config["dataset"],
        io_config=data_config["io"],
    )
    rows = _apply_deterministic_splits(
        rows,
        data_config["dataset"].get("deterministic_split", {}),
    )

    broken_files: list[str] = []
    empty_paths = 0
    for row in rows:
        image_path_value = row.get("image_path")
        if not image_path_value:
            empty_paths += 1
            continue
        absolute_path = root / str(image_path_value)
        if not absolute_path.exists() or not is_image_readable(absolute_path):
            broken_files.append(str(image_path_value))

    image_id_counts = Counter(str(row["image_id"]) for row in rows if row.get("image_id"))
    duplicate_image_ids = sorted(
        [image_id for image_id, count in image_id_counts.items() if count > 1]
    )

    if duplicate_image_ids:
        raise ValueError(f"Duplicate image_id values detected: {duplicate_image_ids}")
    if empty_paths:
        raise ValueError(f"Empty image_path values detected: {empty_paths}")

    extra_fields = sorted({key for row in rows for key in row.keys()} - set(MANIFEST_COLUMNS))
    fieldnames = MANIFEST_COLUMNS + extra_fields

    write_csv(rows, manifest_csv_path, fieldnames)
    write_parquet(rows, manifest_parquet_path, fieldnames)

    summary = _summarize(
        rows,
        broken_files=broken_files,
        duplicate_image_ids=duplicate_image_ids,
        empty_paths=empty_paths,
        adapter_counts=adapter_counts,
    )
    summary["paths"] = {
        "raw_input_dir": raw_dir.relative_to(root).as_posix(),
        "manifest_csv_path": manifest_csv_path.relative_to(root).as_posix(),
        "manifest_parquet_path": manifest_parquet_path.relative_to(root).as_posix(),
        "summary_path": summary_path.relative_to(root).as_posix(),
    }
    summary["dataset_root_exists"] = raw_dir.exists()
    summary["project_paths"] = {
        key: value.relative_to(root).as_posix() if value != root else "."
        for key, value in base_paths.items()
    }
    write_json(summary, summary_path)

    return ManifestArtifacts(
        manifest_csv_path=manifest_csv_path,
        manifest_parquet_path=manifest_parquet_path,
        summary_path=summary_path,
        summary=summary,
    )
