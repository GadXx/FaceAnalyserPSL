from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

import cv2

from src.face_aesthetics.utils.filesystem import ensure_directory, ensure_parent_directory


JPEG_SIGNATURES = (b"\xff\xd8\xff",)
PNG_SIGNATURES = (b"\x89PNG\r\n\x1a\n",)
BMP_SIGNATURES = (b"BM",)
WEBP_SIGNATURES = (b"RIFF",)


def _is_supported_signature(header: bytes, suffix: str) -> bool:
    normalized = suffix.lower()

    if normalized in {".jpg", ".jpeg"}:
        return any(header.startswith(signature) for signature in JPEG_SIGNATURES)
    if normalized == ".png":
        return any(header.startswith(signature) for signature in PNG_SIGNATURES)
    if normalized == ".bmp":
        return any(header.startswith(signature) for signature in BMP_SIGNATURES)
    if normalized == ".webp":
        return header.startswith(WEBP_SIGNATURES[0]) and header[8:12] == b"WEBP"

    return False


def is_image_readable(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            header = file.read(16)
    except OSError:
        return False

    return _is_supported_signature(header, path.suffix)


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    ensure_parent_directory(path)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    ensure_parent_directory(path)

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet export requires pandas and pyarrow in the active Python environment."
        ) from exc

    frame = pd.DataFrame(rows, columns=fieldnames)
    frame.to_parquet(path, index=False)


def write_parquet_shard(
    rows: list[dict[str, Any]],
    dataset_dir: Path,
    fieldnames: list[str],
    shard_name: str,
) -> Path:
    ensure_directory(dataset_dir)

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet export requires pandas and pyarrow in the active Python environment."
        ) from exc

    shard_path = dataset_dir / shard_name
    frame = pd.DataFrame(rows, columns=fieldnames)
    frame.to_parquet(shard_path, index=False)
    return shard_path


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_parent_directory(path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True, sort_keys=True)


def read_parquet(path: Path, columns: list[str] | None = None) -> list[dict[str, Any]]:
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading parquet requires pandas and pyarrow in the active Python environment."
        ) from exc

    def _read_single_parquet(parquet_path: Path) -> Any:
        table = pq.read_table(parquet_path)
        frame = table.to_pandas()
        if columns is not None:
            for column in columns:
                if column not in frame.columns:
                    frame[column] = None
            frame = frame.loc[:, columns]
        return frame

    if path.is_dir():
        shard_paths = sorted(path.glob("*.parquet"))
        if not shard_paths:
            return []

        frames = [_read_single_parquet(shard_path) for shard_path in shard_paths]
        frame = (
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame(columns=columns)
        )
    else:
        frame = _read_single_parquet(path)
    return frame.to_dict(orient="records")


def iter_parquet_batches(
    path: Path,
    *,
    batch_size: int,
    offset: int = 0,
    limit: int | None = None,
) -> Iterator[list[dict[str, Any]]]:
    try:
        import pyarrow.dataset as ds
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Chunked parquet reading requires pyarrow in the active Python environment."
        ) from exc

    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(batch_size=batch_size)
    yielded_rows = 0
    skipped_rows = 0
    for record_batch in scanner.to_batches():
        rows = record_batch.to_pandas().to_dict(orient="records")
        if offset and skipped_rows < offset:
            drop_count = min(offset - skipped_rows, len(rows))
            skipped_rows += drop_count
            rows = rows[drop_count:]
        if not rows:
            continue

        if limit is not None:
            remaining = limit - yielded_rows
            if remaining <= 0:
                break
            rows = rows[:remaining]

        yield rows
        yielded_rows += len(rows)
        if limit is not None and yielded_rows >= limit:
            break


def load_image(path: Path) -> Any:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def save_image(path: Path, image: Any) -> None:
    ensure_parent_directory(path)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise ValueError(f"Unable to write image: {path}")
