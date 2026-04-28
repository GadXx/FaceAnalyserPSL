from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent_directory(path: Path) -> None:
    ensure_directory(path.parent)


def iter_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []

    for pattern in patterns:
        files.extend(path for path in root.glob(pattern) if path.is_file())

    return sorted(set(files))


def to_relative_path(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()
