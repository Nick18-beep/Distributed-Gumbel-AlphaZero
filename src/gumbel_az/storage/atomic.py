"""Cross-platform atomic file writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(path: Path, data: Any) -> Path:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return atomic_write_text(path, text)


def atomic_write_yaml(path: Path, data: Any) -> Path:
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    return atomic_write_text(path, text)
