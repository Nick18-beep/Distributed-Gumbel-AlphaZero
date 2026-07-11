"""Torch checkpoint registry."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

from gumbel_az.storage.atomic import atomic_write_json


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _entry_for_version(index_path: Path, version: int) -> dict[str, Any]:
    index = _load_json(index_path, {"checkpoints": []})
    for entry in index["checkpoints"]:
        if entry["version"] == version:
            return cast(dict[str, Any], entry)
    raise FileNotFoundError(f"checkpoint version {version} is not registered")


def _resolve_checkpoint_path(root: Path, stored_path: str | Path) -> Path:
    path = Path(stored_path)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"checkpoint path escapes checkpoint root: {stored_path}")
    return resolved


class CheckpointManager:
    """Small registry around torch.save checkpoints."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.latest_path = self.root / "latest.json"
        self.best_path = self.root / "best.json"

    def checkpoint_dir(self, version: int) -> Path:
        return self.root / f"ckpt_{version:06d}"

    def save(
        self,
        *,
        version: int,
        state: Any,
        metadata: dict[str, Any],
        best: bool = False,
    ) -> Path:
        final_dir = self.checkpoint_dir(version)
        staging_dir = self.root / f".ckpt_{version:06d}.tmp"
        if final_dir.exists():
            raise FileExistsError(final_dir)
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)
        payload = {
            "state": state,
            "metadata": {
                "version": version,
                "created_at": _utc_now(),
                **metadata,
            },
        }
        try:
            torch.save(payload, staging_dir / "checkpoint.pt")
            staging_dir.replace(final_dir)
        except BaseException:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise

        entry = {
            "version": version,
            "path": final_dir.name,
            "metadata": payload["metadata"],
        }
        index = _load_json(self.index_path, {"checkpoints": []})
        checkpoints = [item for item in index["checkpoints"] if item["version"] != version]
        checkpoints.append(entry)
        checkpoints.sort(key=lambda item: item["version"])
        atomic_write_json(self.index_path, {"checkpoints": checkpoints})
        atomic_write_json(self.latest_path, entry)
        if best:
            atomic_write_json(self.best_path, entry)
        return final_dir

    def load(
        self,
        version: int | None = None,
        *,
        best: bool = False,
        map_location: str | torch.device = "cpu",
    ) -> Any:
        if best:
            pointer = _load_json(self.best_path, None)
            if pointer is None:
                raise FileNotFoundError(self.best_path)
            checkpoint_dir = _resolve_checkpoint_path(self.root, pointer["path"])
        elif version is None:
            pointer = _load_json(self.latest_path, None)
            if pointer is None:
                raise FileNotFoundError(self.latest_path)
            checkpoint_dir = _resolve_checkpoint_path(self.root, pointer["path"])
        else:
            checkpoint_dir = _resolve_checkpoint_path(
                self.root,
                _entry_for_version(self.index_path, version)["path"],
            )
        checkpoint_file = checkpoint_dir / "checkpoint.pt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(checkpoint_file)
        return torch.load(checkpoint_file, map_location=map_location, weights_only=True)

    def promote(self, version: int) -> None:
        entry = _entry_for_version(self.index_path, version)
        atomic_write_json(self.best_path, entry)
