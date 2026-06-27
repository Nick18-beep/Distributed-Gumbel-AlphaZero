"""Orbax-backed checkpoint registry."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp

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
            return entry
    raise FileNotFoundError(f"checkpoint version {version} is not registered")


class CheckpointManager:
    """Small registry around Orbax PyTree checkpoints."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.latest_path = self.root / "latest.json"
        self.best_path = self.root / "best.json"
        self._checkpointer = ocp.PyTreeCheckpointer()

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

        payload = {
            "state": state,
            "metadata": {
                "version": version,
                "created_at": _utc_now(),
                **metadata,
            },
        }
        try:
            self._checkpointer.save(staging_dir, payload)
            staging_dir.replace(final_dir)
        except BaseException:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise

        entry = {
            "version": version,
            "path": str(final_dir.resolve()),
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

    def load(self, version: int | None = None, *, best: bool = False) -> Any:
        if best:
            pointer = _load_json(self.best_path, None)
            if pointer is None:
                raise FileNotFoundError(self.best_path)
            checkpoint_dir = Path(pointer["path"])
        elif version is None:
            pointer = _load_json(self.latest_path, None)
            if pointer is None:
                raise FileNotFoundError(self.latest_path)
            checkpoint_dir = Path(pointer["path"])
        else:
            checkpoint_dir = Path(_entry_for_version(self.index_path, version)["path"])

        if not checkpoint_dir.exists():
            raise FileNotFoundError(checkpoint_dir)
        return self._checkpointer.restore(checkpoint_dir)

    def promote(self, version: int) -> None:
        entry = _entry_for_version(self.index_path, version)
        atomic_write_json(self.best_path, entry)
