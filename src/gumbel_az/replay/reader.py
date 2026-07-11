"""Replay shard reader."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from gumbel_az.replay.codec import decode_samples
from gumbel_az.replay.schema import SCHEMA_VERSION
from gumbel_az.replay.validation import validate_sample
from gumbel_az.storage.atomic import atomic_write_json


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%fZ")


class ReplayReader:
    """Read replay shards from a local replay directory."""

    def __init__(self, replay_dir: Path) -> None:
        self.replay_dir = replay_dir
        self.index_path = replay_dir / "index.json"
        self.quarantine_dir = replay_dir / "quarantine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def _index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"schema_version": SCHEMA_VERSION, "shards": [], "total_samples": 0}
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported replay index schema_version {data.get('schema_version')}; "
                f"expected {SCHEMA_VERSION}"
            )
        return cast(dict[str, Any], data)

    def shard_paths_metadata(self) -> list[dict[str, Any]]:
        return list(self._index().get("shards", []))

    def resolve_shard_path(self, stored_path: str | Path) -> Path:
        """Resolve an index path while rejecting entries outside the replay tree."""
        path = Path(stored_path)
        resolved = path.resolve() if path.is_absolute() else (self.replay_dir / path).resolve()
        replay_root = self.replay_dir.resolve()
        if not resolved.is_relative_to(replay_root):
            raise ValueError(f"replay shard path escapes replay directory: {stored_path}")
        return resolved

    def _remove_from_index(self, shard: Path) -> None:
        index = self._index()
        retained = []
        removed_samples = 0
        for entry in index.get("shards", []):
            try:
                indexed_path = self.resolve_shard_path(entry["path"])
            except (KeyError, TypeError, ValueError):
                retained.append(entry)
                continue
            if indexed_path == shard.resolve():
                removed_samples += int(entry.get("samples", 0))
            else:
                retained.append(entry)
        if len(retained) != len(index.get("shards", [])):
            index["shards"] = retained
            index["total_samples"] = max(
                0,
                int(index.get("total_samples", 0)) - removed_samples,
            )
            atomic_write_json(self.index_path, index)

    def _quarantine(self, shard: Path) -> Path:
        target = self.quarantine_dir / f"{_utc_stamp()}_{shard.name}"
        shutil.move(str(shard), target)
        self._remove_from_index(shard)
        return target

    def read_shard(self, shard_path: Path) -> list[dict[str, Any]]:
        try:
            samples = decode_samples(shard_path.read_bytes())
            for sample in samples:
                validate_sample(sample)
            return samples
        except Exception:
            if shard_path.exists():
                self._quarantine(shard_path)
            raise

    def read_all(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for entry in self.shard_paths_metadata():
            samples.extend(self.read_shard(self.resolve_shard_path(entry["path"])))
        return samples
