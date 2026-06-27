"""Replay shard reader."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gumbel_az.replay.codec import decode_samples
from gumbel_az.replay.schema import SCHEMA_VERSION
from gumbel_az.replay.validation import validate_sample


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
        return data

    def shard_paths_metadata(self) -> list[dict[str, Any]]:
        return list(self._index().get("shards", []))

    def _quarantine(self, shard: Path) -> Path:
        target = self.quarantine_dir / f"{_utc_stamp()}_{shard.name}"
        shutil.move(str(shard), target)
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
            samples.extend(self.read_shard(Path(entry["path"])))
        return samples
