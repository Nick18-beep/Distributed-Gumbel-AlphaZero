"""Replay shard writer."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gumbel_az.replay.codec import encode_samples
from gumbel_az.replay.schema import SCHEMA_VERSION
from gumbel_az.replay.validation import validate_sample
from gumbel_az.storage.atomic import atomic_write_json


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "shards": [], "total_samples": 0}
    return json.loads(path.read_text(encoding="utf-8"))


class ReplayWriter:
    """Append-only replay shard writer."""

    def __init__(self, replay_dir: Path) -> None:
        self.replay_dir = replay_dir
        self.shards_dir = replay_dir / "shards"
        self.quarantine_dir = replay_dir / "quarantine"
        self.index_path = replay_dir / "index.json"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            atomic_write_json(
                self.index_path,
                {"schema_version": SCHEMA_VERSION, "shards": [], "total_samples": 0},
            )

    def _next_shard_path(self, index: dict[str, Any]) -> Path:
        shard_id = len(index.get("shards", [])) + 1
        while True:
            candidate = self.shards_dir / f"shard_{shard_id:09d}.msgpack.zst"
            if not candidate.exists():
                return candidate
            shard_id += 1

    def _normalize_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(sample)
        if "schema_version" in normalized and normalized["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported replay sample schema_version {normalized['schema_version']}; "
                f"expected {SCHEMA_VERSION}"
            )
        normalized["schema_version"] = SCHEMA_VERSION
        normalized["timestamp"] = _utc_now()
        validate_sample(normalized)
        return normalized

    def write_shard(self, samples: list[dict[str, Any]]) -> Path:
        if not samples:
            raise ValueError("cannot write an empty replay shard")
        normalized = [self._normalize_sample(sample) for sample in samples]
        payload = encode_samples(normalized)
        index = _load_index(self.index_path)
        if index.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported replay index schema_version {index.get('schema_version')}; "
                f"expected {SCHEMA_VERSION}"
            )
        destination = self._next_shard_path(index)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=self.shards_dir,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, destination)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

        entry = {
            "path": str(destination.resolve()),
            "samples": len(normalized),
            "created_at": _utc_now(),
        }
        index.setdefault("shards", []).append(entry)
        index["total_samples"] = int(index.get("total_samples", 0)) + len(normalized)
        atomic_write_json(self.index_path, index)
        return destination
