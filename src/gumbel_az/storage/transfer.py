"""LAN transfer helpers for replay shards and checkpoints."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from gumbel_az.replay.codec import decode_samples
from gumbel_az.replay.schema import SCHEMA_VERSION
from gumbel_az.replay.validation import validate_sample
from gumbel_az.storage.atomic import atomic_write_json


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _file_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%fZ")


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return safe.strip("._") or "item"


def _copy_file_atomic(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as output:
            with source.open("rb") as input_stream:
                shutil.copyfileobj(input_stream, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, destination)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return destination


def _copy_file_to_temp(source: Path, directory: Path, *, prefix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as output:
            with source.open("rb") as input_stream:
                shutil.copyfileobj(input_stream, output)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ReplayImportResult:
    imported: bool
    path: Path
    samples: int
    quarantined_path: Path | None = None
    error: str | None = None


class ReplayTransfer:
    """Validate and atomically import replay shards uploaded by LAN workers."""

    def __init__(self, replay_dir: Path) -> None:
        self.replay_dir = replay_dir
        self.shards_dir = replay_dir / "shards"
        self.quarantine_dir = replay_dir / "quarantine"
        self.index_path = replay_dir / "index.json"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def validate_shard(self, shard_path: Path) -> int:
        samples = decode_samples(shard_path.read_bytes())
        if not samples:
            raise ValueError("replay shard is empty")
        for sample in samples:
            validate_sample(sample)
        return len(samples)

    def _next_shard_path(self, index: dict) -> Path:
        shard_id = len(index["shards"]) + 1
        while True:
            candidate = self.shards_dir / f"shard_{shard_id:09d}.msgpack.zst"
            if not candidate.exists():
                return candidate
            shard_id += 1

    def import_shard(self, source: Path, *, worker_id: str = "worker") -> ReplayImportResult:
        timestamp = _utc_now()
        safe_worker_id = _safe_component(worker_id)
        if not source.exists():
            return ReplayImportResult(
                imported=False,
                path=source,
                samples=0,
                error=f"source shard does not exist: {source}",
            )
        staging = _copy_file_to_temp(source, self.shards_dir, prefix=".incoming.")
        try:
            sample_count = self.validate_shard(staging)
        except Exception as exc:
            quarantine_path = (
                self.quarantine_dir
                / f"{_file_stamp()}_{safe_worker_id}_{_safe_component(source.name)}"
            )
            os.replace(staging, quarantine_path)
            return ReplayImportResult(
                imported=False,
                path=source,
                samples=0,
                quarantined_path=quarantine_path,
                error=repr(exc),
            )

        index = _load_json(
            self.index_path,
            {"schema_version": SCHEMA_VERSION, "shards": [], "total_samples": 0},
        )
        destination = self._next_shard_path(index)
        try:
            os.replace(staging, destination)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
        entry = {
            "path": str(destination.resolve()),
            "samples": sample_count,
            "created_at": timestamp,
            "uploaded_by": worker_id,
        }
        index["shards"].append(entry)
        index["total_samples"] = int(index.get("total_samples", 0)) + sample_count
        atomic_write_json(self.index_path, index)
        return ReplayImportResult(imported=True, path=destination, samples=sample_count)


class CheckpointSync:
    """Atomic checkpoint tree download/copy helper for LAN workers."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def sync_pointer(self, source_root: Path, pointer: str, destination_root: Path) -> Path:
        if pointer not in {"latest", "best"}:
            raise ValueError("pointer must be 'latest' or 'best'")
        pointer_path = source_root / f"{pointer}.json"
        metadata = _load_json(pointer_path, {})
        if "path" not in metadata:
            raise FileNotFoundError(pointer_path)
        source_checkpoint = Path(metadata["path"])
        if not source_checkpoint.exists():
            raise FileNotFoundError(source_checkpoint)
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / source_checkpoint.name
        staging = destination_root / f".{source_checkpoint.name}.tmp"
        if destination.exists():
            if not destination.is_dir():
                raise NotADirectoryError(destination)
        else:
            if staging.exists():
                shutil.rmtree(staging)
            try:
                shutil.copytree(source_checkpoint, staging)
                staging.replace(destination)
            except BaseException:
                if staging.exists():
                    shutil.rmtree(staging)
                raise
        local_metadata = {**metadata, "path": str(destination.resolve())}
        atomic_write_json(destination_root / f"{pointer}.json", local_metadata)
        return destination
