"""Run resume helpers."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gumbel_az.config import load_config
from gumbel_az.config.schema import AppConfig
from gumbel_az.replay.codec import decode_samples
from gumbel_az.replay.schema import SCHEMA_VERSION
from gumbel_az.replay.validation import validate_sample
from gumbel_az.storage.atomic import atomic_write_json


@dataclass(frozen=True)
class ResumeContext:
    run_dir: Path
    run_state: dict[str, Any]
    config: AppConfig
    replay_index: dict[str, Any]
    latest_checkpoint: dict[str, Any] | None
    best_checkpoint: dict[str, Any] | None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_run_state(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "run_state.json"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"invalid run_state.json in {run_dir}")
    return state


def load_resolved_config(run_dir: Path) -> AppConfig:
    return load_config(run_dir / "config.resolved.yaml")


def load_replay_index(run_dir: Path) -> dict[str, Any]:
    index_path = run_dir / "replay" / "index.json"
    if not index_path.exists():
        return {"schema_version": SCHEMA_VERSION, "shards": [], "total_samples": 0}
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "shards" not in data:
        raise ValueError(f"invalid replay index: {index_path}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported replay index schema_version {data.get('schema_version')}; "
            f"expected {SCHEMA_VERSION}"
        )
    seen_paths: set[Path] = set()
    total_samples = 0
    for entry in data["shards"]:
        path = Path(entry["path"]).resolve()
        if path in seen_paths:
            raise ValueError(f"duplicate replay shard in index: {path}")
        if not path.exists():
            raise FileNotFoundError(path)
        seen_paths.add(path)
        total_samples += int(entry.get("samples", 0))
    if int(data.get("total_samples", 0)) != total_samples:
        raise ValueError(
            f"replay index total_samples mismatch: {data.get('total_samples')} != {total_samples}"
        )
    return data


def _load_pointer(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid checkpoint pointer: {path}")
    return data


def _validate_checkpoint_pointer(pointer: dict[str, Any]) -> None:
    checkpoint_dir = Path(pointer["path"])
    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)
    required = ("_CHECKPOINT_METADATA", "_METADATA")
    missing = [name for name in required if not (checkpoint_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"incomplete checkpoint {checkpoint_dir}; missing {', '.join(missing)}"
        )


def load_checkpoint_pointers(run_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    from gumbel_az.model.checkpoint import CheckpointManager

    manager = CheckpointManager(run_dir / "checkpoints")
    latest = _load_pointer(manager.latest_path)
    best = _load_pointer(manager.best_path)
    for pointer in (latest, best):
        if pointer is not None:
            _validate_checkpoint_pointer(pointer)
    return latest, best


def rebuild_replay_index(run_dir: Path) -> dict[str, Any]:
    replay_dir = run_dir / "replay"
    shards_dir = replay_dir / "shards"
    quarantine_dir = replay_dir / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    total_samples = 0
    for shard in sorted(shards_dir.glob("*.msgpack.zst")):
        try:
            samples = decode_samples(shard.read_bytes())
            for sample in samples:
                validate_sample(sample)
        except Exception:
            target = quarantine_dir / f"{_utc_now().replace(':', '')}_{shard.name}"
            shutil.move(str(shard), target)
            continue
        entries.append(
            {
                "path": str(shard.resolve()),
                "samples": len(samples),
                "created_at": _utc_now(),
            }
        )
        total_samples += len(samples)
    index = {
        "schema_version": SCHEMA_VERSION,
        "shards": entries,
        "total_samples": total_samples,
    }
    atomic_write_json(replay_dir / "index.json", index)
    return index


def load_resume_context(run_dir: Path, *, rebuild_replay: bool = False) -> ResumeContext:
    run_dir = run_dir.resolve()
    state = load_run_state(run_dir)
    config = load_resolved_config(run_dir)
    replay_index = rebuild_replay_index(run_dir) if rebuild_replay else load_replay_index(run_dir)
    latest, best = load_checkpoint_pointers(run_dir)
    return ResumeContext(
        run_dir=run_dir,
        run_state=state,
        config=config,
        replay_index=replay_index,
        latest_checkpoint=latest,
        best_checkpoint=best,
    )
