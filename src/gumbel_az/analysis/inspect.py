"""Inspection helpers for runs, replay and checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gumbel_az.replay import ReplayReader


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    records = []
    errors = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "error": str(exc)})
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                errors.append({"line": line_number, "error": "record is not a JSON object"})
    return records, errors


def inspect_replay(replay_dir: Path) -> dict[str, Any]:
    if not replay_dir.exists():
        raise FileNotFoundError(f"replay directory does not exist: {replay_dir}")
    if not replay_dir.is_dir():
        raise NotADirectoryError(f"replay path is not a directory: {replay_dir}")
    reader = ReplayReader(replay_dir)
    index = _read_json(replay_dir / "index.json", {"shards": [], "total_samples": 0})
    shard_entries = index.get("shards", []) if isinstance(index, dict) else []
    value_count = 0
    value_sum = 0.0
    value_min: float | None = None
    value_max: float | None = None
    illegal_positive = 0
    invalid_shapes = 0
    entropy_count = 0
    entropy_sum = 0.0
    loaded_samples = 0
    missing_shards = 0
    read_errors = []
    for entry in shard_entries:
        if not isinstance(entry, dict) or "path" not in entry:
            read_errors.append({"path": None, "error": "invalid shard index entry"})
            continue
        shard_path = Path(entry["path"])
        if not shard_path.exists():
            missing_shards += 1
            continue
        try:
            samples = reader.read_shard(shard_path)
        except Exception as exc:  # noqa: BLE001 - inspect should report all shard failures.
            read_errors.append({"path": str(shard_path), "error": repr(exc)})
            continue
        loaded_samples += len(samples)
        for sample in samples:
            value = float(sample["value_target"])
            value_count += 1
            value_sum += value
            value_min = value if value_min is None else min(value_min, value)
            value_max = value if value_max is None else max(value_max, value)
            legal = np.asarray(sample["legal_action_mask"], dtype=bool)
            policy = np.asarray(sample["policy_target"], dtype=np.float32)
            if legal.shape != policy.shape:
                invalid_shapes += 1
                continue
            illegal_positive += int(np.count_nonzero(policy[~legal] > 0.0))
            positive = policy[policy > 0.0]
            if positive.size:
                entropy_sum += float(-np.sum(positive * np.log(positive)))
                entropy_count += 1
    return {
        "replay_dir": str(replay_dir.resolve()),
        "shards": len(shard_entries),
        "missing_shards": missing_shards,
        "read_errors": read_errors,
        "total_samples": int(index.get("total_samples", loaded_samples))
        if isinstance(index, dict)
        else loaded_samples,
        "loaded_samples": loaded_samples,
        "illegal_policy_positive_count": illegal_positive,
        "invalid_shape_count": invalid_shapes,
        "policy_entropy_mean": entropy_sum / entropy_count if entropy_count else 0.0,
        "value_target_mean": value_sum / value_count if value_count else 0.0,
        "value_target_min": value_min if value_min is not None else 0.0,
        "value_target_max": value_max if value_max is not None else 0.0,
    }


def inspect_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"checkpoint path is not a directory: {checkpoint_dir}")
    index = _read_json(checkpoint_dir / "index.json", {"checkpoints": []})
    latest = _read_json(checkpoint_dir / "latest.json", None)
    best = _read_json(checkpoint_dir / "best.json", None)
    entries = index.get("checkpoints", []) if isinstance(index, dict) else []
    versions = [
        entry["version"] for entry in entries if isinstance(entry, dict) and "version" in entry
    ]
    return {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "versions": versions,
        "count": len(versions),
        "invalid_entry_count": len(entries) - len(versions),
        "latest_version": latest.get("version") if isinstance(latest, dict) else None,
        "best_version": best.get("version") if isinstance(best, dict) else None,
    }


def inspect_run(run_dir: Path) -> dict[str, Any]:
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run path is not a directory: {run_dir}")
    state = _read_json(run_dir / "run_state.json", {})
    metrics, metric_errors = _read_jsonl(run_dir / "logs" / "metrics.jsonl")
    events, event_errors = _read_jsonl(run_dir / "logs" / "events.jsonl")
    replay_report = inspect_replay(run_dir / "replay") if (run_dir / "replay").exists() else None
    checkpoint_report = (
        inspect_checkpoint(run_dir / "checkpoints") if (run_dir / "checkpoints").exists() else None
    )
    metric_values: dict[str, list[float]] = {}
    for record in metrics:
        for key, value in record.get("metrics", {}).items():
            if isinstance(value, int | float):
                metric_values.setdefault(key, []).append(float(value))
    metric_summary = {
        key: {
            "last": values[-1],
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for key, values in metric_values.items()
        if values
    }
    promotion_history = []
    if isinstance(state.get("eval"), dict):
        promotion_history.append(state["eval"])
    promotion_history.extend(
        record.get("metrics", {})
        for record in metrics
        if "checkpoint_promoted" in record.get("metrics", {})
    )
    eval_matches = [event for event in events if event.get("event") == "eval_match_completed"]
    return {
        "run_dir": str(run_dir.resolve()),
        "state": state,
        "metrics": metric_summary,
        "event_count": len(events),
        "parse_errors": {
            "metrics": metric_errors,
            "events": event_errors,
        },
        "replay": replay_report,
        "checkpoints": checkpoint_report,
        "promotion_history": promotion_history,
        "eval_matches": eval_matches,
    }
