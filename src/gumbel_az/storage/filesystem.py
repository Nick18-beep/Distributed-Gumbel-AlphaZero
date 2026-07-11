"""Run-directory layout on the local filesystem."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from gumbel_az.config.schema import AppConfig
from gumbel_az.storage.atomic import atomic_write_json


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    logs_dir: Path
    events_path: Path
    metrics_path: Path
    run_state_path: Path
    resolved_config_path: Path


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "run"


def create_run_directory(config: AppConfig, *, base_dir: Path | None = None) -> RunPaths:
    root = (base_dir if base_dir is not None else config.run.output_dir).resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%fZ")
    run_id = f"{timestamp}_{_slugify(config.run.name)}"
    run_dir = root / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=False)

    paths = RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        logs_dir=logs_dir,
        events_path=logs_dir / "events.jsonl",
        metrics_path=logs_dir / "metrics.jsonl",
        run_state_path=run_dir / "run_state.json",
        resolved_config_path=run_dir / "config.resolved.yaml",
    )
    atomic_write_json(
        root / "latest.json",
        {
            "run_id": run_id,
            "run_dir": run_id,
            "created_at": timestamp,
        },
    )
    return paths


def existing_run_paths(run_dir: Path) -> RunPaths:
    run_dir = run_dir.resolve()
    return RunPaths(
        run_id=run_dir.name,
        run_dir=run_dir,
        logs_dir=run_dir / "logs",
        events_path=run_dir / "logs" / "events.jsonl",
        metrics_path=run_dir / "logs" / "metrics.jsonl",
        run_state_path=run_dir / "run_state.json",
        resolved_config_path=run_dir / "config.resolved.yaml",
    )
