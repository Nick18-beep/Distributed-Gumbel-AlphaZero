"""Append-only JSONL writers for events and metrics."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class JsonlWriter:
    """Small line-buffered JSONL writer with explicit fsync per record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        payload = {"timestamp": _utc_now(), **record}
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


class MetricWriter(JsonlWriter):
    """JSONL writer specialized for scalar metrics."""

    def write_metrics(self, step: int, metrics: dict[str, int | float | str | bool]) -> None:
        self.write({"step": step, "metrics": metrics})
