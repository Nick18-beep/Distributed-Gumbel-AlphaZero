"""Execution backend contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gumbel_az.config.schema import AppConfig


@dataclass(frozen=True)
class ExecutionResult:
    run_id: str
    run_dir: Path
    status: str


class ExecutionBackend(Protocol):
    name: str

    def run(self, config: AppConfig) -> ExecutionResult:
        """Run or initialize work described by ``config``."""
