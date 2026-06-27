"""Default config paths."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_config_path() -> Path:
    return project_root() / "configs" / "connect_four.yaml"
