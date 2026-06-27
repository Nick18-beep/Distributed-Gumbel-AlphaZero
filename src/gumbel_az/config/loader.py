"""YAML loading and resolved-config persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gumbel_az.config.overrides import apply_overrides
from gumbel_az.config.schema import AppConfig
from gumbel_az.storage.atomic import atomic_write_yaml


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def load_config(path: Path, overrides: list[str] | None = None) -> AppConfig:
    data = apply_overrides(read_yaml(path), overrides)
    return AppConfig.model_validate(data)


def save_resolved_config(config: AppConfig, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "config.resolved.yaml"
    data = config.model_dump(mode="json")
    return atomic_write_yaml(output_path, data)
