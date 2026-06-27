"""Simple dotted-key overrides for YAML config dictionaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml


def parse_override(override: str) -> tuple[list[str], Any]:
    if "=" not in override:
        raise ValueError(f"override must be KEY=VALUE: {override}")
    key, raw_value = override.split("=", 1)
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError(f"override key is empty: {override}")
    value = yaml.safe_load(raw_value)
    return parts, value


def apply_overrides(data: dict[str, Any], overrides: list[str] | None = None) -> dict[str, Any]:
    result = deepcopy(data)
    for override in overrides or []:
        parts, value = parse_override(override)
        current: dict[str, Any] = result
        for part in parts[:-1]:
            next_value = current.get(part)
            if not isinstance(next_value, dict):
                next_value = {}
                current[part] = next_value
            current = next_value
        current[parts[-1]] = value
    return result
