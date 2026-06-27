"""Replay sample validation."""

from __future__ import annotations

from typing import Any

import numpy as np

from gumbel_az.replay.schema import REQUIRED_FIELDS, SCHEMA_VERSION


def validate_sample(sample: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_FIELDS.difference(sample))
    if missing:
        raise ValueError(f"replay sample missing fields: {', '.join(missing)}")
    if sample.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported replay sample schema_version {sample.get('schema_version')}; "
            f"expected {SCHEMA_VERSION}"
        )

    legal = np.asarray(sample["legal_action_mask"], dtype=np.bool_)
    policy = np.asarray(sample["policy_target"], dtype=np.float32)
    if legal.ndim != 1:
        raise ValueError("legal_action_mask must be a 1D array")
    if policy.shape != legal.shape:
        raise ValueError("policy_target shape must match legal_action_mask")
    if bool(np.any(policy[~legal] > 1.0e-6)):
        raise ValueError("policy_target assigns positive probability to illegal actions")
    if not np.all(np.isfinite(policy)):
        raise ValueError("policy_target contains non-finite values")
    total = float(np.sum(policy))
    if total <= 0.0:
        raise ValueError("policy_target must have positive mass")

    value = float(sample["value_target"])
    if not np.isfinite(value):
        raise ValueError("value_target must be finite")
    if value < -1.0 or value > 1.0:
        raise ValueError("value_target must be in [-1, 1]")
