"""Common model output types."""

from __future__ import annotations

from typing import NamedTuple

import jax


class NetworkOutput(NamedTuple):
    policy_logits: jax.Array
    value: jax.Array
