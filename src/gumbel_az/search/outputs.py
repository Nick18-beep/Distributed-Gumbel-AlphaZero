"""Search output data structures."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax


class SearchOutput(NamedTuple):
    policy_target: jax.Array
    selected_action: jax.Array
    root_value: jax.Array
    visit_counts: jax.Array
    q_values: jax.Array
    prior_logits: jax.Array
    search_metadata: dict[str, Any]
