"""Search backend protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import jax

from gumbel_az.config.schema import SearchConfig
from gumbel_az.model.common import NetworkOutput
from gumbel_az.search.outputs import SearchOutput


class SearchBackend(Protocol):
    name: str

    def search(
        self,
        *,
        root_observation: jax.Array,
        root_legal_mask: jax.Array,
        network_apply: Callable[[jax.Array], NetworkOutput],
        recurrent_fn: Callable[[Any, jax.Array, jax.Array, Any], tuple[Any, Any]],
        rng_key: jax.Array,
        config: SearchConfig,
        root_embedding: Any | None = None,
    ) -> SearchOutput:
        """Run search from root observations."""
