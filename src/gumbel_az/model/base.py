"""Network factory contract."""

from __future__ import annotations

from typing import Protocol

import jax

from gumbel_az.model.common import NetworkOutput


class NetworkFactory(Protocol):
    name: str

    def init(
        self,
        rng_key: jax.Array,
        observation_shape: tuple[int, ...],
        num_actions: int,
    ) -> dict:
        """Initialize network parameters."""

    def apply(self, params: dict, observations: jax.Array, train: bool = False) -> NetworkOutput:
        """Apply network to a batch of observations."""
