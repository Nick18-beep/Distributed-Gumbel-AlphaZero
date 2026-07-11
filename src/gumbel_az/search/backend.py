"""Search backend protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch

from gumbel_az.config.schema import SearchConfig
from gumbel_az.model.common import NetworkOutput
from gumbel_az.search.outputs import SearchOutput


class SearchBackend(Protocol):
    name: str
    device: torch.device

    def search(
        self,
        *,
        root_observation: torch.Tensor,
        root_legal_mask: torch.Tensor,
        network_apply: Callable[[torch.Tensor], NetworkOutput],
        rng: torch.Generator,
        config: SearchConfig,
        root_embedding: Any | None = None,
    ) -> SearchOutput:
        """Run search from root observations."""
