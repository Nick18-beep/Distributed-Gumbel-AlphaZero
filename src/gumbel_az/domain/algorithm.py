"""Training algorithm protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch

from gumbel_az.search.outputs import SearchOutput

if TYPE_CHECKING:
    from gumbel_az.model.common import NetworkOutput
    from gumbel_az.selfplay.trajectory import Trajectory


class TrainingAlgorithm(Protocol):
    @property
    def name(self) -> str: ...

    def select_action(
        self,
        *,
        game_state: Any,
        network_apply: Callable[[torch.Tensor], NetworkOutput],
        rng: torch.Generator,
        temperature: float,
    ) -> SearchOutput:
        """Select an action from the current game state."""

    def temperature_for_move(self, move_index: int) -> float: ...

    def generate_targets(
        self,
        trajectory: Trajectory,
        final_rewards: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Convert a completed trajectory into replay targets."""
