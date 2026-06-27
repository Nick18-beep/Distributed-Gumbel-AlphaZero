"""Training algorithm protocol."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

from gumbel_az.search.outputs import SearchOutput


class TrainingAlgorithm(Protocol):
    name: str

    def select_action(
        self,
        *,
        game_state: Any,
        network_apply,
        rng: torch.Generator,
        temperature: float,
    ) -> SearchOutput:
        """Select an action from the current game state."""

    def generate_targets(self, trajectory, final_rewards: np.ndarray) -> list[dict]:
        """Convert a completed trajectory into replay targets."""
