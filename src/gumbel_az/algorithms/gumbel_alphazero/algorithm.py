"""Gumbel AlphaZero algorithm glue around PyTorch search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from gumbel_az.config.schema import SearchConfig
from gumbel_az.domain.game import GameAdapter
from gumbel_az.model.common import NetworkOutput
from gumbel_az.search.backend import SearchBackend
from gumbel_az.search.outputs import SearchOutput
from gumbel_az.selfplay.trajectory import Trajectory


def _unbatch_output(output: SearchOutput) -> SearchOutput:
    return SearchOutput(
        policy_target=output.policy_target[0],
        selected_action=output.selected_action[0],
        root_value=output.root_value[0],
        visit_counts=output.visit_counts[0],
        q_values=output.q_values[0],
        prior_logits=output.prior_logits[0],
        search_metadata=output.search_metadata,
    )


@dataclass(frozen=True)
class GumbelAlphaZeroAlgorithm:
    game: GameAdapter
    search_backend: SearchBackend
    search_config: SearchConfig
    temperature_moves: int
    name: str = "gumbel_alphazero"

    def select_action(
        self,
        *,
        game_state: Any,
        network_apply: Callable[[torch.Tensor], NetworkOutput],
        rng: torch.Generator,
        temperature: float = 0.0,
    ) -> SearchOutput:
        observation = torch.as_tensor(
            self.game.canonical_observation(game_state)[None, ...],
            dtype=torch.float32,
            device=self.search_backend.device,
        )
        legal_mask = torch.as_tensor(
            self.game.legal_action_mask(game_state)[None, ...],
            dtype=torch.bool,
            device=self.search_backend.device,
        )
        output = self.search_backend.search(
            root_observation=observation,
            root_legal_mask=legal_mask,
            network_apply=network_apply,
            rng=rng,
            config=self.search_config,
            root_embedding=[game_state],
        )
        unbatched = _unbatch_output(output)
        if temperature <= 0.0:
            return unbatched
        policy = unbatched.policy_target.clamp_min(0.0)
        policy = policy / policy.sum().clamp_min(1.0e-8)
        sampled = torch.multinomial(policy, num_samples=1, generator=rng).squeeze(0)
        return unbatched._replace(selected_action=sampled)

    def temperature_for_move(self, move_index: int) -> float:
        return 1.0 if move_index < self.temperature_moves else 0.0

    def generate_targets(
        self,
        trajectory: Trajectory,
        final_rewards: np.ndarray,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for step in trajectory.steps:
            value_target = float(final_rewards[step.to_play])
            samples.append(
                {
                    "game_name": trajectory.game_name,
                    "algorithm_name": self.name,
                    "state_or_observation": step.observation,
                    "legal_action_mask": step.legal_action_mask,
                    "policy_target": step.policy_target,
                    "value_target": value_target,
                    "to_play": step.to_play,
                    "move_index": step.move_index,
                    "game_id": trajectory.game_id,
                    "model_version": trajectory.model_version,
                    "search_stats": step.search_stats,
                }
            )
        return samples
