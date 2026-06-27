"""Self-play trajectory data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax


@dataclass(frozen=True)
class TrajectoryStep:
    observation: jax.Array
    legal_action_mask: jax.Array
    policy_target: jax.Array
    action: int
    root_value: float
    to_play: int
    move_index: int
    search_stats: dict[str, Any]


@dataclass(frozen=True)
class Trajectory:
    game_id: str
    game_name: str
    algorithm_name: str
    model_version: int
    steps: list[TrajectoryStep]
    final_rewards: jax.Array
