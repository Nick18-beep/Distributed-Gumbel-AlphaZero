"""Self-play worker using Gumbel AlphaZero search."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import uuid4

import jax
import numpy as np

from gumbel_az.algorithms import create_algorithm
from gumbel_az.config.schema import AppConfig
from gumbel_az.envs import create_game
from gumbel_az.model import create_network
from gumbel_az.replay import ReplayWriter
from gumbel_az.search import MctxSearchBackend
from gumbel_az.selfplay.trajectory import Trajectory, TrajectoryStep


@dataclass(frozen=True)
class SelfPlayResult:
    games: int
    positions: int
    illegal_action_rate: float
    policy_entropy_mean: float
    root_value_mean: float
    replay_shard: str
    seconds: float

    @property
    def games_per_sec(self) -> float:
        return self.games / max(self.seconds, 1.0e-9)

    @property
    def positions_per_sec(self) -> float:
        return self.positions / max(self.seconds, 1.0e-9)


def _entropy(policy: jax.Array) -> float:
    probs = np.asarray(policy, dtype=np.float32)
    positive = probs[probs > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


class SelfPlayWorker:
    def __init__(
        self,
        config: AppConfig,
        *,
        replay_writer: ReplayWriter,
        params: Any | None = None,
    ) -> None:
        self.config = config
        self.game = create_game(config.game.name)
        self.network = create_network(config.model, num_actions=self.game.num_actions)
        self.params = params
        if self.params is None:
            self.params = self.network.init(
                jax.random.PRNGKey(config.run.seed),
                self.game.observation_shape,
                self.game.num_actions,
            )
        self.algorithm = create_algorithm(
            config,
            game=self.game,
            search_backend=MctxSearchBackend(),
        )
        self.replay_writer = replay_writer
        self.model_version = 0
        self._select_fn = jax.jit(self._select_action)

    def _network_apply(self, params: Any, observations: jax.Array):
        return self.network.apply(params, observations, train=False)

    def _select_action(self, params, state, rng_key, temperature):
        def network_apply(observations: jax.Array):
            return self._network_apply(params, observations)

        return self.algorithm.select_action(
            game_state=state,
            network_apply=network_apply,
            rng_key=rng_key,
            temperature=temperature,
        )

    def play_game(self, rng_key: jax.Array, *, game_id: str | None = None) -> Trajectory:
        state = self.game.init(rng_key)
        steps: list[TrajectoryStep] = []
        game_id = game_id or uuid4().hex
        illegal_actions = 0

        for move_index in range(self.game.max_moves):
            if bool(self.game.is_terminal(state)):
                break
            key, rng_key = jax.random.split(rng_key)
            observation = self.game.canonical_observation(state)
            legal_mask = self.game.legal_action_mask(state)
            to_play = int(self.game.current_player(state))
            temperature = self.algorithm.temperature_for_move(move_index)
            output = self._select_fn(self.params, state, key, temperature)
            (
                selected_action,
                root_value,
                policy_target,
                visit_counts,
                observation_host,
                legal_mask_host,
            ) = jax.device_get(
                (
                    output.selected_action,
                    output.root_value,
                    output.policy_target,
                    output.visit_counts,
                    observation,
                    legal_mask,
                )
            )
            action = int(selected_action)
            if action < 0 or action >= self.game.num_actions or not bool(legal_mask_host[action]):
                illegal_actions += 1
                raise ValueError(f"search selected illegal action {action} at move {move_index}")

            steps.append(
                TrajectoryStep(
                    observation=observation_host,
                    legal_action_mask=legal_mask_host,
                    policy_target=policy_target,
                    action=action,
                    root_value=float(root_value),
                    to_play=to_play,
                    move_index=move_index,
                    search_stats={
                        "selected_action": action,
                        "root_value": float(root_value),
                        "policy_entropy": _entropy(policy_target),
                        "visit_counts": np.asarray(visit_counts).tolist(),
                    },
                )
            )
            state = self.game.step(state, action)

        final_rewards = self.game.rewards(state)
        if illegal_actions:
            raise ValueError(f"illegal actions encountered: {illegal_actions}")
        return Trajectory(
            game_id=game_id,
            game_name=self.game.name,
            algorithm_name=self.algorithm.name,
            model_version=self.model_version,
            steps=steps,
            final_rewards=final_rewards,
        )

    def play_batch(self, num_games: int, seed: int) -> tuple[list[Trajectory], SelfPlayResult]:
        if num_games <= 0:
            raise ValueError("num_games must be positive")
        start = perf_counter()
        trajectories = []
        samples = []
        entropies = []
        root_values = []
        for index in range(num_games):
            trajectory = self.play_game(jax.random.PRNGKey(seed + index))
            trajectories.append(trajectory)
            samples.extend(self.algorithm.generate_targets(trajectory, trajectory.final_rewards))
            entropies.extend(step.search_stats["policy_entropy"] for step in trajectory.steps)
            root_values.extend(step.root_value for step in trajectory.steps)

        shard = self.replay_writer.write_shard(samples)
        elapsed = perf_counter() - start
        positions = sum(len(trajectory.steps) for trajectory in trajectories)
        result = SelfPlayResult(
            games=num_games,
            positions=positions,
            illegal_action_rate=0.0,
            policy_entropy_mean=float(np.mean(entropies)) if entropies else 0.0,
            root_value_mean=float(np.mean(root_values)) if root_values else 0.0,
            replay_shard=str(shard),
            seconds=elapsed,
        )
        return trajectories, result
