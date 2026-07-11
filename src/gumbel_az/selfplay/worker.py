"""PyTorch self-play worker using Gumbel AlphaZero search."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from gumbel_az.algorithms import create_algorithm
from gumbel_az.config.schema import AppConfig
from gumbel_az.envs import create_game
from gumbel_az.model import create_network
from gumbel_az.replay import ReplayWriter
from gumbel_az.search import TorchGumbelSearchBackend
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


def _entropy(policy: np.ndarray) -> float:
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
        model: torch.nn.Module | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.game = create_game(config.game.name)
        self.network = create_network(config.model, num_actions=self.game.num_actions)
        resolved_model = model
        if resolved_model is None:
            resolved_model = self.network.init(
                config.run.seed,
                self.game.observation_shape,
                self.game.num_actions,
                device=self.device,
            )
        self.model: torch.nn.Module = resolved_model
        self.model.to(self.device)
        self.model.eval()
        self.algorithm = create_algorithm(
            config,
            game=self.game,
            search_backend=TorchGumbelSearchBackend(game=self.game, device=self.device),
        )
        self.replay_writer = replay_writer
        self.model_version = 0

    def _network_apply(self, observations: torch.Tensor) -> Any:
        self.model.eval()
        return self.model(observations.to(self.device, non_blocking=True))

    def _make_generator(self, seed: int) -> torch.Generator:
        try:
            generator = torch.Generator(device=self.device)
        except (RuntimeError, TypeError):
            generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def play_game(self, seed: int, *, game_id: str | None = None) -> Trajectory:
        generator = self._make_generator(seed)
        state = self.game.init(seed)
        steps: list[TrajectoryStep] = []
        game_id = game_id or uuid4().hex
        illegal_actions = 0

        for move_index in range(self.game.max_moves):
            if bool(self.game.is_terminal(state)):
                break
            observation = self.game.canonical_observation(state)
            legal_mask = self.game.legal_action_mask(state)
            to_play = int(self.game.current_player(state))
            temperature = self.algorithm.temperature_for_move(move_index)
            output = self.algorithm.select_action(
                game_state=state,
                network_apply=self._network_apply,
                rng=generator,
                temperature=temperature,
            )
            selected_action = int(output.selected_action.detach().cpu().item())
            policy_target = output.policy_target.detach().cpu().numpy().astype(np.float32)
            visit_counts = output.visit_counts.detach().cpu().numpy().astype(np.float32)
            root_value = float(output.root_value.detach().cpu().item())
            if (
                selected_action < 0
                or selected_action >= self.game.num_actions
                or not bool(legal_mask[selected_action])
            ):
                illegal_actions += 1
                raise ValueError(
                    f"search selected illegal action {selected_action} at move {move_index}"
                )

            steps.append(
                TrajectoryStep(
                    observation=np.asarray(observation, dtype=np.float32),
                    legal_action_mask=np.asarray(legal_mask, dtype=bool),
                    policy_target=policy_target,
                    action=selected_action,
                    root_value=root_value,
                    to_play=to_play,
                    move_index=move_index,
                    search_stats={
                        "backend": "torch_gumbel",
                        "selected_action": selected_action,
                        "root_value": root_value,
                        "policy_entropy": _entropy(policy_target),
                        "visit_counts": visit_counts.tolist(),
                    },
                )
            )
            state = self.game.step(state, selected_action)

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
        entropies: list[float] = []
        root_values: list[float] = []
        for index in range(num_games):
            trajectory = self.play_game(seed + index)
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
