"""Single-process evaluation arena."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gumbel_az.config.schema import AppConfig
from gumbel_az.envs import create_game
from gumbel_az.eval.opponents import random_legal_action
from gumbel_az.logging import JsonlWriter
from gumbel_az.model import create_network


@dataclass(frozen=True)
class EvalResult:
    checkpoint_version: int
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    games_per_sec: float


class Arena:
    """Evaluate a candidate checkpoint against a random baseline."""

    def __init__(
        self,
        config: AppConfig,
        *,
        eval_dir: Path,
        event_writer: JsonlWriter | None = None,
    ) -> None:
        self.config = config
        self.game = create_game(config.game.name)
        self.network = create_network(config.model, num_actions=self.game.num_actions)
        self.matches_writer = JsonlWriter(eval_dir / "matches.jsonl")
        self.event_writer = event_writer
        self._greedy_action = jax.jit(self._greedy_action_impl)

    def _greedy_action_impl(
        self,
        params: Any,
        observation: jax.Array,
        legal: jax.Array,
    ) -> jax.Array:
        output = self.network.apply(params, observation[None, ...], train=False)
        logits = jnp.where(legal, output.policy_logits[0], -jnp.inf)
        return jnp.argmax(logits)

    def _network_action(self, params: Any, state: Any) -> int:
        observation = self.game.canonical_observation(state)
        legal = self.game.legal_action_mask(state)
        return int(self._greedy_action(params, observation, legal))

    def play_vs_random(
        self,
        *,
        params: Any,
        checkpoint_version: int,
        game_index: int,
        rng: np.random.Generator,
    ) -> float:
        state = self.game.init(jax.random.PRNGKey(self.config.run.seed + 10_000 + game_index))
        candidate_player = game_index % self.game.num_players

        for _ in range(self.game.max_moves):
            if bool(self.game.is_terminal(state)):
                break
            current_player = int(self.game.current_player(state))
            if current_player == candidate_player:
                action = self._network_action(params, state)
            else:
                action = random_legal_action(self.game.legal_action_mask(state), rng)
            state = self.game.step(state, action)

        reward = float(np.asarray(self.game.rewards(state))[candidate_player])
        record = {
            "checkpoint_version": checkpoint_version,
            "game_index": game_index,
            "candidate_player": candidate_player,
            "opponent": "random",
            "moves": int(state.move_count),
            "reward": reward,
            "result": "win" if reward > 0 else "loss" if reward < 0 else "draw",
        }
        self.matches_writer.write(record)
        if self.event_writer is not None:
            self.event_writer.write({"event": "eval_match_completed", **record})
        return reward

    def evaluate_vs_random(self, *, params: Any, checkpoint_version: int) -> EvalResult:
        start = perf_counter()
        rng = np.random.default_rng(self.config.run.seed + checkpoint_version)
        rewards = [
            self.play_vs_random(
                params=params,
                checkpoint_version=checkpoint_version,
                game_index=index,
                rng=rng,
            )
            for index in range(self.config.eval.games)
        ]
        wins = sum(1 for reward in rewards if reward > 0.0)
        losses = sum(1 for reward in rewards if reward < 0.0)
        draws = len(rewards) - wins - losses
        elapsed = perf_counter() - start
        return EvalResult(
            checkpoint_version=checkpoint_version,
            games=len(rewards),
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=wins / max(len(rewards), 1),
            games_per_sec=len(rewards) / max(elapsed, 1.0e-9),
        )

    def play_vs_params(
        self,
        *,
        candidate_params: Any,
        opponent_params: Any,
        checkpoint_version: int,
        opponent_version: int,
        game_index: int,
    ) -> float:
        state = self.game.init(jax.random.PRNGKey(self.config.run.seed + 20_000 + game_index))
        candidate_player = game_index % self.game.num_players

        for _ in range(self.game.max_moves):
            if bool(self.game.is_terminal(state)):
                break
            current_player = int(self.game.current_player(state))
            params = candidate_params if current_player == candidate_player else opponent_params
            state = self.game.step(state, self._network_action(params, state))

        reward = float(np.asarray(self.game.rewards(state))[candidate_player])
        record = {
            "checkpoint_version": checkpoint_version,
            "opponent_checkpoint_version": opponent_version,
            "game_index": game_index,
            "candidate_player": candidate_player,
            "opponent": f"checkpoint:{opponent_version}",
            "moves": int(state.move_count),
            "reward": reward,
            "result": "win" if reward > 0 else "loss" if reward < 0 else "draw",
        }
        self.matches_writer.write(record)
        if self.event_writer is not None:
            self.event_writer.write({"event": "eval_match_completed", **record})
        return reward

    def evaluate_vs_params(
        self,
        *,
        candidate_params: Any,
        opponent_params: Any,
        checkpoint_version: int,
        opponent_version: int,
    ) -> EvalResult:
        start = perf_counter()
        rewards = [
            self.play_vs_params(
                candidate_params=candidate_params,
                opponent_params=opponent_params,
                checkpoint_version=checkpoint_version,
                opponent_version=opponent_version,
                game_index=index,
            )
            for index in range(self.config.eval.games)
        ]
        wins = sum(1 for reward in rewards if reward > 0.0)
        losses = sum(1 for reward in rewards if reward < 0.0)
        draws = len(rewards) - wins - losses
        elapsed = perf_counter() - start
        return EvalResult(
            checkpoint_version=checkpoint_version,
            games=len(rewards),
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=wins / max(len(rewards), 1),
            games_per_sec=len(rewards) / max(elapsed, 1.0e-9),
        )
