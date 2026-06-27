"""Gumbel AlphaZero algorithm glue around MCTX search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import mctx

from gumbel_az.config.schema import SearchConfig
from gumbel_az.search.outputs import SearchOutput


def _batch_state(state: Any) -> Any:
    return jax.tree.map(lambda value: value[None, ...], state)


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
    game: Any
    search_backend: Any
    search_config: SearchConfig
    temperature_moves: int
    name: str = "gumbel_alphazero"

    def _recurrent_fn(self, network_apply):
        game = self.game

        def recurrent_fn(params, rng_key, action, embedding):
            del params, rng_key
            previous_player = game.current_player(embedding)
            next_state = jax.vmap(game.step)(embedding, action)
            next_observation = jax.vmap(game.canonical_observation)(next_state)
            network_output = network_apply(next_observation)
            terminal = jax.vmap(game.is_terminal)(next_state)
            rewards = jax.vmap(game.rewards)(next_state)
            reward = jnp.take_along_axis(
                rewards,
                previous_player.astype(jnp.int32)[:, None],
                axis=1,
            ).squeeze(-1)
            return (
                mctx.RecurrentFnOutput(
                    reward=reward,
                    discount=jnp.where(terminal, 0.0, -1.0),
                    prior_logits=network_output.policy_logits,
                    value=network_output.value,
                ),
                next_state,
            )

        return recurrent_fn

    def select_action(
        self,
        *,
        game_state: Any,
        network_apply,
        rng_key: jax.Array,
        temperature: float = 0.0,
    ) -> SearchOutput:
        search_key, action_key = jax.random.split(rng_key)
        batched_state = _batch_state(game_state)
        root_observation = jax.vmap(self.game.canonical_observation)(batched_state)
        root_legal_mask = jax.vmap(self.game.legal_action_mask)(batched_state)
        output = self.search_backend.search(
            root_observation=root_observation,
            root_legal_mask=root_legal_mask,
            network_apply=network_apply,
            recurrent_fn=self._recurrent_fn(network_apply),
            rng_key=search_key,
            config=self.search_config,
            root_embedding=batched_state,
        )
        unbatched = _unbatch_output(output)
        temperature_array = jnp.asarray(temperature, dtype=jnp.float32)
        safe_temperature = jnp.maximum(temperature_array, 1.0e-8)
        positive_policy = unbatched.policy_target > 0.0
        logits = jnp.where(
            positive_policy,
            jnp.log(jnp.maximum(unbatched.policy_target, 1.0e-8)) / safe_temperature,
            -jnp.inf,
        )
        sampled_action = jax.random.categorical(action_key, logits)
        selected_action = jnp.where(
            temperature_array <= 0.0,
            unbatched.selected_action,
            sampled_action,
        )
        return unbatched._replace(selected_action=selected_action)

    def temperature_for_move(self, move_index: int) -> float:
        return 1.0 if move_index < self.temperature_moves else 0.0

    def generate_targets(self, trajectory, final_rewards: jax.Array) -> list[dict]:
        samples: list[dict] = []
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
