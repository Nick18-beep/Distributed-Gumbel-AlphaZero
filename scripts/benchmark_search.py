"""Benchmark batched MCTX search after a warmup call."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mctx

from gumbel_az.config import load_config
from gumbel_az.envs.custom.connect_four import COLUMNS, ConnectFourGame
from gumbel_az.model.common import NetworkOutput
from gumbel_az.search import MctxSearchBackend


def _batched_state(game: ConnectFourGame, batch_size: int):
    state = game.init()
    return jax.tree.map(lambda value: jnp.repeat(value[None, ...], batch_size, axis=0), state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/connect_four_cpu_debug.yaml"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    game = ConnectFourGame()
    state = _batched_state(game, args.batch_size)
    observations = jax.vmap(game.canonical_observation)(state)
    legal_mask = jax.vmap(game.legal_action_mask)(state)

    def network_apply(observation):
        batch_size = observation.shape[0]
        return NetworkOutput(
            policy_logits=jnp.zeros((batch_size, COLUMNS), dtype=jnp.float32),
            value=jnp.zeros((batch_size,), dtype=jnp.float32),
        )

    def recurrent_fn(params, rng_key, action, embedding):
        del params, rng_key
        next_state = jax.vmap(game.step)(embedding, action)
        output = network_apply(jax.vmap(game.canonical_observation)(next_state))
        return (
            mctx.RecurrentFnOutput(
                reward=jnp.zeros(action.shape, dtype=jnp.float32),
                discount=jnp.where(jax.vmap(game.is_terminal)(next_state), 0.0, 1.0),
                prior_logits=output.policy_logits,
                value=output.value,
            ),
            next_state,
        )

    backend = MctxSearchBackend()
    search = jax.jit(
        lambda key: (
            backend.search(
                root_observation=observations,
                root_legal_mask=legal_mask,
                network_apply=network_apply,
                recurrent_fn=recurrent_fn,
                rng_key=key,
                config=config.search,
                root_embedding=state,
            ).policy_target
        )
    )

    search(jax.random.PRNGKey(0)).block_until_ready()
    start = time.perf_counter()
    for index in range(args.iterations):
        search(jax.random.PRNGKey(index + 1)).block_until_ready()
    elapsed = time.perf_counter() - start
    searches = args.batch_size * args.iterations
    print(
        f"searches={searches} elapsed_sec={elapsed:.6f} searches_per_sec={searches / elapsed:.2f}"
    )


if __name__ == "__main__":
    main()
