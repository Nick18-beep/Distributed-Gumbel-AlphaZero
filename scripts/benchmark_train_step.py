"""Benchmark a warm train_step on synthetic Connect Four batches."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp

from gumbel_az.config import load_config
from gumbel_az.envs import create_game
from gumbel_az.model import create_network
from gumbel_az.model.optimizer import create_optimizer
from gumbel_az.training.train_state import create_train_state, train_step


def main() -> None:
    config = load_config(Path("configs/connect_four_cpu_debug.yaml"))
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)
    params = network.init(
        jax.random.PRNGKey(config.run.seed),
        game.observation_shape,
        game.num_actions,
    )
    tx, schedule = create_optimizer(config.training)
    state = create_train_state(params=params, apply_fn=network.apply, tx=tx)
    batch = {
        "observation": jnp.zeros(
            (config.training.batch_size, *game.observation_shape), dtype=jnp.float32
        ),
        "policy_target": jnp.full(
            (config.training.batch_size, game.num_actions),
            1.0 / game.num_actions,
            dtype=jnp.float32,
        ),
        "value_target": jnp.zeros((config.training.batch_size,), dtype=jnp.float32),
    }

    state, _ = train_step(state, batch, schedule(state.step))
    jax.block_until_ready(state.step)
    start = perf_counter()
    iterations = 20
    for _ in range(iterations):
        state, metrics = train_step(state, batch, schedule(state.step))
    jax.block_until_ready(metrics["total_loss"])
    elapsed = perf_counter() - start
    samples_per_sec = iterations * config.training.batch_size / max(elapsed, 1.0e-9)
    print(f"train_step_samples_per_sec={samples_per_sec:.2f}")


if __name__ == "__main__":
    main()
