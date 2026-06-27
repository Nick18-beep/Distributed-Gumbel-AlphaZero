from __future__ import annotations

import jax
import jax.numpy as jnp

from gumbel_az.config import load_config
from gumbel_az.envs import create_game
from gumbel_az.model import create_network
from gumbel_az.model.optimizer import create_optimizer
from gumbel_az.training import create_train_state, train_step

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def test_train_step_is_finite_and_updates_params() -> None:
    config = load_config(CONFIG)
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)
    params = network.init(jax.random.PRNGKey(0), game.observation_shape, game.num_actions)
    tx, schedule = create_optimizer(config.training)
    state = create_train_state(params=params, apply_fn=network.apply, tx=tx)
    observations = jnp.ones(
        (config.training.batch_size, *game.observation_shape),
        dtype=jnp.float32,
    )
    value_target = jnp.ones((config.training.batch_size,), dtype=jnp.float32) * 0.5
    batch = {
        "observation": observations,
        "policy_target": (
            jnp.ones((config.training.batch_size, game.num_actions)) / game.num_actions
        ),
        "value_target": value_target,
    }

    new_state, metrics = train_step(state, batch, schedule(state.step))

    assert int(new_state.step) == 1
    assert bool(jnp.isfinite(metrics["total_loss"]))
    assert bool(jnp.isfinite(metrics["policy_loss"]))
    assert bool(jnp.isfinite(metrics["value_loss"]))
    assert bool(jnp.isfinite(metrics["grad_norm"]))
    assert metrics["grad_norm"] > 0.0
