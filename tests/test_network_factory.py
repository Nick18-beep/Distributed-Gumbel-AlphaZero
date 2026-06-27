from __future__ import annotations

import jax
import jax.numpy as jnp

from gumbel_az.config import load_config
from gumbel_az.envs import create_game
from gumbel_az.model import create_network

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def test_mlp_init_is_deterministic_and_uses_num_actions() -> None:
    config = load_config(CONFIG_DIR / "connect_four_cpu_debug.yaml")
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions + 2)
    key = jax.random.PRNGKey(0)

    params_a = network.init(key, game.observation_shape, game.num_actions + 2)
    params_b = network.init(key, game.observation_shape, game.num_actions + 2)
    output = network.apply(params_a, jnp.zeros((3, *game.observation_shape), dtype=jnp.float32))

    assert jax.tree.all(jax.tree.map(lambda a, b: jnp.array_equal(a, b), params_a, params_b))
    assert output.policy_logits.shape == (3, game.num_actions + 2)
    assert output.value.shape == (3,)
    assert jnp.all(output.value <= 1.0)
    assert jnp.all(output.value >= -1.0)


def test_resnet_forward_shapes() -> None:
    config = load_config(CONFIG_DIR / "connect_four.yaml")
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)

    params = network.init(jax.random.PRNGKey(1), game.observation_shape, game.num_actions)
    output = network.apply(params, jnp.zeros((2, *game.observation_shape), dtype=jnp.float32))

    assert output.policy_logits.shape == (2, game.num_actions)
    assert output.value.shape == (2,)


def test_model_init_rejects_num_actions_mismatch() -> None:
    config = load_config(CONFIG_DIR / "connect_four_cpu_debug.yaml")
    game = create_game(config.game.name)
    network = create_network(config.model, num_actions=game.num_actions)

    try:
        network.init(jax.random.PRNGKey(0), game.observation_shape, game.num_actions + 1)
    except ValueError as exc:
        assert "num_actions" in str(exc)
    else:
        raise AssertionError("expected num_actions mismatch to fail")
