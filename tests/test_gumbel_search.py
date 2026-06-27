from __future__ import annotations

import jax
import jax.numpy as jnp
import mctx

from gumbel_az.config import load_config
from gumbel_az.envs.custom.connect_four import COLUMNS, ConnectFourGame
from gumbel_az.model.common import NetworkOutput
from gumbel_az.search.masking import masked_policy
from gumbel_az.search.mctx_backend import MctxSearchBackend

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def _batched_state(game: ConnectFourGame, batch_size: int):
    state = game.init()
    return jax.tree.map(lambda value: jnp.repeat(value[None, ...], batch_size, axis=0), state)


def test_mctx_search_smoke_connect_four_masks_illegal_actions_and_is_deterministic() -> None:
    config = load_config(CONFIG)
    game = ConnectFourGame()
    state = _batched_state(game, 2)
    observations = jax.vmap(game.canonical_observation)(state)
    legal_mask = jax.vmap(game.legal_action_mask)(state)
    legal_mask = legal_mask.at[:, -1].set(False)

    def network_apply(observation):
        batch_size = observation.shape[0]
        logits = jnp.arange(COLUMNS, dtype=jnp.float32)
        return NetworkOutput(
            policy_logits=jnp.repeat(logits[None, :], batch_size, axis=0),
            value=jnp.zeros((batch_size,), dtype=jnp.float32),
        )

    def recurrent_fn(params, rng_key, action, embedding):
        del params, rng_key
        next_state = jax.vmap(game.step)(embedding, action)
        next_obs = jax.vmap(game.canonical_observation)(next_state)
        output = network_apply(next_obs)
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
    output_a = backend.search(
        root_observation=observations,
        root_legal_mask=legal_mask,
        network_apply=network_apply,
        recurrent_fn=recurrent_fn,
        rng_key=jax.random.PRNGKey(0),
        config=config.search,
        root_embedding=state,
    )
    output_b = backend.search(
        root_observation=observations,
        root_legal_mask=legal_mask,
        network_apply=network_apply,
        recurrent_fn=recurrent_fn,
        rng_key=jax.random.PRNGKey(0),
        config=config.search,
        root_embedding=state,
    )

    assert output_a.policy_target.shape == (2, COLUMNS)
    assert output_a.visit_counts.shape == (2, COLUMNS)
    assert output_a.q_values.shape == (2, COLUMNS)
    assert output_a.prior_logits.shape == (2, COLUMNS)
    assert output_a.policy_target[:, -1].tolist() == [0.0, 0.0]
    assert jnp.allclose(jnp.sum(output_a.policy_target, axis=-1), 1.0)
    assert jnp.array_equal(output_a.selected_action, output_b.selected_action)
    assert jnp.allclose(output_a.policy_target, output_b.policy_target)


def test_mctx_search_output_is_jittable() -> None:
    config = load_config(CONFIG)
    game = ConnectFourGame()
    state = _batched_state(game, 1)
    observations = jax.vmap(game.canonical_observation)(state)
    legal_mask = jax.vmap(game.legal_action_mask)(state)

    def network_apply(observation):
        return NetworkOutput(
            policy_logits=jnp.zeros((observation.shape[0], COLUMNS), dtype=jnp.float32),
            value=jnp.zeros((observation.shape[0],), dtype=jnp.float32),
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

    @jax.jit
    def run_search(key):
        return backend.search(
            root_observation=observations,
            root_legal_mask=legal_mask,
            network_apply=network_apply,
            recurrent_fn=recurrent_fn,
            rng_key=key,
            config=config.search,
            root_embedding=state,
        )

    output = run_search(jax.random.PRNGKey(0))

    assert output.policy_target.shape == (1, COLUMNS)
    assert int(output.search_metadata["num_simulations"]) == config.search.simulations_per_move


def test_masked_policy_assigns_zero_to_illegal_actions() -> None:
    logits = jnp.asarray([[1.0, 2.0, 3.0]])
    legal = jnp.asarray([[True, False, True]])
    probs = masked_policy(logits, legal)

    assert probs[0, 1] == 0.0
    assert jnp.isclose(jnp.sum(probs), 1.0)


def test_mctx_search_rejects_shape_mismatch() -> None:
    config = load_config(CONFIG)
    backend = MctxSearchBackend()

    def network_apply(observation):
        return NetworkOutput(
            policy_logits=jnp.zeros((observation.shape[0], COLUMNS), dtype=jnp.float32),
            value=jnp.zeros((observation.shape[0],), dtype=jnp.float32),
        )

    def recurrent_fn(params, rng_key, action, embedding):
        raise AssertionError("recurrent_fn should not be called")

    try:
        backend.search(
            root_observation=jnp.zeros((1, 6, 7, 2), dtype=jnp.float32),
            root_legal_mask=jnp.ones((1, COLUMNS + 1), dtype=bool),
            network_apply=network_apply,
            recurrent_fn=recurrent_fn,
            rng_key=jax.random.PRNGKey(0),
            config=config.search,
        )
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("expected shape mismatch to fail")


def test_mctx_search_rejects_non_bool_legal_mask() -> None:
    config = load_config(CONFIG)
    backend = MctxSearchBackend()

    def network_apply(observation):
        return NetworkOutput(
            policy_logits=jnp.zeros((observation.shape[0], COLUMNS), dtype=jnp.float32),
            value=jnp.zeros((observation.shape[0],), dtype=jnp.float32),
        )

    def recurrent_fn(params, rng_key, action, embedding):
        raise AssertionError("recurrent_fn should not be called")

    try:
        backend.search(
            root_observation=jnp.zeros((1, 6, 7, 2), dtype=jnp.float32),
            root_legal_mask=jnp.ones((1, COLUMNS), dtype=jnp.int32),
            network_apply=network_apply,
            recurrent_fn=recurrent_fn,
            rng_key=jax.random.PRNGKey(0),
            config=config.search,
        )
    except TypeError as exc:
        assert "bool" in str(exc)
    else:
        raise AssertionError("expected non-bool mask to fail")


def test_mctx_search_rejects_roots_without_legal_actions() -> None:
    config = load_config(CONFIG)
    backend = MctxSearchBackend()

    def network_apply(observation):
        return NetworkOutput(
            policy_logits=jnp.zeros((observation.shape[0], COLUMNS), dtype=jnp.float32),
            value=jnp.zeros((observation.shape[0],), dtype=jnp.float32),
        )

    def recurrent_fn(params, rng_key, action, embedding):
        raise AssertionError("recurrent_fn should not be called")

    try:
        backend.search(
            root_observation=jnp.zeros((1, 6, 7, 2), dtype=jnp.float32),
            root_legal_mask=jnp.zeros((1, COLUMNS), dtype=bool),
            network_apply=network_apply,
            recurrent_fn=recurrent_fn,
            rng_key=jax.random.PRNGKey(0),
            config=config.search,
        )
    except ValueError as exc:
        assert "legal action" in str(exc)
    else:
        raise AssertionError("expected all-illegal root to fail")
