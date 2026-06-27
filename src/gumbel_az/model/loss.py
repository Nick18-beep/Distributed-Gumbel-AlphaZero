"""Training losses for policy/value networks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gumbel_az.model.common import NetworkOutput


def policy_loss(policy_logits: jax.Array, policy_target: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(policy_logits, axis=-1)
    return -jnp.mean(jnp.sum(policy_target * log_probs, axis=-1))


def value_loss(value: jax.Array, value_target: jax.Array) -> jax.Array:
    return jnp.mean(jnp.square(value - value_target))


def total_loss(
    outputs: NetworkOutput,
    batch: dict[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    p_loss = policy_loss(outputs.policy_logits, batch["policy_target"])
    v_loss = value_loss(outputs.value, batch["value_target"])
    loss = p_loss + v_loss
    return loss, {
        "policy_loss": p_loss,
        "value_loss": v_loss,
        "total_loss": loss,
    }
