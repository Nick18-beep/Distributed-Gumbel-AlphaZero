"""Q-value transform utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def normalize_q_values(
    q_values: jax.Array,
    legal_mask: jax.Array,
    epsilon: float = 1.0e-8,
) -> jax.Array:
    q_min = jnp.min(jnp.where(legal_mask, q_values, jnp.inf), axis=-1, keepdims=True)
    q_max = jnp.max(jnp.where(legal_mask, q_values, -jnp.inf), axis=-1, keepdims=True)
    normalized = (q_values - q_min) / (q_max - q_min + epsilon)
    return jnp.where(legal_mask, normalized, jnp.zeros_like(normalized))
