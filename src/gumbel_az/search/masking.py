"""Action masking helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def apply_legal_mask(
    logits: jax.Array,
    legal_mask: jax.Array,
    illegal_value: float = -1.0e9,
) -> jax.Array:
    return jnp.where(legal_mask, logits, jnp.asarray(illegal_value, dtype=logits.dtype))


def masked_policy(logits: jax.Array, legal_mask: jax.Array) -> jax.Array:
    masked_logits = apply_legal_mask(logits, legal_mask)
    probs = jax.nn.softmax(masked_logits, axis=-1)
    return jnp.where(legal_mask, probs, jnp.zeros_like(probs))
