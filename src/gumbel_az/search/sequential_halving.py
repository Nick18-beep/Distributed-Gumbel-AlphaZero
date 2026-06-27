"""Sequential halving candidate helper."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def select_candidates(
    scores: jax.Array,
    legal_mask: jax.Array,
    max_num_considered_actions: int,
) -> jax.Array:
    max_actions = min(max_num_considered_actions, scores.shape[-1])
    masked_scores = jnp.where(legal_mask, scores, -jnp.inf)
    order = jnp.argsort(masked_scores, axis=-1)[..., ::-1]
    return order[..., :max_actions]


def sequential_halving(
    scores: jax.Array,
    legal_mask: jax.Array,
    max_num_considered_actions: int,
) -> jax.Array:
    candidates = select_candidates(scores, legal_mask, max_num_considered_actions)
    rounds = max(1, math.ceil(math.log2(candidates.shape[-1])))
    active = candidates
    for _ in range(rounds):
        keep = max(1, active.shape[-1] // 2)
        candidate_scores = jnp.take_along_axis(scores, active, axis=-1)
        order = jnp.argsort(candidate_scores, axis=-1)[..., ::-1]
        active = jnp.take_along_axis(active, order[..., :keep], axis=-1)
    return active[..., 0]
