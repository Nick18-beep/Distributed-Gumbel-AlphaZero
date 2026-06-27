from __future__ import annotations

import jax.numpy as jnp

from gumbel_az.search.q_transform import normalize_q_values


def test_normalize_q_values_masks_illegal_actions() -> None:
    q_values = jnp.asarray([[1.0, 3.0, 5.0, -10.0]])
    legal = jnp.asarray([[True, True, True, False]])

    transformed = normalize_q_values(q_values, legal)

    assert transformed.tolist() == [[0.0, 0.5, 1.0, 0.0]]
