from __future__ import annotations

import jax.numpy as jnp

from gumbel_az.search.sequential_halving import select_candidates, sequential_halving


def test_select_candidates_respects_legal_mask() -> None:
    scores = jnp.asarray([[0.1, 0.9, 0.8, 2.0]])
    legal = jnp.asarray([[True, True, True, False]])

    candidates = select_candidates(scores, legal, 2)

    assert candidates.tolist() == [[1, 2]]


def test_sequential_halving_returns_best_legal_action() -> None:
    scores = jnp.asarray([[0.1, 0.9, 0.8, 2.0]])
    legal = jnp.asarray([[True, True, True, False]])

    selected = sequential_halving(scores, legal, 3)

    assert selected.tolist() == [1]


def test_sequential_halving_clamps_candidate_count_to_action_count() -> None:
    scores = jnp.asarray([[0.1, 0.9, 0.8]])
    legal = jnp.asarray([[True, True, True]])

    candidates = select_candidates(scores, legal, 99)
    selected = sequential_halving(scores, legal, 99)

    assert candidates.shape == (1, 3)
    assert selected.tolist() == [1]
