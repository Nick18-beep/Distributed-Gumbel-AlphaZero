from __future__ import annotations

import torch

from gumbel_az.search.sequential_halving import sequential_halving, top_k_candidates


def test_select_candidates_respects_legal_mask() -> None:
    scores = torch.asarray([[0.1, 0.9, 0.8, 2.0]])
    legal = torch.asarray([[True, True, True, False]])

    candidates = top_k_candidates(scores, legal, 2)

    assert candidates.tolist() == [[1, 2]]


def test_sequential_halving_returns_best_legal_action() -> None:
    scores = torch.asarray([[0.1, 0.9, 0.8, 2.0]])
    legal = torch.asarray([[True, True, True, False]])

    selected = sequential_halving(scores, legal, 3)

    assert selected.tolist() == [1]


def test_sequential_halving_clamps_candidate_count_to_action_count() -> None:
    scores = torch.asarray([[0.1, 0.9, 0.8]])
    legal = torch.asarray([[True, True, True]])

    candidates = top_k_candidates(scores, legal, 99)
    selected = sequential_halving(scores, legal, 99)

    assert candidates.shape == (1, 3)
    assert selected.tolist() == [1]


def test_candidates_pad_with_best_legal_action_when_few_legal_moves() -> None:
    scores = torch.asarray([[0.1, 4.0, 0.8, 2.0]])
    legal = torch.asarray([[False, False, True, False]])

    candidates = top_k_candidates(scores, legal, 4)
    selected = sequential_halving(scores, legal, 4)

    assert candidates.tolist() == [[2, 2, 2, 2]]
    assert selected.tolist() == [2]
