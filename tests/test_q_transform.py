from __future__ import annotations

import torch

from gumbel_az.search.q_transform import completed_by_mix_value


def test_normalize_q_values_masks_illegal_actions() -> None:
    q_values = torch.asarray([[1.0, 3.0, 5.0, -10.0]])
    legal = torch.asarray([[True, True, True, False]])

    transformed = completed_by_mix_value(q_values, legal)

    assert transformed.tolist() == [[0.0, 0.5, 1.0, 0.0]]
