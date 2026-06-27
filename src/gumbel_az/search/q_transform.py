"""Q-value normalization helpers."""

from __future__ import annotations

import torch


def completed_by_mix_value(q_values: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    legal = legal_mask.bool()
    q_min = torch.where(legal, q_values, torch.full_like(q_values, torch.inf)).amin(
        dim=-1,
        keepdim=True,
    )
    q_max = torch.where(legal, q_values, torch.full_like(q_values, -torch.inf)).amax(
        dim=-1,
        keepdim=True,
    )
    scale = torch.clamp(q_max - q_min, min=1.0e-8)
    normalized = (q_values - q_min) / scale
    return torch.where(legal, normalized, torch.zeros_like(normalized))
