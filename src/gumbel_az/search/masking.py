"""Legal action masking helpers for PyTorch tensors."""

from __future__ import annotations

import torch


def apply_legal_mask(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    illegal_value: float = -1.0e9,
) -> torch.Tensor:
    return torch.where(legal_mask.bool(), logits, torch.full_like(logits, illegal_value))


def masked_policy(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    masked_logits = apply_legal_mask(logits, legal_mask)
    probs = torch.softmax(masked_logits, dim=-1)
    return torch.where(legal_mask.bool(), probs, torch.zeros_like(probs))
