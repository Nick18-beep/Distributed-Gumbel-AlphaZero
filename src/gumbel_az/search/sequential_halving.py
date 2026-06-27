"""Sequential halving utilities for PyTorch search."""

from __future__ import annotations

import torch


def top_k_candidates(
    scores: torch.Tensor,
    legal_mask: torch.Tensor,
    max_num_considered_actions: int,
) -> torch.Tensor:
    masked = torch.where(legal_mask.bool(), scores, torch.full_like(scores, -torch.inf))
    k = min(max_num_considered_actions, scores.shape[-1])
    candidates = torch.topk(masked, k=k, dim=-1).indices
    candidate_legal = torch.gather(legal_mask.bool(), dim=-1, index=candidates)
    best_legal = torch.argmax(masked, dim=-1, keepdim=True)
    return torch.where(candidate_legal, candidates, best_legal.expand_as(candidates))


def sequential_halving(
    scores: torch.Tensor,
    legal_mask: torch.Tensor,
    max_num_considered_actions: int,
) -> torch.Tensor:
    active = top_k_candidates(scores, legal_mask, max_num_considered_actions)
    while active.shape[-1] > 1:
        candidate_scores = torch.gather(scores, dim=-1, index=active)
        keep = max(1, (active.shape[-1] + 1) // 2)
        order = torch.topk(candidate_scores, k=keep, dim=-1).indices
        active = torch.gather(active, dim=-1, index=order)
    return active.squeeze(-1)
