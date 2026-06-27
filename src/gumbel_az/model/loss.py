"""PyTorch policy/value losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gumbel_az.model.common import NetworkOutput


def policy_loss(policy_logits: torch.Tensor, policy_target: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(policy_logits, dim=-1)
    return -(policy_target * log_probs).sum(dim=-1).mean()


def value_loss(value: torch.Tensor, value_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(value, value_target)


def total_loss(
    outputs: NetworkOutput,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    p_loss = policy_loss(outputs.policy_logits, batch["policy_target"])
    v_loss = value_loss(outputs.value, batch["value_target"])
    loss = p_loss + v_loss
    return loss, {
        "policy_loss": p_loss.detach(),
        "value_loss": v_loss.detach(),
        "total_loss": loss.detach(),
    }
