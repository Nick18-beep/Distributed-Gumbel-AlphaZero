"""Common PyTorch model outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NetworkOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor
