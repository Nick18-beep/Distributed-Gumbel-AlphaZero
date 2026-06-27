"""PyTorch network factory protocol."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class NetworkFactory(Protocol):
    name: str

    def build(self, observation_shape: tuple[int, ...], num_actions: int) -> nn.Module:
        """Create a PyTorch policy/value module."""

    def init(
        self,
        seed: int,
        observation_shape: tuple[int, ...],
        num_actions: int,
        *,
        device: torch.device | str = "cpu",
    ) -> nn.Module:
        """Create an initialized module on device."""
