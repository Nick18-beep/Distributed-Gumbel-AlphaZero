"""PyTorch MLP policy/value model."""

from __future__ import annotations

import torch
from torch import nn

from gumbel_az.model.common import NetworkOutput


class MLPPolicyValue(nn.Module):
    def __init__(
        self,
        *,
        observation_shape: tuple[int, ...],
        hidden_size: int,
        num_actions: int,
    ) -> None:
        super().__init__()
        features = 1
        for dim in observation_shape:
            features *= dim
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> NetworkOutput:
        hidden = self.body(observations.float())
        return NetworkOutput(
            policy_logits=self.policy_head(hidden),
            value=torch.tanh(self.value_head(hidden)).squeeze(-1),
        )


class MLPNetworkFactory:
    name = "mlp_small"

    def __init__(self, *, hidden_size: int, num_actions: int) -> None:
        self.hidden_size = hidden_size
        self.num_actions = num_actions

    def build(self, observation_shape: tuple[int, ...], num_actions: int) -> nn.Module:
        if num_actions != self.num_actions:
            raise ValueError(f"num_actions mismatch: {num_actions} != {self.num_actions}")
        return MLPPolicyValue(
            observation_shape=observation_shape,
            hidden_size=self.hidden_size,
            num_actions=num_actions,
        )

    def init(
        self,
        seed: int,
        observation_shape: tuple[int, ...],
        num_actions: int,
        *,
        device: torch.device | str = "cpu",
    ) -> nn.Module:
        torch.manual_seed(seed)
        model = self.build(observation_shape, num_actions).to(device)
        return model
