"""PyTorch residual board policy/value model."""

from __future__ import annotations

import torch
from torch import nn

from gumbel_az.model.common import NetworkOutput


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ResNetBoard(nn.Module):
    def __init__(
        self,
        *,
        observation_shape: tuple[int, ...],
        channels: int,
        blocks: int,
        num_actions: int,
    ) -> None:
        super().__init__()
        if len(observation_shape) != 3:
            raise ValueError(f"resnet_board expects HWC observation, got {observation_shape}")
        rows, columns, input_channels = observation_shape
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * rows * columns, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(rows * columns, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> NetworkOutput:
        x = observations.float().permute(0, 3, 1, 2).contiguous()
        hidden = self.blocks(self.stem(x))
        return NetworkOutput(
            policy_logits=self.policy_head(hidden),
            value=self.value_head(hidden).squeeze(-1),
        )


class ResNetBoardFactory:
    name = "resnet_board"

    def __init__(self, *, channels: int, blocks: int, num_actions: int) -> None:
        self.channels = channels
        self.blocks = blocks
        self.num_actions = num_actions

    def build(self, observation_shape: tuple[int, ...], num_actions: int) -> nn.Module:
        if num_actions != self.num_actions:
            raise ValueError(f"num_actions mismatch: {num_actions} != {self.num_actions}")
        return ResNetBoard(
            observation_shape=observation_shape,
            channels=self.channels,
            blocks=self.blocks,
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
        return self.build(observation_shape, num_actions).to(device)
