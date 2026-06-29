"""PyTorch residual board policy/value model."""

from __future__ import annotations

import torch
from torch import nn

from gumbel_az.model.common import NetworkOutput


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        conv_kernel_size: tuple[int, int],
        batch_norm_momentum: float,
    ) -> None:
        super().__init__()
        padding = (conv_kernel_size[0] // 2, conv_kernel_size[1] // 2)
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=conv_kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(channels, momentum=batch_norm_momentum),
            nn.ReLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=conv_kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(channels, momentum=batch_norm_momentum),
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
        conv_kernel_size: tuple[int, int] = (3, 3),
        policy_head_channels: int = 2,
        value_head_channels: int = 1,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        if len(observation_shape) != 3:
            raise ValueError(f"resnet_board expects HWC observation, got {observation_shape}")
        rows, columns, input_channels = observation_shape
        padding = (conv_kernel_size[0] // 2, conv_kernel_size[1] // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                channels,
                kernel_size=conv_kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(channels, momentum=batch_norm_momentum),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *(
                ResidualBlock(
                    channels,
                    conv_kernel_size=conv_kernel_size,
                    batch_norm_momentum=batch_norm_momentum,
                )
                for _ in range(blocks)
            )
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, policy_head_channels, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_head_channels * rows * columns, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, value_head_channels, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_head_channels * rows * columns, channels),
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

    def __init__(
        self,
        *,
        channels: int,
        blocks: int,
        num_actions: int,
        conv_kernel_size: tuple[int, int] = (3, 3),
        policy_head_channels: int = 2,
        value_head_channels: int = 1,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        self.channels = channels
        self.blocks = blocks
        self.num_actions = num_actions
        self.conv_kernel_size = conv_kernel_size
        self.policy_head_channels = policy_head_channels
        self.value_head_channels = value_head_channels
        self.batch_norm_momentum = batch_norm_momentum

    def build(self, observation_shape: tuple[int, ...], num_actions: int) -> nn.Module:
        if num_actions != self.num_actions:
            raise ValueError(f"num_actions mismatch: {num_actions} != {self.num_actions}")
        return ResNetBoard(
            observation_shape=observation_shape,
            channels=self.channels,
            blocks=self.blocks,
            num_actions=num_actions,
            conv_kernel_size=self.conv_kernel_size,
            policy_head_channels=self.policy_head_channels,
            value_head_channels=self.value_head_channels,
            batch_norm_momentum=self.batch_norm_momentum,
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
