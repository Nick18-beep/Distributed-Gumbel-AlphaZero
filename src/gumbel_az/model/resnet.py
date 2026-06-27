"""Small residual board network."""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from gumbel_az.model.common import NetworkOutput


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = nn.relu(nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x))
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        return nn.relu(x + residual)


class ResNetBoardModule(nn.Module):
    channels: int
    blocks: int
    num_actions: int

    @nn.compact
    def __call__(self, observations: jax.Array, train: bool = False) -> NetworkOutput:
        del train
        x = nn.relu(nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(observations))
        for _ in range(self.blocks):
            x = ResidualBlock(self.channels)(x)
        flat = x.reshape((x.shape[0], -1))
        policy_hidden = nn.relu(nn.Dense(self.channels)(flat))
        value_hidden = nn.relu(nn.Dense(self.channels)(flat))
        policy_logits = nn.Dense(self.num_actions)(policy_hidden)
        value = nn.tanh(nn.Dense(1)(value_hidden)).squeeze(-1)
        return NetworkOutput(policy_logits=policy_logits, value=value)


@dataclass(frozen=True)
class ResNetBoardFactory:
    channels: int
    blocks: int
    num_actions: int
    name: str = "resnet_board"

    def init(
        self,
        rng_key: jax.Array,
        observation_shape: tuple[int, ...],
        num_actions: int,
    ) -> dict:
        if num_actions != self.num_actions:
            raise ValueError(
                f"ResNetBoardFactory expected num_actions={self.num_actions}, got {num_actions}"
            )
        module = ResNetBoardModule(
            channels=self.channels,
            blocks=self.blocks,
            num_actions=self.num_actions,
        )
        dummy = jnp.zeros((1, *observation_shape), dtype=jnp.float32)
        return module.init(rng_key, dummy, train=False)["params"]

    def apply(self, params: dict, observations: jax.Array, train: bool = False) -> NetworkOutput:
        module = ResNetBoardModule(
            channels=self.channels,
            blocks=self.blocks,
            num_actions=self.num_actions,
        )
        return module.apply({"params": params}, observations, train=train)
