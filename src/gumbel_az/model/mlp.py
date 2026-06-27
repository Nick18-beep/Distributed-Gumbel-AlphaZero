"""Small MLP network for CPU tests and debug runs."""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from gumbel_az.model.common import NetworkOutput


class MLPModule(nn.Module):
    hidden_size: int
    num_actions: int

    @nn.compact
    def __call__(self, observations: jax.Array, train: bool = False) -> NetworkOutput:
        del train
        x = observations.reshape((observations.shape[0], -1))
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        policy_logits = nn.Dense(self.num_actions)(x)
        value = nn.tanh(nn.Dense(1)(x)).squeeze(-1)
        return NetworkOutput(policy_logits=policy_logits, value=value)


@dataclass(frozen=True)
class MLPNetworkFactory:
    hidden_size: int
    num_actions: int
    name: str = "mlp_small"

    def init(
        self,
        rng_key: jax.Array,
        observation_shape: tuple[int, ...],
        num_actions: int,
    ) -> dict:
        if num_actions != self.num_actions:
            raise ValueError(
                f"MLPNetworkFactory expected num_actions={self.num_actions}, got {num_actions}"
            )
        module = MLPModule(hidden_size=self.hidden_size, num_actions=self.num_actions)
        dummy = jnp.zeros((1, *observation_shape), dtype=jnp.float32)
        return module.init(rng_key, dummy, train=False)["params"]

    def apply(self, params: dict, observations: jax.Array, train: bool = False) -> NetworkOutput:
        module = MLPModule(hidden_size=self.hidden_size, num_actions=self.num_actions)
        return module.apply({"params": params}, observations, train=train)
