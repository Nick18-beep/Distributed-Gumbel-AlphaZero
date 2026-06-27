"""Model registry."""

from __future__ import annotations

from gumbel_az.config.schema import ModelConfig
from gumbel_az.model.base import NetworkFactory
from gumbel_az.model.mlp import MLPNetworkFactory
from gumbel_az.model.resnet import ResNetBoardFactory


def registered_models() -> tuple[str, ...]:
    return ("mlp_small", "resnet_board")


def create_network(config: ModelConfig, *, num_actions: int) -> NetworkFactory:
    if config.name == "mlp_small":
        if config.hidden_size is None:
            raise ValueError("mlp_small requires hidden_size")
        return MLPNetworkFactory(hidden_size=config.hidden_size, num_actions=num_actions)
    if config.name == "resnet_board":
        if config.channels is None or config.blocks is None:
            raise ValueError("resnet_board requires channels and blocks")
        return ResNetBoardFactory(
            channels=config.channels,
            blocks=config.blocks,
            num_actions=num_actions,
        )
    available = ", ".join(registered_models())
    raise KeyError(f"unknown model {config.name!r}; available models: {available}")
