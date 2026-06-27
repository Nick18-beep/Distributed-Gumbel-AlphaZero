"""Algorithm registry."""

from __future__ import annotations

from gumbel_az.config.schema import AppConfig
from gumbel_az.domain.algorithm import TrainingAlgorithm


def registered_algorithms() -> tuple[str, ...]:
    return ("gumbel_alphazero",)


def create_algorithm(config: AppConfig, *, game, search_backend) -> TrainingAlgorithm:
    if config.algorithm.name == "gumbel_alphazero":
        from gumbel_az.algorithms.gumbel_alphazero.algorithm import (
            GumbelAlphaZeroAlgorithm,
        )

        return GumbelAlphaZeroAlgorithm(
            game=game,
            search_backend=search_backend,
            search_config=config.search,
            temperature_moves=config.selfplay.temperature_moves,
        )
    available = ", ".join(registered_algorithms())
    raise KeyError(
        f"unknown algorithm {config.algorithm.name!r}; available algorithms: {available}"
    )
