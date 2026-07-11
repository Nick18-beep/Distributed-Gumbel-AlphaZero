"""Registry for game adapters."""

from __future__ import annotations

from collections.abc import Callable

from gumbel_az.domain.game import GameAdapter

GameFactory = Callable[[], GameAdapter]

_REGISTRY: dict[str, GameFactory] = {}


def register_game(name: str, factory: GameFactory) -> None:
    if name in _REGISTRY:
        raise ValueError(f"game already registered: {name}")
    _REGISTRY[name] = factory


def registered_games() -> tuple[str, ...]:
    _ensure_builtin_games()
    return tuple(sorted(_REGISTRY))


def create_game(name: str) -> GameAdapter:
    _ensure_builtin_games()
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        available = ", ".join(registered_games())
        raise KeyError(f"unknown game {name!r}; available games: {available}") from exc


def _ensure_builtin_games() -> None:
    if "connect_four" not in _REGISTRY:
        from gumbel_az.envs.custom.connect_four import ConnectFourGame

        register_game("connect_four", lambda: ConnectFourGame())
