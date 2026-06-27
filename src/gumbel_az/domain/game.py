"""Game adapter protocol."""

from __future__ import annotations

from typing import Any, Protocol


class GameAdapter(Protocol):
    name: str
    num_players: int
    num_actions: int
    observation_shape: tuple[int, ...]
    max_moves: int
    supports_jit: bool
    supports_vmap: bool

    def init(self, rng_key: Any = None) -> Any: ...

    def legal_action_mask(self, state: Any) -> Any: ...

    def step(self, state: Any, action: int) -> Any: ...

    def is_terminal(self, state: Any) -> Any: ...

    def current_player(self, state: Any) -> Any: ...

    def canonical_observation(self, state: Any) -> Any: ...

    def terminal_value(self, state: Any) -> Any: ...

    def rewards(self, state: Any) -> Any: ...

    def symmetries(self, sample: dict[str, Any]) -> list[dict[str, Any]]: ...

    def render_text(self, state: Any) -> str: ...
