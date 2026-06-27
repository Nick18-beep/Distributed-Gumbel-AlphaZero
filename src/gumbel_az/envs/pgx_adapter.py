"""PGX game adapter placeholder."""

from __future__ import annotations


class PGXGameAdapter:
    """Placeholder for future PGX-backed games.

    Connect Four currently uses a custom adapter to keep terminal rewards, canonical
    observations and exhaustive tests under direct project control. PGX remains the
    preferred external-library path for future board-game adapters when semantics match.
    """

    def __init__(self, game_name: str) -> None:
        self.game_name = game_name
        raise NotImplementedError("PGX adapter is scheduled after the custom Connect Four baseline")
