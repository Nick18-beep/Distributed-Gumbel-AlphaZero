"""Gymnasium adapter placeholder for compatibility environments."""

from __future__ import annotations


class GymnasiumGameAdapter:
    def __init__(self, env_id: str) -> None:
        self.env_id = env_id
        raise NotImplementedError("Gymnasium adapter is a future compatibility adapter")
