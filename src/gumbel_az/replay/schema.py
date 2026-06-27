"""Replay schema constants."""

from __future__ import annotations

SCHEMA_VERSION = 1

REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "timestamp",
        "game_name",
        "algorithm_name",
        "state_or_observation",
        "legal_action_mask",
        "policy_target",
        "value_target",
        "to_play",
        "move_index",
        "game_id",
        "model_version",
        "search_stats",
    }
)
