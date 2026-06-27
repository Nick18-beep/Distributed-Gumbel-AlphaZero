"""Checkpoint promotion policy."""

from __future__ import annotations

from gumbel_az.eval.arena import EvalResult


def should_promote(
    result: EvalResult,
    *,
    min_games: int,
    promotion_win_rate: float,
) -> bool:
    if result.games < min_games:
        return False
    return result.win_rate >= promotion_win_rate
