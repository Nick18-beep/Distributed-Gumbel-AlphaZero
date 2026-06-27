"""Evaluation opponents."""

from __future__ import annotations

from typing import Any

import numpy as np


def random_legal_action(legal_action_mask: Any, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(np.asarray(legal_action_mask, dtype=bool))
    if legal.size == 0:
        raise ValueError("no legal actions available")
    return int(rng.choice(legal))
