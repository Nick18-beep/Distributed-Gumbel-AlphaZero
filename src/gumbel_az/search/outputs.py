"""Search output data structures."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch


class SearchOutput(NamedTuple):
    policy_target: torch.Tensor
    selected_action: torch.Tensor
    root_value: torch.Tensor
    visit_counts: torch.Tensor
    q_values: torch.Tensor
    prior_logits: torch.Tensor
    search_metadata: dict[str, Any]
