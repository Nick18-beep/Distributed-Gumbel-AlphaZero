"""PyTorch-native Gumbel AlphaZero style root search."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from gumbel_az.config.schema import SearchConfig
from gumbel_az.search.masking import apply_legal_mask
from gumbel_az.search.outputs import SearchOutput
from gumbel_az.search.q_transform import completed_by_mix_value
from gumbel_az.search.sequential_halving import sequential_halving, top_k_candidates


class TorchGumbelSearchBackend:
    name = "torch_gumbel"

    def __init__(self, game: Any | None = None, device: torch.device | str | None = None) -> None:
        self.game = game
        self.device = torch.device(device or "cpu")

    def _child_values(
        self,
        *,
        root_embedding: Any | None,
        network_apply,
        batch_size: int,
        num_actions: int,
    ) -> torch.Tensor:
        if self.game is None or root_embedding is None:
            return torch.zeros((batch_size, num_actions), dtype=torch.float32, device=self.device)

        q_values = torch.zeros((batch_size, num_actions), dtype=torch.float32, device=self.device)
        is_tuple_batch = isinstance(root_embedding, tuple) and not hasattr(
            root_embedding,
            "_fields",
        )
        states = (
            list(root_embedding)
            if isinstance(root_embedding, list) or is_tuple_batch
            else [root_embedding]
        )
        if len(states) != batch_size:
            raise ValueError(
                "root_embedding batch size must match root_observation batch size: "
                f"{len(states)} != {batch_size}"
            )
        child_observations: list[np.ndarray] = []
        child_refs: list[tuple[int, int, int]] = []
        for batch_index, state in enumerate(states):
            current_player = int(self.game.current_player(state))
            legal = np.asarray(self.game.legal_action_mask(state), dtype=bool)
            for action in np.flatnonzero(legal):
                next_state = self.game.step(state, int(action))
                rewards = np.asarray(self.game.rewards(next_state), dtype=np.float32)
                if bool(self.game.is_terminal(next_state)):
                    q_values[batch_index, int(action)] = float(rewards[current_player])
                    continue
                child_observations.append(self.game.canonical_observation(next_state))
                child_refs.append((batch_index, int(action), current_player))
        if not child_observations:
            return q_values

        obs = torch.as_tensor(
            np.stack(child_observations).astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            output = network_apply(obs)
            values = output.value.detach()
        for index, (batch_index, action, _player) in enumerate(child_refs):
            q_values[batch_index, action] = -values[index]
        return q_values

    def _uniform_like(self, prior_logits: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        try:
            return torch.rand(
                prior_logits.shape,
                generator=rng,
                device=self.device,
                dtype=prior_logits.dtype,
            )
        except RuntimeError:
            return torch.rand(
                prior_logits.shape,
                generator=rng,
                dtype=prior_logits.dtype,
            ).to(self.device)

    def search(
        self,
        *,
        root_observation: torch.Tensor,
        root_legal_mask: torch.Tensor,
        network_apply,
        rng: torch.Generator,
        config: SearchConfig,
        root_embedding: Any | None = None,
    ) -> SearchOutput:
        root_observation = root_observation.to(self.device, non_blocking=True)
        root_legal_mask = root_legal_mask.to(self.device, non_blocking=True).bool()
        if root_legal_mask.ndim != 2:
            raise ValueError("root_legal_mask must have shape [batch, actions]")
        if not bool(torch.all(torch.any(root_legal_mask, dim=-1)).item()):
            raise ValueError("search requires at least one legal action per root")

        with torch.inference_mode():
            network_output = network_apply(root_observation)
        if network_output.policy_logits.shape != root_legal_mask.shape:
            raise ValueError(
                "root_legal_mask shape must match policy logits shape: "
                f"{tuple(root_legal_mask.shape)} != {tuple(network_output.policy_logits.shape)}"
            )
        prior_logits = apply_legal_mask(network_output.policy_logits, root_legal_mask)
        q_values = self._child_values(
            root_embedding=root_embedding,
            network_apply=network_apply,
            batch_size=root_legal_mask.shape[0],
            num_actions=root_legal_mask.shape[1],
        )
        q_scores = completed_by_mix_value(q_values, root_legal_mask)
        uniform = self._uniform_like(prior_logits, rng).clamp_(1.0e-8, 1.0 - 1.0e-8)
        gumbel = -torch.log(-torch.log(uniform)) * config.gumbel_scale
        root_scores = prior_logits + gumbel + q_scores
        candidates = top_k_candidates(
            root_scores,
            root_legal_mask,
            config.max_num_considered_actions,
        )
        selected = sequential_halving(
            root_scores,
            root_legal_mask,
            config.max_num_considered_actions,
        )
        improved_logits = apply_legal_mask(prior_logits + q_scores, root_legal_mask)
        candidate_mask = torch.zeros_like(root_legal_mask)
        candidate_mask.scatter_(dim=-1, index=candidates, value=True)
        improved_logits = apply_legal_mask(improved_logits, root_legal_mask & candidate_mask)
        policy_target = torch.softmax(improved_logits, dim=-1)
        policy_target = torch.where(root_legal_mask, policy_target, torch.zeros_like(policy_target))
        policy_target = policy_target / policy_target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        visit_counts = torch.zeros_like(policy_target)
        visit_counts.scatter_add_(
            dim=-1,
            index=candidates,
            src=torch.ones_like(candidates, dtype=policy_target.dtype),
        )
        visit_counts.scatter_add_(
            dim=-1,
            index=selected[:, None],
            src=torch.full(
                (selected.shape[0], 1),
                float(config.simulations_per_move),
                dtype=policy_target.dtype,
                device=self.device,
            ),
        )
        return SearchOutput(
            policy_target=policy_target.detach(),
            selected_action=selected.detach(),
            root_value=network_output.value.detach(),
            visit_counts=visit_counts.detach(),
            q_values=q_values.detach(),
            prior_logits=prior_logits.detach(),
            search_metadata={
                "backend": self.name,
                "num_simulations": config.simulations_per_move,
                "max_num_considered_actions": config.max_num_considered_actions,
            },
        )
