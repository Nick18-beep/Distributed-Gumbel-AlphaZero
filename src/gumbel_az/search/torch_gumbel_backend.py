"""PyTorch-native Gumbel AlphaZero style root search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from gumbel_az.config.schema import SearchConfig
from gumbel_az.search.masking import apply_legal_mask
from gumbel_az.search.outputs import SearchOutput
from gumbel_az.search.q_transform import completed_by_mix_value
from gumbel_az.search.sequential_halving import sequential_halving, top_k_candidates


@dataclass
class _MctsNode:
    state: Any
    current_player: int
    legal: np.ndarray
    prior: np.ndarray
    value: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, _MctsNode] = field(default_factory=dict)
    child_visit_count: np.ndarray | None = None
    child_value_sum: np.ndarray | None = None

    def __post_init__(self) -> None:
        actions = int(self.legal.shape[0])
        self.child_visit_count = np.zeros(actions, dtype=np.int32)
        self.child_value_sum = np.zeros(actions, dtype=np.float32)


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

    def _evaluate_state(
        self,
        state: Any,
        network_apply,
        *,
        legal_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        observation = torch.as_tensor(
            self.game.canonical_observation(state)[None, ...],
            dtype=torch.float32,
            device=self.device,
        )
        legal = (
            np.asarray(legal_override, dtype=bool)
            if legal_override is not None
            else np.asarray(self.game.legal_action_mask(state), dtype=bool)
        )
        with torch.inference_mode():
            output = network_apply(observation)
        logits = apply_legal_mask(
            output.policy_logits,
            torch.as_tensor(legal[None, ...], dtype=torch.bool, device=self.device),
        )
        prior = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.float32)
        prior = np.where(legal, prior, 0.0)
        prior_sum = float(prior.sum())
        if prior_sum <= 0.0:
            prior = np.where(legal, 1.0 / max(1, int(legal.sum())), 0.0).astype(np.float32)
        else:
            prior /= prior_sum
        return prior, float(output.value[0].detach().cpu().item())

    def _make_node(
        self,
        state: Any,
        network_apply,
        *,
        legal_override: np.ndarray | None = None,
    ) -> _MctsNode:
        legal = (
            np.asarray(legal_override, dtype=bool)
            if legal_override is not None
            else np.asarray(self.game.legal_action_mask(state), dtype=bool)
        )
        prior, value = self._evaluate_state(state, network_apply, legal_override=legal)
        return _MctsNode(
            state=state,
            current_player=int(self.game.current_player(state)),
            legal=legal,
            prior=prior,
            value=value,
        )

    def _root_gumbel_scores(
        self,
        prior_logits: torch.Tensor,
        legal_mask: torch.Tensor,
        rng: torch.Generator,
        config: SearchConfig,
    ) -> np.ndarray:
        uniform = self._uniform_like(prior_logits, rng).clamp_(1.0e-8, 1.0 - 1.0e-8)
        gumbel = -torch.log(-torch.log(uniform)) * config.gumbel_scale
        scores = apply_legal_mask(prior_logits + gumbel, legal_mask)
        return scores[0].detach().cpu().numpy().astype(np.float32)

    def _select_mcts_action(
        self,
        node: _MctsNode,
        *,
        root_scores: np.ndarray | None,
        candidate_mask: np.ndarray | None,
    ) -> int:
        assert node.child_visit_count is not None
        assert node.child_value_sum is not None
        legal = node.legal if candidate_mask is None else node.legal & candidate_mask
        legal_actions = np.flatnonzero(legal)
        if legal_actions.size == 0:
            legal_actions = np.flatnonzero(node.legal)
        total_visits = max(1, int(node.child_visit_count[legal_actions].sum()))
        scores = np.full(node.legal.shape, -np.inf, dtype=np.float32)
        cpuct = 1.5
        for action in legal_actions:
            visits = int(node.child_visit_count[action])
            q_value = 0.0 if visits == 0 else float(node.child_value_sum[action] / visits)
            prior_bonus = cpuct * float(node.prior[action]) * np.sqrt(total_visits) / (1 + visits)
            scores[action] = q_value + prior_bonus
            if root_scores is not None and visits == 0:
                scores[action] += 0.01 * float(root_scores[action])
        return int(np.argmax(scores))

    def _simulate(
        self,
        node: _MctsNode,
        network_apply,
        *,
        root_scores: np.ndarray | None = None,
        candidate_mask: np.ndarray | None = None,
    ) -> float:
        if bool(self.game.is_terminal(node.state)):
            rewards = np.asarray(self.game.rewards(node.state), dtype=np.float32)
            return float(rewards[node.current_player])
        action = self._select_mcts_action(
            node,
            root_scores=root_scores,
            candidate_mask=candidate_mask,
        )
        next_state = self.game.step(node.state, action)
        rewards = np.asarray(self.game.rewards(next_state), dtype=np.float32)
        if bool(self.game.is_terminal(next_state)):
            value = float(rewards[node.current_player])
        else:
            child = node.children.get(action)
            if child is None:
                child = self._make_node(next_state, network_apply)
                node.children[action] = child
                value = -child.value
            else:
                value = -self._simulate(child, network_apply)
        assert node.child_visit_count is not None
        assert node.child_value_sum is not None
        node.visit_count += 1
        node.value_sum += value
        node.child_visit_count[action] += 1
        node.child_value_sum[action] += value
        return value

    def _mcts_root_values(
        self,
        *,
        root_state: Any,
        prior_logits: torch.Tensor,
        legal_mask: torch.Tensor,
        network_apply,
        rng: torch.Generator,
        config: SearchConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        root_legal = legal_mask.detach().cpu().numpy().astype(bool)
        root = self._make_node(root_state, network_apply, legal_override=root_legal)
        root_scores = self._root_gumbel_scores(
            prior_logits[None, :],
            legal_mask[None, :],
            rng,
            config,
        )
        candidate_count = min(config.max_num_considered_actions, int(root.legal.sum()))
        candidate_indices = np.argsort(root_scores)[-candidate_count:]
        candidate_mask = np.zeros_like(root.legal, dtype=bool)
        candidate_mask[candidate_indices] = True
        simulations = max(1, int(config.simulations_per_move))
        for _ in range(simulations):
            self._simulate(
                root,
                network_apply,
                root_scores=root_scores,
                candidate_mask=candidate_mask,
            )
        assert root.child_visit_count is not None
        assert root.child_value_sum is not None
        visits = root.child_visit_count.astype(np.float32)
        q_values = np.zeros_like(visits, dtype=np.float32)
        visited = root.child_visit_count > 0
        q_values[visited] = root.child_value_sum[visited] / root.child_visit_count[visited]
        for action in np.flatnonzero(root.legal & ~visited):
            next_state = self.game.step(root.state, int(action))
            if bool(self.game.is_terminal(next_state)):
                rewards = np.asarray(self.game.rewards(next_state), dtype=np.float32)
                q_values[action] = float(rewards[root.current_player])
        visits = np.where(root.legal, visits, 0.0).astype(np.float32)
        if float(visits.sum()) <= 0.0:
            visits = np.where(root.legal, 1.0, 0.0).astype(np.float32)
        return visits, q_values, candidate_mask

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
        if self.game is not None and root_embedding is not None:
            is_tuple_batch = isinstance(root_embedding, tuple) and not hasattr(
                root_embedding,
                "_fields",
            )
            states = (
                list(root_embedding)
                if isinstance(root_embedding, list) or is_tuple_batch
                else [root_embedding]
            )
            if len(states) != root_legal_mask.shape[0]:
                raise ValueError(
                    "root_embedding batch size must match root_observation batch size: "
                    f"{len(states)} != {root_legal_mask.shape[0]}"
                )
            visit_arrays = []
            q_arrays = []
            candidate_arrays = []
            for batch_index, state in enumerate(states):
                visits, q_array, candidate_array = self._mcts_root_values(
                    root_state=state,
                    prior_logits=prior_logits[batch_index],
                    legal_mask=root_legal_mask[batch_index],
                    network_apply=network_apply,
                    rng=rng,
                    config=config,
                )
                visit_arrays.append(visits)
                q_arrays.append(q_array)
                candidate_arrays.append(candidate_array)
            visit_counts = torch.as_tensor(
                np.stack(visit_arrays),
                dtype=torch.float32,
                device=self.device,
            )
            q_values = torch.as_tensor(
                np.stack(q_arrays),
                dtype=torch.float32,
                device=self.device,
            )
            candidate_mask = torch.as_tensor(
                np.stack(candidate_arrays),
                dtype=torch.bool,
                device=self.device,
            )
            policy_target = visit_counts / visit_counts.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            selected = torch.argmax(visit_counts, dim=-1)
        else:
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
            policy_target = torch.where(
                root_legal_mask,
                policy_target,
                torch.zeros_like(policy_target),
            )
            policy_target = policy_target / policy_target.sum(dim=-1, keepdim=True).clamp_min(
                1.0e-8
            )
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
