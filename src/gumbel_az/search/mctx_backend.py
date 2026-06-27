"""MCTX-backed Gumbel search."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import mctx

from gumbel_az.config.schema import SearchConfig
from gumbel_az.model.common import NetworkOutput
from gumbel_az.search.masking import apply_legal_mask
from gumbel_az.search.outputs import SearchOutput


class MctxSearchBackend:
    name = "mctx"

    def _qtransform(self, config: SearchConfig):
        if config.q_transform == "completed_by_mix_value":
            return mctx.qtransform_completed_by_mix_value
        if config.q_transform == "parent_siblings":
            return mctx.qtransform_by_parent_and_siblings
        raise ValueError(f"unsupported q_transform: {config.q_transform}")

    def search(
        self,
        *,
        root_observation: jax.Array,
        root_legal_mask: jax.Array,
        network_apply: Callable[[jax.Array], NetworkOutput],
        recurrent_fn: Callable[[Any, jax.Array, jax.Array, Any], tuple[Any, Any]],
        rng_key: jax.Array,
        config: SearchConfig,
        root_embedding: Any | None = None,
    ) -> SearchOutput:
        if not hasattr(mctx, "gumbel_muzero_policy"):
            raise RuntimeError("installed mctx does not expose gumbel_muzero_policy")

        network_output = network_apply(root_observation)
        if root_legal_mask.dtype != jnp.bool_:
            raise TypeError(f"root_legal_mask must be bool, got {root_legal_mask.dtype}")
        if network_output.policy_logits.shape != root_legal_mask.shape:
            raise ValueError(
                "root_legal_mask shape must match policy logits shape: "
                f"{root_legal_mask.shape} != {network_output.policy_logits.shape}"
            )
        try:
            has_legal_action = bool(jnp.all(jnp.any(root_legal_mask, axis=-1)))
        except jax.errors.TracerBoolConversionError:
            has_legal_action = True
        if not has_legal_action:
            raise ValueError("search requires at least one legal action per root")
        prior_logits = apply_legal_mask(network_output.policy_logits, root_legal_mask)
        embedding = root_observation if root_embedding is None else root_embedding
        root = mctx.RootFnOutput(
            prior_logits=prior_logits,
            value=network_output.value,
            embedding=embedding,
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.simulations_per_move,
            invalid_actions=~root_legal_mask,
            qtransform=self._qtransform(config),
            max_num_considered_actions=config.max_num_considered_actions,
            gumbel_scale=config.gumbel_scale,
        )
        policy_target = jnp.where(root_legal_mask, policy_output.action_weights, 0.0)
        normalizer = jnp.sum(policy_target, axis=-1, keepdims=True)
        policy_target = policy_target / jnp.maximum(normalizer, 1.0e-8)

        tree = policy_output.search_tree
        visit_counts = getattr(tree, "children_visits", jnp.zeros_like(policy_target))
        q_values = getattr(tree, "children_qvalues", jnp.zeros_like(policy_target))
        if visit_counts.ndim == policy_target.ndim + 1:
            visit_counts = visit_counts[:, 0, :]
        if q_values.ndim == policy_target.ndim + 1:
            q_values = q_values[:, 0, :]

        return SearchOutput(
            policy_target=policy_target,
            selected_action=policy_output.action,
            root_value=network_output.value,
            visit_counts=visit_counts,
            q_values=q_values,
            prior_logits=prior_logits,
            search_metadata={
                "num_simulations": jnp.asarray(config.simulations_per_move, dtype=jnp.int32),
                "max_num_considered_actions": jnp.asarray(
                    config.max_num_considered_actions,
                    dtype=jnp.int32,
                ),
            },
        )
