"""Train state and jitted train step."""

from __future__ import annotations

from collections.abc import Callable

import flax.struct
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from gumbel_az.model.common import NetworkOutput
from gumbel_az.model.loss import total_loss


class GAZTrainState(TrainState):
    apply_fn: flax.struct.PyTreeNode = flax.struct.field(pytree_node=False)


def create_train_state(
    *,
    params: dict,
    apply_fn: Callable[[dict, jax.Array, bool], NetworkOutput],
    tx: optax.GradientTransformation,
) -> GAZTrainState:
    return GAZTrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def _global_norm(tree) -> jax.Array:
    leaves = jax.tree.leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


@jax.jit
def train_step(
    state: GAZTrainState,
    batch: dict[str, jax.Array],
    learning_rate: jax.Array,
) -> tuple[GAZTrainState, dict[str, jax.Array]]:
    def loss_fn(params: dict) -> tuple[jax.Array, dict[str, jax.Array]]:
        outputs = state.apply_fn(params, batch["observation"], True)
        return total_loss(outputs, batch)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    metrics = {
        **metrics,
        "loss": loss,
        "learning_rate": learning_rate,
        "grad_norm": _global_norm(grads),
    }
    return new_state, metrics
