"""Optimizer construction."""

from __future__ import annotations

import optax

from gumbel_az.config.schema import TrainingConfig


def learning_rate_schedule(config: TrainingConfig) -> optax.Schedule:
    decay_steps = max(config.steps_per_iteration, 1)
    schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=decay_steps,
        alpha=0.1,
    )
    warmup_steps = getattr(config, "warmup_steps", 0)
    if warmup_steps:
        schedule = optax.join_schedules(
            [
                optax.linear_schedule(0.0, config.learning_rate, warmup_steps),
                optax.cosine_decay_schedule(
                    init_value=config.learning_rate,
                    decay_steps=decay_steps,
                    alpha=0.1,
                ),
            ],
            boundaries=[warmup_steps],
        )
    return schedule


def create_optimizer(config: TrainingConfig) -> tuple[optax.GradientTransformation, optax.Schedule]:
    schedule = learning_rate_schedule(config)
    transforms: list[optax.GradientTransformation] = []
    clip_norm = getattr(config, "gradient_clip_norm", None)
    if clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(clip_norm))
    transforms.append(
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config.weight_decay,
        )
    )
    return optax.chain(*transforms), schedule
