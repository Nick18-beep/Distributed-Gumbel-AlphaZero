"""PyTorch optimizer and learning-rate schedule helpers."""

from __future__ import annotations

import math

import torch

from gumbel_az.config.schema import TrainingConfig


class WarmupCosineSchedule:
    def __init__(self, config: TrainingConfig) -> None:
        self.base_lr = config.learning_rate
        self.warmup_steps = max(0, config.warmup_steps)
        self.total_steps = max(config.steps_per_iteration, self.warmup_steps + 1)

    def __call__(self, step: int) -> float:
        step = max(0, int(step))
        if self.warmup_steps and step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(self.warmup_steps)
        denom = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / denom))
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def create_optimizer(
    model: torch.nn.Module,
    config: TrainingConfig,
) -> tuple[torch.optim.Optimizer, WarmupCosineSchedule]:
    if config.optimizer != "adamw":
        raise ValueError(f"unsupported optimizer: {config.optimizer}")
    schedule = WarmupCosineSchedule(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    return optimizer, schedule
