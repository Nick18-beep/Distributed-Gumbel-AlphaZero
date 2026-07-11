"""PyTorch train state and train step."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from gumbel_az.config.schema import TrainingConfig
from gumbel_az.model.loss import total_loss
from gumbel_az.model.optimizer import WarmupCosineSchedule


@dataclass
class TorchTrainState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    schedule: WarmupCosineSchedule
    step: int
    device: torch.device
    scaler: Any | None = None
    compile_enabled: bool = False
    compile_mode: str = "eager"


def _global_grad_norm(model: torch.nn.Module) -> float:
    norms = [param.grad.detach().norm(2) for param in model.parameters() if param.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).detach().cpu())


def _validate_grad_norm(value: float, *, amp_overflow_is_managed: bool) -> tuple[float, bool]:
    if math.isfinite(value):
        return value, False
    if amp_overflow_is_managed:
        return 0.0, True
    raise FloatingPointError(f"non-finite gradient norm: {value}")


def train_step(
    state: TorchTrainState,
    batch: dict[str, torch.Tensor],
    config: TrainingConfig,
) -> tuple[TorchTrainState, dict[str, float]]:
    state.model.train()
    lr = state.schedule(state.step)
    for group in state.optimizer.param_groups:
        group["lr"] = lr
    state.optimizer.zero_grad(set_to_none=True)
    use_amp = state.device.type == "cuda"
    with torch.autocast(device_type=state.device.type, enabled=use_amp):
        outputs = state.model(batch["observation"])
        loss, loss_metrics = total_loss(outputs, batch)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite training loss: {float(loss.detach().cpu())}")
    if state.scaler is not None and use_amp:
        state.scaler.scale(loss).backward()
        state.scaler.unscale_(state.optimizer)
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), config.gradient_clip_norm)
        grad_norm, amp_overflow = _validate_grad_norm(
            _global_grad_norm(state.model),
            amp_overflow_is_managed=True,
        )
        state.scaler.step(state.optimizer)
        state.scaler.update()
    else:
        loss.backward()  # type: ignore[no-untyped-call]
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), config.gradient_clip_norm)
        grad_norm, amp_overflow = _validate_grad_norm(
            _global_grad_norm(state.model),
            amp_overflow_is_managed=False,
        )
        state.optimizer.step()
    state.step += 1
    metrics = {key: float(value.detach().cpu()) for key, value in loss_metrics.items()}
    metrics.update(
        {
            "loss": float(loss.detach().cpu()),
            "learning_rate": float(lr),
            "grad_norm": grad_norm,
            "amp_overflow": float(amp_overflow),
        }
    )
    return state, metrics
