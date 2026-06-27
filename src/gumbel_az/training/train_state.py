"""PyTorch train state and train step."""

from __future__ import annotations

from dataclasses import dataclass

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
    scaler: torch.amp.GradScaler | None = None
    compile_enabled: bool = False
    compile_mode: str = "eager"


def _global_grad_norm(model: torch.nn.Module) -> float:
    norms = [
        param.grad.detach().norm(2)
        for param in model.parameters()
        if param.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).detach().cpu())


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
        grad_norm = _global_grad_norm(state.model)
        state.scaler.step(state.optimizer)
        state.scaler.update()
    else:
        loss.backward()
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), config.gradient_clip_norm)
        grad_norm = _global_grad_norm(state.model)
        state.optimizer.step()
    state.step += 1
    metrics = {key: float(value.detach().cpu()) for key, value in loss_metrics.items()}
    metrics.update(
        {
            "loss": float(loss.detach().cpu()),
            "learning_rate": float(lr),
            "grad_norm": grad_norm,
        }
    )
    return state, metrics
