"""PyTorch trainer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

import numpy as np
import torch

from gumbel_az.config.schema import AppConfig
from gumbel_az.envs import create_game
from gumbel_az.logging import MetricWriter
from gumbel_az.model import create_network
from gumbel_az.model.checkpoint import CheckpointManager
from gumbel_az.model.optimizer import create_optimizer
from gumbel_az.replay.reader import ReplayReader
from gumbel_az.replay.sampler import ReplaySampler
from gumbel_az.runtime import detect_torch_runtime
from gumbel_az.training.train_state import TorchTrainState, train_step


@dataclass(frozen=True)
class TrainLoopResult:
    state: TorchTrainState
    checkpoint_version: int
    steps: int
    samples_seen: int
    samples_per_sec: float
    replay_sample_age_mean: float
    latest_metrics: dict[str, float]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sample_age_seconds(samples: list[dict[str, Any]]) -> float:
    ages = []
    now = _utc_now()
    for sample in samples:
        timestamp = sample.get("timestamp")
        if not isinstance(timestamp, str):
            continue
        try:
            created = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        ages.append((now - created).total_seconds())
    return float(np.mean(ages)) if ages else 0.0


def _checkpoint_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    original = getattr(model, "_orig_mod", None)
    if isinstance(original, torch.nn.Module):
        return original.state_dict()
    return model.state_dict()


def _config_hash(config: AppConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _maybe_compile_model(
    model: torch.nn.Module,
    *,
    compile_policy: str,
    device: torch.device,
    run_name: str,
) -> tuple[torch.nn.Module, str]:
    if compile_policy == "off":
        return model, "eager"
    if not hasattr(torch, "compile"):
        return model, "fallback"
    should_compile = compile_policy == "on"
    if compile_policy == "auto":
        is_debug_run = "debug" in run_name.lower()
        should_compile = device.type == "cuda" and not is_debug_run and sys.platform != "darwin"
    if not should_compile:
        return model, "eager"
    if device.type == "cuda" and importlib.util.find_spec("triton") is None:
        return model, "fallback" if compile_policy == "on" else "eager"
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
    except Exception:
        return model, "fallback"
    if isinstance(compiled, torch.nn.Module):
        return compiled, "compiled"
    return model, "fallback"


def _compiled_original_model(model: torch.nn.Module) -> torch.nn.Module | None:
    original = getattr(model, "_orig_mod", None)
    return original if isinstance(original, torch.nn.Module) else None


def _load_model_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    original = _compiled_original_model(model)
    target = original if original is not None else model
    target.load_state_dict(state_dict)


class Trainer:
    def __init__(
        self,
        config: AppConfig,
        *,
        replay_reader: ReplayReader,
        checkpoint_manager: CheckpointManager,
        metric_writer: MetricWriter | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        runtime = detect_torch_runtime()
        self.device = torch.device(device or runtime.device)
        self.game = create_game(config.game.name)
        self.network = create_network(config.model, num_actions=self.game.num_actions)
        self.replay_reader = replay_reader
        self.replay_sampler = ReplaySampler(replay_reader, config.replay.window_samples)
        self.checkpoint_manager = checkpoint_manager
        self.metric_writer = metric_writer
        self.model = self.network.init(
            config.run.seed,
            self.game.observation_shape,
            self.game.num_actions,
            device=self.device,
        )
        self.model, compile_mode = _maybe_compile_model(
            self.model,
            compile_policy=config.training.compile,
            device=self.device,
            run_name=config.run.name,
        )
        if compile_mode == "compiled":
            try:
                self.model.eval()
                warmup = torch.zeros(
                    (1, *self.game.observation_shape),
                    dtype=torch.float32,
                    device=self.device,
                )
                with torch.inference_mode():
                    self.model(warmup)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            except Exception:
                original = _compiled_original_model(self.model)
                if original is None:
                    raise
                self.model = original
                compile_mode = "fallback"
        self.optimizer, self.schedule = create_optimizer(self.model, config.training)
        scaler_factory = getattr(torch.amp, "GradScaler", None)
        scaler = (
            scaler_factory("cuda")
            if self.device.type == "cuda" and scaler_factory is not None
            else None
        )
        self.state = TorchTrainState(
            model=self.model,
            optimizer=self.optimizer,
            schedule=self.schedule,
            step=0,
            device=self.device,
            scaler=scaler,
            compile_enabled=compile_mode == "compiled",
            compile_mode=compile_mode,
        )

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint.get("state", {})
        model_state = state.get("model_state_dict")
        if not isinstance(model_state, dict):
            raise ValueError("checkpoint is missing model_state_dict")
        _load_model_state_dict(self.state.model, model_state)
        optimizer_state = state.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            self.state.optimizer.load_state_dict(optimizer_state)
        scaler_state = state.get("scaler_state_dict")
        if self.state.scaler is not None and isinstance(scaler_state, dict):
            self.state.scaler.load_state_dict(scaler_state)
        scheduler_state = state.get("scheduler_state_dict")
        if isinstance(scheduler_state, dict):
            self.state.schedule.base_lr = float(
                scheduler_state.get("base_lr", self.state.schedule.base_lr)
            )
            self.state.schedule.warmup_steps = int(
                scheduler_state.get("warmup_steps", self.state.schedule.warmup_steps)
            )
            self.state.schedule.total_steps = int(
                scheduler_state.get("total_steps", self.state.schedule.total_steps)
            )
        self.state.step = int(state.get("step", checkpoint.get("metadata", {}).get("version", 0)))

    def _fallback_state_to_eager(self, state: TorchTrainState) -> TorchTrainState:
        original = _compiled_original_model(state.model)
        if original is None:
            return state
        optimizer_state = state.optimizer.state_dict()
        scaler_state = None if state.scaler is None else state.scaler.state_dict()
        optimizer, schedule = create_optimizer(original, self.config.training)
        try:
            optimizer.load_state_dict(optimizer_state)
        except ValueError:
            pass
        scaler_factory = getattr(torch.amp, "GradScaler", None)
        scaler = (
            scaler_factory("cuda")
            if state.device.type == "cuda" and scaler_factory is not None
            else None
        )
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        return TorchTrainState(
            model=original,
            optimizer=optimizer,
            schedule=schedule,
            step=state.step,
            device=state.device,
            scaler=scaler,
            compile_enabled=False,
            compile_mode="fallback",
        )

    def _save_checkpoint(self, state: TorchTrainState) -> int:
        checkpoint_version = int(state.step)
        self.checkpoint_manager.save(
            version=checkpoint_version,
            state={
                "model_state_dict": _checkpoint_model_state_dict(state.model),
                "optimizer_state_dict": state.optimizer.state_dict(),
                "scheduler_state_dict": {
                    "base_lr": state.schedule.base_lr,
                    "warmup_steps": state.schedule.warmup_steps,
                    "total_steps": state.schedule.total_steps,
                },
                "step": state.step,
                "scaler_state_dict": None if state.scaler is None else state.scaler.state_dict(),
            },
            metadata={
                "training_step": checkpoint_version,
                "game": self.game.name,
                "algorithm": self.config.algorithm.name,
                "model": self.config.model.name,
                "runtime": "torch",
                "device": str(self.device),
                "compile_mode": state.compile_mode,
                "config_hash": _config_hash(self.config),
            },
        )
        return checkpoint_version

    def _augmented_samples(self) -> list[dict[str, Any]]:
        samples = self.replay_sampler.samples()
        augmented: list[dict[str, Any]] = []
        for sample in samples:
            augmented.extend(self.game.symmetries(sample))
        return augmented

    def _sample_batch(self, tensors: dict[str, torch.Tensor], step: int) -> dict[str, torch.Tensor]:
        return self.replay_sampler.sample_tensors(
            tensors,
            batch_size=self.config.training.batch_size,
            seed=self.config.run.seed + step,
            replace_if_needed=True,
            device=self.device,
        )

    def run(self, *, max_steps: int | None = None) -> TrainLoopResult:
        steps = max_steps or self.config.training.steps_per_iteration
        samples = self._augmented_samples()
        sample_arrays = self.replay_sampler.arrays_from(samples)
        sample_tensors = self.replay_sampler.tensors_from_arrays(
            sample_arrays,
            pin_memory=self.device.type == "cuda",
        )
        replay_age = _sample_age_seconds(samples)
        start = perf_counter()
        latest_metrics: dict[str, float] = {}
        state = self.state
        for _ in range(steps):
            batch = self._sample_batch(sample_tensors, int(state.step))
            try:
                state, latest_metrics = train_step(state, batch, self.config.training)
            except Exception:
                if not state.compile_enabled:
                    raise
                state = self._fallback_state_to_eager(state)
                batch = self._sample_batch(sample_tensors, int(state.step))
                state, latest_metrics = train_step(state, batch, self.config.training)
            if self.metric_writer is not None:
                self.metric_writer.write_metrics(
                    int(state.step),
                    {
                        "policy_loss": latest_metrics["policy_loss"],
                        "value_loss": latest_metrics["value_loss"],
                        "total_loss": latest_metrics["total_loss"],
                        "learning_rate": latest_metrics["learning_rate"],
                        "grad_norm": latest_metrics["grad_norm"],
                        "amp_overflow": latest_metrics["amp_overflow"],
                        "replay_sample_age_mean": replay_age,
                        "compile_mode": state.compile_mode,
                    },
                )
            if int(state.step) % self.config.training.checkpoint_every_steps == 0:
                self._save_checkpoint(state)
        self.state = state
        elapsed = perf_counter() - start
        checkpoint_version = int(state.step)
        if checkpoint_version % self.config.training.checkpoint_every_steps != 0:
            checkpoint_version = self._save_checkpoint(state)
        return TrainLoopResult(
            state=state,
            checkpoint_version=checkpoint_version,
            steps=steps,
            samples_seen=steps * self.config.training.batch_size,
            samples_per_sec=(steps * self.config.training.batch_size) / max(elapsed, 1.0e-9),
            replay_sample_age_mean=replay_age,
            latest_metrics=latest_metrics,
        )


def greedy_action_from_model(
    model: torch.nn.Module,
    observation: np.ndarray,
    legal: np.ndarray,
    *,
    device: torch.device | str = "cpu",
) -> int:
    model.eval()
    device = torch.device(device)
    with torch.inference_mode():
        obs = torch.as_tensor(observation[None, ...], dtype=torch.float32, device=device)
        output = model(obs)
        logits = output.policy_logits[0].detach().cpu().numpy()
    masked = np.where(np.asarray(legal, dtype=bool), logits, -np.inf)
    return int(np.argmax(masked))
