"""Local scheduler decisions for single-process orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from gumbel_az.config.schema import AppConfig

SchedulerMode = Literal["prioritize_selfplay", "balanced", "prioritize_training"]


@dataclass(frozen=True)
class SchedulerSignals:
    replay_samples_available: int
    samples_generated_per_sec: float = 0.0
    samples_consumed_per_sec: float = 0.0
    train_steps_per_sec: float = 0.0
    selfplay_queue_depth: int = 0
    replay_write_queue_depth: int = 0
    checkpoint_pending: bool = False
    evaluation_pending: bool = False
    model_staleness: int = 0
    cpu_utilization: float | None = None
    gpu_utilization_if_available: float | None = None
    memory_available: int | None = None


@dataclass(frozen=True)
class SchedulerDecision:
    mode: SchedulerMode
    allow_selfplay: bool
    allow_training: bool
    max_selfplay_batches_in_flight: int
    replay_write_queue_limit: int
    evaluation_games_budget: int
    shard_max_samples: int
    reason: str

    def to_event(self) -> dict[str, Any]:
        return asdict(self)


class LocalScheduler:
    """Simple deterministic control policy for local single-process runs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def decide(self, signals: SchedulerSignals) -> SchedulerDecision:
        replay = self.config.replay
        selfplay = self.config.selfplay
        eval_games = self.config.eval.games if self.config.eval.enabled else 0

        if signals.replay_write_queue_depth > 1:
            return SchedulerDecision(
                mode="balanced",
                allow_selfplay=False,
                allow_training=signals.replay_samples_available >= replay.min_samples_to_train,
                max_selfplay_batches_in_flight=0,
                replay_write_queue_limit=1,
                evaluation_games_budget=0,
                shard_max_samples=max(1, selfplay.shard_max_samples // 2),
                reason="replay_write_backpressure",
            )

        if signals.replay_samples_available < replay.min_samples_to_train:
            return SchedulerDecision(
                mode="prioritize_selfplay",
                allow_selfplay=True,
                allow_training=False,
                max_selfplay_batches_in_flight=1,
                replay_write_queue_limit=1,
                evaluation_games_budget=0,
                shard_max_samples=selfplay.shard_max_samples,
                reason="replay_below_min_samples_to_train",
            )

        if signals.replay_samples_available < replay.low_watermark:
            return SchedulerDecision(
                mode="prioritize_selfplay",
                allow_selfplay=True,
                allow_training=True,
                max_selfplay_batches_in_flight=1,
                replay_write_queue_limit=1,
                evaluation_games_budget=0,
                shard_max_samples=selfplay.shard_max_samples,
                reason="replay_below_low_watermark",
            )

        if signals.evaluation_pending:
            return SchedulerDecision(
                mode="balanced",
                allow_selfplay=True,
                allow_training=True,
                max_selfplay_batches_in_flight=1,
                replay_write_queue_limit=1,
                evaluation_games_budget=eval_games,
                shard_max_samples=selfplay.shard_max_samples,
                reason="evaluation_pending",
            )

        if signals.checkpoint_pending or signals.model_staleness > 1:
            return SchedulerDecision(
                mode="balanced",
                allow_selfplay=True,
                allow_training=True,
                max_selfplay_batches_in_flight=1,
                replay_write_queue_limit=1,
                evaluation_games_budget=0,
                shard_max_samples=selfplay.shard_max_samples,
                reason="checkpoint_or_staleness_pending",
            )

        if signals.replay_samples_available > replay.high_watermark:
            return SchedulerDecision(
                mode="prioritize_training",
                allow_selfplay=False,
                allow_training=True,
                max_selfplay_batches_in_flight=0,
                replay_write_queue_limit=1,
                evaluation_games_budget=0,
                shard_max_samples=selfplay.shard_max_samples,
                reason="replay_above_high_watermark",
            )

        return SchedulerDecision(
            mode="balanced",
            allow_selfplay=True,
            allow_training=True,
            max_selfplay_batches_in_flight=1,
            replay_write_queue_limit=1,
            evaluation_games_budget=eval_games if signals.evaluation_pending else 0,
            shard_max_samples=selfplay.shard_max_samples,
            reason="within_watermarks",
        )
