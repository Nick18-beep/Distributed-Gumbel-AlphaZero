from __future__ import annotations

from pathlib import Path

from gumbel_az.config import load_config
from gumbel_az.orchestration import LocalScheduler, SchedulerSignals

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def test_scheduler_prioritizes_selfplay_below_low_watermark() -> None:
    config = load_config(DEBUG_CONFIG)
    decision = LocalScheduler(config).decide(SchedulerSignals(replay_samples_available=0))

    assert decision.mode == "prioritize_selfplay"
    assert decision.allow_selfplay
    assert not decision.allow_training
    assert decision.reason == "replay_below_min_samples_to_train"


def test_scheduler_blocks_training_below_min_samples_to_train() -> None:
    config = load_config(DEBUG_CONFIG, ["replay.min_samples_to_train=8"])
    decision = LocalScheduler(config).decide(SchedulerSignals(replay_samples_available=1))

    assert decision.mode == "prioritize_selfplay"
    assert decision.allow_selfplay
    assert not decision.allow_training
    assert decision.reason == "replay_below_min_samples_to_train"


def test_scheduler_allows_training_after_min_samples_below_low_watermark() -> None:
    config = load_config(DEBUG_CONFIG, ["replay.min_samples_to_train=8"])
    decision = LocalScheduler(config).decide(
        SchedulerSignals(replay_samples_available=config.replay.min_samples_to_train)
    )

    assert decision.mode == "prioritize_selfplay"
    assert decision.allow_selfplay
    assert decision.allow_training
    assert decision.reason == "replay_below_low_watermark"


def test_scheduler_backpressure_does_not_train_below_min_samples() -> None:
    config = load_config(DEBUG_CONFIG, ["replay.min_samples_to_train=8"])
    decision = LocalScheduler(config).decide(
        SchedulerSignals(
            replay_samples_available=config.replay.min_samples_to_train - 1,
            replay_write_queue_depth=2,
        )
    )

    assert not decision.allow_selfplay
    assert not decision.allow_training
    assert decision.reason == "replay_write_backpressure"


def test_scheduler_prioritizes_training_above_high_watermark() -> None:
    config = load_config(DEBUG_CONFIG)
    decision = LocalScheduler(config).decide(
        SchedulerSignals(replay_samples_available=config.replay.high_watermark + 1)
    )

    assert decision.mode == "prioritize_training"
    assert not decision.allow_selfplay
    assert decision.allow_training


def test_scheduler_applies_replay_write_backpressure() -> None:
    config = load_config(DEBUG_CONFIG)
    decision = LocalScheduler(config).decide(
        SchedulerSignals(
            replay_samples_available=config.replay.low_watermark,
            replay_write_queue_depth=2,
        )
    )

    assert not decision.allow_selfplay
    assert decision.reason == "replay_write_backpressure"
