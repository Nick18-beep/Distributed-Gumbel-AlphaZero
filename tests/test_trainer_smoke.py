from __future__ import annotations

import json
import math
from pathlib import Path

from gumbel_az.config import load_config
from gumbel_az.model.checkpoint import CheckpointManager
from gumbel_az.replay import ReplayReader, ReplayWriter
from gumbel_az.selfplay.worker import SelfPlayWorker
from gumbel_az.training.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def test_trainer_consumes_replay_and_saves_checkpoint(tmp_path: Path) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "selfplay.games_per_iteration=1",
            "search.simulations_per_move=2",
            "training.batch_size=4",
            "training.steps_per_iteration=3",
            "training.checkpoint_every_steps=2",
        ],
    )
    replay_dir = tmp_path / "replay"
    worker = SelfPlayWorker(config, replay_writer=ReplayWriter(replay_dir))
    worker.play_batch(1, config.run.seed)

    checkpoint_manager = CheckpointManager(tmp_path / "checkpoints")
    trainer = Trainer(
        config,
        replay_reader=ReplayReader(replay_dir),
        checkpoint_manager=checkpoint_manager,
    )
    result = trainer.run(max_steps=3)

    assert result.checkpoint_version == 3
    assert result.samples_seen == 12
    assert result.samples_per_sec > 0.0
    assert math.isfinite(result.latest_metrics["total_loss"])
    assert (tmp_path / "checkpoints" / "latest.json").exists()
    index = json.loads((tmp_path / "checkpoints" / "index.json").read_text(encoding="utf-8"))
    assert [entry["version"] for entry in index["checkpoints"]] == [2, 3]


def test_trainer_uses_replacement_for_small_replay(tmp_path: Path) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "selfplay.games_per_iteration=1",
            "search.simulations_per_move=2",
            "training.batch_size=64",
            "training.steps_per_iteration=1",
        ],
    )
    replay_dir = tmp_path / "replay"
    worker = SelfPlayWorker(config, replay_writer=ReplayWriter(replay_dir))
    worker.play_batch(1, config.run.seed)

    result = Trainer(
        config,
        replay_reader=ReplayReader(replay_dir),
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
    ).run(max_steps=1)

    assert result.samples_seen == 64
