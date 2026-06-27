from __future__ import annotations

from pathlib import Path

import numpy as np

from gumbel_az.config import load_config
from gumbel_az.replay import ReplayReader, ReplayWriter
from gumbel_az.selfplay import SelfPlayWorker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def test_selfplay_generates_replay_and_is_deterministic(tmp_path: Path) -> None:
    config = load_config(CONFIG, ["search.simulations_per_move=4"])
    worker_a = SelfPlayWorker(config, replay_writer=ReplayWriter(tmp_path / "a" / "replay"))
    worker_b = SelfPlayWorker(config, replay_writer=ReplayWriter(tmp_path / "b" / "replay"))

    trajectories_a, result_a = worker_a.play_batch(1, seed=123)
    trajectories_b, result_b = worker_b.play_batch(1, seed=123)
    samples = ReplayReader(tmp_path / "a" / "replay").read_all()

    assert result_a.games == 1
    assert result_a.positions == len(samples)
    assert result_a.illegal_action_rate == 0.0
    assert len(trajectories_a[0].steps) > 0
    assert [step.action for step in trajectories_a[0].steps] == [
        step.action for step in trajectories_b[0].steps
    ]
    for sample in samples:
        legal = np.asarray(sample["legal_action_mask"], dtype=bool)
        policy = np.asarray(sample["policy_target"], dtype=np.float32)
        assert float(np.sum(policy[~legal])) == 0.0
        assert -1.0 <= sample["value_target"] <= 1.0


def test_selfplay_batch_generates_requested_games(tmp_path: Path) -> None:
    config = load_config(CONFIG, ["search.simulations_per_move=4"])
    worker = SelfPlayWorker(config, replay_writer=ReplayWriter(tmp_path / "replay"))

    trajectories, result = worker.play_batch(2, seed=7)

    assert len(trajectories) == 2
    assert result.games == 2
    assert result.positions == sum(len(trajectory.steps) for trajectory in trajectories)
