from __future__ import annotations

import json
from pathlib import Path

from gumbel_az.config import load_config
from gumbel_az.execution import SingleProcessExecutionBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def test_single_process_backend_initializes_run(tmp_path: Path) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "selfplay.games_per_iteration=1",
            "stop.max_games=1",
            "search.simulations_per_move=4",
            "training.batch_size=4",
            "training.steps_per_iteration=2",
            "stop.max_train_steps=2",
            "eval.games=2",
        ],
    )

    result = SingleProcessExecutionBackend().run(config)

    assert result.status == "completed"
    assert result.run_dir.is_dir()
    assert (result.run_dir / "config.resolved.yaml").exists()
    assert (result.run_dir / "logs" / "events.jsonl").exists()
    assert (result.run_dir / "logs" / "metrics.jsonl").exists()
    state = json.loads((result.run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["backend"] == "single_process"
    assert Path(state["config_path"]).name == "config.resolved.yaml"
    assert state["train_step"] == 2
    assert state["games_seen"] == 1
    assert state["samples_seen"] > 0
    assert state["eval"]["promoted"] is True
    assert state["eval"]["promotion_reason"] == "initial_best"
    assert (result.run_dir / "replay" / "index.json").exists()
    assert (result.run_dir / "checkpoints" / "latest.json").exists()
    assert (result.run_dir / "checkpoints" / "best.json").exists()
    assert (result.run_dir / "eval" / "matches.jsonl").exists()
    events = (result.run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert '"event": "training_completed"' in events
    assert '"train_step": 2' in events


def test_single_process_backend_rejects_other_backend(tmp_path: Path) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "execution.backend=local_multiprocess",
        ],
    )

    backend = SingleProcessExecutionBackend()
    try:
        backend.run(config)
    except ValueError as exc:
        assert "local_multiprocess" in str(exc)
    else:
        raise AssertionError("expected backend mismatch to fail")


def test_single_process_backend_runs_multiple_iterations(tmp_path: Path) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "stop.max_iterations=2",
            "selfplay.games_per_iteration=1",
            "stop.max_games=2",
            "search.simulations_per_move=2",
            "training.batch_size=4",
            "training.steps_per_iteration=1",
            "stop.max_train_steps=2",
            "eval.games=2",
        ],
    )

    result = SingleProcessExecutionBackend().run(config)

    state = json.loads((result.run_dir / "run_state.json").read_text(encoding="utf-8"))
    replay_index = json.loads(
        (result.run_dir / "replay" / "index.json").read_text(encoding="utf-8")
    )
    checkpoint_index = json.loads(
        (result.run_dir / "checkpoints" / "index.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "completed"
    assert state["iterations_completed"] == 2
    assert state["games_seen"] == 2
    assert state["train_step"] == 2
    assert len(replay_index["shards"]) == 2
    assert [entry["version"] for entry in checkpoint_index["checkpoints"]] == [1, 2]


def test_single_process_backend_respects_max_train_steps_before_extra_selfplay(
    tmp_path: Path,
) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "stop.max_iterations=3",
            "selfplay.games_per_iteration=1",
            "stop.max_games=3",
            "search.simulations_per_move=2",
            "training.batch_size=4",
            "training.steps_per_iteration=1",
            "stop.max_train_steps=1",
            "eval.games=2",
        ],
    )

    result = SingleProcessExecutionBackend().run(config)

    state = json.loads((result.run_dir / "run_state.json").read_text(encoding="utf-8"))
    replay_index = json.loads(
        (result.run_dir / "replay" / "index.json").read_text(encoding="utf-8")
    )
    assert state["iterations_completed"] == 1
    assert state["games_seen"] == 1
    assert state["train_step"] == 1
    assert len(replay_index["shards"]) == 1


def test_single_process_backend_can_disable_eval(tmp_path: Path) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "selfplay.games_per_iteration=1",
            "stop.max_games=1",
            "search.simulations_per_move=2",
            "training.batch_size=4",
            "training.steps_per_iteration=1",
            "stop.max_train_steps=1",
            "eval.enabled=false",
        ],
    )

    result = SingleProcessExecutionBackend().run(config)

    state = json.loads((result.run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["eval"] is None
    assert not (result.run_dir / "eval" / "matches.jsonl").exists()
