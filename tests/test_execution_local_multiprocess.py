from __future__ import annotations

import json
from pathlib import Path

import gumbel_az.execution.local_multiprocess as local_multiprocess_module
from gumbel_az.config import load_config
from gumbel_az.execution.local_multiprocess import LocalMultiprocessExecutionBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def test_local_multiprocess_backend_smoke(tmp_path: Path) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "execution.backend=local_multiprocess",
            "selfplay.games_per_iteration=1",
            "stop.max_games=1",
            "search.simulations_per_move=2",
            "training.batch_size=4",
            "training.steps_per_iteration=1",
            "stop.max_train_steps=1",
            "eval.games=2",
        ],
    )

    result = LocalMultiprocessExecutionBackend().run(config)

    state = json.loads((result.run_dir / "run_state.json").read_text(encoding="utf-8"))
    events = (result.run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert result.status == "completed"
    assert state["backend"] == "local_multiprocess"
    assert state["worker_processes_started"] == 1
    assert state["games_seen"] == 1
    assert state["train_step"] == 1
    assert (result.run_dir / "replay" / "index.json").exists()
    assert (result.run_dir / "checkpoints" / "latest.json").exists()
    assert "worker_process_completed" in events


def test_local_multiprocess_records_worker_failure(tmp_path: Path) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "execution.backend=local_multiprocess",
            "game.name=missing_game",
            "selfplay.games_per_iteration=1",
            "stop.max_games=1",
        ],
    )

    try:
        LocalMultiprocessExecutionBackend().run(config)
    except RuntimeError as exc:
        assert "missing_game" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("worker failure should propagate to parent")

    run_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert run_dirs
    state = json.loads((run_dirs[-1] / "run_state.json").read_text(encoding="utf-8"))
    events = (run_dirs[-1] / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert state["status"] == "failed"
    assert "worker_error" in state
    assert "worker_process_failed" in events


def test_local_multiprocess_records_parent_training_failure(tmp_path: Path, monkeypatch) -> None:
    class FailingOrchestrator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self):
            raise RuntimeError("trainer boom")

    monkeypatch.setattr(local_multiprocess_module, "RunOrchestrator", FailingOrchestrator)
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "execution.backend=local_multiprocess",
            "selfplay.games_per_iteration=1",
            "stop.max_games=1",
            "search.simulations_per_move=2",
            "training.batch_size=4",
            "training.steps_per_iteration=1",
            "stop.max_train_steps=1",
            "eval.games=2",
        ],
    )

    try:
        LocalMultiprocessExecutionBackend().run(config)
    except RuntimeError as exc:
        assert "trainer boom" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("parent training failure should propagate")

    run_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert run_dirs
    state = json.loads((run_dirs[-1] / "run_state.json").read_text(encoding="utf-8"))
    events = (run_dirs[-1] / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert state["status"] == "failed"
    assert state["failure_stage"] == "local_multiprocess_training"
    assert state["error"] == "RuntimeError('trainer boom')"
    assert "local_multiprocess_training_failed" in events
