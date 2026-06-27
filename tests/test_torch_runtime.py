from __future__ import annotations

import json
from pathlib import Path

from gumbel_az.config import load_config
from gumbel_az.execution import SingleProcessExecutionBackend
from gumbel_az.replay import ReplayReader
from gumbel_az.runtime import RuntimeBackend, detect_runtime_backend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def test_runtime_backend_detects_torch() -> None:
    backend = detect_runtime_backend()

    assert backend.torch_available
    assert backend.name == "torch"
    assert backend.device in {"cpu", "cuda", "mps"}


def test_single_process_uses_torch_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "selfplay.games_per_iteration=1",
            "selfplay.batch_size=1",
            "replay.min_samples_to_train=1",
            "training.batch_size=4",
            "training.steps_per_iteration=1",
            "training.checkpoint_every_steps=1",
            "stop.max_iterations=1",
            "stop.max_games=1",
            "stop.max_train_steps=1",
            "search.simulations_per_move=2",
            "eval.games=2",
        ],
    )
    monkeypatch.setattr(
        "gumbel_az.execution.single_process.detect_runtime_backend",
        lambda: RuntimeBackend(
            name="torch",
            torch_available=True,
            device="cpu",
            device_count=1,
            reason="test forced PyTorch",
        ),
    )

    result = SingleProcessExecutionBackend().run(config)

    state = json.loads((result.run_dir / "run_state.json").read_text(encoding="utf-8"))
    events = (result.run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8")
    metrics = (result.run_dir / "logs" / "metrics.jsonl").read_text(encoding="utf-8")

    assert result.status == "completed"
    assert state["runtime_backend"] == "torch"
    assert "test forced PyTorch" in state["runtime_backend_reason"]
    assert "runtime_backend_selected" in events
    assert '"runtime_backend": "torch"' in events
    assert "runtime_backend_is_torch" in metrics
    assert (result.run_dir / "replay" / "index.json").exists()
    assert (result.run_dir / "checkpoints" / "latest.json").exists()
    samples = ReplayReader(result.run_dir / "replay").read_all()
    assert samples
    assert samples[0]["search_stats"]["backend"] == "torch_gumbel"
