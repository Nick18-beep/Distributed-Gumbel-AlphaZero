from __future__ import annotations

import json
import subprocess
import sys
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
    assert backend.name in {"jax", "torch"}


def test_runtime_backend_can_force_torch(monkeypatch) -> None:
    monkeypatch.setenv("GAZ_FORCE_RUNTIME_BACKEND", "torch")

    backend = detect_runtime_backend()

    assert backend.name == "torch"
    assert "GAZ_FORCE_RUNTIME_BACKEND=torch" in backend.reason


def test_runtime_backend_rejects_invalid_force_value(monkeypatch) -> None:
    monkeypatch.setenv("GAZ_FORCE_RUNTIME_BACKEND", "bad")

    backend = detect_runtime_backend()

    assert backend.name == "none"
    assert "unsupported GAZ_FORCE_RUNTIME_BACKEND" in backend.reason


def test_execution_imports_when_jax_import_is_blocked() -> None:
    code = """
import importlib.abc
import sys

class BlockJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jax' or fullname.startswith('jax.'):
            raise ImportError('blocked jax for fallback test')
        return None

sys.meta_path.insert(0, BlockJax())
from gumbel_az.execution.single_process import SingleProcessExecutionBackend
from gumbel_az.selfplay.torch_fallback import TorchFallbackSelfPlayWorker
print(SingleProcessExecutionBackend.name, TorchFallbackSelfPlayWorker.__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "single_process TorchFallbackSelfPlayWorker" in result.stdout


def test_doctor_reports_fallback_when_jax_import_is_blocked() -> None:
    code = """
import importlib.abc
import sys

class BlockJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jax' or fullname.startswith('jax.'):
            raise ImportError('blocked jax for fallback test')
        return None

sys.meta_path.insert(0, BlockJax())
from gumbel_az.cli.doctor import run_doctor
messages = []
run_doctor(
    fix=False,
    distributed=False,
    cuda=False,
    reporter=lambda message, err=False: messages.append(message),
)
print('\\n'.join(messages))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[WARN] jax:" in result.stdout
    assert "runtime backend: torch" in result.stdout


def test_single_process_uses_torch_fallback_when_jax_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "selfplay.games_per_iteration=1",
            "stop.max_games=1",
            "search.simulations_per_move=4",
        ],
    )
    monkeypatch.setattr(
        "gumbel_az.execution.single_process.detect_runtime_backend",
        lambda: RuntimeBackend(
            name="torch",
            jax_available=False,
            torch_available=True,
            reason="test forced PyTorch fallback",
        ),
    )

    result = SingleProcessExecutionBackend().run(config)

    state = json.loads((result.run_dir / "run_state.json").read_text(encoding="utf-8"))
    events = (result.run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8")
    metrics = (result.run_dir / "logs" / "metrics.jsonl").read_text(encoding="utf-8")

    assert result.status == "completed_torch_fallback"
    assert state["runtime_backend"] == "torch"
    assert "test forced PyTorch fallback" in state["runtime_backend_reason"]
    assert "runtime_backend_selected" in events
    assert '"runtime_backend": "torch"' in events
    assert "runtime_backend_is_torch_fallback" in metrics
    assert (result.run_dir / "replay" / "index.json").exists()
    samples = ReplayReader(result.run_dir / "replay").read_all()
    assert samples
    assert samples[0]["search_stats"]["backend"] == "torch_fallback"
