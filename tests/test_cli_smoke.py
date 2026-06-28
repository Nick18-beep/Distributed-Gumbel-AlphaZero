from __future__ import annotations

import builtins
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gumbel_az.cli import main as cli_main
from gumbel_az.cli.main import app

runner = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def _hide_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ray":
            raise ModuleNotFoundError("No module named 'ray'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "run" in result.output
    assert "doctor" in result.output


def test_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "0.0.0" in result.output


def test_init_in_isolated_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert Path("artifacts").is_dir()
    assert Path("artifacts/runs").is_dir()
    assert Path("artifacts/cache").is_dir()


def test_doctor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "[OK] python:" in result.output
    assert "doctor summary:" in result.output


def test_run_initializes_single_process_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(DEBUG_CONFIG),
            "--set",
            "selfplay.games_per_iteration=1",
            "--set",
            "stop.max_games=1",
            "--set",
            "search.simulations_per_move=4",
            "--set",
            "training.batch_size=4",
            "--set",
            "training.steps_per_iteration=1",
            "--set",
            "stop.max_train_steps=1",
            "--set",
            "eval.games=2",
        ],
    )

    assert result.exit_code == 0
    assert "run completed:" in result.output
    assert (tmp_path / "artifacts" / "runs" / "latest.json").exists()


def test_run_accepts_execution_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(DEBUG_CONFIG),
            "--execution",
            "single_process",
            "--set",
            "selfplay.games_per_iteration=1",
            "--set",
            "stop.max_games=1",
            "--set",
            "search.simulations_per_move=2",
            "--set",
            "training.batch_size=4",
            "--set",
            "training.steps_per_iteration=1",
            "--set",
            "stop.max_train_steps=1",
            "--set",
            "eval.games=2",
        ],
    )

    assert result.exit_code == 0
    assert "run completed:" in result.output


def test_registered_dev_config_commands_run_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = ("selfplay", "train", "eval")
    monkeypatch.chdir(tmp_path)

    for command in commands:
        result = runner.invoke(
            app,
            [
                command,
                "--config",
                str(DEBUG_CONFIG),
                "--set",
                "run.output_dir=artifacts/runs",
                "--set",
                "selfplay.games_per_iteration=1",
                "--set",
                "selfplay.batch_size=1",
                "--set",
                "search.simulations_per_move=2",
                "--set",
                "replay.min_samples_to_train=1",
                "--set",
                "replay.low_watermark=1",
                "--set",
                "training.batch_size=4",
                "--set",
                "training.steps_per_iteration=1",
                "--set",
                "training.checkpoint_every_steps=1",
                "--set",
                "eval.games=2",
                "--set",
                "stop.max_games=1",
                "--set",
                "stop.max_train_steps=1",
            ],
        )

        assert result.exit_code == 0
        assert f"{command} completed:" in result.output


def test_selfplay_accepts_games_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "selfplay",
            "--config",
            str(DEBUG_CONFIG),
            "--games",
            "1",
            "--set",
            "selfplay.batch_size=1",
            "--set",
            "search.simulations_per_move=2",
        ],
    )

    assert result.exit_code == 0
    assert "selfplay completed:" in result.output


def test_benchmark_accepts_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "bench"

    result = runner.invoke(
        app,
        [
            "benchmark",
            "--config",
            str(DEBUG_CONFIG),
            "--output-dir",
            str(output_dir),
            "--set",
            "selfplay.games_per_iteration=1",
            "--set",
            "selfplay.batch_size=1",
            "--set",
            "search.simulations_per_move=2",
            "--set",
            "replay.min_samples_to_train=1",
            "--set",
            "training.batch_size=4",
            "--set",
            "training.steps_per_iteration=1",
            "--set",
            "training.checkpoint_every_steps=1",
            "--set",
            "eval.games=2",
            "--set",
            "stop.max_games=1",
            "--set",
            "stop.max_train_steps=1",
        ],
    )

    assert result.exit_code == 0
    assert "benchmark written:" in result.output
    assert any(output_dir.glob("benchmark_*.jsonl"))


def test_lan_ray_without_ray_reports_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _hide_ray(monkeypatch)

    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(PROJECT_ROOT / "configs" / "connect_four_lan.yaml"),
            "--set",
            "execution.backend=lan_ray",
        ],
    )

    assert result.exit_code == 1
    assert "Run failed:" in result.output
    assert "Ray is not installed" in result.output


def test_cluster_status_without_ray_reports_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _hide_ray(monkeypatch)

    result = runner.invoke(app, ["cluster", "status"])

    assert result.exit_code == 1
    assert "Ray is not installed" in result.output


def test_cluster_head_resolves_wildcard_bind_host_to_node_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_main, "_detect_lan_ip", lambda: "192.168.1.10")

    assert cli_main._ray_node_ip("0.0.0.0") == "192.168.1.10"
    assert cli_main._ray_node_ip("::") == "192.168.1.10"
    assert cli_main._ray_node_ip("192.168.1.20") == "192.168.1.20"


def test_ray_cluster_env_enables_experimental_non_linux_multinode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER", raising=False)
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")

    env = cli_main._ray_cluster_env()

    assert env["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] == "1"


def test_cluster_worker_passes_non_linux_ray_multinode_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ray":
            return object()
        return real_import(name, globals, locals, fromlist, level)

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    monkeypatch.setattr(cli_main, "_detect_lan_ip", lambda: "192.168.1.161")
    monkeypatch.setattr(cli_main, "_ray_cli_path", lambda: "ray")
    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)

    result = runner.invoke(
        app,
        [
            "cluster",
            "worker",
            "--head",
            "192.168.1.12:6379",
            "--config",
            str(PROJECT_ROOT / "configs" / "connect_four_lan.yaml"),
        ],
    )

    assert result.exit_code == 0
    assert "Ray multi-node on macOS is experimental" in result.output
    assert "ray worker connected: 192.168.1.12:6379 from 192.168.1.161" in result.output
    assert calls[0]["command"] == [
        "ray",
        "start",
        "--address=192.168.1.12:6379",
        "--node-ip-address=192.168.1.161",
    ]
    assert calls[0]["env"]["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] == "1"


def test_cluster_head_wait_workers_uses_resolved_lan_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    waits: list[dict] = []
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ray":
            return object()
        return real_import(name, globals, locals, fromlist, level)

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, 0)

    def fake_wait(**kwargs) -> None:
        waits.append(kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(cli_main, "_detect_lan_ip", lambda: "192.168.1.12")
    monkeypatch.setattr(cli_main, "_ray_cli_path", lambda: "ray")
    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_main, "_wait_for_ray_workers", fake_wait)

    result = runner.invoke(
        app,
        [
            "cluster",
            "head",
            "--config",
            str(PROJECT_ROOT / "configs" / "connect_four_lan.yaml"),
            "--host",
            "0.0.0.0",
            "--port",
            "6380",
            "--wait-workers",
            "--min-workers",
            "2",
            "--timeout-sec",
            "10",
            "--poll-sec",
            "0.5",
        ],
    )

    assert result.exit_code == 0
    assert "ray head address for workers: 192.168.1.12:6380" in result.output
    assert (
        "worker command: gaz cluster worker --head 192.168.1.12:6380"
        in result.output
    )
    assert calls[0]["command"] == [
        "ray",
        "start",
        "--head",
        "--node-ip-address=192.168.1.12",
        "--port=6380",
        "--include-dashboard=false",
        "--disable-usage-stats",
    ]
    assert waits == [
        {
            "head": "192.168.1.12:6380",
            "min_workers": 2,
            "timeout_sec": 10.0,
            "poll_sec": 0.5,
        }
    ]


def test_cluster_wait_passes_parameters_to_worker_watcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[dict] = []
    monkeypatch.setattr(cli_main, "_wait_for_ray_workers", lambda **kwargs: waits.append(kwargs))

    result = runner.invoke(
        app,
        [
            "cluster",
            "wait",
            "--head",
            "192.168.1.12:6379",
            "--min-workers",
            "3",
            "--timeout-sec",
            "12",
            "--poll-sec",
            "0.25",
        ],
    )

    assert result.exit_code == 0
    assert waits == [
        {
            "head": "192.168.1.12:6379",
            "min_workers": 3,
            "timeout_sec": 12.0,
            "poll_sec": 0.25,
        }
    ]


def test_config_command_accepts_override() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(DEBUG_CONFIG),
            "--set",
            "run.seed=123",
            "--set",
            "selfplay.games_per_iteration=1",
            "--set",
            "stop.max_games=1",
            "--set",
            "search.simulations_per_move=4",
            "--set",
            "training.batch_size=4",
            "--set",
            "training.steps_per_iteration=1",
            "--set",
            "stop.max_train_steps=1",
            "--set",
            "eval.games=2",
        ],
    )

    assert result.exit_code == 0
    assert "run completed:" in result.output


def test_config_command_rejects_invalid_config(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("run:\n  name: incomplete\n", encoding="utf-8")

    result = runner.invoke(app, ["run", "--config", str(config)])

    assert result.exit_code == 2
    assert "Invalid config" in result.output


def test_run_unknown_algorithm_reports_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(DEBUG_CONFIG),
            "--set",
            "algorithm.name=random_baseline",
            "--set",
            "selfplay.games_per_iteration=1",
            "--set",
            "stop.max_games=1",
            "--set",
            "search.simulations_per_move=2",
            "--set",
            "training.batch_size=4",
            "--set",
            "training.steps_per_iteration=1",
            "--set",
            "stop.max_train_steps=1",
            "--set",
            "eval.games=2",
        ],
    )

    assert result.exit_code == 1
    assert "Run failed:" in result.output
    assert "unknown algorithm" in result.output
    assert "Traceback" not in result.output


def test_local_multiprocess_unknown_algorithm_reports_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(DEBUG_CONFIG),
            "--set",
            "execution.backend=local_multiprocess",
            "--set",
            "algorithm.name=random_baseline",
            "--set",
            "selfplay.games_per_iteration=1",
            "--set",
            "stop.max_games=1",
            "--set",
            "search.simulations_per_move=2",
            "--set",
            "training.batch_size=4",
            "--set",
            "training.steps_per_iteration=1",
            "--set",
            "stop.max_train_steps=1",
            "--set",
            "eval.games=2",
        ],
    )

    assert result.exit_code == 1
    assert "Run failed:" in result.output
    assert "self-play worker failed" in result.output
    assert "unknown algorithm" in result.output
    assert "Traceback" not in result.output


def test_resume_reports_not_implemented(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Path("run")
    run_dir.mkdir()
    (run_dir / "config.resolved.yaml").write_text(
        DEBUG_CONFIG.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (run_dir / "run_state.json").write_text(
        '{"status": "completed", "train_step": 2, "games_seen": 1}',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["resume", str(run_dir)])

    assert result.exit_code == 0
    assert "resume state:" in result.output
    assert "train_step=2" in result.output


def test_inspect_run_reports_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Path("run")
    run_dir.mkdir()
    (run_dir / "run_state.json").write_text('{"status": "completed"}', encoding="utf-8")

    result = runner.invoke(app, ["inspect", "run", str(run_dir)])

    assert result.exit_code == 0
    assert '"status": "completed"' in result.output


def test_inspect_rejects_unknown_subject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path("target")
    path.mkdir()

    result = runner.invoke(app, ["inspect", "unknown", str(path)])

    assert result.exit_code != 0
    assert "subject must be one of" in result.output
