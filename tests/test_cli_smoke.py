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
    monkeypatch.delenv("RAY_raylet_start_wait_time_s", raising=False)
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")

    env = cli_main._ray_cluster_env()

    assert env["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] == "1"
    assert env["RAY_raylet_start_wait_time_s"] == "60"


def test_ray_fixed_ports_expands_worker_range() -> None:
    assert cli_main._ray_fixed_ports(
        head_port=6379,
        node_manager_port=6380,
        min_worker_port=10002,
        max_worker_port=10004,
    ) == [6379, 6380, 10002, 10003, 10004]


def test_ray_port_args_require_complete_worker_range() -> None:
    with pytest.raises(cli_main.typer.BadParameter, match="must be set together"):
        cli_main._ray_port_args(min_worker_port=10002)

    with pytest.raises(cli_main.typer.BadParameter, match="cannot be greater"):
        cli_main._ray_port_args(min_worker_port=10101, max_worker_port=10002)


def test_ray_storage_args_are_forwarded() -> None:
    assert cli_main._ray_storage_args(
        temp_dir=Path("/tmp/ray-gaz"),
        plasma_directory=Path("/tmp"),
        object_spilling_directory=Path("/tmp/ray-gaz-spill"),
    ) == [
        "--temp-dir=/tmp/ray-gaz",
        "--plasma-directory=/tmp",
        "--object-spilling-directory=/tmp/ray-gaz-spill",
    ]


def test_ray_worker_storage_defaults_use_short_paths_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")

    assert cli_main._ray_worker_storage_defaults() == (
        Path("/tmp/ray-gaz"),
        Path("/tmp"),
        Path("/tmp/ray-gaz-spill"),
    )
    assert cli_main._ray_worker_storage_defaults(temp_dir=Path("/custom/ray")) == (
        Path("/custom/ray"),
        Path("/tmp"),
        Path("/tmp/ray-gaz-spill"),
    )


def test_cluster_head_reports_busy_ports_before_ray_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ray":
            return object()
        return real_import(name, globals, locals, fromlist, level)

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(cli_main, "_detect_lan_ip", lambda: "192.168.1.12")
    monkeypatch.setattr(cli_main, "_ray_cli_path", lambda: "ray")
    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(
        cli_main,
        "_is_local_tcp_port_available",
        lambda port: port not in {6385, 6386},
    )

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
            "6379",
            "--dashboard-agent-grpc-port",
            "6385",
            "--metrics-export-port",
            "6386",
        ],
    )

    assert result.exit_code == 1
    assert "local TCP port(s) are already in use: 6385, 6386" in result.output
    assert "gaz cluster stop" in result.output
    assert calls == []


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
    monkeypatch.setattr(cli_main, "_ensure_local_ray_ports_available", lambda ports: None)

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
        "--disable-usage-stats",
        "--temp-dir=/tmp/ray-gaz",
        "--plasma-directory=/tmp",
        "--object-spilling-directory=/tmp/ray-gaz-spill",
    ]
    assert calls[0]["env"]["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] == "1"


def test_cluster_worker_passes_fixed_ray_ports(
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
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    monkeypatch.setattr(cli_main, "_ray_cli_path", lambda: "ray")
    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_main, "_ensure_local_ray_ports_available", lambda ports: None)

    result = runner.invoke(
        app,
        [
            "cluster",
            "worker",
            "--head",
            "192.168.1.12:6379",
            "--node-ip",
            "192.168.1.161",
            "--config",
            str(PROJECT_ROOT / "configs" / "connect_four_lan.yaml"),
            "--node-manager-port",
            "6380",
            "--object-manager-port",
            "6381",
            "--runtime-env-agent-port",
            "6382",
            "--dashboard-agent-listen-port",
            "6384",
            "--dashboard-agent-grpc-port",
            "6385",
            "--metrics-export-port",
            "6386",
            "--min-worker-port",
            "10002",
            "--max-worker-port",
            "10101",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["command"] == [
        "ray",
        "start",
        "--address=192.168.1.12:6379",
        "--node-ip-address=192.168.1.161",
        "--disable-usage-stats",
        "--node-manager-port=6380",
        "--object-manager-port=6381",
        "--runtime-env-agent-port=6382",
        "--dashboard-agent-listen-port=6384",
        "--dashboard-agent-grpc-port=6385",
        "--metrics-export-port=6386",
        "--min-worker-port=10002",
        "--max-worker-port=10101",
    ]


def test_cluster_worker_keep_alive_polls_after_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    keep_alive_calls: list[dict] = []
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ray":
            return object()
        return real_import(name, globals, locals, fromlist, level)

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, 0)

    def fake_keep_alive(**kwargs) -> None:
        keep_alive_calls.append(kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    monkeypatch.setattr(cli_main, "_ray_cli_path", lambda: "ray")
    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_main, "_keep_ray_worker_alive", fake_keep_alive)
    monkeypatch.setattr(cli_main, "_ensure_local_ray_ports_available", lambda ports: None)

    result = runner.invoke(
        app,
        [
            "cluster",
            "worker",
            "--head",
            "192.168.1.12:6379",
            "--node-ip",
            "192.168.1.161",
            "--config",
            str(PROJECT_ROOT / "configs" / "connect_four_lan.yaml"),
            "--keep-alive",
            "--keep-alive-poll-sec",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["command"] == [
        "ray",
        "start",
        "--address=192.168.1.12:6379",
        "--node-ip-address=192.168.1.161",
        "--disable-usage-stats",
    ]
    assert keep_alive_calls == [
        {
            "head": "192.168.1.12:6379",
            "poll_sec": 10.0,
            "ray_env": calls[0]["env"],
        }
    ]


def test_ray_worker_keep_alive_stops_local_ray_on_status_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["ray", "status"]:
            return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli_main, "_ray_cli_path", lambda: "ray")
    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_main, "sleep", lambda seconds: None)

    with pytest.raises(cli_main.typer.Exit) as exc_info:
        cli_main._keep_ray_worker_alive(
            head="192.168.1.12:6379",
            poll_sec=10.0,
            ray_env={},
        )

    assert exc_info.value.exit_code == 1
    assert calls == [
        ["ray", "status", "--address=192.168.1.12:6379"],
        ["ray", "stop", "--force"],
    ]


def test_cluster_worker_cleans_up_after_ray_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ray":
            return object()
        return real_import(name, globals, locals, fromlist, level)

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["ray", "start"]:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(cli_main, "_ray_cli_path", lambda: "ray")
    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_main, "_ensure_local_ray_ports_available", lambda ports: None)

    result = runner.invoke(
        app,
        [
            "cluster",
            "worker",
            "--head",
            "192.168.1.12:6379",
            "--node-ip",
            "192.168.1.161",
            "--config",
            str(PROJECT_ROOT / "configs" / "connect_four_lan.yaml"),
        ],
    )

    assert result.exit_code == 1
    assert calls[0][:2] == ["ray", "start"]
    assert calls[1] == ["ray", "stop", "--force"]


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
    monkeypatch.setattr(cli_main, "_ensure_local_ray_ports_available", lambda ports: None)

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
            "6379",
            "--wait-workers",
            "--min-workers",
            "2",
            "--timeout-sec",
            "10",
            "--poll-sec",
            "0.5",
            "--node-manager-port",
            "6380",
            "--object-manager-port",
            "6381",
            "--runtime-env-agent-port",
            "6382",
            "--dashboard-agent-listen-port",
            "6384",
            "--dashboard-agent-grpc-port",
            "6385",
            "--metrics-export-port",
            "6386",
            "--min-worker-port",
            "10002",
            "--max-worker-port",
            "10101",
        ],
    )

    assert result.exit_code == 0
    assert "ray head address for workers: 192.168.1.12:6379" in result.output
    assert (
        "worker command: gaz cluster worker --head 192.168.1.12:6379"
        in result.output
    )
    assert "--node-ip WORKER_LAN_IP" in result.output
    assert "configs/connect_four_lan.yaml" in result.output
    assert "configs\\connect_four_lan.yaml" not in result.output
    assert "--metrics-export-port=6386" in result.output
    assert "--min-worker-port=10002" in result.output
    assert calls[0]["command"] == [
        "ray",
        "start",
        "--head",
        "--node-ip-address=192.168.1.12",
        "--port=6379",
        "--include-dashboard=false",
        "--disable-usage-stats",
        "--node-manager-port=6380",
        "--object-manager-port=6381",
        "--runtime-env-agent-port=6382",
        "--dashboard-agent-listen-port=6384",
        "--dashboard-agent-grpc-port=6385",
        "--metrics-export-port=6386",
        "--min-worker-port=10002",
        "--max-worker-port=10101",
    ]
    assert waits == [
        {
            "head": "192.168.1.12:6379",
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


def test_cluster_stop_runs_ray_stop_and_orphan_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli_main, "_ray_cli_path", lambda: "ray")
    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_main, "_cleanup_ray_orphan_processes", lambda: 2)

    result = runner.invoke(app, ["cluster", "stop"])

    assert result.exit_code == 0
    assert calls == [["ray", "stop", "--force"]]
    assert "ray stopped; cleaned orphan process(es): 2" in result.output


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


def test_resume_status_only_reports_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    result = runner.invoke(app, ["resume", str(run_dir), "--status-only"])

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
