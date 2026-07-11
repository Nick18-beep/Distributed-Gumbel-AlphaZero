from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from gumbel_az.analysis.inspect import inspect_checkpoint, inspect_replay, inspect_run
from gumbel_az.benchmark import run_benchmark
from gumbel_az.cli.main import app
from gumbel_az.config import load_config
from gumbel_az.execution import SingleProcessExecutionBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def _small_run(tmp_path: Path):
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
            "eval.games=2",
        ],
    )
    return SingleProcessExecutionBackend().run(config)


def test_inspect_reports_run_replay_and_checkpoint(tmp_path: Path) -> None:
    result = _small_run(tmp_path)

    run_report = inspect_run(result.run_dir)
    replay_report = inspect_replay(result.run_dir / "replay")
    checkpoint_report = inspect_checkpoint(result.run_dir / "checkpoints")

    assert run_report["state"]["status"] == "completed"
    assert replay_report["total_samples"] > 0
    assert replay_report["illegal_policy_positive_count"] == 0
    assert checkpoint_report["latest_version"] == 1


def test_inspect_run_reports_jsonl_parse_errors(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    (run_dir / "run_state.json").write_text('{"status": "partial"}', encoding="utf-8")
    (logs_dir / "metrics.jsonl").write_text(
        '{"step": 0, "metrics": {"ok": 1}}\n{"step": 1, "metrics": \n',
        encoding="utf-8",
    )
    (logs_dir / "events.jsonl").write_text('["not", "object"]\n', encoding="utf-8")

    report = inspect_run(run_dir)

    assert report["metrics"]["ok"]["last"] == 1.0
    assert report["parse_errors"]["metrics"]
    assert report["parse_errors"]["events"][0]["error"] == "record is not a JSON object"


def test_inspect_replay_missing_directory_fails(tmp_path: Path) -> None:
    missing = tmp_path / "missing-replay"

    try:
        inspect_replay(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("inspect_replay should fail on missing directories")


def test_inspect_replay_reports_missing_indexed_shard(tmp_path: Path) -> None:
    result = _small_run(tmp_path)
    replay_dir = result.run_dir / "replay"
    shard = next((replay_dir / "shards").glob("*.msgpack.zst"))
    shard.unlink()

    report = inspect_replay(replay_dir)

    assert report["missing_shards"] == 1
    assert report["loaded_samples"] == 0
    assert report["read_errors"] == []


def test_inspect_checkpoint_tolerates_malformed_index_entries(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "index.json").write_text(
        json.dumps({"checkpoints": [{"version": 1}, {"missing": "version"}, "bad"]}),
        encoding="utf-8",
    )
    (checkpoint_dir / "latest.json").write_text('{"version": 1}', encoding="utf-8")

    report = inspect_checkpoint(checkpoint_dir)

    assert report["versions"] == [1]
    assert report["count"] == 1
    assert report["invalid_entry_count"] == 2
    assert report["latest_version"] == 1


def test_cli_inspect_outputs_json(tmp_path: Path) -> None:
    result = _small_run(tmp_path)
    runner = CliRunner()

    cli_result = runner.invoke(app, ["inspect", "run", str(result.run_dir)])

    assert cli_result.exit_code == 0
    report = json.loads(cli_result.output)
    assert report["state"]["status"] == "completed"
    assert report["replay"]["total_samples"] > 0


def test_benchmark_smoke_writes_jsonl(tmp_path: Path) -> None:
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
            "eval.games=2",
        ],
    )

    output = run_benchmark(config, output_dir=tmp_path / "benchmarks")
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert output.exists()
    assert {record["benchmark"] for record in records} >= {
        "metadata",
        "train_step",
        "selfplay",
        "replay_read",
        "checkpoint",
        "evaluation",
    }
    selfplay_record = next(record for record in records if record["benchmark"] == "selfplay")
    assert selfplay_record["warmup_games"] == 1
    assert selfplay_record["measured_games"] == 1
    metadata = next(record for record in records if record["benchmark"] == "metadata")
    assert metadata["workspace_temporary"] is True
    assert not list((tmp_path / "benchmarks").glob(".gaz-benchmark-*"))
    assert not list(tmp_path.glob("*_connect-four-cpu-debug"))


def test_cli_benchmark_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "benchmark",
            "--config",
            str(DEBUG_CONFIG),
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
    assert "benchmark written:" in result.output
    assert list((tmp_path / "artifacts" / "benchmarks").glob("benchmark_*.jsonl"))
