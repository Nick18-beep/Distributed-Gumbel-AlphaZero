from __future__ import annotations

import json
import shutil
from pathlib import Path

from gumbel_az.config import load_config
from gumbel_az.config.loader import save_resolved_config
from gumbel_az.execution import SingleProcessExecutionBackend
from gumbel_az.orchestration import load_resume_context, rebuild_replay_index
from gumbel_az.replay import ReplayWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = PROJECT_ROOT / "configs" / "connect_four_cpu_debug.yaml"


def test_resume_context_loads_run_state_config_replay_and_checkpoint(tmp_path: Path) -> None:
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
    result = SingleProcessExecutionBackend().run(config)

    context = load_resume_context(result.run_dir)

    assert context.run_state["status"] == "completed"
    assert context.config.run.name == config.run.name
    assert context.replay_index["total_samples"] > 0
    assert context.latest_checkpoint is not None
    assert context.latest_checkpoint["version"] == 1
    assert context.best_checkpoint is not None


def test_single_process_resume_continues_latest_checkpoint_in_same_run_dir(
    tmp_path: Path,
) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "selfplay.games_per_iteration=1",
            "stop.max_games=1",
            "stop.max_iterations=1",
            "search.simulations_per_move=2",
            "training.batch_size=4",
            "training.steps_per_iteration=1",
            "training.checkpoint_every_steps=1",
            "stop.max_train_steps=1",
            "eval.games=2",
        ],
    )
    first = SingleProcessExecutionBackend().run(config)
    resumed_config = load_config(
        first.run_dir / "config.resolved.yaml",
        [
            "stop.max_games=2",
            "stop.max_iterations=1",
            "stop.max_train_steps=2",
        ],
    )

    second = SingleProcessExecutionBackend().resume(resumed_config, first.run_dir)
    context = load_resume_context(first.run_dir)

    assert second.run_dir == first.run_dir
    assert context.run_state["status"] == "completed"
    assert context.run_state["train_step"] == 2
    assert context.latest_checkpoint is not None
    assert context.latest_checkpoint["version"] == 2


def test_resume_trains_from_existing_replay_when_max_games_is_reached(
    tmp_path: Path,
) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "selfplay.games_per_iteration=1",
            "stop.max_games=1",
            "stop.max_iterations=1",
            "search.simulations_per_move=2",
            "training.batch_size=4",
            "training.steps_per_iteration=1",
            "training.checkpoint_every_steps=1",
            "stop.max_train_steps=1",
            "eval.games=2",
        ],
    )
    first = SingleProcessExecutionBackend().run(config)
    resumed_config = load_config(
        first.run_dir / "config.resolved.yaml",
        [
            "stop.max_games=1",
            "stop.max_iterations=1",
            "stop.max_train_steps=2",
        ],
    )

    second = SingleProcessExecutionBackend().resume(resumed_config, first.run_dir)
    context = load_resume_context(first.run_dir)
    events = (first.run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8")

    assert second.status == "completed"
    assert context.run_state["train_step"] == 2
    assert context.run_state["games_seen"] == 1
    assert "max_games_reached_existing_replay_available" in events


def test_resume_noops_when_stop_limits_are_already_satisfied(tmp_path: Path) -> None:
    config = load_config(
        DEBUG_CONFIG,
        [
            f"run.output_dir={tmp_path.as_posix()}",
            "selfplay.games_per_iteration=1",
            "stop.max_games=1",
            "stop.max_iterations=1",
            "search.simulations_per_move=2",
            "training.batch_size=4",
            "training.steps_per_iteration=1",
            "training.checkpoint_every_steps=1",
            "stop.max_train_steps=1",
            "eval.games=2",
        ],
    )
    first = SingleProcessExecutionBackend().run(config)
    resumed_config = load_config(first.run_dir / "config.resolved.yaml")

    second = SingleProcessExecutionBackend().resume(resumed_config, first.run_dir)
    context = load_resume_context(first.run_dir)
    events = (first.run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8")

    assert second.status == "completed"
    assert context.run_state["train_step"] == 1
    assert context.latest_checkpoint is not None
    assert context.latest_checkpoint["version"] == 1
    assert "resume_no_training_needed" in events


def test_rebuild_replay_index_quarantines_corrupted_shards(tmp_path: Path) -> None:
    config = load_config(DEBUG_CONFIG)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_resolved_config(config, run_dir)
    (run_dir / "run_state.json").write_text(
        json.dumps({"status": "interrupted", "train_step": 0}),
        encoding="utf-8",
    )
    writer = ReplayWriter(run_dir / "replay")
    sample = {
        "game_name": "connect_four",
        "algorithm_name": "gumbel_alphazero",
        "state_or_observation": [[[0.0, 0.0]]],
        "legal_action_mask": [True],
        "policy_target": [1.0],
        "value_target": 0.0,
        "to_play": 0,
        "move_index": 0,
        "game_id": "g",
        "model_version": 0,
        "search_stats": {},
    }
    writer.write_shard([sample])
    corrupt = run_dir / "replay" / "shards" / "shard_999999999.msgpack.zst"
    corrupt.write_bytes(b"not-a-valid-shard")
    (run_dir / "replay" / "index.json").unlink()

    index = rebuild_replay_index(run_dir)

    assert index["total_samples"] == 1
    assert not corrupt.exists()
    assert list((run_dir / "replay" / "quarantine").iterdir())


def test_resume_context_rebuilds_replay_index(tmp_path: Path) -> None:
    config = load_config(DEBUG_CONFIG)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_resolved_config(config, run_dir)
    (run_dir / "run_state.json").write_text(
        json.dumps({"status": "interrupted", "train_step": 0}),
        encoding="utf-8",
    )
    source_replay = tmp_path / "source_replay"
    writer = ReplayWriter(source_replay)
    sample = {
        "game_name": "connect_four",
        "algorithm_name": "gumbel_alphazero",
        "state_or_observation": [[[0.0, 0.0]]],
        "legal_action_mask": [True],
        "policy_target": [1.0],
        "value_target": 0.0,
        "to_play": 0,
        "move_index": 0,
        "game_id": "g",
        "model_version": 0,
        "search_stats": {},
    }
    writer.write_shard([sample])
    (run_dir / "replay" / "shards").mkdir(parents=True)
    shutil.copytree(source_replay / "shards", run_dir / "replay" / "shards", dirs_exist_ok=True)

    context = load_resume_context(run_dir, rebuild_replay=True)

    assert context.replay_index["total_samples"] == 1


def test_resume_rejects_incomplete_checkpoint_pointer(tmp_path: Path) -> None:
    config = load_config(DEBUG_CONFIG)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_resolved_config(config, run_dir)
    (run_dir / "run_state.json").write_text(
        json.dumps({"status": "interrupted", "train_step": 1}),
        encoding="utf-8",
    )
    checkpoint_dir = run_dir / "checkpoints" / "ckpt_000001"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "checkpoints" / "latest.json").write_text(
        json.dumps({"version": 1, "path": str(checkpoint_dir)}),
        encoding="utf-8",
    )

    try:
        load_resume_context(run_dir)
    except FileNotFoundError as exc:
        assert "incomplete checkpoint" in str(exc)
    else:
        raise AssertionError("expected incomplete checkpoint pointer to fail")


def test_resume_rejects_duplicate_replay_index_entries(tmp_path: Path) -> None:
    config = load_config(DEBUG_CONFIG)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_resolved_config(config, run_dir)
    (run_dir / "run_state.json").write_text(
        json.dumps({"status": "interrupted", "train_step": 0}),
        encoding="utf-8",
    )
    writer = ReplayWriter(run_dir / "replay")
    sample = {
        "game_name": "connect_four",
        "algorithm_name": "gumbel_alphazero",
        "state_or_observation": [[[0.0, 0.0]]],
        "legal_action_mask": [True],
        "policy_target": [1.0],
        "value_target": 0.0,
        "to_play": 0,
        "move_index": 0,
        "game_id": "g",
        "model_version": 0,
        "search_stats": {},
    }
    writer.write_shard([sample])
    index_path = run_dir / "replay" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["shards"].append(dict(index["shards"][0]))
    index["total_samples"] = 2
    index_path.write_text(json.dumps(index), encoding="utf-8")

    try:
        load_resume_context(run_dir)
    except ValueError as exc:
        assert "duplicate replay shard" in str(exc)
    else:
        raise AssertionError("expected duplicate replay shard to fail")


def test_resume_rejects_replay_total_mismatch(tmp_path: Path) -> None:
    config = load_config(DEBUG_CONFIG)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_resolved_config(config, run_dir)
    (run_dir / "run_state.json").write_text(
        json.dumps({"status": "interrupted", "train_step": 0}),
        encoding="utf-8",
    )
    writer = ReplayWriter(run_dir / "replay")
    sample = {
        "game_name": "connect_four",
        "algorithm_name": "gumbel_alphazero",
        "state_or_observation": [[[0.0, 0.0]]],
        "legal_action_mask": [True],
        "policy_target": [1.0],
        "value_target": 0.0,
        "to_play": 0,
        "move_index": 0,
        "game_id": "g",
        "model_version": 0,
        "search_stats": {},
    }
    writer.write_shard([sample])
    index_path = run_dir / "replay" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["total_samples"] = 999
    index_path.write_text(json.dumps(index), encoding="utf-8")

    try:
        load_resume_context(run_dir)
    except ValueError as exc:
        assert "total_samples mismatch" in str(exc)
    else:
        raise AssertionError("expected replay total mismatch to fail")


def test_resume_rejects_replay_index_schema_mismatch(tmp_path: Path) -> None:
    config = load_config(DEBUG_CONFIG)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_resolved_config(config, run_dir)
    (run_dir / "run_state.json").write_text(
        json.dumps({"status": "interrupted", "train_step": 0}),
        encoding="utf-8",
    )
    writer = ReplayWriter(run_dir / "replay")
    sample = {
        "game_name": "connect_four",
        "algorithm_name": "gumbel_alphazero",
        "state_or_observation": [[[0.0, 0.0]]],
        "legal_action_mask": [True],
        "policy_target": [1.0],
        "value_target": 0.0,
        "to_play": 0,
        "move_index": 0,
        "game_id": "g",
        "model_version": 0,
        "search_stats": {},
    }
    writer.write_shard([sample])
    index_path = run_dir / "replay" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["schema_version"] = 999
    index_path.write_text(json.dumps(index), encoding="utf-8")

    try:
        load_resume_context(run_dir)
    except ValueError as exc:
        assert "unsupported replay index schema_version" in str(exc)
    else:
        raise AssertionError("expected replay index schema mismatch to fail")
