from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import jax.numpy as jnp
import pytest

from gumbel_az.execution.heartbeat import HeartbeatRegistry
from gumbel_az.execution.lan_ray import (
    HeadController,
    WorkerActorCore,
    make_ray_head_actor,
    make_ray_worker_actor,
)
from gumbel_az.execution.messages import WorkerCapabilities, utc_now
from gumbel_az.execution.task_lease import TaskLeaseManager
from gumbel_az.replay import ReplayWriter
from gumbel_az.replay.codec import encode_samples


def _capabilities(worker_id: str = "worker-1") -> WorkerCapabilities:
    return WorkerCapabilities(
        worker_id=worker_id,
        hostname="host",
        platform="test",
        cpu_count=4,
        runtime_backend="jax",
        jax_devices=("cpu:0",),
        has_gpu=False,
    )


def _sample(index: int = 0) -> dict:
    return {
        "game_name": "connect_four",
        "algorithm_name": "gumbel_alphazero",
        "state_or_observation": jnp.zeros((6, 7, 2), dtype=jnp.float32),
        "legal_action_mask": jnp.asarray([True, True, False, True, True, True, True]),
        "policy_target": jnp.asarray([0.2, 0.2, 0.0, 0.2, 0.2, 0.1, 0.1]),
        "value_target": 1.0 if index % 2 == 0 else -1.0,
        "to_play": index % 2,
        "move_index": index,
        "game_id": f"game-{index}",
        "model_version": 0,
        "search_stats": {"root_value": 0.0},
    }


def test_heartbeat_registry_marks_stale_workers_lost() -> None:
    registry = HeartbeatRegistry(timeout_seconds=1.0)
    registry.register(_capabilities())
    registry.heartbeat("worker-1", status="busy")
    registry.get("worker-1").last_heartbeat_at = utc_now() - timedelta(seconds=2)

    lost = registry.mark_stale_workers_lost()

    assert lost == ["worker-1"]
    assert registry.get("worker-1").status == "lost"


def test_task_lease_expires_and_retries() -> None:
    manager = TaskLeaseManager(lease_seconds=1.0, max_attempts=2)
    manager.submit("selfplay", {"games": 1}, task_id="task-1")

    first = manager.acquire("worker-a")
    assert first is not None
    manager.expire_leases(now=utc_now() + timedelta(seconds=2))
    second = manager.acquire("worker-b")

    assert second is not None
    assert second.task_id == "task-1"
    assert second.worker_id == "worker-b"
    assert second.attempts == 2


def test_task_lease_rejects_completion_after_expiration() -> None:
    manager = TaskLeaseManager(lease_seconds=0.001, max_attempts=2)
    manager.submit("selfplay", {"games": 1}, task_id="task-1")
    leased = manager.acquire("worker-a")
    assert leased is not None
    leased.lease_expires_at = utc_now() - timedelta(seconds=1)

    with pytest.raises(ValueError, match="lease has expired"):
        manager.complete("task-1", leased.lease_id)

    snapshot = manager.snapshot()
    assert snapshot["task-1"]["status"] == "pending"
    assert snapshot["task-1"]["worker_id"] is None


def test_head_controller_registration_commands_and_leases(tmp_path: Path) -> None:
    head = HeadController(run_dir=tmp_path)
    head.register_worker(_capabilities())
    task = head.submit_task("selfplay", {"games": 1}, task_id="task-1")
    acquired = head.acquire_task("worker-1")
    assert acquired is not None

    head.pause_worker("worker-1")
    head.request_checkpoint_sync("worker-1", "best")
    commands = head.worker_commands("worker-1")
    completed = head.complete_task("worker-1", task["task_id"], acquired["lease_id"])

    assert commands == ["pause", "sync_checkpoint:best"]
    assert completed["status"] == "completed"
    assert head.snapshot().workers["worker-1"]["status"] == "idle"


def test_head_controller_rejects_unknown_worker_before_leasing(tmp_path: Path) -> None:
    head = HeadController(run_dir=tmp_path)
    head.submit_task("selfplay", {"games": 1}, task_id="task-1")

    with pytest.raises(KeyError, match="unknown worker"):
        head.acquire_task("missing-worker")

    snapshot = head.snapshot()
    assert snapshot.tasks["task-1"]["status"] == "pending"
    assert snapshot.tasks["task-1"]["worker_id"] is None


def test_head_controller_rejects_completion_from_wrong_worker(tmp_path: Path) -> None:
    head = HeadController(run_dir=tmp_path)
    head.register_worker(_capabilities("worker-a"))
    head.register_worker(_capabilities("worker-b"))
    head.submit_task("selfplay", {"games": 1}, task_id="task-1")
    acquired = head.acquire_task("worker-a")
    assert acquired is not None

    with pytest.raises(ValueError, match="not leased to worker"):
        head.complete_task("worker-b", "task-1", acquired["lease_id"])

    snapshot = head.snapshot()
    assert snapshot.tasks["task-1"]["status"] == "leased"
    assert snapshot.tasks["task-1"]["worker_id"] == "worker-a"


def test_replay_upload_imports_valid_shard_and_quarantines_corrupt(tmp_path: Path) -> None:
    worker_replay = tmp_path / "worker" / "replay"
    valid_shard = ReplayWriter(worker_replay).write_shard([_sample(0), _sample(1)])
    corrupt = tmp_path / "corrupt.msgpack.zst"
    corrupt.write_bytes(b"not-zstd")
    empty = tmp_path / "empty.msgpack.zst"
    empty.write_bytes(encode_samples([]))
    head = HeadController(run_dir=tmp_path / "head")
    orphan = tmp_path / "head" / "replay" / "shards" / "shard_000000001.msgpack.zst"
    orphan.write_bytes(b"orphan")

    valid_result = head.upload_replay_shard("worker-1", str(valid_shard))
    corrupt_result = head.upload_replay_shard("worker:1/unsafe", str(corrupt))
    empty_result = head.upload_replay_shard("worker-1", str(empty))
    missing_result = head.upload_replay_shard("worker-1", str(tmp_path / "missing.zst"))

    index = json.loads((tmp_path / "head" / "replay" / "index.json").read_text("utf-8"))
    assert valid_result["imported"] is True
    assert valid_result["samples"] == 2
    assert Path(valid_result["path"]).name == "shard_000000002.msgpack.zst"
    assert orphan.read_bytes() == b"orphan"
    assert valid_result["bytes_per_sec"] > 0.0
    assert valid_result["samples_per_sec"] > 0.0
    assert index["total_samples"] == 2
    assert corrupt_result["imported"] is False
    assert corrupt_result["quarantined_path"] is not None
    assert ":" not in Path(corrupt_result["quarantined_path"]).name
    assert "/" not in Path(corrupt_result["quarantined_path"]).name
    assert corrupt_result["error"]
    assert empty_result["imported"] is False
    assert "empty" in empty_result["error"]
    assert empty_result["quarantined_path"] is not None
    assert missing_result["imported"] is False
    assert missing_result["quarantined_path"] is None
    assert "does not exist" in missing_result["error"]


def test_worker_checkpoint_download_is_atomic_tree_copy(tmp_path: Path) -> None:
    source = tmp_path / "head" / "checkpoints"
    checkpoint = source / "ckpt_000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "payload").write_text("ok", encoding="utf-8")
    (source / "latest.json").write_text(
        json.dumps({"version": 1, "path": str(checkpoint)}),
        encoding="utf-8",
    )
    worker = WorkerActorCore(worker_id="worker-1")

    synced = worker.download_checkpoint(source, tmp_path / "worker" / "checkpoints", "latest")
    local_pointer = json.loads(
        (tmp_path / "worker" / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )

    assert Path(synced).name == "ckpt_000001"
    assert (Path(synced) / "payload").read_text(encoding="utf-8") == "ok"
    assert (tmp_path / "worker" / "checkpoints" / "latest.json").exists()
    assert local_pointer["path"] == str(Path(synced).resolve())


def test_worker_checkpoint_download_reuses_existing_immutable_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "head" / "checkpoints"
    checkpoint = source / "ckpt_000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "payload").write_text("source", encoding="utf-8")
    (source / "latest.json").write_text(
        json.dumps({"version": 1, "path": str(checkpoint)}),
        encoding="utf-8",
    )
    destination_root = tmp_path / "worker" / "checkpoints"
    destination = destination_root / "ckpt_000001"
    destination.mkdir(parents=True)
    (destination / "payload").write_text("local", encoding="utf-8")
    worker = WorkerActorCore(worker_id="worker-1")

    synced = worker.download_checkpoint(source, destination_root, "latest")

    assert Path(synced) == destination
    assert (destination / "payload").read_text(encoding="utf-8") == "local"


def test_ray_actor_factories_keep_ray_optional() -> None:
    try:
        assert make_ray_head_actor() is not None
        assert make_ray_worker_actor() is not None
    except RuntimeError as exc:
        assert "Ray is not installed" in str(exc) or "Ray could not be imported" in str(exc)
