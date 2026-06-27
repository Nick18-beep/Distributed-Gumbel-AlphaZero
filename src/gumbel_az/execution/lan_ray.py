"""Optional Ray-backed LAN execution primitives."""

from __future__ import annotations

import os
import platform
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from gumbel_az.config.loader import save_resolved_config
from gumbel_az.config.schema import AppConfig
from gumbel_az.execution.base import ExecutionResult
from gumbel_az.execution.heartbeat import HeartbeatRegistry
from gumbel_az.execution.messages import WorkerCapabilities
from gumbel_az.execution.task_lease import TaskLeaseManager
from gumbel_az.logging import JsonlWriter, MetricWriter
from gumbel_az.orchestration.run import RunOrchestrator
from gumbel_az.runtime import detect_runtime_backend
from gumbel_az.storage import create_run_directory
from gumbel_az.storage.atomic import atomic_write_json
from gumbel_az.storage.transfer import CheckpointSync, ReplayTransfer


def _require_ray():
    try:
        import ray  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ray is not installed; run `uv sync --extra distributed`.") from exc
    except Exception as exc:
        sys.modules.pop("ray", None)
        raise RuntimeError(f"Ray could not be imported: {type(exc).__name__}: {exc}") from exc
    return ray


def _enable_ray_experimental_multinode_if_needed() -> None:
    if sys.platform in {"win32", "darwin"}:
        os.environ.setdefault("RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER", "1")


def detect_worker_capabilities(worker_id: str | None = None) -> WorkerCapabilities:
    runtime = detect_runtime_backend()
    devices: tuple[str, ...] = ()
    has_gpu = False
    if runtime.name == "jax":
        try:
            import jax

            devices = tuple(str(device) for device in jax.devices())
            has_gpu = any("gpu" in device.lower() or "cuda" in device.lower() for device in devices)
        except Exception:
            devices = ()
    return WorkerCapabilities(
        worker_id=worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}",
        hostname=socket.gethostname(),
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 1,
        runtime_backend=runtime.name,
        jax_devices=devices,
        has_gpu=has_gpu,
    )


@dataclass(frozen=True)
class HeadSnapshot:
    workers: dict[str, dict]
    tasks: dict[str, dict]
    commands: dict[str, list[str]]


class HeadController:
    """Logical LAN head controller independent from Ray transport."""

    def __init__(
        self,
        *,
        run_dir: Path,
        heartbeat_timeout_seconds: float = 30.0,
        lease_seconds: float = 60.0,
        max_attempts: int = 3,
        event_writer: JsonlWriter | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.heartbeat = HeartbeatRegistry(timeout_seconds=heartbeat_timeout_seconds)
        self.leases = TaskLeaseManager(lease_seconds=lease_seconds, max_attempts=max_attempts)
        self.replay_transfer = ReplayTransfer(run_dir / "replay")
        self.checkpoint_sync = CheckpointSync(run_dir / "checkpoints")
        self.event_writer = event_writer

    def register_worker(self, capabilities: WorkerCapabilities) -> dict:
        return self.heartbeat.register(capabilities).to_json()

    def worker_heartbeat(self, worker_id: str, *, status: str = "idle") -> dict:
        return self.heartbeat.heartbeat(worker_id, status=status).to_json()

    def submit_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        *,
        task_id: str | None = None,
    ) -> dict:
        return self.leases.submit(task_type, payload, task_id=task_id).to_json()

    def acquire_task(self, worker_id: str) -> dict | None:
        self.heartbeat.get(worker_id)
        record = self.leases.acquire(worker_id)
        if record is None:
            return None
        self.heartbeat.heartbeat(worker_id, status="busy")
        return record.to_json()

    def complete_task(self, worker_id: str, task_id: str, lease_id: str) -> dict:
        self.heartbeat.get(worker_id)
        record = self.leases.complete(task_id, lease_id, worker_id=worker_id)
        self.heartbeat.heartbeat(worker_id, status="idle")
        return record.to_json()

    def fail_task(self, worker_id: str, task_id: str, lease_id: str, error: str) -> dict:
        self.heartbeat.get(worker_id)
        record = self.leases.fail(task_id, lease_id, error, retry=True, worker_id=worker_id)
        self.heartbeat.heartbeat(worker_id, status="idle")
        return record.to_json()

    def expire(self) -> dict[str, list[str]]:
        lost = self.heartbeat.mark_stale_workers_lost()
        expired = self.leases.expire_leases()
        return {"lost_workers": lost, "expired_tasks": [task.task_id for task in expired]}

    def upload_replay_shard(self, worker_id: str, shard_path: str) -> dict:
        source = Path(shard_path)
        size_bytes = source.stat().st_size if source.exists() else 0
        started = perf_counter()
        result = self.replay_transfer.import_shard(source, worker_id=worker_id)
        elapsed = perf_counter() - started
        payload = {
            "imported": result.imported,
            "path": str(result.path),
            "samples": result.samples,
            "bytes": size_bytes,
            "seconds": elapsed,
            "bytes_per_sec": size_bytes / max(elapsed, 1.0e-9),
            "samples_per_sec": result.samples / max(elapsed, 1.0e-9),
            "error": result.error,
            "quarantined_path": None
            if result.quarantined_path is None
            else str(result.quarantined_path),
        }
        if self.event_writer is not None:
            self.event_writer.write(
                {"event": "lan_replay_upload", "worker_id": worker_id, **payload}
            )
        return payload

    def worker_commands(self, worker_id: str) -> list[str]:
        return self.heartbeat.pop_commands(worker_id)

    def command_worker(self, worker_id: str, command: str) -> None:
        self.heartbeat.enqueue_command(worker_id, command)

    def pause_worker(self, worker_id: str) -> None:
        self.command_worker(worker_id, "pause")

    def resume_worker(self, worker_id: str) -> None:
        self.command_worker(worker_id, "resume")

    def request_checkpoint_sync(self, worker_id: str, pointer: str = "latest") -> None:
        if pointer not in {"latest", "best"}:
            raise ValueError("pointer must be 'latest' or 'best'")
        self.command_worker(worker_id, f"sync_checkpoint:{pointer}")

    def snapshot(self) -> HeadSnapshot:
        workers = self.heartbeat.snapshot()
        return HeadSnapshot(
            workers=workers,
            tasks=self.leases.snapshot(),
            commands={worker_id: data["commands"] for worker_id, data in workers.items()},
        )


class WorkerActorCore:
    """Worker-side helper independent from Ray actor transport."""

    def __init__(self, *, worker_id: str | None = None) -> None:
        self.capabilities = detect_worker_capabilities(worker_id)

    def heartbeat_payload(self, *, status: str = "idle") -> dict[str, Any]:
        return {"worker_id": self.capabilities.worker_id, "status": status}

    def generate_replay_shard(
        self,
        config_data: dict[str, Any],
        *,
        work_dir: Path,
        games: int,
        seed: int,
    ) -> dict[str, Any]:
        config = AppConfig.model_validate(config_data)
        replay_dir = work_dir / self.capabilities.worker_id / "replay"
        runtime = detect_runtime_backend()
        if runtime.name == "jax":
            from gumbel_az.replay import ReplayWriter
            from gumbel_az.selfplay.worker import SelfPlayWorker

            worker = SelfPlayWorker(config, replay_writer=ReplayWriter(replay_dir))
        elif runtime.name == "torch":
            from gumbel_az.replay import ReplayWriter
            from gumbel_az.selfplay.torch_fallback import TorchFallbackSelfPlayWorker

            worker = TorchFallbackSelfPlayWorker(config, replay_writer=ReplayWriter(replay_dir))
        else:
            raise RuntimeError(runtime.reason)
        _, result = worker.play_batch(games, seed)
        return {
            "worker_id": self.capabilities.worker_id,
            "runtime_backend": runtime.name,
            "replay_shard": result.replay_shard,
            "games": result.games,
            "positions": result.positions,
            "games_per_sec": result.games_per_sec,
            "positions_per_sec": result.positions_per_sec,
        }

    def download_checkpoint(self, source_root: Path, destination_root: Path, pointer: str) -> str:
        synced = CheckpointSync(destination_root).sync_pointer(
            source_root,
            pointer,
            destination_root,
        )
        return str(synced)


def make_ray_head_actor():
    ray = _require_ray()

    @ray.remote
    class RayHeadController(HeadController):
        pass

    return RayHeadController


def make_ray_worker_actor():
    ray = _require_ray()

    @ray.remote
    class RayWorkerActor(WorkerActorCore):
        pass

    return RayWorkerActor


class LanRayExecutionBackend:
    """Run the training lifecycle after connecting to a Ray LAN cluster."""

    name = "lan_ray"

    def run(self, config: AppConfig) -> ExecutionResult:
        if config.execution.backend != self.name:
            raise ValueError(
                f"LanRayExecutionBackend cannot run backend {config.execution.backend!r}"
            )
        _enable_ray_experimental_multinode_if_needed()
        ray = _require_ray()
        paths = create_run_directory(config)
        save_resolved_config(config, paths.run_dir)
        event_writer = JsonlWriter(paths.events_path)
        metric_writer = MetricWriter(paths.metrics_path)
        if not ray.is_initialized():
            address = config.cluster.head_address or "auto"
            ray.init(address=address, ignore_reinit_error=True)
        runtime_backend = detect_runtime_backend()
        state = {
            "run_id": paths.run_id,
            "backend": self.name,
            "status": "initializing",
            "cluster_head_address": config.cluster.head_address,
            "config_path": str(paths.resolved_config_path),
        }
        atomic_write_json(paths.run_state_path, state)
        event_writer.write({"event": "lan_ray_initialized", **state})
        metric_writer.write_metrics(0, {"lan_ray_initialized": True})
        event_writer.write(
            {
                "event": "lan_ray_head_training_started",
                "run_id": paths.run_id,
                "cluster_head_address": config.cluster.head_address,
                "runtime_backend": runtime_backend.name,
                "note": (
                    "Ray cluster connection is active; training lifecycle is running "
                    "on the head node."
                ),
            }
        )
        return RunOrchestrator(
            config,
            paths=paths,
            runtime_backend=runtime_backend,
            event_writer=event_writer,
            metric_writer=metric_writer,
        ).run()
