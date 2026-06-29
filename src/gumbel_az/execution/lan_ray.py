"""Optional Ray-backed LAN execution primitives."""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import tempfile
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


def _lan_progress_message(record: dict[str, Any]) -> str | None:
    event = record.get("event")
    if event == "lan_ray_initialized":
        return f"[lan_ray] connected to Ray cluster: {record.get('cluster_head_address')}"
    if event == "lan_ray_head_training_started":
        return "[lan_ray] training lifecycle starting on head node"
    if event == "lan_ray_no_remote_workers":
        return "[lan_ray] no remote workers available; using head self-play"
    if event == "lan_ray_remote_selfplay_scheduled":
        return (
            "[lan_ray] scheduled remote self-play: "
            f"node={record.get('ray_node_address')} "
            f"actors={record.get('actors')} "
            f"games={record.get('games')}"
        )
    if event == "lan_ray_remote_selfplay_completed":
        imported = record.get("imported", {})
        return (
            "[lan_ray] remote worker completed: "
            f"worker={record.get('worker_id')} "
            f"node={record.get('ray_node_address')} "
            f"games={record.get('games')} "
            f"positions={record.get('positions')} "
            f"imported_samples={imported.get('samples', 0)}"
        )
    if event == "lan_ray_remote_selfplay_failed":
        return (
            "[lan_ray] remote worker failed: "
            f"node={record.get('ray_node_address')} "
            f"error={record.get('error')}"
        )
    if event == "runtime_backend_selected":
        return (
            "[run] runtime selected: "
            f"{record.get('runtime_backend')} device={record.get('device')} "
            f"reason={record.get('reason')}"
        )
    if event == "scheduler_decision":
        decision = record.get("decision", {})
        return (
            "[run] scheduler: "
            f"iteration={record.get('iteration')} "
            f"stage={record.get('stage')} "
            f"selfplay={decision.get('allow_selfplay')} "
            f"training={decision.get('allow_training')} "
            f"reason={decision.get('reason')}"
        )
    if event == "selfplay_completed":
        return (
            "[run] self-play completed: "
            f"iteration={record.get('iteration')} "
            f"games={record.get('games')} "
            f"positions={record.get('positions')}"
        )
    if event == "selfplay_skipped":
        return (
            "[run] self-play skipped: "
            f"iteration={record.get('iteration')} "
            f"reason={record.get('reason')} "
            f"replay_samples={record.get('replay_samples_available')}"
        )
    if event == "training_completed":
        return (
            "[run] training checkpoint: "
            f"iteration={record.get('iteration')} "
            f"step={record.get('train_step')} "
            f"checkpoint={record.get('checkpoint_version')}"
        )
    return None


class ConsoleEventWriter(JsonlWriter):
    """JSONL writer that mirrors high-signal LAN training events to stdout."""

    def write(self, record: dict[str, Any]) -> None:
        super().write(record)
        message = _lan_progress_message(record)
        if message is not None:
            print(message, flush=True)


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
    devices: tuple[str, ...] = (runtime.device,)
    has_gpu = runtime.device == "cuda"
    return WorkerCapabilities(
        worker_id=worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}",
        hostname=socket.gethostname(),
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 1,
        runtime_backend=runtime.name,
        torch_device=runtime.device,
        torch_devices=devices,
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
        work_dir: str | os.PathLike[str] | None = None,
        games: int,
        seed: int,
        worker_slot: int = 0,
    ) -> dict[str, Any]:
        config = AppConfig.model_validate(config_data)
        if work_dir is None:
            root = Path(tempfile.gettempdir()) / "gumbel_az_ray_workers"
        else:
            root = Path(work_dir)
        replay_dir = root / self.capabilities.worker_id / f"slot_{worker_slot:03d}" / "replay"
        runtime = detect_runtime_backend()
        if runtime.name != "torch":
            raise RuntimeError(runtime.reason)
        import torch

        torch_threads = max(1, int(os.environ.get("GAZ_RAY_ACTOR_TORCH_THREADS", "1")))
        torch.set_num_threads(torch_threads)
        print(
            "[gaz-worker] self-play starting "
            f"worker={self.capabilities.worker_id} "
            f"slot={worker_slot} games={games} device={runtime.device} "
            f"torch_threads={torch_threads}",
            flush=True,
        )
        from gumbel_az.replay import ReplayWriter
        from gumbel_az.selfplay.worker import SelfPlayWorker

        worker = SelfPlayWorker(
            config,
            replay_writer=ReplayWriter(replay_dir),
            device=runtime.device,
        )
        _, result = worker.play_batch(games, seed)
        shard_path = Path(result.replay_shard)
        print(
            "[gaz-worker] self-play completed "
            f"worker={self.capabilities.worker_id} "
            f"slot={worker_slot} games={result.games} positions={result.positions} "
            f"games_per_sec={result.games_per_sec:.3f}",
            flush=True,
        )
        return {
            "worker_id": self.capabilities.worker_id,
            "worker_slot": worker_slot,
            "runtime_backend": runtime.name,
            "device": runtime.device,
            "replay_shard": result.replay_shard,
            "replay_shard_name": shard_path.name,
            "replay_shard_bytes": shard_path.read_bytes(),
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


def _current_ray_node_id(ray: Any, alive_nodes: list[dict[str, Any]]) -> str | None:
    try:
        return str(ray.get_runtime_context().get_node_id())
    except (AttributeError, RuntimeError):
        if alive_nodes:
            return str(alive_nodes[0].get("NodeID", ""))
        return None


def _remote_ray_nodes(ray: Any) -> list[dict[str, Any]]:
    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    current_node_id = _current_ray_node_id(ray, alive_nodes)
    if current_node_id is None:
        return []
    return [
        node
        for node in alive_nodes
        if str(node.get("NodeID", "")) != current_node_id
    ]


def _ray_node_affinity_options(node_id: str) -> dict[str, Any]:
    try:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except Exception as exc:
        raise RuntimeError(
            "Ray node-affinity scheduling is unavailable; upgrade Ray or use "
            "single_process/local_multiprocess."
        ) from exc
    return {
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=False,
        )
    }


def _ray_node_address(node: dict[str, Any]) -> str:
    return str(
        node.get(
            "NodeManagerAddress",
            node.get("NodeManagerHostname", "unknown"),
        )
    )


def _ray_node_cpu_count(node: dict[str, Any]) -> int:
    resources = node.get("Resources", {})
    if not isinstance(resources, dict):
        return 1
    try:
        return max(1, int(float(resources.get("CPU", 1))))
    except (TypeError, ValueError):
        return 1


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
        event_writer = ConsoleEventWriter(paths.events_path)
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
        remote_nodes = _remote_ray_nodes(ray)
        remote_stats = {
            "remote_workers_available": len(remote_nodes),
            "remote_workers_scheduled": 0,
            "remote_workers_completed": 0,
            "remote_workers_failed": 0,
            "remote_games": 0,
            "remote_positions": 0,
            "remote_replay_samples_imported": 0,
        }
        if remote_nodes:
            games = min(
                config.stop.max_games or config.selfplay.games_per_iteration,
                config.selfplay.games_per_iteration,
            )
            worker_actor = make_ray_worker_actor()
            node_actor_counts: list[tuple[dict[str, Any], int]] = []
            remaining_actor_budget = games
            for node in remote_nodes:
                if remaining_actor_budget <= 0:
                    break
                actor_count = min(_ray_node_cpu_count(node), remaining_actor_budget)
                node_actor_counts.append((node, actor_count))
                remaining_actor_budget -= actor_count
            actor_plan: list[tuple[dict[str, Any], int, int]] = []
            slot = 0
            total_actors = sum(actor_count for _, actor_count in node_actor_counts)
            base_games_per_actor = games // total_actors
            extra_games = games % total_actors
            for node, actor_count in node_actor_counts:
                node_games = 0
                for _ in range(actor_count):
                    actor_games = base_games_per_actor + (1 if slot < extra_games else 0)
                    actor_plan.append((node, slot, actor_games))
                    node_games += actor_games
                    slot += 1
                event_writer.write(
                    {
                        "event": "lan_ray_remote_selfplay_scheduled",
                        "ray_node_id": str(node.get("NodeID", "")),
                        "ray_node_address": _ray_node_address(node),
                        "actors": actor_count,
                        "games": node_games,
                        "cpu_count": _ray_node_cpu_count(node),
                    }
                )
            remote_stats["remote_workers_scheduled"] = len(actor_plan)
            actors = []
            for node, _, _ in actor_plan:
                node_id = str(node.get("NodeID", ""))
                actors.append(
                    worker_actor.options(
                        **_ray_node_affinity_options(node_id),
                        num_cpus=1,
                    ).remote()
                )
            futures = [
                actor.generate_replay_shard.remote(
                    config.model_dump(mode="json"),
                    work_dir=None,
                    games=actor_games,
                    worker_slot=worker_slot,
                    seed=config.run.seed + 100_000 + index * 10_000,
                )
                for index, (actor, (_, worker_slot, actor_games)) in enumerate(
                    zip(actors, actor_plan, strict=True)
                )
                if actor_games > 0
            ]
            head = HeadController(run_dir=paths.run_dir, event_writer=event_writer)
            upload_dir = paths.run_dir / "ray_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            for index, future in enumerate(futures):
                node = actor_plan[index][0]
                try:
                    payload = ray.get(future)
                except Exception as exc:
                    event_writer.write(
                        {
                            "event": "lan_ray_remote_selfplay_failed",
                            "ray_node_id": str(node.get("NodeID", "")),
                            "ray_node_address": _ray_node_address(node),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    remote_stats["remote_workers_failed"] += 1
                    continue
                shard_path = upload_dir / f"{payload['worker_id']}_{payload['replay_shard_name']}"
                shard_path.write_bytes(payload.pop("replay_shard_bytes"))
                imported = head.upload_replay_shard(payload["worker_id"], str(shard_path))
                remote_stats["remote_workers_completed"] += 1
                remote_stats["remote_games"] += int(payload.get("games", 0))
                remote_stats["remote_positions"] += int(payload.get("positions", 0))
                remote_stats["remote_replay_samples_imported"] += int(imported.get("samples", 0))
                event_writer.write(
                    {
                        "event": "lan_ray_remote_selfplay_completed",
                        **payload,
                        "ray_node_id": str(node.get("NodeID", "")),
                        "ray_node_address": _ray_node_address(node),
                        "imported": imported,
                    }
                )
        else:
            event_writer.write(
                {
                    "event": "lan_ray_no_remote_workers",
                    "run_id": paths.run_id,
                    "note": "No remote Ray worker nodes were available; using head self-play.",
                }
            )
        state.update(remote_stats)
        atomic_write_json(paths.run_state_path, state)
        result = RunOrchestrator(
            config,
            paths=paths,
            runtime_backend=runtime_backend,
            event_writer=event_writer,
            metric_writer=metric_writer,
            skip_initial_selfplay_if_replay_available=remote_stats[
                "remote_replay_samples_imported"
            ]
            > 0,
        ).run()
        if paths.run_state_path.exists():
            final_state = json.loads(paths.run_state_path.read_text(encoding="utf-8"))
            final_state.update(remote_stats)
            atomic_write_json(paths.run_state_path, final_state)
        return result
