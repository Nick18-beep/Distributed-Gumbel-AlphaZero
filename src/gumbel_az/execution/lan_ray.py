"""Optional Ray-backed LAN execution primitives."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
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
from gumbel_az.storage.filesystem import RunPaths, existing_run_paths
from gumbel_az.storage.transfer import CheckpointSync, ReplayTransfer

_GIB = 1024**3
_SELFPLAY_ACTOR_MEMORY_BYTES = 2 * _GIB
_HEAD_MEMORY_RESERVE_BYTES = 4 * _GIB
_WORKER_MEMORY_RESERVE_BYTES = 2 * _GIB


def _lan_progress_message(record: dict[str, Any]) -> str | None:
    event = record.get("event")
    if event == "lan_ray_initialized":
        return f"[lan_ray] connected to Ray cluster: {record.get('cluster_head_address')}"
    if event == "lan_ray_head_training_started":
        return "[lan_ray] training lifecycle starting on head node"
    if event == "lan_ray_no_remote_workers":
        return "[lan_ray] no remote workers available; using head self-play"
    if event == "lan_ray_remote_selfplay_skipped":
        return f"[lan_ray] remote self-play skipped: {record.get('reason')}"
    if event == "lan_ray_remote_selfplay_scheduled":
        return (
            "[lan_ray] scheduled remote self-play: "
            f"node={record.get('ray_node_address')} "
            f"actors={record.get('actors')} "
            f"cpus_per_actor={record.get('cpus_per_actor')} "
            f"memory_gib={record.get('memory_gib')} "
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


def _require_ray() -> Any:
    try:
        import ray
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ray is not installed; run `uv sync --extra distributed`.") from exc
    except Exception as exc:
        sys.modules.pop("ray", None)
        raise RuntimeError(f"Ray could not be imported: {type(exc).__name__}: {exc}") from exc
    return ray


def _enable_ray_experimental_multinode_if_needed() -> None:
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
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
    workers: dict[str, dict[str, Any]]
    tasks: dict[str, dict[str, Any]]
    commands: dict[str, list[str]]


@dataclass(frozen=True)
class RaySelfPlayNodePlan:
    actor_count: int
    cpus_per_actor: int
    cpu_count: int
    memory_bytes: int | None
    actor_limit: int


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

    def register_worker(self, capabilities: WorkerCapabilities) -> dict[str, Any]:
        return self.heartbeat.register(capabilities).to_json()

    def worker_heartbeat(self, worker_id: str, *, status: str = "idle") -> dict[str, Any]:
        return self.heartbeat.heartbeat(worker_id, status=status).to_json()

    def submit_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        *,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        return self.leases.submit(task_type, payload, task_id=task_id).to_json()

    def acquire_task(self, worker_id: str) -> dict[str, Any] | None:
        self.heartbeat.get(worker_id)
        record = self.leases.acquire(worker_id)
        if record is None:
            return None
        self.heartbeat.heartbeat(worker_id, status="busy")
        return record.to_json()

    def complete_task(self, worker_id: str, task_id: str, lease_id: str) -> dict[str, Any]:
        self.heartbeat.get(worker_id)
        record = self.leases.complete(task_id, lease_id, worker_id=worker_id)
        self.heartbeat.heartbeat(worker_id, status="idle")
        return record.to_json()

    def fail_task(
        self,
        worker_id: str,
        task_id: str,
        lease_id: str,
        error: str,
    ) -> dict[str, Any]:
        self.heartbeat.get(worker_id)
        record = self.leases.fail(task_id, lease_id, error, retry=True, worker_id=worker_id)
        self.heartbeat.heartbeat(worker_id, status="idle")
        return record.to_json()

    def expire(self) -> dict[str, list[str]]:
        lost = self.heartbeat.mark_stale_workers_lost()
        expired = self.leases.expire_leases()
        return {"lost_workers": lost, "expired_tasks": [task.task_id for task in expired]}

    def upload_replay_shard(self, worker_id: str, shard_path: str) -> dict[str, Any]:
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
        device: str | None = None,
        checkpoint_payload_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        config = AppConfig.model_validate(config_data)
        temporary_work_dir = work_dir is None
        if work_dir is None:
            root = Path(tempfile.gettempdir()) / "gumbel_az_ray_workers"
        else:
            root = Path(work_dir)
        worker_root = root / self.capabilities.worker_id
        slot_root = worker_root / f"slot_{worker_slot:03d}"
        replay_dir = slot_root / "replay"
        runtime = detect_runtime_backend()
        if runtime.name != "torch":
            raise RuntimeError(runtime.reason)
        import torch

        selected_device = torch.device(device or runtime.device)
        torch_threads = max(
            1,
            int(os.environ.get("GAZ_RAY_ACTOR_TORCH_THREADS", "1")),
        )
        torch.set_num_threads(torch_threads)
        print(
            "[gaz-worker] self-play starting "
            f"worker={self.capabilities.worker_id} "
            f"slot={worker_slot} games={games} device={selected_device} "
            f"torch_threads={torch_threads}",
            flush=True,
        )
        from gumbel_az.replay import ReplayWriter
        from gumbel_az.selfplay.worker import SelfPlayWorker

        worker = SelfPlayWorker(
            config,
            replay_writer=ReplayWriter(replay_dir),
            device=selected_device,
        )
        if checkpoint_payload_bytes is not None:
            checkpoint = torch.load(
                BytesIO(checkpoint_payload_bytes),
                map_location=selected_device,
                weights_only=True,
            )
            worker.model.load_state_dict(checkpoint["state"]["model_state_dict"])
            worker.model_version = int(checkpoint.get("metadata", {}).get("version", 0))
        _, result = worker.play_batch(games, seed)
        shard_path = Path(result.replay_shard)
        shard_bytes = shard_path.read_bytes()
        print(
            "[gaz-worker] self-play completed "
            f"worker={self.capabilities.worker_id} "
            f"slot={worker_slot} games={result.games} positions={result.positions} "
            f"games_per_sec={result.games_per_sec:.3f}",
            flush=True,
        )
        payload = {
            "worker_id": self.capabilities.worker_id,
            "worker_slot": worker_slot,
            "runtime_backend": runtime.name,
            "device": str(selected_device),
            "replay_shard": result.replay_shard,
            "replay_shard_name": shard_path.name,
            "replay_shard_bytes": shard_bytes,
            "games": result.games,
            "positions": result.positions,
            "games_per_sec": result.games_per_sec,
            "positions_per_sec": result.positions_per_sec,
        }
        if temporary_work_dir:
            shutil.rmtree(slot_root, ignore_errors=True)
            for directory in (worker_root, root):
                try:
                    directory.rmdir()
                except OSError:
                    break
        return payload

    def download_checkpoint(self, source_root: Path, destination_root: Path, pointer: str) -> str:
        synced = CheckpointSync(destination_root).sync_pointer(
            source_root,
            pointer,
            destination_root,
        )
        return str(synced)


def make_ray_head_actor() -> Any:
    ray = _require_ray()

    @ray.remote
    class RayHeadController(HeadController):
        pass

    return RayHeadController


def make_ray_worker_actor() -> Any:
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
    return [node for node in alive_nodes if str(node.get("NodeID", "")) != current_node_id]


def _selfplay_ray_nodes(ray: Any, *, include_head: bool) -> list[tuple[dict[str, Any], bool]]:
    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    current_node_id = _current_ray_node_id(ray, alive_nodes)
    if current_node_id is None:
        return []
    head_node: dict[str, Any] | None = None
    remote_nodes: list[tuple[dict[str, Any], bool]] = []
    for node in alive_nodes:
        is_head = str(node.get("NodeID", "")) == current_node_id
        if is_head:
            head_node = node
            continue
        remote_nodes.append((node, False))
    if include_head and head_node is not None and "Resources" in head_node:
        remote_nodes.append((head_node, True))
    return remote_nodes


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
    except (OverflowError, TypeError, ValueError):
        return 1


def _ray_node_memory_bytes(node: dict[str, Any]) -> int | None:
    resources = node.get("Resources", {})
    if not isinstance(resources, dict):
        return None
    try:
        value = int(float(resources["memory"]))
    except (KeyError, OverflowError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def _ray_selfplay_node_plan(
    config: AppConfig,
    node: dict[str, Any],
    *,
    is_head: bool,
    remaining_actor_budget: int,
) -> RaySelfPlayNodePlan:
    node_cpu_count = _ray_node_cpu_count(node)
    cpu_reserve = 2 if is_head else 1
    usable_cpus = max(1, node_cpu_count - cpu_reserve)
    configured = (
        config.cluster.head_selfplay_actors
        if is_head
        else config.cluster.max_selfplay_actors_per_node
    )
    memory_bytes = _ray_node_memory_bytes(node)
    if configured == "auto":
        if memory_bytes is None:
            memory_actor_limit = 1 if is_head else min(2, usable_cpus)
        else:
            memory_reserve = _HEAD_MEMORY_RESERVE_BYTES if is_head else _WORKER_MEMORY_RESERVE_BYTES
            memory_actor_limit = max(
                1,
                (memory_bytes - memory_reserve) // _SELFPLAY_ACTOR_MEMORY_BYTES,
            )
        actor_limit = min(usable_cpus, memory_actor_limit)
    else:
        actor_limit = min(usable_cpus, int(configured))
    actor_count = min(actor_limit, remaining_actor_budget)
    cpus_per_actor = max(1, usable_cpus // max(1, actor_count))
    return RaySelfPlayNodePlan(
        actor_count=actor_count,
        cpus_per_actor=cpus_per_actor,
        cpu_count=node_cpu_count,
        memory_bytes=memory_bytes,
        actor_limit=actor_limit,
    )


def _allocate_games_by_weight(total_games: int, weights: list[int]) -> list[int]:
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("game allocation requires positive actor weights")
    if total_games < len(weights):
        raise ValueError("game allocation requires at least one game per actor")
    allocations = [1] * len(weights)
    remaining = total_games - len(weights)
    if remaining == 0:
        return allocations
    total_weight = sum(weights)
    weighted = [remaining * weight for weight in weights]
    for index, value in enumerate(weighted):
        allocations[index] += value // total_weight
    leftover = total_games - sum(allocations)
    remainder_order = sorted(
        range(len(weights)),
        key=lambda index: weighted[index] % total_weight,
        reverse=True,
    )
    for index in remainder_order[:leftover]:
        allocations[index] += 1
    return allocations


def _checkpoint_payload_bytes(run_dir: Path) -> bytes | None:
    checkpoint_root = (run_dir / "checkpoints").resolve()
    latest = checkpoint_root / "latest.json"
    if not latest.exists():
        return None
    pointer = json.loads(latest.read_text(encoding="utf-8"))
    stored_path = Path(pointer["path"])
    checkpoint_dir = (
        stored_path.resolve()
        if stored_path.is_absolute()
        else (checkpoint_root / stored_path).resolve()
    )
    if not checkpoint_dir.is_relative_to(checkpoint_root):
        raise ValueError(f"checkpoint path escapes checkpoint root: {pointer['path']}")
    checkpoint_file = checkpoint_dir / "checkpoint.pt"
    if not checkpoint_file.exists():
        return None
    return checkpoint_file.read_bytes()


def _put_shared_argument(ray: Any, value: Any, *, shared: bool) -> Any:
    put = getattr(ray, "put", None)
    return put(value) if shared and callable(put) else value


def _worker_config_payload(config: AppConfig) -> dict[str, Any]:
    """Return worker-compatible config without head-only scheduling fields."""
    payload = config.model_dump(mode="json")
    cluster = payload.get("cluster")
    if isinstance(cluster, dict):
        cluster.pop("max_selfplay_actors_per_node", None)
        cluster.pop("head_selfplay_actors", None)
    return payload


def _ready_futures(
    ray: Any,
    futures: list[Any],
    *,
    timeout_sec: float | None = None,
) -> Iterator[tuple[int, Any]]:
    """Return futures in completion order when the Ray API is available."""
    wait = getattr(ray, "wait", None)
    if not callable(wait):
        yield from enumerate(futures)
        return
    pending = list(enumerate(futures))
    deadline = None if timeout_sec is None else perf_counter() + timeout_sec
    while pending:
        timeout = None if deadline is None else max(0.0, deadline - perf_counter())
        ready, _ = wait(
            [future for _, future in pending],
            num_returns=1,
            timeout=timeout,
        )
        if not ready:
            raise TimeoutError(f"timed out waiting for {len(pending)} Ray self-play worker(s)")
        ready_future = ready[0]
        position = next(
            index for index, (_, future) in enumerate(pending) if future == ready_future
        )
        yield pending.pop(position)


def _safe_upload_component(value: Any) -> str:
    basename = re.split(r"[\\/]", str(value))[-1]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", basename)
    return safe.strip("._") or "item"


def _kill_ray_actors(ray: Any, actors: list[Any]) -> None:
    kill = getattr(ray, "kill", None)
    if not callable(kill):
        return
    for actor in actors:
        try:
            kill(actor, no_restart=True)
        except Exception:
            continue


class LanRayExecutionBackend:
    """Run the training lifecycle after connecting to a Ray LAN cluster."""

    name = "lan_ray"

    def run(self, config: AppConfig) -> ExecutionResult:
        if config.execution.backend != self.name:
            raise ValueError(
                f"LanRayExecutionBackend cannot run backend {config.execution.backend!r}"
            )
        paths = create_run_directory(config)
        save_resolved_config(config, paths.run_dir)
        return self._run_with_paths(config, paths, resume=False)

    def resume(self, config: AppConfig, run_dir: Path) -> ExecutionResult:
        if config.execution.backend != self.name:
            raise ValueError(
                f"LanRayExecutionBackend cannot run backend {config.execution.backend!r}"
            )
        paths = existing_run_paths(run_dir)
        save_resolved_config(config, paths.run_dir)
        return self._run_with_paths(config, paths, resume=True)

    def _run_with_paths(
        self,
        config: AppConfig,
        paths: RunPaths,
        *,
        resume: bool,
    ) -> ExecutionResult:
        lifecycle_started = perf_counter()
        _enable_ray_experimental_multinode_if_needed()
        ray = _require_ray()
        event_writer = ConsoleEventWriter(paths.events_path)
        metric_writer = MetricWriter(paths.metrics_path)
        if not ray.is_initialized():
            address = config.cluster.head_address or "auto"
            ray.init(address=address, ignore_reinit_error=True)
        runtime_backend = detect_runtime_backend()
        previous_state: dict[str, Any] = {}
        if resume and paths.run_state_path.exists():
            try:
                previous_state = json.loads(paths.run_state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous_state = {}
        state = {
            **previous_state,
            "run_id": paths.run_id,
            "backend": self.name,
            "status": "initializing",
            "cluster_head_address": config.cluster.head_address,
            "config_path": str(paths.resolved_config_path),
            "resumed_from_checkpoint": resume,
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
        actor_nodes = _selfplay_ray_nodes(
            ray,
            include_head=bool(remote_nodes) and config.cluster.head_selfplay_actors != 0,
        )
        remote_stats = {
            "remote_workers_available": len(remote_nodes),
            "remote_workers_scheduled": 0,
            "remote_workers_completed": 0,
            "remote_workers_failed": 0,
            "remote_games": 0,
            "remote_positions": 0,
            "remote_replay_samples_imported": 0,
        }
        games = 0
        if actor_nodes:
            max_iterations = config.stop.max_iterations or 1
            total_game_budget = config.stop.max_games or (
                config.selfplay.games_per_iteration * max_iterations
            )
            previous_games_seen = int(previous_state.get("games_seen", 0)) if resume else 0
            remaining_game_budget = max(0, total_game_budget - previous_games_seen)
            games = min(remaining_game_budget, config.selfplay.games_per_iteration)
        if actor_nodes and games > 0:
            worker_actor = make_ray_worker_actor()
            node_actor_counts: list[tuple[dict[str, Any], bool, RaySelfPlayNodePlan]] = []
            remaining_actor_budget = games
            for node, is_head in actor_nodes:
                if remaining_actor_budget <= 0:
                    break
                node_plan = _ray_selfplay_node_plan(
                    config,
                    node,
                    is_head=is_head,
                    remaining_actor_budget=remaining_actor_budget,
                )
                node_actor_counts.append((node, is_head, node_plan))
                remaining_actor_budget -= node_plan.actor_count
            actor_slots: list[tuple[dict[str, Any], bool, int, int]] = []
            slot = 0
            total_actors = sum(plan.actor_count for _, _, plan in node_actor_counts)
            for node, is_head, node_plan in node_actor_counts:
                for _ in range(node_plan.actor_count):
                    actor_slots.append((node, is_head, slot, node_plan.cpus_per_actor))
                    slot += 1
            if total_actors != len(actor_slots):
                raise RuntimeError("Ray actor plan size mismatch")
            game_allocations = _allocate_games_by_weight(
                games,
                [cpus_per_actor for _, _, _, cpus_per_actor in actor_slots],
            )
            actor_plan = [
                (node, is_head, worker_slot, actor_games, cpus_per_actor)
                for (node, is_head, worker_slot, cpus_per_actor), actor_games in zip(
                    actor_slots,
                    game_allocations,
                    strict=True,
                )
            ]
            actor_offset = 0
            for node, is_head, node_plan in node_actor_counts:
                node_games = sum(
                    game_allocations[actor_offset : actor_offset + node_plan.actor_count]
                )
                actor_offset += node_plan.actor_count
                event_writer.write(
                    {
                        "event": "lan_ray_remote_selfplay_scheduled",
                        "ray_node_id": str(node.get("NodeID", "")),
                        "ray_node_address": _ray_node_address(node),
                        "is_head_node": is_head,
                        "actors": node_plan.actor_count,
                        "actor_limit": node_plan.actor_limit,
                        "cpus_per_actor": node_plan.cpus_per_actor,
                        "games": node_games,
                        "cpu_count": node_plan.cpu_count,
                        "memory_gib": (
                            None
                            if node_plan.memory_bytes is None
                            else round(node_plan.memory_bytes / _GIB, 2)
                        ),
                    }
                )
            remote_stats["remote_workers_scheduled"] = len(actor_plan)
            checkpoint_payload = _checkpoint_payload_bytes(paths.run_dir) if resume else None
            if checkpoint_payload is None and resume:
                raise RuntimeError(
                    f"cannot resume {paths.run_dir}: latest checkpoint payload not found"
                )
            actors = []
            try:
                for node, _, _, _, cpus_per_actor in actor_plan:
                    node_id = str(node.get("NodeID", ""))
                    actors.append(
                        worker_actor.options(
                            **_ray_node_affinity_options(node_id),
                            num_cpus=cpus_per_actor,
                            runtime_env={
                                "env_vars": {
                                    "GAZ_RAY_ACTOR_TORCH_THREADS": str(cpus_per_actor),
                                }
                            },
                            max_restarts=1,
                            max_task_retries=1,
                        ).remote()
                    )
            except Exception:
                _kill_ray_actors(ray, actors)
                raise
            shared = len(actors) > 1
            config_payload = _put_shared_argument(
                ray,
                _worker_config_payload(config),
                shared=shared,
            )
            checkpoint_argument = _put_shared_argument(
                ray,
                checkpoint_payload,
                shared=shared and checkpoint_payload is not None,
            )
            futures = [
                actor.generate_replay_shard.remote(
                    config_payload,
                    work_dir=None,
                    games=actor_games,
                    worker_slot=worker_slot,
                    seed=config.run.seed + 100_000 + index * 10_000,
                    device="cpu" if is_head else None,
                    checkpoint_payload_bytes=checkpoint_argument,
                )
                for index, (
                    actor,
                    (_, is_head, worker_slot, actor_games, cpus_per_actor),
                ) in enumerate(zip(actors, actor_plan, strict=True))
                if actor_games > 0
            ]
            head = HeadController(run_dir=paths.run_dir, event_writer=event_writer)
            upload_dir = paths.run_dir / "ray_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            try:
                for index, future in _ready_futures(
                    ray,
                    futures,
                    timeout_sec=(
                        None
                        if config.stop.max_wall_time_sec is None
                        else max(
                            0.0,
                            config.stop.max_wall_time_sec - (perf_counter() - lifecycle_started),
                        )
                    ),
                ):
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
                    shard_path = upload_dir / (
                        f"{index:06d}_"
                        f"{_safe_upload_component(payload['worker_id'])}_"
                        f"{_safe_upload_component(payload['replay_shard_name'])}"
                    )
                    shard_path.write_bytes(payload.pop("replay_shard_bytes"))
                    try:
                        imported = head.upload_replay_shard(
                            payload["worker_id"],
                            str(shard_path),
                        )
                    finally:
                        shard_path.unlink(missing_ok=True)
                    remote_stats["remote_workers_completed"] += 1
                    if imported.get("imported"):
                        remote_stats["remote_games"] += int(payload.get("games", 0))
                        remote_stats["remote_positions"] += int(payload.get("positions", 0))
                        remote_stats["remote_replay_samples_imported"] += int(
                            imported.get("samples", 0)
                        )
                    event_writer.write(
                        {
                            "event": "lan_ray_remote_selfplay_completed",
                            **payload,
                            "ray_node_id": str(node.get("NodeID", "")),
                            "ray_node_address": _ray_node_address(node),
                            "imported": imported,
                        }
                    )
            finally:
                try:
                    upload_dir.rmdir()
                except OSError:
                    pass
                _kill_ray_actors(ray, actors)
        elif not actor_nodes:
            event_writer.write(
                {
                    "event": "lan_ray_no_remote_workers",
                    "run_id": paths.run_id,
                    "note": "No remote Ray worker nodes were available; using head self-play.",
                }
            )
        else:
            event_writer.write(
                {
                    "event": "lan_ray_remote_selfplay_skipped",
                    "run_id": paths.run_id,
                    "reason": "max_games_reached",
                    "remote_workers_available": len(remote_nodes),
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
            skip_initial_selfplay_if_replay_available=False,
            resume=resume,
            started_at=lifecycle_started,
        ).run()
        if paths.run_state_path.exists():
            final_state = json.loads(paths.run_state_path.read_text(encoding="utf-8"))
            final_state.update(remote_stats)
            atomic_write_json(paths.run_state_path, final_state)
        return result
