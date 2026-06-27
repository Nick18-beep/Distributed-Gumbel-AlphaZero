"""Worker heartbeat registry for LAN execution."""

from __future__ import annotations

from datetime import datetime, timedelta

from gumbel_az.execution.messages import WorkerCapabilities, WorkerRecord, utc_now


class HeartbeatRegistry:
    """Track worker registration and liveness with explicit timeouts."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self._workers: dict[str, WorkerRecord] = {}

    def register(self, capabilities: WorkerCapabilities) -> WorkerRecord:
        record = WorkerRecord(
            worker_id=capabilities.worker_id,
            capabilities=capabilities,
            status="idle",
            registered_at=utc_now(),
            last_heartbeat_at=utc_now(),
        )
        self._workers[capabilities.worker_id] = record
        return record

    def heartbeat(self, worker_id: str, *, status: str = "idle") -> WorkerRecord:
        if worker_id not in self._workers:
            raise KeyError(f"unknown worker {worker_id!r}")
        if status not in {"idle", "busy", "stopping"}:
            raise ValueError(f"invalid worker status {status!r}")
        record = self._workers[worker_id]
        record.status = status  # type: ignore[assignment]
        record.last_heartbeat_at = utc_now()
        return record

    def enqueue_command(self, worker_id: str, command: str) -> None:
        if worker_id not in self._workers:
            raise KeyError(f"unknown worker {worker_id!r}")
        self._workers[worker_id].commands.append(command)

    def pop_commands(self, worker_id: str) -> list[str]:
        if worker_id not in self._workers:
            raise KeyError(f"unknown worker {worker_id!r}")
        commands = list(self._workers[worker_id].commands)
        self._workers[worker_id].commands.clear()
        return commands

    def mark_stale_workers_lost(self, *, now: datetime | None = None) -> list[str]:
        now = now or utc_now()
        timeout = timedelta(seconds=self.timeout_seconds)
        lost = []
        for worker_id, record in self._workers.items():
            if record.status == "lost":
                continue
            if now - record.last_heartbeat_at > timeout:
                record.status = "lost"
                lost.append(worker_id)
        return lost

    def get(self, worker_id: str) -> WorkerRecord:
        if worker_id not in self._workers:
            raise KeyError(f"unknown worker {worker_id!r}")
        return self._workers[worker_id]

    def snapshot(self) -> dict[str, dict]:
        return {worker_id: record.to_json() for worker_id, record in sorted(self._workers.items())}
