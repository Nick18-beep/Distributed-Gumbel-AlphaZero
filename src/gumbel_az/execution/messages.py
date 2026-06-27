"""Typed messages shared by execution backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class RunTask:
    task_type: Literal["initialize_run"]
    run_dir: Path


@dataclass(frozen=True)
class WorkerHeartbeat:
    worker_id: str
    status: Literal["idle", "busy", "stopping"]


@dataclass(frozen=True)
class WorkerCapabilities:
    worker_id: str
    hostname: str
    platform: str
    cpu_count: int
    runtime_backend: str
    torch_device: str = "cpu"
    torch_devices: tuple[str, ...] = ()
    has_gpu: bool = False


@dataclass
class WorkerRecord:
    worker_id: str
    capabilities: WorkerCapabilities
    status: Literal["idle", "busy", "stopping", "lost"] = "idle"
    registered_at: datetime = field(default_factory=utc_now)
    last_heartbeat_at: datetime = field(default_factory=utc_now)
    commands: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "capabilities": {
                "worker_id": self.capabilities.worker_id,
                "hostname": self.capabilities.hostname,
                "platform": self.capabilities.platform,
                "cpu_count": self.capabilities.cpu_count,
                "runtime_backend": self.capabilities.runtime_backend,
                "torch_device": self.capabilities.torch_device,
                "torch_devices": list(self.capabilities.torch_devices),
                "has_gpu": self.capabilities.has_gpu,
            },
            "status": self.status,
            "registered_at": self.registered_at.isoformat().replace("+00:00", "Z"),
            "last_heartbeat_at": self.last_heartbeat_at.isoformat().replace("+00:00", "Z"),
            "commands": list(self.commands),
        }


@dataclass
class LeaseRecord:
    lease_id: str
    task_id: str
    task_type: str
    payload: dict[str, Any]
    worker_id: str | None = None
    status: Literal["pending", "leased", "completed", "failed"] = "pending"
    attempts: int = 0
    created_at: datetime = field(default_factory=utc_now)
    lease_expires_at: datetime | None = None
    completed_at: datetime | None = None
    last_error: str | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.status != "leased" or self.lease_expires_at is None:
            return False
        return (now or utc_now()) >= self.lease_expires_at

    def lease_to(self, worker_id: str, lease_seconds: float) -> None:
        self.worker_id = worker_id
        self.status = "leased"
        self.attempts += 1
        self.lease_expires_at = utc_now() + timedelta(seconds=lease_seconds)

    def to_json(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "worker_id": self.worker_id,
            "status": self.status,
            "attempts": self.attempts,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "lease_expires_at": None
            if self.lease_expires_at is None
            else self.lease_expires_at.isoformat().replace("+00:00", "Z"),
            "completed_at": None
            if self.completed_at is None
            else self.completed_at.isoformat().replace("+00:00", "Z"),
            "last_error": self.last_error,
        }
