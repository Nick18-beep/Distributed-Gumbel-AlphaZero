"""Task lease manager for retryable LAN work."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from uuid import uuid4

from gumbel_az.execution.messages import LeaseRecord, utc_now


class TaskLeaseManager:
    """Bounded in-memory lease table with explicit expiration and retry."""

    def __init__(self, *, lease_seconds: float = 60.0, max_attempts: int = 3) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self._tasks: dict[str, LeaseRecord] = {}

    def submit(self, task_type: str, payload: dict, *, task_id: str | None = None) -> LeaseRecord:
        task_id = task_id or uuid4().hex
        if task_id in self._tasks:
            raise ValueError(f"task already exists: {task_id}")
        record = LeaseRecord(
            lease_id=uuid4().hex,
            task_id=task_id,
            task_type=task_type,
            payload=dict(payload),
        )
        self._tasks[task_id] = record
        return record

    def acquire(self, worker_id: str) -> LeaseRecord | None:
        self.expire_leases()
        for record in self._tasks.values():
            if record.status != "pending":
                continue
            if record.attempts >= self.max_attempts:
                record.status = "failed"
                record.last_error = "max attempts exceeded"
                continue
            record.lease_id = uuid4().hex
            record.lease_to(worker_id, self.lease_seconds)
            return record
        return None

    def complete(
        self,
        task_id: str,
        lease_id: str,
        *,
        worker_id: str | None = None,
    ) -> LeaseRecord:
        record = self._require_leased(task_id, lease_id, worker_id=worker_id)
        record.status = "completed"
        record.completed_at = utc_now()
        return record

    def fail(
        self,
        task_id: str,
        lease_id: str,
        error: str,
        *,
        retry: bool = True,
        worker_id: str | None = None,
    ) -> LeaseRecord:
        record = self._require_leased(task_id, lease_id, worker_id=worker_id)
        record.last_error = error
        if retry and record.attempts < self.max_attempts:
            record.status = "pending"
            record.worker_id = None
            record.lease_expires_at = None
        else:
            record.status = "failed"
            record.completed_at = utc_now()
        return record

    def expire_leases(self, *, now: datetime | None = None) -> list[LeaseRecord]:
        now = now or utc_now()
        expired = []
        for record in self._tasks.values():
            if not record.is_expired(now):
                continue
            expired.append(record)
            record.worker_id = None
            record.lease_expires_at = None
            if record.attempts >= self.max_attempts:
                record.status = "failed"
                record.last_error = "lease expired; max attempts exceeded"
                record.completed_at = now
            else:
                record.status = "pending"
                record.last_error = "lease expired"
        return expired

    def pending_or_leased(self) -> Iterable[LeaseRecord]:
        return (record for record in self._tasks.values() if record.status in {"pending", "leased"})

    def snapshot(self) -> dict[str, dict]:
        return {task_id: record.to_json() for task_id, record in sorted(self._tasks.items())}

    def _require_leased(
        self,
        task_id: str,
        lease_id: str,
        *,
        worker_id: str | None = None,
    ) -> LeaseRecord:
        if task_id not in self._tasks:
            raise KeyError(f"unknown task {task_id!r}")
        record = self._tasks[task_id]
        if record.status != "leased" or record.lease_id != lease_id:
            raise ValueError(f"task {task_id!r} is not held by lease {lease_id!r}")
        if worker_id is not None and record.worker_id != worker_id:
            raise ValueError(f"task {task_id!r} is not leased to worker {worker_id!r}")
        if record.is_expired():
            self.expire_leases()
            raise ValueError(f"task {task_id!r} lease has expired")
        return record
