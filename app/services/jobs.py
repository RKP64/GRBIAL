from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TraceEvent:
    seq: int
    ts: str
    kind: str            # chunk | info | warning | error
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "ts": self.ts, "kind": self.kind,
                "message": self.message, "data": self.data}


@dataclass
class Job:
    id: str
    kind: str
    domain: str
    state: JobState = JobState.QUEUED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    total: int = 0
    done: int = 0
    counters: dict[str, int] = field(default_factory=lambda: {
        "nodes_added": 0, "edges_added": 0, "rejects": 0, "chunk_errors": 0
    })
    error: str | None = None
    files: list[str] = field(default_factory=list)
    trace: deque[TraceEvent] = field(default_factory=lambda: deque(maxlen=500))
    rejects: deque[dict] = field(default_factory=lambda: deque(maxlen=500))
    _seq: int = 0
    _subscribers: list[asyncio.Queue] = field(default_factory=list)
    _cancel: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def progress(self) -> float:
        return round(self.done / self.total, 4) if self.total else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "domain": self.domain,
            "state": self.state.value, "progress": self.progress,
            "done": self.done, "total": self.total, "counters": dict(self.counters),
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "error": self.error, "files": self.files,
        }

    def emit(self, kind: str, message: str, **data: Any) -> None:
        self._seq += 1
        ev = TraceEvent(self._seq, datetime.now(timezone.utc).isoformat(), kind, message, data)
        self.trace.append(ev)
        payload = {"type": "trace", "job": self.summary(), "event": ev.as_dict()}
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()


class JobRegistry:
    """In-process registry.

    Deliberately behind a small interface: swapping to Cosmos or Redis for
    multi-replica deployments replaces this class only.
    """

    def __init__(self, max_jobs: int = 200) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque(maxlen=max_jobs)

    def create(self, kind: str, domain: str, files: list[str] | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, domain=domain, files=files or [])
        if len(self._order) == self._order.maxlen and self._order:
            self._jobs.pop(self._order[0], None)
        self._jobs[job.id] = job
        self._order.append(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        ids = list(self._order)[::-1][:limit]
        return [self._jobs[i].summary() for i in ids if i in self._jobs]

    async def stream(self, job: Job) -> AsyncIterator[dict[str, Any]]:
        """Server-sent event source: replay current state, then live updates."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        job._subscribers.append(queue)
        try:
            yield {"type": "snapshot", "job": job.summary(),
                   "events": [e.as_dict() for e in list(job.trace)[-50:]]}
            while True:
                if job.state in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED) and queue.empty():
                    yield {"type": "final", "job": job.summary()}
                    return
                try:
                    yield await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield {"type": "heartbeat", "job": job.summary()}
        finally:
            if queue in job._subscribers:
                job._subscribers.remove(queue)


registry = JobRegistry()
