from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Iterable, Mapping


class RequestState(str, Enum):
    WAITING = "waiting"
    FETCHING = "fetching"
    READY = "ready"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


class RequestLane(str, Enum):
    NO_FETCH = "no_fetch"
    KV_FETCH = "kv_fetch"


@dataclass(frozen=True, slots=True)
class RuntimeRequest:
    request_id: str
    fetch_key: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.fetch_key is not None and not self.fetch_key:
            raise ValueError("fetch_key must not be empty when provided")

    @property
    def lane(self) -> RequestLane:
        return RequestLane.KV_FETCH if self.fetch_key is not None else RequestLane.NO_FETCH


@dataclass(frozen=True, slots=True)
class SchedulerLimits:
    max_concurrent_fetches: int = 1
    max_running_no_fetch: int = 1
    max_running_fetch: int = 1

    def __post_init__(self) -> None:
        if self.max_concurrent_fetches < 0:
            raise ValueError("max_concurrent_fetches must be non-negative")
        if self.max_running_no_fetch < 0:
            raise ValueError("max_running_no_fetch must be non-negative")
        if self.max_running_fetch < 0:
            raise ValueError("max_running_fetch must be non-negative")


@dataclass(frozen=True, slots=True)
class SchedulerEvent:
    request_id: str
    from_state: RequestState
    to_state: RequestState
    reason: str


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    request_id: str
    lane: RequestLane
    state: RequestState
    sequence: int
    fetch_key: str | None


@dataclass(frozen=True, slots=True)
class SchedulerStep:
    events: tuple[SchedulerEvent, ...]
    started_fetches: tuple[str, ...]
    started_running: tuple[str, ...]


@dataclass(slots=True)
class _RequestRecord:
    request: RuntimeRequest
    state: RequestState
    sequence: int


class IsolatedFetchScheduler:
    """Deterministic scheduler model with separate no-fetch and KV-fetch lanes.

    No-fetch requests never queue behind remote KV restoration: they use their own
    ready/running lane and capacity.  KV-fetch requests first enter FETCHING, then
    become READY only when the caller reports fetch completion.
    """

    def __init__(self, limits: SchedulerLimits | None = None) -> None:
        self.limits = limits or SchedulerLimits()
        self._records: dict[str, _RequestRecord] = {}
        self._next_sequence = 0
        self._waiting_no_fetch: Deque[str] = deque()
        self._ready_no_fetch: Deque[str] = deque()
        self._running_no_fetch: set[str] = set()
        self._waiting_fetch: Deque[str] = deque()
        self._fetching: set[str] = set()
        self._ready_fetch: Deque[str] = deque()
        self._running_fetch: set[str] = set()

    def submit(self, request: RuntimeRequest) -> None:
        if request.request_id in self._records:
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        record = _RequestRecord(request=request, state=RequestState.WAITING, sequence=self._next_sequence)
        self._next_sequence += 1
        self._records[request.request_id] = record
        if request.lane is RequestLane.KV_FETCH:
            self._waiting_fetch.append(request.request_id)
        else:
            self._waiting_no_fetch.append(request.request_id)

    def state(self, request_id: str) -> RequestState:
        return self._record(request_id).state

    def request(self, request_id: str) -> RuntimeRequest:
        return self._record(request_id).request

    def snapshot(self) -> tuple[RequestSnapshot, ...]:
        return tuple(
            RequestSnapshot(
                request_id=record.request.request_id,
                lane=record.request.lane,
                state=record.state,
                sequence=record.sequence,
                fetch_key=record.request.fetch_key,
            )
            for record in sorted(self._records.values(), key=lambda item: item.sequence)
        )

    def advance(
        self,
        *,
        fetch_completed: Iterable[str] = (),
        fetch_failed: Iterable[str] = (),
        completed: Iterable[str] = (),
    ) -> SchedulerStep:
        events: list[SchedulerEvent] = []
        started_fetches: list[str] = []
        started_running: list[str] = []
        completed_set = set(completed)
        fetch_failed_set = set(fetch_failed)
        fetch_completed_set = set(fetch_completed)
        if fetch_failed_set & fetch_completed_set:
            raise ValueError("a request cannot be both fetch_completed and fetch_failed in one step")

        self._finish_running(completed_set, events)
        self._fail_fetches(fetch_failed_set, events)
        self._complete_fetches(fetch_completed_set, events)
        self._start_fetches(events, started_fetches)
        self._promote_no_fetch(events)
        self._start_ready_no_fetch(events, started_running)
        self._start_ready_fetch(events, started_running)
        return SchedulerStep(events=tuple(events), started_fetches=tuple(started_fetches), started_running=tuple(started_running))

    def _record(self, request_id: str) -> _RequestRecord:
        try:
            return self._records[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request_id {request_id!r}") from exc

    def _transition(
        self,
        request_id: str,
        to_state: RequestState,
        reason: str,
        events: list[SchedulerEvent],
    ) -> None:
        record = self._record(request_id)
        from_state = record.state
        if from_state is to_state:
            return
        record.state = to_state
        events.append(SchedulerEvent(request_id=request_id, from_state=from_state, to_state=to_state, reason=reason))

    def _ordered_ids(self, request_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted(request_ids, key=lambda request_id: self._record(request_id).sequence))

    def _finish_running(self, completed: set[str], events: list[SchedulerEvent]) -> None:
        for request_id in self._ordered_ids(completed):
            record = self._record(request_id)
            if record.state is not RequestState.RUNNING:
                raise ValueError(f"request {request_id!r} is not running")
            if request_id in self._running_no_fetch:
                self._running_no_fetch.remove(request_id)
            elif request_id in self._running_fetch:
                self._running_fetch.remove(request_id)
            else:
                raise ValueError(f"request {request_id!r} is not in a running lane")
            self._transition(request_id, RequestState.FINISHED, "request_completed", events)

    def _fail_fetches(self, failed: set[str], events: list[SchedulerEvent]) -> None:
        for request_id in self._ordered_ids(failed):
            record = self._record(request_id)
            if record.state is not RequestState.FETCHING:
                raise ValueError(f"request {request_id!r} is not fetching")
            self._fetching.remove(request_id)
            self._transition(request_id, RequestState.FAILED, "fetch_failed", events)

    def _complete_fetches(self, completed: set[str], events: list[SchedulerEvent]) -> None:
        for request_id in self._ordered_ids(completed):
            record = self._record(request_id)
            if record.state is not RequestState.FETCHING:
                raise ValueError(f"request {request_id!r} is not fetching")
            self._fetching.remove(request_id)
            self._ready_fetch.append(request_id)
            self._transition(request_id, RequestState.READY, "fetch_completed", events)

    def _start_fetches(self, events: list[SchedulerEvent], started_fetches: list[str]) -> None:
        while len(self._fetching) < self.limits.max_concurrent_fetches and self._waiting_fetch:
            request_id = self._waiting_fetch.popleft()
            self._fetching.add(request_id)
            self._transition(request_id, RequestState.FETCHING, "fetch_started", events)
            started_fetches.append(request_id)

    def _promote_no_fetch(self, events: list[SchedulerEvent]) -> None:
        while self._waiting_no_fetch:
            request_id = self._waiting_no_fetch.popleft()
            self._ready_no_fetch.append(request_id)
            self._transition(request_id, RequestState.READY, "no_fetch_ready", events)

    def _start_ready_no_fetch(self, events: list[SchedulerEvent], started_running: list[str]) -> None:
        while len(self._running_no_fetch) < self.limits.max_running_no_fetch and self._ready_no_fetch:
            request_id = self._ready_no_fetch.popleft()
            self._running_no_fetch.add(request_id)
            self._transition(request_id, RequestState.RUNNING, "no_fetch_started", events)
            started_running.append(request_id)

    def _start_ready_fetch(self, events: list[SchedulerEvent], started_running: list[str]) -> None:
        while len(self._running_fetch) < self.limits.max_running_fetch and self._ready_fetch:
            request_id = self._ready_fetch.popleft()
            self._running_fetch.add(request_id)
            self._transition(request_id, RequestState.RUNNING, "fetch_request_started", events)
            started_running.append(request_id)
