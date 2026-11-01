# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from collections import deque
import copy
from dataclasses import dataclass, field
import itertools
import threading
import time
from typing import Callable


class RenderCallTimeout(TimeoutError):
    def __init__(self, call_id, label, disposition):
        self.call_id = call_id
        self.label = label
        self.disposition = disposition
        super().__init__(f"render call {call_id} ({label}) exceeded its deadline: {disposition}")


class RenderSchedulerClosed(RuntimeError):
    def __init__(self, reason, call_id=None, label=None):
        self.reason = str(reason)
        self.call_id = call_id
        self.label = label
        if call_id is None:
            message = f"render scheduler is closed: {self.reason}"
        else:
            message = f"render call {call_id} ({label}) was retired during close: {self.reason}"
        super().__init__(message)


@dataclass
class _RenderCall:
    call_id: str
    label: str
    func: object
    args: tuple
    kwargs: dict
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    state: str = "queued"
    value: object = None
    error: BaseException = None
    caller_disposition: str = "waiting"
    started_at: float = None
    finished_at: float = None
    late_outcome_recorded: bool = False
    completion_ready: threading.Event = field(default_factory=threading.Event)
    completion_snapshot: tuple = None


class RenderLoopScheduler:
    """Run deadline-aware calls on the existing render-loop owner."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, on_enqueue: Callable[[], None] = None):
        self._lock = threading.RLock()
        self._on_enqueue = on_enqueue
        self._calls = deque()
        self._ids = itertools.count(1)
        self._clock = clock
        self._accepting = True
        self._close_reason = None
        self._running = None
        self._retired_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._rejected_count = 0
        self._late_outcomes = []

    def call(self, func, args, kwargs, *, deadline, label):
        deadline = float(deadline)
        label = str(label)
        with self._lock:
            call_id = f"render-{next(self._ids)}"
            if not self._accepting:
                self._rejected_count += 1
                raise RenderSchedulerClosed(self._close_reason, call_id, label)
            if self._clock() >= deadline:
                self._retired_count += 1
                raise RenderCallTimeout(call_id, label, "retired_before_start")
            call = _RenderCall(
                call_id=call_id,
                label=label,
                func=func,
                args=tuple(args),
                kwargs=dict(kwargs),
                deadline=deadline,
            )
            self._calls.append(call)
            if self._on_enqueue is not None:
                self._on_enqueue()

        while True:
            remaining = call.deadline - self._clock()
            if remaining > 0:
                call.done.wait(remaining)
            with self._lock:
                now = self._clock()
                if call.state == "running" and call.completion_ready.is_set():
                    self._commit_ready_completion_locked(call)
                if call.state == "queued":
                    if now < call.deadline:
                        continue
                    error = RenderCallTimeout(call.call_id, call.label, "retired_before_start")
                    self._retire_locked(call, error)
                    raise error
                if call.state == "running":
                    if now < call.deadline:
                        continue
                    call.caller_disposition = "timed_out_in_flight"
                    raise RenderCallTimeout(call.call_id, call.label, "timed_out_in_flight")
                if call.state == "retired":
                    call.caller_disposition = "raised"
                    raise call.error
                if call.caller_disposition == "timed_out_in_flight":
                    raise RenderCallTimeout(call.call_id, call.label, "timed_out_in_flight")
                if call.state == "completed":
                    call.caller_disposition = "returned"
                    return call.value
                if call.state == "failed":
                    call.caller_disposition = "raised"
                    raise call.error
                raise AssertionError(f"unknown render call state: {call.state}")

    def _retire_locked(self, call, error):
        call.state = "retired"
        call.error = error
        call.caller_disposition = "raised"
        call.func = None
        call.args = ()
        call.kwargs = {}
        self._retired_count += 1
        call.done.set()

    def _complete_locked(self, call, state, value, error, finished_at):
        call.state = state
        call.value = value
        call.error = error
        call.finished_at = finished_at
        self._running = None
        if state == "completed":
            self._completed_count += 1
        else:
            self._failed_count += 1

        completed_late = call.finished_at >= call.deadline
        caller_already_timed_out = call.caller_disposition == "timed_out_in_flight"
        if completed_late or caller_already_timed_out:
            call.caller_disposition = "timed_out_in_flight"
            self._record_late_outcome_locked(call)
        elif state == "completed":
            call.caller_disposition = "returned"
        else:
            call.caller_disposition = "raised"
        call.done.set()

    def _commit_ready_completion_locked(self, call):
        if call.state != "running" or not call.completion_ready.is_set():
            return False
        state, value, error, finished_at = call.completion_snapshot
        self._complete_locked(call, state, value, error, finished_at)
        return True

    def _record_late_outcome_locked(self, call):
        if call.late_outcome_recorded:
            return
        outcome = {
            "call_id": call.call_id,
            "label": call.label,
            "status": call.state,
            "deadline": call.deadline,
            "started_at": call.started_at,
            "finished_at": call.finished_at,
        }
        if call.error is not None:
            outcome["error"] = {
                "type": type(call.error).__name__,
                "message": str(call.error),
            }
        self._late_outcomes.append(outcome)
        call.late_outcome_recorded = True

    def run_one(self):
        while True:
            with self._lock:
                if not self._calls:
                    return False
                call = self._calls.popleft()
                if call.state != "queued":
                    continue
                started_at = self._clock()
                if started_at >= call.deadline:
                    error = RenderCallTimeout(call.call_id, call.label, "retired_before_start")
                    self._retire_locked(call, error)
                    continue
                call.state = "running"
                call.started_at = started_at
                self._running = call
                break

        value = None
        error = None
        try:
            value = call.func(*call.args, **call.kwargs)
        except BaseException as caught:
            finished_at = self._clock()
            error = caught
            state = "failed"
        else:
            finished_at = self._clock()
            state = "completed"
        call.completion_snapshot = (state, value, error, finished_at)
        call.completion_ready.set()
        with self._lock:
            self._commit_ready_completion_locked(call)
        if error is not None and not isinstance(error, Exception):
            raise error
        return True

    def has_queued_calls(self):
        with self._lock:
            return any(call.state == "queued" for call in self._calls)

    def close(self, reason):
        reason = str(reason)
        with self._lock:
            if self._running is not None:
                self._commit_ready_completion_locked(self._running)
            if self._accepting:
                self._accepting = False
                self._close_reason = reason
            for call in self._calls:
                if call.state != "queued":
                    continue
                error = RenderSchedulerClosed(self._close_reason, call.call_id, call.label)
                self._retire_locked(call, error)

    def receipt(self):
        with self._lock:
            if self._running is not None:
                self._commit_ready_completion_locked(self._running)
            queued_count = sum(call.state == "queued" for call in self._calls)
            running = None
            if self._running is not None:
                running = {
                    "call_id": self._running.call_id,
                    "label": self._running.label,
                    "deadline": self._running.deadline,
                    "started_at": self._running.started_at,
                    "state": self._running.state,
                    "caller_disposition": self._running.caller_disposition,
                }
            return {
                "accepting": self._accepting,
                "close_reason": self._close_reason,
                "drained": queued_count == 0 and running is None,
                "queued_count": queued_count,
                "running": running,
                "retired_count": self._retired_count,
                "completed_count": self._completed_count,
                "failed_count": self._failed_count,
                "rejected_count": self._rejected_count,
                "late_outcomes": copy.deepcopy(self._late_outcomes),
            }
