# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""CPU-only measured JointState validation and per-joint FK target selection."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class JointFeedbackConfig:
    """Opt-in measured input configured exclusively by the scene YAML."""

    topic: str = "/joint_states_measured"
    timeout_s: float = 0.5
    stale_policy: str = "hold"

    @classmethod
    def from_scene(cls, scene: dict) -> JointFeedbackConfig | None:
        """Validate feedback configuration; absence disables all subscriptions."""
        raw = (scene.get("libuipc") or {}).get("joint_feedback")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("libuipc.joint_feedback must be a mapping")
        topic = raw.get("topic", "/joint_states_measured")
        if not isinstance(topic, str) or not topic.startswith("/") or topic in ("/", "/joint_states"):
            raise ValueError(
                "libuipc.joint_feedback.topic must be an absolute input topic, distinct from /joint_states"
            )
        timeout = float(raw.get("timeout_s", 0.5))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("libuipc.joint_feedback.timeout_s must be finite and positive")
        policy = raw.get("stale_policy", "hold")
        if policy not in ("hold", "command"):
            raise ValueError("libuipc.joint_feedback.stale_policy must be hold or command")
        return cls(topic, timeout, policy)


class JointFeedbackSelector:
    """Keep measured, commanded, and selected positions separate under one lock.

    Bindings map ROS leaf names to ``(q_index, lower_limit, upper_limit)``.
    Limits use joint DOF indices at model binding time, whereas selected
    positions use configuration indices. Callbacks never touch Warp arrays.
    """

    def __init__(
        self,
        config: JointFeedbackConfig,
        bindings: dict,
        initial_q: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.bindings = dict(bindings)
        self._clock = clock
        self._lock = threading.Lock()
        self.reset(initial_q)

    def reset(self, initial_q: Any) -> None:
        """Reseed hold positions and discard all measurement freshness."""
        with self._lock:
            self._selected = np.array(initial_q, dtype=np.float32, copy=True)
            self._measured = {}
            self._arrivals = {}
            self._command = {}
            self._selection = {name: "hold" for name in self.bindings}
            self._received = self._mapped = self._rejected = self._stale = self._ignored = 0
            self._last_rejection = ""

    def receive(self, names: Any, positions: Any) -> bool:
        """Accept a complete valid message atomically using local arrival time."""
        with self._lock:
            self._received += 1
            try:
                if len(names) == 0 or len(names) != len(positions):
                    raise ValueError("empty or mismatched name/position arrays")
                if len(set(names)) != len(names):
                    raise ValueError("duplicate joint names")
                update = {}
                ignored = 0
                for name, value in zip(names, positions):
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(f"nonfinite position: {name}")
                    # Upstream models may retain joints Main fixes in the
                    # scene; those names must not renew any mapped lease.
                    if name not in self.bindings:
                        ignored += 1
                        continue
                    update[name] = value
                if not update:
                    raise ValueError("no mapped joint names")
            except (TypeError, ValueError, OverflowError) as exc:
                self._rejected += 1
                self._last_rejection = str(exc)
                return False
            now = self._clock()
            self._measured.update(update)
            self._arrivals.update({name: now for name in update})
            self._mapped += len(update)
            self._ignored += ignored
            return True

    def select(self, command_q: Any) -> np.ndarray:
        """Select fresh measured joints, holding or using valid latched commands."""
        with self._lock:
            now = self._clock()
            self._stale = 0
            for name, (index, lower, upper) in self.bindings.items():
                command = float(command_q[index])
                self._command[name] = command
                stamp = self._arrivals.get(name)
                if stamp is not None and 0 <= now - stamp <= self.config.timeout_s:
                    self._selected[index] = min(upper, max(lower, self._measured[name]))
                    self._selection[name] = "measured"
                else:
                    self._stale += 1
                    self._selection[name] = "hold"
                    if self.config.stale_policy == "command" and math.isfinite(command):
                        self._selected[index] = min(upper, max(lower, command))
                        self._selection[name] = "command"
            return self._selected.copy()

    def snapshot(self) -> dict:
        """Copy diagnostics without modifying the scene recording schema."""
        with self._lock:
            return {
                "received": self._received,
                "mapped": self._mapped,
                "rejected": self._rejected,
                "ignored": self._ignored,
                "last_rejection": self._last_rejection,
                "stale": self._stale,
                "commanded": dict(self._command),
                "measured": dict(self._measured),
                "selected": {name: float(self._selected[index]) for name, (index, _, _) in self.bindings.items()},
                "selection": dict(self._selection),
            }


__all__ = ["JointFeedbackConfig", "JointFeedbackSelector"]
