# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Application-neutral extraction of RLinf's Newton ROS stepping path.

The existing engine protocol has no command-id ACK. See DESIGN.md before
using arrival_v1 completion as a physical tick/commit guarantee.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from ._process import _terminate_process_group
from ._ros_bridge import RosEngineBridge
from .config import ClientConfig


@dataclass(frozen=True)
class WorldSchema:
    joint_names: tuple[str, ...]  # per-world names, common ordering
    body_paths: tuple[str, ...]
    num_envs: int


@dataclass(frozen=True)
class WorldSnapshot:
    schema: WorldSchema
    joint_positions: np.ndarray  # (N,J), owned read-only copies
    joint_velocities: np.ndarray  # (N,J)
    body_poses: np.ndarray  # (N,B,7), xyz + quaternion WXYZ
    observation_id: int  # client-local read sequence, NOT an engine tick
    joint_sequence: int
    body_sequence: int
    joint_stamp: float  # publisher wall clock, NOT sim_time
    body_stamp: float
    reset_epoch: int
    completion: str = "arrival_v1"


class RenderBridge(Protocol):
    def render(self, snapshot: WorldSnapshot) -> Any:
        """Apply poses and wait for rendering in the application's renderer."""


@dataclass(frozen=True)
class StepResult:
    snapshot: WorldSnapshot
    rendered: Any = None


def _layout(names: tuple[str, ...], worlds: int) -> tuple[tuple[str, ...], np.ndarray]:
    """Validate uniform worlds and return indices in world-major wire order."""
    if not names or len(set(names)) != len(names):
        raise ValueError("state names must be nonempty and unique")
    qualified = [bool(re.match(r"env_\d+/", name)) for name in names]
    if worlds == 1 and not any(qualified):
        return names, np.arange(len(names))
    if not all(qualified):
        raise ValueError("state mixes world-qualified and unqualified names")
    groups = [[] for _ in range(worlds)]
    for index, name in enumerate(names):
        match = re.fullmatch(r"env_(\d+)/(.*)", name)
        if match is None or int(match[1]) >= worlds or not match[2]:
            raise ValueError(f"invalid world-qualified name: {name}")
        groups[int(match[1])].append((match[2], index))
    common = tuple(name for name, _ in groups[0])
    if not common or any(set(name for name, _ in group) != set(common) for group in groups):
        raise ValueError("worlds have different or missing state layouts")
    order = [dict(group)[name] for group in groups for name in common]
    return common, np.array(order)


def _readonly(array: np.ndarray) -> np.ndarray:
    # bytes-backed arrays cannot have WRITEABLE switched back on by consumers.
    array = np.ascontiguousarray(array, dtype=np.float32)
    return np.frombuffer(array.tobytes(), dtype=np.float32).reshape(array.shape)


class SimulationClient:
    """One exclusive, serialized Newton session, optionally owning its process.

    ``spawn=False`` attaches to an engine launched with this exact configuration.
    step/reset/start/close reject concurrent calls; a transport failure after a
    command poisons the session until close/start. No retry can replay actions.
    """

    completion = "arrival_v1"

    def __init__(
        self,
        config: ClientConfig,
        *,
        spawn: bool = True,
        ready_timeout_s: float = 600,
        step_timeout_s: float = 5,
        log_path: str | None = None,
        renderer: RenderBridge | None = None,
    ):
        if any(not math.isfinite(x) or x <= 0 for x in (ready_timeout_s, step_timeout_s)):
            raise ValueError("client timeouts must be finite and positive")
        self.config = config
        self._spawn = spawn
        self._ready_timeout = ready_timeout_s
        self._step_timeout = step_timeout_s
        self.log_path = log_path
        self._renderer = renderer
        self._bridge = None
        self._process = None
        self._log = None
        self._started = False
        self._failed = False
        self._gate = threading.Lock()
        self._schema = None
        self._wire_joint_names = ()
        self._observation_id = 0

    @contextmanager
    def _exclusive(self):
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("another client operation is in progress")
        try:
            yield
        finally:
            self._gate.release()

    def _check_health(self):
        if self._process is not None and self._process.poll() is not None:
            raise RuntimeError(f"Newton process exited ({self._process.returncode}); log: {self.log_path}")
        self._bridge.check_health()

    def _wait(self, predicate, timeout: float, description: str):
        deadline = time.monotonic() + timeout
        while True:
            self._check_health()
            state = self._bridge.read_state()
            if predicate(state):
                return state
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{description} timed out after {timeout:g}s; log: {self.log_path}")
            time.sleep(0.0005)

    def start(self):
        with self._exclusive():
            if self._started:
                self._require_started()
                return self
            if self._process is not None:
                self._dispose()  # refuse to overwrite a process whose cleanup previously failed
            try:
                # Subscribe before spawning so readiness includes fresh samples.
                self._bridge = RosEngineBridge(node_name=f"genie_sim_client_{os.getpid()}")
                self._bridge._expected_envs = self.config.num_envs
                self._bridge.start()
                if self._spawn:
                    if shutil.which("ros2") is None:
                        raise RuntimeError("ros2 is unavailable; source ROS and the colcon install first")
                    if self.log_path is None:
                        self._log = tempfile.NamedTemporaryFile(
                            mode="a", prefix="genie_sim_client_", suffix=".log", delete=False
                        )
                        self.log_path = self._log.name
                    else:
                        self._log = open(self.log_path, "a", buffering=1)
                    self._process = subprocess.Popen(
                        self.config.command, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                # Count distinct messages, not repeated reads of a single cache.
                seen = -1
                previous = None
                stable = 0

                def ready(state):
                    nonlocal seen, previous, stable
                    if state["joint_sequence"] != seen:
                        seen = state["joint_sequence"]
                        names = state["joint_names"]
                        stable = stable + 1 if names == previous else 1
                        previous = names
                    if stable < 3 or not state["joint_names"] or state["body_poses"] is None:
                        return False
                    _layout(state["joint_names"], self.config.num_envs)
                    _layout(state["body_names"], self.config.num_envs)
                    return self._bridge.command_readers() > 0

                state = self._wait(ready, self._ready_timeout, "Newton state/layout readiness")
                self._wire_joint_names = state["joint_names"]
                self._bridge.set_joint_layout(list(self._wire_joint_names))
                self._schema = None
                self._snapshot(state)  # validate shapes/finite values before success
                self._failed = False
                self._started = True
                return self
            except BaseException:
                self._dispose()
                raise

    def _require_started(self):
        if not self._started or self._failed:
            raise RuntimeError("client is closed or failed; close/start the session before use")
        self._check_health()

    def _snapshot(self, state):
        joints, ji = _layout(state["joint_names"], self.config.num_envs)
        bodies, bi = _layout(state["body_names"], self.config.num_envs)
        schema = WorldSchema(joints, bodies, self.config.num_envs)
        if self._schema is not None and schema != self._schema:
            raise RuntimeError("engine topology changed during the session; rebind after close/start")
        arrays = []
        for key, indices, width in [("positions", ji, 1), ("velocities", ji, 1), ("body_poses", bi, 7)]:
            value = np.asarray(state[key], dtype=np.float32)
            expected = (len(indices),) if width == 1 else (len(indices), width)
            if value.shape != expected or not np.isfinite(value).all():
                raise RuntimeError(f"invalid {key}: expected finite state of shape {expected}, got {value.shape}")
            shape = (schema.num_envs, -1) if width == 1 else (schema.num_envs, -1, width)
            arrays.append(_readonly(value[indices].reshape(shape)))
        self._schema = schema
        self._observation_id += 1
        return WorldSnapshot(
            schema,
            *arrays,
            self._observation_id,
            state["joint_sequence"],
            state["body_sequence"],
            state["joint_stamp"],
            state["body_stamp"],
            state["reset_epoch"],
        )

    def snapshot(self) -> WorldSnapshot:
        with self._exclusive():
            self._require_started()
            try:
                return self._snapshot(self._bridge.read_state())
            except BaseException:
                self._failed = True
                raise

    def step(self, joint_positions: np.ndarray, *, render: bool = False, completion: str = "arrival_v1") -> StepResult:
        with self._exclusive():
            self._require_started()
            if completion != self.completion:
                raise NotImplementedError("engine has no command-id commit ACK; only arrival_v1 is supported")
            if render and self._renderer is None:
                raise ValueError("render=True requires a RenderBridge")
            action = np.asarray(joint_positions, dtype=np.float32)
            shape = (self.config.num_envs, len(self._schema.joint_names))
            if action.shape != shape or not np.isfinite(action).all():
                raise ValueError(f"expected finite joint targets of shape {shape}")
            # Preserve RLinf's wire order even if worlds publish in another order.
            _, order = _layout(self._wire_joint_names, self.config.num_envs)
            target = np.empty(action.size, dtype=np.float32)
            target[order] = action.reshape(-1)
            before = self._bridge.read_state()
            try:
                self._bridge.send_action(target)
                state = self._wait(
                    lambda s: s["joint_sequence"] > before["joint_sequence"],
                    self._step_timeout,
                    "joint-state arrival after action",
                )
                # TF is a separate channel; do not hand an old pose batch to rendering.
                state = self._wait(
                    lambda s: s["body_stamp"] >= s["joint_stamp"] and s["body_sequence"] > before["body_sequence"],
                    self._step_timeout,
                    "body-state arrival after action",
                )
                snapshot = self._snapshot(state)
            except BaseException:
                self._failed = True
                raise
            # Rendering failure cannot undo physics or trigger an action retry.
            rendered = self._renderer.render(snapshot) if render else None
            return StepResult(snapshot, rendered)

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
        *,
        seeds: Sequence[int] | None = None,
        instances: Sequence[int] | None = None,
    ) -> WorldSnapshot:
        with self._exclusive():
            self._require_started()
            ids = list(range(self.config.num_envs)) if env_ids is None else list(env_ids)
            if (
                not ids
                or any(
                    isinstance(i, (bool, np.bool_))
                    or not isinstance(i, (int, np.integer))
                    or not 0 <= i < self.config.num_envs
                    for i in ids
                )
                or len(set(ids)) != len(ids)
            ):
                raise ValueError("env_ids must contain unique, in-range integer world IDs")
            ids = [int(i) for i in ids]
            for values in (seeds, instances):
                if values is not None and (
                    len(values) != len(ids)
                    or any(
                        isinstance(x, (bool, np.bool_))
                        or not isinstance(x, (int, np.integer))
                        or not -1 <= x <= np.iinfo(np.int64).max
                        for x in values
                    )
                ):
                    raise ValueError("seeds/instances must match env_ids and contain integers >= -1")
            before = self._bridge.read_state()
            if before["reset_epoch"] < 0:
                raise NotImplementedError("engine does not advertise reset_epoch; stale-state fallback is disabled")
            seeds = None if seeds is None else [int(x) for x in seeds]
            instances = None if instances is None else [int(x) for x in instances]
            seeded = any(values is not None and any(x >= 0 for x in values) for values in (seeds, instances))
            self._bridge.prepare_reset(seeded)
            self._wait(
                lambda _: self._bridge.reset_readers(seeded) > 0, self._step_timeout, "reset subscriber discovery"
            )
            try:
                self._bridge.send_reset(ids, seeds, instances)
                state = self._wait(
                    lambda s: s["reset_epoch"] > before["reset_epoch"], self._step_timeout, "reset epoch"
                )
                state = self._wait(
                    lambda s: s["body_sequence"] > before["body_sequence"] and s["body_stamp"] >= state["reset_stamp"],
                    self._step_timeout,
                    "post-reset body state",
                )
                return self._snapshot(state)
            except BaseException:
                self._failed = True
                raise

    def _dispose(self):
        self._started = False
        self._failed = True
        try:
            if self._bridge is not None:
                self._bridge.close()
        finally:
            self._bridge = None
            try:
                if self._process is not None:
                    _terminate_process_group(self._process)
                self._process = None
            finally:
                if self._log is not None:
                    self._log.close()
                self._log = None
                self._schema = None

    @property
    def is_started(self) -> bool:
        return self._started and not self._failed

    def close(self):
        with self._exclusive():
            self._dispose()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
