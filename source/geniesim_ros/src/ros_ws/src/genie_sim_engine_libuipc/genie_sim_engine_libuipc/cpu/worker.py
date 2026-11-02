# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Independent CPU joint dynamics with bounded latest-value communication.

Keep this module outside engine.newton: a spawned worker must not import the
GPU engine, create a CUDA context, or inherit a running ROS executor.
"""

from __future__ import annotations

import ctypes
import logging
import math
import multiprocessing as mp
import re
import threading
import time
from dataclasses import dataclass

import numpy as np

from .viewer import CpuViewerConfig


@dataclass(frozen=True)
class CpuMuJoCoConfig:
    """Resolve the documented libuipc.cpu_mujoco scene settings."""

    synchronization: str = "latest"
    sync_substeps: int = 1
    sync_timeout_s: float = 0.5
    dynamic_proxies: tuple = ()
    execution: str = "process"
    physics_hz: float = 1000.0
    timeout_s: float = 0.5
    state_timeout_s: float = 0.5
    iterations: int = 50
    tolerance: float = 1e-8
    friction: float = 0.3
    static_collision_margin: float = 0.0
    robot_self_collision: bool = False
    robot_self_collision_excludes: tuple = ()
    drives: tuple = ()
    viewer: CpuViewerConfig = CpuViewerConfig()

    @classmethod
    def from_scene(cls, scene):
        """Disable absent profiles and reject ambiguous or invalid settings."""
        newton_cfg = scene.get("libuipc") or {}
        raw = newton_cfg.get("cpu_mujoco")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("libuipc.cpu_mujoco must be a mapping")
        if newton_cfg.get("joint_feedback") is not None:
            raise ValueError("cpu_mujoco and external joint_feedback are mutually exclusive")
        if set(raw) - set(cls.__dataclass_fields__):
            raise ValueError(f"Unknown cpu_mujoco keys: {sorted(set(raw) - set(cls.__dataclass_fields__))}")
        cfg = cls(**{**raw, "viewer": CpuViewerConfig.from_mapping(raw.get("viewer", {}))})
        if not isinstance(cfg.robot_self_collision, bool):
            raise ValueError("cpu_mujoco.robot_self_collision must be a boolean")
        if not isinstance(cfg.robot_self_collision_excludes, (list, tuple)):
            raise ValueError("cpu_mujoco.robot_self_collision_excludes must be a list of body pairs")
        for pair in cfg.robot_self_collision_excludes:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(not isinstance(name, str) or not name.strip() for name in pair)
                or pair[0] == pair[1]
            ):
                raise ValueError("cpu_mujoco.robot_self_collision_excludes requires distinct nonempty body names")
        if cfg.synchronization not in ("latest", "barrier"):
            raise ValueError("cpu_mujoco.synchronization must be latest or barrier")
        if type(cfg.sync_substeps) is not int or cfg.sync_substeps < 1:
            raise ValueError("cpu_mujoco.sync_substeps must be a positive integer")
        _proxy_names(cfg)
        if cfg.execution not in ("thread", "process"):
            raise ValueError("cpu_mujoco.execution must be thread or process")
        for key in (
            "physics_hz",
            "timeout_s",
            "state_timeout_s",
            "sync_timeout_s",
            "tolerance",
            "friction",
            "static_collision_margin",
        ):
            value = getattr(cfg, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"cpu_mujoco.{key} must be finite")
            allow_zero = key in ("friction", "static_collision_margin")
            if value < 0 or (value == 0 and not allow_zero):
                raise ValueError(f"cpu_mujoco.{key} must be {'nonnegative' if allow_zero else 'positive'}")
        if isinstance(cfg.iterations, bool) or not isinstance(cfg.iterations, int) or cfg.iterations < 1:
            raise ValueError("cpu_mujoco.iterations must be a positive integer")
        if not isinstance(cfg.drives, (list, tuple)):
            raise ValueError("cpu_mujoco.drives must be a list")
        for drive in cfg.drives:
            required = {"pattern", "kp", "kd", "force_limit"}
            optional = {"armature", "damping", "frictionloss"}
            if not isinstance(drive, dict) or not required <= set(drive) or set(drive) - required - optional:
                raise ValueError("Each CPU drive requires pattern, kp, kd and force_limit")
            if not isinstance(drive["pattern"], str) or not drive["pattern"]:
                raise ValueError("CPU drive pattern must be a nonempty regex")
            try:
                re.compile(drive["pattern"])
            except re.error as exc:
                raise ValueError("Invalid CPU drive regex") from exc
            for key in ("kp", "kd", "force_limit", *sorted(optional & set(drive))):
                value = drive[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"CPU drive {key} must be finite")
                if value < 0 or (value == 0 and key in ("kp", "force_limit")):
                    raise ValueError(f"CPU drive {key} must be positive (kd may be zero)")
        return cfg


def _body_names(names):
    """Validate an ordered selection of exact CPU model body names."""
    if isinstance(names, (str, bytes)):
        raise ValueError("CPU body names must be a sequence of nonempty unique strings")
    try:
        names = tuple(names)
    except TypeError as exc:
        raise ValueError("CPU body names must be a sequence of nonempty unique strings") from exc
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("CPU body names must be nonempty and unique")
    return names


def _proxy_names(config):
    """Require an explicit ordered proxy selection and the barrier clock."""
    names = config.dynamic_proxies
    if (
        not isinstance(names, (list, tuple))
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("cpu_mujoco.dynamic_proxies must be a list of unique nonempty body names")
    if names and config.synchronization != "barrier":
        raise ValueError("cpu_mujoco.dynamic_proxies requires barrier synchronization")
    return tuple(names)


def _proxy_state(count, poses, velocities):
    """Copy and validate a complete world-origin pose / world-COM twist frame."""
    poses, velocities = np.asarray(poses), np.asarray(velocities)
    if poses.dtype.kind not in "fiu" or velocities.dtype.kind not in "fiu":
        raise ValueError("CPU proxy poses and velocities must contain real numeric values")
    poses = poses.astype(float, copy=True)
    velocities = velocities.astype(float, copy=True)
    if poses.shape != (count, 7) or velocities.shape != (count, 6):
        raise ValueError(f"CPU proxy state requires poses ({count}, 7) and velocities ({count}, 6)")
    if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
        raise ValueError("CPU proxy state must be finite")
    lengths = np.linalg.norm(poses[:, 3:], axis=1)
    # IPC stores float32 rotations. Accept its roundoff, then normalize the
    # copied quaternion before converting world twists to local free-joint DOFs.
    if not np.allclose(lengths, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("CPU proxy pose quaternion must be normalized")
    poses[:, 3:] /= lengths[:, None]
    return poses, velocities


def _simulation_time(value, expected=None):
    """Validate the authority's completed time, optionally against the prior ACK."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
        or value < 0
        or (expected is not None and not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9))
    ):
        raise ValueError(f"CPU simulation time does not match completed IPC time {expected}")
    return float(value)


class CpuStepper:
    """Execute named position actuators using MuJoCo's native CPU dynamics."""

    def __init__(self, model_bytes, config, names, initial, body_names=()):
        import mujoco

        self.mj = mujoco
        self.config = config
        self.body_names = _body_names(body_names)
        self.proxy_names = _proxy_names(config)
        self.model = model = mujoco.MjModel.from_binary_path("model.mjb", assets={"model.mjb": model_bytes})
        self.body_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.body_names], dtype=int
        )
        for name, body_id in zip(self.body_names, self.body_ids):
            if body_id < 0:
                raise ValueError(f"CPU model is missing body {name}")
        self.proxy_body_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.proxy_names], dtype=int
        )
        proxy_qids, proxy_vids = [], []
        for name, body in zip(self.proxy_names, self.proxy_body_ids):
            if body < 1:
                raise ValueError(f"CPU model is missing proxy body {name}")
            joint = model.body_jntadr[body]
            if (
                model.body_jntnum[body] != 1
                or joint < 0
                or int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE)
                or not math.isfinite(model.body_mass[body])
                or model.body_mass[body] <= 0
            ):
                raise ValueError(f"CPU proxy body {name} requires a finite-mass free joint")
            proxy_qids.append(model.jnt_qposadr[joint] + np.arange(7))
            proxy_vids.append(model.jnt_dofadr[joint] + np.arange(6))
        self.proxy_qids = np.asarray(proxy_qids, dtype=int).reshape((-1, 7))
        self.proxy_vids = np.asarray(proxy_vids, dtype=int).reshape((-1, 6))
        model.opt.timestep = 1.0 / config.physics_hz
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        model.opt.iterations = config.iterations
        model.opt.tolerance = config.tolerance
        model.geom_friction[:, 0] = config.friction
        self.data = data = mujoco.MjData(model)
        joints = {}
        for joint_id in range(model.njnt):
            full_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            name = full_name.rsplit("/", 1)[-1]
            if name in joints:
                raise ValueError(f"Ambiguous CPU joint leaf name: {name}")
            joints[name] = joint_id
        qids, vids, aids, limits = [], [], [], []
        for name in names:
            if name not in joints:
                raise ValueError(f"CPU model is missing joint {name}")
            joint = joints[name]
            if int(model.jnt_type[joint]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise ValueError(f"CPU command joint must be scalar: {name}")
            actuators = np.flatnonzero(
                (model.actuator_trnid[:, 0] == joint) & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            )
            if len(actuators) != 1:
                raise ValueError(f"CPU joint {name} requires exactly one position actuator, got {len(actuators)}")
            matches = [drive for drive in config.drives if re.fullmatch(drive["pattern"], name)]
            if len(matches) != 1:
                raise ValueError(f"CPU joint {name} requires exactly one matching YAML drive")
            drive = matches[0]
            actuator = int(actuators[0])
            model.actuator_gaintype[actuator] = mujoco.mjtGain.mjGAIN_FIXED
            model.actuator_biastype[actuator] = mujoco.mjtBias.mjBIAS_AFFINE
            model.actuator_gainprm[actuator] = 0.0
            model.actuator_gainprm[actuator, 0] = drive["kp"]
            model.actuator_biasprm[actuator] = 0.0
            model.actuator_biasprm[actuator, 1:3] = (-drive["kp"], -drive["kd"])
            model.actuator_gear[actuator] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            model.actuator_forcelimited[actuator] = True
            model.actuator_forcerange[actuator] = (-drive["force_limit"], drive["force_limit"])
            # Imported USD effort limits also create an aggregate joint cap.
            # A larger YAML servo limit must update both or the torso sags.
            model.jnt_actfrclimited[joint] = True
            model.jnt_actfrcrange[joint] = (-drive["force_limit"], drive["force_limit"])
            model.actuator_ctrllimited[actuator] = False
            for key in ("armature", "damping", "frictionloss"):
                if key in drive:
                    getattr(model, "dof_" + key)[model.jnt_dofadr[joint]] = drive[key]
            qids.append(int(model.jnt_qposadr[joint]))
            vids.append(int(model.jnt_dofadr[joint]))
            aids.append(actuator)
            limits.append(model.jnt_range[joint] if model.jnt_limited[joint] else (-np.inf, np.inf))
        self.qids, self.vids, self.aids = map(np.asarray, (qids, vids, aids))
        self.limits = np.asarray(limits)
        self.targets = np.asarray(initial, dtype=float).copy()
        if self.targets.shape != (len(names),) or not np.isfinite(self.targets).all():
            raise ValueError("Invalid CPU initial joint state")
        data.qpos[self.qids] = self.targets
        data.ctrl[self.aids] = self.targets
        self._fresh = np.zeros(len(names), dtype=bool)
        mujoco.mj_forward(model, data)

    def step(self, targets, stamps, *, now):
        """Consume fresh commands; on expiry hold the then-current position."""
        fresh = (stamps > 0) & (now - stamps <= self.config.timeout_s)
        expired = self._fresh & ~fresh
        self.targets[expired] = self.data.qpos[self.qids[expired]]
        self.targets[fresh] = np.clip(targets[fresh], self.limits[fresh, 0], self.limits[fresh, 1])
        self._fresh = fresh
        self.data.ctrl[self.aids] = self.targets
        self.mj.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("CPU MuJoCo produced a nonfinite state")
        # MuJoCo may reset unstable state to finite zeros; warnings are fatal too.
        for kind in (
            self.mj.mjtWarning.mjWARN_BADQPOS,
            self.mj.mjtWarning.mjWARN_BADQVEL,
            self.mj.mjtWarning.mjWARN_BADQACC,
        ):
            if self.data.warning[kind].number:
                raise RuntimeError(f"CPU MuJoCo instability: {kind.name}")
        if self.body_names or self.proxy_names:
            # mj_step integrates qpos/qvel after computing link transforms. Refresh
            # only kinematics and COM motion data, not collisions or dynamics.
            self.mj.mj_kinematics(self.model, self.data)
            self.mj.mj_comPos(self.model, self.data)
            self.mj.mj_comVel(self.model, self.data)

    def state(self):
        """Copy the controlled positions and velocities in the supplied order."""
        return self.data.qpos[self.qids].copy(), self.data.qvel[self.vids].copy()

    def body_state(self):
        """Copy world body poses [xyz, xyzw] and COM twists [linear, angular]."""
        return self._body_state(self.body_ids)

    def proxy_state(self):
        """Copy completed proxy state for diagnostics and the read-only viewer."""
        return self._body_state(self.proxy_body_ids)

    def set_proxy_state(self, poses, velocities):
        """Replace only proxy free joints, once before a CPU barrier batch."""
        poses, velocities = _proxy_state(len(self.proxy_names), poses, velocities)
        qpos = poses[:, (0, 1, 2, 6, 3, 4, 5)]
        qvel = np.empty_like(velocities)
        for index, body in enumerate(self.proxy_body_ids):
            rotation = np.empty(9)
            self.mj.mju_quat2Mat(rotation, qpos[index, 3:])
            rotation = rotation.reshape((3, 3))
            omega = velocities[index, 3:]
            # Free-joint translation is world origin velocity; angular DOFs are
            # body-local. IPC supplies world COM linear and angular velocities.
            qvel[index, :3] = velocities[index, :3] - np.cross(omega, rotation @ self.model.body_ipos[body])
            qvel[index, 3:] = rotation.T @ omega
        self.data.qpos[self.proxy_qids] = qpos
        self.data.qvel[self.proxy_vids] = qvel
        # Contact constraints couple robot and proxies; neither side's previous
        # acceleration remains a valid warm start after replacing a proxy pose.
        self.data.qacc_warmstart[:] = 0.0
        self.mj.mj_kinematics(self.model, self.data)
        self.mj.mj_comPos(self.model, self.data)
        self.mj.mj_comVel(self.model, self.data)

    def _body_state(self, body_ids):
        """Read an exact body selection using MuJoCo's world COM convention."""
        poses = np.concatenate((self.data.xpos[body_ids], self.data.xquat[body_ids][:, (1, 2, 3, 0)]), axis=1)
        velocities = np.empty((len(body_ids), 6), dtype=float)
        for index, body_id in enumerate(body_ids):
            # MuJoCo 3.10 mjOBJ_BODY uses xipos (COM); mjOBJ_XBODY uses xpos.
            # flg_local=0 returns world [angular, linear], opposite Newton order.
            self.mj.mj_objectVelocity(
                self.model, self.data, self.mj.mjtObj.mjOBJ_BODY, int(body_id), velocities[index], 0
            )
        if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
            raise RuntimeError("CPU MuJoCo produced nonfinite body state")
        return poses, velocities[:, (3, 4, 5, 0, 1, 2)]


class _Mailbox:
    """Transfer fixed-size snapshots under a short lock, without a FIFO."""

    def __init__(self, size, context=None):
        self.buffer = context.RawArray("d", size) if context else np.zeros(size, dtype=float)
        self.lock = context.Lock() if context else threading.Lock()

    def read(self):
        """Return a coherent copy, or None immediately if the writer is busy."""
        if not self.lock.acquire(False):
            return None
        try:
            return np.asarray(self.buffer).copy()
        finally:
            self.lock.release()

    def write(self, values):
        """Replace the complete latest snapshot in one bounded critical section."""
        with self.lock:
            np.asarray(self.buffer)[:] = values

    def read_blocking(self, *, deadline):
        """Read an acknowledged snapshot within the transaction's remaining time."""
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self.lock.acquire(timeout=remaining):
            raise TimeoutError("CPU barrier snapshot exceeded transaction deadline")
        try:
            snapshot = np.asarray(self.buffer).copy()
            if time.monotonic() >= deadline:
                raise TimeoutError("CPU barrier snapshot exceeded transaction deadline")
            return snapshot
        finally:
            self.lock.release()


def _run_worker(
    model_bytes,
    config,
    names,
    initial,
    commands,
    states,
    stop,
    ready,
    failed,
    error,
    body_names=(),
    sync_connection=None,
):
    """Run native physics asynchronously, or only for validated numbered requests."""
    try:
        sim = CpuStepper(model_bytes, config, names, initial, body_names=body_names)
        n = len(names)
        current = np.concatenate((initial, np.zeros(n)))
        count = 0
        deadline = time.monotonic()

        def write_snapshot(started, finished):
            q, v = sim.state()
            body_poses, body_velocities = sim.body_state()
            proxy_poses, proxy_velocities = sim.proxy_state()
            penetration = min((contact.dist for contact in sim.data.contact), default=0.0)
            states.write(
                np.concatenate(
                    (
                        [finished, count, sim.data.time, finished - started, sim.data.ncon, penetration],
                        q,
                        v,
                        body_poses.ravel(),
                        body_velocities.ravel(),
                        proxy_poses.ravel(),
                        proxy_velocities.ravel(),
                    )
                )
            )

        ready.set()
        if config.synchronization == "barrier":
            previous, completed = 0, None
            while not stop.is_set():
                if not sync_connection.poll(0.05):
                    continue
                packet = sync_connection.recv()
                if not isinstance(packet, dict) or set(packet) != {
                    "sequence",
                    "targets",
                    "proxy_poses",
                    "proxy_velocities",
                    "completed_sim_time",
                }:
                    raise RuntimeError("invalid CPU barrier request structure")
                sequence = packet["sequence"]
                if type(sequence) is not int or sequence != previous + 1:
                    raise RuntimeError("invalid CPU barrier sequence")
                start_time = _simulation_time(packet["completed_sim_time"], completed)
                targets = packet["targets"]
                if targets is not None:
                    targets = np.asarray(targets, dtype=float)
                    if targets.shape != (n,) or not np.isfinite(targets).all():
                        raise ValueError("invalid frozen CPU targets")
                if sim.proxy_names:
                    sim.set_proxy_state(packet["proxy_poses"], packet["proxy_velocities"])
                elif packet["proxy_poses"] is not None or packet["proxy_velocities"] is not None:
                    raise ValueError("CPU proxy frame supplied without configured proxies")
                # Engine time is the authority, including a new owner after reset.
                sim.data.time = start_time
                update = commands.read()
                if update is not None:
                    current = update
                started = time.monotonic()
                for _ in range(config.sync_substeps):
                    now = time.monotonic()
                    sim.step(
                        current[:n] if targets is None else targets,
                        current[n:] if targets is None else np.full(n, now),
                        now=now,
                    )
                count += config.sync_substeps
                completed = start_time + config.sync_substeps / config.physics_hz
                if not math.isclose(sim.data.time, completed, rel_tol=0.0, abs_tol=1e-9):
                    raise RuntimeError("CPU barrier clock increment mismatch")
                write_snapshot(started, time.monotonic())
                sync_connection.send(dict(sequence=sequence, steps=count, start_time=start_time, sim_time=completed))
                previous = sequence
            return
        while not stop.is_set():
            update = commands.read()
            if update is not None:
                current = update
            started = time.monotonic()
            sim.step(current[:n], current[n:], now=started)
            finished = time.monotonic()
            count += 1
            write_snapshot(started, finished)
            deadline += 1.0 / config.physics_hz
            if deadline < finished:
                deadline = finished
            stop.wait(max(0.0, deadline - time.monotonic()))
    except BaseException as exc:
        error.value = f"{type(exc).__name__}: {exc}".encode()[:4095]
        failed.set()
        ready.set()


class CpuJointWorker:
    """Own CPU lifecycle and named ROS command reception independently of IPC."""

    def __init__(self, model_bytes, config, names, initial, body_names=()):
        self.config = config
        self.names = tuple(names)
        self.body_names = _body_names(body_names)
        self.proxy_names = _proxy_names(config)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("CPU joint names must be nonempty and unique")
        self._indices = {name: index for index, name in enumerate(self.names)}
        self._initial = np.asarray(initial, dtype=float).copy()
        self._model_bytes = model_bytes
        # A thread-based solver also needs a process-shared mailbox for its GUI.
        context = mp.get_context("spawn") if config.execution == "process" or config.viewer.enabled else None
        event = context.Event if context else threading.Event
        self._stop, self._ready, self._failed = event(), event(), event()
        self._error = context.RawArray("c", 4096) if context else ctypes.create_string_buffer(4096)
        self._commands = _Mailbox(2 * len(names), context)
        self._states = _Mailbox(6 + 2 * len(names) + 13 * (len(self.body_names) + len(self.proxy_names)), context)
        self._commands.write(np.concatenate((self._initial, np.zeros(len(names)))))
        self._context = context if config.execution == "process" else None
        self._sync_parent = self._sync_child = None
        self._sync_sequence = 0
        self._sync_sim_time = None
        self._sync_lock = threading.Lock()
        self._sync_broken = False
        if config.synchronization == "barrier":
            pipe = context.Pipe if context else mp.Pipe
            self._sync_parent, self._sync_child = pipe(duplex=True)
        self._worker = None
        self._viewer_context = context if config.viewer.enabled else None
        self._viewer_process = None
        self._viewer_stop = context.Event() if config.viewer.enabled else None
        self._viewer_ready = context.Event() if config.viewer.enabled else None
        self.received = self.rejected = 0

    def start(self):
        """Start an owned CPU worker and fail clearly if model loading fails."""
        if self._worker is not None:
            raise RuntimeError("CPU worker is already started; create a new owner to restart")
        factory = self._context.Process if self._context else threading.Thread
        self._worker = factory(
            target=_run_worker,
            name="geniesim-cpu-mujoco",
            daemon=True,
            args=(
                self._model_bytes,
                self.config,
                self.names,
                self._initial,
                self._commands,
                self._states,
                self._stop,
                self._ready,
                self._failed,
                self._error,
                self.body_names,
                self._sync_child,
            ),
        )
        self._worker.start()
        if self._context and self._sync_child is not None:
            self._sync_child.close()
        if not self._ready.wait(15.0):
            self.close()
            raise RuntimeError("CPU MuJoCo startup timed out")
        try:
            self.check_health()
        except BaseException:
            self.close()
            raise
        if self._viewer_context is not None:
            from .viewer import run_viewer

            process = self._viewer_context.Process(
                target=run_viewer,
                name="geniesim-cpu-mujoco-viewer",
                daemon=True,
                args=(
                    self._model_bytes,
                    self.config,
                    self.names,
                    self._initial,
                    self._states,
                    self._viewer_stop,
                    self._viewer_ready,
                    self.body_names,
                ),
            )
            try:
                process.start()
                self._viewer_process = process
            except Exception:
                logging.getLogger("genie_sim_engine_libuipc.cpu_viewer").warning(
                    "Could not start CPU viewer; physics continues without it", exc_info=True
                )

    def receive(self, names, positions):
        """Validate an entire message before renewing any per-joint lease."""
        try:
            if not names or len(names) != len(positions) or len(set(names)) != len(names):
                raise ValueError("empty, duplicated, or mismatched joint arrays")
            values = np.asarray(positions, dtype=float)
            if values.shape != (len(names),) or not np.isfinite(values).all():
                raise ValueError("invalid joint positions")
            updates = [(self._indices[name], value) for name, value in zip(names, values) if name in self._indices]
            if not updates:
                return False
        except (TypeError, ValueError):
            self.rejected += 1
            return False
        with self._commands.lock:
            buf = np.asarray(self._commands.buffer)
            now = time.monotonic()
            for index, value in updates:
                buf[index] = value
                buf[len(self.names) + index] = now
        self.received += 1
        return True

    def snapshot(self):
        """Read latest actual state without waiting for a CPU step or lock."""
        return self._decode_snapshot(self._states.read())

    def advance_sync(self, *, targets=None, proxy_poses=None, proxy_velocities=None, completed_sim_time=None):
        """Advance exactly one CPU batch seeded from the completed IPC clock/state.

        Every failure after attempting to send is sticky: the worker may already
        have advanced and cannot safely repeat the coupling transaction.
        """
        with self._sync_lock:
            if self._sync_broken:
                raise RuntimeError("CPU barrier synchronization is broken; create a new owner")
            if self.config.synchronization != "barrier" or self._sync_parent is None:
                raise RuntimeError("CPU worker has no barrier synchronization")
            if self._worker is None or self._stop.is_set():
                raise RuntimeError("CPU barrier worker is not started")
            try:
                self.check_health()
            except BaseException:
                self._sync_broken = True
                raise
            start_time = _simulation_time(completed_sim_time, self._sync_sim_time)
            if targets is not None:
                targets = np.asarray(targets, dtype=float).copy()
                if targets.shape != (len(self.names),) or not np.isfinite(targets).all():
                    raise ValueError("invalid frozen CPU targets")
            if self.proxy_names:
                proxy_poses, proxy_velocities = _proxy_state(len(self.proxy_names), proxy_poses, proxy_velocities)
            elif proxy_poses is not None or proxy_velocities is not None:
                raise ValueError("CPU proxy frame supplied without configured dynamic_proxies")
            sequence = self._sync_sequence + 1
            expected_steps = sequence * self.config.sync_substeps
            end_time = start_time + self.config.sync_substeps / self.config.physics_hz
            deadline = time.monotonic() + self.config.sync_timeout_s
            try:
                self._sync_parent.send(
                    dict(
                        sequence=sequence,
                        targets=targets,
                        proxy_poses=proxy_poses,
                        proxy_velocities=proxy_velocities,
                        completed_sim_time=start_time,
                    )
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._sync_parent.poll(remaining):
                    raise TimeoutError(f"CPU barrier sequence {sequence} exceeded {self.config.sync_timeout_s:g}s")
                ack = self._sync_parent.recv()
                if (
                    not isinstance(ack, dict)
                    or set(ack) != {"sequence", "steps", "start_time", "sim_time"}
                    or type(ack["sequence"]) is not int
                    or type(ack["steps"]) is not int
                    or ack["sequence"] != sequence
                    or ack["steps"] != expected_steps
                ):
                    raise RuntimeError("CPU barrier acknowledgement mismatch")
                _simulation_time(ack["start_time"], start_time)
                _simulation_time(ack["sim_time"], end_time)
                self.check_health()
                snapshot = self._decode_snapshot(self._states.read_blocking(deadline=deadline))
                if (
                    snapshot is None
                    or snapshot["steps"] != expected_steps
                    or not math.isclose(snapshot["sim_time"], end_time, rel_tol=0.0, abs_tol=1e-9)
                ):
                    raise RuntimeError("CPU barrier acknowledgement has no matching snapshot")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"CPU barrier sequence {sequence} exceeded {self.config.sync_timeout_s:g}s")
            except BaseException as exc:
                self._sync_broken = True
                if isinstance(exc, (TimeoutError, RuntimeError, KeyboardInterrupt, SystemExit)):
                    raise
                raise RuntimeError("CPU barrier protocol failed") from exc
            self._sync_sequence = sequence
            self._sync_sim_time = end_time
            snapshot["sync_sequence"] = sequence
            snapshot["start_time"] = start_time
            return snapshot

    def _decode_snapshot(self, state):
        if state is None or state[0] == 0.0:
            return None
        size = 6 + 2 * len(self.names) + 13 * (len(self.body_names) + len(self.proxy_names))
        if (
            not isinstance(state, np.ndarray)
            or state.shape != (size,)
            or not np.isfinite(state).all()
            or state[1] < 0
            or state[1] != int(state[1])
            or state[4] < 0
            or state[4] != int(state[4])
        ):
            raise RuntimeError("Invalid CPU state snapshot")
        n = len(self.names)
        body_start = 6 + 2 * n
        velocity_start = body_start + 7 * len(self.body_names)
        proxy_start = velocity_start + 6 * len(self.body_names)
        proxy_velocity_start = proxy_start + 7 * len(self.proxy_names)
        return dict(
            wall_time=state[0],
            steps=int(state[1]),
            sim_time=state[2],
            step_s=state[3],
            contacts=int(state[4]),
            min_contact_dist=state[5],
            positions=state[6 : 6 + n],
            velocities=state[6 + n : body_start],
            body_poses=state[body_start:velocity_start].reshape((-1, 7)),
            body_velocities=state[velocity_start:proxy_start].reshape((-1, 6)),
            proxy_poses=state[proxy_start:proxy_velocity_start].reshape((-1, 7)),
            proxy_velocities=state[proxy_velocity_start:].reshape((-1, 6)),
            received=self.received,
            rejected=self.rejected,
        )

    def check_health(self):
        """Report construction, stepping, or unexpected worker exit failures."""
        if self._failed.is_set():
            raise RuntimeError(self._error.value.decode(errors="replace"))
        if self._worker is not None and not self._worker.is_alive() and not self._stop.is_set():
            raise RuntimeError("CPU MuJoCo worker exited unexpectedly")

    def is_alive(self):
        """Report whether the owned worker still runs."""
        return self._worker is not None and self._worker.is_alive()

    def close(self):
        """Stop only this worker; never shut down a shared executor or context."""
        self._stop.set()
        if self._viewer_stop is not None:
            self._viewer_stop.set()
        if self._viewer_process is not None:
            self._viewer_process.join(timeout=3.0)
            if self._viewer_process.is_alive():
                self._viewer_process.terminate()
                self._viewer_process.join(timeout=3.0)
        if self._worker is None:
            for connection in (self._sync_parent, self._sync_child):
                if connection is not None:
                    connection.close()
            return
        self._worker.join(timeout=3.0)
        if self._worker.is_alive() and self._context:
            self._worker.terminate()
            self._worker.join(timeout=3.0)
        if self._worker.is_alive():
            raise RuntimeError("CPU MuJoCo thread did not stop within 3 seconds")
        for connection in (self._sync_parent, self._sync_child):
            if connection is not None:
                connection.close()
