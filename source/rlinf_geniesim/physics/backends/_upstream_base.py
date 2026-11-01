# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Physics backends that own an upstream geniesim_ros subprocess.

RosPhysicsBridge publishes joint targets and receives joint state, body poses,
clock and schema metadata. Lockstep commands advance a fixed number of physics
ticks. Batched upstream worlds are exposed through the vector backend contract."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ...child_lifetime import bind_to_server
from .base import (
    HealthStatus,
    SceneSchema,
    VecPhysicsBackend,
)
from ._ros_bridge import RosPhysicsBridge


# ---------------------------------------------------------------------------
# Engine teardown
# ---------------------------------------------------------------------------

# Escalation ladder used when tearing an engine down: (signal, grace seconds).
_TEARDOWN_LADDER = (
    (signal.SIGINT, 10.0),  # let the engine shut ROS down cleanly
    (signal.SIGTERM, 5.0),
    (signal.SIGKILL, 5.0),
)


def _process_group_gone(proc: subprocess.Popen, pgid: int) -> bool:
    """Is EVERY process in ``pgid`` gone?

    ``proc`` is the ``ros2 run`` launcher, not the engine — the engine is its
    child. ``proc.poll()`` first so an already-exited launcher is reaped:
    an unreaped zombie still counts as a live group member for ``killpg``,
    which would make this report "alive" forever.
    """
    proc.poll()
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Terminate the complete engine process group, escalating if necessary.

    The ros2 launcher may exit before its engine child, so launcher completion alone
    does not establish that the group's ROS publishers and resources are gone."""
    pgid = proc.pid
    for sig, grace in _TEARDOWN_LADDER:
        if _process_group_gone(proc, pgid):
            return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if _process_group_gone(proc, pgid):
                return
            time.sleep(0.05)
    if not _process_group_gone(proc, pgid):
        print(
            f"[engine-teardown] WARNING: process group {pgid} survived "
            f"SIGKILL — an orphaned engine may still be publishing on the "
            f"same ROS topics as its replacement",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Pure AABB math at module scope, with no I/O.
#
# get_body_aabbs() composes a STATIC per-body LOCAL box (published once by
# the engine on /rlinf/body_aabb_local) with the per-tick WORLD pose (already
# memoized by get_body_poses()) to get a world AABB. The only generally-
# correct way to AABB a ROTATED box is to transform all 8 corners and take
# the per-axis min/max — a rotated box's extreme points are always among
# its corners, never at the un-rotated center +/- half-extent.
# ---------------------------------------------------------------------------

# The 8 sign combinations for a box's corners, relative to its center.
_AABB_CORNER_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
    dtype=np.float64,
)  # (8, 3)


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Batched quaternion ``(..., 4)`` ``[w, x, y, z]`` -> rotation matrix
    ``(..., 3, 3)``.

    Same formula as ``eval/api_core_shim.py::_rotmat_from_quat_wxyz`` (kept
    in sync so pose->matrix conventions agree everywhere in the eval
    stack), vectorized over arbitrary leading dims via numpy broadcasting.
    """
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = w * w + x * x + y * y + z * z
    n = np.where(n < 1e-12, 1.0, n)  # degenerate quat -> identity-ish, never divide by 0
    s = 2.0 / n
    r = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    r[..., 0, 0] = 1 - s * (y * y + z * z)
    r[..., 0, 1] = s * (x * y - z * w)
    r[..., 0, 2] = s * (x * z + y * w)
    r[..., 1, 0] = s * (x * y + z * w)
    r[..., 1, 1] = 1 - s * (x * x + z * z)
    r[..., 1, 2] = s * (y * z - x * w)
    r[..., 2, 0] = s * (x * z - y * w)
    r[..., 2, 1] = s * (y * z + x * w)
    r[..., 2, 2] = 1 - s * (x * x + y * y)
    return r


def _world_aabb_from_local(poses: np.ndarray, local_aabb: np.ndarray) -> np.ndarray:
    """Vectorized world AABB from per-body LOCAL boxes + world poses.

    Args:
        poses: ``(N, B, 7)`` ``[x, y, z, qw, qx, qy, qz]`` — ``get_body_poses``'s
            output format.
        local_aabb: ``(B, 6)`` ``[cx, cy, cz, hx, hy, hz]`` — body-frame
            center + half-extents, broadcast over the N (env) axis.

    Returns:
        ``(N, B, 6)`` float32 ``[min_x, min_y, min_z, max_x, max_y, max_z]``.
    """
    poses = np.asarray(poses, dtype=np.float64)
    local_aabb = np.asarray(local_aabb, dtype=np.float64)
    pos = poses[..., 0:3]  # (N, B, 3)
    quat = poses[..., 3:7]  # (N, B, 4) wxyz
    center = local_aabb[..., 0:3]  # (B, 3)
    half = local_aabb[..., 3:6]  # (B, 3)
    corners_local = center[:, None, :] + _AABB_CORNER_SIGNS[None, :, :] * half[:, None, :]  # (B, 8, 3)

    rot = _quat_wxyz_to_rotmat(quat)  # (N, B, 3, 3)
    # world_corner[n,b,k,i] = sum_j rot[n,b,i,j] * corners_local[b,k,j]
    world_corners = np.einsum("nbij,bkj->nbki", rot, corners_local)  # (N,B,8,3)
    world_corners = world_corners + pos[:, :, None, :]

    mins = world_corners.min(axis=2)  # (N, B, 3)
    maxs = world_corners.max(axis=2)  # (N, B, 3)
    return np.concatenate([mins, maxs], axis=-1).astype(np.float32)  # (N,B,6)


class _UpstreamSubprocessBackend(VecPhysicsBackend):
    """Base for backends that delegate physics to an upstream
    geniesim_ros executable.

    Subclasses set the following class-level attributes:

      * ``UPSTREAM_PACKAGE``  — ROS 2 package, e.g. ``"genie_sim_engine"``
      * ``UPSTREAM_EXEC``     — executable name, e.g. ``"genie_sim_engine_newton.py"``
      * ``UPSTREAM_ENGINE``   — value for the ``physics_engine`` parameter
      * ``UPSTREAM_SOLVER``   — value for the ``physics_solver`` parameter
                                (None if the executable does not accept it)
      * ``BACKEND_NAME``      — string under which the class is registered

    Subclasses MAY override:

      * ``_extra_ros_params(self) -> dict[str, str]`` to supply additional
        ``--ros-args -p k:=v`` parameters.
      * ``_supports_batching`` (default ``True``).
    """

    # ---- subclass contract ------------------------------------------------
    UPSTREAM_PACKAGE: str = ""
    UPSTREAM_EXEC: str = ""
    UPSTREAM_ENGINE: str = ""
    UPSTREAM_SOLVER: Optional[str] = None
    BACKEND_NAME: str = ""

    # Flip in subclass if the upstream does not parallelize over worlds.
    _supports_batching: bool = True

    # reset_epoch fallback warning latch. Class-level (not per-instance) and
    # deliberately referenced via the base class name in reset() rather than
    # ``type(self)`` — the point is ONE warning for the whole process (an old
    # engine build without the feature shouldn't spam this on every reset(),
    # nor once per backend instance if more than one is ever constructed).
    _epoch_fallback_warned: bool = False

    # ---- ctor / lifecycle -------------------------------------------------

    def __init__(
        self,
        scene_yaml: str,
        robot_urdf: str,
        num_envs: int,
        physics_hz: float,
        device: str = "cuda:0",
        *,
        stage_manifest: str = "",
        headless: bool = True,
        ros_log_level: str = "warn",
        # First bring-up on a FRESH container compiles all warp/mjwarp CUDA
        # kernels (~3-5 min, empty ~/.cache/warp) — 60s timed that out. Real
        # crashes still fail fast via the subprocess poll inside _wait_ready.
        ready_timeout_s: float = 600.0,
        step_timeout_s: float = 5.0,
        control_hz: float = 0.0,
        **backend_kwargs: Any,
    ) -> None:
        super().__init__(
            scene_yaml=scene_yaml,
            robot_urdf=robot_urdf,
            num_envs=num_envs,
            physics_hz=physics_hz,
            device=device,
            **backend_kwargs,
        )
        if num_envs < 1:
            raise ValueError(f"{type(self).__name__}: num_envs must be >= 1 (got {num_envs})")
        # The engine batches N worlds on the GPU from one template.
        # ``num_envs`` reaches it as a ROS parameter, and this backend drives
        # it as a real vector env: step takes (N, A) and readings reshape to
        # (N, ...).
        self._stage_manifest = stage_manifest
        self._headless = bool(headless)
        self._ros_log_level = ros_log_level
        self._ready_timeout_s = float(ready_timeout_s)
        self._step_timeout_s = float(step_timeout_s)
        # Lockstep: 1 control step == round(physics_hz/control_hz) physics
        # ticks, command-driven (see common/loop.py). 0 = free-run (legacy).
        self._control_hz = float(control_hz)

        # Populated by start():
        self._proc: Optional[subprocess.Popen] = None
        self._pid: Optional[int] = None
        self._bridge: Optional[RosPhysicsBridge] = None
        self._action_dim: int = 0
        self._state_dim: int = 0
        self._joint_names: list[str] = []
        # Latch the joint layout only once the upstream's /joint_states count
        # is stable — the engine can publish a transient nominal layout before
        # the Newton model settles (see _wait_ready).
        self._latch_prev_n: int = -1
        self._latch_stable: int = 0
        self._latch_required: int = 3
        self._body_names: list[str] = []
        self._last_send_ms = self._last_wait_ms = 0.0
        self._last_step_ms: float = 0.0
        self._last_seen_sim_time: float = -1.0

    # ------------------------------------------------------------------
    # Subprocess management
    # ------------------------------------------------------------------

    def _build_argv(self) -> list[str]:
        """Construct ``ros2 run <pkg> <exec> <scene> [--gui] --ros-args -p ...``."""
        if not self.UPSTREAM_PACKAGE or not self.UPSTREAM_EXEC:
            raise RuntimeError(
                f"{type(self).__name__}: UPSTREAM_PACKAGE / UPSTREAM_EXEC " f"must be set on the subclass"
            )

        scene = str(Path(self.scene_yaml).resolve())
        argv: list[str] = [
            "ros2",
            "run",
            self.UPSTREAM_PACKAGE,
            self.UPSTREAM_EXEC,
            scene,
        ]

        # The engine parses ``--gui`` before ``--ros-args``. Pass it when the
        # caller wants a GUI, and also for the windowless ovrtx visualizer,
        # which the engine otherwise disables along with every other one.
        force_gui = (not self._headless) or (
            str(self._extra_ros_params().get("physics_engine_visualizer", "")).strip().lower() == "ovrtx"
        )
        if force_gui:
            argv.append("--gui")

        ros_params: dict[str, str] = {
            "physics_engine": self.UPSTREAM_ENGINE,
            "physics_hz": f"{self._physics_hz}",
            "headless": "true" if self._headless else "false",
            # Native multi-world replicate count — the engine replicates the
            # single-env template into N worlds; 1 selects a single world.
            "num_envs": f"{int(self._num_envs)}",
        }
        if self.UPSTREAM_SOLVER:
            ros_params["physics_solver"] = self.UPSTREAM_SOLVER
        if self._stage_manifest:
            # Upstream's EngineSession requires this — without it,
            # genie_sim_engine_newton.py raises "parameter
            # 'stage_manifest' is required" and exits.
            ros_params["stage_manifest"] = str(Path(self._stage_manifest).resolve())

        ros_params.update(self._step_mode_ros_params())
        ros_params.update(self._extra_ros_params())

        argv += ["--ros-args", "--log-level", self._ros_log_level]
        for k, v in ros_params.items():
            argv += ["-p", f"{k}:={v}"]

        return argv

    def _step_mode_ros_params(self) -> dict[str, str]:
        """Derive lockstep burst length from the control and physics rates.

        Loop pacing uses step_* ROS parameters; physics_* parameters are forwarded to
        the engine constructor. A lockstep burst advances fixed simulation time and
        waits for the next command without the realtime idle-timeout setting."""
        if self._control_hz <= 0:
            # No control rate declared: leave the engine on its default
            # (``realtime``). Sending ``lockstep`` with no K would be worse than
            # not asking for it.
            return {}
        ratio = self._physics_hz / self._control_hz
        k = max(1, round(ratio))
        if abs(ratio - k) > 1e-6:
            eff = self._physics_hz / k
            print(
                f"[{type(self).__name__}] WARN: physics_hz="
                f"{self._physics_hz:g} is NOT an integer multiple of "
                f"control_hz={self._control_hz:g} (ratio {ratio:.3f}). "
                f"Lockstep needs whole ticks -> using K={k} -> EFFECTIVE "
                f"control rate {eff:.2f} Hz ({k * 1000.0 / self._physics_hz:.1f} ms "
                f"sim-time per step), not the requested "
                f"{self._control_hz:g} Hz. Pick divisible rates, e.g. "
                f"physics_hz=120 for control_hz=30, or control_hz=25 at "
                f"physics_hz=100.",
                flush=True,
            )
        return {"step_mode": "lockstep", "step_lockstep_ticks": str(k)}

    def _extra_ros_params(self) -> dict[str, str]:
        """Subclass hook: extra ``-p key:=value`` pairs for the upstream."""
        return {}

    def _verify_environment(self) -> None:
        """Validate ROS availability and required scene/manifest files before launch."""
        if shutil.which("ros2") is None:
            raise RuntimeError(
                "'ros2' is not on PATH. Build the selected ROS workspace, then source "
                "source/rlinf_geniesim/scripts/rlinf_env.sh from the repository root."
            )
        if not Path(self.scene_yaml).is_file():
            raise FileNotFoundError(f"scene_yaml not found: {self.scene_yaml}")
        if not self._stage_manifest:
            raise ValueError(
                f"{type(self).__name__}: stage_manifest is required. "
                f"Supply a generated manifest JSON through task-init manifest.path "
                f"or the backend constructor's stage_manifest argument."
            )
        if not Path(self._stage_manifest).is_file():
            raise FileNotFoundError(f"stage_manifest not found: {self._stage_manifest}")

    def start(self) -> None:
        if self._started:
            return
        self._verify_environment()

        argv = self._build_argv()

        env = os.environ.copy()
        if self._device.startswith("cuda:"):
            env.setdefault("CUDA_VISIBLE_DEVICES", self._device.split(":", 1)[1])
        # Idle and lockstep policies are ROS parameters supplied by _build_argv.

        # Capture child stdout+stderr to a log file (see render bridge for
        # rationale — visibility + no PIPE back-pressure).
        backend_tag = getattr(self, "BACKEND_NAME", type(self).__name__)
        log_path = f"/tmp/rlinf_{backend_tag}_engine_{os.getpid()}.log"
        self._log_path = log_path
        # Rebuilds reuse the server PID; append to preserve earlier engine logs.
        self._log_fp = open(log_path, "a", buffering=1)
        self._log_fp.write(f"\n===== engine start (server pid {os.getpid()}) " f"=====\n")
        print(
            f"[{type(self).__name__}] child stdout/stderr -> {log_path}",
            flush=True,
        )

        argv, env = bind_to_server(argv, env)
        self._proc = subprocess.Popen(
            argv,
            env=env,
            stdout=self._log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # Put the child (and its grandchildren spawned by `ros2 run`)
            # in their own process group so close() can SIGINT the whole
            # group with a single os.killpg — otherwise SIGINT goes to
            # `ros2 run` only and the actual engine subprocess is left
            # orphaned + zombified.
            start_new_session=True,
        )
        self._pid = self._proc.pid

        # Bring up the rclpy bridge before waiting for ready, so we can
        # detect readiness via "first /joint_states received".
        self._bridge = RosPhysicsBridge(node_name=f"rlinf_{self.BACKEND_NAME}_bridge")
        self._bridge.start()

        self._wait_ready()
        self._started = True

    def _wait_ready(self) -> None:
        """Block until the upstream publishes its first ``/joint_states``.

        The first message gives us joint_names + joint count, which we
        latch as the layout used for ``send_action`` and ``state_dim``.
        """
        deadline = time.monotonic() + self._ready_timeout_s
        while time.monotonic() < deadline:
            # Bail early if the subprocess died.
            if self._proc is None or self._proc.poll() is not None:
                tail = ""
                log_path = getattr(self, "_log_path", "")
                if log_path:
                    try:
                        with open(log_path, "r") as f:
                            tail = f.read()
                    except Exception:  # noqa: BLE001
                        tail = ""
                self._cleanup_after_failure()
                raise RuntimeError(
                    f"{type(self).__name__}: upstream "
                    f"'{self.UPSTREAM_EXEC}' exited with code "
                    f"{self._proc.returncode if self._proc else '?'}.\n"
                    f"Log: {log_path}\n"
                    f"Tail of stdout/stderr:\n{tail[-2000:]}"
                )
            if self._bridge and self._bridge.has_state():
                pos = self._bridge.latest_joint_pos()
                # The upstream can publish a transient /joint_states (nominal
                # joint set) before the Newton model settles — e.g. mjwarp
                # drops loop-closure joints to equality constraints, shrinking
                # the count (50 -> 46). Latching the FIRST message would fix
                # action_dim/state_dim to the transient size and mismatch every
                # later step. Latch only after the count is stable across
                # ``_latch_required`` consecutive reads.
                if pos is not None:
                    n = int(pos.shape[0])
                    if n == self._latch_prev_n:
                        self._latch_stable += 1
                    else:
                        self._latch_prev_n = n
                        self._latch_stable = 1
                    if self._latch_stable >= self._latch_required:
                        # Lay out /joint_command with the real joint names
                        # from /joint_states, so the engine applies actions by
                        # name; generic joint_%d names resolve to nothing.
                        # With several worlds the names carry an ``env_<w>/``
                        # prefix that routes each world's commands.
                        names = self._bridge.latest_joint_names()
                        # The joint COUNT stabilises before the names carry
                        # world numbers, and latching the bare names would
                        # leave every /joint_command name unknown. Poll until
                        # the prefix appears.
                        if self._num_envs > 1 and not (
                            names is not None and len(names) == n and all(nm.startswith("env_") for nm in names)
                        ):
                            time.sleep(0.05)
                            continue
                        if names is not None and len(names) == n:
                            self._joint_names = list(names)
                        else:
                            self._joint_names = [f"joint_{i}" for i in range(n)]
                        self._bridge.set_joint_layout(self._joint_names)
                        # ``n`` is the FLAT joint count across ALL worlds; the
                        # per-env action/state dim is n // num_envs (replicate
                        # is uniform). num_envs==1 -> per_env == n (unchanged).
                        _ne = max(1, int(self._num_envs))
                        per_env = n // _ne if (n % _ne == 0) else n
                        self._action_dim = per_env
                        # state = (qpos, qvel) flattened per env.
                        self._state_dim = 2 * per_env
                        return
            time.sleep(0.05)

        self._cleanup_after_failure()
        raise TimeoutError(
            f"{type(self).__name__}: upstream did not publish " f"/joint_states within {self._ready_timeout_s}s"
        )

    def _cleanup_after_failure(self) -> None:
        """Best-effort cleanup if start() / _wait_ready() failed."""
        if self._bridge is not None:
            try:
                self._bridge.close()
            except Exception:  # noqa: BLE001
                pass
            self._bridge = None
        if self._proc is not None:
            # Same group teardown as close(): terminate()/kill() only reach
            # the `ros2 run` launcher and would leave the engine orphaned.
            try:
                _terminate_process_group(self._proc)
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        if not self._started and self._proc is None and self._bridge is None:
            return
        # Order: bridge first (so we stop spinning callbacks), then proc.
        bridge = self._bridge
        proc = self._proc
        self._bridge = None
        self._proc = None
        self._started = False

        # A restart brings a fresh bridge whose tf sequence starts at 0, so
        # stale caches would be served on a sequence collision. ``_body_names``
        # must go too, or a restarted backend looks like it already has names.
        self._body_names = []
        self._bp_seq = -1
        self._bp_cache = None
        self._ba_seq = -1
        self._ba_cache = None
        self._ba_table = None
        self._ba_table_key = None
        self._ba_warned_no_table = False
        self._ba_warned_missing = False
        self._ba_warned_no_names = False

        if bridge is not None:
            try:
                bridge.close()
            except Exception:  # noqa: BLE001
                pass

        if proc is None:
            return
        _terminate_process_group(proc)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def step(self, actions: np.ndarray) -> None:
        self._ensure_started()
        self._validate_actions(actions)
        if self._bridge is None:
            raise RuntimeError("ROS bridge not running")

        t0 = time.monotonic()

        # actions is (num_envs, action_dim); flatten WORLD-MAJOR (env_0's
        # joints, then env_1's, ...) to match the world-indexed /joint_command
        # layout latched from /joint_states. num_envs==1 -> (action_dim,).
        target = np.asarray(actions, dtype=np.float32).reshape(-1)
        # Latch the /joint_states ARRIVAL count BEFORE sending, then wait for it
        # to advance: the lockstep engine publishes exactly one message per
        # K-tick burst, so "seq advanced" == "THIS command's burst finished".
        # (The old sim_time-freshness check could be satisfied EARLY by
        # mid-burst /clock signals — steps then read state 1-3 ticks stale and
        # the render drove stale poses, silently breaking the lockstep contract.)
        seq0 = self._bridge.js_seq()
        self._bridge.send_action(target)
        _sent = time.monotonic()
        self._last_send_ms = (_sent - t0) * 1000

        deadline = time.monotonic() + self._step_timeout_s
        while time.monotonic() < deadline:
            if self._bridge.js_seq() > seq0:
                self._last_seen_sim_time = self._bridge.latest_sim_time()
                break
            time.sleep(0.0005)
        else:
            raise TimeoutError(
                f"{type(self).__name__}.step: no /joint_states arrival "
                f"within {self._step_timeout_s}s (seq stayed {seq0})"
            )

        self._last_wait_ms = (time.monotonic() - _sent) * 1000
        self._last_step_ms = (time.monotonic() - t0) * 1000.0

    def reset(
        self,
        env_ids: Optional[Sequence[int]] = None,
        seeds: Optional[Sequence[int]] = None,
        instances: Optional[Sequence[int]] = None,
    ) -> None:
        self._ensure_started()
        if self._bridge is None:
            return
        # Per-world reset: publish the target world indices on /reset_envs.
        # The engine restores those worlds to the build-time init state
        # (joint_q + zeroed joint_qd + init PD targets), without rebuilding
        # the model. ``env_ids=None`` means "reset all worlds", sent as an
        # EMPTY list (the engine-side "all" convention).
        ne = max(1, int(self._num_envs))
        if env_ids is None:
            ids: list[int] = []
        else:
            ids = [int(w) for w in env_ids if 0 <= int(w) < ne]
            if not ids:
                return
        # Capture BOTH credentials before send_reset — whichever wait branch
        # below fires needs its own baseline from strictly before the engine
        # could have started processing this call's reset.
        seq0 = self._bridge.js_seq()
        e0 = self._bridge.js_reset_epoch()
        self._bridge.send_reset(ids, seeds=seeds, instances=instances)
        # Wait until the engine has actually applied the reset.
        #
        # Waiting for the sequence to advance is not enough: the engine
        # publishes every tick while free-running, so any pre-reset publish
        # satisfies it and the caller reads pre-reset state.
        #
        # reset_epoch closes that race: the engine bumps it only after the
        # whole batch is applied and sends it with every joint state, both
        # written under one lock.
        import time as _time  # noqa: PLC0415

        deadline = _time.monotonic() + 2.0
        if e0 == -1:
            # Engine build without the reset_epoch feature (or no
            # /joint_states observed yet at all) — fall back to the old
            # seq-advance wait. See the class attribute above for why the
            # warning is gated at the class, not the instance.
            if not _UpstreamSubprocessBackend._epoch_fallback_warned:
                _UpstreamSubprocessBackend._epoch_fallback_warned = True
                print("[engine-wrapper] engine without reset_epoch — " "falling back to seq wait", flush=True)
            while self._bridge.js_seq() <= seq0:
                if _time.monotonic() > deadline:
                    print(
                        "[engine-wrapper] reset apply-confirm timed out (2s) — " "returning possibly-stale state",
                        flush=True,
                    )
                    break
                _time.sleep(0.004)
            return
        while self._bridge.js_reset_epoch() <= e0:
            if _time.monotonic() > deadline:
                print(
                    "[engine-wrapper] reset apply-confirm timed out (2s) — " "returning possibly-stale state",
                    flush=True,
                )
                break
            _time.sleep(0.004)
        # The epoch confirmation above covers the joint mirror only; body
        # poses arrive on another channel that can lag on a cold start. Both
        # channels share the engine clock and poses are published strictly
        # after joint states within a tick, so a pose batch stamped no earlier
        # than the confirming tick is post-reset. Shares the same 2-second
        # budget as the epoch wait.
        epoch_stamp = self._bridge.js_epoch_stamp()
        while self._bridge.tf_stamp() < epoch_stamp:
            if _time.monotonic() > deadline:
                print(
                    "[engine-wrapper] reset body-pose freshness wait timed "
                    "out (2s) — returning possibly-stale body_poses",
                    flush=True,
                )
                break
            _time.sleep(0.004)

    # ------------------------------------------------------------------
    # State readers
    # ------------------------------------------------------------------

    def get_joint_states(self) -> np.ndarray:
        self._ensure_started()
        if self._bridge is None:
            raise RuntimeError("bridge not running")
        ne = max(1, int(self._num_envs))
        per = int(self._action_dim)  # per-env joint count (= flat n // num_envs)
        pos = self._bridge.latest_joint_pos()
        vel = self._bridge.latest_joint_vel()
        if pos is None:
            # No data yet — return zeros to keep callers' shapes valid.
            return np.zeros((ne, per, 2), dtype=np.float32)
        if vel is None or vel.shape != pos.shape:
            vel = np.zeros_like(pos)
        # /joint_states is flat WORLD-MAJOR (env_0's joints, then env_1's, ...);
        # reshape to (num_envs, per_env). Fall back to single-env wrap if the
        # flat count doesn't divide evenly (shouldn't happen post-latch).
        if per > 0 and pos.shape[0] == ne * per:
            pos = pos.reshape(ne, per)
            vel = vel.reshape(ne, per)
        else:
            pos = pos[None, :]
            vel = vel[None, :]
        out = np.stack([pos, vel], axis=-1).astype(np.float32)  # (N, per, 2)
        return out

    def get_state(self) -> np.ndarray:
        joints = self.get_joint_states()  # (N, per, 2)
        return joints.reshape(joints.shape[0], -1)  # (N, 2*per)

    def _check_pose_finite(self, arr: np.ndarray) -> None:
        """Report transitions into and out of non-finite body poses.

        Invalid poses can make geometry disappear while the renderer still returns an
        image, so identify affected bodies before forwarding the observation."""
        ok = bool(np.isfinite(arr).all())
        was_bad = bool(getattr(self, "_pose_nonfinite", False))
        if ok:
            if was_bad:
                self._pose_nonfinite = False
                self._pose_nonfinite_msg = ""
                print("[engine-wrapper] body poses are finite again", flush=True)
            return
        self._pose_nonfinite = True
        bad = ~np.isfinite(arr).all(axis=-1)  # (N, B)
        n_bad = int(bad.sum())
        names = getattr(self, "_body_names", None) or []
        where = []
        for e, b in zip(*np.nonzero(bad)):
            nm = names[b] if b < len(names) else f"body[{b}]"
            where.append(f"env{e}:{nm}")
            if len(where) >= 6:
                break
        msg = (
            f"body poses contain non-finite values (NaN/Inf): "
            f"{n_bad}/{bad.size} poses are bad, first {', '.join(where)}" + (" …" if n_bad > len(where) else "")
        )
        self._pose_nonfinite_msg = msg
        if not was_bad:
            print(
                f"[engine-wrapper] error: {msg}. Physics has diverged: the "
                f"renderer will cull this geometry, the observation becomes "
                f"a uniform grey image that looks normal but holds nothing, "
                f"and reward and scoring are equally untrustworthy. Discard "
                f"data from this frame on.",
                file=sys.stderr,
                flush=True,
            )

    def get_body_poses(self) -> np.ndarray:
        self._ensure_started()
        if self._bridge is None:
            raise RuntimeError("bridge not running")
        # MEMOIZED on the TF arrival count: the eval checker shim calls this
        # PER POSE QUERY (dozens of times per step at 8 envs), and the per-call
        # name-grouping over ~500 frames cost ~30ms/step. Poses only change
        # when a new burst publishes, so one parse per tf_seq serves everyone.
        # Callers must treat the returned array as READ-ONLY (they do: handler
        # copies out, the render path copies per row).
        _seq = self._bridge.tf_seq()
        if _seq == getattr(self, "_bp_seq", -1) and getattr(self, "_bp_cache", None) is not None:
            return self._bp_cache
        ne = max(1, int(self._num_envs))
        poses, names = self._bridge.latest_body_poses()
        if poses is None:
            return np.zeros((ne, 0, 7), dtype=np.float32)
        if ne <= 1 or names is None:
            if names is not None:
                self._body_names = names
            self._bp_seq, self._bp_cache = _seq, poses[None, ...].astype(np.float32)  # (1, B, 7)
            self._check_pose_finite(self._bp_cache)
            return self._bp_cache
        # Multi-world: /tf_render frames are world-indexed 'env_<w>/...'.
        # Group rows by world (world-major publish order is preserved within
        # each group), then stack to (num_envs, B, 7).
        groups: list[list[int]] = [[] for _ in range(ne)]
        base_names: list[str] = []
        for i, nm in enumerate(names):
            if not nm.startswith("env_"):
                continue
            try:
                w = int(nm[4:].split("/", 1)[0])
            except (ValueError, IndexError):
                continue
            if 0 <= w < ne:
                groups[w].append(i)
                if w == 0:
                    base_names.append(nm.split("/", 1)[1])
        b = min((len(g) for g in groups), default=0)
        if b == 0:
            self._bp_seq, self._bp_cache = _seq, poses[None, ...].astype(np.float32)  # fallback
            self._check_pose_finite(self._bp_cache)
            return self._bp_cache
        out = np.zeros((ne, b, 7), dtype=np.float32)
        for w in range(ne):
            out[w] = poses[groups[w][:b]]
        self._body_names = base_names[:b]
        self._bp_seq, self._bp_cache = _seq, out
        self._check_pose_finite(out)
        return out  # (N, B, 7)

    def get_body_aabbs(self) -> np.ndarray:
        """Return world AABBs shaped (N, B, 6), aligned with the body schema.

        Compose the latched, name-aligned local boxes with current body poses and cache
        by tf_seq. Until the local table arrives, return zero boxes and warn once."""
        self._ensure_started()
        poses = self.get_body_poses()  # (N, B, 7); also refreshes _body_names
        ne, b = int(poses.shape[0]), int(poses.shape[1])
        if self._bridge is None:
            return np.zeros((ne, b, 6), dtype=np.float32)
        _seq = self._bridge.tf_seq()
        if _seq == getattr(self, "_ba_seq", -1) and getattr(self, "_ba_cache", None) is not None:
            return self._ba_cache
        if not self._body_names:
            # No body names have arrived yet: _wait_ready() waits for
            # /joint_states, not for the pose broadcast. Aligning against an
            # empty name list would build a (0, 6) table, and caching that
            # would poison every later call. Return zeros and warn once.
            out = np.zeros((ne, b, 6), dtype=np.float32)
            if not getattr(self, "_ba_warned_no_names", False):
                self._ba_warned_no_names = True
                print(
                    f"[{type(self).__name__}] WARN: get_body_aabbs: body "
                    f"names not populated yet (no /tf_render observed) — "
                    f"returning zero AABBs until poses/names arrive",
                    flush=True,
                )
            self._ba_seq, self._ba_cache = _seq, out
            return out
        local = self._aligned_local_aabb_table(b)
        if local is None:
            out = np.zeros((ne, b, 6), dtype=np.float32)
            if not getattr(self, "_ba_warned_no_table", False):
                self._ba_warned_no_table = True
                print(
                    f"[{type(self).__name__}] WARN: get_body_aabbs: "
                    f"/rlinf/body_aabb_local hasn't arrived yet — returning "
                    f"zero AABBs (engine may not publish it, e.g. a "
                    f"non-mjwarp solver)",
                    flush=True,
                )
        else:
            out = _world_aabb_from_local(poses, local)
        self._ba_seq, self._ba_cache = _seq, out
        return out

    def _aligned_local_aabb_table(self, b: int) -> Optional[np.ndarray]:
        """Align latched local AABBs with the current per-env body names.

        Return ``None`` until the table arrives. Cache by the complete name tuple
        so late or changed body names cannot reuse a stale alignment. Empty name
        sets are never cached.
        """
        names_key = tuple(self._body_names)
        cached = getattr(self, "_ba_table", None)
        cached_key = getattr(self, "_ba_table_key", None)
        if cached is not None and names_key and cached_key == names_key:
            return cached
        if self._bridge is None:
            return None
        names, table = self._bridge.latest_body_aabb_local()
        if not names or table is None or table.size == 0:
            return None
        # Normalize BOTH sides to a bare label (strip any leading '/')
        # before matching: _body_names carries a leading '/' at N==1 (raw
        # /tf_render prim paths, see get_body_poses) but not at N>1
        # (already stripped by the world-grouping there); the wire
        # contract's published names are always bare. Normalizing makes
        # the lookup robust to either dialect instead of depending on
        # num_envs.
        by_name = {str(nm).lstrip("/"): row for nm, row in zip(names, table)}
        body_names = list(self._body_names)
        aligned = np.zeros((b, 6), dtype=np.float32)
        missing: list[str] = []
        for i in range(b):
            bn = body_names[i] if i < len(body_names) else None
            row = by_name.get(str(bn).lstrip("/")) if bn is not None else None
            if row is None:
                missing.append(bn if bn is not None else f"<index {i}>")
                continue
            aligned[i] = row
        if missing and not getattr(self, "_ba_warned_missing", False):
            self._ba_warned_missing = True
            print(
                f"[{type(self).__name__}] WARN: get_body_aabbs: "
                f"{len(missing)}/{b} body name(s) missing from "
                f"/rlinf/body_aabb_local (zero local box): {missing[:5]}"
                + (f" +{len(missing) - 5} more" if len(missing) > 5 else ""),
                flush=True,
            )
        # Only persist the cache for a non-empty name key (defense in
        # depth — see docstring; the real gate is the caller-side guard in
        # get_body_aabbs). An uncached result is simply recomputed next
        # call, which is cheap and correct; caching a bogus empty-name
        # alignment is the one thing that must never happen.
        if names_key:
            self._ba_table = aligned
            self._ba_table_key = names_key
        return aligned

    def get_object_poses(self) -> np.ndarray:
        # Object poses arrive in /tf_render alongside body
        # poses without a distinguishing namespace, so we don't separate
        # them yet. Return empty until the upstream tags them.
        self._ensure_started()
        return np.zeros((1, 0, 7), dtype=np.float32)

    def get_camera_poses(self) -> np.ndarray:
        # No upstream camera FK service; the renderer can fall back to
        # computing camera poses from CameraSpec.link + body_poses.
        self._ensure_started()
        return np.zeros((1, 0, 7), dtype=np.float32)

    def get_sim_time(self) -> np.ndarray:
        self._ensure_started()
        if self._bridge is None:
            return np.zeros((1,), dtype=np.float64)
        return np.array([self._bridge.latest_sim_time()], dtype=np.float64)

    def _strip_to_per_env(self, names: list[str]) -> list[str]:
        """Reduce a FLAT world-major name list to the PER-ENV set.

        ``self._joint_names`` (and a raw ``/tf_render`` frame list) are
        flat across worlds — ``N*per_env`` entries, world-major, and for
        ``num_envs>1`` each is world-indexed ``env_<w>/<name>`` . The SceneSchema alignment guarantee (see
        ``base.py::SceneSchema``) requires ``joint_names[j]`` /
        ``body_link_names[i]`` to align with the PER-ENV axis of
        ``get_joint_states()`` (N, per, 2) / ``get_body_poses()``
        (N, B, 7). So take world 0's slice and strip the ``env_<w>/``
        prefix. ``num_envs==1`` returns the names unchanged.
        """
        ne = max(1, int(self._num_envs))
        if not names:
            return list(names)
        per = (len(names) // ne) if (ne > 1 and len(names) % ne == 0) else len(names)
        out: list[str] = []
        for nm in names[:per]:
            if ne > 1 and nm.startswith("env_") and "/" in nm:
                out.append(nm.split("/", 1)[1])
            else:
                out.append(nm)
        return out

    def get_static_prop_aabbs(self) -> dict[str, np.ndarray]:
        """Return latched world AABBs keyed by checker-facing static prop paths.

        Static props need no live pose composition. Return an empty mapping when the
        engine has not published the table."""
        self._ensure_started()
        if self._bridge is None:
            return {}
        table = self._bridge.latest_static_prop_aabb()
        return dict(table) if table else {}

    def get_scene_schema(self) -> SceneSchema:
        """Build a per-environment schema from the observed joint and body names.

        Strip world prefixes so each name indexes the corresponding per-env axis of
        the batched getters. body_paths use synthesized /World/body_<i> identifiers."""
        self._ensure_started()
        # ``_body_names`` (set by get_body_poses) is already per-env +
        # stripped; the fallback /tf_render list is flat, so reduce it.
        body_names = list(self._body_names)
        if not body_names:
            _poses, names = self._bridge.latest_body_poses() if self._bridge else (None, None)
            if names:
                body_names = self._strip_to_per_env(list(names))
        joint_names = self._strip_to_per_env(list(self._joint_names))
        return SceneSchema(
            body_paths=[f"/World/body_{i}" for i in range(len(body_names))],
            body_kinds=["robot_link"] * len(body_names),
            body_link_names=body_names,
            joint_names=joint_names,
            object_names=[],
            cloth_meshes=[],
            static_props=sorted(self.get_static_prop_aabbs()),
        )

    def get_scene_joint_states(self) -> list[dict]:
        """Return one ordered joint-position mapping per environment.

        Scene joints include door hinges and drawer slides. Names are bare compiled
        joint labels, routed by their ``env_<w>/`` prefix. The engine does not publish
        their owning object prims, so the checker shim cannot attribute these labels
        to objects using its prim-path lookup. Robot joint mappings do not fill that
        gap. Return empty mappings until scene state arrives or if no scene joints
        exist; lookup failures are handled best-effort.
        """
        self._ensure_started()
        ne = max(1, int(self._num_envs))
        out: list[dict] = [dict() for _ in range(ne)]
        if self._bridge is None:
            return out
        names, pos = self._bridge.latest_scene_joint_states()
        if names is None or pos is None:
            return out
        for nm, p in zip(names, pos):
            nm = str(nm)
            if ne > 1 and nm.startswith("env_") and "/" in nm:
                w_str, rest = nm[4:].split("/", 1)
                try:
                    w = int(w_str)
                except ValueError:
                    continue
                if 0 <= w < ne:
                    out[w][rest] = float(p)
            else:
                # num_envs==1 (bare label) — or an un-prefixed label at N>1,
                # which we can't attribute to a world, so it lands in env 0.
                out[0][nm] = float(p)
        return out

    def get_body_link_names(self) -> list[str]:
        """Live snapshot of upstream-published body names.

        Unlike the SceneSchema (latched once at startup) this reflects
        whatever upstream is currently broadcasting on ``/tf_render``.
        Returns an empty list if no ``/tf_render`` has arrived yet.
        """
        if self._body_names:
            return list(self._body_names)
        if self._bridge is None:
            return []
        _, names = self._bridge.latest_body_poses()
        return list(names) if names else []

    def wait_for_body_link_names(
        self, timeout_s: float = 10.0, stable_reads: int = 3, poll_s: float = 0.05
    ) -> list[str]:
        """Wait for a nonempty body list that agrees across ``stable_reads``.

        Startup waits for joint states, so body TF may arrive later. Each TF
        message carries the complete body set. Return an empty list on timeout so
        callers can fall back to their configured buffer-size estimates.
        """
        deadline = time.monotonic() + float(timeout_s)
        prev: list[str] = []
        stable = 0
        while time.monotonic() < deadline:
            names = self.get_body_link_names()
            if names:
                if names == prev:
                    stable += 1
                    if stable >= int(stable_reads):
                        return list(names)
                else:
                    prev = list(names)
                    stable = 1
            time.sleep(float(poll_s))
        return []

    def set_qpos(self, env_id: int, qpos: np.ndarray) -> None:
        # Direct joint teleportation requires an upstream service.
        raise NotImplementedError(f"{type(self).__name__}.set_qpos requires an upstream " f"set_qpos service")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def supports_batching(self) -> bool:
        return self._supports_batching

    @property
    def action_dim(self) -> int:
        if self._action_dim == 0:
            raise RuntimeError(
                f"{type(self).__name__}.action_dim is only known after "
                f"start() observes the first /joint_states message"
            )
        return self._action_dim

    @property
    def state_dim(self) -> int:
        if self._state_dim == 0:
            raise RuntimeError(f"{type(self).__name__}.state_dim is only known after start()")
        return self._state_dim

    @property
    def backend_name(self) -> str:
        if not self.BACKEND_NAME:
            raise RuntimeError(f"{type(self).__name__}: BACKEND_NAME class attribute " f"must be set")
        return self.BACKEND_NAME

    # ------------------------------------------------------------------
    # Health / metrics
    # ------------------------------------------------------------------

    def health_check(self) -> HealthStatus:
        alive = self._started and self._proc is not None and self._proc.poll() is None and self._bridge is not None
        if not alive:
            err = "subprocess or bridge not running"
        else:
            # A diverged pose is "the process lives but its data is useless",
            # which the health snapshot has to show; otherwise the only trace
            # is one stderr line in the engine log.
            err = getattr(self, "_pose_nonfinite_msg", "") or ""
        return HealthStatus(
            alive=alive,
            pid=self._pid,
            last_step_ms=self._last_step_ms,
            cuda_oom=False,
            error_msg=err,
        )

    def get_metrics(self) -> dict[str, float]:
        return {
            "last_step_ms": float(self._last_step_ms),
            "last_sim_time": float(self._last_seen_sim_time),
        }


__all__ = ["_UpstreamSubprocessBackend"]
