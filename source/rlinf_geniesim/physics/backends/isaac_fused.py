# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""PhysX physics and tiled rendering in one Isaac Sim subprocess.

One FusedIsaacBackend serves both vector interfaces. start owns one subprocess;
close is idempotent because the handler may invoke it through both interfaces.
FusedStateShm exchanges lockstep commands/state, while TiledFrameShm carries RGB.
update_poses is a no-op because physics and rendering already share the stage."""

from __future__ import annotations

import copy
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from . import register_engine
from .base import SceneSchema, VecPhysicsBackend
from ._upstream_base import _world_aabb_from_local
from ..fused_shm import CMD_RESET, CMD_STEP, FusedStateShm, STATUS_OK
from ...child_lifetime import bind_to_server
from ...comm.shm.sem import SemPair
from ...render.bridges._hotswap_protocol import clear_stale_control_files
from ...render.bridges._render_randomization import RenderRandomizationMixin
from ...render.bridges.base import CameraSpec, RenderResult, VecRenderBackend
from ...render.tiled_shm import TiledFrameShm


def _instance_generation():
    """Return a process-wide generation for resource names across backend rebuilds.

    The PID is shared by overlapping backend generations, so it cannot identify
    ready files, schemas, semaphores, or shared-memory segments by itself."""
    n = _INSTANCE_GEN[0]
    _INSTANCE_GEN[0] = n + 1
    return n


_INSTANCE_GEN = [0]


@register_engine("isaac_fused")
class FusedIsaacBackend(RenderRandomizationMixin, VecPhysicsBackend, VecRenderBackend):
    """One subprocess implementing both physics and rendering interfaces.

    The two base initializers have distinct signatures and are not cooperative;
    this constructor calls each explicitly."""

    BACKEND_NAME = "isaac_fused"
    # One Kit process holds all environments and their tiled cameras.
    SUPPORTS_BATCHING = True
    # Observation cameras use the same tiled RGB layout as isaac_tiled.
    OUTPUT_CHANNELS = ["rgb"]

    def __init__(
        self,
        scene_yaml: str,
        robot_urdf: str,
        num_envs: int,
        physics_hz: float,
        device: str = "cuda:0",
        *,
        # ---- VecRenderBackend's required constructor args ----
        cameras: Optional[Sequence[CameraSpec]] = None,
        render_hz: float = 30.0,
        # ---- Scene / stage wiring (mirrors isaac_tiled.py) ----
        stage_manifest: str = "",
        env_spacing: float = 15.0,
        headless: bool = True,
        # Each step advances round(physics_hz/control_hz) physics ticks.
        # Zero selects one tick; unlike ROS backends, fused has no free-running mode.
        control_hz: float = 0.0,
        # ---- Subprocess launch ----
        script_path: str = "",
        python_exec: str = "",
        state_shm_name: str = "",
        req_sem_name: str = "",
        resp_sem_name: str = "",
        tiled_shm_name: str = "",
        schema_out: str = "",
        ready_file: str = "",
        # Startup includes physics stage construction and RTX shader compilation.
        ready_timeout_s: float = float(os.environ.get("RLINF_FUSED_READY_TIMEOUT", "180")),
        # The round trip includes both physics and rendering.
        step_timeout_s: float = 30.0,
        render_randomization: Optional[dict] = None,
        randomize_ack_timeout_s: float = 30.0,
        reset_settle_time_s: float = 1.0,
        **backend_kwargs: Any,
    ) -> None:
        # See the class docstring: call both parent __init__s directly rather
        # than chaining through super() (they are not cooperative).
        VecPhysicsBackend.__init__(
            self,
            scene_yaml=scene_yaml,
            robot_urdf=robot_urdf,
            num_envs=num_envs,
            physics_hz=physics_hz,
            device=device,
            **backend_kwargs,
        )
        VecRenderBackend.__init__(
            self, num_envs=num_envs, cameras=list(cameras or []), render_hz=render_hz, **backend_kwargs
        )

        if num_envs < 1:
            raise ValueError(f"{type(self).__name__}: num_envs must be >= 1 (got {num_envs})")

        # Design doc §5 ("action convention"): actions[:, j] is joint j's position
        # target, 1 action per joint, in ``get_scene_schema()['joint_names']``
        # order (which itself is ``Articulation.dof_names`` verbatim on the
        # subprocess side — no reordering on either end). So action_dim is
        # NOT an independent knob here; making it one would let the two sides
        # disagree on the very alignment the SceneSchema contract promises.
        #
        # These are NOT constructor args: (num_joints, num_bodies,
        # num_objects) are only knowable AFTER the subprocess has actually
        # built the Isaac stage (URDF/USD import, articulation discovery,
        # object spawn from the manifest) — asking the CALLER to already
        # know them before start() would mean guessing, and a wrong guess
        # would size the SHM once, wrong, with no way to detect it. So these
        # start at the "not known yet" sentinel
        # (0 / empty) and are filled in by ``_attach_state_shm`` (called from
        # ``start()``) once the subprocess's own schema JSON reports them —
        # exactly the same "only valid after start()" contract the ABC
        # documents and the ROS-backed engines already live under (see
        # ``_upstream_base.py``'s ``action_dim``/``state_dim`` properties).
        self._num_joints = 0
        self._num_bodies = 0
        self._num_objects = 0
        self._action_dim = 0
        self._state_dim = 0

        self._stage_manifest = stage_manifest
        # Settle after a reset, run by the node inside the RESET command and
        # before the reset frame is rendered (sim_server then disables the
        # handler's own settle for fused, so it happens exactly once).
        self._reset_settle_time_s = max(0.0, float(reset_settle_time_s))
        # Render-side generalization (lights, material, camera pose), the same
        # machinery as isaac_tiled (RenderRandomizationMixin). The node polls
        # randomize_cmd.json in its manifest's directory during each reset.
        self._render_randomization: dict = dict(render_randomization or {})
        self._randomize_ack_timeout_s = float(randomize_ack_timeout_s)
        self._randomize_generation = 0
        self._material_candidates: Optional[dict] = None
        self._last_render_plans: dict = {}
        self._reload_control_dir: Optional[str] = str(Path(stage_manifest).resolve().parent) if stage_manifest else None
        #: Which benchmark instance this stage was BUILT for, read from the
        #: manifest in start(). ``None`` = unknown. Used by reset() to tell a
        #: real instance SWITCH apart from a request for the instance already
        #: loaded -- the latter is a no-op, not a capability gap.
        self._build_instance: Optional[int] = None
        self._env_spacing = float(env_spacing)
        self._headless = bool(headless)
        self._control_hz = float(control_hz) if control_hz > 0 else float(physics_hz)
        self._ready_timeout_s = float(ready_timeout_s)
        self._step_timeout_s = float(step_timeout_s)

        self._script_path = script_path or self._default_fused_script()
        self._python_exec = python_exec or os.environ.get("RLINF_ISAAC_PYTHON", "") or "python3"

        # Unique per instance so concurrent runs on one machine never collide
        # on a global SHM/semaphore name (same reasoning as isaac_tiled.py's
        # ``_tiled_shm_name`` default).
        #
        # The pid alone is NOT unique per instance: a reload builds a NEW
        # backend in the SAME server process, so the pid — and therefore
        # every derived name — is byte-identical to the generation it is
        # replacing. Combined with ``close()`` not removing the schema/ready
        # files, the rebuilt generation would immediately find its
        # PREDECESSOR's ready-file, skip the wait, and read the
        # PREDECESSOR's schema — attaching with the previous task's dims
        # against a segment that no longer exists, which surfaces as "the
        # subprocess's own segment disagrees with the schema it reported"
        # even though the subprocess itself did nothing wrong.
        #
        # So: keep the pid (it is what separates two CONCURRENT servers) and
        # add a monotonically increasing generation. Same process now gets a
        # fresh name per rebuild; a different process still gets its own.
        pid = os.getpid()
        gen = _instance_generation()
        tag = f"{pid}_{gen}"
        #: Kept as an attribute (not just a local) because start() names the
        #: child's log with it -- every artifact this instance owns must share
        #: the same generation-unique identity, or a rebuild collides with its
        #: predecessor on whichever one was left pid-only.
        self._tag = tag
        self._state_shm_name = state_shm_name or f"rlinf_fused_state_{tag}"
        self._tiled_shm_name = tiled_shm_name or f"rlinf_fused_tiled_{tag}"
        self._req_sem_name = req_sem_name or f"/rlinf_fused_req_{tag}"
        self._resp_sem_name = resp_sem_name or f"/rlinf_fused_resp_{tag}"
        self._schema_out = schema_out or f"/tmp/rlinf_fused_schema_{tag}.json"
        # Dedicated readiness sentinel, separate from schema_out's mere
        # existence: the server writes schema JSON, THEN touches this file —
        # a zero-byte "touch" cannot be observed half-written the way a JSON
        # file theoretically could be, so this is the one thing this side
        # polls for, and the schema (read only after this appears) is then
        # trusted to be complete without its own retry-on-parse-error dance.
        #
        # The file's CONTENT is the subprocess's pid (``isaacsim_fused_sim``
        # writes ``os.getpid()``); ``_wait_ready`` checks that this pid's
        # process GROUP matches the one this side launched
        # (``os.getpgid(who) == self._proc.pid``), NOT that the pid is
        # literally equal to ``self._proc.pid``. A direct pid compare fails
        # whenever ``RLINF_ISAAC_PYTHON`` points at a wrapper such as
        # ``/isaac-sim/python.sh`` that invokes the real interpreter as a
        # plain call instead of ``exec``ing it: ``self._proc.pid`` is then
        # the WRAPPER's pid, while the subprocess's own pid (what it writes
        # here) is a different, bash-forked child -- the two never match,
        # but both share the same process-group id (see ``start_new_session``
        # below and ``_wait_ready``'s docstring for why). This is the
        # handshake that makes "exists" mean "MY subprocess is up" rather
        # than "some subprocess, possibly a dead one, once got this far".
        self._ready_file = ready_file or f"/tmp/rlinf_fused_ready_{tag}"

        # Populated by start() / close():
        self._state_shm: Optional[FusedStateShm] = None
        self._sem: Optional[SemPair] = None
        self._tiled_shm: Optional[TiledFrameShm] = None
        self._proc: Optional[subprocess.Popen] = None
        self._pid: Optional[int] = None
        self._log_path: str = ""
        self._log_fp = None
        self._schema_cache: Optional[SceneSchema] = None
        self._last_tiled_frame_id: int = 0
        self._last_send_ms = self._last_wait_ms = 0.0
        self._last_step_ms: float = 0.0
        self._frame_index: int = 0
        self._last_barrier_dropped: bool = True
        self._frame_drop_warned: bool = False
        # Idempotency guard for close() — see class docstring: the handler
        # calls close() twice on this one object (once per role).
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    @staticmethod
    def _default_fused_script() -> str:
        """Resolve the fused node script, honoring RLINF_FUSED_SIM_SCRIPT.

        File validation is deferred to start() so construction needs no simulator.
        """
        env = os.environ.get("RLINF_FUSED_SIM_SCRIPT")
        if env:
            return env
        rel = Path("geniesim_ros") / "src" / "ros_ws" / "src" / "genie_sim_render" / "scripts" / "isaacsim_fused_sim.py"
        here = Path(__file__).resolve()
        for parent in here.parents:
            cand = parent / rel
            if cand.is_file():
                return str(cand)
        return str(here.parents[2] / rel)

    def _build_argv(self) -> list[str]:
        """Build the fused node command and its resource ownership arguments.

        The backend chooses resource names and creates the semaphore pair.
        The node creates dimension-dependent state SHM and writes its schema
        before signaling readiness; the backend then attaches using that schema.
        """
        if not self._stage_manifest:
            raise ValueError(f"{type(self).__name__}: stage_manifest is required")
        resolved_manifest = Path(self._stage_manifest).resolve()
        argv = [
            self._python_exec,
            self._script_path,
            "--manifest",
            str(resolved_manifest),
            "--shm-name",
            self._tiled_shm_name,
            "--render-hz",
            f"{self._render_hz}",
            "--num-envs",
            f"{self._num_envs}",
            "--env-spacing",
            f"{self._env_spacing}",
            "--headless",
            "1" if self._headless else "0",
            "--state-shm",
            self._state_shm_name,
            "--req-sem",
            self._req_sem_name,
            "--resp-sem",
            self._resp_sem_name,
            "--schema-out",
            self._schema_out,
            "--ready-file",
            self._ready_file,
            "--physics-hz",
            f"{self._physics_hz}",
            "--control-hz",
            f"{self._control_hz}",
            "--reset-settle-time-s",
            f"{self._reset_settle_time_s}",
        ]
        # Per-joint-class PD gains. Without this the articulation runs on the
        # robot USD's own DriveAPI values, which are NOT what this repo means
        # by "the robot's gains" -- every tuning result, and the newton
        # backend's behaviour, come from this yaml. With the asset's own
        # gains a commanded +0.3 rad on an arm joint reached
        # only +0.15 rad in 20 control steps, i.e. the arm lagged the headset
        # badly during teleop. Same resolution the newton backend uses
        # (newton_mjwarp._rlinf_physics_params_file), so both are driven from
        # one file.
        _pp = self._rlinf_physics_params_file()
        if _pp:
            argv += ["--physics-params-file", _pp]
        return argv

    @staticmethod
    def _rlinf_physics_params_file() -> str:
        """Absolute path of the installed ``physics_params_rlinf.yaml``.

        Empty string when not found, so the node falls back to the USD gains
        with a log line rather than failing to start -- that log line is the
        only sign grasping reverted to untuned behaviour.
        """
        try:
            from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415

            path = os.path.join(
                get_package_share_directory("genie_sim_engine_core"), "config", "physics_params_rlinf.yaml"
            )
            return path if os.path.isfile(path) else ""
        except Exception:  # noqa: BLE001
            return ""

    def _verify_environment(self) -> None:
        if not self._stage_manifest:
            raise ValueError(f"{type(self).__name__}: stage_manifest is required")
        if not Path(self._script_path).is_file():
            raise FileNotFoundError(
                f"fused sim script not found: {self._script_path} " f"(set RLINF_FUSED_SIM_SCRIPT to override)"
            )
        if shutil.which(self._python_exec) is None and not Path(self._python_exec).is_file():
            raise RuntimeError(
                f"python interpreter '{self._python_exec}' not found "
                f"(set RLINF_ISAAC_PYTHON to the isaacsim-capable python)"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._verify_environment()
        if self._reload_control_dir:
            # Both randomize generation counters start at 0, so a cmd/ack left
            # behind by a previous process would be taken for this one's.
            clear_stale_control_files(self._reload_control_dir)

        # FusedStateShm's dimensions (action_dim, num_joints, num_bodies,
        # num_objects) are NOT knowable on this side yet — they depend on the
        # Isaac stage the subprocess is about to build (URDF joint count,
        # manifest object count), which does not exist until the subprocess
        # runs. So THIS side cannot create that segment; it can only create
        # what needs no such knowledge: the SemPair. The subprocess creates
        # FusedStateShm itself once it knows its own dimensions (see
        # ``_attach_state_shm`` below for the client-side half of this).
        #
        # SemPair ownership: comm/shm/sem.py's close() docstring says
        # "creator (server) unlinks" but the code only ever checks
        # ``self._created`` (set from the ``create`` kwarg passed in) — the
        # "(server)" is descriptive of the USUAL caller, not a rule the code
        # enforces. This side passes create=True, so this side unlinks in
        # close() below.
        self._sem = SemPair(self._req_sem_name, self._resp_sem_name, create=True)

        try:
            argv = self._build_argv()
            # Generation-unique, like every other artifact of this instance
            # (see the constructor). A pid-only name made every rebuild in the
            # same server truncate its predecessor's log via ``open(..., "w")``
            # -- so the only record of a FAILED generation, which is the one an
            # operator actually needs, was destroyed by the retry that followed
            # it. ``self._tag`` is the same ``<pid>_<generation>`` suffix the
            # SHM/schema/ready names use.
            log_path = f"/tmp/rlinf_{self.BACKEND_NAME}_{self._tag}.log"
            self._log_path = log_path
            self._log_fp = open(log_path, "w", buffering=1)
            print(f"[{type(self).__name__}] child stdout/stderr -> {log_path}", flush=True)
            argv, env = bind_to_server(argv, os.environ.copy())
            self._proc = subprocess.Popen(
                argv,
                env=env,
                stdout=self._log_fp,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,  # killpg in close() reaches the
                # whole process group, not just
                # this direct child
            )
            self._pid = self._proc.pid

            self._wait_ready()  # blocks until --ready-file appears
            self._attach_state_shm()  # reads schema, derives dims, attaches
            self._attach_tiled_shm()  # blocks until TiledFrameShm appears
            self._started = True
        except Exception:
            # Partial bring-up: tear everything down rather than leak a Kit
            # process or a half-created semaphore pair that would make the
            # NEXT start() attempt fail with a confusing "already exists".
            self.close()
            raise

    def _wait_ready(self) -> None:
        """Wait for this launch's ready file before reading its schema and SHM.

        The node writes its PID after creating the stage, state SHM and schema.
        Accept only a PID in the launched process group: an Isaac Python wrapper
        may fork the interpreter, so the writer need not have the wrapper's PID.
        A missing, incomplete or foreign ready file remains pending until the
        child exits or the readiness timeout expires."""
        deadline = time.monotonic() + self._ready_timeout_s
        ready_path = Path(self._ready_file)
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"{type(self).__name__}: fused sim subprocess exited "
                    f"with code {self._proc.returncode} before touching "
                    f"{ready_path}.\nLog tail:\n{self._read_log_tail()}"
                )
            # Read-then-compare rather than a bare exists(): see docstring.
            # A file that is foreign (predecessor generation, or a manual
            # leftover) must NOT satisfy the wait — it would license a schema
            # read that describes a stage this subprocess never built.
            try:
                who = ready_path.read_text().strip()
            except (FileNotFoundError, OSError):
                who = ""
            if who:
                # Process-GROUP membership, not pid equality -- see the
                # docstring above for why a direct pid compare is wrong
                # (RLINF_ISAAC_PYTHON is commonly a non-exec wrapper). Every
                # failure mode here means "not (yet) confirmed to be ours",
                # so all of them fall through to the poll-and-retry below
                # rather than raising out of this loop; a REAL dead child is
                # already caught above via ``self._proc.poll()`` with a more
                # specific error, so none of these except-branches need to
                # duplicate that check.
                matches = False
                if self._proc is not None:
                    try:
                        matches = os.getpgid(int(who)) == self._proc.pid
                    except ValueError:
                        # Content isn't a valid pid yet -- a torn/partial
                        # write caught mid-flight; keep polling.
                        pass
                    except ProcessLookupError:
                        # ``who`` names a pid that no longer exists: a
                        # stale ready-file (predecessor generation, or a
                        # manual leftover), never our own child (that case
                        # is the poll() check above).
                        pass
                    except PermissionError:
                        # Can't inspect this pid's process group (e.g. it
                        # belongs to another user) -- conservatively treat
                        # as "not ours" rather than raising out of the wait.
                        pass
                if matches:
                    print(f"[{type(self).__name__}] ready file present: " f"{ready_path} (child pid {who})", flush=True)
                    return
                # Present but written by someone else. Loop rather than fail:
                # the child is alive and may still be building its stage, and
                # a hard error here would be indistinguishable from a real
                # startup failure to the operator.
            time.sleep(0.2)
        raise RuntimeError(
            f"{type(self).__name__}: ready file '{self._ready_file}' did "
            f"not appear within {self._ready_timeout_s:.0f}s.\n"
            f"Log tail:\n{self._read_log_tail()}"
        )

    def _attach_state_shm(self) -> None:
        """Read the schema JSON, derive dimensions from it, then ATTACH
        (``create=False``) to the FusedStateShm segment the subprocess
        already created.

        Called only after ``_wait_ready`` confirms the ready-file exists, so
        the schema file is guaranteed fully written and the state SHM is
        guaranteed to already exist under ``self._state_shm_name`` — this
        method does not itself need a long retry/timeout loop the way
        ``_wait_ready`` and ``_attach_tiled_shm`` do; a short one only guards
        against filesystem-visibility jitter, not a real race.

        This is where ``FusedStateShm._validate_header`` becomes genuinely
        load-bearing (per this backend's corrected design): if the dims we
        derive from the schema JSON disagree with what the subprocess
        actually sized the segment with, THAT validation is what catches it,
        loudly, instead of every subsequent get_* call silently reading the
        wrong shape.
        """
        schema_path = Path(self._schema_out)
        deadline = time.monotonic() + 10.0
        last_err: Optional[BaseException] = None
        js: Optional[SceneSchema] = None
        while time.monotonic() < deadline:
            try:
                js = json.loads(schema_path.read_text())
                break
            except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                last_err = e
                time.sleep(0.05)
        if js is None:
            raise RuntimeError(
                f"{type(self).__name__}: schema file '{schema_path}' was "
                f"not readable even after --ready-file appeared. "
                f"Last error: {last_err!r}"
            )
        self._schema_cache = js
        print(
            f"[{type(self).__name__}] schema ready: "
            f"{len(js.get('joint_names', []))} joints, "
            f"{len(js.get('body_paths', []))} bodies, "
            f"{len(js.get('object_names', []))} objects",
            flush=True,
        )

        # Design doc §5: 1 action per joint, so action_dim == num_joints —
        # not an independent number the schema or the subprocess get to pick.
        self._num_joints = len(js.get("joint_names", []))
        self._num_bodies = len(js.get("body_paths", []))
        self._num_objects = len(js.get("object_names", []))
        # Static per-body local AABB table (see get_body_aabbs). Absent on a
        # subprocess older than this field -- keep None and degrade to zero
        # boxes rather than guessing a shape.
        _ab = js.get("body_local_aabbs")
        self._body_aabb_local = np.asarray(_ab, dtype=np.float32) if _ab else None
        # Static scene-prop world AABBs (see get_static_prop_aabbs). Distinct
        # from body_local_aabbs above: props have no rigid body and no pose
        # row to compose with, so the subprocess computes and ships the
        # WORLD box directly (build_schema's static_prop_world_aabbs) rather
        # than a body-local box the client composes with live poses. Keyed
        # by the path as the eval checkers REQUEST it (e.g.
        # "/World/storage_bin"), not the resolved stage path. Absent /
        # empty on a subprocess older than this field, or when the task's
        # eval_static_props allowlist named nothing -- keep {} and degrade
        # to the shim's existing zero-box path rather than guessing a shape.
        _spa = js.get("static_prop_world_aabbs")
        self._static_prop_aabbs = (
            {str(k): [np.asarray(box, dtype=np.float32) for box in v] for k, v in _spa.items()} if _spa else {}
        )
        self._action_dim = self._num_joints
        self._state_dim = 2 * self._num_joints

        try:
            self._state_shm = FusedStateShm(
                self._state_shm_name,
                num_envs=self._num_envs,
                action_dim=self._action_dim,
                num_joints=self._num_joints,
                num_bodies=self._num_bodies,
                num_objects=self._num_objects,
                num_scene_joints=len(js.get("scene_joint_paths", [])),
                create=False,
            )
        except FileNotFoundError as e:
            # Distinct from the dims-mismatch case below on purpose: a
            # missing segment and a dimension disagreement point at
            # different causes (this side reading a stale schema with no
            # matching segment, vs. the two sides genuinely disagreeing on
            # shape), and a shared message that blames the subprocess for
            # the first case would be a misdiagnosis.
            raise RuntimeError(
                f"{type(self).__name__}: no FusedStateShm segment named "
                f"'{self._state_shm_name}' to attach to (dims derived from "
                f"schema {self._schema_out}: num_envs={self._num_envs}, "
                f"action_dim={self._action_dim}, num_joints={self._num_joints}, "
                f"num_bodies={self._num_bodies}, num_objects={self._num_objects}). "
                f"The subprocess creates this segment before writing its "
                f"ready-file, so reaching here means either the schema just "
                f"read is not from THIS subprocess (check the ready-file's "
                f"pid against the child's), or the child died between "
                f"creating it and now.\nLog tail:\n{self._read_log_tail()}"
            ) from e
        except ValueError as e:
            raise RuntimeError(
                f"{type(self).__name__}: failed to attach FusedStateShm "
                f"'{self._state_shm_name}' with dims derived from the "
                f"schema (num_envs={self._num_envs}, "
                f"action_dim={self._action_dim}, num_joints={self._num_joints}, "
                f"num_bodies={self._num_bodies}, num_objects={self._num_objects}) "
                f"— the subprocess's own segment disagrees with the schema "
                f"it reported.\nLog tail:\n{self._read_log_tail()}"
            ) from e

    def _read_log_tail(self, n: int = 2000) -> str:
        if not self._log_path:
            return ""
        try:
            with open(self._log_path, "r") as f:
                return f.read()[-n:]
        except Exception:  # noqa: BLE001
            return ""

    def _obs_tile_dims(self) -> tuple[int, int, int]:
        """``(n_tiles, h, w)`` the fused subprocess will write to the tiled
        image SHM — identical logic/rationale to isaac_tiled.py's
        ``_obs_tile_dims`` (one obs camera per env, tiled `[env_i][cam_j]`,
        common width / max height across cameras). Duplicated rather than
        imported: importing a concrete render bridge from a physics-backend
        module would be an odd layering, and the logic is a dozen lines of
        pure JSON parsing with a hard-coded fallback, not worth sharing.
        """
        h, w, n_cams = 400, 640, 1
        try:
            with open(self._stage_manifest, "r") as f:
                man = json.load(f)
            obs = [c for c in (man.get("cameras") or []) if not c.get("is_free_cam")]
            if obs:
                n_cams = len(obs)
                h = max(int(c.get("height", 0)) for c in obs) or h
                w = int(obs[0].get("width", w))
        except Exception:  # noqa: BLE001
            if self._cameras:
                n_cams = len(self._cameras)
                if "image_hw" in self._cameras[0]:
                    h = max(int(c["image_hw"][0]) for c in self._cameras if "image_hw" in c) or h
                    w = int(self._cameras[0]["image_hw"][1])
        return self._num_envs * n_cams, h, w

    def _attach_tiled_shm(self) -> None:
        n_tiles, h, w = self._obs_tile_dims()
        deadline = time.monotonic() + self._ready_timeout_s
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"{type(self).__name__}: fused sim subprocess exited "
                    f"with code {self._proc.returncode} before creating the "
                    f"tiled-SHM.\nLog tail:\n{self._read_log_tail()}"
                )
            try:
                self._tiled_shm = TiledFrameShm(self._tiled_shm_name, num_envs=n_tiles, height=h, width=w, create=False)
                print(
                    f"[{type(self).__name__}] attached tiled-SHM " f"'{self._tiled_shm_name}' ({n_tiles},{h},{w},3)",
                    flush=True,
                )
                return
            except FileNotFoundError as e:
                last_err = e
            except ValueError as e:
                last_err = e
            time.sleep(0.2)
        raise RuntimeError(
            f"{type(self).__name__}: tiled-SHM '{self._tiled_shm_name}' "
            f"({n_tiles},{h},{w},3) did not appear within "
            f"{self._ready_timeout_s:.0f}s. Last error: {last_err!r}"
        )

    def close(self) -> None:
        """Idempotent by construction: the handler calls this once as
        "engine.close()" and once as "bridge.close()" on this SAME object
        (design doc §2). The ``_closed`` flag makes every call after the
        first a true no-op rather than relying on "the resources are already
        None so every branch below happens to skip" — that pattern is
        fragile if any branch is ever reordered.
        """
        if self._closed:
            return
        self._closed = True

        # Best-effort graceful shutdown over the wire before killpg. This is
        # deliberately best-effort (broad except): a subprocess that is
        # already dead, wedged, or never got far enough to attach the SHM/sem
        # must not prevent teardown from proceeding to the killpg fallback
        # below, which is the mechanism that actually guarantees termination.
        if (
            self._state_shm is not None
            and self._sem is not None
            and self._proc is not None
            and self._proc.poll() is None
        ):
            try:
                from ..fused_shm import CMD_SHUTDOWN  # noqa: PLC0415 -- avoid

                # an unused-import lint hit on the happy path; this whole
                # block is a courtesy, not a load-bearing shutdown method.
                self._state_shm.set_command(CMD_SHUTDOWN)
                self._sem.post_request()
            except Exception:  # noqa: BLE001
                pass

        proc = self._proc
        self._proc = None
        self._started = False
        self._shm_attached = False
        self._schema_bound = False
        if proc is not None:
            pgid = proc.pid
            try:
                os.killpg(pgid, signal.SIGINT)
                proc.wait(timeout=10.0)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(pgid, signal.SIGTERM)
                    proc.wait(timeout=5.0)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        pass

        if self._log_fp is not None:
            try:
                self._log_fp.close()
            except Exception:  # noqa: BLE001
                pass
            self._log_fp = None

        # Tear down IPC LAST, after the subprocess is provably gone, so
        # unlinking can never race a still-exiting process still touching the
        # segments (same ordering rationale as isaac_tiled.py's close()).
        # This side created the SemPair -> this side unlinks it.
        if self._sem is not None:
            try:
                self._sem.close(unlink=True)
            except Exception:  # noqa: BLE001
                pass
            self._sem = None
        if self._state_shm is not None:
            # The SUBPROCESS created FusedStateShm (see start()/
            # _attach_state_shm — dims are only known once the stage is
            # built). This side only ever attached, so it only detaches;
            # unlinking here would be unlinking a segment we don't own, and
            # would race the subprocess's own cleanup on a clean exit.
            try:
                self._state_shm.close(unlink=False)
            except Exception:  # noqa: BLE001
                pass
            self._state_shm = None
        if self._tiled_shm is not None:
            # The SUBPROCESS created TiledFrameShm (unchanged ownership from
            # isaac_tiled.py). A clean subprocess exit (SIGINT) may already
            # have unlinked it; SIGTERM/SIGKILL does not. Unlink
            # unconditionally once the process is gone, same as
            # isaac_tiled.py's close() — a leftover segment under this pid-
            # keyed name would otherwise block the NEXT run using this pid
            # (rare but not impossible after a pid wraparound) from creating
            # it fresh.
            try:
                self._tiled_shm.close(unlink=True)
            except Exception:  # noqa: BLE001
                pass
            self._tiled_shm = None

    # ------------------------------------------------------------------
    # Main loop: physics (VecPhysicsBackend)
    # ------------------------------------------------------------------

    def _wait_for_response(self, what: str) -> None:
        assert self._sem is not None
        try:
            self._sem.wait_response(timeout=self._step_timeout_s)
        except Exception as e:  # posix_ipc.BusyError (timeout) or SignalError
            import posix_ipc  # noqa: PLC0415 -- lazy, see comm/shm/sem.py

            if isinstance(e, posix_ipc.BusyError):
                alive = self._proc is not None and self._proc.poll() is None
                raise TimeoutError(
                    f"{type(self).__name__}.{what}: no response from the "
                    f"fused Isaac subprocess within {self._step_timeout_s}s "
                    f"(subprocess alive={alive})"
                ) from e
            raise

    def step(self, actions: np.ndarray, need_images: bool = True) -> None:
        """Advance physics and optionally render within one subprocess round trip.

        ``need_images=False`` skips rendering, leaving the tiled frame ID
        unchanged. The next image-bearing step publishes a fresh frame for
        ``render_barrier`` to detect."""
        self._ensure_started()
        self._validate_actions(actions)
        assert self._state_shm is not None and self._sem is not None
        t0 = time.monotonic()
        self._state_shm.actions[:] = np.asarray(actions, dtype=np.float32)
        self._state_shm.set_command(CMD_STEP, want_image=bool(need_images))
        self._sem.post_request()
        _sent = time.monotonic()
        self._last_send_ms = (_sent - t0) * 1000
        self._wait_for_response("step")
        if self._state_shm.status != STATUS_OK:
            raise RuntimeError(f"{type(self).__name__}.step: " f"{self._state_shm.read_error()}")
        self._last_wait_ms = (time.monotonic() - _sent) * 1000
        self._last_step_ms = (time.monotonic() - t0) * 1000.0

    def reset(
        self,
        env_ids: Optional[Sequence[int]] = None,
        seeds: Optional[Sequence[int]] = None,
        instances: Optional[Sequence[int]] = None,
    ) -> None:
        """Reset selected environments with optional seeds and instance IDs.

        ``seeds`` and ``instances`` follow the selected environment order.
        Instance poses come from the superset manifest's
        ``instance_switch.instance_body_poses`` table; switching does not rebuild
        the stage. An omitted instance entry or ``-1`` keeps the current
        instance. The node rejects instance IDs outside that table."""
        self._ensure_started()
        assert self._state_shm is not None and self._sem is not None
        ne = self._num_envs
        ids = list(range(ne)) if env_ids is None else [int(i) for i in env_ids if 0 <= int(i) < ne]
        if not ids:
            return

        mask = np.zeros((ne,), dtype=np.int32)
        inst_arr = np.full((ne,), -1, dtype=np.int64)  # -1 = leave this env
        seeds_arr = np.full((ne,), -1, dtype=np.int64)  # -1 = verbatim replay
        for k, e in enumerate(ids):
            mask[e] = 1
            if seeds is not None:
                if k >= len(seeds):
                    raise ValueError(
                        f"{type(self).__name__}.reset: seeds has "
                        f"{len(seeds)} entries but {len(ids)} env(s) are "
                        f"being reset (env_ids={env_ids!r})"
                    )
                seeds_arr[e] = int(seeds[k])
            if instances is not None and k < len(instances):
                inst_arr[e] = int(instances[k])

        # Render-side generalization goes out BEFORE the reset command: the
        # node renders the reset frame inside that command, so the values must
        # already be waiting for it (an on_reset after the fact would leave the
        # first frame with the old lights and materials). The ack is checked
        # once the reset has returned.
        pending = self._send_render_randomization(ids, None if seeds is None else [int(seeds_arr[e]) for e in ids])

        self._state_shm.reset_mask[:] = mask
        self._state_shm.reset_seeds[:] = seeds_arr
        self._state_shm.reset_instances[:] = inst_arr
        # Always True: a full reset is expressed as an all-ones mask, not via
        # this flag — there is no case where we post a RESET with a mask we
        # want the server to disregard.
        self._state_shm.set_command(CMD_RESET, reset_mask_valid=True)
        self._sem.post_request()
        self._wait_for_response("reset")
        if self._state_shm.status != STATUS_OK:
            raise RuntimeError(f"{type(self).__name__}.reset: " f"{self._state_shm.read_error()}")
        if pending is not None:
            self._await_render_randomization(pending)

    def render_proc_alive(self) -> bool:
        """Whether the fused node process is running (the randomize channel
        refuses to wait on a dead node)."""
        return self._proc is not None and self._proc.poll() is None

    # ---- State readers --------------------------------------------------
    # Every getter below returns a COPY of the SHM-backed array, never a live
    # view: the caller may hold onto the returned array across the NEXT
    # step()/reset() call (e.g. sim_handler stashes a pre-reset joint-state
    # snapshot for the settle window — see sim_handler.py's ``hold_now``),
    # and the next step() overwrites these SAME SHM bytes in place. Handing
    # back a view would silently mutate an array the caller believes is a
    # frozen snapshot.

    def get_state(self) -> np.ndarray:
        """``(num_envs, 2*num_joints)`` = flattened ``[pos, vel]`` per joint.

        Meaning taken from ``_upstream_base.py``'s ``get_state()``
        (``joints.reshape(joints.shape[0], -1)`` over ``get_joint_states()``)
        — the newton/isaac_physx backends' own definition of "state" is
        exactly this, so matching it here keeps ``state_dim`` comparable
        across backends for the same task.
        """
        js = self.get_joint_states()  # already a copy, (N, J, 2)
        return js.reshape(js.shape[0], -1)  # (N, 2*J)

    def get_joint_states(self) -> np.ndarray:
        self._ensure_started()
        assert self._state_shm is not None
        return self._state_shm.joint_states.copy()

    def get_scene_joint_states(self) -> list[dict]:
        """Per-env scene-joint positions for eval, in PhysX DOF order.

        Labels keep the full env-relative USD path so the shim can match
        the owning object. Unavailable joints stay NaN: dropping individual
        entries would shift the checker's joint_index onto a different DOF.
        These DOFs never enter the robot action/state.
        """
        self._ensure_started()
        assert self._state_shm is not None
        names = self._schema_cache.get("scene_joint_paths", [])
        return [{name: float(q) for name, q in zip(names, row)} for row in self._state_shm.scene_joint_positions]

    def get_body_poses(self) -> np.ndarray:
        self._ensure_started()
        assert self._state_shm is not None
        return self._state_shm.body_poses.copy()

    def get_object_poses(self) -> np.ndarray:
        self._ensure_started()
        assert self._state_shm is not None
        return self._state_shm.object_poses.copy()

    def get_body_aabbs(self) -> np.ndarray:
        """``(N, B, 6)`` world-frame ``[min, max]``, ordered like
        ``get_body_poses()``.

        Same construction as the newton path: a STATIC body-frame box table
        (shipped once in the schema, since geometry does not change at
        runtime) composed with the live poses by ``_world_aabb_from_local``.
        Sharing that function is the point -- the eval checkers must not see
        two engines' AABBs disagree because one of them re-derived the corner
        math.

        Zero boxes when the subprocess published no table, matching newton's
        "table hasn't arrived" degradation. ``api_core_shim.get_obj_aabb``
        already treats AABBs as best-effort; returning zeros keeps that
        contract instead of raising NotImplementedError at callers that have
        no such guard.
        """
        self._ensure_started()
        poses = self.get_body_poses()
        table = getattr(self, "_body_aabb_local", None)
        if table is None or len(table) == 0:
            return np.zeros((poses.shape[0], poses.shape[1], 6), dtype=np.float32)
        return _world_aabb_from_local(poses, table)

    def get_static_prop_aabbs(self) -> dict[str, list[np.ndarray]]:
        """``{requested_path: [[min_x,min_y,min_z,max_x,max_y,max_z], ...]}``
        per-env world AABBs for STATIC scene props this task's
        ``eval_static_props`` allowlist names.

        Outer dict keyed by requested path (e.g. ``/World/storage_bin``).
        Inner list indexed by env_id, each element a 6-element WORLD-frame
        box ``[min_x, min_y, min_z, max_x, max_y, max_z]``.

        Populated once from the schema (see ``_attach_state_shm`` /
        ``static_prop_world_aabbs`` in ``isaacsim_fused_sim.build_schema``),
        not per-step: these props never move, so their world box is a
        build-time constant, same reasoning as ``get_body_aabbs``'s local
        table but WORLD-frame already (no live pose to compose with -- props
        have no body row). Keys are the path as the checker REQUESTED it
        (e.g. ``/World/storage_bin``), not the resolved ``/Background/...``
        stage path, so ``api_core_shim`` can look it up with the exact
        string the checker already has.

        Returns ``{}`` when the schema carried no table (older subprocess,
        or this task's allowlist named nothing) -- matching the base class's
        contract, this does NOT raise.
        """
        self._ensure_started()
        return dict(getattr(self, "_static_prop_aabbs", None) or {})

    def get_camera_poses(self) -> np.ndarray:
        # Same gap as _upstream_base.py's get_camera_poses: no FK service on
        # this wire either (FusedStateShm carries no camera-pose region).
        # Renderer/caller falls back to CameraSpec.link + body_poses.
        self._ensure_started()
        return np.zeros((self._num_envs, 0, 7), dtype=np.float32)

    def get_sim_time(self) -> np.ndarray:
        self._ensure_started()
        assert self._state_shm is not None
        return self._state_shm.sim_time.copy()

    def get_scene_schema(self) -> SceneSchema:
        """Read once at ``start()`` (via ``_attach_state_shm``) from the JSON
        the subprocess writes to ``--schema-out``; cached here. Returns a deep
        copy so a caller mutating the result (or mutating what
        ``bind_scene_schema`` stores) can never corrupt this instance's cache
        for a later ``get_scene_schema()`` call.
        """
        self._ensure_started()
        if self._schema_cache is None:
            # Should already be populated by start(); re-read defensively
            # rather than crash if something cleared it.
            self._schema_cache = json.loads(Path(self._schema_out).read_text())
        return copy.deepcopy(self._schema_cache)

    def set_qpos(self, env_id: int, qpos: np.ndarray) -> None:
        # Same gap as _upstream_base.py's set_qpos: FusedStateShm has no
        # direct-write command (only STEP / RESET / SHUTDOWN), so there is no
        # wire path to overwrite qpos without integrating physics.
        raise NotImplementedError(
            f"{type(self).__name__}.set_qpos requires a wire command "
            f"FusedStateShm does not have (only STEP/RESET/SHUTDOWN)"
        )

    # ------------------------------------------------------------------
    # Main loop: render (VecRenderBackend)
    # ------------------------------------------------------------------

    def attach_shm(self, frame_shm: Any, layout: Any) -> None:
        self._ensure_started()
        self._frame_shm = frame_shm
        self._layout = layout
        self._shm_attached = True

    def bind_scene_schema(self, schema: SceneSchema) -> None:
        self._ensure_started()
        self._schema = schema
        # Record the built instance so reset() can distinguish "switch to the
        # instance I am already on" (a no-op every caller emits, because
        # sim_handler always passes the env's current instance) from a genuine
        # switch to a DIFFERENT instance (which this backend cannot do).
        try:
            with open(self._stage_manifest, "r") as _mf:
                _bt = (json.load(_mf) or {}).get("_benchmark_task") or {}
            if _bt.get("instance_id") is not None:
                self._build_instance = int(_bt["instance_id"])
                print(f"[{type(self).__name__}] stage built for instance " f"{self._build_instance}", flush=True)
        except Exception:  # noqa: BLE001 -- absence just means "unknown"
            self._build_instance = None
        self._schema_bound = True

    def update_poses(
        self,
        body_poses: np.ndarray,
        sim_time: np.ndarray,
        camera_poses: Optional[np.ndarray] = None,
        object_poses: Optional[np.ndarray] = None,
        cloth_vertices: Optional[np.ndarray] = None,
        tick: int = 0,
        will_collect: bool = True,
        body_names: Optional[list] = None,  # ignored, for API compat
        **kwargs,  # catch any future additions
    ) -> None:
        """Keep the shared physics/render stage unchanged.

        Fused physics already updates the stage rendered by the tiled camera.
        The shared render interface requires this method, but no pose transfer
        is needed between processes."""
        self._ensure_started()
        return

    def render(self) -> RenderResult:
        self.render_barrier()
        return self.render_copy()

    def render_barrier(self) -> bool:
        """Check whether a completed round trip published a fresh tiled frame.

        The subprocess writes state and images before posting its response
        semaphore, so this method does not poll. It returns ``False`` and marks
        the frame dropped when the tiled frame ID has not advanced."""
        self._ensure_started()
        self._ensure_shm_attached()
        if self._tiled_shm is None:
            self._last_barrier_dropped = True
            return False
        if self._proc is not None and self._proc.poll() is not None:
            raise RuntimeError(
                f"{type(self).__name__}: fused Isaac subprocess exited "
                f"(code={self._proc.returncode}) — no frame will arrive"
            )
        fid = self._tiled_shm.frame_id
        satisfied = fid > self._last_tiled_frame_id
        if satisfied:
            self._last_tiled_frame_id = fid
        elif not self._frame_drop_warned:
            # Should not happen under the design's contract (every STEP/RESET
            # round trip renders exactly one new frame) — if it does, it is a
            # subprocess-side bug worth knowing about, but not fatal to this
            # one frame (see render_barrier's docstring: this is defense in
            # depth, not the expected path).
            print(
                f"[{type(self).__name__}] WARNING: tiled-SHM frame_id did "
                f"not advance after a completed step()/reset() round trip "
                f"(frame_id={fid}) — the fused subprocess should render "
                f"every round trip; treating this frame as dropped",
                flush=True,
            )
            self._frame_drop_warned = True
        self._last_barrier_dropped = not satisfied
        return satisfied

    @staticmethod
    def _fill_rgb_from_tiled(rgb_buf: np.ndarray, tiled_rgb: np.ndarray) -> bool:
        """Copy each de-tiled per-env head frame into ``rgb_buf[env, cam]``.

        Same fast-path-plus-fallback shape as isaac_tiled.py's
        ``_fill_rgb_from_tiled`` (bulk reshape when the layout matches
        exactly, per-tile loop otherwise) — duplicated rather than imported
        for the same reason as ``_obs_tile_dims`` above.
        """
        n_envs = int(rgb_buf.shape[0])
        n_cams = int(rgb_buf.shape[1])
        H, W = int(rgb_buf.shape[2]), int(rgb_buf.shape[3])
        n_tiles = int(tiled_rgb.shape[0])
        if n_tiles == n_envs * n_cams and int(tiled_rgb.shape[1]) == H and int(tiled_rgb.shape[2]) == W:
            src = tiled_rgb if tiled_rgb.shape[-1] == 3 else tiled_rgb[..., :3]
            rgb_buf[:] = src.reshape(n_envs, n_cams, H, W, 3)
            return False
        dropped = n_tiles != n_envs * n_cams
        for i in range(n_envs):
            for j in range(n_cams):
                t = i * n_cams + j
                if t >= n_tiles:
                    dropped = True
                    continue
                tile = tiled_rgb[t]
                if int(tile.shape[0]) == H and int(tile.shape[1]) == W:
                    rgb_buf[i, j] = tile[..., :3]
                else:
                    dropped = True
        return dropped

    def render_copy(self) -> RenderResult:
        dropped = self._last_barrier_dropped
        stamp_arr = (
            self._state_shm.sim_time.copy()
            if self._state_shm is not None
            else np.zeros((self._num_envs,), dtype=np.float64)
        )

        if self._tiled_shm is not None and self._frame_shm is not None:
            try:
                rgb_buf = self._frame_shm.field("rgb_buf")
                copy_dropped = self._fill_rgb_from_tiled(rgb_buf, self._tiled_shm.rgb)
                dropped = dropped or copy_dropped
                self._frame_shm.field("frame_index").fill(self._frame_index + 1)
                # Unlike isaac_tiled.py (whose tiled-SHM carries no
                # timestamp, forcing it to stash sim_time from update_poses),
                # we have the AUTHORITATIVE per-env sim_time straight from
                # FusedStateShm — no stashing needed since update_poses is a
                # no-op here.
                self._frame_shm.field("frame_sim_time")[0] = float(stamp_arr[0]) if stamp_arr.size else 0.0
                self._frame_shm.field("frame_ready").fill(True)
                self._frame_shm.field("frame_tick").fill(int(self._tiled_shm.rendered_tick))
            except KeyError:
                pass

        self._frame_index += 1
        # Rendering happened INSIDE step()'s round trip, not as a separate
        # timed phase here — last_step_ms (on the physics side of this same
        # object) already includes it. Reporting 0 here would be misleading
        # as "free"; reporting step's own number would double-count if a
        # caller sums both. Report step's number anyway (see health_check's
        # docstring for the same tradeoff) since some callers only look at
        # the render side's latency and a fused backend has no OTHER number
        # to give them.
        latency_ms = self._last_step_ms
        return RenderResult(
            sim_time=stamp_arr,
            frame_index=self._frame_index,
            latency_ms=latency_ms,
            dropped=dropped,
        )

    # ------------------------------------------------------------------
    # Health / metrics
    # ------------------------------------------------------------------

    def health_check(self) -> dict:
        """Overridden (rather than inheriting either parent's default) on
        purpose: ``VecPhysicsBackend.health_check`` and ``VecRenderBackend.
        health_check`` are both NON-abstract methods with the SAME NAME,
        returning differently-shaped TypedDicts (``HealthStatus`` has
        ``last_step_ms``; ``RenderHealthStatus`` has ``last_render_ms``).
        Plain multiple inheritance would silently resolve to
        ``VecPhysicsBackend``'s version (leftmost in the MRO) for BOTH roles,
        so any caller treating this object as "the bridge" and reading
        ``health["last_render_ms"]`` would KeyError. Returning the union of
        both fields here is correct for whichever role calls it.
        """
        self._ensure_started()
        alive = self._started and (self._proc is None or self._proc.poll() is None)
        return {
            "alive": alive,
            "pid": self._pid,
            "last_step_ms": float(self._last_step_ms),
            "last_render_ms": float(self._last_step_ms),
            "cuda_oom": False,
            "error_msg": "",
        }

    def get_metrics(self) -> dict[str, float]:
        return {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def supports_batching(self) -> bool:
        return self.SUPPORTS_BATCHING

    @property
    def action_dim(self) -> int:
        # Same "only known after start()" contract as the ROS backends
        # (_upstream_base.py) and for the same underlying reason: the
        # subprocess must build the Isaac stage before the joint count is
        # real. See __init__'s comment on why these are not constructor args.
        if self._action_dim == 0:
            raise RuntimeError(
                f"{type(self).__name__}.action_dim is only known after "
                f"start() reads the schema the subprocess reports"
            )
        return self._action_dim

    @property
    def state_dim(self) -> int:
        if self._state_dim == 0:
            raise RuntimeError(f"{type(self).__name__}.state_dim is only known after " f"start()")
        return self._state_dim

    @property
    def output_channels(self) -> list[str]:
        return list(self.OUTPUT_CHANNELS)

    @property
    def backend_name(self) -> str:
        return self.BACKEND_NAME


__all__ = ["FusedIsaacBackend"]
