# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""GenieSimVecEnv — a gym-style vectorized environment over the sim protocol.

Holds one comm client and translates RL semantics into the transport verbs
(``handshake`` / ``step`` / ``reset``). Imports no engine, USD, or ROS code.

Action layout: ``[arm(14 = L7+R7), gripper(2)]``, plus ``waist(5)`` when
``control_waist`` is set, as PD position targets scattered into the sim's full
joint vector; non-controlled joints hold their reset targets.

Observation layout: ``states`` is 24-dim
``[left_arm(7), right_arm(7), gripper_L, gripper_R, waist(5), head(3)]``.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from rlinf_geniesim.comm.base import STATUS_READY

# Wire constant for "the server has no task yet"; the construction-time
# handshake uses it to tell an idle server from a real handshake failure.
from rlinf_geniesim.comm.zmq._wire import NO_ACTIVE_TASK

# Sensor imaging generalization: entirely client-side, no engine/render
# coupling -- see _apply_sensor_randomization.
from rlinf_geniesim.randomize import RandomizationSpec, Sampler, substreams
from rlinf_geniesim.randomize import sensor as sensor_randomize


# Top-level task-init sections, used only to warn about typos in
# ``env.init(config=...)``. Not a whitelist: unknown sections are still sent.
_KNOWN_INIT_SECTIONS = frozenset(
    {
        "rates",
        "action",
        "observation",
        "episode",
        "reward",
        "termination",
        "randomization",
        "seed",
        "multi_env",
        "manifest",
        "debug",
        # SHM buffer size estimates (num_bodies_estimate / num_objects_estimate).
        # The layout is fixed when the world is built, so this can only be given
        # at init time.
        "scene",
    }
)

try:  # torch is present in RLinf workers; optional for host unit tests
    import torch

    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    torch = None
    _HAS_TORCH = False


def _to_host_array(x) -> np.ndarray:
    """Move a value to host memory; ``np.asarray`` cannot read GPU memory."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.cpu()
    return np.asarray(x)


# Benchmark G2_omnipicker control groups (mirrors
# geniesim_benchmark.utils.name_utils.ROBOT_CONFIGS — inlined so this module
# imports without the benchmark package; override via config for other robots).
_G2_LEFT_ARM = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
_G2_RIGHT_ARM = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
_G2_GRIPPER = ["idx41_gripper_l_outer_joint1", "idx81_gripper_r_outer_joint1"]
_G2_WAIST = [f"idx0{i}_body_joint{i}" for i in range(1, 6)]
_G2_HEAD = [f"idx1{i}_head_joint{i}" for i in range(1, 4)]

# The defaults above, keyed by config field, so ``_resolve_layout`` can tell
# "the caller never set this" from "the caller asked for exactly this". The
# distinction decides whether a mismatch is auto-resolved or reported: an
# explicit request that does not fit the scene is a mistake worth surfacing,
# while an untouched default is just this module not knowing the robot yet.
_JOINT_GROUP_DEFAULTS = {
    "left_arm_joints": _G2_LEFT_ARM,
    "right_arm_joints": _G2_RIGHT_ARM,
    "gripper_joints": _G2_GRIPPER,
    "waist_joints": _G2_WAIST,
    "head_joints": _G2_HEAD,
}


def _autoresolve_joint_groups(names: list) -> tuple:
    """Select benchmark joint groups that match the advertised joint names.

    Return the unique matching group layout and candidate robot names. Return
    (None, reason) if the benchmark is unavailable, no layout fits, or matching
    robot configurations disagree. The caller decides whether to use defaults."""
    try:
        from geniesim_benchmark.utils.name_utils import ROBOT_CONFIGS  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        # Deliberately soft. This module inlines the constants above so it
        # imports without the benchmark package; losing that property to a
        # convenience feature would be a bad trade.
        return None, f"geniesim_benchmark not importable ({type(exc).__name__})"

    have = set(names)
    fields = tuple(_JOINT_GROUP_DEFAULTS)
    hits, sigs = [], {}
    for robot, conf in ROBOT_CONFIGS.items():
        groups = {f: list(conf.get(f) or []) for f in fields}
        if any(not g for g in groups.values()):
            continue
        if not all(j in have for g in groups.values() for j in g):
            continue
        hits.append(robot)
        sigs[tuple(tuple(groups[f]) for f in fields)] = groups
    if not hits:
        return None, "no ROBOT_CONFIGS entry has all of its joints in this scene"
    if len(sigs) > 1:
        return None, (
            f"candidates {hits} disagree on the groups, so there is no "
            f"single answer -- set the joint groups explicitly"
        )
    return next(iter(sigs.values())), hits


class SimStateNonFiniteError(ValueError, RuntimeError):
    """Joint state from the server contains NaN or Inf.

    A distinct type so a loop can catch it, record the episode as failed, and
    continue rather than losing the whole run.
    """

    def __init__(self, message: str, *, bad_envs=()) -> None:
        super().__init__(message)
        self.bad_envs = tuple(int(e) for e in bad_envs)


@dataclass
class GenieSimVecEnvConfig:
    # --- session ---
    bootstrap_path: str = "/tmp/boot.json"
    comm_backend: str = "shm"
    comm_kwargs: dict = field(default_factory=dict)  # backend extras (zmq: address=...)
    spawn_server: bool = False  # False = attach to a running server
    task_init_yaml: str = ""  # used when spawn_server=True
    server_python: str = "python3"
    server_script: str = "source/rlinf_geniesim/scripts/sim_server.py"
    server_log: str = "/tmp/rlinf_vecenv_sim.log"
    serve_timeout_s: float = 600.0
    step_timeout_s: float = 60.0

    # --- task identity (for task_descriptions; auto-read from instructions.json)
    task_name: str = ""
    sub_task_name: str = ""
    instance_id: int = 0
    task_description: str = ""  # explicit override wins

    # --- action space (benchmark-aligned) ---
    left_arm_joints: list = field(default_factory=lambda: list(_G2_LEFT_ARM))
    right_arm_joints: list = field(default_factory=lambda: list(_G2_RIGHT_ARM))
    gripper_joints: list = field(default_factory=lambda: list(_G2_GRIPPER))
    waist_joints: list = field(default_factory=lambda: list(_G2_WAIST))
    head_joints: list = field(default_factory=lambda: list(_G2_HEAD))
    control_waist: bool = False  # +5 action dims when True

    # --- observation ---
    # obs-dict key -> camera index in the sim's rgb_buf camera axis
    # (camera order == benchmark camera_list: head, left wrist, right wrist).
    image_keys: dict = field(default_factory=lambda: {"main_images": 0, "wrist_images": 1, "wrist_images_right": 2})

    # --- episode ---
    max_episode_steps: int = 200
    auto_reset: bool = True
    ignore_terminations: bool = False

    # --- sensor imaging generalization ---
    # Resolved sensor-side spec DOMAIN -- same shape as
    # ``rlinf_geniesim.randomize.RandomizationSpec.sensor``
    # (``{"noise": {...}, "occlusion": {...}, "drop_frame": {...}}``), NOT
    # per-env drawn values. The caller is expected to have produced this via
    # ``RandomizationSpec.resolve(task_name, yaml_randomization,
    # bench_root=...).sensor`` -- this env has no server round-trip to fetch
    # it itself; execution is entirely client-side. Default ``{}`` = off,
    # and off means byte-for-byte identical to this feature not existing:
    # ``_wrap_obs`` never even calls into ``randomize.sensor`` when this is
    # empty (see ``_apply_sensor_randomization``).
    sensor: dict = field(default_factory=dict)

    # --- torch boundary ---
    to_torch: bool = True
    device: str = "cpu"


# Sentinel for GenieSimVecEnv.__init__'s `to_torch`/`device` params: both
# already have real, non-None defaults (True / "cpu"), so `None` can't double
# Distinguish omitted options from explicit overrides in the legacy constructor.
_UNSET = object()


class GenieSimVecEnv:
    """Vectorized (N envs in ONE sim server) gym-style env for RLinf."""

    # cfg fields the injected construction retires; passing one raises.
    _RETIRED_CFG_FIELDS = (
        "comm_backend",
        "comm_kwargs",
        "bootstrap_path",
        "spawn_server",
        "task_init_yaml",
        "server_python",
        "server_script",
        "server_log",
        "task_name",
        "sub_task_name",
        "serve_timeout_s",
        "step_timeout_s",
    )

    def __init__(self, client_or_cfg, *, to_torch=_UNSET, device=_UNSET, **cfg_overrides):
        """Two forms of construction.

        * ``GenieSimVecEnv(client, to_torch=..., device=..., **preferences)``
          reuses a connection the caller already opened; the task comes from
          ``env.init()``. This is the recommended form.
        * ``GenieSimVecEnv(cfg)`` is the legacy path, kept for one release
          cycle with a ``DeprecationWarning``.
        """
        if client_or_cfg is None:
            raise TypeError(
                "the first argument must be a connected SimClient "
                "(recommended) or a GenieSimVecEnvConfig (legacy), not None"
            )
        if isinstance(client_or_cfg, GenieSimVecEnvConfig):
            import warnings  # noqa: PLC0415

            warnings.warn(
                "GenieSimVecEnv(cfg) is deprecated; use "
                "GenieSimVecEnv(client, ...) plus env.init({...}). The address "
                "is given once in make_client and the task comes from "
                "env.init. This form is kept for one release cycle.",
                DeprecationWarning,
                stacklevel=2,
            )
            # to_torch/device are keyword-only and never land in
            # cfg_overrides, so fold them into the same rejected set.
            bad_kw = dict(cfg_overrides)
            if to_torch is not _UNSET:
                bad_kw["to_torch"] = to_torch
            if device is not _UNSET:
                bad_kw["device"] = device
            if bad_kw:
                raise TypeError(
                    f"the legacy GenieSimVecEnv(cfg) form takes no extra "
                    f"keyword arguments, including to_torch and device: "
                    f"{sorted(bad_kw)}. Put them in cfg itself."
                )
            cfg = client_or_cfg
            self._owns_client = True
        else:
            bad = [k for k in cfg_overrides if k in self._RETIRED_CFG_FIELDS]
            if bad:
                raise ValueError(
                    f"these fields are retired in the injected construction "
                    f"and would have no effect, so they are rejected: "
                    f"{sorted(bad)}. The address is given once in make_client; "
                    f"task_name/sub_task_name come from env.init(); "
                    f"spawn_server conflicts with reusing a client; "
                    f"serve_timeout_s/step_timeout_s are replaced by "
                    f"make_client(timeout_s=...)."
                )
            cfg = GenieSimVecEnvConfig(
                to_torch=True if to_torch is _UNSET else to_torch, device="cpu" if device is _UNSET else device
            )
            for k, v in cfg_overrides.items():
                if not hasattr(cfg, k):
                    raise TypeError(f"unknown GenieSimVecEnv argument: {k!r}")
                setattr(cfg, k, v)
            self._client = client_or_cfg
            self._owns_client = False
        self.cfg = cfg
        self.num_envs = 0  # filled at handshake
        self._proc: Optional[subprocess.Popen] = None
        self._dims = None
        self._schema: dict = {}
        self._joint_names: list[str] = []
        self._hold_targets: Optional[np.ndarray] = None  # (N, J) full targets
        self._elapsed = None  # (N,) int64
        self._unconfirmed_reset_envs: set[int] = set()
        # Sensor imaging generalization -- per-env concrete config
        # (``RandomizationPlan.sensor`` shape) and the rng that drives this
        # env's ongoing frame-level draws for the CURRENT episode, keyed by
        # env_id. Both re-derived from scratch on every reset() (see
        # _reset_sensor_randomization); absent/None entries mean "no
        # augmentation this episode" (cfg.sensor empty, or this env's last
        # reset used seed < 0). ``_sensor_cache`` is the dirt-mask cache
        # (keyed (camera_name, h, w, ratio)), shared across envs and the
        # env's whole lifetime -- see randomize.sensor's module docstring.
        self._sensor_cfg: dict[int, dict] = {}
        self._sensor_rng: dict[int, Optional[np.random.Generator]] = {}
        self._sensor_cache: dict[tuple, np.ndarray] = {}
        self._is_start = True
        self._task_desc: str = ""
        # Instruction text comes from the server's `task_metadata`.
        self._sub_task_name = getattr(cfg, "sub_task_name", "") or ""
        self._server_instr_by_instance = None
        # With no task on the server, construction succeeds but the layout
        # stays empty until the first `init()`.
        self._layout_ready = False
        if self._owns_client:
            self._connect()  # legacy: open our own connection
        else:
            self._handshake_refresh()  # injected: only fetch dims/schema
        if self._dims is not None:
            self._resolve_layout()
            self._init_metrics()

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    @property
    def client(self):
        """Return the owned or caller-injected communication client.

        Control-plane calls such as ``set_backends`` are safe here. Switch tasks
        through ``env.init()`` so cached dimensions and joint indices are refreshed.
        """
        return self._client

    def effective_randomization(self) -> dict:
        """Read server randomization state and add local sensor plans.

        When applied.last_reset is available, annotate its existing per-env entries
        with sensor settings. Otherwise expose local plans under applied.sensor_only.
        An empty sensor mapping means disabled or not reset. This query requires a
        connected ZMQ client but does not require env.init to have run."""
        out = self._client.randomization_readback()
        sensor_by_env = {e: (self._sensor_cfg.get(e) or {}) for e in range(self.num_envs)}
        applied = out.get("applied")
        last_reset = applied.get("last_reset") if applied else None
        if last_reset is not None:
            for env_key, plan in (last_reset.get("per_env") or {}).items():
                plan["sensor"] = sensor_by_env.get(int(env_key), {})
        else:
            if applied is None:
                applied = {}
                out["applied"] = applied
            applied["sensor_only"] = {str(e): cfg for e, cfg in sensor_by_env.items()}
        return out

    def init(self, task: dict, config: Optional[dict] = None, timeout_s: float = 900.0) -> None:
        """Create a task, or switch to another one. Same call, same object.

        ``task`` is the task section. It needs at least ``sub_task_name``,
        the real task name. ``task_name``, the "background + robot"
        combination, is required the first time and inherited afterwards.
        ``num_envs`` is parallelism and is inherited when omitted. Any other
        task key (``instances``, ``superset_instances``,
        ``reset_settle_time_s``) is passed through unchanged.

        ``config`` carries the other top-level task-init sections: ``rates``,
        ``multi_env``, ``reward``, ``randomization``, ``manifest``, ``debug``,
        ``observation`` and so on. Omitted sections fall back to the server
        default the first time and are inherited afterwards. This is the only
        way to express them, since the server holds no startup yaml. Most
        callers need none of them: the defaults are the normal values
        (physics_hz 120, control_hz 30, success_from_checker true,
        env_spacing 15.0).

        The server merges this with the declared backends and builds the
        world, returning new dims and schema. This object then refreshes in
        place, so references the caller already holds stay valid.
        """
        sub = str(task.get("sub_task_name") or "")
        if not sub:
            raise ValueError(
                "env.init needs a sub_task_name (the real task name; "
                "task_name is the background-plus-robot combination)"
            )
        # Send the engine section only if the caller declared one; a
        # client-guessed default would be rejected by the backends lock.
        backends = {}
        if self._client.backends_declared():
            backends = dict(self._client.get_backends())
        backends["num_envs"] = int(task.get("num_envs", self.num_envs) or 1)
        # num_envs moves into backends; an empty task_name omits the key
        # entirely, since the deep merge would otherwise overwrite a good
        # value with an empty one.
        task_section = {k: v for k, v in task.items() if k != "num_envs"}
        task_section["sub_task_name"] = sub
        if not str(task.get("task_name") or ""):
            task_section.pop("task_name", None)
        payload = {"task": task_section, "backends": backends}
        for section, value in (config or {}).items():
            if section in ("task", "backends"):
                raise ValueError(
                    f"env.init's config must not carry the {section!r} "
                    f"section: the task section goes in the first argument "
                    f"and the engine section in client.set_backends(). Two "
                    f"places to write one thing will drift."
                )
            if section == "comm":
                raise ValueError(
                    "env.init's config must not carry a 'comm' section: "
                    "which port to bind is decided when the server starts "
                    "(sim_server --comm-config) and a client cannot change it."
                )
            if section not in _KNOWN_INIT_SECTIONS:
                # Warn rather than reject: the server may add sections later.
                warnings.warn(
                    f"{section!r} in env.init(config=...) is not a known "
                    f"task-init section. It was sent to the server as given, "
                    f"but a typo will be silently ignored there. Known "
                    f"sections: {sorted(_KNOWN_INIT_SECTIONS)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            payload[section] = value
        self._layout_ready = False
        self._dims, schema = self._client.init(payload, timeout_s=timeout_s)
        self._schema = json.loads(schema) if isinstance(schema, str) else (schema or {})
        self.num_envs = int(self._dims.num_envs)
        self._adopt_active_task(sub)

    def attach_task(self, sub_task_name: str) -> None:
        """Adopt a task the server already holds: the second half of
        ``init()`` with no build request.

        Re-handshakes for dims and schema, re-resolves the layout, and clears
        the per-subtask instruction cache. Raises if the server has no task,
        rather than doing an ``init()`` the caller never asked for.
        """
        sub = str(sub_task_name or "")
        if not sub:
            raise ValueError(
                "attach_task needs a sub_task_name (the real task name; "
                "task_name is the background-plus-robot combination)"
            )
        self._handshake_refresh()
        if self._dims is None:
            raise RuntimeError(
                f"attach_task({sub!r}): the server has no active task. This "
                f"method only adopts a task that is already loaded; it does "
                f"not build one. Load a task on the server first, or call "
                f"env.init() explicitly."
            )
        self._adopt_active_task(sub)

    def _adopt_active_task(self, sub: str) -> None:
        """Refresh in place onto subtask ``sub`` using the current dims and
        schema. Shared second half of ``init`` and ``attach_task``; references
        the caller already holds stay valid."""
        self._sub_task_name = sub
        # Order matters: re-resolve the layout first, so the action scatter
        # and observation gather indices are recomputed against the new
        # schema's joint names, then clear episode state, then reset metrics.
        self._hold_targets = None
        self._elapsed = None
        self._unconfirmed_reset_envs = set()
        self._instr_cache = {}
        # Server-side instruction table is keyed by SUB TASK, not by
        # instance -- a task switch invalidates the whole cached table, not
        # just the per-instance entries `_instr_cache` already clears above.
        # Leaving this set would silently answer the NEW sub task's
        # `_instr_table()` calls with the OLD sub task's instructions.
        self._server_instr_by_instance = None
        self._resolve_layout()
        self._init_metrics()
        self._is_start = True

    def _handshake_refresh(self) -> None:
        """Fetch dims and schema at construction time.

        A server with no task is not an error: ``NO_ACTIVE_TASK`` is absorbed
        as the uninitialised state and the first ``env.init()`` fills it in.
        Other handshake errors propagate unchanged.
        """
        self._layout_ready = False
        try:
            dims, schema = self._client.handshake()
        except RuntimeError as exc:
            if NO_ACTIVE_TASK not in str(exc):
                raise
            self._dims = None
            self._schema = {}
            self.num_envs = 0
            return
        self._dims = dims
        self._schema = json.loads(schema) if isinstance(schema, str) else (schema or {})
        self.num_envs = int(self._dims.num_envs)

    def _require_layout(self, what: str) -> None:
        """Guard every entry point that needs a resolved layout, so an
        uninitialised env gives a named error rather than an AttributeError or
        an empty num_envs=0 array."""
        if not getattr(self, "_layout_ready", False):
            raise RuntimeError(
                f"{what}: this env has no resolved task layout. Call env.init(...) "
                f"to initialize a task, or env.attach_task(...) to adopt an active task."
            )

    def _connect(self) -> None:
        cfg = self.cfg
        if cfg.spawn_server:
            # Removed with the server's --task-init path; name the two steps
            # that replace it.
            raise RuntimeError(
                "spawn_server=True is no longer supported: the server starts "
                "without a task, and this path relied on the removed "
                "--task-init. Use two steps instead: start the server "
                "(sim_server.py --comm-config <comm.yaml>, or "
                "scripts/start_sim_server.sh), then construct with "
                "GenieSimVecEnv(client) + env.init({'task_name': ..., "
                "'sub_task_name': ...})."
            )
        from rlinf_geniesim.comm import make_client  # noqa: PLC0415

        self._client = make_client(
            cfg.comm_backend, bootstrap_path=cfg.bootstrap_path, timeout_s=cfg.step_timeout_s, **cfg.comm_kwargs
        )
        self._handshake_refresh()

    def close(self) -> None:
        # The caller owns an injected client; closing the env only drops its reference.
        if not self._owns_client:
            self._client = None
            return
        # Explicitly shut down a server spawned by this env. Otherwise cleanup
        # follows the transport client's close() contract.
        try:
            if self._client is not None:
                if self._proc is not None:
                    self._client.shutdown()  # we own the server
                else:
                    self._client.close()
        except Exception:  # noqa: BLE001
            pass
        if self._proc is not None and self._proc.poll() is None:
            try:
                import signal  # noqa: PLC0415

                os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
                self._proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                self._proc.kill()

    # ------------------------------------------------------------------
    # Layout: action scatter + obs gather indices (by joint NAME)
    # ------------------------------------------------------------------

    def _resolve_layout(self) -> None:
        self._layout_ready = False
        cfg = self.cfg
        names = list(self._schema.get("joint_names") or [])
        if not names:
            raise RuntimeError("handshake schema has no joint_names")
        self._joint_names = names
        idx = {n: i for i, n in enumerate(names)}

        # Fit the joint groups to the robot the server actually loaded.
        #
        # Only groups still holding their default are eligible: a caller who
        # named the joints explicitly gets their request honoured, and if it
        # does not fit they get the error below rather than a silent swap to
        # something else. Same precedence this repo uses everywhere else --
        # explicit beats inferred, and inferred beats a stale default.
        inferred = getattr(self, "_inferred_joint_groups", {})
        for f, previous in inferred.items():
            if getattr(cfg, f) == previous:
                setattr(cfg, f, list(_JOINT_GROUP_DEFAULTS[f]))
        stale = [
            f
            for f, dflt in _JOINT_GROUP_DEFAULTS.items()
            if getattr(cfg, f) == dflt and any(n not in idx for n in getattr(cfg, f))
        ]
        if stale:
            groups, why = _autoresolve_joint_groups(names)
            if groups:
                for f in stale:
                    setattr(cfg, f, list(groups[f]))
                    inferred[f] = list(groups[f])
                self._inferred_joint_groups = inferred
                print(
                    f"[vec_env] joint groups auto-resolved for this scene "
                    f"(candidates {why}): " + ", ".join(f"{f}={getattr(cfg, f)}" for f in stale),
                    flush=True,
                )
            else:
                # Not fatal here. _need below reports which joints are missing,
                # which is the more useful half; this only adds why the lookup
                # could not help.
                print(
                    f"[vec_env] these joint groups do not fit the scene and "
                    f"could not be resolved automatically: {stale} ({why})",
                    flush=True,
                )

        def _need(group: list[str], label: str) -> list[int]:
            missing = [n for n in group if n not in idx]
            if missing:
                raise RuntimeError(f"{label} joints not in sim schema: {missing}")
            return [idx[n] for n in group]

        self._arm_idx = _need(cfg.left_arm_joints + cfg.right_arm_joints, "arm")
        self._grip_idx = _need(cfg.gripper_joints, "gripper")
        self._waist_idx = _need(cfg.waist_joints, "waist")
        self._head_idx = _need(cfg.head_joints, "head")
        self._act_idx = list(self._arm_idx) + list(self._grip_idx)
        if cfg.control_waist:
            self._act_idx += list(self._waist_idx)
        self.action_dim = len(self._act_idx)
        # benchmark obs order: L-arm(7) R-arm(7) gripL gripR waist(5) head(3)
        self._obs_state_idx = self._arm_idx + self._grip_idx + self._waist_idx + self._head_idx
        self.state_dim = len(self._obs_state_idx)
        # Per-env instruction state.  The id is RESAMPLED on every reset
        # (uniform over the instance's instructions.json, seeded per env so
        # identical seeds — e.g. a GRPO group — get identical goals) and is
        # sent to the sim, which swaps the success checker to problem{id}:
        # language string and reward stay welded to the same goal.
        self._instr_cache: dict = {}
        self._env_instance_cl = [int(self.cfg.instance_id)] * self.num_envs
        self._env_instr_id = [1] * self.num_envs
        self._env_task_desc = [self._instruction_text(self.cfg.instance_id, 1)] * self.num_envs
        self._instr_rng = np.random.default_rng()
        # Last line, so an exception partway through leaves "no layout"
        # rather than a half-open state.
        self._layout_ready = True

    def _instr_table(self, instance_id: int) -> list:
        """Return cached ``(id, text)`` entries from server metadata.

        Use a synthetic entry before task initialization or when metadata is
        unavailable (timeout or unsupported transport). Application errors propagate
        so the displayed instruction cannot silently disagree with the checker.
        Entries without explicit IDs use their one-based table positions.
        """
        instance_id = int(instance_id)
        if instance_id in self._instr_cache:
            return self._instr_cache[instance_id]
        cfg = self.cfg
        table = []
        # Do not cache on timeout: freezing the synthetic placeholder would
        # make every later reset fail against ids that do not exist.
        transient = False
        if cfg.task_description:
            table = [(1, str(cfg.task_description))]  # explicit override
        elif self._sub_task_name:
            try:
                md = self._server_instructions()  # whole catalogue, cached
                entries = md.get(instance_id) or []
                table = [
                    (int(e["id"]) if "id" in e else i + 1, str(e.get("text") or e.get("instruction") or ""))
                    for i, e in enumerate(entries)
                ]
            except (TimeoutError, NotImplementedError) as exc:
                # Absorb only the two unreadable cases; anything else
                # propagates unchanged.
                transient = isinstance(exc, TimeoutError)
                import warnings  # noqa: PLC0415

                if isinstance(exc, NotImplementedError):
                    reason = (
                        "this transport has no task_metadata verb "
                        "(shm), and the local-disk instruction source "
                        "is gone, so it has no real source at all"
                    )
                else:
                    reason = f"server timed out or is unreachable: {exc!r}"
                # Timeouts are not cached, so track "warn once per instance"
                # here or a slow server floods the log.
                warned = getattr(self, "_instr_warned", None)
                if warned is None:
                    warned = self._instr_warned = set()
                if instance_id not in warned:
                    warned.add(instance_id)
                    warnings.warn(
                        f"sub_task_name={self._sub_task_name!r} instance="
                        f"{instance_id} uses a synthetic placeholder rather "
                        f"than the server's real instruction text ({reason}). "
                        f"Scoring still follows the server's real problem ids, "
                        f"so language and reward may no longer agree.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                table = []
        if not table:
            # The placeholder looks like a real value, so warn: scoring still
            # follows the server's real problem ids.
            import warnings  # noqa: PLC0415

            if not self._sub_task_name:
                why = (
                    "this env has no subtask name yet: call "
                    "env.attach_task('<sub_task_name>') when the server "
                    "already holds the task, or env.init({...}) when it "
                    "does not"
                )
            else:
                why = (
                    f"the server returned an empty instruction table for "
                    f"sub_task_name={self._sub_task_name!r} instance "
                    f"{instance_id}; the name or the id may be wrong"
                )
            warnings.warn(
                f"task_descriptions is a synthetic placeholder "
                f"{(self._sub_task_name or 'task').replace('_', ' ')!r},"
                f"rather than the server's real instruction text: {why}. "
                f"Scoring still follows the server's real problem ids, so "
                f"language and reward may no longer agree.",
                RuntimeWarning,
                stacklevel=2,
            )
            table = [(1, (self._sub_task_name or "task").replace("_", " "))]
        if not transient:
            self._instr_cache[instance_id] = table
        return table

    def _server_instructions(self) -> dict:
        """``{instance_id: [{"id"?, "text"}, ...]}``, fetched once and cached.

        Sourced from the server's ``task_metadata`` rather than local disk.
        ``sub_task_name`` is deliberately not passed: it only makes the server
        compute ``switchable_instance_ids``, which instruction text does not
        need.
        """
        cached = getattr(self, "_server_instr_by_instance", None)
        if cached is not None:
            return cached
        md = self._client.task_metadata()
        entry = (md.get("sub_tasks") or {}).get(self._sub_task_name) or {}
        table = {int(k): v for k, v in (entry.get("instructions_by_instance") or {}).items()}
        self._server_instr_by_instance = table
        return table

    def _instruction_text(self, instance_id: int, instr_id: int) -> str:
        for i, txt in self._instr_table(instance_id):
            if i == int(instr_id):
                return txt
        return self._instr_table(instance_id)[0][1]

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def _joint_pos(self, out) -> np.ndarray:
        """(N, J) joint positions from the step/reset result."""
        js = np.asarray(out.joint_states)
        if js.ndim == 3 and js.shape[-1] == 2:
            return js[..., 0]
        # fallback: interleaved states [pos, vel]
        st = np.asarray(out.states)
        return np.ascontiguousarray(st[:, 0::2])

    def _to_out(self, x: np.ndarray):
        if self.cfg.to_torch and _HAS_TORCH:
            return torch.as_tensor(x).to(self.cfg.device)
        return x

    def _ready_status(self) -> list:
        """``infos["status"]``: every env is READY, always. A reset returns
        with its envs ready to act (``GenieSimHandler.reset``); the key stays
        because RLinf's GenieSim env reads it."""
        return [STATUS_READY] * self.num_envs

    def _assert_state_finite(self, jp: np.ndarray, out) -> None:
        """Reject NaN/Inf joint state before it reaches the observation.

        Placed at the entry of ``_wrap_obs``, after unpacking and before any
        processing, so ``jp`` here is exactly what the server sent and the
        error can say whether the sim produced bad values or the client did.

        The message reports how many non-finite values the other fields of the
        same frame carry: joints and body poses bad together means the solver
        diverged, one channel bad alone points at that channel. The two lead
        in different directions and the distinction cannot be recovered later.

        Cost is one vectorized ``isfinite`` over (N, J), which is negligible.
        """
        bad = ~np.isfinite(jp)  # (N, J)
        bad_rows = bad.any(axis=1)
        # Velocity/body failures can precede a bad joint position. These
        # responses carry every env, including on partial reset, so row ids
        # are already the caller's global environment ids.
        for field in ("joint_states", "body_poses", "object_poses"):
            value = getattr(out, field, None)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.size and arr.ndim and arr.shape[0] == jp.shape[0]:
                bad_rows |= (~np.isfinite(arr)).reshape(jp.shape[0], -1).any(axis=1)
        if not bad_rows.any():
            return
        bad_envs = np.flatnonzero(bad_rows)
        e0 = int(bad_envs[0])
        idx0 = np.flatnonzero(bad[e0]).tolist()
        # The diagnostic path must not fail: an AttributeError raised while
        # reporting an error would bury the real cause. Every field below
        # tolerates being absent.
        names = getattr(self, "_joint_names", None) or []
        shown = ", ".join(f"{i}({names[i]})" if i < len(names) else str(i) for i in idx0[:6]) + (
            " …" if len(idx0) > 6 else ""
        )

        def _nbad(field: str) -> str:
            arr = getattr(out, field, None)
            if arr is None:
                return f"{field}=n/a"
            a = np.asarray(arr)
            return f"{field}={int((~np.isfinite(a)).sum())}/{a.size}"

        others = ", ".join(_nbad(f) for f in ("states", "body_poses", "object_poses", "rewards"))
        _el = getattr(self, "_elapsed", None)
        elapsed = _el.tolist() if _el is not None else "n/a"
        sub = getattr(self, "_sub_task_name", "") or "?"
        raise SimStateNonFiniteError(
            f"joint state from the server holds {int(bad.sum())} non-finite "
            f"values (NaN/Inf): envs {bad_envs.tolist()} affected, env{e0} "
            f"has {len(idx0)}/{jp.shape[1]} bad joints at indices {shown}."
            f"\n  other fields this frame: {others}"
            f"\n  context: sub_task={sub!r} elapsed={elapsed}"
            f"\n  This is SIM-side state, not policy output. The observation "
            f"is already bad here and is refused rather than handed to the "
            f"policy, which would emit NaN actions and be blamed by the "
            f"action guard instead."
            f"\n  Usual cause: the physics solver diverged (contacts or "
            f"constraints failing to converge, velocities exploding, "
            f"positions going NaN). The server's stderr normally carries a "
            f"matching non-finite body-pose error at the same moment."
            f"\n  Recovery: resetting that env usually works, since a reset "
            f"clears the solver's cross-step carryover. Evaluation and "
            f"training loops should catch SimStateNonFiniteError, record the "
            f"episode as failed, and continue rather than ending the run.",
            bad_envs=bad_envs,
        )

    def _wrap_obs(self, out) -> dict:
        """Sim result -> RLinf obs dict (benchmark-aligned states + routed
        images + instruction)."""
        jp = self._joint_pos(out)  # (N, J)
        self._assert_state_finite(jp, out)
        states = jp[:, self._obs_state_idx].astype(np.float32)  # (N, 24)
        obs: dict[str, Any] = {
            "states": self._to_out(states),
            "task_descriptions": list(self._env_task_desc),
        }
        raw_images = getattr(out, "images", None)
        if raw_images is None:
            # A step without images preserves the previous observation frame.
            for key in self.cfg.image_keys:
                obs[key] = None
            return obs
        im = np.asarray(raw_images)  # (N,C,H,W,3)
        n_cams = im.shape[1] if im.ndim == 5 else 0
        # The frame buffer is allocated at the MAX camera height; crop each
        # camera back to its own obs height (head 400, wrist 528) so every camera
        # keeps its benchmark aspect. cam_heights empty => uniform (no crop).
        cam_h = tuple(getattr(self._dims, "cam_heights", ()) or ())
        cropped: dict[str, np.ndarray] = {}
        for key, cam in self.cfg.image_keys.items():
            ci = int(cam)
            if 0 <= ci < n_cams:
                frame = im[:, ci]  # (N, Hmax, W, 3)
                if ci < len(cam_h) and 0 < int(cam_h[ci]) <= frame.shape[1]:
                    frame = frame[:, : int(cam_h[ci])]  # -> (N, Hc, W, 3)
                cropped[key] = np.ascontiguousarray(frame)
        # Sensor imaging generalization -- MUST run after the
        # per-camera crop (each frame's own h/w feeds the dirt-mask cache
        # key) and before self._to_out (numpy in, numpy out; _to_out may
        # already have handed a torch tensor to the caller).
        cropped = self._apply_sensor_randomization(cropped)
        for key, frame in cropped.items():
            obs[key] = self._to_out(frame)
        return obs

    def _reset_sensor_randomization(self, ids: np.ndarray, seed_vec: np.ndarray) -> None:
        """Replace sensor plans and RNGs for the environments being reset.

        Derive independent sensor substreams from the finalized per-env seeds.
        Seed -1 disables sensor augmentation for that environment. Other environments
        keep their current plans and RNG state."""
        if not self.cfg.sensor:
            return
        seed_map = {int(e): int(seed_vec[e]) for e in ids if int(seed_vec[e]) >= 0}
        plans = Sampler(RandomizationSpec(physics={}, sensor=self.cfg.sensor)).sample(seed_map) if seed_map else {}
        for e in ids:
            e = int(e)
            s = int(seed_vec[e])
            if s >= 0:
                plan = plans.get(e)
                self._sensor_cfg[e] = plan.sensor if plan is not None else {}
                self._sensor_rng[e] = substreams(s, e)["sensor"]
            else:
                self._sensor_cfg[e] = {}
                self._sensor_rng[e] = None

    def _apply_sensor_randomization(self, cropped: dict) -> dict:
        """Post-process cropped per-camera frames with sensor imaging
        generalization (noise / dirt occlusion / drop frame).

        ``cropped``: obs key -> (N, H, W, 3) uint8, already cropped to that
        camera's own height (see ``_wrap_obs``).

        Returns ``cropped`` UNCHANGED (same dict, same arrays, no copy) when
        the feature is off (``self.cfg.sensor`` empty, or no env currently
        has an active plan/rng) — this is the fast path that guarantees
        ``np.array_equal(off_output, raw)``: no float roundtrip happens on
        this path at all, not even a no-op one.
        """
        if not self.cfg.sensor or not cropped:
            return cropped
        rngs = [self._sensor_rng.get(e) for e in range(self.num_envs)]
        cfgs = [self._sensor_cfg.get(e) or {} for e in range(self.num_envs)]
        if not any(r is not None for r in rngs) or not any(cfgs):
            return cropped
        return sensor_randomize.apply_camera_image_augmentation(rngs, self._sensor_cache, cropped, cfgs)

    def _map_actions(self, actions) -> np.ndarray:
        """Policy actions (N, action_dim) -> full sim joint targets (N, J).
        Non-controlled joints hold their reset-time targets (benchmark
        semantics: init pose commanded once, PD holds it)."""
        if actions is None:
            return np.array(self._hold_targets, dtype=np.float32)
        a = actions
        if _HAS_TORCH and isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        a = np.asarray(a, dtype=np.float32)
        if a.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"actions shape {a.shape} != ({self.num_envs}, {self.action_dim})")
        full = np.array(self._hold_targets, dtype=np.float32)  # (N, J)
        full[:, self._act_idx] = a
        return full

    # ------------------------------------------------------------------
    # Gym surface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        env_ids=None,
        seeds=None,
        instance_ids=None,
        instruction_ids=None,
        block: bool = True,
    ):
        """Reset all environments or the selected env_ids, then return ready observations.

        seeds is an (N,) array of per-environment randomization seeds; seed instead
        spawns reproducible per-environment seeds. A seed of -1 restores the authored
        snapshot and disables sensor randomization for that environment.

        instance_ids selects instances within the loaded topology/superset; -1 keeps
        the active instance. instruction_ids pins instructions in the selected
        instance's table; -1 requests sampling and invalid IDs raise ValueError.
        Omitting instruction_ids preserves the uniform sampler's RNG draw sequence.

        Sensor plans use the same seeds and remain active until the next reset.
        block is accepted for caller compatibility; reset always completes before
        returning.
        """
        self._require_layout("env.reset()")
        mask = np.zeros((self.num_envs,), dtype=np.float32)
        if env_ids is None:
            mask[:] = 1.0
            ids = np.arange(self.num_envs)
        else:
            ids = (
                np.asarray(env_ids.cpu() if (_HAS_TORCH and isinstance(env_ids, torch.Tensor)) else env_ids)
                .reshape(-1)
                .astype(np.int64)
            )
            mask[ids] = 1.0

        def _scatter(vec: np.ndarray, raw, what: str) -> None:
            """Scatter ``raw`` into ``vec`` by the position of each reset env.

            Length may match ``env_ids`` (one-to-one with the subset) or
            ``num_envs`` (positional over all envs). Any other length raises,
            because filling from index 0 would write a value meant for one env
            into another without reporting anything.
            """
            v = (
                np.asarray(raw.cpu() if (_HAS_TORCH and isinstance(raw, torch.Tensor)) else raw)
                .reshape(-1)
                .astype(np.int64)
            )
            if len(v) == len(ids):
                vec[ids] = v
            elif len(v) == self.num_envs:
                vec[:] = v
            else:
                raise ValueError(
                    f"{what} has length {len(v)}, which is neither the "
                    f"number of envs being reset ({len(ids)}, one-to-one with "
                    f"env_ids) nor num_envs ({self.num_envs}, positional)"
                )

        seed_vec = np.full((self.num_envs,), -1, dtype=np.int64)
        if seeds is not None:
            _scatter(seed_vec, seeds, "seeds")
        elif seed is not None and int(seed) >= 0:
            # seed < 0 means "no seed", matching the -1 sentinel of the array
            # form. Passing it to default_rng directly would raise, making one
            # semantics behave differently between the scalar and array forms.
            spawned = np.random.default_rng(int(seed)).integers(0, 2**31 - 1, size=self.num_envs)
            seed_vec[ids] = spawned[ids]
        inst_vec = np.full((self.num_envs,), -1, dtype=np.int64)
        if instance_ids is not None:
            _scatter(inst_vec, instance_ids, "instance_ids")
        explicit_instr_vec = np.full((self.num_envs,), -1, dtype=np.int64)
        if instruction_ids is not None:
            _scatter(explicit_instr_vec, instruction_ids, "instruction_ids")
        # Resample the instruction on every reset, except for envs named in
        # ``instruction_ids``. The per-env generator is seeded from the reset
        # seed when one is given, so the same seed yields the same layout and
        # the same goal.
        #
        # Validate explicit IDs before sampling; publish local episode state
        # only after the reset succeeds.
        for e in ids:
            explicit_id = int(explicit_instr_vec[e])
            if explicit_id < 0:
                continue
            inst_e = int(inst_vec[e]) if inst_vec[e] >= 0 else self._env_instance_cl[e]
            table = self._instr_table(inst_e)
            if not any(t[0] == explicit_id for t in table):
                raise ValueError(
                    f"instruction_ids[{e}]={explicit_id} not found in "
                    f"instance {inst_e}'s instruction table "
                    f"(valid ids={[t[0] for t in table]})"
                )
        instr_vec = np.full((self.num_envs,), -1, dtype=np.int64)
        next_instances = list(self._env_instance_cl)
        next_instruction_ids = list(self._env_instr_id)
        next_descriptions = list(self._env_task_desc)
        for e in ids:
            inst_e = int(inst_vec[e]) if inst_vec[e] >= 0 else self._env_instance_cl[e]
            inst_vec[e] = inst_e
            table = self._instr_table(inst_e)
            explicit_id = int(explicit_instr_vec[e])
            if explicit_id >= 0:
                iid, txt = next(t for t in table if t[0] == explicit_id)
            else:
                rng = np.random.default_rng(int(seed_vec[e]) + 0x5EED) if seed_vec[e] >= 0 else self._instr_rng
                iid, txt = table[int(rng.integers(0, len(table)))]
            instr_vec[e] = iid
            next_instances[e] = inst_e
            next_instruction_ids[e] = iid
            next_descriptions[e] = txt
        pending = self._unconfirmed_reset_envs
        pending.update(int(e) for e in ids)
        out = self._client.reset(mask, seeds=seed_vec, instances=inst_vec, instructions=instr_vec)
        jp = self._joint_pos(out)  # (N, J)
        # Never let a bad reset frame become a future PD hold command.
        self._assert_state_finite(jp, out)
        self._env_instance_cl = next_instances
        self._env_instr_id = next_instruction_ids
        self._env_task_desc = next_descriptions
        self._reset_sensor_randomization(ids, seed_vec)
        if self._hold_targets is None:
            self._hold_targets = jp.astype(np.float32).copy()
        else:
            self._hold_targets[ids] = jp[ids].astype(np.float32)
        if self._elapsed is None:
            self._elapsed = np.zeros((self.num_envs,), dtype=np.int64)
        self._elapsed[ids] = 0
        self._reset_metrics(ids)
        instr = list(self._env_task_desc)
        reset_success = (
            np.asarray(out.success).reshape(-1).astype(np.int64).tolist()
            if getattr(out, "success", None) is not None
            else [0] * self.num_envs
        )
        obs = self._wrap_obs(out)
        pending.difference_update(int(e) for e in ids)
        return obs, {
            "instruction_ids": np.asarray(self._env_instr_id, dtype=np.int64),
            # Same three keys step() returns (see its docstring for the
            # step_instruction == instruction caveat) — reset counts as a
            # frame too, so callers get a consistent infos shape everywhere.
            "instruction": instr,
            "step_instruction": list(instr),  # own list — see step()'s note
            "is_success": reset_success,
            "status": self._ready_status(),
        }

    def _step_once(self, actions_2d, *, need_images=True):
        """Run one (N, A) action step without resetting mid-chunk.

        need_images accepts a bool or an (N,) mask. Unselected image rows are zero;
        infos["image_envs"] identifies real rows independently of their pixel values.

        infos["instruction"] and infos["step_instruction"] both contain the current
        whole-task description. infos["is_success"] reports task completion separately
        from termination; infos["status"] is STATUS_READY after synchronous reset.
        """
        full = self._map_actions(actions_2d)
        out = self._client.step(full, need_images=need_images)
        obs = self._wrap_obs(out)

        rewards = np.asarray(out.rewards, dtype=np.float32).reshape(-1).copy()
        terminations = np.asarray(out.terminations).reshape(-1).astype(bool).copy()
        truncations = np.asarray(out.truncations).reshape(-1).astype(bool).copy()
        success = (
            np.asarray(out.success).reshape(-1).astype(bool).copy()
            if getattr(out, "success", None) is not None
            else np.zeros_like(terminations)
        )

        self._elapsed += 1
        truncations |= self._elapsed >= self.cfg.max_episode_steps
        rewards = self._shape_reward(rewards, out)

        infos = self._record_metrics(rewards, terminations, success)
        instr = list(self._env_task_desc)
        infos["instruction"] = instr
        # own list object (not the same reference as "instruction"): keeps the
        # two keys independently mutable by a caller without cross-pollution,
        # and keeps the equality in tests a real check against a shared
        # oracle rather than an "is this the same object" tautology.
        infos["step_instruction"] = list(instr)  # see docstring: == instruction for now
        infos["is_success"] = success.astype(np.int64).tolist()
        infos["status"] = self._ready_status()
        if self.cfg.ignore_terminations:
            infos["episode"]["success_at_end"] = self._to_out(terminations.copy())
            terminations[:] = False
        # Staged scores travel only when the transport result actually
        # carries a snapshot (zmq with evaluation on). Otherwise the attribute
        # is absent and the key is omitted. This must land before
        # `_handle_auto_reset` deep-copies infos, so a same-step auto-reset
        # carries it into `final_info`.
        stage_scores = getattr(out, "stage_scores", None)
        if stage_scores is not None:
            infos["stage_scores"] = stage_scores
        # Per-env image mask (zmq): the indices of the rows for which the
        # server actually returned an image. Omitted when every row came back,
        # or when no images were requested.
        image_envs = getattr(out, "image_envs", None)
        if image_envs is not None:
            infos["image_envs"] = list(image_envs)

        return (obs, self._to_out(rewards), self._to_out(terminations), self._to_out(truncations), infos)

    #: Whether the server recognizes ``step_chunk``. ``None`` = not probed
    #: yet; once an old server is found to reply ``unknown verb``, this is
    #: pinned to ``False`` so we stop wasting a try on every chunk.
    _step_chunk_supported = None

    def _can_batch_chunk(self, chunk: int, return_per_step_obs: bool) -> bool:
        """Whether this chunk can run as "one round trip".

        Five cases must fall back to per-substep:

        * ``RLINF_STEP_CHUNK=0`` -- explicitly disabled (kept for A/B
          testing and as a kill switch if something goes wrong);
        * ``chunk == 1`` -- already only one round trip, batching saves
          nothing;
        * ``return_per_step_obs`` -- the caller wants every frame's image,
          but the batched path only renders the last substep;
        * ``_shape_reward`` overridden by a subclass -- it takes the whole
          ``out`` per substep (e.g. reading ``out.body_poses`` for dense
          shaping), but the batched path only has the last substep's
          ``out`` on hand. Better to not batch than to quietly feed it the
          wrong frame;
        * **the client itself has no ``step_chunk``** -- ``step_chunk`` is
          specific to the zmq transport; the shm backend's client has no
          such method, and any third-party client might not either. This
          must be checked up front with ``hasattr``: without that check it
          is a bare ``AttributeError`` crashing straight through, not a
          graceful fallback (the ``unknown verb`` fallback below only
          covers the "server is too old" case).
        """
        if os.environ.get("RLINF_STEP_CHUNK", "1") == "0":
            return False
        if chunk <= 1 or return_per_step_obs:
            return False
        if type(self)._shape_reward is not GenieSimVecEnv._shape_reward:
            return False
        if not callable(getattr(self._client, "step_chunk", None)):
            return False
        # This does **not** need a RESETTING guard. An async-reset env would
        # need one: the batched path only has the last substep's ``out``,
        # cannot pin down a RESETTING->READY edge, and ``_hold_targets``
        # would end up reading poses that are several steps stale. But this
        # env has only one kind of reset -- blocking, and it can already act
        # by the time ``reset`` returns (see ``reset``'s "``block`` is
        # accepted and ignored" and ``_ready_status`` always returning all
        # READY), so the RESETTING state does not exist here. Do not add
        # such a guard: ``self._env_status``/``STATUS_RESETTING`` are not
        # attributes of this class, and referencing them raises
        # ``AttributeError``.
        return self._step_chunk_supported is not False

    def _chunk_via_substeps(self, a, need_images, return_per_step_obs):
        """Runs a chunk one substep at a time: one ZMQ round trip per
        substep."""
        chunk = a.shape[1]
        obs_list, rew, terms, truncs, succ = [], [], [], [], []
        info = None
        for i in range(chunk):
            last = i == chunk - 1
            # ``need_images`` may be a per-env mask, so it cannot be combined
            # with ``and``: that raises on an ndarray and silently discards the
            # mask for a list. Branch explicitly.
            want_img = need_images if (return_per_step_obs or last) else False
            obs, r, te, tr, info = self._step_once(a[:, i], need_images=want_img)
            if return_per_step_obs or last:
                obs_list.append(obs)
            rew.append(_to_host_array(r))
            terms.append(_to_host_array(te))
            truncs.append(_to_host_array(tr))
            succ.append(np.asarray(info["is_success"], dtype=bool))
        return obs_list, rew, terms, truncs, succ, info

    def _chunk_via_one_roundtrip(self, a, need_images):
        """Runs the whole chunk as one ZMQ round trip, then backfills the
        per-substep bookkeeping locally once it returns.

        The only difference from the per-substep path is **how many round
        trips** happen: each substep's reward / termination / truncation /
        success is still reconstructed one at a time and still goes
        through ``_record_metrics`` and ``_elapsed`` individually, so
        episode stats, the step cap, and auto-reset's trigger condition
        are all byte-for-byte identical between the two paths.

        Falls back to per-substep when the server is too old (does not
        recognize ``step_chunk``), and remembers that conclusion so it
        never retries. **But a mismatched reply format does not fall
        back** -- that means the two sides disagree about the protocol,
        and quietly running per-substep anyway would just make every
        substep's reward wrong.
        """
        chunk = a.shape[1]
        # (chunk, N, A): the server steps through substeps along axis 0.
        acts = np.ascontiguousarray(np.stack([self._map_actions(a[:, i]) for i in range(chunk)]))
        try:
            out = self._client.step_chunk(acts, need_images=need_images)
        except RuntimeError as exc:
            if "unknown verb" not in str(exc):
                raise
            self._step_chunk_supported = False
            return self._chunk_via_substeps(a, need_images, False)
        self._step_chunk_supported = True

        scal = getattr(out, "chunk_scalars", None)
        expected = (4, chunk, self.num_envs)
        if scal is None or tuple(scal.shape) != expected:
            raise RuntimeError(
                "step_chunk reply carries no usable chunk_scalars field "
                f"(got {None if scal is None else tuple(scal.shape)}, "
                f"expected {expected}); client and server disagree on the "
                "wire format"
            )

        rew, terms, truncs, succ = [], [], [], []
        info = None
        for i in range(chunk):
            rewards = np.asarray(scal[0, i], dtype=np.float32).copy()
            terminations = scal[1, i].astype(bool)
            truncations = scal[2, i].astype(bool)
            success = scal[3, i].astype(bool)
            succ.append(success)
            self._elapsed += 1
            truncations = truncations | (self._elapsed >= self.cfg.max_episode_steps)
            info = self._record_metrics(rewards, terminations, success)
            if self.cfg.ignore_terminations:
                info["episode"]["success_at_end"] = self._to_out(terminations.copy())
                terminations = np.zeros_like(terminations)
            rew.append(rewards)
            terms.append(terminations)
            truncs.append(truncations)

        # The non-episode parts of info only keep the last substep's -- in
        # the per-substep path ``info`` is overwritten every substep too,
        # so what's left at the end is also the last substep's; both paths
        # agree.
        instr = list(self._env_task_desc)
        info["instruction"] = instr
        info["step_instruction"] = list(instr)
        info["is_success"] = scal[3, -1].astype(np.int64).tolist()
        # Byte-for-byte identical to the per-substep path (see
        # ``_step_once``): this branch only has blocking reset, no
        # RESETTING, and ``_ready_status()`` always returns all READY.
        # Do **not** update ``_hold_targets`` here: this branch's
        # ``reset()`` has already seeded it directly with the post-reset
        # joint positions (see ``self._hold_targets[ids] = jp[ids]`` in
        # ``reset``).
        info["status"] = self._ready_status()
        stage_scores = getattr(out, "stage_scores", None)
        if stage_scores is not None:
            info["stage_scores"] = stage_scores
        image_envs = getattr(out, "image_envs", None)
        if image_envs is not None:
            info["image_envs"] = list(image_envs)
        return [self._wrap_obs(out)], rew, terms, truncs, succ, info

    def step(self, actions, auto_reset: bool = True, need_images=True, return_per_step_obs: bool = False):
        """Run one action chunk. ``actions`` is always ``(N, chunk, A)``; a
        single step is chunk=1.

        Chunk semantics, matching isaaclab: no reset happens inside a chunk; a
        mid-chunk done is latched and reported in the last column only;
        ``auto_reset``, if enabled, fires once at the chunk end; an env that
        finished mid-chunk keeps simulating its remaining sub-steps, and since
        the verdict is latched this cannot change its score.

        ``rewards`` / ``terminations`` / ``truncations`` are always
        ``(N, chunk)``: per-step rewards are not summed away, because in-chunk
        discount accounting and stage-score timing both need them.

        By default only the chunk's last frame is rendered and returned, which
        cuts a chunked rollout's render cost to 1/chunk. Pass
        ``return_per_step_obs=True`` for every frame.
        """
        self._require_layout("env.step()")
        pending = self._unconfirmed_reset_envs
        if pending:
            raise RuntimeError(
                f"env.step(): reset is unconfirmed for envs {sorted(pending)}; "
                "retry env.reset(...) for those envs before stepping."
            )
        a, was_2d = self._as_chunk_actions(actions)  # (N, chunk, A)
        chunk = a.shape[1]
        if self._can_batch_chunk(chunk, return_per_step_obs):
            obs_list, rew, terms, truncs, succ, info = self._chunk_via_one_roundtrip(a, need_images)
        else:
            obs_list, rew, terms, truncs, succ, info = self._chunk_via_substeps(a, need_images, return_per_step_obs)
        chunk_rewards = np.stack(rew, axis=1)  # (N, chunk)
        chunk_success = np.stack(succ, axis=1).astype(bool)  # (N, chunk)
        raw_terms = np.stack(terms, axis=1).astype(bool)
        raw_truncs = np.stack(truncs, axis=1).astype(bool)
        past_terms, past_truncs = raw_terms.any(axis=1), raw_truncs.any(axis=1)

        final_obs = obs_list if return_per_step_obs else obs_list[-1]
        if (past_terms | past_truncs).any() and auto_reset and self.cfg.auto_reset:
            tail, info = self._handle_auto_reset(
                past_terms | past_truncs, obs_list[-1] if return_per_step_obs else final_obs, info
            )
            if return_per_step_obs:
                final_obs[-1] = tail
            else:
                final_obs = tail

        if self.cfg.auto_reset or self.cfg.ignore_terminations:
            chunk_terms = np.zeros_like(raw_terms)
            chunk_terms[:, -1] = past_terms
            chunk_truncs = np.zeros_like(raw_truncs)
            chunk_truncs[:, -1] = past_truncs
        else:
            chunk_terms, chunk_truncs = raw_terms, raw_truncs
        if was_2d:
            # Migration contract: 2D in, old shape out. Squeezing the chunk
            # axis keeps every field equivalent to the pre-change behaviour, so
            # callers still passing 2D do not silently receive (N, 1).
            chunk_rewards = chunk_rewards[:, 0]
            chunk_terms, chunk_truncs = chunk_terms[:, 0], chunk_truncs[:, 0]
        # Per-substep success, ``(N, chunk)``. ``info["is_success"]`` only
        # carries the last substep's single value, so an env that
        # succeeded mid-chunk is no longer visible by the last substep,
        # yet the caller (rlinf's ``GenieSimEnv``) needs to know **which
        # step** it succeeded on to do per-substep bookkeeping.
        #
        # This must be attached here, not right after the loop ends:
        # ``_handle_auto_reset`` replaces ``info`` wholesale with the
        # post-reset one, so attaching it earlier would get overwritten
        # (the old one goes into ``final_info`` along with everything
        # else, but the top level describes the chunk that just ran, and
        # it needs to carry this).
        info["chunk_success"] = chunk_success
        return (final_obs, self._to_out(chunk_rewards), self._to_out(chunk_terms), self._to_out(chunk_truncs), info)

    def _as_chunk_actions(self, actions) -> tuple[np.ndarray, bool]:
        """Normalise to ``(N, chunk, A)`` and report whether the input used
        the migration-era 2D shape. ``was_2d`` decides whether ``step``
        squeezes the chunk axis out of its returns."""
        a = actions
        if _HAS_TORCH and isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        a = np.asarray(a, dtype=np.float32)
        was_2d = a.ndim == 2
        if was_2d:
            import warnings  # noqa: PLC0415

            warnings.warn(
                "step() actions are always (N, chunk, A). A 2D (N, A) input "
                "is promoted to (N, 1, A) during migration, with the returns "
                "squeezed back to match the old behaviour. This becomes a "
                "hard error after one release cycle.",
                DeprecationWarning,
                stacklevel=3,
            )
            a = a[:, None, :]
        if a.ndim != 3 or a.shape[0] != self.num_envs or a.shape[2] != self.action_dim:
            raise ValueError(f"actions shape {a.shape} != ({self.num_envs}, chunk, " f"{self.action_dim})")
        # Non-finite actions must be stopped here rather than fed into
        # physics. The damage is delayed and unrecoverable: the observation of
        # the step itself still looks normal, several clean steps later
        # everything is NaN, and no reset brings it back. Because symptom and
        # cause are several steps apart, catching it at the entry is the only
        # place where the error can name what happened.
        if not np.isfinite(a).all():
            bad = np.argwhere(~np.isfinite(a))
            e, c, d = (int(x) for x in bad[0])
            raise ValueError(
                f"actions contain {len(bad)} non-finite values (NaN/Inf); "
                f"the first is at env={e} chunk_step={c} dim={d} "
                f"(value {a[e, c, d]}). Refusing to send: a non-finite action "
                f"drives the physics solver into a NaN state that a reset "
                f"cannot recover, with no error reported anywhere. Check the "
                f"policy output — the usual causes are early-training "
                f"divergence, too large a learning rate, or mixed-precision "
                f"overflow."
            )
        return a, was_2d

    def chunk_step(self, chunk_actions):
        """Thin alias for ``step``, kept for one release cycle. Returns
        exactly what ``step`` returns; an alias must not create a second set
        of semantics."""
        import warnings  # noqa: PLC0415

        # This alias's return TYPE changed too: obs and infos are now a
        # single dict rather than a per-sub-step list, so an out-of-tree
        # `obs_list[-1]` hits KeyError(-1). Warn, or there is nothing to point
        # at.
        warnings.warn(
            "chunk_step() is deprecated; call step() directly. The two "
            "return exactly the same thing, and the shapes have changed: obs "
            "and infos are now a single dict for the chunk's last frame "
            "(no longer a per-sub-step list, so `obs[-1]` gives "
            "KeyError(-1)), and rewards/terminations/truncations are "
            "(N, chunk). For per-frame observations pass "
            "step(..., return_per_step_obs=True).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.step(chunk_actions)

    def _handle_auto_reset(self, dones: np.ndarray, final_obs, infos):
        final_obs = copy.deepcopy(final_obs)
        final_info = copy.deepcopy(infos)
        ids = np.nonzero(dones)[0]
        obs, infos = self.reset(env_ids=ids)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_observation"] = self._to_out(dones.copy())
        infos["_final_info"] = self._to_out(dones.copy())
        infos["_elapsed_steps"] = self._to_out(dones.copy())
        return obs, infos

    # ------------------------------------------------------------------
    # Hooks / metrics / RLinf accessories
    # ------------------------------------------------------------------

    def _shape_reward(self, rewards: np.ndarray, out) -> np.ndarray:
        """Subclass hook: dense shaping from sim facts (e.g. ``out.body_poses``).
        Base = the sim's sparse task signal, untouched."""
        return rewards

    def _init_metrics(self) -> None:
        n = self.num_envs
        self._m_return = np.zeros(n, dtype=np.float64)
        self._m_length = np.zeros(n, dtype=np.int64)
        self._m_success_once = np.zeros(n, dtype=bool)

    def _reset_metrics(self, env_ids) -> None:
        self._m_return[env_ids] = 0.0
        self._m_length[env_ids] = 0
        self._m_success_once[env_ids] = False

    def _record_metrics(self, rewards, terminations, success) -> dict:
        self._m_return += rewards
        self._m_length += 1
        self._m_success_once |= success
        ep = {
            "return": self._to_out(self._m_return.copy()),
            "length": self._to_out(self._m_length.copy()),
            "success_once": self._to_out(self._m_success_once.copy()),
            "success_at_end": self._to_out(success.copy()),
        }
        return {"episode": ep}

    @property
    def elapsed_steps(self):
        # `_elapsed` is `None` right after construction AND right after
        # `init()` (a task switch resets it, same as a fresh build) until
        # the first `reset()` populates it -- a caller querying elapsed
        # steps in that window (e.g. right after switching tasks, before
        # its first reset) means "no steps yet", not a crash. Having no task
        # at all is a different case: num_envs is unknown, so the array would
        # have length 0 and the wrong shape. _require_layout catches that.
        self._require_layout("env.elapsed_steps")
        if self._elapsed is None:
            return self._to_out(np.zeros((self.num_envs,), dtype=np.int64))
        return self._to_out(self._elapsed.copy())

    @property
    def task_descriptions(self) -> list:
        """Per-env current global instruction text — the same list backing
        ``obs["task_descriptions"]`` and ``infos["instruction"]``
        (``_env_task_desc`` is the single source of truth; this is just a
        public accessor for it). With no active task this raises by name via
        ``_require_layout`` rather than an ``AttributeError``."""
        self._require_layout("env.task_descriptions")
        return list(self._env_task_desc)

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, v: bool) -> None:
        self._is_start = bool(v)

    @property
    def info_logging_keys(self):
        return ["return", "length", "success_once", "success_at_end"]

    def update_reset_state_ids(self) -> None:
        """Reset-state curriculum hook (RLinf calls this after each rollout).
        Per-reset instance/pose selection is a sim-side extension (reset verb
        parameter) — until then every reset restores the build-time state."""
        return


__all__ = ["GenieSimVecEnv", "GenieSimVecEnvConfig", "SimStateNonFiniteError"]
