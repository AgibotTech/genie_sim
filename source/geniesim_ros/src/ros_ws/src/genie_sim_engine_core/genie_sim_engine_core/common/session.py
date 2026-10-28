# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared session bootstrap for physics-engine entry points.

SimpleLogger — parameterised prefix logger.

EngineSession — owns physics-params loading through ``set_topology``.
This base class is **backend-agnostic**: it does not import any
particular backend's USD helpers.  Two narrow hooks
(:meth:`EngineSession._wait_for_manifest` and
:meth:`EngineSession._snapshot_body_transforms`) cover the two places
where this loop must read from a USD stage; backend distributions
provide subclasses that implement them.

The companion file ``genie_sim_engine_newton/common/session.py`` ships
``GenieSimEngineSession`` for the in-tree backends; external backends
(for example a MuJoCo-only adapter) subclass ``EngineSession`` directly.

Entry points do:
  1. Bootstrap (SimulationApp or kernel warmup).
  2. Construct concrete EngineSession subclass(node_name, params,
     simulation_app, physics_engine, logger).
  3. Log the engine-specific "ready" message using session attributes.
  4. Optionally append to session.post_step_hooks.
  5. Call session.startup(headless).
  6. Define a render_hook closure (entry-point-specific).
  7. Call session.run(render_hook, exit_check).
"""

from __future__ import annotations

import json
import os
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

from genie_sim_engine_core.common.loop import EngineRunLoop
from genie_sim_engine_core.common.pacing import DEFAULT_STEP_MODE
from genie_sim_engine_core.common.params import EngineNodeParams, load_physics_params, parse_init_joint_pos
from genie_sim_engine_core.common.scene_config import (
    parse_convert_joints_to_fixed,
    parse_pin_base_mode,
    parse_pin_base_to_world,
)
from genie_sim_engine_core.engine.base import PhysicsEngine, audit_lifecycle
from genie_sim_engine_core.engine.publish_plan import PublishPlan

# ---------------------------------------------------------------------------
# Launcher physics_* → engine kwargs forwarding
# ---------------------------------------------------------------------------
#
# Contract: a launcher YAML can add a new physics knob without touching this
# module, provided the backend actually accepts it (either by naming it on
# ``__init__``, or by taking ``**kwargs``). Session-owned physics_* keys —
# ROS/loop concerns that were never meant to reach the backend constructor —
# live in the allowlist below and are excluded before forwarding.
#
# Anything dropped is LOGGED (not silently ignored) so a typo like
# ``physics_solver_iteratons`` doesn't vanish without a trace.

#: physics_* launcher keys that this module handles itself; NOT engine kwargs.
#: When adding a new physics_* key: if the session/loop handles it, add here;
#: if the backend handles it, add to the backend's __init__. If it truly needs
#: to reach both, thread it explicitly via ``engine_kwargs`` at the call site
#: rather than relying on the prefix forwarder.
_SESSION_ONLY_PHYSICS_KEYS: frozenset = frozenset(
    {
        "physics_engine",  # backend name (passed positionally)
        "physics_engine_visualizer",  # which visualizer to spawn (ROS-side)
        "physics_params_file",  # YAML path (load_physics_params)
        "physics_hz",  # loop cadence (EngineNodeParams)
        "physics_solver",  # sentinel; each backend resolves internally
        "physics_solver_substep",  # threaded via engine_kwargs above
        "physics_solver_iterations",  # threaded via engine_kwargs above
        "physics_solver_mass_matrix_interval",  # threaded via engine_kwargs above
    }
)


def _forward_physics_launcher_params(
    engine_kwargs: dict,
    params: dict,
    physics_engine: str,
    logger,
) -> None:
    """Forward launcher ``physics_*`` params into ``engine_kwargs`` in place.

    Filters against the backend's declared ``__init__`` signature (via
    :meth:`PhysicsEngine.backend_abi`) when the backend does NOT accept
    ``**kwargs``. Backends that DO accept ``**kwargs`` receive every
    physics_* key that isn't in :data:`_SESSION_ONLY_PHYSICS_KEYS` — that
    is the point of ``**kwargs``.

    Keys already present in ``engine_kwargs`` (explicitly threaded above
    the call site) win: the launcher never clobbers an explicit assignment.

    Dropped keys are logged so a launcher-YAML typo is loud, not silent.
    Falls back to the pre-ABI broad forward when reflection is unavailable
    (unregistered backends, custom subclasses without a proper signature).
    """
    try:
        abi = PhysicsEngine.backend_abi(physics_engine)
        accepted: set = set(abi.accepted_names)
        accepts_kwargs: bool = abi.accepts_extra_kwargs
    except Exception:  # noqa: BLE001 — unregistered backend: keep the old broad rule
        accepted, accepts_kwargs = None, True

    dropped: list = []
    for k, v in params.items():
        if not (isinstance(k, str) and k.startswith("physics_")):
            continue
        if k in _SESSION_ONLY_PHYSICS_KEYS or k in engine_kwargs:
            continue
        if accepted is not None and not accepts_kwargs and k not in accepted:
            dropped.append(k)
            continue
        engine_kwargs[k] = v

    if dropped and logger is not None:
        logger.warn(
            f"[session] dropped {len(dropped)} launcher physics_* param(s) not "
            f"accepted by backend {physics_engine!r}: {sorted(dropped)}. "
            f"Rename, add to the backend's __init__, or extend "
            f"_SESSION_ONLY_PHYSICS_KEYS in common/session.py."
        )


# ---------------------------------------------------------------------------
# scene_yaml_defaults — launcher-level base beneath every scene yaml
# ---------------------------------------------------------------------------
#
# A launcher YAML may carry a ``scene_yaml_defaults:`` block whose shape is a
# SCENE yaml, not a params block:
#
#     genie_sim_engine:
#       ros__parameters:
#         scene_yaml_defaults:
#           newton:
#             mjc_contact:
#               robot: {friction: [1.5, 0.5, 0.2]}
#
# It is deep-merged UNDERNEATH the scene yaml the manifest names, so a fleet
# operator sets a profile once and any individual scene can still override one
# key of it without restating the rest.  That direction is the whole point:
# launcher-wins would make the scene yaml unable to disagree with the launcher,
# which is the opposite of how ``dds_hz`` and friends behave, and those ARE
# per-run policy.  This block is per-FLEET default.
#
# Merge is per-key and recursive on mappings only.  A LIST is a leaf: a scene
# naming ``friction: [1.0, 1.0, 0.0001]`` means that triplet, not that triplet
# element-wise merged with the launcher's.


def _deep_merge_under(base: dict, over: dict) -> dict:
    """``over`` wins; mappings recurse, everything else is replaced whole."""
    out = dict(base)
    for k, v in (over or {}).items():
        prev = out.get(k)
        if isinstance(prev, dict) and isinstance(v, dict):
            out[k] = _deep_merge_under(prev, v)
        else:
            out[k] = v
    return out


def _read_scene_yaml_defaults(params: dict, logger) -> dict:
    """Pull ``scene_yaml_defaults`` out of the node params as a real mapping.

    ``_merge_params_file`` hands mappings through intact, so the params-file
    route arrives as a ``dict``.  A ``-p scene_yaml_defaults:=...`` CLI
    override arrives as a string, which is parsed as YAML — the same text the
    launcher file would have held.  Anything else warns and is ignored rather
    than silently applying nothing.
    """
    raw = (params or {}).get("scene_yaml_defaults")
    if raw in (None, "", {}):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            import yaml  # noqa: PLC0415

            parsed = yaml.safe_load(raw)
        except Exception as exc:  # noqa: BLE001
            parsed = None
            if logger is not None:
                logger.warn(f"scene_yaml_defaults: not parseable as YAML ({exc}); ignoring")
        if isinstance(parsed, dict):
            return parsed
        if parsed is not None and logger is not None:
            logger.warn(f"scene_yaml_defaults: expected a mapping, got {type(parsed).__name__}; ignoring")
        return {}
    if logger is not None:
        logger.warn(f"scene_yaml_defaults: expected a mapping, got {type(raw).__name__}; ignoring")
    return {}


def _resolve_physics_params_file(requested: str, logger) -> str:
    """Resolve the ``physics_params_file`` ROS parameter to a real path.

    Three accepted forms, one key — see ``geniesim_ros/AGENTS.md``,
    "one knob, one route":

    * **empty** -> the shipped base profile,
      ``genie_sim_engine_core/config/physics_params.yaml``.  This is the
      historical behaviour and stays the default so a launcher that says
      nothing keeps working.
    * **a bare filename** (``physics_params_rlinf.yaml``) -> resolved
      against that same ``config/`` directory.  This is what launcher
      YAMLs should write: an absolute path is not portable across an
      install tree, a source tree and another machine, and until every
      launcher could name a profile portably, none of them named one at
      all — the whole profile mechanism sat with zero callers while two
      tuned profiles shipped next to each other.
    * **an absolute path** -> used as given, for a profile kept outside
      the package.

    Resolution is logged unconditionally.  It used to be silent, which is
    precisely how a dead selector survived: nothing in a bring-up log
    distinguished "loaded the profile you asked for" from "fell back".
    """
    share_config = ""
    try:
        from ament_index_python.packages import get_package_share_directory

        share_config = os.path.join(get_package_share_directory("genie_sim_engine_core"), "config")
    except Exception as exc:  # noqa: BLE001 - ament missing is not fatal here
        logger.warn(f"could not resolve the genie_sim_engine_core config dir: {exc}")

    requested = (requested or "").strip()
    if not requested:
        if not share_config:
            return ""
        resolved = os.path.join(share_config, "physics_params.yaml")
        logger.info(f"physics_params_file unset — using the base profile {resolved}")
        return resolved

    if os.path.isabs(requested):
        resolved = requested
    elif share_config:
        resolved = os.path.join(share_config, requested)
    else:
        resolved = requested

    if not os.path.isfile(resolved):
        # Loud, because the consequence is silent: ``load_physics_params``
        # falls back to built-in defaults, so a typo'd profile name runs a
        # DIFFERENT set of physics than the launcher asked for and nothing
        # downstream can tell.
        logger.error(
            f"physics_params_file: {requested!r} resolves to {resolved}, which does not exist — "
            f"falling back to built-in defaults, NOT to the base profile. Physics will not match "
            f"the launcher's intent. Fix the name or ship the file."
        )
        return ""

    logger.info(f"physics_params_file: {requested!r} -> {resolved}")
    return resolved


class SimpleLogger:
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def debug(self, m: str) -> None:
        print(f"[{self._prefix}] DEBUG: {m}", flush=True)

    def info(self, m: str) -> None:
        print(f"[{self._prefix}] {m}", flush=True)

    def warn(self, m: str) -> None:
        print(f"[{self._prefix}] WARN: {m}", flush=True)

    # Python's ``logging.Logger`` exposes both ``warn`` (deprecated)
    # and ``warning`` (standard).  Modern code paths (and everything
    # ported from external libs) call ``.warning``; alias here so
    # callers can use either without knowing which logger flavour
    # they received.
    def warning(self, m: str) -> None:
        self.warn(m)

    def error(self, m: str) -> None:
        print(f"[{self._prefix}] ERROR: {m}", flush=True)


class EngineSession:
    """Bootstrap a physics engine from a params dict + manifest.

    After construction: ``core`` is initialised, sim is built, topology
    is set.  Call ``startup(headless)`` then ``run(render_hook,
    exit_check)`` to enter the loop.

    Subclass responsibilities
    -------------------------
    Two abstract hooks isolate the USD-stage helpers that today only
    one backend distribution ships:

    * :meth:`_wait_for_manifest` — block until ``manifest_path`` exists
      and is parseable.
    * :meth:`_snapshot_body_transforms` — read body world poses out of
      the live USD stage owned by ``sim``.

    A concrete subclass provides these (typically by delegating to
    ``genie_sim_engine_isaacsim.kit.stage``, or to a backend-native USD
    reader elsewhere).

    Public attributes
    -----------------
    core            : genie_sim_engine_py module (pybind .so)
    logger          : the logger passed in
    sim             : PhysicsEngine instance
    physics_hz      : float
    render_hz       : float
    realtime_factor : float
    dt              : 1.0 / physics_hz
    scene_usda      : resolved path, for log messages
    post_step_hooks : list of Callable[[float], None]
    """

    def __init__(
        self,
        node_name: str,
        params: dict,
        simulation_app: Any,
        physics_engine: str,
        logger: Any,
    ) -> None:
        import genie_sim_engine_py as _core  # deferred — .so requires ROS env

        self.core = _core
        self.logger = logger
        self.post_step_hooks: list = []

        # --- typed params (IsaacLab @configclass pattern) ---
        ep = EngineNodeParams.from_dict(params)

        physics_params_file = _resolve_physics_params_file(ep.physics_params_file, logger)
        phys = load_physics_params(physics_params_file, logger)

        init_joint_pos = parse_init_joint_pos(ep.init_joint_pos_json, logger)
        if init_joint_pos:
            logger.info(
                f"init_joint_pos: {len(init_joint_pos)} joint(s) requested "
                f"(deg for revolute, m for prismatic) — will apply at articulation init"
            )

        render_hz = ep.render_hz if ep.render_hz > 0.0 else float(phys.render_target_hz)
        if render_hz <= 0.0:
            logger.warn(f"render_hz must be > 0; using {phys.render_target_hz}")
            render_hz = float(phys.render_target_hz)

        # --- manifest ---
        if not ep.stage_manifest:
            raise RuntimeError("parameter 'stage_manifest' is required")
        manifest_path = str(Path(ep.stage_manifest).resolve())

        self._wait_for_manifest(manifest_path, logger)
        # Typed manifest read — Manifest.read validates the schema
        # version and resolves every path field against ``base_path``,
        # replacing the ad-hoc ``_abs`` helper this function used to
        # inline. If the stage dir is stale (older schema), the launcher
        # gets a clear ``SchemaVersionError`` diagnostic and stops
        # rather than diving into a KeyError deep in an engine adapter.
        from genie_sim_engine_core.schema import (
            Manifest,
            SceneConfig,
            SchemaError,
            SchemaVersionError,
        )

        try:
            manifest = Manifest.read(Path(manifest_path))
        except SchemaVersionError as exc:
            logger.error(
                f"manifest.json schema version mismatch: {exc}. "
                "Re-run the assemble pipeline (delete the stage dir if needed)."
            )
            raise
        except SchemaError as exc:
            logger.error(f"manifest.json unreadable: {exc}")
            raise

        # rlinf fallback: init_joint_pos travels in the manifest (rather than
        # the ROS param) when the caller bypasses ros2 launch args. The typed
        # Manifest above doesn't carry this field (core-owned schema), so peek
        # the raw JSON — already proven parseable by the successful read above.
        # parse_init_joint_pos accepts a dict, so feed it through the same path.
        if not init_joint_pos:
            with open(manifest_path, "r") as f:
                manifest_raw = json.load(f)
            if manifest_raw.get("init_joint_pos"):
                init_joint_pos = parse_init_joint_pos(manifest_raw.get("init_joint_pos"), logger)
                if init_joint_pos:
                    logger.info(f"init_joint_pos: {len(init_joint_pos)} joint(s) from manifest")

        robot_prefix = manifest.robot_prefix or "genie"
        scene_usda = str(manifest.scene_usda) if manifest.scene_usda else ""
        robot_usda = str(manifest.robot_usda) if manifest.robot_usda else ""
        render_layer_usda = str(manifest.render_layer_usda) if manifest.render_layer_usda else ""
        robot_from_urdf = manifest.robot_from_urdf

        # --- scene yaml ---
        scene_yaml_path = str(manifest.scene_yaml) if manifest.scene_yaml else ""
        scene_cfg: dict = {}
        scene_typed = None
        if scene_yaml_path and os.path.isfile(scene_yaml_path):
            try:
                scene_typed = SceneConfig.read(Path(scene_yaml_path))
                scene_cfg = scene_typed.to_dict()
            except SchemaError as exc:
                logger.warn(f"failed to read scene_yaml {scene_yaml_path}: {exc} — using defaults")
        elif scene_yaml_path:
            logger.warn(f"scene_yaml {scene_yaml_path} not found — using defaults")
        else:
            logger.warn("manifest missing 'scene_yaml' field — using defaults for dynamic params")

        # Launcher-level base, merged BENEATH whatever the scene said.  Both
        # views are rebuilt from the merged dict so the typed accessors below
        # and ``engine_kwargs["scene_cfg"]`` cannot disagree about what the
        # active configuration is.
        scene_defaults = _read_scene_yaml_defaults(params, logger)
        if scene_defaults:
            scene_cfg = _deep_merge_under(scene_defaults, scene_cfg)
            try:
                scene_typed = SceneConfig.from_dict(
                    scene_cfg,
                    source_path=Path(scene_yaml_path) if scene_yaml_path else None,
                )
            except SchemaError as exc:
                # The dict view still carries the merge; only the typed one is
                # lost, and the fall-throughs below already handle that.
                scene_typed = None
                logger.warn(f"scene_yaml_defaults merged but the result failed the scene schema: {exc}")
            logger.info(
                f"scene_yaml_defaults: merged {len(scene_defaults)} launcher block(s) "
                f"({', '.join(sorted(map(str, scene_defaults)))}) beneath "
                f"{scene_yaml_path or '<no scene yaml>'}; the scene's own keys win"
            )

        # Two independent runtime-behavior flags from the scene yaml.
        # Prefer the typed accessors when the parse succeeded; fall
        # through to the dict-based parsers otherwise so scenes that
        # currently fail the schema (unlikely — all 40 real scenes
        # parse) still see the same behaviour as pre-schema code.
        if scene_typed is not None:
            pin_base_to_world = scene_typed.robot.robot_source.pin_base_to_world
            pin_base_mode = scene_typed.robot.robot_source.pin_base_mode
            convert_joints_to_fixed = list(scene_typed.robot.robot_source.convert_joints_to_fixed)
        else:
            _robot_section = scene_cfg.get("robot") if isinstance(scene_cfg.get("robot"), dict) else {}
            pin_base_to_world = parse_pin_base_to_world(_robot_section)
            pin_base_mode = parse_pin_base_mode(_robot_section)
            convert_joints_to_fixed = parse_convert_joints_to_fixed(_robot_section)

        # --- pybind init ---
        # base_frame: "base_footprint" asks the C++ publish_odom to split
        # base_link's pose into ``odom -> base_footprint`` (ground-projected)
        # plus ``base_footprint -> base_link`` (dynamic residual).  Done
        # unconditionally so every scene -- mobile or pinned, mobile-base
        # or arm-only -- has the same TF topology.
        _base_frame = "base_footprint"
        _core.init_ros(
            node_name=node_name,
            namespace="",
            fake_slam=ep.fake_slam,
            executor_threads=2,
            base_frame=_base_frame,
            # Selects which body-pose publisher the bridge constructs:
            # ``/tf_render`` (every body) instead of ``/tf_scene`` (scene
            # only). Must be decided HERE — publisher topic names are fixed at
            # construction — not at publish time.
            use_tf_render=ep.use_tf_render,
        )
        _core.init_scheduler(
            render_target_hz=render_hz,
            render_safety_ms=float(phys.render_safety_ms),
            physics_hz=ep.physics_hz,
            rtf=ep.realtime_factor,
        )

        # --- engine ---
        # physics_solver_substep / physics_solver_iterations use 0 as the
        # "engine picks its own default" sentinel; each backend substitutes
        # its own fallback when 0 is passed.  Explicit non-zero values from
        # the CLI / launcher YAML override the engine default.
        newton_solvers_path = os.path.join(os.path.dirname(manifest_path), "newton_solvers.json")
        engine_kwargs = {
            "robot_prefix": robot_prefix,
            "scene_usda": scene_usda,
            "robot_usda": robot_usda,
            "render_layer_usda": render_layer_usda,
            "physics_hz": ep.physics_hz,
            "num_envs": ep.num_envs,
            "render_hz": render_hz,
            "dds_hz": ep.dds_hz,
            "simulation_app": simulation_app,
            "logger": logger,
            "params": phys,
            "robot_from_urdf": robot_from_urdf,
            "init_joint_pos": init_joint_pos,
            "runtime_usd_dump_path": os.path.join(os.path.dirname(manifest_path), "robot_runtime.usda"),
            "pin_base_to_world": pin_base_to_world,
            "pin_base_mode": pin_base_mode,
            "convert_joints_to_fixed": list(convert_joints_to_fixed or []),
            "newton_solvers_path": newton_solvers_path,
            "scene_cfg": scene_cfg,
            "scene_yaml_path": scene_yaml_path,
            "physics_solver": ep.physics_solver,
            "physics_solver_substep": ep.physics_solver_substep,
            "physics_solver_iterations": ep.physics_solver_iterations,
            "physics_solver_mass_matrix_interval": ep.physics_solver_mass_matrix_interval,
            "render_mode": ep.render_mode,
            "mujoco_pd_ke": ep.mujoco_pd_ke,
            "mujoco_pd_kd": ep.mujoco_pd_kd,
            "mujoco_njmax": ep.mujoco_njmax,
            "mujoco_nconmax": ep.mujoco_nconmax,
            # /contact_states publish policy — tri-state strings; the
            # backend merges these with the scene YAML's ``contact_publish``
            # block (launcher wins on non-empty values).
            "contact_publish_scope": ep.contact_publish_scope,
            "contact_publish_include_self_contacts": ep.contact_publish_include_self_contacts,
            # SysID recording — see EngineNodeParams docstring.  Backends
            # that don't implement recording ignore these silently.
            # ``launcher_yaml_path`` is forwarded so engine-side
            # consumers can reference the exact launcher YAML that
            # produced their parameters — ``plugins_params_path`` is
            # already the temp file bringup materialises from the
            # launcher YAML (see physics_backend.launch.py
            # ``_OVERRIDABLE_FROM_YAML`` block), so no additional
            # wiring in launch is required.
            "sysid_mode": ep.sysid_mode,
            "sysid_label": ep.sysid_label,
            "sysid_max_ticks": ep.sysid_max_ticks,
            "sysid_chunk_size": ep.sysid_chunk_size,
            "sysid_auto_stop": ep.sysid_auto_stop,
            # Stimulus name + config route to the recorder's
            # ``load_stimulus`` factory.  See EngineNodeParams field
            # docstrings for the accepted names / config shape.
            "sysid_stimulus": ep.sysid_stimulus,
            "sysid_stimulus_config": ep.sysid_stimulus_config,
            "launcher_yaml_path": str(params.get("launcher_yaml_path", "") or ""),
        }
        # Forward launcher ``physics_*`` params to the backend constructor.
        # See :func:`_forward_physics_launcher_params` for the contract
        # (backend-ABI aware, honours ``**kwargs`` receivers, logs drops).
        # ``num_envs`` / ``mujoco_njmax`` / ``mujoco_nconmax`` are threaded
        # explicitly via ``engine_kwargs`` above — feature's engine_base
        # declares all three as defaulted params, so unconditional passing
        # works here.
        _forward_physics_launcher_params(engine_kwargs, params, physics_engine, logger)
        # ⚠ DIVERGENCE from rolling, deliberate (see docs/MERGE_LIST.md).
        # Rolling threads ``num_envs`` / ``mujoco_njmax`` / ``mujoco_nconmax`` /
        # ``pin_base_mode`` / ``use_tf_render`` HERE, each behind an
        # ``if <non-default>`` guard, because its backends do not declare those
        # kwargs and ``PhysicsEngine.create``'s ABI check would raise. This
        # branch declares all of them on ``engine_base`` (and passes
        # ``use_tf_render`` straight to ``init_ros``), so they are threaded
        # unconditionally in the ``engine_kwargs`` literal above instead.
        # When this branch merges INTO rolling, rolling's guards must NOT be
        # dropped until its backends declare the kwargs — that is Batch 5.
        sim = PhysicsEngine.create(physics_engine, **engine_kwargs)

        _, body_frames = self._snapshot_body_transforms(sim.stage, sim.body_paths)
        body_frames = list(body_frames)

        # v2 protocol: split robot vs scene bodies at the publish boundary so
        # ``/tf_scene`` carries scene state only (robot links ride ``/tf`` via
        # robot_state_publisher). Newton's frame IDs are already absolute prim
        # paths, so a ``/<robot_prefix>/`` prefix check is enough. An empty
        # mask means "publish everything as scene" (v1 behaviour).
        #
        # Order is load-bearing: this runs BEFORE the multi-env prefixing
        # below. The test is a prefix match, and an ``env_<w>/`` prefix would
        # make every robot link fail it — the whole robot would then be
        # silently routed to the scene channel, with no exception and no log,
        # simply vanishing off ``/tf``.
        #
        # The mask is computed unconditionally, ``use_tf_render`` or not: it is
        # simply true information about the body set. Render mode does not
        # blank it out — ``publish_tf_render`` ignores it by construction, so
        # "render mode carries the robot" holds no matter what lands here.
        robot_root = f"/{robot_prefix}/"
        body_is_robot = [
            bool(name) and (name == f"/{robot_prefix}" or name.startswith(robot_root)) for name in body_frames
        ]

        _ne = int(sim.num_envs)
        if _ne > 1 and body_frames and len(body_frames) % _ne == 0:
            _per = len(body_frames) // _ne
            body_frames = [f"env_{i // _per}/{body_frames[i].lstrip('/')}" for i in range(len(body_frames))]

        # v2 protocol: split ``/joint_states`` (robot only) from
        # ``/joint_states_scene`` (articulated scene DoFs — drawers, doors,
        # cart wheels). The URDF conversion stage baked the URDF joint set
        # into ``manifest.robot["joint_names"]``; anything the sim reports
        # that is not in that set is a scene DoF. Free-joint bodies never
        # enter ``sim.joint_names``.
        #
        # With no ``robot.joint_names`` in the manifest everything counts as
        # robot and ``/joint_states_scene`` stays quiet — equivalent to v1.
        #
        # Multi-env matters here too: at ``num_envs > 1`` the backend's joint
        # names carry an ``env_<w>/`` prefix (see
        # ``genie_sim_engine_newton/topology.py``) while
        # ``manifest.robot["joint_names"]`` holds the BARE URDF names. Match
        # on the bare form or every robot joint falls through to the scene
        # side and ``/joint_states`` goes empty.
        robot_joint_names_raw = manifest.robot.get("joint_names") if isinstance(manifest.robot, dict) else None
        robot_joint_names_set = set(robot_joint_names_raw or [])
        self.sim_joint_names: list = list(sim.joint_names)

        def _bare_joint_name(n: str) -> str:
            """Strip the multi-env ``env_<w>/`` prefix; no-op at num_envs==1."""
            if _ne > 1 and n.startswith("env_") and "/" in n:
                return n.split("/", 1)[1]
            return n

        if robot_joint_names_set:
            self.robot_joint_indices = [
                i for i, n in enumerate(self.sim_joint_names) if _bare_joint_name(n) in robot_joint_names_set
            ]
            self.scene_joint_indices = [
                i for i, n in enumerate(self.sim_joint_names) if _bare_joint_name(n) not in robot_joint_names_set
            ]
        else:
            self.robot_joint_indices = list(range(len(self.sim_joint_names)))
            self.scene_joint_indices = []
        self.robot_joint_names: list = [self.sim_joint_names[i] for i in self.robot_joint_indices]
        self.scene_joint_names: list = [self.sim_joint_names[i] for i in self.scene_joint_indices]

        # A backend may instead report its scene DoFs on a channel of its own,
        # because it never put them in ``sim.joint_names`` to begin with.
        # newton-standalone does exactly that: a door hinge in the joint map
        # would land in ``_joint_name_to_dof``, and ``_configure_pd_drives``
        # forces stiff POSITION-mode PD on every entry there — every door and
        # drawer in the scene would come up motorized. So it keeps them out,
        # which also means the subtraction above finds nothing and
        # ``/joint_states_scene`` would sit silent forever.
        #
        # When present, that channel WINS and the index path is abandoned
        # (indices into ``get_joint_states`` are meaningless for joints that
        # aren't in it — hence the empty list, not the computed one). The loop
        # reads the values through ``sim.get_scene_joint_states()``. The Isaac
        # Sim backends report nothing here and keep the subtraction.
        #
        # Requires BOTH halves: names without a readback (or vice versa) is a
        # half-implemented backend, and silently trusting one would publish
        # zeros or mismatched lengths. Fall back rather than guess.
        #
        # Both halves are declared on the ABC with defaults (``[]`` / ``None``),
        # so read them straight rather than probing with ``getattr`` — the
        # question is not whether the members exist but whether THIS backend
        # overrode the readback, which the identity check answers exactly.
        # Calling it here to find out is not an option: at bootstrap the
        # backend's state mirrors may not be populated yet, and ``None`` is
        # also the legitimate answer for a scene that simply has no articulated
        # DoFs.
        _direct_names = list(sim.scene_joint_names or [])
        _has_direct_read = type(sim).get_scene_joint_states is not PhysicsEngine.get_scene_joint_states
        self.scene_joint_direct: bool = bool(_direct_names) and _has_direct_read
        if _direct_names and not _has_direct_read:
            logger.warn(
                "[session] backend reports scene_joint_names but implements no "
                "get_scene_joint_states — ignoring it and falling back to the "
                "manifest subtraction; /joint_states_scene may stay quiet"
            )
        if self.scene_joint_direct:
            self.scene_joint_names = _direct_names
            self.scene_joint_indices = []
            logger.info(
                f"[session] /joint_states_scene ← backend scene DoF channel: "
                f"{len(_direct_names)} joint(s) "
                f"(disjoint from the {len(self.robot_joint_names)} on /joint_states) "
                f"{_direct_names[:6]}" + (f" +{len(_direct_names) - 6} more" if len(_direct_names) > 6 else "")
            )
        elif self.scene_joint_names:
            logger.info(
                f"[session] /joint_states_scene ← manifest subtraction: "
                f"{len(self.scene_joint_names)} scene DoF(s) sliced out of "
                f"sim.joint_names {self.scene_joint_names[:6]}"
                + (f" +{len(self.scene_joint_names) - 6} more" if len(self.scene_joint_names) > 6 else "")
            )

        _core.set_topology(list(self.robot_joint_names), list(body_frames), body_is_robot)

        # v2: freeze the whole routing decision into one value and hand it to
        # the run loop by constructor argument. Everything the publish path
        # needs to know is now known HERE, so nothing downstream re-derives
        # it and nothing is stamped onto the backend object.
        #
        # ``use_tf_render`` selects the body-pose channel: the loop picks
        # between ``_core.publish_body_tf_scene`` (v2 scene-only, on
        # ``/tf_scene``) and ``_core.publish_tf_render`` (single-topic
        # every-body, on ``/tf_render``). ``init_ros`` built the matching
        # publisher; the other is null, so calling the wrong entry point is a
        # silent C++ no-op — which is exactly why the flag has to travel with
        # the rest of the plan rather than being inferred at publish time.
        self.publish_plan = PublishPlan(
            robot_joint_indices=tuple(self.robot_joint_indices),
            scene_joint_names=tuple(self.scene_joint_names),
            scene_joint_indices=tuple(self.scene_joint_indices),
            scene_joint_direct=self.scene_joint_direct,
            use_tf_render=bool(ep.use_tf_render),
        )
        if ep.use_tf_render:
            logger.info(
                "[session] use_tf_render=on — every body pose (robot links "
                "included) goes on /tf_render via publish_tf_render; "
                "/tf_scene is not advertised "
                "(no robot_state_publisher in this deployment)"
            )

        self.sim = sim
        # One bring-up line per reset-contract omission, naming the class and
        # the fix.  Here rather than in the backend because every backend this
        # session builds -- including the downstream ones that subclass
        # EngineSession -- passes through this point exactly once.
        audit_lifecycle(sim, logger)
        self.physics_hz = ep.physics_hz
        self.render_hz = render_hz
        self.dds_hz = ep.dds_hz
        self.realtime_factor = ep.realtime_factor
        # Loop pacing policy.  Stashed, not acted on: the pacer is built
        # inside ``EngineRunLoop.spin`` where the wall-tick and publish
        # period it needs are already resolved.  See docs/step_modes.md.
        self.step_mode = ep.step_mode
        self.step_idle_timeout = ep.step_idle_timeout
        self.step_lockstep_ticks = ep.step_lockstep_ticks
        self.dt = 1.0 / ep.physics_hz
        self.scene_usda = scene_usda

    # ------------------------------------------------------------------
    # USD-helper hooks — subclasses bind these to a concrete backend's
    # stage helpers. ``_wait_for_manifest`` has a concrete default here
    # since every backend needs the same "block until this JSON exists"
    # semantics; only ``_snapshot_body_transforms`` is genuinely backend-
    # specific (USD stage vs. mjModel body arrays).
    # ------------------------------------------------------------------

    def _wait_for_manifest(
        self,
        path: str,
        logger: Any,
        retries: int = 30,
        interval: float = 1.0,
    ) -> None:
        """Block until ``path`` exists, logging once per ``interval``.

        The assemble step (upstream of every backend) writes a JSON
        manifest to a known path; this method waits for it before
        starting the physics loop. Pure stdlib — no backend coupling —
        so subclasses only need to override it for genuinely different
        wait strategies (e.g. a network manifest).
        """
        for _ in range(retries):
            if Path(path).exists():
                return
            logger.info(f"Waiting for manifest: {path}")
            time.sleep(interval)
        raise FileNotFoundError(
            f"manifest not found after {retries}s: {path}\n"
            "  → run 'geniesim assemble' (or the equivalent launch step) to generate it"
        )

    @abstractmethod
    def _snapshot_body_transforms(self, stage: Any, body_paths: list) -> tuple: ...

    def startup(self, headless: bool) -> None:
        self.sim.startup(headless)
        self.sim.note_render_target(self.render_hz)

    def run(self, render_hook=None, exit_check=None) -> int:
        EngineRunLoop(
            self.core,
            self.sim,
            self.dt,
            self.logger,
            post_step_hooks=self.post_step_hooks,
            rtf=self.realtime_factor,
            dds_hz=self.dds_hz,
            step_mode=self.step_mode or DEFAULT_STEP_MODE,
            step_idle_timeout=self.step_idle_timeout,
            step_lockstep_ticks=self.step_lockstep_ticks,
            publish_plan=self.publish_plan,
        ).spin(
            render_hook=render_hook,
            exit_check=exit_check,
        )
        return 0
