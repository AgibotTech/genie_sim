# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Direct Newton engine — cloth behaviour matches example_cloth_franka.py.

Franka demo params, in meter scale:
  particle_radius      0.008 m
  contact_margin       0.008 m
  self_contact_radius  0.002 m
  soft_contact_ke      1e4   (VBD stiffness is dimensionless)
  robot_contact_ke     5e4
  density              0.02
  tri_ke / tri_ka      1e4
  sim_substeps         10
  VBD iterations       5

Step structure is identical to the demo:
  for each substep:
    clear forces
    robot step  (particle_count=0, gravity=0, shape_contact_pair_count=0)
    restore particle_count + gravity
    cloth collision + VBD step
    swap states

Performance optimisations (vs. naive uncaptured + USD-Set writeback):
  * 10 substeps captured into a single CUDA graph (~0.4ms/frame total)
  * Kit-free: no Fabric / Hydra render kernels racing on the default
    CUDA stream.

ASSET REQUIREMENTS for cloth USDs:
  * metersPerUnit = 1.0 (raw vertex values are already meters)
  * No parent xformOp:scale or non-trivial xforms — pipeline reads raw
    points directly without applying transforms.
  * Hierarchy must include an Xform parent containing a Mesh child
    (e.g. /Root/shirt with Mesh as a child of /Root). Referencing a Mesh
    as the default prim under a destination Xform doesn't compose cleanly.

If a cloth USD doesn't meet these requirements, bake it into a clean form
(see tools/usd/bake_cloth_meters.py for a one-shot conversion script that
applies the parent xform stack + metersPerUnit into raw vertex data).
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warp as wp

from genie_sim_engine_core.common.lifecycle_ros import RosResetCoordinator
from genie_sim_engine_core.engine.base import PhysicsEngine
from genie_sim_engine_core.engine.lifecycle import EpisodeHandle, EpisodeSummary
from genie_sim_engine_core.schema.scene import PinBaseMode

# Module-level kernels live in ``kernels.py``.
#
# Newton-standalone is Kit-free, so the only writeback kernel is
# ``_kernel_velocity_inject`` (no Fabric path).

from genie_sim_engine_newton.engine.newton.setup import (
    _DebugPubsMixin,
    _InitPoseMixin,
    _ModelMixin,
    _NormalizeMixin,
    _RuntimeMixin,
    _SolverMixin,
    _StageMixin,
    _StaticFactsMixin,
)
from genie_sim_engine_newton.engine.newton.cloth import _ClothMixin
from genie_sim_engine_newton.engine.newton.control import _ControlMixin
from genie_sim_engine_newton.engine.newton.lifecycle import _LifecycleMixin
from genie_sim_engine_newton.engine.newton.plugin import _PluginMixin
from genie_sim_engine_newton.engine.newton.state import _StateMixin
from genie_sim_engine_newton.engine.newton.stats import _StatsMixin
from genie_sim_engine_newton.engine.newton.topology import _TopologyMixin


def resolve_pin_base_mode(pin_base_to_world: bool, explicit_mode: str) -> PinBaseMode:
    """Back-compat alias for :meth:`PinBaseMode.resolve`.

    The rule and the mode vocabulary live in
    ``genie_sim_engine_core.schema.scene.PinBaseMode`` — one declaration
    that the schema validator and every routing consumer read.  This name
    is kept because callers (and ``tools/test_pin_base_mode.py``) import
    it from here.  The return value is a ``str`` subclass, so existing
    ``== "weld"`` comparisons and f-strings are unaffected.
    """
    return PinBaseMode.resolve(pin_base_to_world, explicit_mode)


class _NewtonStandaloneBase(
    # Build phases — fixed order mirrors execution order in ``_build``.
    _RuntimeMixin,  # _build TOC + _warmup + _capture_graph + _dump_runtime_usd
    _StageMixin,  # phase 1: stage open + USD overrides
    _ModelMixin,  # phases 2-3: ModelBuilder + add_usd + finalize
    _NormalizeMixin,  # phase 4: mass clamp + group/articulation unification
    _SolverMixin,  # phases 5-6: states + robot solver + cloth solver
    _DebugPubsMixin,  # phase 7: rclpy marker publishers
    _InitPoseMixin,  # phase 8: name maps + init_joint_pos + state sync
    _StaticFactsMixin,  # phase 9: latched /rlinf/body_aabb_local
    # Non-build mixins (control, state I/O, etc.).
    _LifecycleMixin,
    _ClothMixin,
    _PluginMixin,
    _TopologyMixin,
    _ControlMixin,
    _StateMixin,
    _StatsMixin,
    PhysicsEngine,
):
    """Shared init, properties, and physics loop for Newton standalone engines.

    Concrete subclasses implement the three template methods:
      _open_stage(newton_scene)  — how to open/populate the USD stage
      _warmup_renders()          — no-op
      _configure_viewport()      — no-op

    Newton-standalone is Kit-free; the only concrete subclass is
    ``NewtonHeadlessEngine``.
    """

    # ---------------------------------------------------------------
    # Solver tuning — match example_cloth_franka.py exactly.
    #
    # franka demo (verbatim):
    #     self.fps          = 50            # ← main control variable
    #     self.sim_substeps = 10            # fixed
    #     self.iterations   = 5             # fixed
    #     self.frame_dt     = 1 / fps
    #     self.sim_dt       = frame_dt / sim_substeps
    #
    # Our engine maps:
    #     physics_hz                  ← franka's ``fps`` (user-controlled)
    #     physics_solver_substep      ← franka's ``sim_substeps`` (default 10)
    #     physics_solver_iterations   ← franka's ``iterations`` (default 5)
    #     sim_dt = (1 / physics_hz) / substeps                  (same formula)
    #
    # Just like franka, the user controls real-time speed via physics_hz.
    # Substeps + iterations stay at franka's tuned values unless explicitly
    # overridden. At physics_hz = 50  → sim_dt = 2.000 ms (franka exact).
    # At physics_hz = 60   → sim_dt = 1.667 ms (franka's other common rate).
    # At physics_hz = 100  → sim_dt = 1.000 ms (smaller, more accurate).
    # ---------------------------------------------------------------
    _DEFAULT_SUBSTEPS = 10  # franka self.sim_substeps
    _DEFAULT_ITERATIONS = 5  # franka self.iterations

    _PARTICLE_RADIUS = 0.008  # 0.8 cm
    _CLOTH_BODY_CONTACT_M = 0.008  # 0.8 cm — cloth↔rigid margin
    _SELF_CONTACT_RADIUS = 0.002  # 0.2 cm
    _SELF_CONTACT_MARGIN = 0.002  # 0.2 cm

    _SOFT_CONTACT_KE = 1e4
    _SOFT_CONTACT_KD = 1e-2
    _SOFT_CONTACT_MU = 0.25  # self_contact_friction in demo

    _ROBOT_KE = 5e4
    _ROBOT_KD = 1e-3
    _ROBOT_KU = 1.5

    _TRI_KE = 1e4
    _TRI_KA = 1e4
    _TRI_KD = 1.5e-6
    _EDGE_KE = 5.0
    _EDGE_KD = 1e-2
    _DENSITY = 0.02

    # ---------------------------------------------------------------

    def __init__(
        self,
        *,
        robot_prefix: str,
        scene_usda: str,
        robot_usda: str,
        render_layer_usda: str,
        physics_hz: float,
        render_hz: float,
        simulation_app: Any,
        logger: Any,
        params: Any,
        robot_from_urdf: bool,
        init_joint_pos: Any,
        runtime_usd_dump_path: str,
        num_envs: int = 1,
        dds_hz: float = 0.0,
        pin_base_to_world: bool = False,
        pin_base_mode: str = "",  # "" = weld; "weld" / "mocap" select explicitly
        convert_joints_to_fixed: list | None = None,
        newton_solvers_path: str = "",
        scene_cfg: dict | None = None,
        scene_yaml_path: str = "",
        physics_solver: str = "fsvbd",
        physics_solver_substep: int = 0,  # 0 = auto from physics_hz
        physics_solver_iterations: int = 0,  # 0 = auto from substeps
        physics_solver_mass_matrix_interval: int = 0,  # 0 = engine default (= sim_substeps); pass a large value to effectively disable auto rebuild
        render_mode: str = "raster",
        mujoco_pd_ke: float = 0.0,  # 0 = engine default (50000 N·m/rad)
        mujoco_pd_kd: float = 0.0,  # 0 = engine default (500 N·m·s/rad)
        # MuJoCo constraint-buffer capacity (mjwarp adapters only).  Declared
        # HERE, not just on ``make_adapter``, because ``PhysicsEngine.create``
        # validates launcher kwargs against this signature
        # (``engine/base.py::_validate_create_kwargs``) and this class takes no
        # ``**kwargs``: a launcher that sets the documented
        # ``mujoco_njmax`` / ``mujoco_nconmax`` node params
        # (``launcher_newton_mjwarp.yaml``) would otherwise raise an ABI
        # mismatch ``TypeError`` before the engine ever starts.  ``0`` = leave
        # it to the adapter; see ``adapters/__init__.py::make_adapter`` for how
        # the stock and multi-env adapters read that zero differently.
        mujoco_njmax: int = 0,  # 0 = adapter default / auto-scale
        mujoco_nconmax: int = 0,  # 0 = adapter default / auto-scale
        # /contact_states publish policy — accepted from the launcher for ABI
        # parity with the mujoco engine, but no-op on this backend: newton
        # does not currently publish /contact_states.  Stored so a future
        # port of the mujoco adapter's merge logic (scene YAML +
        # launcher-wins) can pick them up without another signature change.
        contact_publish_scope: str = "",
        contact_publish_include_self_contacts: str = "",
        # SysID recording knobs — accepted for ABI parity with what
        # ``common/session.py`` unconditionally forwards, but **no-op on this
        # backend**: newton-standalone implements no trajectory recorder.
        # Same pattern (and same reason) as the contact_publish_* pair above.
        #
        # Without these the engine will not construct AT ALL: session.py puts
        # every one of them into ``engine_kwargs`` and
        # ``PhysicsEngine._validate_create_kwargs`` raises on any name the
        # constructor does not accept (no ``**kwargs`` here, by design — the
        # strict ABI check is what catches launcher/engine drift). Upstream's
        # own sysid commits (6ca41cbbbf / 99b95e9ba2 / 46731549a9, 2026-08-11)
        # added the forwarding without the matching signature, so rolling's
        # newton path hits the same TypeError; this is the accept-and-ignore
        # half they are missing, not an RLinf-specific need.
        sysid_mode: str = "",
        sysid_label: str = "",
        sysid_max_ticks: int = 0,
        sysid_chunk_size: int = 0,
        sysid_auto_stop: bool = True,
        sysid_stimulus: str = "",
        sysid_stimulus_config: str = "",
        launcher_yaml_path: str = "",
    ) -> None:
        self._robot_prefix_str = robot_prefix
        self._scene_usda = scene_usda
        self._robot_usda = robot_usda
        self._render_layer_usda = render_layer_usda
        self._physics_hz = physics_hz
        self._render_hz = render_hz
        # Native multi-world replicate count (1 = single env). Read in the
        # model build to replicate the single-env template into N worlds.
        self._num_envs = int(num_envs) if int(num_envs) > 0 else 1
        self._simulation_app = simulation_app
        self._logger = logger
        self._params = params
        # Gripper name-matching fallbacks come from config; without this
        # the built-in G2 defaults stay in force.
        from genie_sim_engine_newton.common.gripper_annotations import (  # noqa: PLC0415
            configure_naming as _configure_gripper_naming,
        )

        _configure_gripper_naming(getattr(params, "gripper_naming", None), logger)
        # ROS reset coordinator — brings up /reset_envs subscriber and
        # coordinates episode resets. Backend-agnostic; _LifecycleMixin
        # implements the actual episode_reset(env_ids, seed, instances).
        self._lifecycle = RosResetCoordinator(self, logger)
        # SimulationControl state (see genie_sim_engine_core.engine.lifecycle).
        # ``_paused`` gates the physics advance inside ``step`` itself, not just
        # the run loop's own flag: an RL consumer holding the engine directly
        # calls ``pause()`` and then keeps calling ``step()`` on its own clock,
        # and the loop's flag is not in that path.
        self._paused: bool = False
        self._episode_id: int = 0
        self._episode_started_at_tick: int = 0
        self._robot_from_urdf = robot_from_urdf
        self._init_joint_pos = init_joint_pos
        self._convert_joints_to_fixed = list(convert_joints_to_fixed or [])
        self._pin_base_to_world = bool(pin_base_to_world)
        # Path for the post-build USD snapshot of the live composed stage
        # (robot.usda + newton_scene.usda + any runtime authoring). Empty
        # disables the dump. Written once in startup() after _warmup so
        # the snapshot reflects exactly what ``add_usd`` parsed plus the
        # cloth-scene sublayer — see ``_dump_runtime_usd``.
        self._runtime_usd_dump_path = runtime_usd_dump_path
        self._newton_solvers_path = newton_solvers_path
        self._scene_cfg = scene_cfg or {}
        # Resolve the base-pin mode NOW — both inputs (``pin_base_to_world``
        # and ``scene_cfg``) are already in hand.  ``weld``/``free`` are the
        # two pre-existing behaviors (byte-identical to before this flag
        # existed); ``mocap`` is new — see ``PinBaseMode`` for the full
        # semantics.  Consumed by:
        #   * ``setup/stage.py::_apply_runtime_fix_policies`` — whether to
        #     deactivate the URDF root_joint weld (weld: no: mocap/free: yes);
        #   * ``adapters/__init__.make_adapter`` — ``pin_base_joint=True``
        #     when mocap, so the mjwarp adapter registers the robot root
        #     FREE joint into the same kinematic-pin machinery used for
        #     KINEMATIC scene bodies;
        #   * ``_pick_substep_body`` below — mocap needs REAL gravity +
        #     shape contacts (like ``free``) since the pin clamp (not a
        #     scene-wide gravity/contact kill) is what holds the base.
        self._pin_base_mode_explicit = str(pin_base_mode or "")
        self._pin_base_mode = resolve_pin_base_mode(self._pin_base_to_world, self._pin_base_mode_explicit)
        _pbm_norm = self._pin_base_mode_explicit.strip().lower()
        if not self._pin_base_to_world:
            _pbm_reason = "pin_base_to_world=false"
        elif _pbm_norm in ("weld", "mocap"):
            _pbm_reason = "explicit"
        else:
            _pbm_reason = "default"
        if _pbm_reason == "pin_base_to_world=false" and _pbm_norm:
            # Accepted, not refused: the boolean outranks the mode
            # (``PinBaseMode.resolve``).  Say so loudly — silently
            # dropping a requested mode is how a scene ends up not doing
            # what its yaml appears to ask for.
            self._logger.warn(
                f"[engine] pin_base_mode={_pbm_norm!r} requested but pin_base_to_world=false "
                f"outranks it — resolved to {self._pin_base_mode}"
            )
        else:
            self._logger.info(f"[engine] pin_base_mode={self._pin_base_mode} ({_pbm_reason})")
        self._scene_yaml_path = scene_yaml_path
        # ``_scene_plugin`` is loaded lazily in ``_build`` from the scene
        # yaml's ``newton.scene_plugin`` field; ``None`` until then.
        self._scene_plugin: Any = None
        self._scene_plugin_path: str = ""
        # Sim-time accumulator the loop drives via ``step(dt, …)``; passed
        # to the scene plugin's on_post_step / on_render hooks so they can
        # phase a controller without re-reading the host clock.
        self._sim_time_acc: float = 0.0
        # Render-mode choice — forwarded to ``configure_viewport_for_debug``
        # after _build/_warmup so the live viewport agrees with the
        # bootstrap-time kit args. See ``runtime.bootstrap.RENDER_MODE_MAP``
        # for the canonical mode list.
        self._render_mode = (render_mode or "raster").strip().lower()
        self._physics_solver = physics_solver.strip().lower()
        # Solver-specific substep defaults.
        # mujoco-warp is an implicit integrator that converges in 1 pass per frame.
        # vbd/featherstone (explicit) need 10 substeps for cloth stability.
        if self._physics_solver.startswith(("mujoco-warp", "mujoco_warp")):
            _substep_default = 1
            _iter_default = 0  # let SolverMuJoCo use its own default (100)
        else:
            _substep_default = self._DEFAULT_SUBSTEPS
            _iter_default = self._DEFAULT_ITERATIONS
        self._sim_substeps = int(physics_solver_substep) if int(physics_solver_substep) > 0 else _substep_default
        self._vbd_iterations = int(physics_solver_iterations) if int(physics_solver_iterations) > 0 else _iter_default
        # 0 sentinel → adapter falls back to ``sim_substeps`` (= current
        # behaviour, M rebuilt once per frame).  Any positive value
        # OVERRIDES the default; the user passes e.g. 100000 to
        # effectively cache M forever (suitable for slow-motion cloth
        # folding where joint velocities stay well below 1 rad/s and
        # M(q) drift is negligible across many frames).
        self._mass_matrix_interval = int(physics_solver_mass_matrix_interval)
        self._mujoco_pd_ke = float(mujoco_pd_ke)
        self._mujoco_pd_kd = float(mujoco_pd_kd)
        self._mujoco_njmax = int(mujoco_njmax)
        self._mujoco_nconmax = int(mujoco_nconmax)
        self._contact_publish_scope = (contact_publish_scope or "").strip().lower()
        self._contact_publish_include_self_contacts = (contact_publish_include_self_contacts or "").strip().lower()
        if self._contact_publish_scope or self._contact_publish_include_self_contacts:
            self._logger.info(
                f"[genie_sim_engine_newton] /contact_states not published on this backend; "
                f"ignoring contact_publish_scope={self._contact_publish_scope!r} "
                f"contact_publish_include_self_contacts={self._contact_publish_include_self_contacts!r}"
            )
        if str(sysid_mode or "").strip():
            # Same reason the pair above warns: the signature accepts these so
            # the engine constructs at all, and accepting them silently would
            # leave an operator watching for a recording that is never written.
            self._logger.warn(
                "[genie_sim_engine_newton] SysID recording is not implemented on "
                "newton-standalone; ignoring "
                f"sysid_mode={sysid_mode!r}, sysid_label={sysid_label!r}, "
                f"sysid_max_ticks={sysid_max_ticks}, sysid_chunk_size={sysid_chunk_size}, "
                f"sysid_auto_stop={sysid_auto_stop}, sysid_stimulus={sysid_stimulus!r}, "
                f"sysid_stimulus_config={sysid_stimulus_config!r}, "
                f"launcher_yaml_path={launcher_yaml_path!r}"
            )

        # Runtime state
        # Inline OVRtx visualizer (physics_engine_visualizer:=ovrtx) attaches
        # a wp.Event here.  When set, ``tick_extras`` records it on Warp's
        # current stream after each physics step so the OVRtx render thread
        # can GPU-wait on the latest physics commit without CPU sync.
        # See docs/ovrtx_sync.md.
        self._physics_step_event: Any = None
        self._stage: Any = None
        # Diagnostic: when GENIESIM_STEP_GPU_SYNC is truthy, ``step()``
        # blocks on a ``wp.synchronize_device`` after launching the
        # captured graph and emits a 1Hz log of the GPU completion time.
        # OFF by default — turning it on forces every frame to wait for
        # the full cloth+robot graph (~22 ms on the fr3+tshirt scene),
        # which strips the parallel overlap with the Hydra render and
        # drops render to 0 Hz. Useful for one-shot profiling to settle
        # whether physics is GPU-bound; keep off in normal use.
        self._step_gpu_sync = os.environ.get("GENIESIM_STEP_GPU_SYNC", "").strip().lower() in ("1", "true", "yes", "on")
        self._model: Any = None
        self._state_0: Any = None
        self._state_1: Any = None
        self._control: Any = None
        self._robot_solver: Any = None
        # Set only by ``setup/solver.py::_build_mujoco_xpbd_coupled_proxy``
        # when a scene opts into ``newton.coupling``. Non-None is what
        # ``_pick_substep_body`` keys the coupled substep on.
        self._coupled_solver: Any = None
        # Solver-specific behavior lives in a SolverAdapter.  The adapter
        # owns the position-target buffer (mjwarp -> control.joint_target_q,
        # featherstone -> dedicated wp.array) and the per-substep robot
        # step.  apply_commands routes /joint_command writes through
        # self._adapter.target_buffer().
        from genie_sim_engine_newton.engine.newton.adapters import make_adapter

        # When ``runtime_usd_dump_path`` is set, default the MJCF dump
        # to a sibling file ``robot_runtime.xml`` in the same scene
        # directory.  Operator gets a matched pair of post-load
        # snapshots — USD on the Newton side, MJCF on the mjwarp side —
        # so the two model representations can be inspected and diffed.
        mjcf_dump = ""
        if self._runtime_usd_dump_path:
            mjcf_dump = os.path.splitext(self._runtime_usd_dump_path)[0] + ".xml"
        # IMU sites/sensors to bake into the dumped robot_runtime.xml.
        # Declared per-scene under the top-level ``imu:`` key (sibling of
        # ``cameras:`` / ``lidars:``); absent for robots with no IMU, in
        # which case the mjcf post-process step no-ops.  Static XML only —
        # not a runtime publisher.
        imu_cfg = self._scene_cfg.get("imu")
        # ``newton.mujoco_enable_multiccd`` — MuJoCo's multi-point convex
        # contact manifolds.  Scene-homed rather than physics_params-homed
        # because MultiCCD requires zero-margin convex pairs, which is a
        # property of the scene's collision geometry and not a global contact
        # preference. Absent stays ``None`` — each adapter
        # keeps its own default, which differs between the stock and multi-env
        # ones; see ``adapters/__init__.py::make_adapter``.
        #
        # YAML strings are accepted because a scene author writing
        # ``mujoco_enable_multiccd: "true"`` should not silently get the
        # opposite of what they asked for — a bare ``bool("false")`` is True.
        _newton_cfg = self._scene_cfg.get("newton") if isinstance(self._scene_cfg, dict) else None
        _multiccd = _newton_cfg.get("mujoco_enable_multiccd") if isinstance(_newton_cfg, dict) else None
        if isinstance(_multiccd, str):
            _mc = _multiccd.strip().lower()
            if _mc in ("true", "1", "yes", "on"):
                _multiccd = True
            elif _mc in ("false", "0", "no", "off"):
                _multiccd = False
            else:
                self._logger.warn(
                    f"[newton] scene newton.mujoco_enable_multiccd={_multiccd!r} is not a "
                    f"boolean — ignoring it and leaving the adapter default"
                )
                _multiccd = None
        elif _multiccd is not None:
            _multiccd = bool(_multiccd)
        from genie_sim_engine_newton.common.joint_params import load_scene_joint_params

        self._joint_params = load_scene_joint_params(self._scene_cfg, engine="newton", logger=self._logger)
        self._adapter = make_adapter(
            self._physics_solver,
            logger=self._logger,
            mujoco_pd_ke=self._mujoco_pd_ke,
            mujoco_pd_kd=self._mujoco_pd_kd,
            mujoco_njmax=self._mujoco_njmax,
            mujoco_nconmax=self._mujoco_nconmax,
            mujoco_save_to_mjcf=mjcf_dump,
            physics_params=self._params,
            joint_params=self._joint_params,
            physics_hz=self._physics_hz,
            imu_cfg=imu_cfg,
            pin_base_joint=PinBaseMode.of(self).needs_base_clamp,
            mujoco_enable_multiccd=_multiccd,
        )
        # Base-pin gate — an adapter without the per-substep clamp raises
        # early instead of silently deactivating the URDF root_joint weld
        # and leaving the base under full gravity with nothing holding it,
        # while the log still reads ``pin_base_mode=mocap``.  Same shape
        # as the cloth gate in ``setup/model.py``.
        #
        # We RAISE where the Isaac launchers only warn-and-downgrade, and
        # that asymmetry is a decision, not drift: refuse when the fallback
        # is unsafe, downgrade when it is merely weaker.  Here there is no
        # fallback at all — the weld is off and nothing replaces it.  On
        # Isaac the weld still holds the base and only reset-time
        # re-placement is lost, so refusing would reject a scene that would
        # otherwise run correctly.  Rationale in engine_core
        # docs/pipeline.md#base-pin-runtime-toggle; please keep both halves.
        if PinBaseMode.of(self).needs_base_clamp and not getattr(self._adapter, "supports_base_pin", False):
            raise RuntimeError(
                f"pin_base_mode=mocap needs an adapter with a per-substep base clamp; "
                f"physics_solver={self._adapter.name} has none. Use "
                f"physics_solver:=mujoco_warp_multi_env "
                f"(launcher_newton_mjwarp_multi_env.yaml), or drop pin_base_mode "
                f"to use the default weld."
            )
        self._cloth_solver: Any = None
        self._contacts: Any = None
        self._gravity_zero: Any = None
        self._gravity_earth: Any = None
        # ``newton.debug.pub_deformable_marker`` — debug-only sibling
        # rclpy publisher (visualization_msgs/Marker TRIANGLE_LIST of the
        # cloth / FEM surface, particles indexed by tri_indices).
        # Lazily constructed in _build when the flag is true.  ``None``
        # otherwise; tick_extras skips the publish path when None.
        self._deformable_pub: Any = None
        # ``newton.debug.pub_deformable_pointcloud`` — debug-only sibling
        # rclpy publisher (sensor_msgs/PointCloud2 of every particle in
        # state.particle_q).  Same on/off semantics as the marker pub
        # above; the two are independent so an operator can run either,
        # both, or neither.
        self._deformable_pc_pub: Any = None
        # ``newton.debug.pub_object_marker`` — debug-only sibling rclpy
        # publisher (visualization_msgs/MarkerArray of every free-joint
        # body's pose).  Same on/off semantics.
        self._object_pub: Any = None
        # Sim-time of next debug-pub fire.  Both debug publishers share
        # this gate so they fire at ``render_hz`` (not physics_hz × subs).
        # See ``tick_extras`` for the advance step.
        self._next_debug_pub_time: float = 0.0
        self._joint_names: List[str] = []
        self._joint_prim_map: Dict[str, str] = {}
        # ``_joint_name_to_dof`` is the qd (velocity / DOF vector) index
        # — the right thing to use for ``control.joint_target_q``,
        # ``control.joint_target_qd``, ``state.joint_qd``, and the
        # per-DOF controlled-mask in the featherstone / AVBD adapters.
        # ``_joint_name_to_q_idx`` is the q (configuration vector) index,
        # needed when reading ``state.joint_q`` for the joint position.
        # The two coincide for revolute / prismatic / fixed joints
        # (q_count == dof_count) and diverge by +1 for every preceding
        # FREE joint (q_count=7, dof_count=6 — quaternion vs angular
        # velocity).  Conflating them makes every commanded joint move
        # its neighbour once ``pin_base_to_world: false`` puts a FREE
        # joint at the head of the articulation, so the two maps are
        # kept strictly distinct.
        #
        # The two name maps are populated from a single ``JointIndex``
        # snapshot in ``_build_joint_map``.  ``self._jindex`` keeps the
        # full snapshot around so downstream consumers (adapters,
        # state.py, control.py) can reach richer
        # per-joint metadata — joint type, q/qd slice widths, the
        # ``copy_q_to_qd`` helper — without re-walking the Newton
        # ``model.joint_*`` arrays.  Stays ``None`` until
        # ``_build_joint_map`` runs (i.e. until ``add_usd`` has
        # populated those arrays); callers that pre-empt that order
        # should guard with ``if self._jindex is not None``.
        self._joint_name_to_dof: Dict[str, int] = {}
        self._joint_name_to_q_idx: Dict[str, int] = {}
        # Scene-articulation DoFs (door hinge / drawer slide) — the joints
        # ``_build_joint_map`` keeps OFF the controllable surface but still
        # reports on ``/joint_states_scene`` for benchmark / eval consumers.
        # Deliberately a channel of its own rather than entries in the two
        # maps above: those define what ``/joint_command`` can drive and what
        # ``_configure_pd_drives`` clamps into POSITION-mode PD, and a scene
        # prop belongs to neither. Names are bare (``env_<w>/`` prefixed at
        # num_envs>1); the index arrays are parallel to it, into
        # ``state.joint_q`` (q) and ``state.joint_qd`` / ``joint_f`` (qd).
        self._scene_joint_names: List[str] = []
        self._scene_joint_q_idx: List[int] = []
        self._scene_joint_dof_idx: List[int] = []
        from genie_sim_engine_newton.engine.newton.joint_index import JointIndex  # noqa: PLC0415

        self._jindex: Optional[JointIndex] = None
        self._body_paths: List[str] = []
        # Set by _build_body_map (topology.py) if the active adapter
        # implements local_body_aabb (mjwarp only) — see body_aabb_local.
        self._body_aabb_local: Optional[Any] = None

        # Joints that fix_base / fix_head / fix_body collapsed to FixedJoint
        # at runtime.  Featherstone has 0 DOFs for them; Newton's
        # ``model.joint_q`` doesn't carry their values.  But the URDF still
        # describes them as revolute, so robot_state_publisher / RViz /
        # downstream consumers need to see them in ``/joint_states`` to
        # build a consistent kinematic-chain TF tree.  This dict is the
        # single source of truth for "what value to publish" for those
        # joints — populated once at lifecycle build time, queried every
        # tick by ``state.get_joint_states``.  Empty when no fix_* flag
        # is on (then everything goes through the normal joint_q path).
        self._static_joint_q: Dict[str, float] = {}

        # CUDA graph
        self._cuda_graph: Any = None
        self._sim_dt_captured: float = 0.0

        # Pinned-host mirrors of state_0.joint_q / joint_qd / body_q /
        # particle_q.
        # ``AsyncMirror`` runs an async device→host wp.copy on the
        # captured graph's stream after every step (engine.step below);
        # ``get_joint_states`` / ``get_body_transforms`` / ``get_odom``
        # read the resulting host buffer with no GPU contact.  The
        # mirror lags by ONE physics tick — see async_mirror.py for the
        # full ping-pong design.  Trade: 10 ms of /joint_states lag at
        # 100 Hz vs ~6 ms of CPU stall every tick on the synchronous
        # path.
        from genie_sim_engine_newton.engine.newton.async_mirror import AsyncMirror  # noqa: PLC0415

        self._jq_mirror = AsyncMirror()
        self._jqd_mirror = AsyncMirror()
        self._bq_mirror = AsyncMirror()
        self._pq_mirror = None

        # Stats
        self._tick_count = 0
        self._step_ms_acc = 0.0
        self._step_ms_max = 0.0
        self._writeback_ms_acc = 0.0
        self._writeback_ms_max = 0.0
        self._render_ms_acc = 0.0
        self._render_ms_max = 0.0
        self._render_count = 0
        self._render_target_hz = 30.0  # set by run loop via note_render_target
        self._t_log_start = time.monotonic()
        self._t_loop_start = time.monotonic()
        self._frame_id = 0
        self._LOG_EVERY = int(physics_hz * 5)
        # Per-phase timings (run loop pushes via note_phase_timing)
        self._phase_step_ms_acc = 0.0
        self._phase_extras_ms_acc = 0.0
        self._phase_render_ms_acc = 0.0
        self._phase_render_sync_ms_acc = 0.0
        self._phase_render_sync_max = 0.0
        # Publish-phase timings (run loop pushes via note_publish_phase)
        self._publish_clock_ms_acc = 0.0
        self._publish_joints_ms_acc = 0.0
        self._publish_bodies_ms_acc = 0.0
        self._publish_odom_ms_acc = 0.0

        self._build()
        self._warmup()
        # One-shot USD snapshot of the live composed stage AFTER add_usd
        # parsing + cloth-scene sublayer resolve.
        self._dump_runtime_usd()
        self._capture_graph()

    # ------------------------------------------------------------------
    # PhysicsEngine interface
    # ------------------------------------------------------------------

    @property
    def stage(self) -> Any:
        return self._stage

    @property
    def robot_prefix(self) -> str:
        return self._robot_prefix_str

    @property
    def joint_names(self) -> List[str]:
        return self._joint_names

    @property
    def scene_joint_names(self) -> List[str]:
        """Bare names of the articulated-scene DoFs, for
        ``/joint_states_scene``.

        Empty on a robot-only scene. These are NOT in :attr:`joint_names` —
        the two sets are disjoint by construction (see
        ``_build_joint_map``), which is what lets ``EngineSession`` use this
        channel directly instead of subtracting the manifest's robot joint
        set out of :attr:`joint_names`.
        """
        return self._scene_joint_names

    @property
    def joint_prim_map(self) -> Dict[str, str]:
        return self._joint_prim_map

    @property
    def body_paths(self) -> List[str]:
        return self._body_paths

    @property
    def body_aabb_local(self) -> Any:
        """Per-env-body-indexed LOCAL AABB table (``(len(body_paths), 6)``
        ``[cx,cy,cz,hx,hy,hz]``), or ``None`` (mjwarp only — see
        ``_build_body_map`` / ``adapters/mujoco_warp.py::local_body_aabb``).
        Set once at build time; consumed by ``common/session.py`` to
        publish ``/rlinf/body_aabb_local``."""
        return getattr(self, "_body_aabb_local", None)

    # ------------------------------------------------------------------
    # SimulationControl — see genie_sim_engine_core.engine.lifecycle.
    # ``episode_reset`` / ``teleport`` live in ``_LifecycleMixin``
    # (engine/newton/lifecycle.py), which owns the state-write machinery.
    # ------------------------------------------------------------------

    @property
    def autostart(self) -> bool:
        """True — this backend spins as soon as the run loop has it.

        Newton-standalone is built by the ROS realtime path, which steps on
        construction. An RL consumer that wants the idle-until-``episode_begin``
        shape drives ``step()`` itself and simply does not start the loop.
        """
        return True

    @property
    def num_envs(self) -> int:
        """Parallel worlds, from the ``num_envs`` constructor argument."""
        return self._num_envs

    @property
    def reset_epoch(self) -> int:
        """Resets completed, bumped at the tail of ``episode_reset``.

        ``getattr`` with a default because the counter is initialised in
        ``_snapshot_reset_state``, which runs during the build: the run loop
        may read this while the engine is still coming up.
        """
        return int(getattr(self, "_reset_epoch", 0) or 0)

    @property
    def activity(self) -> Any:
        """The ``RosResetCoordinator``, which is already an ``ActivitySignal``.

        It owns both halves the pacer needs: the ``/joint_command`` arrival
        stamp (``last_cmd_wall`` / ``idle_activity_ok``) and the pending-reset
        queue (``reset_pending``). Returning it is what makes the idle gate
        and lockstep engagement work on this backend at all -- the pacer used
        to look those names up on the engine, where they have never existed,
        so every probe missed and both features silently degraded to
        free-running.
        """
        return self._lifecycle

    def pause(self) -> None:
        """Join the completed tick and optional RGB capture before acknowledging."""
        from genie_sim_engine_newton.engine.newton.state_sync import pause_at_tick_boundary

        pause_at_tick_boundary(self)

    def resume(self) -> None:
        """Lift the :meth:`pause` gate. Idempotent."""
        if self._paused:
            self._paused = False
            self._logger.info("[newton-standalone] resumed")

    def episode_begin(self, seed: int) -> EpisodeHandle:
        """Open an episode at the current frame; does NOT reset any world.

        Resetting here would make ``episode_begin`` unusable for the common
        RL shape of "reset a subset, then open episodes for those worlds" —
        the caller decides which worlds move via
        :meth:`episode_reset`. Seeding is likewise the caller's:
        ``episode_reset(env_ids, seeds=...)`` carries per-env seeds, which one
        scalar here cannot express, so the seed is recorded on the handle
        rather than applied.
        """
        self._episode_id += 1
        self._episode_started_at_tick = int(self._frame_id)
        return EpisodeHandle(
            episode_id=self._episode_id,
            seed=int(seed),
            started_at_tick=self._episode_started_at_tick,
            metadata={"num_envs": self._num_envs, "reset_epoch": int(self._lifecycle.reset_epoch)},
        )

    def episode_end(self, reason: str) -> EpisodeSummary:
        """Close the current episode. Leaves state alone for post-hoc reads.

        Tick counts come from ``_frame_id``, the physics frame counter, so a
        pause between begin and end does not inflate them.
        """
        ended = int(self._frame_id)
        return EpisodeSummary(
            episode_id=self._episode_id,
            reason=str(reason),
            ended_at_tick=ended,
            tick_count=max(0, ended - self._episode_started_at_tick),
            metadata={"reset_epoch": int(self._lifecycle.reset_epoch)},
        )

    def raw_model(self) -> Any:
        """The Newton ``Model``. Escape hatch — no cross-backend guarantees."""
        return self._model

    def raw_data(self) -> Any:
        """The live Newton ``State`` (``state_0``).

        Solver-internal data (mjwarp's ``Data``) is one level further down,
        through the adapter; this stays at the Newton layer so a consumer is
        not silently handed a different backend's structs depending on which
        solver got built.
        """
        return self._state_0

    def startup(self, headless: bool) -> None:
        # Match franka: fps → frame_dt → sim_dt = frame_dt / sim_substeps.
        frame_dt_ms = 1000.0 / self._physics_hz
        sim_dt_ms = frame_dt_ms / self._sim_substeps
        total_iters = self._sim_substeps * self._vbd_iterations
        self._logger.info(
            f"[newton-standalone] ready — physics_hz={self._physics_hz:.0f}  "
            f"frame_dt={frame_dt_ms:.3f}ms  (1× real time)"
        )
        self._logger.info(
            f"[newton-standalone]   sim_substeps={self._sim_substeps} (franka default 10)  "
            f"iterations={self._vbd_iterations} (franka default 5)"
        )
        self._logger.info(
            f"[newton-standalone]   sim_dt={sim_dt_ms:.3f}ms  "
            f"total_iters/frame={total_iters}  "
            f"(= sim_substeps × iterations, franka-style)"
        )
        self._logger.info(
            f"[newton-standalone] step GPU sync diagnostic: "
            f"{'ENABLED' if self._step_gpu_sync else 'disabled'} "
            f"(set GENIESIM_STEP_GPU_SYNC=0 to turn off)"
        )

    def _update_record_snapshot_mirror(self, *, create: bool = False):
        """Allocate on demand; retire safely when the consumer disappears."""
        mirror = self._pq_mirror
        if not self._record_snapshot_enabled():
            if mirror is not None:
                # A queued DMA still owns the pinned memory. Wait only on this
                # enabled -> disabled transition before dropping its buffers.
                mirror.synchronize()
                self._pq_mirror = None
        elif mirror is None and create:
            from genie_sim_engine_newton.engine.newton.async_mirror import AsyncMirror  # noqa: PLC0415

            self._pq_mirror = AsyncMirror()
        return self._pq_mirror

    def step(self, dt: float, step_start: float) -> float:
        """One physics frame (10 substeps). Uses CUDA graph if captured.

        With ``GENIESIM_STEP_GPU_SYNC=1`` set, also issues a
        ``wp.synchronize_device("cuda:0")`` after the launch and reports
        the GPU completion time via the 1Hz log. This is a DIAGNOSTIC
        knob — it stalls the Python thread until every queued kernel
        finishes, so don't leave it on for normal runs (it strips the
        async overlap that lets render and physics share the GPU). Use
        it to settle whether ``get_joint_states`` sync time is bound by
        the captured graph itself or by something queued outside it
        (e.g. cloth post-step BVH rebuild, ROS publish-side numpy() syncs).
        """
        t0 = time.monotonic()

        if self._paused:
            # Advance nothing. The mirrors are deliberately NOT refreshed
            # here: they already hold the last stepped state, and a reset or
            # teleport applied while paused flushes them itself (see
            # ``_commit_state_write``), so a publish during a pause carries
            # the current state either way.
            #
            # Sim time is the caller's to keep — this gate stops physics, not
            # ``/clock``. The ROS lifecycle verbs pause both (the run loop
            # holds its own flag as well); a direct ``pause()`` on the engine
            # stops only the solver.
            return (time.monotonic() - t0) * 1000.0

        if self._cloth_solver is not None and hasattr(self._cloth_solver, "rebuild_bvh"):
            # BVH must be rebuilt each frame; can't be inside the captured
            # graph. Only ``SolverVBD`` exposes ``rebuild_bvh`` — XPBD and
            # Style3D maintain their own broad-phase internally and the
            # method isn't part of their API. ``hasattr`` guard keeps the
            # call VBD-only without branching on the solver class name.
            self._cloth_solver.rebuild_bvh(self._state_0)

        if self._cuda_graph is not None:
            # Single GPU launch for all 10 substeps — eliminates 10× kernel-
            # launch overhead and Python loop costs. Rebuild_bvh is the only
            # required outside-graph step.
            wp.capture_launch(self._cuda_graph)
        else:
            # Fallback: uncaptured Python loop (used during warmup, or if
            # capture failed). State swap is Python-level here.
            self._simulate_substeps(dt, captured=False)

        # Scene-plugin per-frame callback.  Runs OUTSIDE the captured CUDA
        # graph (post the substep loop) so it can do host-side work like
        # state-machine updates and writing the next frame's kinematic
        # body poses.  See plugin.py for the contract.
        self._sim_time_acc += dt
        self._call_plugin(
            "on_post_step",
            self._state_0,
            self._sim_time_acc,
            dt,
            self._plugin_ctx(),
        )

        particle_mirror = self._update_record_snapshot_mirror(create=True)

        # Async device→host mirror refresh — see ``async_mirror.py``
        # for the ping-pong design.  Each mirror enqueues a wp.copy
        # onto the source-device's current stream (the same one the
        # captured graph just ran on) so the memcpy is FIFO-ordered
        # behind the kernel writes; the host returns immediately.
        # Readers in state.py see the OTHER slot, which the previous
        # tick wrote, and check its event before using it.
        if self._state_0 is not None:
            jq_src = getattr(self._state_0, "joint_q", None)
            jqd_src = getattr(self._state_0, "joint_qd", None)
            bq_src = getattr(self._state_0, "body_q", None)
            pq_src = getattr(self._state_0, "particle_q", None) if particle_mirror is not None else None
            primary = jq_src or jqd_src or bq_src or pq_src
            if primary is not None:
                stream = wp.get_stream(primary.device)
                if jq_src is not None:
                    self._jq_mirror.refresh(jq_src, stream)
                if jqd_src is not None:
                    self._jqd_mirror.refresh(jqd_src, stream)
                if bq_src is not None:
                    self._bq_mirror.refresh(bq_src, stream)
                if pq_src is not None:
                    particle_mirror.refresh(pq_src, stream)
        # Optional GPU completion timing (diagnostic; default off).
        if getattr(self, "_step_gpu_sync", False):
            t_sync0 = time.monotonic()
            try:
                wp.synchronize_device("cuda:0")
            except Exception:
                pass
            t_sync = (time.monotonic() - t_sync0) * 1000.0
            if not hasattr(self, "_step_gpu_sync_t0"):
                self._step_gpu_sync_t0 = time.monotonic()
                self._step_gpu_sync_acc = 0.0
                self._step_gpu_sync_max = 0.0
                self._step_gpu_sync_n = 0
            self._step_gpu_sync_acc += t_sync
            self._step_gpu_sync_max = max(self._step_gpu_sync_max, t_sync)
            self._step_gpu_sync_n += 1
            if (time.monotonic() - self._step_gpu_sync_t0) >= 1.0 and self._step_gpu_sync_n > 0:
                avg = self._step_gpu_sync_acc / self._step_gpu_sync_n
                self._logger.info(
                    f"[newton-standalone] step GPU sync (1Hz, {self._step_gpu_sync_n} calls): "
                    f"avg={avg:.3f}ms  max={self._step_gpu_sync_max:.3f}ms  "
                    f"(time for the captured cloth+robot graph to finish on the GPU)"
                )
                self._step_gpu_sync_t0 = time.monotonic()
                self._step_gpu_sync_acc = 0.0
                self._step_gpu_sync_max = 0.0
                self._step_gpu_sync_n = 0

        total_ms = (time.monotonic() - t0) * 1000.0
        self._step_ms_acc += total_ms
        if total_ms > self._step_ms_max:
            self._step_ms_max = total_ms
        self._frame_id += 1
        return total_ms

    def _simulate_substeps(self, dt: float = 0.0, captured: bool = True) -> None:
        """Inner substep body. Called either inside CUDA graph capture
        (``captured=True`` — uses ``.assign()`` to swap states, no Python ops)
        or uncaptured (``captured=False`` — uses Python state swap).

        When captured, ``dt`` is taken from ``self._sim_dt_captured`` set at
        capture time (graph baked-in constant).

        Per-substep work is delegated to ``self._substep_body``, a method
        reference bound once in ``_build`` based on the active scene
        composition.  Each branch (plain rigid, franka-VBD-cloth, etc.)
        owns its own model-state mutations so the orchestrator here
        stays minimal and a new solver mode just adds another
        ``_substep_body_*`` method without touching this loop.
        """
        sub_dt = self._sim_dt_captured if captured else (dt / self._sim_substeps)

        for _k in range(self._sim_substeps):
            # Clear BOTH state buffers — the franka cloth demo does this
            # and it matters because state_0 ↔ state_1 swap each substep.
            # After the swap, the new state_0 (= old state_1) carries
            # whatever forces the previous substep wrote; without
            # clearing we leak per-substep stale impulses into the next
            # integration.  Universal across substep bodies.
            self._state_0.clear_forces()
            self._state_1.clear_forces()

            self._substep_body(sub_dt)

            # Swap states. Inside CUDA graph capture we can't do a Python swap;
            # use .assign() (a GPU memcpy) so state_0 holds the latest result.
            if captured:
                # CRITICAL: copy the FULL state forward, not just body/particle.
                # Featherstone writes new joint_q / joint_qd to state_1 each
                # substep (see SolverFeatherstone — its kernels' outputs include
                # state_out.joint_q / state_out.joint_qd). Without rolling
                # those forward, every substep re-reads the same initial
                # joint_q and the captured 10-substep frame effectively does
                # only ONE step of motion (and even that gets thrown away
                # next frame because state_0.joint_q is still the init pose).
                # The user-visible symptom is "robot is frozen" / "commands
                # have no effect" even though apply_commands wrote the target.
                if self._state_0.joint_q is not None and self._state_1.joint_q is not None:
                    self._state_0.joint_q.assign(self._state_1.joint_q)
                if self._state_0.joint_qd is not None and self._state_1.joint_qd is not None:
                    self._state_0.joint_qd.assign(self._state_1.joint_qd)
                if self._state_0.particle_q is not None:
                    self._state_0.particle_q.assign(self._state_1.particle_q)
                    self._state_0.particle_qd.assign(self._state_1.particle_qd)
                if self._state_0.body_q is not None:
                    self._state_0.body_q.assign(self._state_1.body_q)
                    self._state_0.body_qd.assign(self._state_1.body_qd)
            else:
                self._state_0, self._state_1 = self._state_1, self._state_0

    # ------------------------------------------------------------------
    # Per-substep bodies — one branch per solver composition.
    #
    # Selected once in ``_build`` and stored as ``self._substep_body``;
    # the captured CUDA graph bakes whichever branch was active at
    # capture time.  Add a new solver here by writing a new
    # ``_substep_body_<name>`` method and extending ``_pick_substep_body``
    # — keeps assumptions about gravity / contacts / cloth interleave
    # local to each branch instead of compounding in a shared loop.
    # ------------------------------------------------------------------

    def _substep_body_kinematic_control(self, sub_dt) -> None:
        """One rigid-solver substep in *kinematic-control mode*:
        ``particle_count=0``, ``gravity=0``, ``shape_contact_pair_count=0``.

        Default for cloth-free scenes.  The rigid solver tracks
        JOINT_TARGET / velocity-injection set points cleanly without
        fighting an inertial / contact load — no body droop under
        gravity, no wheel-vs-floor contact bouncing.  IsaacSim's
        URDF→USD converter always emits ``PhysicsFixedJoint
        "root_joint"`` between ``/robot`` and ``base_link``, so the
        base is welded to world by the importer; zero gravity inside
        the rigid step is therefore safe (no drift possible).
        """
        pc = self._model.particle_count
        self._model.particle_count = 0
        self._model.gravity.assign(self._gravity_zero)
        self._model.shape_contact_pair_count = 0
        self._adapter.substep(self._model, self._state_0, self._state_1, self._control, sub_dt)
        if getattr(self._state_0, "particle_f", None) is not None:
            self._state_0.particle_f.zero_()
        self._model.particle_count = pc
        self._model.gravity.assign(self._gravity_earth)

    def _substep_body_franka_vbd_cloth(self, sub_dt) -> None:
        """One rigid-solver substep with REAL gravity and shape contacts,
        followed by ``model.collide`` + VBD cloth ``step``.

        Only ``particle_count`` is zeroed (the rigid step ignores
        cloth particles; cloth gets its own step below).  Gravity and
        rigid contacts run normally — Featherstone handles dynamic
        bodies the standard way.  The URDF importer welds
        ``base_link`` to world via ``PhysicsFixedJoint 'root_joint'``
        (``pin_base_to_world=True``), so the robot doesn't drift under
        gravity regardless.  Passive dynamic bodies in the scene
        (hanger, dropped objects) get gravity through the standard
        path.
        """
        pc = self._model.particle_count
        self._model.particle_count = 0
        self._adapter.substep(self._model, self._state_0, self._state_1, self._control, sub_dt)
        if getattr(self._state_0, "particle_f", None) is not None:
            self._state_0.particle_f.zero_()
        self._model.particle_count = pc

        # Cloth step (VBD + collision) — uses real gravity, the cloth
        # solver consults its own particle contacts.
        # NOTE: ``model.collide(state_0, ...)`` intentionally uses the
        # PRE-rigid-step state. With ``integrate_with_external_rigid_solver=
        # True`` VBD reads ``body_q_prev_for_particles = state_in.body_q``
        # (= state_0) and ``body_q_for_particles = state_out.body_q``
        # (= state_1) to infer rigid body velocity for cloth-rigid friction.
        # Forwarding state_1.body_q into state_0.body_q before collide
        # zeroes that inferred velocity and breaks cloth-rigid coupling on
        # any moving body (hanger drop, gripper close, etc.). Substep dt
        # is 1.67 ms — body displacement per substep is small enough that
        # contact normals computed against state_0 are still valid for the
        # cloth penalty force evaluated against state_1 inside VBD.
        self._model.collide(self._state_0, self._contacts)
        # SolverXPBD has no ``integrate_with_external_rigid_solver`` flag
        # (only SolverVBD / SolverStyle3D do).  When XPBD's step() runs
        # against a model with body_count > 0 it re-integrates rigid
        # bodies AND re-projects rigid joint / contact constraints — on
        # the MJW combo path this fights the implicit solve MJW just
        # finished, and the welded ``[base, head, body]`` chassis chain
        # accumulates drift each substep until the wheels detach.
        # Snapshot body state before XPBD's step and restore after; cloth
        # particles still see the pre-XPBD body_q via ``model.collide``
        # above (computed against state_0), so particle-vs-rigid contact
        # is unaffected.  No-op for VBD / Style3D — they don't touch
        # body_q in external-rigid mode.
        _xpbd_freeze = getattr(self, "_cloth_solver_pref", "") == "xpbd" and self._state_1.body_q is not None
        if _xpbd_freeze:
            _saved_body_q = wp.clone(self._state_1.body_q)
            _saved_body_qd = wp.clone(self._state_1.body_qd) if self._state_1.body_qd is not None else None
        self._cloth_solver.step(self._state_0, self._state_1, self._control, self._contacts, sub_dt)
        if _xpbd_freeze:
            self._state_1.body_q.assign(_saved_body_q)
            if _saved_body_qd is not None:
                self._state_1.body_qd.assign(_saved_body_qd)

    def _substep_body_mujoco_xpbd_coupled_proxy(self, sub_dt) -> None:
        """Advance rigid bodies and particles through ``SolverCoupledProxy``.

        Unlike the sequential rigid-then-cloth paths below, proxy momentum
        harvested from the XPBD destination is fed back to the MuJoCo source,
        so a grasped cloth or FEM body pushes back on the gripper instead of
        only being pushed. Collision refresh stays per-substep, matching
        Newton's own MuJoCo+XPBD example.
        """
        self._model.collide(self._state_0, self._contacts)
        self._coupled_solver.step(self._state_0, self._state_1, self._control, self._contacts, sub_dt)

    def _pick_substep_body(self):
        """Bind ``self._substep_body`` based on the active solver
        composition.  Called from ``_build`` after the cloth solver and
        adapter have been constructed; the choice is frozen for the
        lifetime of the engine (and the captured CUDA graph).

        Selection matrix (cloth present, pin_base_mode — see
        ``PinBaseMode``):
          * cloth, *           → ``_substep_body_franka_vbd_cloth``
            (kinematic-control rigid + VBD cloth)
          * no cloth, weld     → ``_substep_body_kinematic_control``
            (rigid step with gravity=0, contacts=0; URDF root_joint
            FixedJoint anchors the base — stationary humanoid / arm
            bench)
          * no cloth, mocap    → ``_substep_body_plain``
            (real gravity + shape contacts, SAME as ``free`` below — the
            base root_joint stays FREE; unlike ``free``, it is NOT left to
            ordinary floor/wheel contact.  ``adapters/mujoco_warp.py``'s
            kinematic-pin machinery clamps it to the authored reference
            every substep, so per-reset base randomization sticks. Using
            ``kinematic_control`` here instead would zero gravity/contacts
            for the WHOLE model — fine for a stationary bench, but it
            freezes every dynamic object in a benchmark manipulation
            scene, which is exactly the case mocap exists for.)
          * no cloth, free     → ``_substep_body_plain``
            (real gravity + shape contacts; root_joint has been
            deactivated so the base is free under physics — mobile
            robot, contact tuning becomes operator's responsibility)
        """
        cloth = self._cloth_solver is not None
        pin_mode = PinBaseMode.of(self)
        # Second half of the base-pin gate.  ``__init__`` refused an
        # adapter that CANNOT clamp the base; this refuses the one that
        # could but DIDN'T — the clamp is only knowable once the solver
        # is built, and without it ``mocap`` means an unpinned base under
        # full gravity with nothing holding it, which looks like a
        # mysterious fall rather than a configuration error.
        if pin_mode.needs_base_clamp:
            _reg = getattr(self._adapter, "base_pin_registered", None)
            if callable(_reg) and not _reg():
                raise RuntimeError(
                    "pin_base_mode=mocap resolved but the adapter registered no base "
                    "pin: no non-Objects FREE root joint was found in the built model "
                    "(the URDF root_joint may not have been deactivated, or the asset "
                    "has no free root). The base would be left under full gravity with "
                    "nothing holding it — see the [mjwarp-adapter] base pin warning "
                    "above."
                )
        # ``keeps_root_weld`` is the question this matrix actually asks: a
        # FixedJoint is holding the base, so the substep can zero gravity
        # and contacts instead of letting physics carry it.
        pinned = pin_mode.keeps_root_weld
        adapter_name = getattr(self._adapter, "name", "")

        # AVBD path: SolverVBD integrates rigid + cloth in one step.
        # No separate cloth_solver, no model.collide between rigid and
        # cloth — adapter.substep does everything inside its solver.step.
        # (lifecycle.py sets self._cloth_solver = None on this path so
        # the ``cloth`` flag above is False; this branch fires before
        # the cloth/pinned/plain matrix.)
        # Coupled path first: it owns BOTH solvers, so none of the flags the
        # matrix below keys on describe it. ``_build_mujoco_xpbd_coupled_proxy``
        # sets ``_cloth_solver = None``, which would otherwise route a cloth
        # scene into the no-cloth branches.
        if self._coupled_solver is not None:
            self._substep_body = self._substep_body_mujoco_xpbd_coupled_proxy
            self._logger.info(
                "[newton-standalone] substep body: mujoco-xpbd-coupled-proxy "
                "(two-way proxy momentum feedback; collision refresh every substep)"
            )
        elif adapter_name == "avbd":
            self._substep_body = self._substep_body_avbd_unified
            self._logger.info(
                f"[newton-standalone] substep body: {adapter_name}-unified "
                "(adapter integrates rigid and deformable state in one solve)"
            )
        elif cloth:
            self._substep_body = self._substep_body_franka_vbd_cloth
            self._logger.info(
                "[newton-standalone] substep body: franka-VBD-cloth "
                "(rigid step under real gravity + shape contacts; cloth step "
                "under real gravity + particle contacts)"
            )
        elif pinned:
            self._substep_body = self._substep_body_kinematic_control
            self._logger.info(
                "[newton-standalone] substep body: kinematic-control "
                "(rigid step with gravity=0, shape_contacts=0; URDF root_joint "
                "FixedJoint anchors the base.)"
            )
        elif pin_mode.needs_base_clamp:
            self._substep_body = self._substep_body_plain
            self._logger.info(
                "[newton-standalone] substep body: plain (pin_base_mode=mocap) "
                "(real gravity + shape contacts, same as the unpinned path — "
                "the base root_joint stays FREE; adapters/mujoco_warp.py's "
                "kinematic-pin machinery clamps it to the authored reference "
                "every substep, so per-reset base randomization sticks without "
                "relying on floor/wheel contact to hold position.)"
            )
        else:
            self._substep_body = self._substep_body_plain
            self._logger.info(
                "[newton-standalone] substep body: plain "
                "(real gravity + shape contacts; root_joint deactivated, "
                "base is mobile under physics — tune geom_solref/solimp on "
                "collision shapes if bouncing instability appears)"
            )

    def _substep_body_plain(self, sub_dt) -> None:
        """One rigid-solver substep under real gravity + shape contacts.
        Used for mobile-robot scenes (``pin_base_to_world: false``) where
        floor contact through the wheels keeps the chassis anchored.
        Pair with contact tuning (geom_solref/solimp) on collision
        shapes for the robot's mass distribution.
        """
        self._adapter.substep(self._model, self._state_0, self._state_1, self._control, sub_dt)

    def _substep_body_avbd_unified(self, sub_dt) -> None:
        """One unified AVBD step — rigid + cloth advance together.

        The AVBD adapter constructed a single ``SolverVBD`` instance with
        ``integrate_with_external_rigid_solver=False``, so its
        ``solver.step`` integrates rigid bodies (via Augmented VBD) and
        cloth particles (via VBD) in one pass.  No ``model.collide`` /
        ``cloth_solver.step`` plumbing is needed — that whole inter-solver
        coupling layer (which the Featherstone+VBD path uses) is replaced
        by the unified solver's internal contact pipeline.

        ``particle_count`` and ``shape_contact_pair_count`` stay at their
        finalize-time values; the solver's contact pass handles both
        cloth-vs-rigid and rigid-vs-rigid.
        """
        self._adapter.substep(self._model, self._state_0, self._state_1, self._control, sub_dt)

    def tick_extras(self) -> None:
        """Per-tick work that ALWAYS fires (every physics step).

        Kit-free newton-standalone has no Fabric writeback to drive.
        Kept as a stats heartbeat: increment the tick counter and emit
        the 1 Hz stats log.

        When the inline OVRtx visualizer is attached
        (``physics_engine_visualizer:=ovrtx``), this also records the
        per-tick CUDA event the OVRtx render thread waits on — see
        ``docs/ovrtx_sync.md``.
        """
        if self._model is None:
            return
        tw0 = time.monotonic()

        # Inline OVRtx handshake: cheap (~µs).  Records on Warp's current
        # stream — the same stream the Newton CUDA graph just launched on,
        # so the OVRtx thread's stream-side wait sees a committed body_q.
        if self._physics_step_event is not None:
            wp.record_event(self._physics_step_event)

        # Recording uses the particle mirror rather than ``state_0.numpy()``.
        # It therefore has exactly the same one-tick delay as /joint_states
        # and /tf_scene and never inserts a synchronous GPU read in the loop.
        particle_mirror = self._update_record_snapshot_mirror()
        if particle_mirror is not None:
            particle_snapshot = particle_mirror.read()
            body_snapshot, body_paths = self.get_body_transforms()
            self._call_plugin(
                "on_record_snapshot",
                particle_snapshot,
                body_snapshot,
                body_paths,
                self._sim_time_acc,
                self._plugin_ctx(),
            )

        wb_ms = (time.monotonic() - tw0) * 1000.0
        self._writeback_ms_acc += wb_ms
        if wb_ms > self._writeback_ms_max:
            self._writeback_ms_max = wb_ms

        self._tick_count += 1
        # 1Hz wall-clock stats log
        if (time.monotonic() - self._t_log_start) >= 1.0:
            self._print_stats()

        # Debug-only publishers run at ``render_hz``, not physics_hz ×
        # substeps — RViz can't render any faster than render_hz anyway,
        # and publishing every physics tick wastes wire bandwidth + GIL
        # time inside the hot path.  Shared gate so both pubs land on the
        # same beat; advances in sim-time so it's not skewed by realtime
        # factor or wall-clock jitter.  ``render_hz <= 0`` (headless) =>
        # publish every tick (cheap fallback for the rare case where a
        # user wants raw cadence).
        _hz = float(self._render_hz)
        _due = _hz <= 0.0 or self._sim_time_acc >= self._next_debug_pub_time
        if _due and (
            self._deformable_pub is not None or self._deformable_pc_pub is not None or self._object_pub is not None
        ):
            # Debug-only: publish deformable / cloth surface as a
            # TRIANGLE_LIST Marker.  Gated by
            # ``newton.debug.pub_deformable_marker`` (default off); the
            # ``is None`` branch costs one attribute lookup per tick in
            # production.
            if self._deformable_pub is not None:
                try:
                    pq = getattr(self._state_0, "particle_q", None)
                    if pq is not None:
                        self._deformable_pub.publish(pq, self._sim_time_acc)
                except Exception as exc:  # noqa: BLE001
                    self._logger.warn(f"[newton-standalone] deformable marker publish failed: {exc!r}")
                    self._deformable_pub = None  # disarm on first failure to avoid log spam

            # Debug-only: publish raw particle_q as a sensor_msgs/PointCloud2.
            # Gated by ``newton.debug.pub_deformable_pointcloud``; same
            # disarm-on-failure pattern as the marker pub above.
            if self._deformable_pc_pub is not None:
                try:
                    pq = getattr(self._state_0, "particle_q", None)
                    if pq is not None:
                        self._deformable_pc_pub.publish(pq, self._sim_time_acc)
                except Exception as exc:  # noqa: BLE001
                    self._logger.warn(f"[newton-standalone] deformable pointcloud publish failed: {exc!r}")
                    self._deformable_pc_pub = None

            # Debug-only: publish free-joint rigid object markers.  Same
            # gating and disarm-on-failure pattern as the deformable pub.
            if self._object_pub is not None:
                try:
                    bq = getattr(self._state_0, "body_q", None)
                    if bq is not None:
                        self._object_pub.publish(bq, self._sim_time_acc)
                except Exception as exc:  # noqa: BLE001
                    self._logger.warn(f"[newton-standalone] object marker publish failed: {exc!r}")
                    self._object_pub = None

            if _hz > 0.0:
                # Advance by a fixed period; tolerate large physics-side
                # gaps (e.g. after a long pause) by snapping forward
                # rather than letting the gate flood once it catches up.
                period = 1.0 / _hz
                self._next_debug_pub_time = max(self._next_debug_pub_time + period, self._sim_time_acc + period * 0.5)

    def attach_physics_event(self, event: Any) -> None:
        """Attach a ``wp.Event`` recorded after every physics step.

        Used by the inline OVRtx visualizer so its render thread can
        ``wp.Stream.wait_event(event)`` to gate the per-frame sync kernel
        on the latest physics commit, with no CPU-side wait on the physics
        thread.  Idempotent — pass ``None`` to detach.
        """
        self._physics_step_event = event

    def sync_visual_state(self) -> None:
        """Explicit force-sync hook.  No-op on Kit-free
        newton-standalone — there is no Fabric or USD-side mirror to
        push state into; the Newton GL viewer reads ``state_0`` GPU
        arrays directly via ``viewer.log_state()`` each render frame.
        Kept for ``PhysicsEngine`` interface compatibility.
        """
        return

    def shutdown(self) -> None:
        # The AABB publisher's node IS the latch on /rlinf/body_aabb_local —
        # ROS 2 durability is writer-side, so the retained writer is what lets
        # a wrapper attaching after engine build still read the table.  It is
        # therefore torn down here and nowhere earlier.  Best effort: a failed
        # teardown must not stop the rest of shutdown.
        try:
            from genie_sim_engine_newton.engine.newton.benchmark_publisher import (  # noqa: PLC0415
                shutdown as shutdown_benchmark_publisher,
            )

            shutdown_benchmark_publisher()
        except Exception:  # noqa: BLE001
            pass
        self._model = None
        self._stage = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
