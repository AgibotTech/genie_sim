# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""MuJoCo-Warp solver adapter.

Encapsulates the pre-construction PD-drive setup,
``SolverMuJoCo`` construction with internal-buffer sizes, and seeding of
``control.joint_target_q`` with the init pose.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import newton
import warp as wp

from genie_sim_engine_newton.engine.newton.adapters.base import (
    SolverAdapter,
    apply_actuator_force_cap,
    resolve_mjc_impratio_cone,
    resolve_mjc_iterations,
)

from .base_mocap import BaseMocapResetMixin

# Threshold for treating ``joint_target_ke`` as "unset" by the URDF.
#
# Isaac Sim's URDF importer writes a vestigial ``DriveAPI`` with
# ``stiffness≈625`` on every revolute joint regardless of effort values
# in the source URDF.  Newton's ``add_usd`` reads that 625 into
# ``model.joint_target_ke``.  Anything below this threshold is treated as
# "the importer wrote a placeholder; substitute the launcher default."
#
# 626 is chosen to catch exactly the vestigial 625 while preserving
# intentional values >= 626 from XACRO ``<drive>`` tags (e.g. head ke=1000,
# arm-tip ke=8000).
_VESTIGIAL_KE_THRESHOLD = 626.0

# Default PD gains when the launcher YAML doesn't set
# ``mujoco_pd_ke`` / ``mujoco_pd_kd`` AND the URDF doesn't author
# ``joint_target_ke`` / ``joint_target_kd``.
#
# Fallback only — used when ``joint_effort_limit`` is also unavailable
# so we can't scale per-joint.  The preferred path is per-joint
# ``kp = effort_limit * _KE_TO_EFFORT_FACTOR`` so the spring never
# demands more torque than the actuator can deliver at small errors.
_DEFAULT_KE = 100.0
_DEFAULT_KD = 10.0

# Per-joint kp = effort_limit * _KE_TO_EFFORT_FACTOR.
# 10 means a 0.1 rad error demands exactly ``effort_limit`` — small
# corrections track linearly; large goals saturate during the
# transient and unclamp as error shrinks, which is the right behavior
# for position control on torque-limited actuators.  Going higher
# (e.g. 50) gives stiffer hold but saturates earlier.  Going lower
# (e.g. 2) gives softer hold but is unmistakable to read in RViz.
_KE_TO_EFFORT_FACTOR = 10.0

# kd = _KD_TO_KE_FRACTION * sqrt(kp).
# Critical damping for a unit-inertia link: kd = 2 * sqrt(kp).  We use
# 2.0 to target critical damping under the worst-case assumption that
# m_eff ≈ 1 kg·m².  Most arm / body links are well under that, so we
# end up slightly over-damped — desirable, because mjwarp's
# ``implicitfast`` integrator is stable past nominal damping ratios
# and over-damping suppresses the ringing that follows every wbc
# step-input.  An under-damped value (< 1.0) makes the joint
# oscillate around each new goal, and the second-order modes from
# 42 joints fighting each other look like an explosion to the
# operator even when no single joint diverges.
_KD_TO_KE_FRACTION = 2.0

# Mimic-master gains for the gripper master joint (the one that owns
# the open/close target via /joint_command).  Tuned to match the
# reference G2 MJCF (``gainprm="5" biasprm="0 -5"`` on
# ``position_idx31_gripper_l_inner_joint1``) — extremely soft, because
# the kinematic coupling between gripper joints comes from the mjwarp
# equality constraints Newton's importer derives from
# Gripper tuning constants — DELETED.  All gripper-related values
# (master kp/kd, master effort cap, mimic eq solref/solimp, joint
# armature/damping/frictionloss) now come from
# ``physics_params.yaml::usd_drive_api.gripper`` (see
# ``common/params.py:GripperDriveParams`` for the field list).  The
# adapter reads ``self._physics_params.drive_gripper.*`` everywhere
# a gripper-specific value is needed.  This removes the
# ``_GRIPPER_MASTER_KE/KD``, ``_MIMIC_AUTHORED_KE_FLOOR``,
# ``_MIMIC_JOINT_ARMATURE/FRICTIONLOSS/DAMPING`` constants and the
# accompanying overlay-snapshot logic — the YAML is the single
# source of truth.

# Default per-DOF actuator effort limit when the URDF didn't author
# ``<limit effort=...>`` for a joint.  mjwarp converts
# ``joint_effort_limit`` into mujoco's per-joint ``actfrclimited=True``
# + ``actfrcrange=(-eff, +eff)``; if eff is 0 (the Newton default for
# unspecified URDF effort), the actuator force is clamped to 0 N·m and
# the robot behaves like jelly regardless of how high ``ke`` is.  This
# value clamps "effectively unauthored" effort limits (anything <=
# ``_EFFORT_LIMIT_FLOOR``) up to ``_DEFAULT_EFFORT_LIMIT`` so the PD
# can actually deliver torque.  500 N·m is large enough for a humanoid
# arm but bounded enough that a runaway PD doesn't impart absurd
# velocity to the floating base.
_DEFAULT_EFFORT_LIMIT = 500.0
_EFFORT_LIMIT_FLOOR = 1.0  # anything <= this is treated as "unauthored"

# MuJoCo internal constraint-buffer sizes.
#
#   njmax    = max constraint Jacobian rows (contacts + limits + equality)
#   nconmax  = max active contact pairs
#
# Under-allocation produces ``nefc overflow`` / ``nconmax overflow`` spam
# and silently drops contacts.  Sized with 2x headroom over the observed
# peak on the supermarket scene (peak njmax ~750).
_DEFAULT_NJMAX = 1024
_DEFAULT_NCONMAX = 512


class MuJoCoWarpAdapter(BaseMocapResetMixin, SolverAdapter):
    def __init__(
        self,
        *,
        pd_ke: float = 0.0,
        pd_kd: float = 0.0,
        njmax: int = _DEFAULT_NJMAX,
        nconmax: int = _DEFAULT_NCONMAX,
        effort_limit: float = 0.0,
        save_to_mjcf: str = "",
        physics_params: Any = None,
        joint_params: Any = None,
        physics_hz: float = 0.0,
        imu_cfg: Any = None,
        enable_multiccd: Optional[bool] = None,
    ) -> None:
        # ``pd_ke`` / ``pd_kd`` from the launcher remain the GLOBAL
        # blanket override — non-zero means "force this on every DOF
        # regardless of class".  Useful for tuning experiments and
        # CI smoke runs.  Zero means "use ``physics_params`` per-class
        # tables" (the production path); see ``prepare_model``.
        self._pd_ke = float(pd_ke) if pd_ke > 0.0 else _DEFAULT_KE
        self._pd_kd = float(pd_kd) if pd_kd > 0.0 else _DEFAULT_KD
        self._pd_ke_explicit = pd_ke > 0.0
        self._pd_kd_explicit = pd_kd > 0.0
        self._effort_limit = float(effort_limit) if effort_limit > 0.0 else _DEFAULT_EFFORT_LIMIT
        self._effort_limit_floor = _EFFORT_LIMIT_FLOOR
        self._njmax = int(njmax)
        self._nconmax = int(nconmax)
        self._save_to_mjcf = str(save_to_mjcf) if save_to_mjcf else ""
        # ``newton.mujoco_enable_multiccd`` from the SCENE yaml, tri-state.
        # ``None`` (absent) = leave SolverMuJoCo on its own default, which is
        # ``False`` in newton 1.4.0.  It lives in the scene yaml rather than
        # physics_params.yaml because it is a statement about the SCENE's
        # geometry, not a global contact preference: MultiCCD needs
        # zero-margin convex pairs, which is a property the scene author
        # arranges, for example with zero-margin synthetic grasp boxes.
        # The multi-env adapter reads the
        # same key and defaults the other way — see
        # ``adapters/__init__.py::make_adapter``.
        self._enable_multiccd = None if enable_multiccd is None else bool(enable_multiccd)
        # ``PhysicsParams`` from ``physics_params.yaml`` — when provided,
        # ``prepare_model`` drives per-DOF gains from the per-class
        # tables (body/arm/head → ``art_default``, chassis_drive →
        # ``art_chassis_drive``, chassis_steer → ``art_chassis_steer``,
        # gripper master → ``drive_gripper.master_*``) using the shared
        # name classifier in ``common.joint_classification``.  When
        # ``None`` the adapter falls back to the unscaled effort × 10
        # heuristic (kept so ``test_newton_solver.py`` and any
        # standalone scripts that don't load the YAML still run).
        # Required by the engine path — ``NewtonHeadlessEngine`` will
        # always pass its loaded params.  The wiring guarantees newton-
        # standalone gets the same per-class gains the isaac_newton
        # wrapper would apply at runtime, so behaviour matches across
        # engines for any scene whose joint names match the canonical
        # ``body_*`` / ``arm_*`` / ``head_*`` / ``gripper_*`` /
        # ``chassis_*wheel*`` regex set.
        self._physics_params = physics_params
        self._joint_params = joint_params
        # ``physics_hz`` (outer-frame rate, Hz) is captured here so
        # ``build_solver`` can compute the LIVE substep dt =
        # ``1 / (physics_hz × sim_substeps)`` and post-process the
        # dumped MJCF to carry that value in ``<option timestep>`` plus
        # ``<custom>`` numerics — see
        # ``common/mjcf_postprocess.py:augment_mjcf_timing``.  When 0
        # (e.g. ``test_newton_solver.py`` invocations that don't pass
        # the rate), the post-process is skipped and the dump inherits
        # MuJoCo's 0.002 s default (no-params fallback path).
        self._physics_hz = float(physics_hz) if physics_hz > 0.0 else 0.0
        # IMU sites/sensors to inject into the dumped ``robot_runtime.xml``,
        # from the scene yaml's ``imu:`` block.  ``None`` when the scene
        # declares no IMU — the ``add_imu_sites_and_sensors`` post-process
        # step then no-ops.  Consumed ONLY by the save_to_mjcf dump path;
        # the live mjwarp solver is unaffected (this is a static XML
        # artifact for downstream pure-MuJoCo consumers).
        self._imu_cfg = imu_cfg
        self._target_buffer: Optional[wp.array] = None
        self._solver: Any = None

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mujoco-warp"

    @property
    def supports_cloth(self) -> bool:
        # MuJoCo-Warp itself has no cloth solver, but VBD cloth is a
        # solver-INDEPENDENT extension that composes with whichever
        # rigid solver is active.  ``_substep_body_franka_vbd_cloth``
        # delegates the rigid step to ``self._adapter.substep`` and runs
        # ``model.collide`` + ``cloth_solver.step`` afterwards, so the
        # cloth path doesn't care which adapter computed body_q.
        # Allow the composition; if it produces issues, gate it behind
        # a launcher flag rather than disabling outright.
        return True

    # ------------------------------------------------------------------
    # Pre-finalize
    # ------------------------------------------------------------------

    def register_custom_attributes(self, builder: Any) -> None:
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    # ------------------------------------------------------------------
    # Helpers: per-class gain selection (shared classifier)
    # ------------------------------------------------------------------

    @staticmethod
    def _params_class_for_dof(
        label: str,
        low: Optional[float],
        high: Optional[float],
    ) -> str:
        """Return the ``JK_*`` joint class for one DOF, splitting
        chassis wheel into ``JK_CHASSIS_DRIVE`` / ``JK_CHASSIS_STEER``
        from the Newton-side limits (radians; threshold ≈ 12 rad,
        matching the 700-degree threshold used on the USD-side
        classifier).
        """
        from genie_sim_engine_newton.common.joint_classification import (  # noqa: PLC0415
            JK_CHASSIS_DRIVE,
            JK_CHASSIS_STEER,
            JK_CHASSIS_WHEEL,
            classify_joint_by_name,
            is_chassis_wheel_free,
        )

        # Newton model labels can be path-like (``World/foo/joint``);
        # take the leaf so the regex anchors fire.
        short = label.rsplit("/", 1)[-1] if "/" in label else label
        kind = classify_joint_by_name(short)
        if kind != JK_CHASSIS_WHEEL:
            return kind
        # Convert the USD-degree threshold (700) to Newton's radian
        # unit (≈ 12.217 rad).  Same drive-vs-steer cutoff as
        # ``classify_joint`` on the USD side, just expressed in the
        # native unit so we don't need a degree round-trip on every
        # DOF.
        return JK_CHASSIS_DRIVE if is_chassis_wheel_free(low, high, threshold=12.217) else JK_CHASSIS_STEER

    def _per_class_gains(self) -> Optional[Dict[str, tuple]]:
        """Return ``{JK_*: (ke, kd, effort)}`` from ``self._physics_params``.

        Returns ``None`` when ``physics_params`` is unset — caller
        falls back to the unscaled ``effort × 10`` heuristic.

        Source layers from ``physics_params.yaml``:
          * body / arm / head        ← ``articulation_view_runtime.default``
          * chassis_drive            ← ``articulation_view_runtime.chassis_drive``
          * chassis_steer            ← ``articulation_view_runtime.chassis_steer``
          * gripper master           ← ``usd_drive_api.gripper.master_*``
          * gripper follower         ← (0, 0, 0) — constraint-driven, no actuator

        Layer 2 (``articulation_view_runtime``) is what the isaac_newton
        wrapper applies at runtime via ``set_dof_stiffnesses`` — using
        it here gives newton-standalone the SAME gains the wrapper
        runs, so a scene loaded under either engine behaves the same.
        The gripper master is the one DOF that takes its kp/kd from
        Layer 1 because that's where the gripper tuning lives
        (see the long tuning comment in
        ``physics_params.yaml::usd_drive_api.gripper``).
        """
        from genie_sim_engine_newton.common.joint_classification import (  # noqa: PLC0415
            JK_ARM,
            JK_ARM_MID,
            JK_ARM_SHOULDER,
            JK_ARM_WRIST,
            JK_BODY,
            JK_CHASSIS_DRIVE,
            JK_CHASSIS_STEER,
            JK_GRIPPER,
            JK_HEAD,
            JK_PASSIVE,
        )

        p = self._physics_params
        if p is None:
            return None
        art_def = p.art_default
        art_cd = p.art_chassis_drive
        art_cs = p.art_chassis_steer
        art_body = p.art_body
        art_head = p.art_head
        art_arm_sh = p.art_arm_shoulder
        art_arm_md = p.art_arm_mid
        art_arm_wr = p.art_arm_wrist
        drv_grip = p.drive_gripper
        # Per-sub-class actuator gains.  Reference G2 MJCF tiers
        # body/head/arm by joint inertia; flattening them
        # to ``art_default`` (5e4 / 5e3) trips the static-diff tool
        # as > 2× divergent on every joint.  Generic
        # ``JK_ARM`` (no shoulder/mid/wrist match) falls through
        # to ``art_default`` so any robot with arm joints outside
        # the canonical 1-7 numbering still gets a working tuning.
        return {
            JK_BODY: (art_body.kp, art_body.kd, art_body.max_effort),
            JK_HEAD: (art_head.kp, art_head.kd, art_head.max_effort),
            JK_ARM: (art_def.kp, art_def.kd, art_def.max_effort),
            JK_ARM_SHOULDER: (art_arm_sh.kp, art_arm_sh.kd, art_arm_sh.max_effort),
            JK_ARM_MID: (art_arm_md.kp, art_arm_md.kd, art_arm_md.max_effort),
            JK_ARM_WRIST: (art_arm_wr.kp, art_arm_wr.kd, art_arm_wr.max_effort),
            JK_CHASSIS_DRIVE: (art_cd.kp, art_cd.kd, art_cd.max_effort),
            JK_CHASSIS_STEER: (art_cs.kp, art_cs.kd, art_cs.max_effort),
            # ``JK_GRIPPER`` is split master-vs-follower inside the
            # apply loop (master = drv_grip; follower = zeros) since
            # we need the mimic-follower set to do the split.
            JK_GRIPPER: (drv_grip.master_stiffness, drv_grip.master_damping, art_def.max_effort),
            # Passive scene rigid bodies (auto-created free joints):
            # zero PD, zero effort.  MuJoCo runs these as un-actuated
            # DOFs — gravity, contacts, and joint reactions handle them
            # via the integrator.
            JK_PASSIVE: (0.0, 0.0, 0.0),
        }

    def _per_class_passive_joint(self) -> Optional[Dict[str, tuple]]:
        """Return ``{JK_*: (dof_damping, dof_frictionloss, armature)}`` from
        ``self._physics_params``.

        Source layer for passive-joint values:
          * body / arm / head        ← ``usd_drive_api.default_revolute``
          * chassis_drive / steer    ← ``usd_drive_api.chassis_*_joint``
          * gripper                  ← (None, None, None) — skipped, the
                                       assemble overlay already authors
                                       ``mjc:damping`` + ``physxJoint:
                                       armature`` per gripper joint and
                                       we must not stomp it.

        Returns ``None`` when ``physics_params`` is unset (no-params
        adapter path).

        Why these are independent of the actuator gains in
        ``_per_class_gains``
        ---------------------------------------------------------
        These are passive (no actuator involvement) — applied every
        step regardless of any drive.  In MuJoCo terms,
        ``dof_damping[i] = const`` adds ``-c·qvel[i]`` to the joint
        force, ``dof_frictionloss[i] = const`` adds a stick-slip
        friction model, and ``dof_armature[i] = const`` adds rotor
        inertia to the joint's effective mass (improves solver
        conditioning + raises kp/dt² stability headroom; lowers the
        closed-loop natural frequency on light distal joints).
        They're independent of (and additive with) the actuator's
        PD output, which is why this lookup is separate from
        ``_per_class_gains``.
        """
        from genie_sim_engine_newton.common.joint_classification import (  # noqa: PLC0415
            JK_ARM,
            JK_ARM_MID,
            JK_ARM_SHOULDER,
            JK_ARM_WRIST,
            JK_BODY,
            JK_CHASSIS_DRIVE,
            JK_CHASSIS_STEER,
            JK_GRIPPER,
            JK_HEAD,
            JK_PASSIVE,
        )

        p = self._physics_params
        if p is None:
            return None

        # Map each sub-class to its dedicated DriveParams block.  Fall
        # back to ``drive_default_revolute`` for the generic ``JK_ARM``
        # bucket (joints whose index didn't classify into shoulder /
        # mid / wrist).
        def _t(d):
            return (d.dof_damping, d.dof_frictionloss, d.armature)

        return {
            JK_BODY: _t(p.drive_body),
            JK_HEAD: _t(p.drive_head),
            JK_ARM: _t(p.drive_default_revolute),
            JK_ARM_SHOULDER: _t(p.drive_arm_shoulder),
            JK_ARM_MID: _t(p.drive_arm_mid),
            JK_ARM_WRIST: _t(p.drive_arm_wrist),
            JK_CHASSIS_DRIVE: _t(p.drive_chassis_drive_joint),
            JK_CHASSIS_STEER: _t(p.drive_chassis_steer_joint),
            # JK_GRIPPER handled by the assemble overlay — don't touch.
            JK_GRIPPER: (None, None, None),
            # JK_PASSIVE: don't touch the per-DOF damping/frictionloss
            # for passive scene rigid bodies — leaving them at Newton's
            # finalize defaults lets the body fall and respond naturally.
            # Adding damping here would slow free-fall artificially.
            JK_PASSIVE: (None, None, None),
        }

    # ------------------------------------------------------------------
    # Post-finalize, pre-solver
    # ------------------------------------------------------------------

    def prepare_model(
        self,
        model: Any,
        logger: Any,
        mimic_followers: Optional[Dict[str, list]] = None,
        robot_prefix: str = "",  # noqa: ARG002 — accepted for ABC compatibility
    ) -> None:
        """Author per-DOF PD gains + effort cap on the Newton model
        before ``SolverMuJoCo(model)`` reads them.

        Two paths:

          A. **``physics_params`` provided** (production engine path).
             Every DOF is classified via
             ``common.joint_classification.classify_joint_by_name``
             and looked up in the per-class table built by
             ``_per_class_gains`` (sourced from ``physics_params.yaml``).
             Body / arm / head / chassis_drive / chassis_steer all
             get their kp/kd/max_effort from the matching block;
             gripper masters get ``usd_drive_api.gripper.master_*``;
             gripper followers get zeros (constraint-driven).  Any DOF
             that classifies as ``JK_OTHER`` is collected into a list
             and a single ``RuntimeError`` is raised at the end with
             every offender named — matches the policy in
             ``kit/stage.py:_init_articulation``.  We hard-fail rather
             than silently leaving the offending joint at the URDF
             importer's vestigial default (typically ``ke ≈ 625``,
             ``effort ≈ 0``) which is a near-certain source of
             "joint-doesn't-track-commands" bugs.

          B. **No ``physics_params``** (no-params / standalone-tool path).
             Falls back to the per-effort heuristic ``ke = effort * 10``
             with critical-damping ``kd = 2·√ke``.  Used by
             ``test_newton_solver.py`` invocations that don't load the
             YAML, by operator scripts, and as the safety net
             when the YAML can't be resolved (e.g. share dir missing
             during early development).  The launcher's explicit
             ``mujoco_pd_ke`` / ``mujoco_pd_kd`` blanket override
             still wins over both paths when set, for tuning
             experiments.

        After per-class authoring, the mimic-suppression block runs
        unchanged — when ``mimic_followers`` is non-empty, follower
        actuators are muted (mode=NONE) and the master's authored
        ``ke`` is normalised to the soft mimic value if it looks
        un-overlaid.  See the ``# 4.`` block below.
        """
        mode_arr = model.joint_target_mode
        ke_arr = model.joint_target_ke
        kd_arr = model.joint_target_kd
        effort_arr = getattr(model, "joint_effort_limit", None)
        if mode_arr is None or ke_arr is None or kd_arr is None:
            logger.warn(
                "[mjwarp-adapter] joint_target_{mode,ke,kd} array is None; " "SolverMuJoCo will not apply PD drives"
            )
            return

        import numpy as np  # local; module-top doesn't import np

        mode = mode_arr.numpy().copy()
        ke = ke_arr.numpy().copy()
        kd = kd_arr.numpy().copy()
        effort = effort_arr.numpy().copy() if effort_arr is not None else None
        ke_authored_snapshot = ke.copy()  # before any fallback writes

        # ``JointIndex`` resolves the q-vs-qd asymmetry once. ``ke`` /
        # ``kd`` / ``effort`` / ``mode`` are all dof-sized, so we
        # always index them via ``s.qd_start``. See
        # ``engine/newton/joint_index.py`` for the full rationale.
        from genie_sim_engine_newton.engine.newton.joint_index import JointIndex  # noqa: PLC0415

        jindex = JointIndex(model)
        limit_lo_arr = getattr(model, "joint_limit_lower", None)
        limit_hi_arr = getattr(model, "joint_limit_upper", None)
        limit_lo = limit_lo_arr.numpy() if limit_lo_arr is not None else None
        limit_hi = limit_hi_arr.numpy() if limit_hi_arr is not None else None

        # ------------------------------------------------------------
        # Build per-DOF kind list (one entry per CONTROLLABLE DOF in
        # ``ke``).  Each Newton joint may span multiple DOFs; the
        # classifier acts on the joint label, so we expand the per-
        # joint kind across the joint's qd-slice.
        # ------------------------------------------------------------
        from genie_sim_engine_newton.common.joint_classification import (  # noqa: PLC0415
            JK_GRIPPER,
            JK_OTHER,
            JK_PASSIVE,
        )

        dof_kind: list = [JK_OTHER] * len(ke)
        unknown_joint_names: list = []
        passive_joint_names: list = []  # log-only — these are intentional, not errors
        gripper_follower_set = set()
        gripper_master_set = set()
        if mimic_followers:
            gripper_follower_set = {name for _master, lst in mimic_followers.items() for (name, _c, _o) in lst}
            gripper_master_set = set(mimic_followers.keys())
        # Kept for ``_cap_gripper_master_force``, which runs after compile and
        # no longer has ``mimic_followers`` in scope.  Same content as
        # ``gripper_annotations.gripper_masters(mimic_followers)`` -- reusing the
        # set already derived here rather than deriving it twice.
        self._mimic_masters = gripper_master_set

        for s in jindex.slices():
            if not s.label or not s.is_actuated:
                continue
            if s.qd_start + s.dof_count > len(ke):
                continue
            if s.is_free:
                # Scene bodies created with ModelBuilder.add_body() receive a
                # 6-DOF free joint, and a caller-provided body label also gives
                # that joint a descriptive name (for example
                # ``chef_wok_free_joint``).  ``is_actuated`` is ``dof_count > 0``
                # so a free joint reaches this classifier, and the name-based
                # passive regexes below do not match a label like that — the
                # scene body ends up PD-actuated toward target 0.  Joint type is
                # the authoritative signal: free bodies must remain unactuated
                # regardless of what they are called.
                passive_joint_names.append(s.name)
                for k in range(s.dof_count):
                    dof_kind[s.qd_start + k] = JK_PASSIVE
                continue
            # Pull joint-level limit for chassis drive/steer split
            # (use the first DOF's qd index — adequate for revolute
            # wheels which are 1-DOF anyway).
            lo = float(limit_lo[s.qd_start]) if limit_lo is not None and s.qd_start < len(limit_lo) else None
            hi = float(limit_hi[s.qd_start]) if limit_hi is not None and s.qd_start < len(limit_hi) else None
            kind = self._params_class_for_dof(s.label, lo, hi)
            if kind == JK_OTHER:
                unknown_joint_names.append(s.name)
                continue
            if kind == JK_PASSIVE:
                # Auto-created free joint for a scene rigid body
                # (hanger, dropped object, etc.).  Mark every DOF in
                # the joint's slice as PASSIVE — the gain table maps
                # JK_PASSIVE to (ke=0, kd=0, effort=0), so MuJoCo
                # leaves these DOFs unactuated and Newton's
                # integrator handles them via gravity + contacts.
                passive_joint_names.append(s.name)
                for k in range(s.dof_count):
                    dof_kind[s.qd_start + k] = JK_PASSIVE
                continue
            for k in range(s.dof_count):
                dof_kind[s.qd_start + k] = kind

        if passive_joint_names:
            logger.info(
                f"[mjwarp-adapter] classified {len(passive_joint_names)} joint(s) as "
                f"JK_PASSIVE (auto-created free joints for scene rigid bodies; "
                f"PD gains zeroed, integrator handles dynamics): {passive_joint_names}"
            )

        # ------------------------------------------------------------
        # Hard-fail policy: any joint whose name didn't match a known
        # class is fatal.  Silently inheriting the URDF importer's
        # vestigial defaults is the kind of "this joint sometimes
        # doesn't track commands" symptom that takes hours to bisect
        # — same policy ``kit/stage.py:_init_articulation`` enforces.
        # We collect the full offender list before raising so the
        # operator sees every offender in one pass.
        # ------------------------------------------------------------
        if unknown_joint_names:
            raise RuntimeError(
                "[mjwarp-adapter] joint_classification could not bucket "
                f"{len(unknown_joint_names)} joint(s): {unknown_joint_names}.  "
                f"Add a regex to ``common/joint_classification.py`` "
                f"(``_RE_BODY`` / ``_RE_ARM`` / ``_RE_HEAD`` / ``_RE_GRIPPER`` / "
                f"``_RE_CHASSIS_WHEEL``) so each joint gets a known kind, "
                f"then re-launch.  Aborting startup rather than silently "
                f"applying the unscaled effort×10 fallback (which produced "
                f"e.g. body kp=500 / arm kp=180 — too weak to hold the "
                f"robot under gravity)."
            )

        # ------------------------------------------------------------
        # Per-class gain authoring.  Production path runs when
        # ``physics_params`` is set; otherwise the unscaled effort×10
        # fallback runs (see method docstring path B).
        # ------------------------------------------------------------
        per_class = self._per_class_gains()

        # 1. Effort floor (always) — URDF effort=0 → ``_effort_limit``
        # so mjwarp's ``actfrcrange`` clamp doesn't zero the actuator.
        n_effort_overridden = 0
        if effort is not None:
            mask = effort <= self._effort_limit_floor
            n_effort_overridden = int(mask.sum())
            effort[mask] = self._effort_limit

        if self._pd_ke_explicit or self._pd_kd_explicit:
            # Launcher blanket override — wins over both paths.  Used
            # for tuning experiments and CI smoke runs that want a
            # fixed gain everywhere.
            if self._pd_ke_explicit:
                ke[:] = self._pd_ke
            if self._pd_kd_explicit:
                kd[:] = self._pd_kd
            n_authored_per_class = 0
            n_ke_fallback = int(self._pd_ke_explicit) * len(ke)
            n_kd_fallback = int(self._pd_kd_explicit) * len(kd)
        elif per_class is not None:
            # Production path — drive every DOF from its class table.
            # Precompute a per-DOF "owning joint short name" so the
            # gripper master-vs-follower split below is a single dict
            # lookup, not a linear walk over joint slices per DOF.
            dof_to_owner_name: list = [""] * len(ke)
            for s in jindex.slices():
                for k in range(s.dof_count):
                    if s.qd_start + k < len(ke):
                        dof_to_owner_name[s.qd_start + k] = s.name

            n_authored_per_class = 0
            for i in range(len(ke)):
                kind = dof_kind[i]
                # Resolve the master-vs-follower split for grippers.
                if kind == JK_GRIPPER:
                    short = dof_to_owner_name[i]
                    if short in gripper_follower_set:
                        ke[i] = 0.0
                        kd[i] = 0.0
                        continue
                    # Treat anything else under JK_GRIPPER as a master:
                    # masters are kp/kd from drv_grip (the mimic-suppression
                    # block below will refine if needed).
                # Apply class gains
                cls_gains = per_class.get(kind)
                if cls_gains is None:
                    # Shouldn't happen — unknown_joint_names would have
                    # raised above.  Belt-and-suspenders no-op.
                    continue
                kp_c, kd_c, eff_c = cls_gains
                ke[i] = float(kp_c)
                kd[i] = float(kd_c)
                if effort is not None and eff_c > 0.0 and effort[i] < eff_c:
                    # Raise the effort cap if the class table demands
                    # more than the URDF (e.g. body needs max_effort=5e3
                    # vs URDF's 50).  Never LOWER the cap — URDF's
                    # tighter value wins if it was authored.
                    #
                    # GRIPPER SPECIAL CASE: do NOT raise the gripper's
                    # URDF effort to ``art_default.max_effort`` — the
                    # gripper master is hand-tuned to its hardware
                    # spec (URDF effort=50) and raising it 100× to
                    # 5000 lets the master apply absurd torque under
                    # any non-zero command error.  Use the YAML's
                    # ``drive_gripper.master_max_force`` when set
                    # non-zero; otherwise preserve URDF.
                    if kind == JK_GRIPPER:
                        master_override = float(self._physics_params.drive_gripper.master_max_force)
                        if master_override > 0.0:
                            effort[i] = master_override
                        # else: leave URDF effort intact
                    else:
                        effort[i] = float(eff_c)
                n_authored_per_class += 1
            n_ke_fallback = 0
            n_kd_fallback = 0
        else:
            # Unscaled effort×10 heuristic (no params provided — e.g.
            # test_newton_solver.py without --physics-params).
            n_authored_per_class = 0
            kp_fallback_mask = ke < _VESTIGIAL_KE_THRESHOLD
            if effort is not None:
                scaled_eff = np.minimum(effort, self._effort_limit)
                scaled = scaled_eff * _KE_TO_EFFORT_FACTOR
                ke[kp_fallback_mask] = scaled[kp_fallback_mask]
            else:
                ke[kp_fallback_mask] = self._pd_ke
            kd_fallback_mask = kd <= 0.0
            kd[kd_fallback_mask] = (_KD_TO_KE_FRACTION * np.sqrt(np.maximum(ke[kd_fallback_mask], 1.0))).astype(
                kd.dtype
            )
            n_ke_fallback = int(kp_fallback_mask.sum())
            n_kd_fallback = int(kd_fallback_mask.sum())

        # 3. POSITION mode for any DOF with non-zero ke.  Runs AFTER
        # fallback writes so chassis joints (ke=0 → fallback wrote
        # effort×10) also get POSITION mode and an actuator emitted.
        # Followers with ke=0 stay at EFFORT (no POSITION actuator),
        # which is what we want for constraint-driven joints.
        prev_pos = int((mode == 1).sum())
        forced_pos = int(((mode != 1) & (ke > 0)).sum())
        if forced_pos > 0:
            mode = np.where((mode != 1) & (ke > 0), np.int32(1), mode)

        # 3b. VELOCITY mode for chassis drive DOFs (the free-spin road
        # wheels).  Newton emits a velocity actuator with
        # ``gainprm=[kd, ...] biasprm=[0, 0, -kd, ...]`` which is the
        # exact form the reference G2 MJCF uses for navigation:
        # ``force = kd * ctrl - kd * qvel`` → ctrl is the wheel target
        # velocity.  Without this the chassis_*_joint2 DOFs land at
        # ``mode = EFFORT`` (Newton's default for ke=0) which the
        # converter skips → no actuator emitted → robot can't drive.
        # We use the per-class ``kd`` from
        # ``articulation_view_runtime.chassis_drive`` as the velocity
        # gain (matches reference's gainprm=10 when YAML says kd=10).
        # POSITION-mode forcing above already skipped these (ke=0),
        # so we can stamp them with VELOCITY here without conflict.
        from genie_sim_engine_newton.common.joint_classification import JK_CHASSIS_DRIVE  # noqa: PLC0415

        _MODE_VELOCITY = 2  # newton.JointTargetMode.VELOCITY
        n_velocity_forced = 0
        if per_class is not None:
            for i, kind in enumerate(dof_kind):
                if kind == JK_CHASSIS_DRIVE:
                    mode[i] = _MODE_VELOCITY
                    n_velocity_forced += 1

        # 4. Mimic-follower mute fallback.  When the overlay ran,
        # followers already have ke=0 (which JointTargetMode resolves
        # to EFFORT-mode actuator with idle ctrl) and this branch is
        # a no-op.  When loading an older un-overlaid USD, we still
        # need to mute followers so they don't fight the equality
        # constraint.
        n_followers_muted = 0
        n_masters_softened = 0
        n_mimic_armature_set = 0
        mimic_joint_names: list = []
        if mimic_followers:
            try:
                follower_names = {name for _master, lst in mimic_followers.items() for (name, _c, _o) in lst}
                master_names = set(mimic_followers.keys())
                # Walk JointIndex.slices() instead of a parallel
                # joint_label / qd_start / dof_dim triple — same
                # JointIndex used at the top of this function, so the
                # qd alignment is consistent.
                armature_arr = getattr(model, "joint_armature", None)
                friction_arr = getattr(model, "joint_friction", None)
                armature = armature_arr.numpy().copy() if armature_arr is not None else None
                friction = friction_arr.numpy().copy() if friction_arr is not None else None

                for s in jindex.slices():
                    if not s.is_actuated:
                        continue
                    is_follower = s.name in follower_names
                    is_master = s.name in master_names
                    if not (is_follower or is_master):
                        continue
                    start = s.qd_start
                    count = s.dof_count
                    if is_follower:
                        # Mute fallback — only fires if the snapshot
                        # had a non-zero ke (overlay didn't run).
                        # Always-fires under per-class path: per-class
                        # writes follower ke=0 already (see the gripper
                        # branch in the apply loop), and this just
                        # belt-and-suspenders the mode=NONE so the
                        # actuator is dropped entirely.
                        if float(ke_authored_snapshot[start]) > 0.0 or per_class is not None:
                            mode[start : start + count] = 0
                            n_followers_muted += count
                    else:  # master
                        if per_class is not None:
                            # Per-class path: trust the kp/kd we just
                            # wrote (``drv_grip.master_stiffness`` /
                            # ``master_damping``) from
                            # ``physics_params.yaml``.  Don't read the
                            # USD-authored ke (``_HAND_MASTER_FINAL_KP``
                            # = 5) back over the top — the YAML carries
                            # the tuned values
                            # ``master_stiffness=2.0e3`` /
                            # ``master_damping=200.0``.
                            pass
                        else:
                            # No-per-class path — fall back
                            # to the YAML's master_stiffness /
                            # master_damping unconditionally.
                            # The YAML is the single source of
                            # truth; no "if authored ≤ FLOOR keep
                            # else soft-tune" heuristic applies.
                            ke[start : start + count] = (
                                float(self._physics_params.drive_gripper.master_stiffness)
                                if self._physics_params is not None
                                else float(ke_authored_snapshot[start])
                            )
                            kd[start : start + count] = (
                                float(self._physics_params.drive_gripper.master_damping)
                                if self._physics_params is not None
                                else float(model.joint_target_kd.numpy()[start])
                            )
                            n_masters_softened += count
                    # Armature / friction injection — always pull from
                    # ``drv_grip`` so the YAML stays the single source.
                    # ``drv_grip`` exists when ``physics_params`` was
                    # provided; otherwise the URDF / overlay-authored
                    # values stay (we only fill in zeros to avoid
                    # zero-armature / zero-friction).
                    _drv_grip = self._physics_params.drive_gripper if self._physics_params is not None else None
                    _grip_armature = _drv_grip.armature if _drv_grip is not None else 0.001
                    _grip_friction = _drv_grip.dof_frictionloss if _drv_grip is not None else 0.01
                    if armature is not None and start + count <= len(armature):
                        slot = armature[start : start + count]
                        if (slot <= 0.0).any():
                            slot[slot <= 0.0] = _grip_armature
                            armature[start : start + count] = slot
                    if friction is not None and start + count <= len(friction):
                        slot = friction[start : start + count]
                        if (slot <= 0.0).any():
                            slot[slot <= 0.0] = _grip_friction
                            friction[start : start + count] = slot
                    n_mimic_armature_set += count
                    mimic_joint_names.append(s.name)

                if armature is not None and armature_arr is not None:
                    armature_arr.assign(armature)
                if friction is not None and friction_arr is not None:
                    friction_arr.assign(friction)
                self._mimic_joint_names = mimic_joint_names
            except Exception as exc:
                logger.warn(f"[mjwarp-adapter] mimic suppression fallback failed (continuing): {exc}")

        mode_arr.assign(mode)
        ke_arr.assign(ke)
        kd_arr.assign(kd)
        if effort is not None:
            effort_arr.assign(effort)

        # ------------------------------------------------------------
        # Per-class PASSIVE joint authoring — independent of the
        # actuator gains above.  Writes ``dof_damping`` and
        # ``dof_frictionloss`` (in MuJoCo terms) so the reference
        # G2 MJCF's per-joint passive physics values land on every
        # body / arm / head DOF, instead of the URDF importer's 0 /
        # 0 defaults.  Without these the simulation feels less
        # settled than the reference (verified empirically via the
        # static / dynamic XML diff vs the reference G2 MJCF).
        #
        # Two arrays:
        #   * ``model.mujoco.dof_passive_damping``  (with the
        #     Newton-1.15 angular-DOF ``× π/180`` quirk — we
        #     pre-divide so the post-quirk MJCF value matches the
        #     authored target; see ``docs/newton_quirks.md``).
        #   * ``model.joint_friction``              (no quirk).
        #
        # JK_GRIPPER is intentionally skipped — the assemble overlay
        # in ``_robot_impl._apply_mimic_joint_overlay`` already
        # authors ``mjc:damping`` per gripper joint (0.05 N·m·s/rad)
        # with the same pre-divide, and we must not stomp it.
        # ------------------------------------------------------------
        n_passive_damping_set = 0
        n_passive_friction_set = 0
        n_passive_armature_set = 0
        passive_class = self._per_class_passive_joint()
        if passive_class is not None:
            # Both arrays are optional on Newton model — ``dof_passive_damping``
            # lives under ``model.mujoco`` (custom-attribute namespace
            # added by Newton's mjc resolver) and is only present when
            # the USD has at least one ``mjc:damping`` author or the
            # custom attr has been registered.  ``joint_friction`` is
            # a first-class model field — present on every model.
            mj_attrs = getattr(model, "mujoco", None)
            damp_arr_mjc = getattr(mj_attrs, "dof_passive_damping", None) if mj_attrs is not None else None
            friction_arr_top = getattr(model, "joint_friction", None)
            # ``joint_armature`` is a first-class Newton model field
            # (gripper-mimic block above already mutates it in place).
            # Pull it once, mutate per-DOF, assign back at the end.
            armature_arr_top = getattr(model, "joint_armature", None)

            from genie_sim_engine_newton.common.joint_classification import JK_GRIPPER  # noqa: PLC0415

            damp_np = damp_arr_mjc.numpy().copy() if damp_arr_mjc is not None else None
            friction_np = friction_arr_top.numpy().copy() if friction_arr_top is not None else None
            armature_np = armature_arr_top.numpy().copy() if armature_arr_top is not None else None

            # ``deg_to_rad`` compensation for Newton's ``× π/180`` quirk
            # on angular DOFs.  See ``docs/newton_quirks.md``.  All
            # joints we currently classify are revolute (body / arm /
            # head / chassis_wheel) so we apply uniformly; revisit if
            # we ever classify a prismatic joint into a class with
            # non-zero ``dof_damping``.
            #
            # NOTE — armature has NO π/180 quirk: it's a rotor inertia
            # (kg·m²), not an angular displacement / rate, so the
            # converter writes it verbatim.  Author the physical value.
            import math  # noqa: PLC0415

            deg_to_rad = math.pi / 180.0

            for i in range(len(ke)):
                kind = dof_kind[i]
                if kind == JK_GRIPPER:
                    # Overlay-authored — keep what's in the array.
                    continue
                cls_passive = passive_class.get(kind)
                if cls_passive is None:
                    continue
                target_damp, target_friction, target_armature = cls_passive
                if target_damp is not None and damp_np is not None and i < len(damp_np):
                    # Pre-divide to compensate Newton's angular-DOF
                    # quirk — the converter multiplies by π/180 at MJCF
                    # write time, so authoring ``target / (π/180)``
                    # lands the right value in the dumped MJCF.
                    damp_np[i] = float(target_damp) / deg_to_rad
                    n_passive_damping_set += 1
                if target_friction is not None and friction_np is not None and i < len(friction_np):
                    # No quirk on frictionloss — author verbatim.
                    friction_np[i] = float(target_friction)
                    n_passive_friction_set += 1
                if target_armature is not None and armature_np is not None and i < len(armature_np):
                    # No quirk on armature — author verbatim.  Skip
                    # ``target_armature == 0`` so we don't clobber
                    # an existing non-zero importer-supplied value
                    # with a zero default; YAML 0 means "leave alone".
                    if float(target_armature) > 0.0:
                        armature_np[i] = float(target_armature)
                        n_passive_armature_set += 1

            if damp_np is not None and damp_arr_mjc is not None:
                damp_arr_mjc.assign(damp_np)
            if friction_np is not None and friction_arr_top is not None:
                friction_arr_top.assign(friction_np)
            if armature_np is not None and armature_arr_top is not None:
                armature_arr_top.assign(armature_np)

        # Name-keyed sysid values are authoritative, including explicit zeros.
        # This runs after every class/mimic fallback and before SolverMuJoCo
        # converts the Newton model into its compiled solver arrays.
        from genie_sim_engine_newton.common.joint_params import apply_newton_joint_params

        apply_newton_joint_params(
            model, self._joint_params, mimic_followers=gripper_follower_set, robot_prefix=robot_prefix, logger=logger
        )

        ke_min = float(ke.min()) if len(ke) else 0.0
        ke_max = float(ke.max()) if len(ke) else 0.0
        kd_min = float(kd.min()) if len(kd) else 0.0
        kd_max = float(kd.max()) if len(kd) else 0.0
        path = (
            "per-class (physics_params.yaml)"
            if per_class is not None and not (self._pd_ke_explicit or self._pd_kd_explicit)
            else (
                "launcher pd_ke/pd_kd blanket override"
                if (self._pd_ke_explicit or self._pd_kd_explicit)
                else "unscaled effort×10 fallback"
            )
        )
        logger.info(
            f"[mjwarp-adapter] PD drives ({path}): "
            f"{len(mode)} DOF(s); mode→POS forced on {forced_pos} (was POS: {prev_pos}); "
            f"mode→VEL forced on {n_velocity_forced} (chassis_drive); "
            f"per-class authored on {n_authored_per_class}/{len(ke)} DOF(s); "
            f"ke fallback on {n_ke_fallback}/{len(ke)} DOF(s) → [{ke_min:g}, {ke_max:g}], "
            f"kd fallback on {n_kd_fallback}/{len(kd)} DOF(s) → [{kd_min:g}, {kd_max:g}], "
            f"effort floor on {n_effort_overridden}/{len(effort) if effort is not None else 0}; "
            f"mimic fallback: muted={n_followers_muted}, softened_master={n_masters_softened}, "
            f"armature/friction injected on {n_mimic_armature_set}; "
            f"passive joint per-class: dof_damping set on {n_passive_damping_set}/{len(ke)}, "
            f"dof_frictionloss set on {n_passive_friction_set}/{len(ke)}, "
            f"armature set on {n_passive_armature_set}/{len(ke)}."
        )

    # ------------------------------------------------------------------
    # Solver construction
    # ------------------------------------------------------------------

    def _report_collision_filter_budget(self, logger: Any) -> None:
        """Log how close this model is to MuJoCo's 32-colour filter ceiling.

        ``SolverMuJoCo`` encodes Newton's collision filtering as a graph
        colouring and then writes ``contype = 1 << colour`` per geom.  Colours
        are assigned so shapes that SHOULD collide get DIFFERENT colours, so the
        colour count grows with the number of mutually-colliding shapes.
        ``solver_mujoco.py`` only writes ``contype`` / ``conaffinity`` while
        ``colour < 32``; past that it writes NEITHER, and MuJoCo's defaults then
        make **everything collide with everything, robot self-collision
        included.**

        That failure is completely silent — no exception, no warning from Newton
        — and presents as an inexplicably unstable robot.  Dexterous hands make
        it reachable: the 20-DOF wuji hand pair alone colours to 31 of 32
        (measured on genie_g2_crsB_wuji.urdf), so adding scene props can tip it
        over.  Report the headroom every run, and shout when it is gone.
        """
        try:
            mj_model = getattr(self._solver, "mj_model", None)
            if mj_model is None:
                return
            import numpy as np  # local; module-top doesn't import np

            contype = np.asarray(mj_model.geom_contype)
            conaff = np.asarray(mj_model.geom_conaffinity)
            if contype.size == 0:
                return
            if bool(np.all(contype == 1) and np.all(conaff == 1)):
                logger.warn(
                    "[mjwarp-adapter] COLLISION FILTERING IS OFF: every geom has "
                    "contype=conaffinity=1, meaning the filter graph needed >= 32 colours "
                    "and SolverMuJoCo silently skipped writing the masks. Robot links will "
                    "self-collide and the solve will be unstable. Reduce the number of "
                    "mutually-colliding shapes — share a collision_group, or turn off "
                    "collision on decorative shapes."
                )
                return
            used = max(int(v).bit_length() for v in set(contype.tolist()) if v > 0)
            msg = (
                f"[mjwarp-adapter] collision filter: ~{used}/32 colours used "
                f"({32 - used} headroom) across {contype.size} geoms"
            )
            if used >= 30:
                logger.warn(
                    msg + " — WARNING: at 32 the masks are silently dropped and everything "
                    "self-collides. Adding colliding shapes to this scene is risky."
                )
            else:
                logger.info(msg)
        except Exception as exc:  # a diagnostic must never break startup
            logger.info(f"[mjwarp-adapter] collision filter budget check skipped: {exc!r}")

    def build_solver(
        self,
        model: Any,
        sim_substeps: int,
        sim_iterations: int,  # noqa: ARG002 — PARTICLE-solver count; the rigid
        # solve reads solver.mjc_iterations instead (see resolve_mjc_iterations).
        logger: Any,
        mass_matrix_interval: int = 0,  # noqa: ARG002 — mjwarp has no mass-matrix concept; accepted for ABC compatibility
    ) -> Any:
        # NOT sim_iterations: that is the particle-solver count, and running
        # MuJoCo's rigid constraint solve at a FEM scene's 4 makes frictional
        # grasps creep. 0/unset => SolverMuJoCo's own default (MuJoCo's 100).
        mj_iter = resolve_mjc_iterations(self._physics_params)
        # Anti-creep pair, from ``solver.impratio`` and nowhere else.  Both
        # ``None`` when the key is 0/unset, which is SolverMuJoCo's "leave
        # MuJoCo's defaults alone".  The multi-env adapter applies the same
        # key by poking ``mjm.opt`` post-construction because it also has to
        # fill mjwarp's ``impratio_invsqrt``; here the constructor does it.
        mj_impratio, mj_cone = resolve_mjc_impratio_cone(self._physics_params)
        from genie_sim_engine_newton.common.hand_collision import hand_self_collision_at_put_model

        with hand_self_collision_at_put_model(logger, mjcf_path=self._save_to_mjcf):
            self._solver = newton.solvers.SolverMuJoCo(
                model,
                iterations=mj_iter,
                impratio=mj_impratio,
                cone=mj_cone,
                # ``None`` => don't pass an opinion; SolverMuJoCo's default is False.
                enable_multiccd=bool(self._enable_multiccd),
                update_data_interval=1,
                njmax=self._njmax,
                nconmax=self._nconmax,
                # Native MJCF dump from mjwarp's MjSpec — when set, the
                # solver writes the fully-compiled mujoco model (joints,
                # actuators with gainprm/biasprm, geom inertia, contact
                # params, equality constraints, joint limits) so it can
                # be diffed against authoritative reference XML, loaded
                # in ``mujoco.viewer.launch`` for independent visual
                # inspection, or run through ``mujoco_validate`` /
                # ``mjpython`` checks without spinning up the Newton
                # engine.  Same path the production engine writes its
                # ``robot_runtime.usda`` next to — operator gets matched
                # USD + MJCF snapshots side by side.
                save_to_mjcf=self._save_to_mjcf or None,
            )
        logger.info(
            f"[mjwarp-adapter] robot solver: SolverMuJoCo "
            f"(iterations={mj_iter or 'default(100)'}, substeps={sim_substeps}, "
            f"njmax={self._njmax}, nconmax={self._nconmax}, "
            f"cone={mj_cone or 'default(pyramidal)'}, "
            f"impratio={mj_impratio if mj_impratio else 'default(1)'}, "
            f"multiccd={bool(self._enable_multiccd)})"
        )
        self._report_collision_filter_budget(logger)
        if self._save_to_mjcf:
            logger.info(
                f"[mjwarp-adapter] MJCF dump: wrote mjwarp's compiled mujoco "
                f"model to {self._save_to_mjcf}.  Inspect with "
                f"`mujoco.viewer.launch_passive(mujoco.MjModel.from_xml_path("
                f"...))` or diff vs. a reference MJCF."
            )
            # Post-process the dumped MJCF so its ``<option timestep>``
            # reflects the LIVE substep dt (Newton's ``_convert_to_mjc``
            # never sets ``spec.option.timestep`` — the file otherwise
            # inherits MuJoCo's 0.002 s default regardless of what the
            # engine actually runs at), and so a ``<custom>`` block
            # carries ``physics_hz`` + ``sim_substeps`` for pure-MuJoCo
            # tools that want to replay the runtime cadence.  Skipped
            # silently when ``physics_hz`` wasn't passed to the adapter
            # (e.g. ``test_newton_solver.py`` invocations).
            # Single unified MJCF post-process pipeline.  Each step
            # runs in its own ``try / except`` inside the helper so
            # one step's failure no longer skips the rest.  The same
            # helper drives the isaac_newton wrapper's
            # ``_force_solver_init`` post-dump path — both engines
            # converge on this single source of truth for "what does
            # a well-formed dump look like?".
            from genie_sim_engine_newton.common.mjcf_postprocess import apply_mjcf_postprocess_pipeline  # noqa: PLC0415

            apply_mjcf_postprocess_pipeline(
                mjcf_path=self._save_to_mjcf,
                sim_substeps=int(sim_substeps),
                physics_hz=float(self._physics_hz),
                physics_params=self._physics_params,
                qpos=None,  # init-pose keyframe is wired separately
                # in test_newton_solver.py and lifecycle.py
                # because it must run AFTER _apply_init_joint_pos
                imu_cfg=self._imu_cfg,
                logger=logger,
            )

        # Apply joint-level passive damping on mimic-touched DOFs.
        # SolverMuJoCo already compiled the model at this point, so
        # ``model.joint_target_kd`` etc. are frozen into mjwarp's
        # constants and ``dof_passive_damping`` (the route Newton uses
        # for joint-level damping) was 0 at compile time.  We write
        # straight into ``mjw_model.dof_damping`` — it's a
        # ``(nworld, n_dof)`` ``wp.array``; the captured graph reads
        # this array each substep, so post-build mutation persists.
        # mjwarp's dof index map differs from Newton's; we look each
        # mimic joint up by short name via
        # ``mujoco.mj_name2id(mj_model, mjOBJ_JOINT, name)`` and use
        # ``mj_model.jnt_dofadr`` to get the DOF index.
        self._apply_mimic_dof_damping(logger)
        # Push the YAML-configured mimic equality solref/solimp into the
        # live ``mjw_model.eq_solref`` / ``eq_solimp`` arrays so the
        # constraint stiffness matches whatever the YAML asks for.
        # When YAML uses MuJoCo's default ``[0.02, 1.0]`` this is a
        # no-op (the value already matches).  When the user sets a
        # stiff direct form (``[-10000, -100]``) this enforces it on
        # the live solver.  Mirrored in the dumped MJCF by the
        # ``augment_mjcf_equality_solref`` post-process step.
        self._apply_mimic_eq_solref(logger)
        # Force ceilings, both read from the physics profile and both
        # unconditional -- see each method's docstring for why they are not
        # folded into the loop-closure fix-up.
        self._cap_gripper_master_force(logger)
        self._cap_arm_actuator_force(logger)
        self._snapshot_base_mocap(logger)

        return self._solver

    def _apply_mimic_dof_damping(self, logger: Any) -> None:
        """Write the YAML's ``usd_drive_api.gripper.dof_damping`` into
        ``mjw_model.dof_damping`` for every mimic-touched joint AFTER
        SolverMuJoCo has compiled the mujoco model.

        Reads ``self._physics_params.drive_gripper.dof_damping`` —
        falls back to the prior 0.05 default when params is unset
        (no-params ``test_newton_solver.py`` invocations).  No-op when no
        mimic joints exist or the lookup fails.
        """
        names = getattr(self, "_mimic_joint_names", None) or []
        if not names:
            return
        # Pull the damping value from the YAML.  Same field the
        # ``prepare_model`` mimic-suppression block reads — single
        # source of truth.
        damping_value = (
            float(self._physics_params.drive_gripper.dof_damping) if self._physics_params is not None else 0.05
        )
        solver = getattr(self, "_solver", None)
        mj_model = getattr(solver, "mj_model", None) if solver is not None else None
        mjw_model = getattr(solver, "mjw_model", None) if solver is not None else None
        if mj_model is None or mjw_model is None:
            logger.warn(
                "[mjwarp-adapter] dof_damping: solver.mj_model / mjw_model " "unavailable; passive damping skipped"
            )
            return
        try:
            import mujoco
            import numpy as np

            dof_damping = mjw_model.dof_damping.numpy().copy()
            n_dof = dof_damping.shape[-1]
            applied = []
            missed = []
            mjc_joint_names: list = []
            for j in range(mj_model.njnt):
                full = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
                mjc_joint_names.append(full or "")
            for short in names:
                matches = [j for j, full in enumerate(mjc_joint_names) if full.endswith(short)]
                if not matches:
                    missed.append(short)
                    continue
                j = matches[0]
                dof0 = int(mj_model.jnt_dofadr[j])
                dof_count = 1
                if 0 <= dof0 < n_dof:
                    if dof_damping.ndim > 1:
                        dof_damping[:, dof0 : dof0 + dof_count] = np.maximum(
                            dof_damping[:, dof0 : dof0 + dof_count], damping_value
                        )
                    else:
                        dof_damping[dof0 : dof0 + dof_count] = np.maximum(
                            dof_damping[dof0 : dof0 + dof_count], damping_value
                        )
                    applied.append(short)
            mjw_model.dof_damping.assign(dof_damping)
            logger.info(
                f"[mjwarp-adapter] dof_damping: wrote {damping_value} "
                f"to {len(applied)}/{len(names)} mimic DOF(s) "
                f"(missed: {missed[:4]}{'…' if len(missed) > 4 else ''}).  "
                f"Sourced from physics_params.yaml::usd_drive_api.gripper.dof_damping."
            )
        except Exception as exc:
            logger.warn(
                f"[mjwarp-adapter] dof_damping write failed (continuing without "
                f"passive damping; armature + frictionloss are still in effect): {exc}"
            )

    def _apply_mimic_eq_solref(self, logger: Any) -> None:
        """Push YAML-configured equality solver tuning into
        ``mjw_model.eq_solref`` / ``eq_solimp`` for every gripper
        mimic equality.

        Why this lives at the adapter (post-build) layer
        ------------------------------------------------
        Newton's MJCF converter emits mimic equalities at MuJoCo's
        default ``solref=[0.02, 1.0]`` / ``solimp=[0.9, 0.95, 0.001,
        0.5, 2.0]`` regardless of any USD ``mjc:solref`` /
        ``mjc:solimp`` we might author — the mimic emit path in
        ``solver_mujoco.py`` doesn't consume those custom attributes
        (see ``docs/newton_quirks.md``).  So the runtime gets the
        default soft form unless we mutate the live mjw_model
        explicitly.

        This method writes ``self._physics_params.drive_gripper
        .mimic_eq_solref`` (and ``.mimic_eq_solimp``) into the live
        arrays.  When the YAML values equal MuJoCo's defaults this is
        a no-op write.  When the user picks a stiff form (negative
        solref like ``[-10000, -100]``) it stiffens every gripper
        equality at runtime.

        The dumped MJCF text is patched separately by
        ``common/mjcf_postprocess.py:augment_mjcf_equality_solref`` so
        live and dump always carry identical equality stiffness.
        """
        if self._physics_params is None:
            return
        names = getattr(self, "_mimic_joint_names", None) or []
        if not names:
            return
        drv_grip = self._physics_params.drive_gripper
        target_solref = tuple(float(v) for v in drv_grip.mimic_eq_solref)
        target_solimp = tuple(float(v) for v in drv_grip.mimic_eq_solimp)

        solver = getattr(self, "_solver", None)
        mj_model = getattr(solver, "mj_model", None) if solver is not None else None
        mjw_model = getattr(solver, "mjw_model", None) if solver is not None else None
        if mj_model is None or mjw_model is None:
            logger.warn("[mjwarp-adapter] eq_solref: solver.mj_model / mjw_model unavailable; skipping.")
            return

        try:
            import numpy as np

            # ``mjw_model.eq_*`` arrays are per-equality; one entry per
            # mjcf ``<equality>`` element.  Shape may be (n_eq,) or
            # (nworld, n_eq) depending on the multi-world configuration.
            eq_solref = mjw_model.eq_solref.numpy().copy()
            eq_solimp = mjw_model.eq_solimp.numpy().copy()
            eq_obj1id = mj_model.eq_obj1id  # joint id for joint1 of each equality

            # Build the set of joint indices that correspond to gripper
            # mimic joints (master + followers).  ``names`` already
            # carries the short names from ``prepare_model``.
            import mujoco

            mimic_jids: set = set()
            for short in names:
                for j in range(mj_model.njnt):
                    full = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                    if full.endswith(short):
                        mimic_jids.add(j)
                        break

            n_patched = 0
            for e in range(mj_model.neq):
                if int(eq_obj1id[e]) not in mimic_jids:
                    continue
                if eq_solref.ndim == 1:
                    # Per-equality single value — shouldn't happen for
                    # solref (vec2) but guard anyway.
                    continue
                # Standard case: eq_solref shape (..., n_eq, 2);
                # eq_solimp shape (..., n_eq, 5).
                if eq_solref.ndim == 3:
                    eq_solref[:, e, :] = target_solref
                    eq_solimp[:, e, :] = target_solimp
                else:
                    eq_solref[e, :] = target_solref
                    eq_solimp[e, :] = target_solimp
                n_patched += 1

            mjw_model.eq_solref.assign(eq_solref)
            mjw_model.eq_solimp.assign(eq_solimp)
            logger.info(
                f"[mjwarp-adapter] eq_solref/solimp: wrote {target_solref!r} / "
                f"{target_solimp!r} to {n_patched} mimic equality constraint(s).  "
                f"Sourced from physics_params.yaml::usd_drive_api.gripper."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warn(f"[mjwarp-adapter] eq_solref/solimp write failed (continuing with " f"MuJoCo defaults): {exc}")

    # ------------------------------------------------------------------
    # Target buffer
    # ------------------------------------------------------------------

    def _cap_gripper_master_force(self, logger: Any) -> None:
        """Cap the force on the gripper's MASTER actuators.

        ``master_max_force: 0`` is meant to read as "keep the URDF effort
        limit", but in MuJoCo a forcerange of ``[0, 0]`` means UNLIMITED —
        so closing hammers back and forth at +-1700 N*m and the grip force
        is anything but steady.  With a cap, saturation becomes a stable
        constant-force grasp.

        Priority: YAML ``drive_gripper.master_max_force`` (>0) >
        ``gripper_contact.gripper_master_force_cap`` (shipped 0 = leave
        forcerange alone).  The yaml is the only source — there is no env
        back door here; see ``config/physics_params.yaml``.

        What gets capped is the gripper's driving MASTER, i.e. the joints
        the URDF ``<mimic>`` entries reference (the ``_mimic_masters`` set
        ``prepare_model`` stashed, the two ``inner_joint1`` on G2).
        Without a parsed mimic map ``joint_is_master`` falls back to name
        matching, so scenes with no mimic are capped too.

        **This used to be multi-env only**, and the asymmetry was an
        accident of how the multi-env adapter was split out, not a
        decision: nothing about a force ceiling needs more than one world.
        ``physics_knobs.md`` had to carry a *multi-env only* marker for it,
        which told stock-adapter operators their configured cap was inert.
        The mechanism is shared with the multi-env copy
        (``base.apply_actuator_force_cap``); what stays here is the policy —
        which knob, which joints.
        """
        from genie_sim_engine_newton.common.gripper_annotations import joint_is_master  # noqa: PLC0415

        _cp2 = getattr(self._physics_params, "gripper_contact", None)
        _drv_grip = getattr(self._physics_params, "drive_gripper", None) if self._physics_params is not None else None
        _yaml_cap = float(getattr(_drv_grip, "master_max_force", 0.0) or 0.0)
        _fallback = float(getattr(_cp2, "gripper_master_force_cap", 0.0)) if _cp2 else 0.0
        master_cap = _yaml_cap if _yaml_cap > 0.0 else _fallback
        _cap_src = "yaml" if _yaml_cap > 0.0 else "default"
        # ★ Own log line: this is the line the readback gate greps
        # (``host_run_teleop.sh`` §2d) — keep the format.
        logger.info(f"[mjc] gripper master_max_force={master_cap:g} N*m " f"(source={_cap_src})")
        # ``<= 0`` is the shipped value and means "leave forcerange alone",
        # keeping the effort limit the MJCF/URDF already carries.  It must
        # not reach the cap below: that would write 0 into both bounds,
        # clamping the gripper's drive force to nothing.
        if master_cap <= 0.0:
            return
        masters = getattr(self, "_mimic_masters", set()) or set()
        capped = apply_actuator_force_cap(
            self._solver,
            logger,
            cap=master_cap,
            select=lambda jn: joint_is_master(jn, masters),
            what="gripper",
        )
        if capped:
            logger.info(
                f"[mjwarp-adapter] gripper master forcerange capped to "
                f"+-{master_cap:g} N*m on {len(capped)} actuator(s) "
                f"({', '.join(capped)})"
            )

    def _cap_arm_actuator_force(self, logger: Any) -> None:
        """Cap the arm's position-servo actuators at ``arm_contact.
        arm_max_torque`` (N*m).

        NOT the same knob as ``art_arm_shoulder/mid/wrist.max_effort``:
        that per-class table can only RAISE a joint's effort limit above
        whatever the URDF authored (see the per-class gain lookup above in
        this file) — it exists to backfill a class that needs MORE torque
        than a stingy URDF gives it, not to cap one that has too much.
        This is the opposite: a hard forcerange ceiling, independent of
        whatever the URDF or the class table authored.  It is also the only
        way DOWN from the unauthored-effort floor that lookup installs, so
        it is load-bearing even where the URDF already authors a sane
        limit.

        Why it's needed: these position-servo kp are 32000-60000 N*m/rad,
        and the compiled model's forcerange is empty (``[0, 0]`` =
        unlimited in MuJoCo).  The right arm maps to ~408 N/mm at the end
        effector: 1mm of commanded overshoot into the table applies 408 N,
        2mm applies 816, uncapped.  Measured consequence: an 8.3mm-thick
        coaster gets pressed flat through the table under a 2mm overshoot,
        and a 298g bottle gets launched to 1.14 m/s in a single 0.833ms
        substep.

        ``100.0`` (the shipped RLinf value) matches the URDF's own
        ``<limit effort="100">`` for each arm joint — capping to it
        saturates end-effector pressure at 357 N (0.87mm penetration, then
        holds).

        Only caps actuators whose joint classifies as an arm joint
        (``JK_ARM`` / ``JK_ARM_SHOULDER`` / ``JK_ARM_MID`` /
        ``JK_ARM_WRIST``, via ``classify_joint_by_name`` — name-pattern
        matching only, the same classifier the per-class drive-gain table
        above already trusts).  Body/head/chassis keep their existing
        stiffness untouched: the hand is what slams into the table, body
        and head are not on that force chain, and capping them would only
        degrade pose tracking.  The gripper has its own cap
        (``_cap_gripper_master_force``).

        Shared with the multi-env adapter for the same reason as the
        gripper cap — see that docstring's last paragraph.
        """
        arm_cap = float(getattr(getattr(self._physics_params, "arm_contact", None), "arm_max_torque", 0.0) or 0.0)
        if arm_cap <= 0.0:
            return

        from genie_sim_engine_newton.common.joint_classification import (  # noqa: PLC0415
            JK_ARM,
            JK_ARM_MID,
            JK_ARM_SHOULDER,
            JK_ARM_WRIST,
            classify_joint_by_name,
        )

        _ARM_KINDS = (JK_ARM, JK_ARM_SHOULDER, JK_ARM_MID, JK_ARM_WRIST)
        capped = apply_actuator_force_cap(
            self._solver,
            logger,
            cap=arm_cap,
            select=lambda jn: classify_joint_by_name(jn) in _ARM_KINDS,
            what="arm",
        )
        if capped is not None:
            logger.info(
                f"[mjwarp-adapter] arm position actuators capped to " f"+-{arm_cap:g} N*m on {len(capped)} actuator(s)"
            )

    def init_target_buffer(self, model: Any, control: Any, logger: Any) -> None:
        """Seed ``control.joint_target_q`` with the init pose.

        SolverMuJoCo owns the underlying buffer; we just point our
        ``target_buffer()`` at it and seed it with the current
        ``model.joint_q`` (which holds the init pose at this point because
        ``_apply_init_joint_pos`` has already run).
        """
        if model.joint_q is None or control is None:
            logger.warn("[mjwarp-adapter] cannot init target buffer: " "model.joint_q or control is None")
            return
        tgt = getattr(control, "joint_target_q", None)
        if tgt is None:
            logger.warn(
                "[mjwarp-adapter] control.joint_target_q is None; " "position commands will be silently dropped"
            )
            return
        # ``control.joint_target_q`` is sized to ``model.joint_dof_count``
        # (the qd-space size — one slot per actuated DOF), while
        # ``model.joint_q`` is sized to the q-space which includes the
        # extra coordinates of any FREE joints (3+4 vs 6 per FREE joint).
        # When the model has no FREE joints these match and a direct
        # assign is fine.  Otherwise we'd hit a size mismatch:
        #     "source buffer (N + 4·#free) > destination (N)"
        # The fix is to copy only the q-coordinates that have a 1-to-1
        # qd counterpart — i.e. revolute/prismatic DOFs.  The simplest
        # correct path is to copy as many elements as the destination
        # holds; FREE-joint extra coords end up unused (and they're
        # JK_PASSIVE → zero PD anyway, so target values for them are
        # ignored by MuJoCo's actuator).
        src = model.joint_q
        # [diag-init] Snapshot of joint_q at the moment we seed the
        # target buffer.  If _apply_init_joint_pos wrote non-zero values
        # earlier and they're still here, the path through Newton is
        # clean.  If they're zeros here, the joint_q was clobbered
        # between _apply_init_joint_pos and this point (one of the
        # lifecycle steps in between resets state).  Index trace
        # mirrors the topology log so cross-referencing by DOF index
        # works.
        try:
            jq_now = src.numpy()
            nonzero = [(int(i), float(jq_now[i])) for i in range(len(jq_now)) if abs(float(jq_now[i])) > 1e-9]
            logger.info(
                f"[diag-init] init_target_buffer entry: model.joint_q has "
                f"{len(nonzero)} non-zero entry/entries; "
                f"sample (up to 12): {nonzero[:12]}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warn(f"[diag-init] init_target_buffer joint_q snapshot failed: {exc}")

        if int(tgt.size) == int(src.size):
            tgt.assign(src)
        else:
            # FREE / BALL / D6 joints in the model — ``joint_q`` is
            # sized by q-count (e.g. 7 entries per FREE joint:
            # x,y,z,qw,qx,qy,qz) while ``joint_target_q`` is sized
            # by dof-count (6 per FREE joint: 3 trans + 3 angular
            # velocities).  ``JointIndex.copy_q_to_qd`` maps each
            # joint's q-slice to its qd-slice individually; revolute /
            # prismatic / fixed joints copy 1-to-1, width-mismatch
            # joints (FREE / BALL / D6) leave the qd slot at 0 because
            # there's no element-wise mapping from a quaternion to an
            # angular-velocity vector — those slots are typically
            # classified ``JK_PASSIVE`` anyway and the actuator ignores
            # the target value.
            #
            # The naive equivalent (``tgt = src_q[: tgt.size]``) shifts
            # every joint downstream of a FREE joint by one slot,
            # which presents as a startup kick: at step 1 each
            # actuator sees ``error = prev_joint_q − this_joint_q``
            # and the high-kp PD slams the robot.
            from genie_sim_engine_newton.engine.newton.joint_index import JointIndex  # noqa: PLC0415
            import numpy as _np  # noqa: PLC0415

            src_np = src.numpy()
            jindex = JointIndex(model)
            tgt_np = _np.zeros(int(tgt.size), dtype=src_np.dtype)
            n_copied = jindex.copy_q_to_qd(src_np, tgt_np)
            tgt.assign(tgt_np)
            logger.info(
                f"[mjwarp-adapter] target_buffer size={int(tgt.size)} smaller "
                f"than joint_q size={int(src.size)} (FREE / BALL / D6 joints "
                f"carry width-mismatched q entries); per-joint mapped "
                f"{n_copied}/{int(tgt.size)} dof slots — width-mismatch joints "
                f"stay at 0 (typically JK_PASSIVE; actuator ignores target)"
            )
        # [diag-init] Verify the seed landed in control.joint_target_q.
        # This is the buffer SolverMuJoCo's JOINT_TARGET actuators read
        # to compute ``ke*(ctrl - qpos)`` each step.  Zeros here →
        # actuators see target=0 from the first step → PD pulls the
        # robot back toward all-zeros even though joint_q started at
        # the init pose, which presents as "robot snaps to zero
        # immediately after init".
        try:
            tgt_after = tgt.numpy()
            nonzero_tgt = [
                (int(i), float(tgt_after[i])) for i in range(len(tgt_after)) if abs(float(tgt_after[i])) > 1e-9
            ]
            logger.info(
                f"[diag-init] init_target_buffer exit: control.joint_target_q has "
                f"{len(nonzero_tgt)} non-zero entry/entries after seed; "
                f"sample (up to 12): {nonzero_tgt[:12]}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warn(f"[diag-init] init_target_buffer joint_target_q snapshot failed: {exc}")
        self._target_buffer = tgt

    def target_buffer(self) -> Optional[wp.array]:
        return self._target_buffer

    # ------------------------------------------------------------------
    # Post-joint-map: no-op (PD already configured in prepare_model)
    # ------------------------------------------------------------------

    def post_joint_map(
        self,
        model: Any,
        jindex: Any,
        control: Any,
        logger: Any,
    ) -> None:
        logger.info(
            "[mjwarp-adapter] post_joint_map: no-op " "(PD drives were set in prepare_model before SolverMuJoCo init)"
        )

    # ------------------------------------------------------------------
    # Per-substep robot step (inside captured CUDA graph)
    # ------------------------------------------------------------------

    def substep(
        self,
        model: Any,
        state_in: Any,
        state_out: Any,
        control: Any,
        sim_dt: float,
    ) -> None:
        # mjwarp keeps a SEPARATE gravity copy in ``mjw_model.opt.gravity``
        # (mujoco's option struct); ``solver.step`` reads that, NOT
        # ``model.gravity``.  When the engine's kinematic-control substep
        # writes ``model.gravity = 0`` and we don't propagate it, mjwarp
        # still steps under earth gravity — robot droops, contacts engage,
        # bouncing instability.
        #
        # ``_update_model_properties`` launches a Warp kernel that copies
        # ``model.gravity`` → ``mjw_model.opt.gravity`` (it has a CPU
        # branch that calls ``.numpy()`` which would force a sync; we're
        # always on CUDA here, so the GPU-kernel branch fires and is
        # capturable into the engine's CUDA graph).  Calling it once at
        # the top of every substep keeps mjwarp's gravity in sync with
        # whatever the engine's substep body set on Newton's side.
        self._solver._update_model_properties()
        # SolverMuJoCo's JOINT_TARGET actuators read control.joint_target_q
        # internally; nothing else to inject.
        self._solver.step(state_in, state_out, control, None, sim_dt)
