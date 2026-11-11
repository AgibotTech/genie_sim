# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""MuJoCo-Warp solver adapter.

Encapsulates the pre-construction PD-drive setup,
``SolverMuJoCo`` construction with internal-buffer sizes, and seeding of
``control.joint_target_q`` with the init pose.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import newton
import warp as wp

from genie_sim_engine_newton.engine.newton.adapters.base import (
    SolverAdapter,
    apply_actuator_force_cap,
    resolve_mjc_iterations,
)

from .base_mocap import BaseMocapResetMixin, select_base_mocap_index

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


def _compute_body_aabb_local(model: Any, solver: Any, logger: Any) -> Optional[Any]:
    """Build-time (NOT per-step) per-body LOCAL AABB table, indexed by the
    PER-ENV Newton body index (0..bodies_per_world-1) — the SAME index
    space as ``model.body_label`` / ``session.py``'s ``body_frames`` — so
    the caller can zip it directly against those names before publishing.

    Called exactly once, right after ``SolverMuJoCo`` construction. Reads
    mjwarp's (single-env-template) geometry — ``geom_aabb`` (center+size,
    body-LOCAL after accounting for ``geom_pos``/``geom_quat``, see below),
    ``geom_bodyid``, ``geom_pos``, ``geom_quat`` — unions every geom's
    local box (rotated+translated into the BODY frame) per mjc body, then
    remaps mjc body id -> per-env Newton body id via
    ``solver.mjc_body_to_newton`` (built by ``newton.solvers.SolverMuJoCo``
    — see its ``_convert_to_mjc``: ``mjc_body_to_newton[w, mjc_body]`` is
    the GLOBAL newton body index for world ``w``; row 0 is already the
    per-env-template index since world 0 contributes no offset).

    Bodies with no geometry (pure frames) get an all-zero box, not
    "missing" — this table is DENSE over ``0..bodies_per_world-1``.

    Every "*"-leading-dim mjwarp array here (``geom_aabb``/``geom_pos``/
    ``geom_quat``) is read at index 0 along that dim: it's either size 1
    (the common case — shared, un-randomized geometry) or ``nworld`` (if
    some other code path enabled per-world geometry DR), and index 0 is
    the per-env TEMPLATE value either way (mirrors mjwarp's own broadphase
    kernel, which indexes ``geom_aabb[worldid % ngeom_aabb]`` — i.e. falls
    back to slot 0 whenever ``ngeom_aabb == 1``). All N replicated worlds
    share the same geometry at build time, so slot 0 is correct for all
    of them; only ONE (single-env) table needs to be published.

    Returns ``None`` (caller must not publish) if the model doesn't expose
    the expected mjwarp arrays (e.g. a non-mjwarp solver got here somehow,
    or a mjwarp version without ``mjc_body_to_newton``).
    """
    import numpy as np  # noqa: PLC0415  (module top doesn't import np)

    try:
        mjw_model = getattr(solver, "mjw_model", None)
        mj_model = getattr(solver, "mj_model", None)
        mjc_body_to_newton = getattr(solver, "mjc_body_to_newton", None)
        if mjw_model is None or mj_model is None or mjc_body_to_newton is None:
            logger.warn(
                "[mjwarp-adapter] body_aabb_local: solver.mjw_model / "
                "mj_model / mjc_body_to_newton unavailable; skipping "
                "(get_body_aabbs will fall back to zero AABBs)"
            )
            return None

        world_count = int(getattr(model, "world_count", 1) or 1)
        bodies_per_world = int(model.body_count) // max(1, world_count)
        ngeom = int(mj_model.ngeom)
        if bodies_per_world <= 0:
            return None
        out = np.zeros((bodies_per_world, 6), dtype=np.float32)
        if ngeom == 0:
            return out  # no geoms at all -> every body is a zero box

        geom_bodyid = np.asarray(mjw_model.geom_bodyid.numpy()).reshape(-1)[:ngeom]
        geom_aabb = mjw_model.geom_aabb.numpy()[0]  # (ngeom, 2, 3) center,size
        geom_pos = mjw_model.geom_pos.numpy()[0]  # (ngeom, 3)    geom->body offset
        geom_quat = mjw_model.geom_quat.numpy()[0]  # (ngeom, 4)    WXYZ (matches
        # this file's OWN xquat
        # convention — see
        # ``dw.xquat.numpy()[world] # wxyz``
        # a few hundred lines below).
        newton_of_mjc = np.asarray(mjc_body_to_newton.numpy())[0]  # (nbody_mjc,)

        mins = np.full((bodies_per_world, 3), np.inf, dtype=np.float64)
        maxs = np.full((bodies_per_world, 3), -np.inf, dtype=np.float64)
        has_geom = np.zeros((bodies_per_world,), dtype=bool)

        # 8-corner sign combinations, reused per geom.
        signs = np.array(
            [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
            dtype=np.float64,
        )

        for g in range(ngeom):
            mjc_b = int(geom_bodyid[g])
            if mjc_b < 0 or mjc_b >= len(newton_of_mjc):
                continue
            nb = int(newton_of_mjc[mjc_b])
            if nb < 0 or nb >= bodies_per_world:
                continue  # world body / unmapped static geom — not one of ours

            c = geom_aabb[g, 0].astype(np.float64)
            h = geom_aabb[g, 1].astype(np.float64)
            corners_geom = c[None, :] + signs * h[None, :]  # (8, 3)

            w, x, y, z = (float(v) for v in geom_quat[g])
            n = w * w + x * x + y * y + z * z
            n = n if n > 1e-12 else 1.0
            s = 2.0 / n
            r = np.array(
                [
                    [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                    [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                    [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
                ]
            )
            corners_body = corners_geom @ r.T + geom_pos[g].astype(np.float64)  # (8,3)

            mn = corners_body.min(axis=0)
            mx = corners_body.max(axis=0)
            mins[nb] = np.minimum(mins[nb], mn)
            maxs[nb] = np.maximum(maxs[nb], mx)
            has_geom[nb] = True

        for b in range(bodies_per_world):
            if not has_geom[b]:
                continue  # frame-only body -> zero box (dense table, not "missing")
            out[b, 0:3] = (mins[b] + maxs[b]) * 0.5
            out[b, 3:6] = (maxs[b] - mins[b]) * 0.5

        logger.info(
            f"[mjwarp-adapter] body_aabb_local: computed local AABB for "
            f"{bodies_per_world} body/bodies from {ngeom} geom(s) "
            f"({int(has_geom.sum())} with geometry, "
            f"{bodies_per_world - int(has_geom.sum())} zero-box frame(s))"
        )
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warn(f"[mjwarp-adapter] body_aabb_local: computation failed: {exc!r}")
        return None


@wp.kernel
def _pin_kinematic_dofs_kernel(
    qadr: wp.array(dtype=wp.int32),
    dadr: wp.array(dtype=wp.int32),
    qref: wp.array2d(dtype=wp.float32),
    # outputs
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
):
    """Clamp kinematic-body free-joint state: qpos ← authored reference,
    qvel ← 0.  Launched dim=(nworld, len(qadr)); dadr is shorter (6 vs 7
    slots per body) so it's guarded."""
    world, i = wp.tid()
    qpos[world, qadr[i]] = qref[world, i]
    if i < dadr.shape[0]:
        qvel[world, dadr[i]] = 0.0


@wp.kernel
def _gather_kinematic_ref_kernel(
    qadr: wp.array(dtype=wp.int32),
    qpos: wp.array2d(dtype=wp.float32),
    # outputs
    qref: wp.array2d(dtype=wp.float32),
):
    """Snapshot the pinned bodies' current qpos into the reference buffer
    (called after every explicit state sync — init pose, reset, instance
    switch — so the pin tracks the latest authored placement per world)."""
    world, i = wp.tid()
    qref[world, i] = qpos[world, qadr[i]]


@wp.kernel
def _zero_world_rows_kernel(
    worlds: wp.array(dtype=wp.int32),
    # output
    buf: wp.array2d(dtype=wp.float32),
):
    """Zero ``buf``'s rows for the named worlds. Launched
    dim=(len(worlds), buf.shape[1]); other worlds are untouched, so a
    subset reset never disturbs an env that is still running."""
    i, j = wp.tid()
    buf[worlds[i], j] = 0.0


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

# Per-body linear scaling for the auto path, calibrated on two measured
# scenes: 61-body supermarket peaks at njmax~750 (2x headroom kept by the
# 1024 floor), and the 156-body 20-instance superset probe demanded
# njmax>=2096 / nconmax>=700 (auto gives 3120/1092, ~50% headroom).
_NJMAX_PER_BODY = 20
_NCONMAX_PER_BODY = 7


def resolve_constraint_capacity(
    nbody_per_env: int,
    njmax_explicit: int = 0,
    nconmax_explicit: int = 0,
) -> "tuple[int, int]":
    """Pick (njmax, nconmax) for SolverMuJoCo: explicit positive values win
    verbatim; otherwise scale linearly with the per-env body count, floored
    at the historical defaults so small scenes never shrink."""
    nb = max(0, int(nbody_per_env))
    njmax = int(njmax_explicit) if int(njmax_explicit) > 0 else max(_DEFAULT_NJMAX, _NJMAX_PER_BODY * nb)
    nconmax = int(nconmax_explicit) if int(nconmax_explicit) > 0 else max(_DEFAULT_NCONMAX, _NCONMAX_PER_BODY * nb)
    return njmax, nconmax


class MuJoCoWarpMultiEnvAdapter(BaseMocapResetMixin, SolverAdapter):
    #: This adapter's ``_build_base_pin`` unions the robot root FREE
    #: joint's qpos/qvel slots into the same per-substep clamp it uses for
    #: KINEMATIC scene bodies — the capability ``pin_base_mode: mocap``
    #: is built on.  It is the only adapter that has it.
    supports_base_pin = True

    def __init__(
        self,
        *,
        pd_ke: float = 0.0,
        pd_kd: float = 0.0,
        njmax: int = 0,
        nconmax: int = 0,
        effort_limit: float = 0.0,
        save_to_mjcf: str = "",
        physics_params: Any = None,
        joint_params: Any = None,
        physics_hz: float = 0.0,
        imu_cfg: Any = None,
        pin_base_joint: bool = False,
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
        # 0 = auto-scale sentinel; resolved against the per-env body count
        # in ``build_solver`` via ``resolve_constraint_capacity``. Positive
        # values here are explicit launcher overrides that win verbatim.
        self._njmax = max(0, int(njmax))
        self._nconmax = max(0, int(nconmax))
        self._save_to_mjcf = str(save_to_mjcf) if save_to_mjcf else ""
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
        # ``pin_base_mode=mocap`` (see schema.scene.PinBaseMode):
        # the robot root FREE joint goes into the pin arrays built by
        # ``_build_base_pin`` below, so it gets clamped to its authored
        # reference every substep instead of drifting under ordinary
        # floor/wheel contact.  ``False`` (default) leaves the arrays empty
        # and the per-substep clamp is skipped entirely.
        #
        # The base is the ONLY occupant of those arrays. Scene kinematic
        # bodies used to share them; they no longer reach here at all —
        # ``setup/stage.py::_demote_kinematic_bodies`` turns them into
        # world-static colliders before the model is built.
        self._pin_base_joint = bool(pin_base_joint)
        # ``newton.mujoco_enable_multiccd`` from the scene yaml, tri-state.
        # ``None`` (absent) keeps THIS adapter's long-standing default of ON —
        # see the MULTICCD block in the warmup path for the pinch measurement
        # that put it there.  An explicit ``false`` in a scene turns it off;
        # the stock adapter defaults the other way.  See
        # ``adapters/__init__.py::make_adapter``.
        self._enable_multiccd = True if enable_multiccd is None else bool(enable_multiccd)
        self._target_buffer: Optional[wp.array] = None
        self._solver: Any = None
        # Gripper drive MASTER joint set, populated by ``prepare_model``
        # from ``gripper_masters(mimic_followers)`` and consumed by the
        # H4 force-cap in ``_fix_gripper_loop_constraints``.  Empty until
        # then (and on no-mimic scenes), where H4 falls back to its legacy
        # ``outer_joint1`` regex via ``joint_is_master``.
        self._mimic_masters: set = set()
        # Gripper mimic FOLLOWER joint names (the union of every follower
        # in the mimic map), populated by ``prepare_model``.  Consumed by
        # the H3 loop-constraint fix to build the DRIVEN set
        # (``masters ∪ (followers − couplers)``) structurally, without a
        # ``joint1$`` regex.  Empty on no-mimic scenes.
        self._mimic_follower_names: set = set()
        # Gripper 4-bar loop COUPLER joint names (``newton:isLoopCoupler``
        # truth, legacy ``joint[034]`` regex fallback), threaded in from
        # the solver via ``set_coupler_joints`` (computed where the USD
        # stage is available — the adapter has none).  Consumed by the H3
        # coupler unlock + passive damping in ``_fix_gripper_loop_constraints``.
        self._coupler_joints: set = set()
        # Build-time-only (never per-step) per-body LOCAL AABB table, set
        # by ``build_solver`` -> ``_compute_body_aabb_local``. ``None``
        # until then, or if the computation failed / isn't applicable.
        self._body_aabb_local: Optional[Any] = None
        # Robot BASE mocap body (pin_base_to_world:false native path), set
        # by ``build_solver`` -> ``_snapshot_base_mocap``. ``_base_mocap_idx``
        # is the base's mocap id (``None`` = no mocap base → the reset-time
        # mocap randomization path is inactive; other builds hit the
        # free-joint / weld paths). The two init snapshots are the
        # build-time base pose the per-reset offset is applied RELATIVE to
        # (restore-then-offset — see ``apply_base_mocap_offsets``).
        self._base_mocap_idx: Optional[int] = None
        self._base_mocap_jnt: Optional[Any] = None  # per-world Newton jnt idx of the base fixed joint
        self._base_mocap_init_pos: Optional[Any] = None  # np copy of mjw_data.mocap_pos at build (nworld, nmocap, 3)
        self._base_mocap_init_xp: Optional[Any] = None  # np copy of model.joint_X_p at build (njnt, 7)
        #: Whether ``_build_base_pin`` actually registered a robot root
        #: free joint into the clamp.  ``supports_base_pin`` says this
        #: adapter CAN hold the base; this says it DID.  The engine
        #: asserts it when the resolved mode needs the clamp.
        self._base_pin_registered: bool = False

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mujoco-warp-multi-env"

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
            JK_SCENE_ARTICULATION,
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
            # Articulated scene joints (door hinge / drawer slide): also
            # zero PD — the joint is passive so it can be pushed freely
            # (no ghost drive holding or driving it).  Unlike JK_PASSIVE
            # it DOES get non-zero passive damping/frictionloss from the
            # ``scene_articulation`` block (see _per_class_passive_joint).
            JK_SCENE_ARTICULATION: (0.0, 0.0, 0.0),
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
            JK_SCENE_ARTICULATION,
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
            # JK_SCENE_ARTICULATION: door hinge / drawer slide — apply the
            # ``scene_articulation`` block's passive damping + frictionloss
            # so the joint settles instead of oscillating near-frictionlessly
            # (contrast JK_PASSIVE above, whose free bodies must fall
            # naturally).  ``armature`` from the same block (0 ⇒ leave as-is).
            JK_SCENE_ARTICULATION: _t(p.drive_scene_articulation),
        }

    # ------------------------------------------------------------------
    # Post-finalize, pre-solver
    # ------------------------------------------------------------------

    def set_coupler_joints(self, coupler_joints: Optional[Any]) -> None:
        """Thread in the gripper 4-bar loop COUPLER joint name set.

        Called by the solver (``setup/solver.py``) right after
        ``prepare_model``, from the site where the live USD stage is
        available — the adapter has no stage of its own (concern-4).  The
        set is ``read_loop_coupler_joint_names(stage, ...)``: the
        ``newton:isLoopCoupler`` annotation truth, or the legacy
        ``joint[034]`` regex fallback on un-annotated USD.  Consumed by
        the H3 fix in ``_fix_gripper_loop_constraints`` (which runs later,
        in ``build_solver``) via ``gripper_loop_constraint_labels``.
        Mirrors how ``mimic_followers`` is threaded to ``prepare_model``.
        """
        self._coupler_joints = set(coupler_joints or ())

    def prepare_model(
        self,
        model: Any,
        logger: Any,
        mimic_followers: Optional[Dict[str, list]] = None,
        robot_prefix: str = "",
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
        # Stash the gripper MASTER joint set before any early return, so that
        # ``_fix_gripper_loop_constraints`` — which runs later inside
        # ``build_solver`` and has no mimic map of its own — can identify the
        # driving joints from the annotation. ``gripper_masters`` is just the
        # keys of the mimic map, i.e. the joints the URDF ``<mimic>`` entries
        # reference; empty when no mimic map was parsed, in which case that
        # code falls back to name matching.
        from genie_sim_engine_newton.common.gripper_annotations import gripper_masters  # noqa: PLC0415

        self._mimic_masters = gripper_masters(mimic_followers)
        # H3 DRIVEN-set source: the union of every mimic follower name.
        # ``_fix_gripper_loop_constraints`` (which runs later, in
        # ``build_solver``, and has no ``mimic_followers`` of its own)
        # computes the driven joints as ``masters ∪ (followers − couplers)``
        # from this + ``self._mimic_masters`` + ``self._coupler_joints``.
        self._mimic_follower_names = {
            name for _master, lst in (mimic_followers or {}).items() for (name, _c, _o) in lst
        }

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
            JK_SCENE_ARTICULATION,
            classify_scene_joint,
        )
        from genie_sim_engine_newton.engine.newton.joint_index import JT_FREE  # noqa: PLC0415

        dof_kind: list = [JK_OTHER] * len(ke)
        unknown_joint_names: list = []
        passive_joint_names: list = []  # log-only — these are intentional, not errors
        scene_joint_names: list = []  # log-only — ownership-classified scene articulation joints
        gripper_follower_set = set()
        gripper_master_set = set()
        if mimic_followers:
            gripper_follower_set = {name for _master, lst in mimic_followers.items() for (name, _c, _o) in lst}
            gripper_master_set = set(mimic_followers.keys())

        # Joint parent/child body arrays used for the ownership test. A joint
        # whose parent AND child bodies both sit outside the robot prefix is a
        # scene joint (door hinges, drawer slides, free scene bodies) — this
        # keeps props with custom joint names from tripping the hard
        # ``unknown_joint_names`` failure. Passing None, or an empty prefix,
        # disables the test and preserves the old name-only behaviour.
        _jp_arr = getattr(model, "joint_parent", None)
        _jc_arr = getattr(model, "joint_child", None)
        _jp = _jp_arr.numpy() if _jp_arr is not None else None
        _jc = _jc_arr.numpy() if _jc_arr is not None else None
        _body_label = list(getattr(model, "body_label", []) or [])

        def _owned_body_label(arr, joint_idx):
            if arr is None or joint_idx >= len(arr):
                return None
            bidx = int(arr[joint_idx])
            return _body_label[bidx] if 0 <= bidx < len(_body_label) else None

        for s in jindex.slices():
            if not s.label or not s.is_actuated:
                continue
            if s.qd_start + s.dof_count > len(ke):
                continue
            # Identify scene joints by ownership (not by name). Runs before
            # name-based classification so arbitrarily named scene props are
            # still recognised; robot joints get None here and fall through to
            # the name path below.
            #
            # The guard here mirrors topology.py::_is_scene_owned: when the
            # model has no joint_parent/joint_child arrays at all, both labels
            # are None, and is_scene_joint(None, None, prefix) returns True —
            # which would route every controlled robot joint down the zero-gain
            # scene path (the whole robot goes limp). In that case the
            # ownership test is simply disabled and the name path decides.
            scene_kind = (
                classify_scene_joint(
                    is_free=(s.joint_type == JT_FREE),
                    parent_body_label=_owned_body_label(_jp, s.joint_index),
                    child_body_label=_owned_body_label(_jc, s.joint_index),
                    robot_prefix=robot_prefix,
                )
                if (robot_prefix and _jp is not None and _jc is not None)
                else None
            )
            if scene_kind is not None:
                # JK_SCENE_ARTICULATION (hinge/slide) → zero PD + scene
                # passive damping; JK_PASSIVE (free scene body) → zero PD,
                # natural free-fall.  Either way the DOFs are un-driven.
                if scene_kind == JK_SCENE_ARTICULATION:
                    scene_joint_names.append(s.name)
                else:
                    passive_joint_names.append(s.name)
                for k in range(s.dof_count):
                    dof_kind[s.qd_start + k] = scene_kind
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
        if scene_joint_names:
            logger.info(
                f"[mjwarp-adapter] classified {len(scene_joint_names)} joint(s) as "
                f"JK_SCENE_ARTICULATION by body-ownership (non-robot parent+child; "
                f"door hinge / drawer slide — PD gains zeroed, scene_articulation "
                f"passive damping applied): {scene_joint_names}"
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
            # head / chassis_wheel, and the door-style
            # ``scene_articulation`` hinges) so we apply uniformly.
            # KNOWN CAVEAT (registered backlog): a PRISMATIC scene joint
            # (drawer slide) classified ``scene_articulation`` would have
            # its dof_damping over-scaled ~57x by this uniform pre-divide
            # — the class needs a prismatic exception before drawer tasks.
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
        # Constraint-buffer capacity: explicit launcher values win; otherwise
        # auto-scale with the per-env body count (same world_count derivation
        # as _compute_body_aabb_local above).
        world_count = int(getattr(model, "world_count", 1) or 1)
        nbody_per_env = int(model.body_count) // max(1, world_count)
        njmax, nconmax = resolve_constraint_capacity(nbody_per_env, self._njmax, self._nconmax)
        basis = "explicit" if (self._njmax > 0 or self._nconmax > 0) else (f"auto(nbody_per_env={nbody_per_env})")
        # Keep the effective values separately; later capacity checks use them
        # as the denominator.
        self._njmax_eff, self._nconmax_eff = int(njmax), int(nconmax)
        from genie_sim_engine_newton.common.hand_collision import hand_self_collision_at_put_model

        with hand_self_collision_at_put_model(logger, mjcf_path=self._save_to_mjcf):
            self._solver = newton.solvers.SolverMuJoCo(
                model,
                iterations=mj_iter,
                # 0 = mjw_data owns the dynamic state; the Newton side is read-only.
                #
                # The historical interval=1 did one Newton<->mjw round trip per
                # substep, and that is the root cause of resting objects tumbling
                # microscopically (plain mujoco_warp holds a resting error <=0.2 mm
                # on the same model and initial values; Newton's own source also
                # marks that path as potentially introducing numerical instability).
                # What makes 0 viable is that the two jobs that round trip was
                # doing are covered elsewhere:
                #   * pushing engine-side state writes into the solver —
                #     ``sync_state_into_solver``, called once after every direct
                #     write (init pose, reset, randomization);
                #   * holding kinematic scene bodies in place — done at stage
                #     level now (``setup/stage.py::_demote_kinematic_bodies``
                #     makes them world-static colliders), so they are not
                #     simulated bodies and need no per-substep clamp.
                # ``_build_base_pin`` is NOT part of this: it clamps the robot
                # base only, and only under ``pin_base_mode: mocap``.
                update_data_interval=(
                    int(self._physics_params.solver.update_data_interval) if self._physics_params is not None else 1
                ),
                njmax=njmax,
                nconmax=nconmax,
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
            f"njmax={njmax}, nconmax={nconmax} [{basis}])"
        )
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
        # Close the gripper 4-bar loops for real: rebase the connect-
        # equality anchors at the assembled pose, stiffen them, and free
        # the in-loop coupler joints that PD would otherwise lock.
        # Global contact options, the gripper master force cap, and the arm
        # torque cap all go first and unconditionally: the loopfix below
        # returns early on grippers with no CONNECT equalities (G2_90d has
        # none — only JOINT mimic equalities), and none of these three are
        # loop-closure-dependent — see each method's own docstring.
        self._apply_global_contact_options(logger)
        self._cap_gripper_master_force(logger)
        self._cap_arm_actuator_force(logger)
        self._fix_gripper_loop_constraints(logger)
        # Robot base pin (``pin_base_mode: mocap`` only; no-op otherwise) —
        # must come AFTER the solver exists so mj_model/mjw_data are up.
        self._build_base_pin(logger)
        # Robot BASE mocap snapshot (pin_base_to_world:false native path) —
        # also needs mj_model/mjw_data/mjc_mocap_to_newton_jnt, so it runs
        # here. No-op when the model has no mocap base.
        self._snapshot_base_mocap(logger)

        # Per-body LOCAL AABB table — ONE-TIME (build-time) computation,
        # zero per-step cost. Consumed by session.py (publishes it, once,
        # latched, on /rlinf/body_aabb_local) for rlinf's
        # get_body_aabbs() eval-checker support. See ``local_body_aabb``.
        self._body_aabb_local = _compute_body_aabb_local(model, self._solver, logger)

        return self._solver

    def local_body_aabb(self) -> Optional[Any]:
        """Per-env-body-indexed LOCAL AABB table (``(bodies_per_world, 6)``
        ``[cx,cy,cz,hx,hy,hz]``), or ``None`` if unavailable. Duck-typed
        hook — only the mjwarp adapter implements this; callers (e.g.
        ``engine/newton/topology.py::_build_body_map``) probe via
        ``getattr(adapter, "local_body_aabb", None)``, same pattern as
        ``set_contacts`` elsewhere in this engine.
        """
        return self._body_aabb_local

    def _apply_global_contact_options(self, logger: Any) -> None:
        """Solver-wide contact settings: multi-point manifolds + friction cone.

        Deliberately NOT part of ``_fix_gripper_loop_constraints``.  That
        method returns early when the robot has no CONNECT equalities, and
        only closed-chain grippers (G2_omnipicker) have any -- G2_90d's
        gripper carries 10 JOINT mimic equalities and zero CONNECT, so the
        whole tail was skipped and it ran on the pyramidal cone with no real
        static friction, letting a grasped object rotate out of the jaws.

        The split follows ownership: the friction cone and the contact
        manifold are global solver settings and have nothing to do with
        whether the robot has a closed chain.  Anchor rebasing and coupler
        unlocking genuinely are loop-only (they only mean anything when
        CONNECT equalities exist to rebase/unlock) and stay inside
        ``_fix_gripper_loop_constraints``.  The gripper master force cap
        and the arm torque cap are the SAME class of bug as this one --
        also loop-closure-independent and subject to the same early return
        -- and are pulled out into their own ``_cap_gripper_master_force``
        / ``_cap_arm_actuator_force``.
        """
        solver = self._solver
        mjm = getattr(solver, "mj_model", None)
        mw = getattr(solver, "mjw_model", None)
        if mjm is None or mw is None:
            return
        try:
            import mujoco  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            logger.warn(f"[mjwarp-adapter] global contact options skipped ({exc!r})")
            return
        # MULTICCD: multi-point contact manifolds for convex pairs.
        # mjwarp emits a SINGLE contact point per convex pair by
        # default; a box pinched between two curved single-point
        # contacts walks out under lateral disturbance (measured:
        # 140 N symmetric pinch decaying monotonically to 85 N at
        # constant master angle while the arm moved, then release).
        # The flag is read at collision-kernel build time (first
        # collide after this), so setting it here (pre-warmup) works.
        if not self._enable_multiccd:
            logger.info(
                "[mjwarp-adapter] MULTICCD disabled by scene "
                "(newton.mujoco_enable_multiccd: false) — convex pairs fall "
                "back to a single contact point; expect a pinched box to "
                "walk out under lateral disturbance"
            )
        else:
            try:
                import mujoco_warp as _mjw  # noqa: PLC0415

                mw.opt.enableflags |= int(_mjw.EnableBit.MULTICCD)
                mjm.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
                logger.info("[mjwarp-adapter] MULTICCD enabled (multi-point convex contact manifolds)")
            except Exception as _exc:  # noqa: BLE001
                logger.warn(f"[mjwarp-adapter] MULTICCD enable failed: {_exc!r}")

        # Anti-creep: elliptic friction cone + impratio (mjwarp's impratio
        # only takes effect on the elliptic solve path: mu = friction *
        # impratio^-1/2). Regularised friction has no real static friction,
        # so a grasped object creeps out at roughly 1 mm/s.
        # The value comes from ``solver.impratio`` in physics_params.yaml
        # and from nowhere else; 0 (shipped default) leaves the solver's own
        # pyramidal cone alone.
        _cp = getattr(self._physics_params, "solver", None)
        imprat = float(getattr(_cp, "impratio", 0.0)) if _cp else 0.0
        if imprat > 0.0:
            try:
                mjm.opt.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
                mw.opt.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
                mjm.opt.impratio = imprat
                inv = 1.0 / (imprat**0.5)
                if hasattr(mw.opt, "impratio_invsqrt"):
                    try:
                        mw.opt.impratio_invsqrt.fill_(inv)
                    except AttributeError:
                        mw.opt.impratio_invsqrt = inv
                logger.info(
                    f"[mjwarp-adapter] elliptic cone + impratio={imprat:g} "
                    f"(anti-creep: stiffer tangential friction)"
                )
            except Exception as _exc:  # noqa: BLE001
                logger.warn(f"[mjwarp-adapter] cone/impratio config failed: {_exc!r}")

    def _cap_gripper_master_force(self, logger: Any) -> None:
        """Cap the force on the gripper's MASTER actuators. Loop-closure-
        independent — must run unconditionally, NOT from inside
        ``_fix_gripper_loop_constraints`` (that method returns early on
        grippers with no CONNECT equalities, e.g. G2_90d, which has 10
        JOINT mimic equalities and zero CONNECT — see that method's and
        ``_apply_global_contact_options``'s docstrings for the same class
        of bug).

        ``master_max_force: 0`` is meant to read as "keep the URDF effort
        limit", but in MuJoCo a forcerange of [0,0] means UNLIMITED — so
        closing hammers back and forth at +-1700 N*m and the grip force is
        anything but steady. With a cap, saturation becomes a stable
        constant-force grasp.

        Priority: YAML ``drive_gripper.master_max_force`` (>0) >
        ``gripper_contact.gripper_master_force_cap`` (shipped 0 = leave
        forcerange alone). The yaml is the only source — there is no env
        back door here; see ``config/physics_params.yaml``.
        Note that ``master_max_force`` is non-zero in both shipped
        profiles (18.0 shared, 50.0 RLinf), so that last fallback level is
        in practice unreachable.

        What gets capped is the gripper's driving MASTER, i.e. the joints
        the URDF ``<mimic>`` entries reference
        (``gripper_masters(mimic_followers)``, the two inner_joint1 on
        G2). Without a parsed mimic map this falls back to name matching,
        so scenes with no mimic are capped too.

        **No longer multi-env only.**  The stock adapter carries the same
        cap (``mujoco_warp.py::_cap_gripper_master_force``) and the two
        share the write through ``base.apply_actuator_force_cap``; nothing
        about a force ceiling needed more than one world, and while this
        was the only copy ``physics_knobs.md`` had to advertise the knob as
        inert on ``mujoco-warp``.  Policy — which knob, which joints —
        stays in each adapter, per ``adapters/__init__``'s rule that stock
        users inherit nothing from this file.
        """
        from genie_sim_engine_newton.common.gripper_annotations import joint_is_master  # noqa: PLC0415

        _cp2 = getattr(self._physics_params, "gripper_contact", None)
        _drv_grip = getattr(self._physics_params, "drive_gripper", None) if self._physics_params is not None else None
        _yaml_cap = float(getattr(_drv_grip, "master_max_force", 0.0) or 0.0)
        _fallback = float(getattr(_cp2, "gripper_master_force_cap", 0.0)) if _cp2 else 0.0
        _configured = _yaml_cap if _yaml_cap > 0.0 else _fallback
        _cap_src = "yaml" if _yaml_cap > 0.0 else "default"
        master_cap = _configured
        # ★ Own log line: this is the line the readback gate greps
        # (``host_run_teleop.sh`` §2d) — keep the format.
        logger.info(f"[mjc] gripper master_max_force={master_cap:g} N*m " f"(source={_cap_src})")
        # ``<= 0`` is the shipped value and means "leave forcerange
        # alone", keeping the effort limit the MJCF/URDF already carries. It
        # must not fall into the loop below: that would write 0 into both
        # bounds, clamping the gripper's drive force to nothing rather than
        # restoring the behaviour without this change.
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
        arm_max_torque`` (N*m). Loop-closure-independent, same reason as
        ``_cap_gripper_master_force`` — runs unconditionally, not gated by
        ``_fix_gripper_loop_constraints``'s early return.

        NOT the same knob as ``art_arm_shoulder/mid/wrist.max_effort``:
        that per-class table can only RAISE a joint's effort limit above
        whatever the URDF authored (see the per-class gain lookup above in
        this file) — it exists to backfill a class that needs MORE torque
        than a stingy URDF gives it, not to cap one that has too much.
        This is the opposite: a hard forcerange ceiling, independent of
        whatever the URDF or the class table authored.

        Why it's needed: these position-servo kp are 32000-60000 N*m/rad,
        and the compiled model's forcerange is empty ([0,0] = unlimited in
        MuJoCo). The right arm maps to ~408 N/mm at the end effector: 1mm
        of commanded overshoot into the table applies 408 N, 2mm applies
        816, uncapped. Measured consequence: an 8.3mm-thick coaster gets
        pressed flat through the table under a 2mm overshoot, and a 298g
        bottle gets launched to 1.14 m/s in a single 0.833ms substep —
        squarely in the "real launch" 0.75-3.9 m/s band the anti-ejection
        regression set calibrates against.

        100.0 (the shipped RLinf value) matches the URDF's own
        ``<limit effort="100">`` for each arm joint — capping to it
        saturates end-effector pressure at 357 N (0.87mm penetration,
        then holds).

        Only caps actuators whose joint classifies as an arm joint
        (``JK_ARM`` / ``JK_ARM_SHOULDER`` / ``JK_ARM_MID`` /
        ``JK_ARM_WRIST``, via ``classify_joint_by_name`` — name-pattern
        matching only, the same classifier the per-class drive-gain table
        above already trusts). Body/head/chassis keep their existing
        stiffness untouched: the hand is what slams into the table: body
        and head are not on that force chain, and capping them would only
        degrade pose tracking. The gripper has its own cap
        (``_cap_gripper_master_force``).

        **No longer multi-env only** — see the last paragraph of
        ``_cap_gripper_master_force``.
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

    def _fix_gripper_loop_constraints(self, logger: Any) -> None:
        """Make the gripper loop-closure CONNECT equalities functional.

        The robot USD models each Robotiq-style finger as a closed
        five-bar (base -[joint1]-> link1 -[joint3]-> link3 -[joint4]->
        link4 -[joint0]-> link2, plus a spherical pin base<->link2 in
        ``/genie/loop_joints`` marked excludeFromArticulation).  The
        engine re-adds those pins as Newton CONNECT equalities
        (setup/model.py loopfix), but two defects left them cosmetic:

        1. WRONG body2-side anchor: MuJoCo's compiler derives the second
           anchor at qpos0, and Newton emits qpos0=0 which is NOT the
           assembled configuration — the compiled closure held the pin
           12.8 mm out of its hole at every pose (user-visible
           "pin not in the hole"), fighting the linkage instead of
           closing it.  Fix: recompute anchor2 from anchor1 at the
           CURRENT (import = assembled) pose, where the loop residual is
           zero by construction.
        2. LOCKED couplers: joints 0/3/4 of each finger are inside the
           loop but carried full-strength position actuators
           (kp=5000/kd=800) pinning them at fixed angles — a closed
           four-bar with all coupler angles PD-locked has zero mobility;
           commanding the master swept the pads by only ~6 mm.  Fix:
           zero those actuators' gain/bias (the loop + master + mimic
           equality fully determine the motion; measured close sweep
           goes 11.2 cm -> 4.8 cm pad separation with <=0.05 mm loop
           residual).

        Also stiffens the connect solref to the same direct form as the
        mimic equality ([-50000, -500]); the default (0.02, 1.0) let the
        pin wander millimetres under load.
        """
        solver = self._solver
        mjm = getattr(solver, "mj_model", None)
        dw = getattr(solver, "mjw_data", None)
        mw = getattr(solver, "mjw_model", None)
        if mjm is None or dw is None or mw is None:
            return
        try:
            import mujoco  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            connects = [e for e in range(mjm.neq) if mjm.eq_type[e] == mujoco.mjtEq.mjEQ_CONNECT]
            if not connects:
                logger.info("[mjwarp-adapter] gripper loopfix: no connect equalities; skipped")
                return

            # --- 1+3. anchor rebase at the assembled (current) pose + stiffen
            d_tmp = mujoco.MjData(mjm)
            d_tmp.qpos[:] = dw.qpos.numpy()[0][: mjm.nq]
            mujoco.mj_forward(mjm, d_tmp)
            eq_data = mw.eq_data.numpy()
            eq_solref = mw.eq_solref.numpy()
            resid = []
            for e in connects:
                b1 = int(np.ravel(mjm.eq_obj1id[e])[0])
                b2 = int(np.ravel(mjm.eq_obj2id[e])[0])
                a1 = np.array(mjm.eq_data[e][0:3], dtype=np.float64)
                w1 = d_tmp.xpos[b1] + d_tmp.xmat[b1].reshape(3, 3) @ a1
                a2_old = np.array(mjm.eq_data[e][3:6], dtype=np.float64)
                w2_old = d_tmp.xpos[b2] + d_tmp.xmat[b2].reshape(3, 3) @ a2_old
                resid.append(float(np.linalg.norm(w1 - w2_old)) * 1000)
                a2 = d_tmp.xmat[b2].reshape(3, 3).T @ (w1 - d_tmp.xpos[b2])
                eq_data[..., e, 0:3] = a1
                eq_data[..., e, 3:6] = a2
                # k=1e6 (mass-normalized): the pin constraint is IN
                # SERIES with the grip-force path — at -50000 its
                # effective linear stiffness (~0.05 kg link x 5e4 =
                # 2.5 N/mm) capped the pad-on-object force at single-
                # digit newtons no matter the motor torque (measured:
                # master saturated at 50 N*m, block felt 0.1-3.8 N).
                # Oscillation period at 1e6 is ~6.3 ms = ~6.3 substeps
                # at rolling's sub_dt 1.0 ms (substep=10 @ 100Hz, vs.
                # feature's 0.833 ms @ substep=10/120Hz); vanilla
                # close-sweep verified stable with residual <=0.001 mm.
                eq_solref[..., e, 0] = -1000000.0
                eq_solref[..., e, 1] = -2000.0
                # keep mj_model in sync so the MJCF dump reflects reality
                mjm.eq_data[e][3:6] = a2
                mjm.eq_solref[e] = [-1000000.0, -2000.0]
            mw.eq_data.assign(eq_data)
            mw.eq_solref.assign(eq_solref)

            # --- In-loop coupler / driven joint sets (read from the asset
            # annotation, never guessed from names).
            #   * couplers = ``self._coupler_joints``, from the
            #     ``newton:isLoopCoupler`` annotation (name regex as fallback),
            #     handed in by ``set_coupler_joints``;
            #   * driven   = ``masters | (followers - couplers)``, i.e. the mimic
            #     masters plus the followers that are not couplers — derived
            #     from the structure of the mimic map, not from names.
            # ``gripper_loop_constraint_labels`` maps those two name sets onto
            # the compiled MJCF joint labels (tolerating prefixes).
            from genie_sim_engine_newton.common.gripper_annotations import (  # noqa: PLC0415
                gripper_loop_constraint_labels,
            )

            _mjm_joint_names = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in range(mjm.njnt)]
            coupler_labels, driven_labels = gripper_loop_constraint_labels(
                _mjm_joint_names,
                getattr(self, "_coupler_joints", set()) or set(),
                getattr(self, "_mimic_follower_names", set()) or set(),
                getattr(self, "_mimic_masters", set()) or set(),
            )
            logger.info(
                f"[mjwarp-adapter] gripper loopfix: {len(coupler_labels)} coupler joint(s), "
                f"{len(driven_labels)} driven joint(s)"
            )

            # --- 2. free the in-loop coupler joints (zero their actuators)
            gain = mw.actuator_gainprm.numpy()
            bias = mw.actuator_biasprm.numpy()
            freed = []
            for a in range(mjm.nu):
                if mjm.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                    continue
                j = int(np.ravel(mjm.actuator_trnid[a])[0])
                jn2 = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if jn2 in coupler_labels:
                    gain[..., a, 0] = 0.0
                    bias[..., a, 1] = 0.0
                    bias[..., a, 2] = 0.0
                    mjm.actuator_gainprm[a][0] = 0.0
                    mjm.actuator_biasprm[a][1] = 0.0
                    mjm.actuator_biasprm[a][2] = 0.0
                    freed.append(jn2.split("_genie_joints_")[-1])
            mw.actuator_gainprm.assign(gain)
            mw.actuator_biasprm.assign(bias)

            # --- 2b. Joint-level passive damping. The +-50 N*m force cap also
            # clips the PD damping term, degrading a saturated controller into
            # relay-like open/close oscillation, while freely swinging couplers
            # (no PD, only a tiny friction loss) rattle whenever the arm moves.
            # Joint-level dof_damping acts outside the actuator, is unaffected
            # by the force cap, and implicitfast integrates it implicitly, so it
            # is unconditionally stable. 8.0 on driven joints, 1.0 on in-loop
            # couplers — using the same two sets identified above.
            dampv = mw.dof_damping.numpy()
            n_damped = 0
            for j in range(mjm.njnt):
                jn2 = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                val = None
                if jn2 in driven_labels:
                    val = 8.0
                elif jn2 in coupler_labels:
                    val = 1.0
                if val is not None:
                    d0 = int(np.ravel(mjm.jnt_dofadr[j])[0])
                    dampv[..., d0] = np.maximum(dampv[..., d0], val)
                    mjm.dof_damping[d0] = max(float(mjm.dof_damping[d0]), val)
                    n_damped += 1
            mw.dof_damping.assign(dampv)

            logger.info(
                f"[mjwarp-adapter] gripper loopfix: dof_damping on {n_damped} gripper joint(s) "
                f"(8 driven / 1.0 couplers); "
                # Master forcerange capping moved to _cap_gripper_master_force
                # (it applies to every gripper, not just closed-loop ones —
                # cannot hang off this function's early return); that method
                # logs its own line.
                f"{len(connects)} connect "
                f"equalit(ies) re-anchored at the assembled pose (old residuals "
                f"{[f'{r:.1f}mm' for r in resid]}), solref -> (-1000000, -2000); "
                f"{len(freed)} in-loop coupler actuator(s) freed "
                f"({', '.join(freed[:6])}{'…' if len(freed) > 6 else ''}).  "
                f"The gripper four-bars are now mobile closed loops."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warn(f"[mjwarp-adapter] gripper loopfix failed (grasping will be broken): {exc!r}")

    def _build_base_pin(self, logger: Any) -> None:
        """Identify the robot ROOT free joint and prepare the per-substep pin.

        Only used by ``pin_base_mode: mocap``. A compile-time weld makes the
        base immovable, so a reset cannot re-place it; ``mocap`` keeps the FREE
        root joint (so reset can write qpos) and re-asserts the authored
        reference every substep, which is what makes those writes stick.

        The base is the FREE joint whose child body is NOT one of the
        ``/World/Objects/...`` scene props. Its armature is normal — it is a
        real dynamic body — so it is identified by ownership, not by inertia.

        Scene KINEMATIC bodies are NOT handled here: ``setup/stage.py::
        _demote_kinematic_bodies`` turns them into world-static colliders
        before ``add_usd`` ever sees them, so they never reach the solver as
        free joints at all.

        The reference buffer is refreshed by :meth:`sync_state_into_solver`, so
        reset / seeded randomization / instance switching keep working per
        world.
        """
        self._pin_qadr = None
        self._pin_dadr = None
        self._pin_qref = None
        self._base_pin_registered = False
        solver = self._solver
        mjm = getattr(solver, "mj_model", None)
        dw = getattr(solver, "mjw_data", None)
        if mjm is None or dw is None:
            return
        try:
            import mujoco  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            want_base = bool(getattr(self, "_pin_base_joint", False))
            qadr: list = []
            dadr: list = []
            base_names: list = []  # robot root free joint(s), mocap mode only
            for j in range(mjm.njnt):
                if not want_base:
                    break
                if mjm.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                    continue
                # Robot-root candidate: a NORMAL-armature free joint whose
                # child body is not one of the "/World/Objects/..." scene
                # props (same "the only non-object free joint" heuristic
                # ``reset.py::_world_free_joints`` uses on the Newton side;
                # here it's against the compiled MJC body name, which keeps
                # the full USD-path-derived label with "/" -> "_").
                body_id = int(mjm.jnt_bodyid[j])
                body_name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                if "Objects" in body_name:
                    continue
                jnt_name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or f"jnt{j}"
                d0 = int(mjm.jnt_dofadr[j])
                q0 = int(mjm.jnt_qposadr[j])
                qadr.extend(range(q0, q0 + 7))
                dadr.extend(range(d0, d0 + 6))
                base_names.append(f"{jnt_name} (body={body_name})")
            if want_base:
                if not base_names:
                    # Left as a warn here on purpose: this whole method
                    # runs inside a ``except Exception -> warn`` guard, so
                    # a raise would be downgraded to a warning anyway.
                    # ``_base_pin_registered`` stays False and
                    # ``_pick_substep_body`` refuses the build — one place
                    # owns the refusal.
                    logger.warn(
                        "[mjwarp-adapter] base pin: pin_base_mode=mocap requested "
                        "but no non-Objects FREE joint found (root_joint may not have "
                        "been deactivated, or the asset has no free root) — base pin "
                        "NOT registered."
                    )
                elif len(base_names) > 1:
                    logger.warn(
                        f"[mjwarp-adapter] base pin: {len(base_names)} candidate "
                        f"robot-root free joints found (expected 1) — pinning all of "
                        f"them: {base_names}"
                    )
            if not qadr:
                logger.info(
                    "[mjwarp-adapter] base pin: no robot root free joint to pin; "
                    "disabled (pin_base_mode is not mocap, or none was found)"
                )
                return
            device = solver.model.device
            nworld = int(dw.nworld)
            self._pin_qadr = wp.array(np.asarray(qadr, dtype=np.int32), dtype=wp.int32, device=device)
            self._pin_dadr = wp.array(np.asarray(dadr, dtype=np.int32), dtype=wp.int32, device=device)
            self._pin_qref = wp.zeros((nworld, len(qadr)), dtype=wp.float32, device=device)
            self._refresh_kinematic_ref()
            if base_names:
                self._base_pin_registered = True
                logger.info(
                    f"[mjwarp-adapter] base pin: root free joint(s) registered "
                    f"({', '.join(base_names)}) — {len(base_names) * 7} qpos + "
                    f"{len(base_names) * 6} qvel slot(s) clamped per substep, across "
                    f"all {nworld} world(s) (pin_base_mode=mocap)."
                )
        except Exception as exc:  # noqa: BLE001
            self._pin_qadr = None
            logger.warn(f"[mjwarp-adapter] base pin setup failed (base may drift): {exc!r}")

    def _refresh_kinematic_ref(self) -> None:
        """Re-snapshot pinned bodies' qpos into the pin reference buffer."""
        if self._pin_qadr is None or self._pin_qref is None:
            return
        dw = self._solver.mjw_data
        wp.launch(
            _gather_kinematic_ref_kernel,
            dim=(int(dw.nworld), int(self._pin_qadr.shape[0])),
            inputs=[self._pin_qadr, dw.qpos],
            outputs=[self._pin_qref],
            device=self._solver.model.device,
        )

    def sync_state_into_solver(self, state: Any, logger: Any = None) -> None:
        """One-shot push of ``state.joint_q`` / ``state.joint_qd`` into the
        live ``mjw_data.qpos`` / ``qvel``.

        Required companion of ``update_data_interval=0`` (see
        :meth:`build_solver`): the solver never re-reads the Newton state
        on its own, so every engine-side direct state write must be
        followed by exactly one call here or physics silently keeps the
        old configuration.  Current callers:

          * lifecycle init (``_phase_finalize_init_state``) — pushes the
            init joint pose after ``_apply_init_joint_pos``;
          * ``reset.py`` — pushes the restored snapshot (robot joints,
            free-body object poses, randomization / instance switches).

        Safe to call between frames only (no CUDA graph in flight —
        reset/init both satisfy this).  No-op with a warn when the solver
        isn't built yet.
        """
        solver = getattr(self, "_solver", None)
        if solver is None or getattr(solver, "mjw_data", None) is None:
            if logger is not None:
                logger.warn("[mjwarp-adapter] sync_state_into_solver: solver not built; skipped")
            return
        solver._update_mjc_data(solver.mjw_data, solver.model, state)
        # The pushed state carries the (possibly reset / randomized /
        # instance-switched) kinematic-body placements — re-snapshot the
        # pin reference so the per-substep clamp holds the NEW pose.
        self._refresh_kinematic_ref()
        # After pushing qpos/qvel, the derived state (xpos/xquat, contact set,
        # efc_*) must be recomputed. Without it, a one-step-stale derived state
        # describes the configuration that just diverged — the post-reset
        # contact set carries enormous penetration depths, and the next frame
        # applies those stale constraints to a freshly reset robot.
        #
        # This line and ``sync_control_into_solver`` are complementary and must
        # both be present: adding forward() without syncing ctrl merely makes
        # the leftover pre-reset command compute more "accurately", which is
        # worse. Confirm that sync is still there before removing this line.
        try:
            import mujoco_warp as _mjw  # noqa: PLC0415

            _mjw.forward(solver.mjw_model, solver.mjw_data)
        except Exception as exc:  # noqa: BLE001 — a reset must not abort on this
            if logger is not None:
                logger.warn(
                    f"[mjwarp-adapter] forward() failed: {exc!r} — "
                    f"derived state is still pre-reset; physics may "
                    f"diverge immediately"
                )

    def sync_control_into_solver(self, state: Any, control: Any, logger: Any = None) -> None:
        """Push ``control.joint_target_q`` into the live ``mjw_data.ctrl``
        in one shot.

        This is the CONTROL-SIDE counterpart of
        :meth:`sync_state_into_solver`. The state half has always been there
        (reset pushes qpos/qvel); the control half was missing — and
        ``mjw_data.ctrl`` is the input MuJoCo actuators actually read.
        ``_apply_mjc_control`` is only called inside ``solver.step``, so a reset
        changed Newton's ``joint_target_q`` only, leaving ``ctrl`` holding the
        PRE-RESET command.

        Consequence (measured 2026-08-05, see
        ``docs/verify_2026-08-04_teleop_nan/VERIFY.md``): resetting after a
        divergence teleports the robot back to init, zeroes velocities, restores
        the PD targets and clears the solver carryover — all five probes read
        0.000000 — yet ``ctrl`` is 1.1-1.8 rad off the baseline. The next frame
        the actuators drive on that stale command, ``qfrc_actuator`` spikes to
        892-22429 (130-350 when healthy), the arm is pushed into its joint
        limits and physics blows up again: velocity climbs monotonically from 0
        to 2388 rad/s. In other words "reset recovers" only recovered position,
        not control.

        Every actuator-related model field was diffed (gainprm / biasprm /
        forcerange / ctrlrange / gear / dof_armature / dof_damping /
        frictionloss / jnt_range / body_mass) and ALL were unchanged; the only
        thing that differed was ``ctrl`` — so the problem is not in the model,
        it is this one missing sync.

        Called on frame boundaries only (reset qualifies), the same safety
        precondition as the state side. Warns and skips when the solver is not
        built yet, rather than manufacturing a new failure.
        """
        solver = getattr(self, "_solver", None)
        if solver is None or getattr(solver, "mjw_data", None) is None:
            if logger is not None:
                logger.warn("[mjwarp-adapter] sync_control_into_solver: " "solver not built; skipped")
            return
        if control is None:
            return
        try:
            solver._apply_mjc_control(solver.model, state, control, solver.mjw_data)
        except Exception as exc:  # noqa: BLE001 — a reset must not abort on a
            # failed sync
            if logger is not None:
                logger.warn(
                    f"[mjwarp-adapter] sync_control_into_solver "
                    f"failed: {exc!r} — ctrl still holds its pre-reset "
                    f"value; physics may diverge immediately"
                )

    # Solver-state fields carried across steps — the next step reads them as
    # input instead of recomputing from qpos/qvel/ctrl. Every other field
    # (qacc, qfrc_*, cacc, cfrc_*, actuator_*, contact, efc_*) is re-derived
    # each step and needs no clearing.
    #   * qacc_warmstart — the constraint solver's initial guess
    #   * act            — actuator activation state (na==0 for this robot's
    #                      position actuators, cleared anyway so stateful
    #                      actuator models stay correct too)
    # qfrc_applied / xfrc_applied are deliberately absent: this engine never
    # writes those two slots.
    _SOLVER_CARRYOVER_FIELDS = ("qacc_warmstart", "act")

    def clear_solver_carryover(self, env_ids: Any = None, logger: Any = None) -> None:
        """Drop the solver's cross-step carry-over for ``env_ids`` (all
        worlds when None). Companion of :meth:`sync_state_into_solver`.

        Why this is unconditional rather than a NaN special case: a reset
        TELEPORTS the state to a different configuration, so any solver
        state carried over from the pre-teleport configuration is stale by
        definition — a warm start for a pose that no longer exists. Clearing
        it is what reset always meant; NaN recovery just falls out of it.

        That recovery matters on its own: an action that drives the
        integrator to NaN used to brick the sim server permanently, because
        reset restored qpos/qvel (see reset.py) while ``qacc_warmstart``
        kept its NaNs and re-poisoned the very next constraint solve. RL
        exploration produces such actions by design, so "one bad action
        needs a server restart" is not an acceptable failure mode. Verified
        2026-07-30 — see docs/verify_2026-07-30_instance_pin/VERIFY.md.

        Safe between frames only (no CUDA graph in flight — reset satisfies
        this, same as ``sync_state_into_solver``). No-op with a warn when
        the solver isn't built yet.
        """
        import numpy as np  # noqa: PLC0415  (module top doesn't import np)

        solver = getattr(self, "_solver", None)
        dw = getattr(solver, "mjw_data", None) if solver is not None else None
        if dw is None:
            if logger is not None:
                logger.warn("[mjwarp-adapter] clear_solver_carryover: " "solver not built; skipped")
            return
        nworld = int(getattr(dw, "nworld", 0) or 0)
        ids = list(range(nworld)) if env_ids is None else [int(w) for w in env_ids]
        if not ids or nworld <= 0:
            return
        device = getattr(dw.qacc_warmstart, "device", None)
        worlds = wp.array(np.asarray(ids, dtype=np.int32), dtype=wp.int32, device=device)
        cleared = []
        for name in self._SOLVER_CARRYOVER_FIELDS:
            buf = getattr(dw, name, None)
            # ndim guard: these are (nworld, n) 2d arrays; `act` is (nworld, 0)
            # when the model has no stateful actuators — launching dim=(k, 0)
            # is a no-op, but skip it so the log stays honest.
            if buf is None or getattr(buf, "ndim", 0) != 2 or buf.shape[1] == 0:
                continue
            wp.launch(
                _zero_world_rows_kernel, dim=(len(ids), buf.shape[1]), inputs=[worlds], outputs=[buf], device=device
            )
            cleared.append(name)
        if logger is not None and cleared:
            logger.info(f"[mjwarp-adapter] solver carry-over cleared for " f"world(s) {ids}: {', '.join(cleared)}")

    # ------------------------------------------------------------------
    # Robot BASE mocap-body per-reset xy randomization
    # (pin_base_to_world:false native path — see reset.py section 3)
    # ------------------------------------------------------------------

    def base_pin_registered(self) -> bool:
        """True when ``_build_base_pin`` found and clamped a robot root
        free joint — i.e. ``pin_base_mode: mocap`` is actually in force.

        ``supports_base_pin`` (class attribute) says this adapter CAN
        hold the base; this says it DID, and is only knowable after the
        solver is built.  ``_pick_substep_body`` asserts it.
        """
        return bool(getattr(self, "_base_pin_registered", False))

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
        # Pin kinematic bodies BEFORE the solve so collision + constraints
        # see them at their authored pose with zero velocity (mocap
        # semantics — see _build_base_pin).  Plain wp.launch: capture-
        # safe, so it bakes into the CUDA graph on the captured path.
        # Gated at build time by ``pin_base_mode: mocap`` — a non-empty table
        # simply means "there is a base to pin". Scene kinematic bodies never
        # reach here: stage-level demotion turns them into static colliders.
        if self._pin_qadr is not None:
            dw = self._solver.mjw_data
            wp.launch(
                _pin_kinematic_dofs_kernel,
                dim=(int(dw.nworld), int(self._pin_qadr.shape[0])),
                inputs=[self._pin_qadr, self._pin_dadr, self._pin_qref],
                outputs=[dw.qpos, dw.qvel],
                device=self._solver.model.device,
            )
        # SolverMuJoCo's JOINT_TARGET actuators read control.joint_target_q
        # internally; nothing else to inject.
        self._solver.step(state_in, state_out, control, None, sim_dt)
