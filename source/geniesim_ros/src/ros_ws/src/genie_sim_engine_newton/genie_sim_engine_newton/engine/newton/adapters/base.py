# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Solver-adapter abstract base class.

A ``SolverAdapter`` absorbs all solver-specific behavior — PD-drive prep,
solver construction, target-buffer storage — so that ``_LifecycleMixin``
in ``lifecycle.py`` can stay solver-agnostic.

The lifecycle calls hooks at well-defined points; see each method's
docstring for ordering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import warp as wp


def resolve_mjc_iterations(physics_params: Any) -> Optional[int]:
    """Resolve ``SolverMuJoCo(iterations=...)`` from ``solver.mjc_iterations``.

    Returns ``None`` when the key is unset or 0, which leaves SolverMuJoCo
    on its own default (MuJoCo's 100).

    This is deliberately NOT derived from the launcher's
    ``physics_solver_iterations``.  That knob is the PARTICLE-solver
    iteration count: FEM tet scenes run it at 4 because ``substeps=24``
    carries the stiffness.  The combo solvers (``mjxpbd``, ``mjvbd``) drive
    both solvers from one adapter, and forwarding the particle count ran
    MuJoCo's rigid constraint solve at 4 against a default of 100.  A
    frictional contact block cannot converge in 4 iterations, so a grasped
    object creeps out of the jaws — which reads as a friction problem and
    sends you tuning coefficients that were never wrong.

    The two solvers need independent counts, and the second one comes from
    ``physics_params.yaml`` rather than a module constant in an adapter —
    see ``geniesim_ros/AGENTS.md``, "one knob, one route".

    ``physics_params`` is duck-typed: these adapters do not import from
    ``genie_sim_engine_core``, so the read tolerates a params object with
    no ``solver`` section at all.
    """
    solver = getattr(physics_params, "solver", None)
    if solver is None:
        return None
    try:
        n = int(getattr(solver, "mjc_iterations", 0) or 0)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def resolve_mjc_impratio_cone(physics_params: Any) -> "tuple[Optional[float], Optional[str]]":
    """Resolve ``SolverMuJoCo(impratio=..., cone=...)`` from ``solver.impratio``.

    Returns ``(None, None)`` when the key is unset or 0, which leaves
    SolverMuJoCo on MuJoCo's own defaults (``impratio=1``, pyramidal cone).

    **One key, two settings, on purpose.**  mjwarp's ``impratio`` only takes
    effect on the elliptic solve path — the friction bound is
    ``mu = friction * impratio**-0.5``, which the pyramidal cone never
    evaluates.  Setting ``impratio`` without the cone is therefore a no-op
    that reads like a tuning knob, so the cone follows from it rather than
    being a second key an operator can get half-right.  That is why there is
    no ``solver.cone``: see ``geniesim_ros/AGENTS.md``, "one knob, one route".

    Why it matters: the pyramidal cone regularises friction, which has no
    real static regime, so a grasped object creeps out of the jaws at
    roughly 1 mm/s under load.  Same symptom as the iteration bug
    ``resolve_mjc_iterations`` documents, and the same wrong diagnosis
    (friction coefficients) waiting at the end of it.

    ``physics_params`` is duck-typed for the same reason as above: these
    adapters do not import from ``genie_sim_engine_core``.
    """
    solver = getattr(physics_params, "solver", None)
    if solver is None:
        return (None, None)
    try:
        imprat = float(getattr(solver, "impratio", 0.0) or 0.0)
    except (TypeError, ValueError):
        return (None, None)
    if imprat <= 0.0:
        return (None, None)
    return (imprat, "elliptic")


def apply_actuator_force_cap(solver: Any, logger: Any, *, cap: float, select: Any, what: str) -> Optional[list]:
    """Write ``+-cap`` into ``actuator_forcerange`` for the selected joints.

    The MECHANISM only.  Which knob supplies ``cap`` and which joints
    ``select`` accepts stay in the calling adapter, because those are
    policy and they differ: the gripper cap keys off the mimic master set,
    the arm cap off ``classify_joint_by_name``.  What does NOT differ is
    this write, and it was duplicated verbatim in two adapters long enough
    for one copy to acquire the ``actuator_forcelimited`` sweep the other
    lacked.

    ``select`` is called with the MuJoCo joint name behind each actuator
    and returns truthy to cap it.  Only ``mjTRN_JOINT`` actuators are
    considered; tendon and site transmissions are left alone.

    Returns the short names of the actuators written, or ``None`` when the
    cap could not be applied at all (no compiled handles, or no mujoco /
    numpy).  An empty list means it ran and matched nothing — the caller
    decides whether that is worth a log line.

    Both bounds are written through ``frange[..., a, :]``, so the same
    call serves a single-world ``mjw_model`` and a multi-world one: the
    ellipsis spans however many worlds the model carries.

    ``actuator_forcelimited`` is swept at the end rather than per
    actuator because mjwarp reads the flag array, not ``mj_model``'s:
    writing the range without the flag leaves the model carrying a
    forcerange MuJoCo never enforces, which looks exactly like a cap that
    was applied and ignored.
    """
    mjm = getattr(solver, "mj_model", None)
    mw = getattr(solver, "mjw_model", None)
    if mjm is None or mw is None:
        return None
    try:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.warn(f"[mjwarp-adapter] {what} force cap skipped ({exc!r})")
        return None

    frange = mw.actuator_forcerange.numpy()
    capped: list = []
    for a in range(mjm.nu):
        if mjm.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        j = int(np.ravel(mjm.actuator_trnid[a])[0])
        jn = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not select(jn):
            continue
        frange[..., a, 0] = -cap
        frange[..., a, 1] = cap
        mjm.actuator_forcerange[a] = [-cap, cap]
        mjm.actuator_forcelimited[a] = 1
        capped.append(jn.split("_genie_joints_")[-1])
    mw.actuator_forcerange.assign(frange)
    if hasattr(mw, "actuator_forcelimited"):
        fl = mw.actuator_forcelimited.numpy()
        for a in range(mjm.nu):
            if mjm.actuator_forcelimited[a]:
                fl[..., a] = 1
        mw.actuator_forcelimited.assign(fl)
    return capped


# Entries of ``physics_params.yaml`` whose ONLY readers are the MuJoCo-Warp
# adapters -- so on ``featherstone`` / ``avbd`` they are inert.
#
# This is a subset of the *all mjwarp* rows in
# ``genie_sim_engine_core/docs/physics_knobs.md``, on purpose: entries read
# from the shared Newton layer (``setup/model.py``'s pad decomposition) or
# from the MJCF post-processor reach every adapter and must NOT be reported
# as inert.  ``tools/test_knob_reader_scope.py`` holds this tuple against the
# document so a new adapter-only knob cannot quietly skip the warning.
#
# ``(section attribute, leaf attribute, the dotted name the yaml uses)`` --
# the first two are what the code reads off the dataclass, the third is what
# an operator has to go edit.
_ADAPTER_ONLY_PARAM_KEYS = (
    ("solver", "mjc_iterations", "solver.mjc_iterations"),
    ("solver", "impratio", "solver.impratio"),
    ("gripper_contact", "gripper_master_force_cap", "gripper_contact.gripper_master_force_cap"),
    ("arm_contact", "arm_max_torque", "arm_contact.arm_max_torque"),
    # No shipped default on the dataclass, so this one always reads as
    # authored -- which is the truthful answer: every profile sets a gripper
    # kp, and on a non-mjwarp adapter every one of them is ignored.
    ("drive_gripper", "master_stiffness", "usd_drive_api.gripper.master_stiffness"),
)

# Solver keys that route to a MuJoCo-Warp adapter, stock or multi-env.
# Kept here rather than in the factory so the predicate sits next to the
# key list it qualifies.
_MJWARP_SOLVER_KEYS = frozenset(
    (
        "mujoco-warp",
        "mujoco_warp",
        "mujoco-warp-multi-env",
        "mujoco_warp_multi_env",
        "mjvbd",
        "mjxpbd",
    )
)


def _authored_value(section: Any, leaf: str) -> Optional[str]:
    """``repr`` of ``section.leaf`` if it differs from the dataclass default.

    ``None`` means "left at the shipped default", which is not worth
    warning about: every profile carries every key, so the question is not
    whether the key is present but whether anyone set it to something.

    The default comes from ``dataclasses.fields`` rather than a second copy
    of the shipped numbers here -- a hand-maintained copy is one more thing
    to drift, and it would drift silently in the direction of a warning
    that names a value the operator never typed.  A field with no plain
    default, or a params object that is not a dataclass at all, counts as
    authored: over-reporting is recoverable, and the duck-typed path has no
    other evidence to go on.
    """
    try:
        val = getattr(section, leaf)
    except AttributeError:
        return None
    try:
        import dataclasses  # noqa: PLC0415

        for f in dataclasses.fields(type(section)):
            if f.name != leaf:
                continue
            if f.default is not dataclasses.MISSING and val == f.default:
                return None
            break
    except Exception:  # noqa: BLE001
        pass
    return repr(val)


def warn_inert_params(physics_solver: str, physics_params: Any, logger: Any = None) -> list:
    """Warn when the active adapter reads none of the knobs the profile sets.

    ``physics_knobs.md`` opens by saying that setting a value the active
    adapter never reads is silent -- no warning, no error, the yaml just has
    no effect -- and then the only thing standing between an operator and
    that silence is a column in a markdown table.  This makes the one case
    that a column cannot help with loud: a non-MuJoCo adapter, where the
    whole MuJoCo half of the profile is a no-op at once.

    Scoped to the adapter FAMILY on purpose.  Within mjwarp, which adapter
    reads what is checked statically by
    ``tools/test_knob_reader_scope.py`` against the document; across
    families there is nothing to check, because ``featherstone`` / ``avbd``
    are not even handed ``physics_params``.  That is also why the
    check lives here and not in each adapter: the adapters that need
    warning about are the ones that never see the params.

    Returns the list of inert ``"name=value"`` strings so a test can assert
    on it without a logger.  Never raises: a params object it cannot read
    yields an empty list.
    """
    key = (physics_solver or "").strip().lower()
    if key in _MJWARP_SOLVER_KEYS or physics_params is None:
        return []
    inert = []
    for section_attr, leaf, dotted in _ADAPTER_ONLY_PARAM_KEYS:
        section = getattr(physics_params, section_attr, None)
        if section is None:
            continue
        shown = _authored_value(section, leaf)
        if shown is not None:
            inert.append(f"{dotted}={shown}")
    if inert and logger is not None:
        logger.warn(
            f"[newton] physics_solver={physics_solver!r} reads none of the MuJoCo "
            f"tuning knobs, so {len(inert)} entr{'y' if len(inert) == 1 else 'ies'} "
            f"set in the active physics profile have no effect: {', '.join(inert)}. "
            f"Either run a mujoco-warp adapter or stop setting them -- see "
            f"genie_sim_engine_core/docs/physics_knobs.md, the 'read by' column."
        )
    return inert


class SolverAdapter(ABC):
    # ------------------------------------------------------------------
    # Capabilities (lifecycle queries these before build)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier used in log messages.

        e.g. ``"mujoco-warp"``, ``"featherstone"``.
        """

    @property
    @abstractmethod
    def supports_cloth(self) -> bool:
        """Whether this solver handles VBD cloth entries in the scene yaml.

        Lifecycle uses this to gate cloth scenes: an mjwarp solver with
        cloth entries in the scene yaml raises early instead of silently
        running rigid-only.
        """

    #: Whether this solver can clamp the robot root FREE joint to a
    #: settable reference every substep — what ``pin_base_mode: mocap``
    #: is built on (``make_adapter(pin_base_joint=True)``).
    #:
    #: ``engine_base`` gates on this the same way ``setup/model.py`` gates
    #: cloth: an adapter without the clamp raises early instead of
    #: silently dropping the weld and letting the base fall under
    #: gravity with nothing holding it.
    #:
    #: Non-abstract on purpose, unlike ``supports_cloth``: every solver
    #: has to think about cloth, but the base clamp is a single-adapter
    #: extension — making it abstract would mean four edits to say "no"
    #: four times (``mujoco_warp``, ``mujoco_warp_multi_env``,
    #: ``featherstone``, ``avbd`` — ``mjvbd`` / ``mjxpbd`` / ``fsvbd`` are
    #: factory keys onto those classes, not separate adapters).
    supports_base_pin: bool = False

    # ------------------------------------------------------------------
    # Pre-finalize: register custom attributes on the builder
    # ------------------------------------------------------------------

    @abstractmethod
    def register_custom_attributes(self, builder: Any) -> None:
        """Register solver-specific custom attributes on the ModelBuilder.

        Called BEFORE ``builder.add_usd()``.  The attributes a solver
        authors on the builder MUST match the solver instantiated after
        finalize — VBD reads ``edge_rest_angle`` etc that XPBD doesn't
        author, and vice versa.
        """

    # ------------------------------------------------------------------
    # Post-finalize, pre-solver: mutate model arrays
    # ------------------------------------------------------------------

    @abstractmethod
    def prepare_model(
        self,
        model: Any,
        logger: Any,
        mimic_followers: Optional[Dict[str, list]] = None,
        robot_prefix: str = "",
    ) -> None:
        """Mutate model arrays BEFORE solver construction.

        ``robot_prefix`` (the active robot's USD prefix, e.g. ``"genie"``)
        lets an adapter tell robot-owned joints from SCENE-owned ones by
        body ownership — used by the mjwarp adapter to classify articulated
        scene objects (door hinges / drawer slides) without a per-asset
        joint-name regex.  Empty ⇒ ownership check disabled (name-based
        classification only).  Ignored by adapters that don't need it.

        For mjwarp: forces ``joint_target_mode := POSITION`` and overrides
        ``joint_target_ke/kd`` from the launcher fallback (because Isaac's
        URDF importer authors a vestigial DriveAPI that Newton interprets
        as VELOCITY mode and ke=625).

        For Featherstone: no-op (velocity injection bypasses PD torques).
        """

    # ------------------------------------------------------------------
    # Solver construction
    # ------------------------------------------------------------------

    @abstractmethod
    def build_solver(
        self,
        model: Any,
        sim_substeps: int,
        sim_iterations: int,
        logger: Any,
        mass_matrix_interval: int = 0,
    ) -> Any:
        """Construct and return the solver instance.

        Encapsulates solver-specific config (njmax/nconmax for mjwarp,
        update_mass_matrix_interval for Featherstone, etc.).

        ``mass_matrix_interval`` is the Featherstone-specific tunable for
        the M(q) rebuild cadence; ``0`` (default) means "use the
        solver-appropriate default" (Featherstone falls back to
        ``sim_substeps`` = one rebuild per frame).  Adapters that don't
        have a mass-matrix concept (mjwarp) ignore it.
        """

    # ------------------------------------------------------------------
    # Target-position buffer
    # ------------------------------------------------------------------

    @abstractmethod
    def init_target_buffer(
        self,
        model: Any,
        control: Any,
        logger: Any,
    ) -> None:
        """Allocate the position-target buffer and seed with ``model.joint_q``.

        Called AFTER ``_apply_init_joint_pos`` so the seed IS the init pose.
        Buffer must be allocated once; later writes use ``.assign()`` to
        preserve the pointer captured by the CUDA graph.
        """

    @abstractmethod
    def target_buffer(self) -> Optional[wp.array]:
        """The ``wp.array`` ``apply_commands`` scatters into.

        Returns ``None`` if ``init_target_buffer`` hasn't run yet (i.e.
        commands arriving before model build completes are silently
        dropped, which is the correct startup behavior).
        """

    # ------------------------------------------------------------------
    # Post-joint-map: solver-specific tweaks that need joint name → DOF
    # ------------------------------------------------------------------

    @abstractmethod
    def post_joint_map(
        self,
        model: Any,
        jindex: Any,
        control: Any,
        logger: Any,
    ) -> None:
        """Solver-specific setup that needs the resolved joint-index map.

        Called ONCE after ``_build_joint_map`` and ``_apply_init_joint_pos``
        in ``lifecycle.py``.

        ``jindex`` is the engine's ``JointIndex`` snapshot (see
        ``engine/newton/joint_index.py``). Adapters that want a
        ``{name: dof}`` dict shape can call
        ``jindex.name_to_dof()``; adapters that want full slice
        metadata (joint type, q-vs-qd widths, slice extents) iterate
        ``jindex.slices()``. Routing the JointIndex itself avoids a
        per-adapter copy and keeps the q/qd asymmetry resolved in one
        place.

        For mjwarp: no-op (PD already set in ``prepare_model``).

        For Featherstone: applies selective PD on PASSIVE joints (those
        not driven by ROS commands — body/head on the G2).  Velocity
        injection alone has O(dt²) drift under constant external torque
        and lets heavy root chains accumulate into their joint limits
        within one frame; selective PD adds a restoring force on those
        joints while command-driven DOFs (arms/grippers/wheels) stay
        velocity-injection only.
        """

    # ------------------------------------------------------------------
    # Per-substep robot step (called INSIDE the captured CUDA graph)
    # ------------------------------------------------------------------

    @abstractmethod
    def substep(
        self,
        model: Any,
        state_in: Any,
        state_out: Any,
        control: Any,
        sim_dt: float,
    ) -> None:
        """Run one rigid-body substep.

        Called from ``engine.py:_simulate_substeps`` inside the
        ``wp.ScopedCapture()`` block — implementations MUST do only GPU
        ops (``wp.launch``, ``solver.step``, ``arr.assign``); no Python
        branching beyond startup-time decisions, no allocations.

        For mjwarp: ``self._solver.step(state_in, state_out, control, None, sim_dt)``.

        For Featherstone: ``wp.launch(_kernel_velocity_inject, ...)``
        first to write ``joint_qd = target − q`` into ``state_in``,
        then ``self._solver.step(...)``.
        """
