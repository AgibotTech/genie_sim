# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Solver-adapter factory.

Every supported physics solver returns a ``SolverAdapter`` instance;
unknown solvers raise ``ValueError`` so a typo can't silently fall back
to inline default behavior.

Solver taxonomy
---------------
* **Pure solvers** integrate everything (rigid bodies, particles, etc.)
  inside a single ``solver.step`` call — no composition with another
  solver downstream.  Currently: ``avbd``, ``mujoco-warp``.
* **Combo solvers** delegate cloth/particle integration to a separate
  Newton solver (always ``SolverVBD`` in this codebase) that
  ``lifecycle.py`` constructs in addition to the rigid solver returned
  here.  Currently: ``fsvbd`` (Featherstone-rigid + VBD-cloth).

The factory does NOT build the cloth solver — that's
``lifecycle._build``'s job, gated on whether the scene yaml declares any
cloth/particle entry.  Whether the adapter is "pure" vs "combo" is
exposed via ``adapter.supports_cloth`` (pure cloth-supporting → True
forces lifecycle to skip a separate cloth solver; combo adapters report
False so lifecycle adds one).
"""

from __future__ import annotations

from typing import Any, Optional

from genie_sim_engine_newton.engine.newton.adapters.base import SolverAdapter, warn_inert_params


def make_adapter(
    physics_solver: str,
    *,
    mujoco_pd_ke: float = 0.0,
    mujoco_pd_kd: float = 0.0,
    mujoco_njmax: int = 0,
    mujoco_nconmax: int = 0,
    mujoco_save_to_mjcf: str = "",
    physics_params: Optional[Any] = None,
    joint_params: Optional[Any] = None,
    physics_hz: float = 0.0,
    imu_cfg: Optional[Any] = None,
    pin_base_joint: bool = False,
    mujoco_enable_multiccd: Optional[bool] = None,
    logger: Optional[Any] = None,
) -> SolverAdapter:
    """Return an adapter for ``physics_solver``.

    Supported solvers:
      * ``mujoco-warp`` / ``mujoco_warp``  -> :class:`MuJoCoWarpAdapter`   (pure)
      * ``mjvbd`` / ``mjxpbd``             -> :class:`MuJoCoWarpAdapter`   (combo)
      * ``fsvbd`` / ``featherstone``       -> :class:`FeatherstoneAdapter` (combo)
      * ``avbd``                           -> :class:`AVBDAdapter`         (pure)

    ``mjvbd`` is the MJW analogue of ``fsvbd``: MuJoCo-Warp drives the
    rigid articulation, and ``lifecycle.py`` builds a separate
    ``SolverVBD`` with ``integrate_with_external_rigid_solver=True`` for
    cloth particles.  The adapter object is identical to the pure
    ``mujoco-warp`` adapter — the only difference is that the scene yaml
    declares cloth entries, which triggers the cloth-solver build.
    Useful when you want MJW's implicit integrator for the robot while
    keeping VBD cloth (e.g., to compare against the Featherstone+VBD
    path on the same scene with a single-token swap).

    ``fsvbd`` is the pipeline-composition name: Featherstone-for-rigid
    combined with a separate ``SolverVBD`` instance constructed by
    ``lifecycle.py`` for cloth particles.  The two solvers run
    back-to-back inside each substep; rigid bodies are kinematic from
    SolverVBD's point of view (``integrate_with_external_rigid_solver=True``).

    ``avbd`` selects a single ``SolverVBD`` instance in unified mode
    (``integrate_with_external_rigid_solver=False``): one ``solver.step``
    advances both rigid (via Augmented VBD) and cloth (via VBD).
    Lifecycle skips the separate cloth-solver build for this path, and
    the engine binds ``_substep_body_avbd_unified`` instead of
    ``_substep_body_franka_vbd_cloth``.

    ``mujoco_enable_multiccd`` comes from the SCENE yaml
    (``newton.mujoco_enable_multiccd``) and is deliberately TRI-STATE.
    MuJoCo emits a single contact point per convex pair by default, and a box
    pinched between two curved single-point contacts walks out under lateral
    disturbance — measured on the multi-env adapter as a 140 N symmetric
    pinch decaying monotonically to 85 N at constant master angle.  That
    adapter therefore turns MULTICCD on unconditionally today, while the
    stock adapter has never had it at all.  Collapsing those two into one
    plain ``bool`` default would silently pick a side: ``False`` would undo
    the multi-env fix for every scene that does not name the key, and
    ``True`` would enable it for stock scenes nobody has measured.  ``None``
    (the absent case) means "each adapter keeps its own default"; ``True`` /
    ``False`` from a scene override both.

    ``mujoco-warp`` is also pure but rigid-only — cloth scenes refuse
    to start under it (``supports_cloth == False``).

    ``mujoco_save_to_mjcf`` is mjwarp-only: when set to a path the
    SolverMuJoCo writes the compiled MJCF (joints, actuators, gainprm/
    biasprm, contact params, equality constraints, joint limits) so
    operators can diff against a reference MJCF, run ``mujoco.viewer``
    on it, or validate it with ``mjpython``-side tooling.

    ``physics_params`` (mjwarp-only): when set, the adapter classifies
    each DOF via ``common.joint_classification`` and pulls per-class
    ``ke / kd / max_effort`` from ``PhysicsParams.art_body`` /
    ``.art_head`` / ``.art_arm_shoulder`` / ``.art_arm_mid`` /
    ``.art_arm_wrist`` (sub-classed body/head/arm) /
    ``.art_chassis_drive`` / ``.art_chassis_steer`` /
    ``.drive_gripper.master_*``, with ``.art_default`` as the
    fallback for un-classified joints.  Brings newton-standalone
    into parity with the isaac_newton wrapper (which applies these
    values via ``ArticulationView.set_dof_stiffnesses`` at runtime).
    When unset, the adapter falls back to its unscaled ``effort × 10``
    heuristic — used for ``test_newton_solver.py`` invocations that
    don't load the YAML.

    ``imu_cfg`` (mjwarp-only): the scene yaml's ``imu:`` block
    (``parent_body`` + ``sites`` + ``sensors``).  When set, the adapter
    injects the corresponding ``<site>`` / ``<sensor>`` elements into the
    dumped ``robot_runtime.xml`` (see
    ``common.mjcf_postprocess.add_imu_sites_and_sensors``) so the static
    MJCF exposes the G2 chassis IMUs.  ``None`` (scene declares no IMU) →
    the injection step no-ops.  Affects only the save_to_mjcf artifact,
    not the live solver.

    ``mujoco_njmax`` / ``mujoco_nconmax`` (mjwarp-only): constraint-solver
    capacity.  ``0`` means "leave it to the adapter" — and the two adapters
    read that differently ON PURPOSE.  The STOCK adapter keeps its fixed
    ``_DEFAULT_NJMAX``/``_DEFAULT_NCONMAX`` (1024/512), so this function does
    NOT forward zeros into it: passing 0 there would replace a working
    default with no capacity at all — only positive values reach it, so an
    operator can still tune it.  The multi-env adapter instead auto-scales
    from the scene body count when it sees 0, so it takes the values
    verbatim.

    ``pin_base_joint`` (mjwarp multi-env only): set when the caller's
    ``pin_base_mode`` resolves to ``"mocap"``.  The adapter's
    ``_build_base_pin`` then also identifies the robot root FREE joint and
    unions its qpos/qvel slots into the same per-substep clamp used for
    KINEMATIC scene bodies, so the base holds its authored pose (settable
    per reset) without a compile-time weld.  ``False`` (default) is a no-op.
    """
    key = physics_solver.strip().lower()
    # Before dispatch, because the adapters that need this warning are the
    # ones that never receive ``physics_params`` -- ``FeatherstoneAdapter``
    # and ``AVBDAdapter`` take no arguments at all, so they cannot report a
    # profile they are not given.  This is the only place that holds the
    # solver name and the params together for EVERY adapter.
    #
    # ``logger`` is optional so the factory stays callable from a test or a
    # tool without one; the engine passes its node logger.
    warn_inert_params(physics_solver, physics_params, logger)
    # ``mjxpbd`` is an alias for ``mjvbd`` at the adapter layer: both keep
    # MuJoCo-Warp as the rigid solver and let lifecycle build a separate
    # particle solver. Which particle solver (VBD vs XPBD vs Style3D) is
    # chosen by ``newton.solver.prefer`` in the scene yaml.
    # ``mujoco-warp-multi-env`` is a SEPARATE adapter that carries the
    # multi-world / lockstep / grasp-fidelity work. It is deliberately its own
    # file rather than a modification of ``mujoco_warp.py``: users of the
    # stock adapter must not inherit any of it. Only this branch passes the
    # extra constructor arguments those features need — the stock adapter's
    # signature and its njmax/nconmax defaults are untouched.
    if key in ("mujoco-warp-multi-env", "mujoco_warp_multi_env"):
        from genie_sim_engine_newton.engine.newton.adapters.mujoco_warp_multi_env import (
            MuJoCoWarpMultiEnvAdapter,
        )

        return MuJoCoWarpMultiEnvAdapter(
            pd_ke=mujoco_pd_ke,
            pd_kd=mujoco_pd_kd,
            njmax=mujoco_njmax,
            nconmax=mujoco_nconmax,
            save_to_mjcf=mujoco_save_to_mjcf,
            physics_params=physics_params,
            joint_params=joint_params,
            physics_hz=physics_hz,
            imu_cfg=imu_cfg,
            pin_base_joint=pin_base_joint,
            enable_multiccd=mujoco_enable_multiccd,
        )
    if key in ("mujoco-warp", "mujoco_warp", "mjvbd", "mjxpbd"):
        from genie_sim_engine_newton.engine.newton.adapters.mujoco_warp import MuJoCoWarpAdapter

        # Forward the capacity knobs ONLY when the caller set them. The stock
        # adapter's defaults live in its own signature (1024/512), so passing
        # the unset 0 through would overwrite a working capacity with none —
        # ``**{}`` keeps the default binding intact instead.
        _cap = {}
        if mujoco_njmax > 0:
            _cap["njmax"] = mujoco_njmax
        if mujoco_nconmax > 0:
            _cap["nconmax"] = mujoco_nconmax
        return MuJoCoWarpAdapter(
            pd_ke=mujoco_pd_ke,
            pd_kd=mujoco_pd_kd,
            **_cap,
            save_to_mjcf=mujoco_save_to_mjcf,
            physics_params=physics_params,
            joint_params=joint_params,
            physics_hz=physics_hz,
            imu_cfg=imu_cfg,
            enable_multiccd=mujoco_enable_multiccd,
        )
    if joint_params:
        raise ValueError("Selected joint overlays require a MuJoCo-Warp rigid solver on Newton")
    if key in ("fsvbd", "featherstone"):
        from genie_sim_engine_newton.engine.newton.adapters.featherstone import FeatherstoneAdapter

        return FeatherstoneAdapter()
    if key == "avbd":
        from genie_sim_engine_newton.engine.newton.adapters.avbd import AVBDAdapter

        return AVBDAdapter()
    raise ValueError(
        f"unknown physics_solver: {physics_solver!r}; "
        f"expected one of: mujoco-warp (pure), mujoco-warp-multi-env (pure), "
        f"mjvbd | mjxpbd (combo), "
        f"fsvbd | featherstone (combo), avbd (pure)"
    )


__all__ = ["SolverAdapter", "make_adapter"]
