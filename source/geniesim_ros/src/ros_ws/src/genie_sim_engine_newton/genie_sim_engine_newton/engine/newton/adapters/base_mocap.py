# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Robot fixed-root reset shared by single-world and replicated MuJoCo solvers."""

from __future__ import annotations
from typing import Any


def select_base_mocap_index(named_mocaps):
    """Pick the robot BASE mocap body from ``[(mocap_id, body_name), ...]``.

    On a ``pin_base_to_world: false`` build Newton's MJC converter emits
    every WORLD-FIXED root (parent=-1, FIXED joint) as a NATIVE MuJoCo
    mocap body (see ``SolverMuJoCo._convert_to_mjc``: ``mocap=is_fixed_root``).
    The robot's own ``base_link`` becomes such a mocap body, so
    ``reset.py::_world_free_joints`` finds NO root free joint. Free scene
    props keep their FREE joints (NOT mocap); demoted kinematic tables are
    STATIC colliders (NOT mocap) — so in practice the ONLY mocap body is the
    base, but we still filter defensively.

    The base is the (first) mocap whose body name does NOT contain
    ``"Objects"`` — the same "only non-object root" heuristic
    ``_world_free_joints`` / ``_build_base_pin`` use on the Newton side.

    Returns ``(base_idx, others)``: ``base_idx`` is the base's mocap id (or
    ``None`` when every mocap is a scene Object / none exist), ``others`` is
    the list of ``(mocap_id, name)`` NOT chosen (for diagnostics).
    """
    base_idx = None
    others = []
    for mid, name in named_mocaps:
        if "Objects" in (name or ""):
            others.append((mid, name))
            continue
        if base_idx is None:
            base_idx = mid
        else:
            others.append((mid, name))
    return base_idx, others


class BaseMocapResetMixin:
    def _snapshot_base_mocap(self, logger: Any) -> None:
        """Identify the robot BASE mocap body and snapshot its build-time
        (init) pose so reset-time per-world base-xy randomization can be
        applied RELATIVE to it (restore-then-offset; see
        ``apply_base_mocap_offsets``).

        Background — why the base is a mocap body here
        ----------------------------------------------
        On a ``pin_base_to_world: false`` tabletop build the URDF world-weld
        is deactivated, but Newton's MJC converter still emits the robot's
        world-fixed ``base_link`` root as a NATIVE MuJoCo mocap body (no root
        FREE joint — ``reset.py::_world_free_joints`` correctly finds none).
        MuJoCo holds mocap bodies at ``mjw_data.mocap_pos`` EVERY step — that
        IS the position hold (no free-run drift once we move it). Under this
        engine's ``update_data_interval=0`` neither ``solver.step`` nor
        ``sync_state_into_solver`` (``_update_mjc_data`` writes only
        qpos/qvel) touches ``mocap_pos``, and the FROM-Newton refresh
        (``update_mocap_transforms_kernel``) fires only on
        ``notify_model_changed(JOINT_PROPERTIES)`` — never per step — so a
        direct write persists.

        The Newton-side base ``body_q`` (what the render/publish path reads)
        is re-derived EVERY step by ``_update_newton_state``'s
        ``eval_articulation_fk`` from ``model.joint_X_p`` of the base's fixed
        joint (``body_q = joint_X_p * inv(joint_X_c)``; a parent=-1 fixed
        root has identity joint motion). So to make ``body_poses`` reflect
        the offset in STEADY state we must move ``joint_X_p`` too — see
        ``apply_base_mocap_offsets``. We snapshot both buffers' init values
        here (``model.joint_X_p`` is a template Newton never mutates by
        stepping, so it holds the init base pose until WE change it).
        """
        self._base_mocap_idx = None
        self._base_mocap_jnt = None
        self._base_mocap_init_pos = None
        self._base_mocap_init_xp = None
        solver = self._solver
        mjm = getattr(solver, "mj_model", None)
        dw = getattr(solver, "mjw_data", None)
        model = getattr(solver, "model", None)
        if mjm is None or dw is None or getattr(dw, "mocap_pos", None) is None:
            return
        try:
            import mujoco  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            nmocap = int(getattr(mjm, "nmocap", 0) or 0)
            if nmocap <= 0:
                logger.info(
                    "[mjwarp-adapter] base mocap: model has no mocap bodies "
                    "(weld/free base build) — reset-time base-xy via mocap "
                    "inactive"
                )
                return
            named = []
            for b in range(int(mjm.nbody)):
                mid = int(mjm.body_mocapid[b])
                if mid < 0:
                    continue
                nm = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                named.append((mid, nm))
            base_idx, others = select_base_mocap_index(named)
            if base_idx is None:
                logger.warn(
                    f"[mjwarp-adapter] base mocap: {len(named)} mocap body/bodies "
                    f"but all look like scene Objects "
                    f"({[n for _, n in named][:6]}) — no base mocap picked"
                )
                return
            self._base_mocap_idx = int(base_idx)
            self._base_mocap_init_pos = np.asarray(dw.mocap_pos.numpy()).copy()
            base_name = next((n for mid, n in named if mid == base_idx), "?")
            # Per-world Newton joint index of the base fixed joint (for the
            # joint_X_p publish-sync). ``mjc_mocap_to_newton_jnt[w, mocap]``
            # is the GLOBAL (world-major) Newton joint index; row w is world
            # w's copy. Absent on older Newton → mocap_pos still moves the
            # physics, but published base body_q may lag one build-pose.
            mmap = getattr(solver, "mjc_mocap_to_newton_jnt", None)
            if mmap is not None and getattr(model, "joint_X_p", None) is not None:
                self._base_mocap_jnt = np.asarray(mmap.numpy())[:, int(base_idx)].copy()
                self._base_mocap_init_xp = np.asarray(model.joint_X_p.numpy()).copy()
                # Coupled ModelView owns compact copies. FK/readback uses the
                # parent model, so move that root too using exact joint labels.
                parent = getattr(model, "parent", None)
                self._base_mocap_parent_model = parent
                if parent is not None:
                    labels = list(parent.joint_label)
                    self._base_mocap_parent_jnt = np.array(
                        [labels.index(model.joint_label[int(j)]) if int(j) >= 0 else -1 for j in self._base_mocap_jnt]
                    )
                    self._base_mocap_parent_init_xp = parent.joint_X_p.numpy().copy()

            logger.info(
                f"[mjwarp-adapter] base mocap: id={base_idx} name={base_name!r}, "
                f"init pos={np.asarray(self._base_mocap_init_pos)[0, int(base_idx)].tolist()}; "
                + (
                    "joint_X_p publish-sync ON"
                    if self._base_mocap_jnt is not None
                    else "joint_X_p map UNAVAILABLE (published base body_q may not reflect offset)"
                )
                + (f"; other mocap(s): {[n for _, n in others][:4]}" if others else "")
            )
        except Exception as exc:  # noqa: BLE001
            self._base_mocap_idx = None
            logger.warn(f"[mjwarp-adapter] base mocap snapshot failed: {exc!r}")

    def has_base_mocap(self) -> bool:
        """True when the build exposes a robot BASE mocap body (the
        ``pin_base_to_world: false`` native path). ``reset.py`` routes
        ``init_base_xy_thresh`` onto ``apply_base_mocap_offsets`` when this
        is True and no root FREE joint exists."""
        return getattr(self, "_base_mocap_idx", None) is not None

    def apply_base_mocap_offsets(self, offsets, logger: Any = None) -> int:
        """Apply per-world base-xy offsets to the mocap-body robot base.

        ``offsets``: ``{world: (dx, dy)}`` — each RELATIVE to the build-time
        init base pose (restore-then-offset; NOT cumulative across resets),
        mirroring how ``reset.py`` restores ``joint_q`` to init before
        randomizing. A ``(0.0, 0.0)`` entry restores that world's base to
        init (so a prior reset's offset never lingers).

        Writes two buffers, both IN PLACE (``.assign`` — the captured CUDA
        graph's pointers stay valid; the caller runs this between frames,
        no graph in flight):

          * ``mjw_data.mocap_pos[w, base]`` — MuJoCo's per-step hold point;
            this is what keeps the base at the offset under free-run.
          * ``model.joint_X_p[base_jnt_of_world_w]`` — the fixed-root parent
            transform BOTH ``reset.py``'s ``eval_fk`` AND the per-step
            ``_update_newton_state`` (``eval_articulation_fk``) derive the
            base ``body_q`` from. Adding ``(dx,dy,0)`` to its translation
            shifts the published ``body_q`` (and its FK-children) by exactly
            ``(dx,dy,0)`` for a parent=-1 fixed root, in steady state — not
            just the frame right after reset.

        Returns the number of worlds updated; ``0`` (no-op) when there is no
        base mocap or the offsets map is empty.
        """
        if not offsets or getattr(self, "_base_mocap_idx", None) is None:
            return 0
        solver = getattr(self, "_solver", None)
        dw = getattr(solver, "mjw_data", None) if solver is not None else None
        if dw is None or getattr(dw, "mocap_pos", None) is None:
            return 0
        import numpy as np  # noqa: PLC0415

        idx = int(self._base_mocap_idx)
        init_pos = self._base_mocap_init_pos
        # 1. mocap_pos — physics hold. Read CURRENT (preserves worlds NOT in
        # this batch), overwrite only the requested worlds FROM init+offset.
        mp = np.asarray(dw.mocap_pos.numpy())
        nworld = mp.shape[0]
        applied = []
        for w, off in offsets.items():
            wi = int(w)
            if 0 <= wi < nworld and init_pos is not None:
                mp[wi, idx, 0] = float(init_pos[wi, idx, 0]) + float(off[0])
                mp[wi, idx, 1] = float(init_pos[wi, idx, 1]) + float(off[1])
                mp[wi, idx, 2] = float(init_pos[wi, idx, 2])
                applied.append(wi)
        dw.mocap_pos.assign(mp)
        # 2. joint_X_p — published body_q via FK (steady state).
        jnt = getattr(self, "_base_mocap_jnt", None)
        init_xp = self._base_mocap_init_xp
        model = getattr(solver, "model", None)
        if jnt is not None and init_xp is not None and getattr(model, "joint_X_p", None) is not None:
            xp = np.asarray(model.joint_X_p.numpy())
            for w, off in offsets.items():
                wi = int(w)
                nj = int(jnt[wi]) if 0 <= wi < len(jnt) else -1
                if 0 <= nj < xp.shape[0]:
                    xp[nj, 0] = float(init_xp[nj, 0]) + float(off[0])
                    xp[nj, 1] = float(init_xp[nj, 1]) + float(off[1])
            model.joint_X_p.assign(xp)
            parent = getattr(self, "_base_mocap_parent_model", None)
            if parent is not None:
                parent_xp = parent.joint_X_p.numpy()
                for w, off in offsets.items():
                    joint = int(self._base_mocap_parent_jnt[int(w)])
                    if joint >= 0:
                        parent_xp[joint, :2] = self._base_mocap_parent_init_xp[joint, :2] + off
                parent.joint_X_p.assign(parent_xp)

        if logger is not None and applied:
            logger.info(
                f"[mjwarp-adapter] base mocap: xy offset applied to world(s) "
                f"{sorted(applied)} (mocap idx={idx}); "
                + (
                    "+joint_X_p publish-sync"
                    if (jnt is not None and init_xp is not None)
                    else "mocap_pos only (no joint_X_p map)"
                )
            )
        return len(applied)

    def sync_state_into_solver(self, state: Any, logger: Any = None) -> None:
        """Commit a single-world out-of-band state write before readback or stepping."""
        import mujoco_warp

        solver = self._solver
        solver._update_mjc_data(solver.mjw_data, solver.model, state)
        mujoco_warp.forward(solver.mjw_model, solver.mjw_data)

    def sync_control_into_solver(self, state: Any, control: Any, logger: Any = None) -> None:
        if control is not None:
            solver = self._solver
            solver._apply_mjc_control(solver.model, state, control, solver.mjw_data)

    def clear_solver_carryover(self, env_ids=None, logger: Any = None) -> None:
        """The single-world solver must not retain a previous episode's warmstart."""
        if env_ids is not None and set(env_ids) != {0}:
            raise ValueError("single-world solver reset requires world 0")
        for name in ("qacc_warmstart", "act"):
            value = getattr(self._solver.mjw_data, name, None)
            if value is not None and value.size:
                value.zero_()
