# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""AVBD solver adapter — unified rigid + cloth via Newton's ``SolverVBD``.

Newton's ``SolverVBD`` class implements two formulations under one roof:

  * Vertex Block Descent (VBD) for cloth/particle DOFs.
  * Augmented VBD (AVBD) for rigid bodies.

When ``SolverVBD`` is constructed without ``integrate_with_external_rigid_solver``,
it integrates BOTH cloth particles AND rigid bodies in a single ``step`` call.
This adapter wires that mode, so passive scene rigid bodies (hangers,
dropped objects) AND the robot's articulated chain AND the cloth particles
all advance together inside one solver pass.

Why a separate adapter from Featherstone
----------------------------------------
The Featherstone+VBD path uses TWO solvers (``SolverFeatherstone`` for
rigid + a separate ``SolverVBD(integrate_with_external_rigid_solver=True)``
for cloth) composed in ``_substep_body_franka_vbd_cloth``.  That split
inherits Featherstone's quirks: mesh-vs-mesh contact requiring SDF data
that ``add_usd`` doesn't generate, FREE-joint twists clobbered by the
naive velocity-inject kernel, and articulation-membership filtering
breaking when passive bodies share an articulation with the robot.

AVBD-unified avoids the split entirely.  One solver, one contact pass,
one substep.  The lifecycle skips the separate cloth-solver build when
the rigid adapter is AVBD; the substep skips ``model.collide`` +
``cloth_solver.step`` and just calls ``self._adapter.substep``.

What this adapter shares with FeatherstoneAdapter
-------------------------------------------------
The masked-velocity-injection control law (``_kernel_velocity_inject_masked``
+ controlled-DOF mask built from ``joint_name_to_dof``) is solver-agnostic
— it writes ``joint_qd = target − q`` for controlled DOFs before
``solver.step``.  AVBD reads ``joint_qd`` the same way Featherstone does,
so the same control path drives the robot.  Passive DOFs (FREE joints,
JK_PASSIVE) are left alone; AVBD integrates them via gravity + contacts.

Selective PD on passive prefixes is also shared logic, just routed
through ``model.joint_target_*`` exactly as on the Featherstone path.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import newton
import numpy as np
import warp as wp

from genie_sim_engine_newton.engine.newton.adapters.base import SolverAdapter
from genie_sim_engine_newton.engine.newton.kernels import _kernel_velocity_inject_masked

# Default selective-PD gains — same defaults as Featherstone.  AVBD's
# rigid integrator handles gravity natively, so these are only used when
# a robot's body chain genuinely needs PD beyond velocity-injection
# (rare on G2 SP; empty prefix tuple by default disables the feature).
_DEFAULT_PASSIVE_KE = 2000.0
_DEFAULT_PASSIVE_KD = 500.0
_DEFAULT_PASSIVE_PREFIXES: tuple = ()


class AVBDAdapter(SolverAdapter):
    def __init__(
        self,
        *,
        passive_joint_prefixes: Iterable[str] = _DEFAULT_PASSIVE_PREFIXES,
        passive_pd_ke: float = _DEFAULT_PASSIVE_KE,
        passive_pd_kd: float = _DEFAULT_PASSIVE_KD,
        vbd_iterations: int = 5,
        self_contact_radius: float = 0.0025,
        self_contact_margin: float = 0.0025,
        soft_contact_ke: float = 1.0e4,
        rigid_body_particle_contact_buffer_size: int = 512,
        rigid_joint_linear_ke: float = 1.0e5,
        rigid_joint_angular_ke: float = 1.0e5,
        rigid_joint_snap_init: bool = False,
    ) -> None:
        self._passive_joint_prefixes = tuple(passive_joint_prefixes)
        self._passive_pd_ke = float(passive_pd_ke)
        self._passive_pd_kd = float(passive_pd_kd)
        self._vbd_iterations = int(vbd_iterations)
        self._self_contact_radius = float(self_contact_radius)
        self._self_contact_margin = float(self_contact_margin)
        self._soft_contact_ke = float(soft_contact_ke)
        self._rigid_body_particle_contact_buffer_size = int(rigid_body_particle_contact_buffer_size)
        self._rigid_joint_linear_ke = float(rigid_joint_linear_ke)
        self._rigid_joint_angular_ke = float(rigid_joint_angular_ke)
        # When True, set ``rigid_joint_*_k_start`` equal to ``rigid_joint_*_ke``
        # so the FixedJoint constraints are enforced at full stiffness from
        # frame 0.  Without this, AVBD's warmup ramp lets welded bodies sag
        # from their parsed (T-pose) USD transforms toward the constraint-
        # satisfying pose over several frames — visible as a "jumpy at
        # T-pose then settles" startup.
        #
        # Default OFF: snap-init is safe ONLY when EVERY rigid joint in the
        # model is FIXED (the all-welded ``[base, head, body, arm, gripper]``
        # case).  When any revolute joint is active, jamming k_start to ke
        # at frame 0 fights the velocity-injection control law — the joint-
        # position constraint pulls toward q=0 with full stiffness while
        # velocity-inject pushes ``qd = target − q`` toward the init pose,
        # and the model explodes.  Turn this on per-scene only when you've
        # confirmed all rigid joints are FIXED.
        self._rigid_joint_snap_init = bool(rigid_joint_snap_init)

        self._target_buffer: Optional[wp.array] = None
        self._controlled_mask: Optional[wp.array] = None
        # Contacts buffer — set by lifecycle via ``set_contacts`` after
        # ``self._model.contacts()`` has been allocated.  AVBD-unified
        # SolverVBD needs explicit contacts populated each substep
        # (the example_cloth_poker_cards.py pattern: collide then step).
        # ``None`` means "no contact pipeline configured" — solver.step
        # falls back to the older implicit-collide path.
        self._contacts: Any = None
        # Same deferral pattern as FeatherstoneAdapter — post_joint_map
        # may run before init_target_buffer, so we stash the map and
        # apply it once the mask array is allocated.
        self._pending_joint_name_to_dof: Optional[Dict[str, int]] = None
        self._solver: Any = None
        self._sim_substeps: int = 0

    def set_contacts(self, contacts: Any) -> None:
        """Inject the engine's ``self._contacts`` buffer into the adapter.

        Called once by lifecycle after ``self._contacts = self._model.contacts()``.
        The substep then runs ``model.collide(state_in, contacts)`` before
        ``solver.step`` and passes the populated buffer to the solver,
        matching the cloth+rigid pattern in
        ``newton/examples/cloth/example_cloth_poker_cards.py``.

        Featherstone and mjwarp adapters don't define this method — only
        AVBD's unified-rigid+cloth path needs it; lifecycle uses
        ``hasattr(adapter, "set_contacts")`` to decide whether to call.
        """
        self._contacts = contacts

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "avbd"

    @property
    def supports_cloth(self) -> bool:
        # AVBD's parent class (SolverVBD) handles cloth particles inside
        # the same solver instance.  Lifecycle MUST NOT build a separate
        # cloth solver when this adapter is active — see the AVBD branch
        # in lifecycle.py:_build (cloth-solver construction skipped when
        # ``self._adapter.name == "avbd"``).
        return True

    # ------------------------------------------------------------------
    # Pre-finalize
    # ------------------------------------------------------------------

    def register_custom_attributes(self, builder: Any) -> None:
        # SolverVBD authors the same custom attributes for cloth on the
        # builder regardless of whether rigid is integrated externally
        # or internally; the AVBD path doesn't add anything beyond what
        # the cloth side already needs.
        newton.solvers.SolverVBD.register_custom_attributes(builder)

    # ------------------------------------------------------------------
    # Post-finalize, pre-solver
    # ------------------------------------------------------------------

    def prepare_model(
        self,
        model: Any,
        logger: Any,
        mimic_followers: Optional[Dict[str, list]] = None,  # noqa: ARG002
        robot_prefix: str = "",  # noqa: ARG002 — mjwarp-only; accepted for ABC compatibility
    ) -> None:
        # AVBD reads ``joint_target_ke/kd`` natively but the velocity-
        # injection control law overrides that for controlled DOFs each
        # substep, so per-DOF PD gains here are unused by command-driven
        # joints.  Selective PD for genuinely passive joints (body/head
        # if configured) is applied later in post_joint_map, same as
        # the Featherstone path.
        return

    # ------------------------------------------------------------------
    # Solver construction
    # ------------------------------------------------------------------

    def build_solver(
        self,
        model: Any,
        sim_substeps: int,
        sim_iterations: int,
        logger: Any,
        mass_matrix_interval: int = 0,  # noqa: ARG002 — AVBD has no mass-matrix concept
    ) -> Any:
        """Construct ``SolverVBD`` in unified-rigid+cloth mode.

        ``integrate_with_external_rigid_solver=False`` is the key flag:
        with it disabled, ``SolverVBD`` integrates both particles AND
        rigid bodies via AVBD inside ``solver.step``.  No separate
        rigid solver is needed; no separate cloth solver is needed.

        ``mass_matrix_interval`` is accepted for ABC compatibility but
        ignored — VBD has no mass-matrix rebuild concept.
        """
        self._sim_substeps = int(sim_substeps)
        iters = int(sim_iterations) if int(sim_iterations) > 0 else self._vbd_iterations

        # Auto-detect statue mode: if every rigid joint is FIXED (or FREE
        # for the base), nothing actuated → safe to snap k_start to ke.
        # Mixing in any REVOLUTE/PRISMATIC means active DOFs are present
        # and snap-init would fight the velocity-injection control law,
        # so we fall back to the warmup ramp regardless of the user knob.
        snap = self._rigid_joint_snap_init
        try:
            from genie_sim_engine_newton.engine.newton.joint_index import JointIndex, JT_FIXED, JT_FREE  # noqa: PLC0415

            jindex = JointIndex(model)
            if len(jindex):
                unique = sorted({s.joint_type for s in jindex.slices()})
                logger.info(
                    f"[avbd-adapter] joint_type histogram: {unique} "
                    f"(FIXED={JT_FIXED}, FREE={JT_FREE}, total_joints={len(jindex)})"
                )
                # If any FREE joint exists in supposedly-statue mode, dump
                # which body Newton is auto-floating — that's the tumbler axis.
                if JT_FREE in unique:
                    try:
                        jc = model.joint_child.numpy()
                        jp = model.joint_parent.numpy()
                        for s in jindex.slices():
                            if s.joint_type != JT_FREE:
                                continue
                            i = s.joint_index
                            child_idx = int(jc[i])
                            parent_idx = int(jp[i])
                            child_label = (
                                model.body_label[child_idx]
                                if 0 <= child_idx < len(model.body_label)
                                else f"<body#{child_idx}>"
                            )
                            logger.warn(
                                f"[avbd-adapter] FREE joint #{i}: "
                                f"parent_body={parent_idx} child_body={child_idx} "
                                f"({child_label}) — this is Newton's auto-base-joint "
                                f"for an orphan body, the wobble axis of the welded chain"
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warn(f"[avbd-adapter] FREE-joint dump failed: {exc}")
                all_static = all(s.joint_type in (JT_FIXED, JT_FREE) for s in jindex.slices())
                # Did the bake-into-localRot0 actually land in joint_X_p?
                # Sample the FIXED joints whose label ends with "__fixed"
                # (our authored siblings) and dump their parent-xform quat —
                # identity means the parser dropped the bake; non-identity
                # means it stored it and the residual issue is elsewhere
                # (eval_fk membership, etc.).
                try:
                    Xp = model.joint_X_p.numpy()
                    sampled = 0
                    for s in jindex.slices():
                        if s.joint_type != JT_FIXED or "__fixed" not in s.label:
                            continue
                        if "arm_l_joint2" in s.label or "arm_r_joint2" in s.label or "body_joint3" in s.label:
                            tf = Xp[s.joint_index]
                            logger.info(
                                f"[avbd-adapter] joint_X_p[{s.label}]: "
                                f"pos={tf[:3].tolist()} quat(xyzw)={tf[3:].tolist()}"
                            )
                            sampled += 1
                            if sampled >= 4:
                                break
                except Exception as exc:  # noqa: BLE001
                    logger.warn(f"[avbd-adapter] joint_X_p dump failed: {exc}")
                if all_static and not snap:
                    snap = True
                    logger.info(
                        "[avbd-adapter] auto-enabling rigid_joint_snap_init: "
                        "every rigid joint is FIXED/FREE (statue mode)"
                    )
                if (not all_static) and snap:
                    snap = False
                    logger.warn(
                        "[avbd-adapter] auto-disabling rigid_joint_snap_init: "
                        "active REVOLUTE/PRISMATIC DOFs present — snap would "
                        "fight velocity-injection control and explode"
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warn(f"[avbd-adapter] snap_init auto-detect failed ({exc}); honouring knob={snap}")

        # FixedJoint snap-to-init: bypass AVBD's k_start warmup ramp by
        # initializing both linear and angular k_start at full stiffness.
        # See _rigid_joint_snap_init docstring above for why.
        joint_linear_k_start = self._rigid_joint_linear_ke if snap else 100.0
        joint_angular_k_start = self._rigid_joint_angular_ke if snap else 10.0

        self._solver = newton.solvers.SolverVBD(
            model,
            iterations=iters,
            integrate_with_external_rigid_solver=False,  # ← unified rigid+cloth
            particle_self_contact_radius=self._self_contact_radius,
            particle_self_contact_margin=self._self_contact_margin,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=self._self_contact_radius * 2.5,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=self._soft_contact_ke,
            rigid_body_particle_contact_buffer_size=self._rigid_body_particle_contact_buffer_size,
            rigid_joint_linear_ke=self._rigid_joint_linear_ke,
            rigid_joint_angular_ke=self._rigid_joint_angular_ke,
            rigid_joint_linear_k_start=joint_linear_k_start,
            rigid_joint_angular_k_start=joint_angular_k_start,
        )
        logger.info(
            f"[avbd-adapter] unified rigid+cloth solver: SolverVBD "
            f"(iterations={iters}, integrate_with_external_rigid_solver=False, "
            f"joint_snap_init={snap}); "
            f"rigid bodies advanced via AVBD inside solver.step — no "
            f"separate Featherstone or external cloth solver needed"
        )
        return self._solver

    # ------------------------------------------------------------------
    # Target buffer
    # ------------------------------------------------------------------

    def init_target_buffer(self, model: Any, control: Any, logger: Any) -> None:
        """Allocate the per-DOF target buffer and the controlled-DOF mask.

        Same pattern as FeatherstoneAdapter: a wp.array sized to
        ``joint_q``, seeded with the init pose; ``apply_commands``
        mutates it in-place.  Mask is allocated with all-zeros and
        populated either now (if post_joint_map already stashed the
        joint map) or later (when post_joint_map first runs).
        """
        if model.joint_q is None:
            logger.warn("[avbd-adapter] cannot init target buffer: model.joint_q is None")
            return
        self._target_buffer = wp.empty_like(model.joint_q)
        self._target_buffer.assign(model.joint_q)

        n_dofs = int(model.joint_dof_count)
        self._controlled_mask = wp.zeros(n_dofs, dtype=wp.int32, device=model.device)

        if self._pending_joint_name_to_dof is not None:
            self._apply_controlled_mask(self._pending_joint_name_to_dof, logger)
            self._pending_joint_name_to_dof = None
        else:
            logger.info(
                f"[avbd-adapter] target_buffer allocated ({n_dofs} DOFs); "
                f"no joint map yet — controlled_mask stays all-zeros until post_joint_map runs"
            )

    def target_buffer(self) -> Optional[wp.array]:
        return self._target_buffer

    # ------------------------------------------------------------------
    # Post-joint-map: build controlled-DOF mask + selective PD
    # ------------------------------------------------------------------

    def post_joint_map(
        self,
        model: Any,
        jindex: Any,
        control: Any,
        logger: Any,
    ) -> None:
        joint_name_to_dof = jindex.name_to_dof() if jindex is not None else {}

        if self._controlled_mask is None:
            self._pending_joint_name_to_dof = dict(joint_name_to_dof)
            logger.info(
                "[avbd-adapter] post_joint_map: target_buffer not yet allocated; "
                f"stashing {len(self._pending_joint_name_to_dof)} joint→DOF entries for "
                "init_target_buffer to apply"
            )
        else:
            self._apply_controlled_mask(joint_name_to_dof, logger)

        # Selective PD on passive prefixes — same as Featherstone path.
        if model.joint_target_ke is None or model.joint_target_kd is None or model.joint_target_mode is None:
            logger.warn("[avbd-adapter] joint_target_{mode,ke,kd} array is None; skipping selective PD setup")
            return
        if not self._passive_joint_prefixes:
            logger.info("[avbd-adapter] selective PD: disabled (no passive prefixes configured)")
            return
        mode = model.joint_target_mode.numpy().copy()
        ke = model.joint_target_ke.numpy().copy()
        kd = model.joint_target_kd.numpy().copy()
        n_passive = 0
        for name, idx in joint_name_to_dof.items():
            if any(p in name for p in self._passive_joint_prefixes):
                mode[idx] = 1  # POSITION
                ke[idx] = self._passive_pd_ke
                kd[idx] = self._passive_pd_kd
                n_passive += 1
        if n_passive == 0:
            logger.info(
                f"[avbd-adapter] selective PD: no passive joints found " f"(prefixes={self._passive_joint_prefixes!r})"
            )
            return
        model.joint_target_mode.assign(mode)
        model.joint_target_ke.assign(ke)
        model.joint_target_kd.assign(kd)
        logger.info(
            f"[avbd-adapter] selective PD: {n_passive} passive DOF(s) "
            f"matching {self._passive_joint_prefixes!r} -> "
            f"ke={self._passive_pd_ke:.0f}, kd={self._passive_pd_kd:.0f}"
        )

    def _apply_controlled_mask(self, joint_name_to_dof: Dict[str, int], logger: Any) -> None:
        if self._controlled_mask is None:
            logger.warn("[avbd-adapter] _apply_controlled_mask: mask not allocated; skipping")
            return
        n_dofs = int(self._controlled_mask.size)
        mask = np.zeros(n_dofs, dtype=np.int32)
        for _, idx in joint_name_to_dof.items():
            if 0 <= int(idx) < n_dofs:
                mask[int(idx)] = 1
        self._controlled_mask.assign(mask)
        n_controlled = int(mask.sum())
        n_passive_dofs = n_dofs - n_controlled
        logger.info(
            f"[avbd-adapter] controlled-DOF mask: "
            f"{n_controlled}/{n_dofs} DOFs marked controlled "
            f"({n_passive_dofs} uncontrolled — left to AVBD's integrator)"
        )

    # ------------------------------------------------------------------
    # Per-substep robot+cloth step (inside captured CUDA graph)
    # ------------------------------------------------------------------

    def substep(
        self,
        model: Any,
        state_in: Any,
        state_out: Any,
        control: Any,
        sim_dt: float,
    ) -> None:
        """Masked velocity-injection, then ``model.collide`` + unified ``SolverVBD.step``.

        Mirrors the cloth+rigid pattern in
        ``newton/examples/cloth/example_cloth_poker_cards.py``:

          1. (this codebase only) inject controlled-DOF velocity targets
             via the masked kernel.
          2. ``model.collide(state_in, contacts)`` — populates the
             contacts buffer for the upcoming step.  Without this AVBD
             sees zero contact pairs and dynamic bodies fall through
             every static collider in the scene.
          3. ``solver.step(state_in, state_out, control, contacts, dt)``
             with the contacts arg — AVBD's unified rigid+cloth pass
             reads them inside the iteration.

        Lifecycle calls ``set_contacts(self._contacts)`` after creating
        the contacts buffer; if it was never called (no-contacts harness),
        ``self._contacts`` stays ``None`` and we fall back to the
        no-explicit-contacts call which AVBD silently no-ops.
        """
        if (
            self._target_buffer is not None
            and self._controlled_mask is not None
            and state_in.joint_qd is not None
            and state_in.joint_q is not None
        ):
            wp.launch(
                _kernel_velocity_inject_masked,
                dim=model.joint_dof_count,
                inputs=[
                    state_in.joint_q,
                    self._target_buffer,
                    self._controlled_mask,
                ],
                outputs=[state_in.joint_qd],
            )
        if self._contacts is not None:
            model.collide(state_in, self._contacts)
        self._solver.step(state_in, state_out, control, self._contacts, sim_dt)
