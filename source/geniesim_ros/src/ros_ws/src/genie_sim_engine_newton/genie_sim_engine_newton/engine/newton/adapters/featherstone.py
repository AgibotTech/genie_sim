# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Featherstone solver adapter.

Owns:
  * ``SolverFeatherstone`` construction.
  * The ``_target_joint_pos`` velocity-injection buffer.
  * The per-substep masked velocity-injection kernel launch.
  * Selective PD on PASSIVE joints (body / head etc.) — joints that no
    ROS topic ever commands and that therefore have no way to resist
    gravity through the velocity-injection control law.

What this adapter does NOT own:
  * Cloth solvers (SolverVBD / SolverXPBD / SolverStyle3D) — solver-
    independent extensions composed by ``lifecycle.py`` / ``engine.py``.

Why a controlled-DOF mask
--------------------------
Featherstone ignores ``joint_target_mode`` and applies its built-in PD
unconditionally from ``joint_target_ke/kd``.  URDF imports for many
robots author both at 0, so PD is dead-on-arrival; we drive controlled
joints by injecting velocity directly (``joint_qd = target − q``) before
``solver.step`` integrates.  See the franka cloth-folding example for
the validated stable pattern.

The naive form of that kernel runs over EVERY DOF in the model — which
silently breaks the moment the scene contains a passive dynamic body
(hanger, dropped object, ragdoll).  Such bodies' DOFs are never written
to ``target_buffer``, so ``target = q`` always and the kernel writes
``qd = 0`` — overwriting whatever velocity Newton's solver just produced
from gravity / contacts / joint reactions.  Result: the body never moves
under physics.

Fix: build a ``controlled_mask`` once at ``post_joint_map`` time from
``joint_name_to_dof`` (every DOF the controller addresses by name is
"controlled"; everything else is passive-dynamic) and have the kernel
skip uncontrolled DOFs.  Newton's integrator then handles passive bodies
the standard textbook way — velocity-injection becomes a control signal
for a specific subset of joints, not a destructive global override.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import newton
import numpy as np
import warp as wp

from genie_sim_engine_newton.engine.newton.adapters.base import SolverAdapter
from genie_sim_engine_newton.engine.newton.kernels import _kernel_velocity_inject_masked

# Default PD gains applied to PASSIVE joints when ``passive_pd_ke/kd`` is
# not overridden at construction.  PD here is for kinematic-coupling
# stabilisation, not gravity, so gains are intentionally moderate.
_DEFAULT_PASSIVE_KE = 2000.0
_DEFAULT_PASSIVE_KD = 500.0

# Default joint-name substrings that mark a joint as PASSIVE.  Empty by
# default — selective PD off; G2 SP doesn't need it.
_DEFAULT_PASSIVE_PREFIXES: tuple = ()


class FeatherstoneAdapter(SolverAdapter):
    def __init__(
        self,
        *,
        passive_joint_prefixes: Iterable[str] = _DEFAULT_PASSIVE_PREFIXES,
        passive_pd_ke: float = _DEFAULT_PASSIVE_KE,
        passive_pd_kd: float = _DEFAULT_PASSIVE_KD,
    ) -> None:
        self._passive_joint_prefixes = tuple(passive_joint_prefixes)
        self._passive_pd_ke = float(passive_pd_ke)
        self._passive_pd_kd = float(passive_pd_kd)
        self._target_buffer: Optional[wp.array] = None
        self._controlled_mask: Optional[wp.array] = None
        # ``post_joint_map`` may run BEFORE ``init_target_buffer`` (see
        # lifecycle.py call order — selective-PD setup happens before
        # the target buffer is allocated).  Stash the joint map here on
        # the first call; ``init_target_buffer`` applies it once the
        # mask array exists.
        self._pending_joint_name_to_dof: Optional[Dict[str, int]] = None
        self._solver: Any = None
        self._sim_substeps: int = 0

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "featherstone"

    @property
    def supports_cloth(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Pre-finalize
    # ------------------------------------------------------------------

    def register_custom_attributes(self, builder: Any) -> None:
        newton.solvers.SolverFeatherstone.register_custom_attributes(builder)

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
        # Featherstone's built-in PD is bypassed in favor of velocity-
        # injection control for COMMAND-DRIVEN joints; selective PD for
        # PASSIVE joints happens in post_joint_map once the joint map
        # is built.  Nothing to do here.
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
        mass_matrix_interval: int = 0,
    ) -> Any:
        """Construct ``SolverFeatherstone`` with a tunable mass-matrix
        rebuild cadence.

        ``mass_matrix_interval``:
            0 (default) → fall back to ``sim_substeps``.  Matches franka
                          cloth demo: M(q) is rebuilt once per frame.
            N > 0       → use ``N`` as the rebuild interval.  Pass a
                          large value (e.g. 100_000) to effectively
                          disable auto rebuilds.  Safe for slow-motion
                          tasks where M(q) doesn't drift meaningfully
                          across many frames.
        """
        self._sim_substeps = int(sim_substeps)
        interval = int(mass_matrix_interval) if int(mass_matrix_interval) > 0 else self._sim_substeps
        self._solver = newton.solvers.SolverFeatherstone(model, update_mass_matrix_interval=interval)
        if interval > self._sim_substeps:
            note = (
                f" — overrides default sim_substeps={self._sim_substeps}; "
                f"M will be rebuilt every {interval} substeps "
                f"(~{interval / max(self._sim_substeps, 1):.1f} frames)"
            )
        else:
            note = " (= sim_substeps; one rebuild per frame, franka default)"
        logger.info(
            f"[featherstone-adapter] robot solver: SolverFeatherstone "
            f"(update_mass_matrix_interval={interval}){note}"
        )
        return self._solver

    # ------------------------------------------------------------------
    # Target buffer
    # ------------------------------------------------------------------

    def init_target_buffer(self, model: Any, control: Any, logger: Any) -> None:
        """Allocate the per-DOF target buffer and seed with the init pose.

        Featherstone's velocity-injection kernel reads from this buffer
        each substep to compute ``joint_qd = target − q`` for controlled
        DOFs.  Allocated once with ``wp.empty_like(model.joint_q)`` so
        the pointer the CUDA graph captures stays valid; ``apply_commands``
        mutates contents via ``.assign()``.

        The controlled-DOF mask is allocated here and populated either
        from a stash left by an earlier ``post_joint_map`` call (the
        normal lifecycle order in this engine) or left all-zeros if no
        joint map has arrived yet.  All-zeros means velocity-injection
        is a no-op everywhere — Newton runs the model under pure physics.
        """
        if model.joint_q is None:
            logger.warn("[featherstone-adapter] cannot init target buffer: model.joint_q is None")
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
                f"[featherstone-adapter] target_buffer allocated ({n_dofs} DOFs); "
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
        """Populate ``controlled_mask`` (now or deferred) and apply selective PD.

        ``jindex`` is the engine's ``JointIndex`` snapshot (see
        ``engine/newton/joint_index.py``).  We pull the canonical
        ``{name: dof_index}`` dict via ``jindex.name_to_dof()`` —
        every DOF that appears as a value in this map is "controlled":
        the controller will write a target and we want
        velocity-injection to drive it.  Every other DOF (FREE-joint
        twists for passive scene bodies, ragdoll DOFs, anything
        Newton parsed but the controller doesn't know about) is
        passive and Newton's integrator handles it via gravity +
        contacts + joint reactions.

        If this method runs BEFORE ``init_target_buffer`` (the order
        this engine's lifecycle uses today), we stash the map and apply
        it once the mask array exists.  If the mask is already up,
        apply immediately.

        Selective PD (passive prefixes) is applied here unconditionally
        — it touches ``model.joint_target_*`` which is independent of
        the mask buffer.
        """
        joint_name_to_dof = jindex.name_to_dof() if jindex is not None else {}

        if self._controlled_mask is None:
            self._pending_joint_name_to_dof = dict(joint_name_to_dof)
            logger.info(
                "[featherstone-adapter] post_joint_map: target_buffer not yet allocated; "
                f"stashing {len(self._pending_joint_name_to_dof)} joint→DOF entries for "
                "init_target_buffer to apply"
            )
        else:
            self._apply_controlled_mask(joint_name_to_dof, logger)

        # Selective PD on passive prefixes — independent of mask.
        if model.joint_target_ke is None or model.joint_target_kd is None or model.joint_target_mode is None:
            logger.warn("[featherstone-adapter] joint_target_{mode,ke,kd} array is None; skipping selective PD setup")
            return
        if not self._passive_joint_prefixes:
            logger.info("[featherstone-adapter] selective PD: disabled (no passive prefixes configured)")
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
                f"[featherstone-adapter] selective PD: no passive joints found "
                f"(prefixes={self._passive_joint_prefixes!r})"
            )
            return
        model.joint_target_mode.assign(mode)
        model.joint_target_ke.assign(ke)
        model.joint_target_kd.assign(kd)
        logger.info(
            f"[featherstone-adapter] selective PD: {n_passive} passive DOF(s) "
            f"matching {self._passive_joint_prefixes!r} -> "
            f"ke={self._passive_pd_ke:.0f}, kd={self._passive_pd_kd:.0f}"
        )

    def _apply_controlled_mask(self, joint_name_to_dof: Dict[str, int], logger: Any) -> None:
        """Write 1's into ``self._controlled_mask`` for every DOF named
        in ``joint_name_to_dof``.  Called either from ``post_joint_map``
        (mask already up) or ``init_target_buffer`` (using stashed map).
        """
        if self._controlled_mask is None:
            logger.warn("[featherstone-adapter] _apply_controlled_mask: mask not allocated; skipping")
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
            f"[featherstone-adapter] controlled-DOF mask: "
            f"{n_controlled}/{n_dofs} DOFs marked controlled "
            f"({n_passive_dofs} uncontrolled — left to Newton's integrator)"
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
        """Masked velocity-injection then ``solver.step``.

        For each controlled DOF: write ``joint_qd = target − q`` so
        Featherstone integrates kinematically toward the target.
        Uncontrolled DOFs are left alone — Newton's integrator computes
        their joint_qd from gravity + contacts + constraint impulses
        the standard way.
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
        self._solver.step(state_in, state_out, control, None, sim_dt)
