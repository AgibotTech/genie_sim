# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build-phase 4: post-finalize model normalisations.

Mass clamp (Featherstone NaN guard), soft-contact material
constants, collision-group unification, and MJW articulation
unification.  Pure mutations on ``self._model``.
"""

from __future__ import annotations

import json
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warp as wp


class _NormalizeMixin:
    def _phase_normalize_model(self) -> None:
        """Phase 4: post-finalize model normalisations.

        Clamps zero-mass bodies (Featherstone NaN guard), sets soft-contact
        material constants, then collapses collision groups so cross-source
        rigid bodies collide, and on MJW unifies articulations so passive
        free bodies actually step.  Pure mutations on ``self._model``.
        """
        # Clamp zero-mass bodies to prevent Featherstone NaN.
        # USD robots sometimes have massless intermediate links (spine
        # segments, structural bridges) that PhysX handles via virtual mass
        # injection, but Newton's Featherstone inverts the spatial inertia
        # matrix directly — zero mass → singular → NaN on frame 1.
        _MIN_MASS = 1e-4  # 0.1 gram
        _MIN_INERTIA = 1e-8  # kg·m²
        if self._model.body_mass is not None:
            body_mass_np = self._model.body_mass.numpy().copy()
            zero_idx = (body_mass_np < _MIN_MASS).nonzero()[0]
            if len(zero_idx) > 0:
                self._logger.info(
                    f"[newton-standalone] clamping {len(zero_idx)} zero-mass body/bodies "
                    f"to {_MIN_MASS} kg: idx {zero_idx[:8].tolist()}"
                    f"{'...' if len(zero_idx) > 8 else ''}"
                )
                body_mass_np[zero_idx] = _MIN_MASS
                self._model.body_mass = wp.array(body_mass_np, dtype=wp.float32, device=self._model.device)
            if self._model.body_inertia is not None:
                body_inertia_np = self._model.body_inertia.numpy().copy()
                for i in zero_idx:
                    if np.max(np.abs(body_inertia_np[i])) < _MIN_INERTIA:
                        # Solid sphere: I = 2/5·m·r², r = 5 mm
                        r = 0.005
                        I_val = float((2.0 / 5.0) * _MIN_MASS * r * r)
                        body_inertia_np[i] = np.eye(3, dtype=np.float32) * I_val
                self._model.body_inertia = wp.array(body_inertia_np, dtype=wp.mat33, device=self._model.device)

        # Contact material (soft_contact_ke/kd/mu + soft_contact_margin) is set
        # in _phase_build_model (model.py) right after finalize and BEFORE the
        # plugin's on_model_ready hook, so scene plugins can override it; do NOT
        # re-assert it here or the plugin's FEM tuning would be clobbered
        # (noodles tunnel the wok). on_model_ready runs in phase 3, this is
        # phase 4 — re-setting here would always win over the plugin.

        # Per-shape friction is now driven by the per-class
        # ``newton.mjc_contact.<kind>.friction`` table inside
        # ``engine.newton.mjc_contact.apply_pre_finalize``, which writes
        # ``shape_material_mu`` (tangential), ``shape_material_mu_torsional``
        # and ``shape_material_mu_rolling`` separately.  The previous
        # scene-wide ``_ROBOT_KU`` broadcast that lived here was
        # silently overwriting that per-class write — and only set the
        # tangential slot, leaving torsional / rolling at their MuJoCo
        # defaults (0.005 / 0.0001), which is why fingers couldn't grip
        # round objects (the rod would just roll out of the jaw).
        # Removed entirely; tune via the yaml block instead.

        # ----------------------------------------------------------------
        # Collision-group unification.
        # ----------------------------------------------------------------
        # Newton's ModelBuilder assigns a fresh collision group per
        # ``add_usd`` / ``add_urdf`` call by default — pairs ACROSS groups
        # are filtered out of the rigid-vs-rigid contact pass.  In a
        # multi-source scene (robot URDF + scene USDA + sidecar USDAs from
        # scene-yaml ``newton.entries``) this means scene rigid bodies
        # like the hanger or fold-box NEVER collide with the robot, even
        # though both have CollisionAPI.
        #
        # Fix: collapse every shape into group 0 so the per-group filter
        # is a no-op.  Newton's contact pass still respects:
        #   * static-vs-static filtering (those pairs stay culled)
        #   * articulation self-collision flags
        #   * per-shape ``shape_collision_filter_pairs`` (untouched)
        # so we don't open a free-for-all — we only stop INTER-source
        # exclusion that the per-source group assignment created.
        #
        # Keeps cloth-vs-rigid unaffected (cloth uses a separate
        # broadphase that ignores collision groups regardless).
        if (
            getattr(self._model, "shape_collision_group", None) is not None
            and self._model.shape_collision_group.size > 0
        ):
            groups = self._model.shape_collision_group.numpy().copy()
            unique_before = sorted(set(int(g) for g in groups.tolist()))
            # groups[:] = 0
            self._model.shape_collision_group.assign(groups)
            self._logger.info(
                f"[lifecycle] collision-group unification: collapsed "
                f"{len(unique_before)} groups {unique_before!r} -> [0] "
                f"on {len(groups)} shapes (so cross-source bodies — "
                f"robot/scene/yaml-entries — can collide)"
            )

        # ----------------------------------------------------------------
        # Articulation unification (mjwarp ONLY).
        # ----------------------------------------------------------------
        # Newton's ``add_usd`` puts each free-floating ``RigidBodyAPI``
        # prim into its OWN articulation (separate from the URDF robot's
        # articulation), and MuJoCo-Warp's solver only steps the
        # articulations it was constructed with.  Result: passive scene
        # rigid bodies (hangers, dropped objects) with auto-created
        # FREE joints stay frozen at their spawn pose — gravity never
        # accelerates them, contacts never resolve.
        #
        # Fix on mjwarp: merge all articulations into one.
        #
        # newton 1.4.0 keeps BOTH representations live at once, and they are
        # not alternatives — measured on the pinned version, a 3-articulation
        # model finalizes to ``articulation_start == [0, 2, 4, 6]`` (CSR, N+1
        # entries, trailing ``joint_count`` sentinel) AND
        # ``articulation_end == [1, 3, 5]`` (split, N entries, excluding
        # converted loop-closing joints).  Different consumers read different
        # ones, unconditionally:
        #   * ``sim/articulation.py`` (``eval_articulation_fk``) reads
        #     ``articulation_end[id]``;
        #   * ``solvers/coupled/solver_coupled.py`` reads
        #     ``articulation_start[id + 1]`` — the CSR sentinel — while
        #     compacting its ModelView;
        #   * ``utils/selection.py`` reads both.
        # So the merge must leave both valid.  Writing a single-element
        # ``articulation_start`` (which an earlier revision here did, on the
        # theory that ``articulation_end``'s presence meant the CSR form was
        # gone) leaves FK working and puts every sentinel reader out of
        # bounds — invisible until a scene turns on ``newton.coupling``.
        #
        # ``joint_articulation`` is remapped to 0 rather than left alone.  FK
        # only needs owner >= 0, but any consumer that INDEXES the
        # articulation arrays by that owner id — the coupled solver again —
        # reads past the end of a now-1-entry array.  Loop-closure joints
        # keep their -1 and stay correctly skipped.
        #
        # IMPORTANT: this is a NO-OP on the Featherstone path.  Newton's
        # contact pipeline uses articulation membership for
        # SELF-COLLISION filtering (pairs within the same articulation
        # are excluded from rigid-vs-rigid contact).  If we unify on
        # Featherstone, the robot's arm shapes and the hanger's shapes
        # land in the same articulation and become self-collision-filtered;
        # any init-pose overlap then produces unbounded penetration
        # impulses and the integrator NaNs on substep 1.  Featherstone
        # already steps every articulation regardless of count, so it
        # doesn't need the merge in the first place.
        adapter_name = getattr(self._adapter, "name", "")
        if adapter_name.startswith(("mujoco-warp", "mujoco_warp")):
            try:
                artic_start = getattr(self._model, "articulation_start", None)
                artic_end = getattr(self._model, "articulation_end", None)
                joint_articulation = getattr(self._model, "joint_articulation", None)
                joint_count = int(getattr(self._model, "joint_count", 0) or 0)
                if artic_start is not None and joint_count > 0:
                    import warp as _wp  # noqa: PLC0415

                    start_arr = artic_start.numpy()
                    n_art = int(getattr(self._model, "articulation_count", 0) or max(0, len(start_arr) - 1))
                    end_arr = artic_end.numpy() if artic_end is not None else None
                    owner_arr = joint_articulation.numpy() if joint_articulation is not None else None
                    owner_ids = (
                        sorted({int(v) for v in owner_arr.tolist() if int(v) >= 0}) if owner_arr is not None else []
                    )
                    # Unified means every representation agrees, not just the
                    # one this code happens to write.
                    spans_all = (
                        n_art == 1
                        and len(start_arr) == 2
                        and int(start_arr[0]) == 0
                        and int(start_arr[1]) == joint_count
                        and (end_arr is None or (len(end_arr) == 1 and int(end_arr[0]) == joint_count))
                        and all(owner == 0 for owner in owner_ids)
                    )
                    if not spans_all:
                        dev = self._model.device
                        self._model.articulation_start = _wp.array(
                            np.array([0, joint_count], dtype=np.int32), dtype=artic_start.dtype, device=dev
                        )
                        if artic_end is not None:
                            self._model.articulation_end = _wp.array(
                                np.array([joint_count], dtype=np.int32), dtype=artic_end.dtype, device=dev
                            )
                        if joint_articulation is not None and owner_arr is not None:
                            unified_owners = owner_arr.copy()
                            unified_owners[unified_owners >= 0] = 0
                            joint_articulation.assign(unified_owners)
                        artic_labels = getattr(self._model, "articulation_label", None)
                        if isinstance(artic_labels, list):
                            self._model.articulation_label = [artic_labels[0] if artic_labels else "unified"]
                        if hasattr(self._model, "articulation_count"):
                            self._model.articulation_count = 1
                        if hasattr(self._model, "max_joints_per_articulation"):
                            self._model.max_joints_per_articulation = joint_count
                        if hasattr(self._model, "max_dofs_per_articulation"):
                            self._model.max_dofs_per_articulation = int(getattr(self._model, "joint_dof_count", 0) or 0)

                        # World bookkeeping is only rewritten for a SINGLE
                        # world.  Collapsing N articulations to one already
                        # spans every world on the multi-env adapter, and
                        # naming one of them as "the" world would assert
                        # something false about the other 63.  Left alone
                        # there, exactly as it is today — the multi-world
                        # articulation story wants its own measurement, not a
                        # side effect of this fix.
                        world_count = int(getattr(self._model, "world_count", 1) or 1)
                        artic_world = getattr(self._model, "articulation_world", None)
                        if artic_world is not None and world_count <= 1:
                            old_worlds = artic_world.numpy()
                            nonneg = [int(w) for w in old_worlds.tolist() if int(w) >= 0]
                            unified_world = nonneg[0] if nonneg else -1
                            self._model.articulation_world = _wp.array(
                                np.array([unified_world], dtype=np.int32), dtype=artic_world.dtype, device=dev
                            )
                            artic_world_start = getattr(self._model, "articulation_world_start", None)
                            if artic_world_start is not None:
                                starts = [0] * (world_count + 2)
                                if unified_world < 0:
                                    starts = [1] * (world_count + 2)
                                else:
                                    for w in range(world_count):
                                        starts[w + 1] = starts[w] + int(w == unified_world)
                                    starts[-1] = 1
                                self._model.articulation_world_start = _wp.array(
                                    np.array(starts, dtype=np.int32), dtype=artic_world_start.dtype, device=dev
                                )
                        self._logger.info(
                            f"[lifecycle] articulation unification (mjwarp): "
                            f"collapsed {n_art} articulations -> 1 spanning all "
                            f"{joint_count} joints (CSR start sentinel + split end + "
                            f"owner remap {owner_ids!r}->[0]; so passive FREE-joint "
                            f"scene bodies are FK'd + stepped by the solver)"
                        )
                    else:
                        self._logger.info(
                            f"[lifecycle] articulation unification (mjwarp): "
                            f"already 1 articulation spanning {joint_count} "
                            f"joints (no-op)"
                        )
            except Exception as exc:  # noqa: BLE001
                self._logger.warn(f"[lifecycle] articulation unification (mjwarp) failed: {exc!r}")
        else:
            self._logger.info(
                f"[lifecycle] articulation unification: skipped on "
                f"adapter={adapter_name!r} (Featherstone steps every "
                f"articulation; merging would collapse self-collision "
                f"filtering and NaN the integrator on init-pose overlap)"
            )
