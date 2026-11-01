# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Amount of fluid that ended up inside a container, counted from the particles.

Written for the 50 ml pour task, where the pass condition is a particle count,
not a pose: fill a graduated cylinder to 10080-12320 particles (45-55 ml at
224 particles/ml) and stop pouring.

Format: ``"<particle_obj>|<receiver_obj>|<source_obj>|<min>|<max>|<expected_total>"
         "|<settle_checks>|<settle_tol>|<max_tilt_deg>"``

Why particles rather than a filled-volume proxy:

* The particle set is PBD, and PhysX writes ``positions`` on the PointsInstancer
  back to the USD layer every physics step -- verified by playing a delivered
  scene and watching the mean height fall from 0.744127 to 0.743441 over 480
  steps while the count stayed at 22500. ``get_prim_attribute`` is therefore a
  valid read here, unlike for rigid-body transforms under GPU dynamics.
* ``EVALUATION.md`` requires containment to be judged against the receiver's
  real collision walls (inner face, inner floor, rim height), explicitly *not*
  by radius from the container's centre. That is what ``_receiver_cavity``
  derives: it measures the authored ``Wall_*`` / ``Bottom`` colliders once, in
  the receiver's local frame, so swapping the container asset re-derives it.

A pass needs all of the following on the *same* update -- nothing is stitched
together across updates, because "pour 40 ml, wait, top up 10 ml" is not the
task:

1. count inside the receiver's cavity is within ``[min, max]``;
2. the last ``settle_checks`` counts moved by at most ``settle_tol`` particles,
   i.e. pouring has stopped -- the delivered contract asks for "flow has
   stopped", explicitly *not* a flat surface, so no surface-variance term;
3. the system still holds ``expected_total`` particles (a pass cannot come from
   deleting or adding fluid);
4. the receiver is tilted less than ``max_tilt_deg`` from upright;
5. every particle is inside the source's envelope, inside the receiver's
   cavity, or not yet below ``NEAR_TABLE_Z`` -- so a puddle on the table fails
   rather than being ignored. ``NEAR_TABLE_Z`` is the fixed table height of
   this task's 40 scenes, not a general threshold.

What this deliberately does *not* check, and why (see EVALUATION.md section 4):
contact/spacing against the receiver needs per-physics-step sampling, but
checkers are driven every 30 env steps, so an instantaneous touch between those
samples is invisible; recording completeness, fluid material auditing and the
standard export/QC gate live in the data-collection pipeline and are not
observable from the benchmark at all. A passing benchmark episode is therefore a
subset of the delivered contract's "valid recording", never equivalent to it.
"""

import collections
import json
import os

import numpy as np

from geniesim_benchmark.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
from geniesim_benchmark.plugins.logger import Logger

logger = Logger()

# World-Z below which a stray particle counts as spilled onto the table rather
# than as still in flight. Measured for this task's fixed table height; a scene
# with a different table breaks it (EVALUATION.md documents the same constant
# and the same caveat).
NEAR_TABLE_Z = 0.736

# Source envelope from the delivered ASSETS.json bounds, widened per
# EVALUATION.md's far-scatter definition: local radius 75 mm, local height
# -15..130 mm. Used only to decide whether a particle is still "in the jug".
SOURCE_ENVELOPE_RADIUS = 0.075
SOURCE_ENVELOPE_Z = (-0.015, 0.130)

WALL_PREFIX = "Wall_"
BOTTOM_NAME = "Bottom"


def _to_world(points, matrix):
    """Row-vector affine transform: USD composes world = local * M."""
    p = np.asarray(points, dtype=np.float64)
    return (np.c_[p, np.ones(len(p))] @ matrix)[:, :3]


def _to_local(points, matrix_inv):
    """Inverse of ``_to_world``."""
    p = np.asarray(points, dtype=np.float64)
    return (np.c_[p, np.ones(len(p))] @ matrix_inv)[:, :3]


class PourVolume(EvaluateAction):
    def __init__(
        self,
        env,
        particle_obj,
        receiver_obj,
        source_obj,
        min_particles=10080,
        max_particles=12320,
        expected_total=22500,
        settle_checks=6,
        settle_tol=10,
        max_tilt_deg=5.0,
    ):
        super().__init__(env)
        self._holder_name, self._particle_obj = self.placeholder_sparser(particle_obj)
        self.receiver_obj = receiver_obj
        self.source_obj = source_obj
        self.min_particles = int(min_particles)
        self.max_particles = int(max_particles)
        self.expected_total = int(expected_total)
        self.settle_checks = max(1, int(settle_checks))
        self.settle_tol = float(settle_tol)
        self.max_tilt_deg = float(max_tilt_deg)

        self._done_flag = False
        self._history = collections.deque(maxlen=self.settle_checks)
        self._cavity = None  # (inner_radius, floor_z, mouth_z), receiver-local
        self._cavity_error = None
        self._count = 0
        self._inside_jug = 0
        self._spilled = 0
        self._tilt_deg = 0.0
        self._total = 0
        self._best = 0.0
        self._updates = 0
        self._trace = os.environ.get("GENIESIM_POUR_TRACE", "")

    @property
    def particle_obj(self) -> str:
        if self._holder_name:
            return getattr(self, self._particle_obj)
        return self._particle_obj

    def _prim_path(self, obj_name: str) -> str:
        """Workspace-scoped prim path for ``obj_name``.

        Mirrors ``_analyze_obj_name`` but returns the path only, so it can be
        handed to the stage directly rather than to ``get_prim_attribute``
        (which applies the env prefix itself).
        """
        path = self._analyze_obj_name(obj_name)
        return self.api_core._env_path(path)

    def _world_matrix(self, prim_path: str):
        from pxr import Usd, UsdGeom

        prim = self.api_core._stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        return np.array(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )

    def _receiver_cavity(self):
        """Inner radius, floor height and rim height of the receiver, in its own
        local frame.

        Read from the authored colliders rather than from a nominal size: the
        delivered graduated cylinder is 80 flat ``Wall_*`` cubes plus a
        ``Bottom`` plate, and judging containment against its bounding box would
        accept particles in the corners outside the glass.

        Computed once and cached, in local coordinates, so a receiver that gets
        knocked over is still measured correctly -- every update re-derives the
        receiving frame from the receiver's current world transform.
        """
        if self._cavity is not None or self._cavity_error is not None:
            return self._cavity

        from pxr import Usd, UsdGeom

        root = self._prim_path(self.receiver_obj)
        stage = self.api_core._stage
        colliders = stage.GetPrimAtPath(root + "/entity/Colliders")
        if not colliders or not colliders.IsValid():
            colliders = stage.GetPrimAtPath(root)
        if not colliders or not colliders.IsValid():
            self._cavity_error = f"receiver colliders not found under {root}"
            return None

        M = self._world_matrix(root)
        if M is None:
            self._cavity_error = f"receiver prim not found: {root}"
            return None
        M_inv = np.linalg.inv(M)

        bbox = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            ["default", "render", "guide", "proxy"],
            useExtentsHint=False,
        )

        def local_corners(prim):
            r = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            if r.IsEmpty():
                return None
            lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
            pts = np.array(
                [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            )
            return _to_local(pts, M_inv)

        walls = [
            p
            for p in colliders.GetChildren()
            if p.GetName().startswith(WALL_PREFIX)
            and any("Collision" in s for s in p.GetPrimTypeInfo().GetAppliedAPISchemas())
        ]
        if not walls:
            self._cavity_error = f"no {WALL_PREFIX}* colliders under {colliders.GetPath()}"
            return None

        radii, tops, floors = [], [], []
        for w in walls:
            c = local_corners(w)
            if c is None:
                continue
            radii.append(float(np.hypot(c[:, 0], c[:, 1]).min()))
            tops.append(float(c[:, 2].max()))
            floors.append(float(c[:, 2].min()))
        if not radii:
            self._cavity_error = "receiver wall colliders have no extent"
            return None

        inner_radius = min(radii)
        mouth_z = max(tops)

        # The floor plate sits beside the collider group on some assets and
        # inside it on others (the graduated cylinder authors Bottom as a
        # sibling of Colliders/), so look in both places before falling back to
        # the lowest wall corner.
        bottom = None
        entity = stage.GetPrimAtPath(root + "/entity")
        for parent in (colliders, entity):
            if bottom is not None or not parent or not parent.IsValid():
                continue
            for name in (BOTTOM_NAME, "Floor", "Base"):
                p = parent.GetChild(name)
                if p and p.IsValid():
                    bottom = local_corners(p)
                    if bottom is not None:
                        break
        floor_z = float(bottom[:, 2].max()) if bottom is not None else min(floors)

        self._cavity = (inner_radius, floor_z, mouth_z)
        logger.info(
            f"[PourVolume] receiver cavity from {len(walls)} wall colliders: "
            f"inner_r={inner_radius:.5f} m  floor_z={floor_z:.5f}  mouth_z={mouth_z:.5f}"
        )
        return self._cavity

    def _inside_count(self, pts_world, M_inv, cavity, z_offset):
        inner_radius, floor_z, mouth_z = cavity
        local = _to_local(pts_world, M_inv)
        r = np.hypot(local[:, 0], local[:, 1])
        z = local[:, 2] + z_offset
        mask = (r <= inner_radius) & (z >= floor_z) & (z <= mouth_z)
        return int(mask.sum())

    def update(self, delta_time: float) -> float:
        if self._done_flag:
            return super().update(delta_time)

        self._updates += 1
        cavity = self._receiver_cavity()
        if cavity is None:
            if self._updates % 10 == 1:
                logger.warning(f"[PourVolume] receiver cavity unavailable: {self._cavity_error}")
            return super().update(delta_time)

        from pxr import Usd, UsdGeom

        # `particle_obj` names the particle prim itself (``liquid/entity/Particles``),
        # not the fluid's root Xform -- that is where the PointInstancer hangs and
        # where `positions` is authored.
        particle_prim = self._prim_path(self.particle_obj)
        pts = self.api_core.get_prim_attribute(particle_prim, "positions", default=None)
        if pts is None:
            if self._updates % 10 == 1:
                logger.warning(
                    f"[PourVolume] no 'positions' on {particle_prim}; fluid may not be simulating"
                )
            return super().update(delta_time)

        pts = np.asarray(pts, dtype=np.float64)
        self._total = int(len(pts))
        if self._total == 0:
            return super().update(delta_time)

        M_f = self._world_matrix(particle_prim)
        M_r = self._world_matrix(self._prim_path(self.receiver_obj))
        M_s = self._world_matrix(self._prim_path(self.source_obj))
        if M_f is None or M_r is None:
            return super().update(delta_time)

        world = _to_world(pts, M_f)

        # Tilt: the receiver's local +Z against world +Z.
        up = M_r[:3, 2]
        norm = np.linalg.norm(up)
        self._tilt_deg = (
            float(np.degrees(np.arccos(np.clip(abs(up[2]) / norm, -1.0, 1.0))))
            if norm > 0
            else 180.0
        )

        M_r_inv = np.linalg.inv(M_r)
        loc = _to_local(world, M_r_inv)
        self._count = self._inside_count(world, M_r_inv, cavity, 0.0)

        # Endpoint spill rule: at the end every particle must be in the jug's
        # envelope or in the receiver's cavity, and nothing else. This is the
        # contract's *final* condition ("结束时允许液体留在量杯内或量筒内，
        # 但两者之外的粒子数必须为零"), which is the only spill test a checker
        # sampling every 30 env steps can evaluate -- the near-table z < 0.736
        # rule in the contract is a during-pour test that needs per-step samples.
        # A particle stopped above the rim is therefore also a fail, which also
        # covers the contract's "nothing left waiting to fall into the cylinder".
        src = _to_local(world, np.linalg.inv(M_s)) if M_s is not None else None
        if src is not None:
            in_jug = (
                (np.hypot(src[:, 0], src[:, 1]) <= SOURCE_ENVELOPE_RADIUS)
                & (src[:, 2] >= SOURCE_ENVELOPE_Z[0])
                & (src[:, 2] <= SOURCE_ENVELOPE_Z[1])
            )
        else:
            in_jug = np.zeros(len(world), dtype=bool)
        in_receiver = (
            (np.hypot(loc[:, 0], loc[:, 1]) <= cavity[0])
            & (loc[:, 2] >= cavity[1])
            & (loc[:, 2] <= cavity[2])
        )
        stray = ~(in_jug | in_receiver)
        self._inside_jug = int(in_jug.sum())
        self._spilled = int(stray.sum())

        self._history.append(self._count)

        in_range = self.min_particles <= self._count <= self.max_particles
        settled = (
            len(self._history) == self.settle_checks
            and (max(self._history) - min(self._history)) <= self.settle_tol
        )
        total_ok = self._total == self.expected_total
        upright = self._tilt_deg <= self.max_tilt_deg
        no_spill = self._spilled == 0

        self._done_flag = bool(in_range and settled and total_ok and upright and no_spill)
        self._best = max(self._best, min(self._count / max(1, self.max_particles), 1.0))

        if self._trace:
            logger.info(
                "[PourTrace] "
                + json.dumps(
                    {
                        "update_index": self._updates,
                        "env_step": getattr(self.env, "current_step", None),
                        "receiver": self.receiver_obj,
                        "cavity_local": list(cavity),
                        "count_in_receiver": self._count,
                        "count_in_source": self._inside_jug,
                        "spilled": self._spilled,
                        "total": self._total,
                        "tilt_deg": round(self._tilt_deg, 3),
                        "history": list(self._history),
                        "in_range": in_range,
                        "settled": settled,
                        "total_ok": total_ok,
                        "upright": upright,
                        "no_spill": no_spill,
                        "done": self._done_flag,
                    },
                    sort_keys=True,
                )
            )

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"
        self.progress_info["DETAILS"] = {
            "particles_in_receiver": self._count,
            "particles_in_source": self._inside_jug,
            "particles_spilled": self._spilled,
            "particles_total": self._total,
            "target_min": self.min_particles,
            "target_max": self.max_particles,
            "settle_window": list(self._history),
            "settle_tol": self.settle_tol,
            "receiver_tilt_deg": round(self._tilt_deg, 3),
            "receiver_cavity_local": list(self._cavity) if self._cavity else None,
            "cavity_error": self._cavity_error,
        }

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [PourVolume] receiver: {self.receiver_obj}, evt: {event.value}")

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            self.progress_info["SCORE"] = 0
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1 if self._done_flag else round(self._best, 4)
