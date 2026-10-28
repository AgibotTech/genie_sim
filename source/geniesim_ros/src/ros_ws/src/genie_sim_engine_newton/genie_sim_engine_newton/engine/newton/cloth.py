# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton-standalone cloth-injection mixin.

Provides the ``_ClothMixin`` class composed into
``_NewtonStandaloneBase`` via multiple inheritance — see
``engine_base.py`` for the full mixin order.  ``self.X`` references
resolve through the engine's MRO.
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


class _ClothMixin:
    def _load_cloth_entries(self) -> list:
        if not self._newton_solvers_path:
            return []
        f = Path(self._newton_solvers_path)
        if not f.is_file():
            return []
        with f.open() as fp:
            extras = json.load(fp)
        live_newton = self._scene_cfg.get("newton") or {}
        live_params_map = {e["name"]: e.get("params") or {} for e in (live_newton.get("entries") or [])}
        out = []
        for entry in extras.get("entries", []):
            if entry.get("kind") != "cloth":
                continue
            name = entry["name"]
            out.append({**entry, "params": {**entry.get("params", {}), **live_params_map.get(name, {})}})
        return out

    def _load_solver_preference(self) -> str:
        """Return the explicit particle-solver name from newton_solvers.json.

        The newton_extras stage validates that ``newton.solver.prefer`` is set
        to one of ``{vbd, xpbd, style3d}`` whenever a cloth entry exists,
        so by the time we read it here it's guaranteed to be one of those
        three (or ``"none"`` if no cloth). We re-validate anyway so an
        edited or stale ``newton_solvers.json`` doesn't silently pass a
        bogus value into Newton's solver constructors.
        """
        VALID = {"vbd", "xpbd", "style3d", "avbd"}
        # Sources, in priority order:
        #   1. live ``newton.solver.prefer`` in the loaded scene_cfg —
        #      the source of truth a scene-plugin author edits when no
        #      cloth/USD entries exist (assemble_newton skips the
        #      sidecar in that case, so newton_solvers.json may not
        #      exist).
        #   2. ``newton_solvers.json`` written by assemble_newton
        #      (the cloth-path sidecar).
        live_newton = self._scene_cfg.get("newton") or {}
        live_solver = ((live_newton.get("solver") or {}).get("prefer") or "").strip().lower()
        if live_solver:
            if live_solver in VALID:
                return live_solver
            if live_solver == "none":
                return "none"
            raise RuntimeError(
                f"scene yaml newton.solver.prefer={live_solver!r} is not a "
                f"recognised particle solver. Valid: {sorted(VALID)}."
            )
        if not self._newton_solvers_path:
            return "none"
        f = Path(self._newton_solvers_path)
        if not f.is_file():
            return "none"
        try:
            with f.open() as fp:
                extras = json.load(fp)
        except Exception:
            return "none"
        raw = ((extras.get("solver") or {}).get("prefer") or "").strip().lower()
        if raw in VALID:
            return raw
        if raw in ("", "none"):
            return "none"
        raise RuntimeError(
            f"newton_solvers.json has solver.prefer={raw!r} which is not a "
            f"recognised particle solver. Re-run assemble_newton (or fix the "
            f"scene yaml) to set it to one of: {sorted(VALID)}."
        )

    def _inject_cloth(self, builder: Any, entry: dict) -> None:
        import warp as wp
        from pxr import Usd, UsdGeom

        name = entry["name"]
        prim_path = entry.get("usd_prim_path", f"/World/{name}")
        pose = entry.get("pose", [0, 0, 0, 0, 0, 0, 1])
        vel = entry.get("vel", [0, 0, 0])
        p = entry.get("params") or {}

        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            self._logger.warn(f"[newton-standalone] cloth prim not found: {prim_path}")
            return

        mesh_prim = None
        for q in Usd.PrimRange(prim):
            if q.IsA(UsdGeom.Mesh):
                mesh_prim = q
                break
        if mesh_prim is None:
            self._logger.warn(f"[newton-standalone] no Mesh under {prim_path}")
            return

        mesh = UsdGeom.Mesh(mesh_prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idxs = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idxs:
            return

        verts = np.array([[v[0], v[1], v[2]] for v in pts], dtype=np.float32)

        # Sanity log + meter-scale check. The cloth USD MUST be authored at
        # meter-scale (metersPerUnit=1.0, identity parent xforms) so raw
        # vertex values are already meters. If you see bbox values like
        # 6500, your USD has cm-scale raw values + a parent xformOp:scale
        # of 0.01 — bake the scale into the raw verts (see
        # tools/usd/bake_cloth_meters.py) before using here.
        bbox_min = verts.min(0)
        bbox_max = verts.max(0)
        bbox_size = bbox_max - bbox_min
        self._logger.info(
            f"[newton-standalone] cloth mesh '{name}': " f"{len(verts)} verts, bbox(m)={bbox_size.tolist()}"
        )
        if max(bbox_size) > 5.0:
            self._logger.warn(
                f"[newton-standalone] cloth bbox exceeds 5m — "
                f"likely a unit mismatch. Newton sims in meters; "
                f"the USD must be authored at meter-scale. Got bbox={bbox_size}"
            )

        tri: list[int] = []
        cursor = 0
        for n in counts:
            v0 = int(idxs[cursor])
            for k in range(1, n - 1):
                tri += [v0, int(idxs[cursor + k]), int(idxs[cursor + k + 1])]
            cursor += n
        indices = np.array(tri, dtype=np.int32)

        start = builder.particle_count
        builder.add_cloth_mesh(
            pos=wp.vec3(float(pose[0]), float(pose[1]), float(pose[2])),
            rot=wp.quat(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])),
            scale=1.0,
            vel=wp.vec3(float(vel[0]), float(vel[1]), float(vel[2])),
            vertices=verts,
            indices=indices,
            density=float(p.get("density", self._DENSITY)),
            tri_ke=float(p.get("tri_ke", self._TRI_KE)),
            tri_ka=float(p.get("tri_ka", self._TRI_KA)),
            tri_kd=float(p.get("tri_kd", self._TRI_KD)),
            edge_ke=float(p.get("edge_ke", self._EDGE_KE)),
            edge_kd=float(p.get("edge_kd", self._EDGE_KD)),
            particle_radius=float(p.get("particle_radius", self._PARTICLE_RADIUS)),
        )
        end = builder.particle_count
        self._cloth_particle_start = start
        self._cloth_particle_end = end
        self._cloth_usd_prim_path = prim_path
        self._logger.info(
            f"[newton-standalone] cloth '{name}': {end-start} particles "
            f"(tri_ke={p.get('tri_ke', self._TRI_KE)}, VBD)"
        )

    # ------------------------------------------------------------------
    # USDRT body writeback
    # ------------------------------------------------------------------
