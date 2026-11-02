# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Snapshot and restore PhysX PBD fluids without restarting the world."""

import numpy as np


def _position_attribute(prim):
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.PointInstancer):
        return UsdGeom.PointInstancer(prim).GetPositionsAttr()
    return UsdGeom.Points(prim).GetPointsAttr()


def snapshot_fluid_particles(stage, env_root=""):
    from pxr import PhysxSchema

    states = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if env_root and not (path == env_root or path.startswith(env_root + "/")):
            continue
        if not prim.HasAPI(PhysxSchema.PhysxParticleSetAPI):
            continue
        particle_set = PhysxSchema.PhysxParticleSetAPI(prim)
        if not particle_set.GetFluidAttr().Get():
            continue
        # PhysX writes smoothed render positions to points/positions. Read the
        # physical positions when available; simulationPoints itself is read-only.
        points = particle_set.GetSimulationPointsAttr().Get()
        if points is None or not len(points):
            points = _position_attribute(prim).Get()
        if points is None or not len(points):
            raise RuntimeError(f"Fluid particle set has no initial positions: {path}")
        points = np.asarray(points, dtype=np.float32).copy()
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise RuntimeError(f"Fluid particle set has invalid initial positions: {path}")
        states[path] = points
    return states


def restore_fluid_particles(stage, states):
    from pxr import UsdGeom, Vt

    # Validate all prims before writing any state. USD position/velocity changes
    # are consumed by PhysX on the next tick, as for runtime particle authoring.
    prims = {}
    for path in states:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Fluid particle set disappeared before reset: {path}")
        prims[path] = prim
    for path, points in states.items():
        prim = prims[path]
        _position_attribute(prim).Set(Vt.Vec3fArray.FromNumpy(points))
        geometry = UsdGeom.PointInstancer(prim) if prim.IsA(UsdGeom.PointInstancer) else UsdGeom.Points(prim)
        geometry.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.zeros_like(points)))
