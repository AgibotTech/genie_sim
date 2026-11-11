# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bind transforms needed for PhysX rigid PointInstancer position readback."""


def snapshot_rigid_instancer_prototypes(stage):
    """Capture before physics starts, retain across workspace-only replacements.

    PhysX writes rigid instance positions relative to the prototype transform
    at creation. If an external prototype is also an independent rigid body,
    its live transform changes. USD's IncludeProtoXform would apply that change
    a second time, shifting all instance centers away from their physics poses.
    PBD particle instancers do not use this rigid-body convention.
    """
    from pxr import UsdGeom, UsdPhysics

    snapshots = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.PointInstancer):
            continue
        targets = UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets()
        prototypes = [stage.GetPrimAtPath(path) for path in targets]
        if prototypes and all(p.IsValid() and p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prototypes):
            snapshots[str(prim.GetPath())] = tuple(UsdGeom.Xformable(p).GetLocalTransformation() for p in prototypes)
    return snapshots
