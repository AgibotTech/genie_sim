# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Prepare imported sandbox physics before the first simulation tick."""

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def prepare_sandbox_physics(stage, scene_path, world_root="/World"):
    """Repair sandbox-only body hierarchies without changing source asset layers."""
    parts = Path(scene_path).parts
    if ("background", "sandbox") not in zip(parts, parts[1:]):
        return
    root = stage.GetPrimAtPath(world_root)
    if not root:
        return

    prims = list(Usd.PrimRange(root))
    bodies = {
        prim.GetPath()
        for prim in prims
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
    }
    cache = UsdGeom.XformCache()
    nested = []
    for prim in prims:
        if prim.GetPath() not in bodies or UsdGeom.Xformable(prim).GetResetXformStack():
            continue
        parent = prim.GetParent()
        while parent and not parent.IsPseudoRoot():
            if parent.GetPath() in bodies:
                nested.append((prim, cache.GetLocalToWorldTransform(prim)))
                break
            parent = parent.GetParent()

    for prim, matrix in nested:
        transform = Gf.Transform(matrix)
        rotation = transform.GetRotation().GetQuat()
        scale = transform.GetScale()
        reconstructed = Gf.Matrix4d().SetScale(scale)
        reconstructed *= Gf.Matrix4d().SetRotate(rotation)
        reconstructed *= Gf.Matrix4d().SetTranslate(transform.GetTranslation())
        if not Gf.IsClose(matrix, reconstructed, 1e-8):
            raise ValueError(f"Sandbox nested body has an unsupported sheared transform: {prim.GetPath()}")

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        for name, add_op, value in (
            ("translate", xform.AddTranslateOp, transform.GetTranslation()),
            ("orient", xform.AddOrientOp, rotation),
            ("scale", xform.AddScaleOp, scale),
        ):
            attr = prim.GetAttribute(f"xformOp:{name}")
            op = UsdGeom.XformOp(attr) if attr else add_op(UsdGeom.XformOp.PrecisionDouble)
            if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                value = Gf.Quatf(value) if name == "orient" else Gf.Vec3f(value)
            op.Set(value)
        xform.SetXformOpOrder(
            [UsdGeom.XformOp(prim.GetAttribute(f"xformOp:{name}")) for name in ("translate", "orient", "scale")],
            resetXformStack=True,
        )

    for prim in prims:
        if (
            prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and not any(body.HasPrefix(prim.GetPath()) for body in bodies)
        ):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
