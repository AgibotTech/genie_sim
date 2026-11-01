# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bake explicitly fixed tools as collision geometry of their parent link."""

from pathlib import Path


def bake_fixed_attachments(robot_usda: str, paths: list[str], out_path: str) -> str:
    """Write a composition layer; leave the shared robot/tool assets untouched.

    Paths are relative to the robot's default prim. A configured tool follows
    its existing rigid ancestor, keeping its authored local pose and colliders.
    It no longer contributes an independently simulated body or mass override.
    Never infer attachment from hierarchy alone: nested free bodies can be
    intentional, and joint-connected tools need a different representation.
    """
    if not paths:
        return robot_usda
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
        raise ValueError("robot.fixed_attachments must be a list of relative prim paths")
    source = str(Path(robot_usda).resolve())
    output = Path(out_path).resolve()
    if output == Path(source):
        raise ValueError("fixed attachment layer must not overwrite the robot asset")
    original = Usd.Stage.Open(source)
    if not original.GetDefaultPrim():
        raise ValueError(f"robot has no default prim: {source}")
    layer = Sdf.Layer.CreateAnonymous("fixed_attachments.usda")
    layer.subLayerPaths = [source]
    layer.defaultPrim = original.GetDefaultPrim().GetName()
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
    root = stage.GetDefaultPrim()
    joint_bodies = {
        target
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Joint)
        for rel in (UsdPhysics.Joint(prim).GetBody0Rel(), UsdPhysics.Joint(prim).GetBody1Rel())
        for target in rel.GetTargets()
    }
    for path in paths:
        rel = Sdf.Path(path)
        if not path or not rel.IsPrimPath() or rel.IsAbsolutePath() or ".." in path.split("/"):
            raise ValueError(f"fixed attachment must be a relative prim path: {path!r}")
        prim = stage.GetPrimAtPath(root.GetPath().AppendPath(rel))
        if not prim:
            raise ValueError(f"fixed attachment does not exist: {path}")
        parent = prim.GetParent()
        while parent and parent != root and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            parent = parent.GetParent()
        body = UsdPhysics.RigidBodyAPI(parent) if parent else None
        if not body or body.GetRigidBodyEnabledAttr().Get() is False:
            raise ValueError(f"fixed attachment has no enabled rigid parent: {path}")
        node = prim
        while node != parent:
            if UsdGeom.Xformable(node).GetResetXformStack():
                raise ValueError(f"fixed attachment resets its parent transform: {path}")
            node = node.GetParent()
        if any(target.HasPrefix(prim.GetPath()) for target in joint_bodies):
            raise ValueError(f"fixed attachment is already connected by a joint: {path}")
        if any(p != prim and p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(prim)):
            raise ValueError(f"fixed attachment contains nested rigid bodies: {path}")
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        prim.RemoveAPI(UsdPhysics.MassAPI)
        prim.RemoveAppliedSchema("PhysxRigidBodyAPI")
    output.parent.mkdir(parents=True, exist_ok=True)
    layer.Export(str(output))
    return str(output)
