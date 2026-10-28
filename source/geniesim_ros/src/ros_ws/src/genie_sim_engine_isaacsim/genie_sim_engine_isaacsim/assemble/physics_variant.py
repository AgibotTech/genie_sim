# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Compose AS3 physics before inspecting or editing a converted robot.

Isaac Sim 6.2's transformer emits an unselected Physics variant set. The
physx variant includes the common physics payload and retains DriveAPI,
which both Kit backends and the standalone Newton importer consume.
USD imports stay local so assemble discovery remains runtime independent.
"""

from collections import Counter
import re
import xml.etree.ElementTree as ET


def select_physics_variant(stage, robot_root=None, logger=None):
    """Select AS3 physx or legacy PhysX; leave assets without variants unchanged.

    Authors into the caller's edit target. Assembly uses the output root
    layer; runtime uses its in-memory scene, leaving the source asset alone.
    Never choose an arbitrary variant or silently accept a missing payload.
    """
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(robot_root) if robot_root else stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise RuntimeError(f"Robot physics: missing root {robot_root or 'defaultPrim'}")
    variants = root.GetVariantSets()
    if not variants.HasVariantSet("Physics"):
        return None
    physics = variants.GetVariantSet("Physics")
    available = physics.GetVariantNames()
    selected = next((name for name in ("physx", "PhysX") if name in available), None)
    if selected is None:
        raise RuntimeError(f"Robot physics: {root.GetPath()} requires 'physx' or 'PhysX'; available={available}")
    if not physics.SetVariantSelection(selected) or physics.GetVariantSelection() != selected:
        raise RuntimeError(f"Robot physics: failed to select {selected!r} on {root.GetPath()}")
    stage.Load(root.GetPath())
    bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not bodies:
        raise RuntimeError(
            f"Robot physics: {selected!r} on {root.GetPath()} composed zero rigid bodies; check payload paths"
        )
    if logger:
        logger.info(f"[robot-physics] {root.GetPath()}: Physics={selected}, {len(bodies)} rigid bodies")
    return selected


def validate_urdf_joints(stage, urdf_path):
    """Reject lost or duplicate movable joints before authoring/saving overlays."""
    from pxr import UsdPhysics

    urdf = ET.parse(urdf_path).getroot()
    expected = Counter(
        re.sub(r"^idx\d+_", "", joint.get("name", ""))
        for joint in urdf.findall("joint")
        if joint.get("type") in ("revolute", "continuous", "prismatic")
    )
    actual = Counter(
        re.sub(r"^idx\d+_", "", prim.GetName())
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
    )
    if expected != actual:
        raise RuntimeError(
            f"Robot physics: movable-joint inventory mismatch for {urdf_path}: "
            f"expected={sum(expected.values())}, composed={sum(actual.values())}; "
            f"missing={dict(expected - actual)}, unexpected={dict(actual - expected)}"
        )
    return sum(expected.values())
