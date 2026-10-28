# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Synchronize authored mimic followers with the fused robot's initial pose."""
from __future__ import annotations

import math

import numpy as np
from pxr import Usd, UsdPhysics


def synchronize_mimic_joint_positions(robot_root, dof_names, positions):
    """Return a copy of ``positions`` (N, J, radians) with mimic followers posed.

    Supports the G2's single-DOF revolute PhysxMimicJointAPI constraints. The
    PhysxSchema 110.0.7 schema defines ``q + gearing * q_ref + offset = 0``,
    with gearing defaulting to 1 and angular offset to 0 degrees. The Isaac
    tensor API uses radians, so only the offset needs conversion here.

    Relationship targets are resolved by full USD path within ``robot_root``;
    unique prim names then map those joints to the actual articulation DOFs.
    Malformed constraints fail before the caller writes any pose to PhysX.
    """
    result = np.array(positions, copy=True)
    if result.ndim != 2 or result.shape[1] != len(dof_names):
        raise ValueError("Mimic initial pose must have shape (N, number of DOFs)")
    if not robot_root or not robot_root.IsValid():
        raise ValueError("Mimic initial pose requires a valid robot root")
    dof_indices = {name: index for index, name in enumerate(dof_names)}
    if len(dof_indices) != len(dof_names):
        raise ValueError("Mimic initial pose requires unique articulation DOF names")

    joints, by_name, followers = {}, {}, []
    for prim in Usd.PrimRange(robot_root):
        if prim.IsA(UsdPhysics.Joint):
            joints[prim.GetPath()] = prim
            by_name.setdefault(prim.GetName(), []).append(prim.GetPath())
        # Read composed authored tokens: host USD may lack the PhysxSchema
        # plugin, in which case GetAppliedSchemas() omits this same schema.
        schemas = prim.GetMetadata("apiSchemas")
        mimic = [s for s in schemas.GetAppliedItems()
                 if s.startswith("PhysxMimicJointAPI:")] if schemas else []
        if mimic:
            followers.append((prim, mimic))

    def dof_index(prim):
        name = prim.GetName()
        if name not in dof_indices or len(by_name.get(name, [])) != 1:
            raise ValueError(f"Mimic joint {prim.GetPath()} has no unique articulation DOF")
        return dof_indices[name]

    def coefficient(prim, name, default):
        attr = prim.GetAttribute(name)
        value = attr.Get() if attr else None
        if value is None and not (attr and attr.HasAuthoredValueOpinion()):
            value = default
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Mimic coefficient {prim.GetPath()}.{name} is invalid") from exc
        if not math.isfinite(value):
            raise ValueError(f"Mimic coefficient {prim.GetPath()}.{name} must be finite")
        return value

    constraints = {}
    for prim, schemas in followers:
        if not prim.IsA(UsdPhysics.RevoluteJoint) or len(schemas) != 1:
            raise ValueError(f"Mimic joint {prim.GetPath()} must have one revolute constraint")
        axis = schemas[0].split(":", 1)[1]
        if axis not in ("rotX", "rotY", "rotZ"):
            raise ValueError(f"Mimic joint {prim.GetPath()} has unsupported axis {axis}")
        prefix = f"physxMimicJoint:{axis}:"
        relationship = prim.GetRelationship(prefix + "referenceJoint")
        targets = relationship.GetTargets() if relationship else []
        if len(targets) != 1 or targets[0] not in joints:
            raise ValueError(f"Mimic joint {prim.GetPath()} needs one reference joint inside {robot_root.GetPath()}")
        reference = joints[targets[0]]
        if not reference.IsA(UsdPhysics.RevoluteJoint):
            raise ValueError(f"Mimic reference {reference.GetPath()} must be revolute")
        constraints[dof_index(prim)] = (
            dof_index(reference), coefficient(prim, prefix + "gearing", 1.),
            math.radians(coefficient(prim, prefix + "offset", 0.)))

    # Resolve sources before their dependents, independently of USD/DOF order.
    pending = dict(constraints)
    while pending:
        ready = [index for index, (reference, _, _) in pending.items() if reference not in pending]
        if not ready:
            names = [dof_names[index] for index in pending]
            raise ValueError(f"Mimic joint dependency cycle: {names}")
        for index in ready:
            reference, gearing, offset = pending.pop(index)
            result[:, index] = -gearing * result[:, reference] - offset
    return result
