# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Apply Channel B at both the USD seed and articulation tensor boundaries."""

import math


def _is_follower(prim):
    rel = prim.GetRelationship("newton:mimicJoint")
    return bool(rel and rel.HasAuthoredTargets()) or any(
        s.startswith("PhysxMimicJointAPI:") or s == "NewtonMimicAPI" for s in prim.GetAppliedSchemas()
    )


def _set(prim, name, value):
    from pxr import Sdf

    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False)
    if not attr.Set(float(value)):
        raise RuntimeError(f"[joint_params] failed to author {prim.GetPath()}.{name}")


def apply_usd_joint_params(stage, joint_prim_map, resolver, *, physics_engine, logger):
    if not resolver:
        return
    from pxr import UsdPhysics

    resolver.validate_names(joint_prim_map)
    for name, path in joint_prim_map.items():
        values = resolver.joint(name)
        if not values:
            continue
        prim = stage.GetPrimAtPath(path)
        angular = prim.IsA(UsdPhysics.RevoluteJoint)
        token = "angular" if angular else "linear"
        scale = math.pi / 180.0 if angular else 1.0
        follower = _is_follower(prim)
        if follower and any(values.get(f, 0) != 0 for f in ("stiffness", "damping_drive")):
            raise ValueError(f"[joint_params] {name}: mimic follower cannot carry a drive")
        if follower and prim.HasAPI(UsdPhysics.DriveAPI, token):
            # Remove even the Newton-specific class boot seed before reset.
            # The equality constraint alone moves an identified follower.
            for attr in ("stiffness", "damping"):
                _set(prim, f"drive:{token}:physics:{attr}", 0.0)
        if not follower:
            for field, attr in (("stiffness", "stiffness"), ("damping_drive", "damping")):
                if field in values:
                    UsdPhysics.DriveAPI.Apply(prim, token)
                    _set(prim, f"drive:{token}:physics:{attr}", values[field] * scale)
            effort = values.get("max_effort", values.get("effort"))
            if effort is not None:
                _set(prim, f"drive:{token}:physics:maxForce", effort)
        if "armature" in values:
            _set(prim, "physxJoint:armature", values["armature"])

        if physics_engine == "isaac_physx":
            if "velocity" in values:
                raise ValueError(f"[joint_params] {name}.velocity: PhysX overlay velocity limits are not implemented")
            # Isaac 6.2 / PhysX 110.4 exposes actual friction *efforts*,
            # unlike legacy jointFriction (a load-dependent coefficient).
            # Viscosity is authored per degree in angular USD attributes;
            # tensor/model values remain per radian, as the SDK requires.
            if any(f in values for f in ("damping", "frictionloss", "armature")):
                try:
                    applied = prim.ApplyAPI("PhysxJointAxisAPI", token)
                except Exception as exc:
                    raise RuntimeError(
                        "Sysid passive joint parameters require Isaac Sim 6.2 PhysxJointAxisAPI; "
                        "older runtimes must explicitly select robot.joint_overlays: []"
                    ) from exc
                if not applied:
                    raise RuntimeError(f"[joint_params] cannot apply PhysxJointAxisAPI to {path}")
                prefix = f"physxJointAxis:{token}:"
                if "damping" in values:
                    _set(prim, prefix + "viscousFrictionCoefficient", values["damping"] * scale)
                if "frictionloss" in values:
                    for attr in ("staticFrictionEffort", "dynamicFrictionEffort"):
                        _set(prim, prefix + attr, values["frictionloss"])
                    # Disable the old coefficient so it cannot add a second
                    # load-dependent friction term on top of the identified one.
                    _set(prim, "physxJoint:jointFriction", 0.0)
                if "armature" in values:
                    _set(prim, prefix + "armature", values["armature"])
        else:
            # Newton's own schema has precedence over Mjc/Physx. Converted
            # assets can already carry newton:friction/armature, so author
            # those too before the wrapper's reset compiles its solver.
            # newton:damping follows USD per-degree units; mjc:damping is SI
            # on Newton 1.4/1.6 with SchemaResolverMjc (see importer contract).
            for field, attr in (("damping", "mjc:damping"), ("frictionloss", "mjc:frictionloss")):
                if field in values:
                    _set(prim, attr, values[field])
            for field, attr in (("damping", "damping"), ("frictionloss", "friction"), ("armature", "armature")):
                if field in values:
                    _set(prim, f"newton:{attr}", values[field] * (scale if field == "damping" else 1.0))
        logger.info(f"[joint_params] USD {path}: applied per-joint values after class defaults")


def apply_articulation_joint_params(stage, joint_prim_map, joint_index, resolver, kps, kds, max_efforts):
    """Keep SI tensor gains and the visible USD properties consistent."""
    if not resolver:
        return
    from pxr import UsdPhysics

    for name, idx in joint_index.items():
        if not resolver.joint(name):
            continue
        prim = stage.GetPrimAtPath(joint_prim_map[name])
        kps[idx], kds[idx], max_efforts[idx] = resolver.drive(name, kps[idx], kds[idx], max_efforts[idx])
        if _is_follower(prim):
            kps[idx] = kds[idx] = 0.0
            continue
        token = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
        scale = math.pi / 180.0 if token == "angular" else 1.0
        _set(prim, f"drive:{token}:physics:stiffness", kps[idx] * scale)
        _set(prim, f"drive:{token}:physics:damping", kds[idx] * scale)
        _set(prim, f"drive:{token}:physics:maxForce", max_efforts[idx])
