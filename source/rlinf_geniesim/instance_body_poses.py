# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bake instance reset tables from fully composed, unsimulated foreground USD.

The locator table remains useful for checkers and parking. Physics gets a
separate table of complete rigid-body poses, including scene-authored overrides
inside referenced assets. This module never edits a source layer or starts Kit.
"""

from __future__ import annotations

import copy
import hashlib

import numpy as np

# Element counts at or above this use the whole-buffer digest in `_value`.
# Below it, values keep the exact element-wise path: a transform, a quaternion
# or a 3-vector is COMPUTED (matrix products across differently-placed
# instances), so the 1e-6 rounding there is absorbing real numerical noise and
# must stay. Mesh arrays are READ from one shared asset file instead, so their
# elements are bit-identical between instances and nothing is being absorbed.
_DIGEST_MIN_ELEMENTS = 64


def _pose(matrix):
    from pxr import Gf

    q = Gf.Transform(matrix).GetRotation().GetQuat().GetNormalized()
    quat = np.asarray([q.GetReal(), *q.GetImaginary()], dtype=float)
    if quat[0] < 0:
        quat = -quat
    return [float(x) for x in matrix.ExtractTranslation()] + quat.tolist()


def _matrix(row, context="parking pose"):
    from pxr import Gf

    row = np.asarray(row, dtype=float)
    if row.shape != (7,) or not np.isfinite(row).all():
        raise ValueError(f"{context}: expected seven finite pose values")
    _quaternion(row[3:], context)
    matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(float(row[3]), Gf.Vec3d(*row[4:7])).GetNormalized())
    matrix.SetTranslateOnly(Gf.Vec3d(*row[:3]))
    return matrix


def _array_digest(value):
    """Hash a numeric USD array after float64 conversion and rounding.

    The digest preserves numeric int/float equivalence without per-element Python
    recursion. Return None for nonnumeric arrays and Python lists/tuples, which use
    the element-wise signature path."""
    if isinstance(value, (str, bytes, list, tuple)) or not hasattr(value, "__len__"):
        return None
    if len(value) < 8:
        return None
    try:
        arr = np.asarray(value)
    except Exception:  # noqa: BLE001 - any unconvertible array falls back
        return None
    if arr.dtype.kind not in "fiub" or arr.size < _DIGEST_MIN_ELEMENTS:
        return None
    arr = np.ascontiguousarray(np.round(arr.astype(np.float64), 6))
    return ("@array", arr.shape, hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest())


def _value(value):
    """Comparable schema values without USD names or layer metadata."""
    if isinstance(value, (str, bool, int, type(None))):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if hasattr(value, "GetReal"):
        return _value((value.GetReal(), *value.GetImaginary()))
    digest = _array_digest(value)
    if digest is not None:
        return digest
    try:
        return tuple(_value(x) for x in value)
    except TypeError:
        return str(value)


def _quaternion(values, context):
    q = np.asarray(values, dtype=float)
    if not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1.0) > 1e-4:
        raise ValueError(f"{context}: invalid quaternion (must be finite and unit length)")


def _check_matrix(matrix, context):
    arr = np.asarray(matrix)
    if not np.isfinite(arr).all() or not np.allclose(arr[:, 3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{context}: invalid affine transform")
    linear = arr[:3, :3]
    scale = np.linalg.norm(linear, axis=1)
    if np.any(scale < 1e-8) or np.linalg.det(linear) <= 0:
        raise ValueError(f"{context}: zero or reflected scale cannot be baked as a rigid pose")
    axes = linear / scale[:, None]
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError(f"{context}: shear cannot be baked as a rigid pose")


def _validate_xforms(root, cache, context):
    from pxr import Usd, UsdGeom

    ancestors = []
    parent = root.GetParent()
    while parent and not parent.IsPseudoRoot():
        ancestors.append(parent)
        parent = parent.GetParent()
    for prim in [*ancestors, *Usd.PrimRange(root, Usd.TraverseInstanceProxies())]:
        xform = UsdGeom.Xformable(prim)
        if not xform:
            continue
        where = f"{context} {prim.GetPath()}"
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                q = op.Get()
                if q is None:
                    raise ValueError(f"{where}: missing quaternion value")
                _quaternion([q.GetReal(), *q.GetImaginary()], where)
        _check_matrix(cache.GetLocalToWorldTransform(prim), where)


def _stable_path(path, aliases):
    parts = str(path).split("/")
    if len(parts) >= 4 and parts[1:3] == ["World", "Objects"]:
        parts[3] = aliases.get(parts[3], parts[3])
    return "/".join(parts)


def _physics_material(prim):
    """Effective physical parameters, independent of material names or hashes."""
    from pxr import UsdShade

    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    if not material:
        return None
    bound = material.GetPrim()
    schemas = tuple(sorted(s for s in bound.GetAppliedSchemas() if s.startswith(("Physics", "Physx"))))
    if not schemas:
        # ComputeBoundMaterial also falls back to the all-purpose visual
        # binding. A shading-only material contributes no physical parameters.
        return None
    return (
        schemas,
        {a.GetName(): _value(a.Get()) for a in bound.GetAttributes() if a.GetName().startswith(("physics:", "physx"))},
    )


def _signature(root, cache, bodies, aliases, joints):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    result = {}
    root_world = cache.GetLocalToWorldTransform(root)
    constrained = {
        target
        for joint in joints
        for name in ("physics:body0", "physics:body1")
        for target in joint.GetRelationship(name).GetTargets()
    }
    free = [
        b
        for b in bodies
        if b.GetPath() not in constrained and not UsdPhysics.RigidBodyAPI(b).GetKinematicEnabledAttr().Get()
    ]
    body_paths = {b.GetPath() for b in bodies if not UsdPhysics.RigidBodyAPI(b).GetKinematicEnabledAttr().Get()}
    connected = [
        [joint.GetRelationship(name).GetTargets() for name in ("physics:body0", "physics:body1")]
        for joint in joints
        if any(
            t.HasPrefix(root.GetPath())
            for name in ("physics:body0", "physics:body1")
            for t in joint.GetRelationship(name).GetTargets()
        )
    ]
    # Joint body0 is not necessarily anchored: a wholly internal dynamic
    # articulation has a FREE base joint in Newton. Its complete local layout
    # must match, while the slot may move as a whole. Keep world anchoring,
    # cross-slot links and kinematic endpoints on the conservative world check.
    floating = (
        [b for b in bodies if b.GetPath() in constrained]
        if connected and all(len(side) == 1 and side[0] in body_paths for pair in connected for side in pair)
        else []
    )
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        # Visual switching is separate. Physical materials are compared at the
        # consuming prim below, including inherited and external bindings.
        if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader):
            continue
        physics_attrs = {
            a.GetName(): _value(a.Get())
            for a in prim.GetAttributes()
            if a.GetName().startswith(("physics:", "physx")) or ":physics:" in a.GetName()
        }
        geometry = prim.IsA(UsdGeom.Gprim)
        if not physics_attrs and not geometry and prim not in bodies:
            continue
        rel = str(prim.GetPath().MakeRelativePath(root.GetPath()))
        attrs = dict(physics_attrs)
        attrs["@physics_material"] = _physics_material(prim)
        for rel_attr in prim.GetRelationships():
            if rel_attr.GetName().startswith(("physics:", "physx")):
                attrs[rel_attr.GetName()] = tuple(_stable_path(p, aliases) for p in rel_attr.GetTargets())
        if geometry:
            for attr in prim.GetAttributes():
                name = attr.GetName()
                if not name.startswith(("xformOp", "primvars:", "material:", "physics:", "physx")) and name not in {
                    "visibility",
                    "purpose",
                    "extent",
                    "normals",
                    "doubleSided",
                }:
                    attrs[name] = _value(attr.Get())
        world = cache.GetLocalToWorldTransform(prim)
        if prim in free:
            # A free rigid root's full p/q may differ between scene instances.
            # Its scale still changes geometry, inertia and collision shapes.
            linear = np.asarray(world)[:3, :3]
            transform = _value((linear @ linear.T).tolist())
        else:
            parent_body = next((b for b in reversed(free) if b != prim and prim.GetPath().HasPrefix(b.GetPath())), None)
            # Reset writes only dynamic FREE bodies. Static/kinematic shapes
            # cannot inherit a changed locator pose through the reset table.
            collision = UsdPhysics.CollisionAPI(prim)
            if parent_body:
                reference = cache.GetLocalToWorldTransform(parent_body)
            elif any(prim.GetPath().HasPrefix(b.GetPath()) for b in floating):
                reference = root_world
            elif (
                (free or floating)
                and prim not in bodies
                and not (collision and collision.GetCollisionEnabledAttr().Get())
            ):
                # A free asset can also have nonphysical helper geometry (the
                # pot's invisible lowpoly/bbox). This does not pin its body.
                reference = root_world
            else:
                reference = Gf.Matrix4d(1)
            transform = _value(np.asarray(world * reference.GetInverse()).tolist())
        result[rel] = (prim.GetTypeName(), attrs, transform)
    # Locator scaling affects every shape even when its rigid root is nested.
    linear = np.asarray(root_world)[:3, :3]
    result["@locator_scale"] = _value((linear @ linear.T).tolist())
    # Joints may be authored outside their body's object subtree.
    for joint in joints:
        relationships = {
            r.GetName(): tuple(_stable_path(t, aliases) for t in r.GetTargets())
            for r in joint.GetRelationships()
            if r.GetName().startswith(("physics:", "physx"))
        }
        if any(
            t.HasPrefix(root.GetPath())
            for name in ("physics:body0", "physics:body1")
            for t in joint.GetRelationship(name).GetTargets()
        ):
            result["@joint:" + _stable_path(joint.GetPath(), aliases)] = (
                joint.GetTypeName(),
                relationships,
                {
                    a.GetName(): _value(a.Get())
                    for a in joint.GetAttributes()
                    if a.GetName().startswith(("physics:", "physx")) or ":physics:" in a.GetName()
                },
            )
    return result


def _freeze(value):
    if isinstance(value, dict):
        return tuple((k, _freeze(value[k])) for k in sorted(value))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def signature_key(signature):
    """Freeze a physical signature into a hashable key with numeric equality.

    Tuples preserve -0.0 == 0.0, preventing equivalent transforms from splitting
    into separate compatibility groups."""
    return _freeze(signature)


def _nonresettable_prims(root, bodies, joints):
    """Physical prims whose world transforms the FREE-body reset cannot set."""
    from pxr import Usd, UsdGeom, UsdPhysics

    dynamic = {b.GetPath(): b for b in bodies if not UsdPhysics.RigidBodyAPI(b).GetKinematicEnabledAttr().Get()}
    adjacent = {path: set() for path in dynamic}
    anchored = set()
    for joint in joints:
        ends = [joint.GetRelationship(name).GetTargets() for name in ("physics:body0", "physics:body1")]
        inside = [side[0] for side in ends if len(side) == 1 and side[0] in dynamic]
        if not inside:
            continue
        if len(inside) == 2:
            adjacent[inside[0]].add(inside[1])
            adjacent[inside[1]].add(inside[0])
        else:
            # The other endpoint is world, kinematic, or outside this slot.
            anchored.update(inside)
    pending = list(anchored)
    while pending:
        for path in adjacent[pending.pop()]:
            if path not in anchored:
                anchored.add(path)
                pending.append(path)
    movable = set(dynamic) - anchored
    fixed = {b.GetPath(): b for b in bodies if b.GetPath() not in movable}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        collision = UsdPhysics.CollisionAPI(prim)
        if not collision or collision.GetCollisionEnabledAttr().Get() is False:
            continue
        if any(prim.GetPath().HasPrefix(path) for path in movable):
            continue
        fixed[prim.GetPath()] = prim
    if not movable and not fixed:
        # A visual-only object has no body-table entry either. The whole mesh
        # stays at startup when a variant slot is parked or its locator moves.
        fixed = {p.GetPath(): p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Gprim)}
    return fixed.values()


def _open_stage(path, iid):
    from pxr import Usd, UsdGeom

    try:
        stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    except Exception as exc:
        raise ValueError(f"instance {iid}: cannot open scene USD {path!r}: {exc}") from exc
    if stage is None:
        raise ValueError(f"instance {iid}: cannot open scene USD {path!r}")
    errors = stage.GetCompositionErrors()
    if errors:
        raise ValueError(f"instance {iid}: unresolved USD composition/payload: {errors}")
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0, rtol=0, atol=1e-9):
        raise ValueError(f"instance {iid}: metersPerUnit must be 1 for environment-local metre poses")
    return stage


def bake_instance_body_poses(
    *,
    scene_paths: dict[str, str],
    group: dict,
    background_path: str = "",
    corrected_scene_paths: dict[str, str] | None = None,
) -> dict:
    """Return versioned body poses and corrected locator poses, without mutation.

    ``name_alias`` is the sole member-to-slot pairing. ``scene_paths`` should
    already contain any spawn corrections; optional corrected paths override
    it by instance. Background support corrections belong to the caller, so
    ``background_path`` is accepted for the loader contract but never loaded.
    Poses are in the source stage's world frame (the clone's environment frame),
    with scalar-first quaternions. Static slots produce no body-table entries.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    paths = {str(k): v for k, v in scene_paths.items()}
    paths.update({str(k): v for k, v in (corrected_scene_paths or {}).items()})
    locators = copy.deepcopy(group["instance_poses"])
    body_tables, references, signatures, fixed_references = {}, {}, {}, {}
    for iid_value in group["group_instances"]:
        iid = str(iid_value)
        if iid not in paths:
            raise ValueError(f"instance {iid}: missing scene USD path")
        stage = _open_stage(paths[iid], iid)
        cache = UsdGeom.XformCache()
        body_tables[iid] = {}
        aliases = group["name_alias"][iid]
        if len(set(aliases.values())) != len(aliases):
            raise ValueError(f"instance {iid}: name_alias assigns multiple objects to the same slot")
        joints = [p for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()) if p.IsA(UsdPhysics.Joint)]
        for member, slot in aliases.items():
            root = stage.GetPrimAtPath(f"/World/Objects/{member}")
            if not root or not root.IsActive():
                raise ValueError(f"instance {iid}: missing object {member!r} for slot {slot!r}")
            if slot not in locators[iid]:
                raise ValueError(f"instance {iid}: alias slot {slot!r} has no locator pose")
            _validate_xforms(root, cache, f"instance {iid}")
            bodies = [
                p
                for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get()
            ]
            signature = _signature(root, cache, bodies, aliases, joints)
            if slot in signatures and signature != signatures[slot]:
                previous = signatures[slot]
                changed = next(
                    k for k in sorted(previous.keys() | signature.keys()) if previous.get(k) != signature.get(k)
                )
                raise ValueError(
                    f"incompatible physical structure for slot {slot!r} in instance {iid} "
                    f"at {changed!r}; regroup or rebuild"
                )
            signatures[slot] = signature
            root_world = cache.GetLocalToWorldTransform(root)
            locators[iid][slot] = _pose(root_world)
            fixed = {}
            for prim in _nonresettable_prims(root, bodies, joints):
                rel = str(prim.GetPath().MakeRelativePath(root.GetPath()))
                world = cache.GetLocalToWorldTransform(prim)
                fixed[rel] = (world, world * _matrix(_pose(root_world)).GetInverse())
            if slot in fixed_references:
                for rel, (world, _) in fixed.items():
                    prior = fixed_references[slot][rel][0]
                    if not np.allclose(np.asarray(world), np.asarray(prior), atol=1e-6, rtol=0):
                        raise ValueError(
                            f"slot {slot!r} in instance {iid}: physical prim {rel!r} "
                            "moves but reset cannot move a static, kinematic or "
                            "world-constrained structure; regroup or rebuild"
                        )
            else:
                fixed_references[slot] = fixed
            relative = {}
            for body in bodies:
                rel = str(body.GetPath().MakeRelativePath(root.GetPath()))
                key = slot if rel == "." else f"{slot}/{rel}"
                world = cache.GetLocalToWorldTransform(body)
                body_tables[iid][key] = _pose(world)
                # Parking rows carry p/q, not the locator's authored scale.
                # Remove only that p/q so the reference's scale still affects
                # nested body offsets when the slot is parked elsewhere.
                relative[key] = world * _matrix(_pose(root_world)).GetInverse()
            references.setdefault(slot, relative)
    for iid_value in group["group_instances"]:
        iid = str(iid_value)
        used = set(group["name_alias"][iid].values())
        for slot, row in locators[iid].items():
            if slot not in used:
                if slot not in references:
                    raise ValueError(f"instance {iid}: parked slot {slot!r} has no real reference object")
                parked = _matrix(row, f"instance {iid} parked slot {slot!r}")
                for rel, (world, relative) in fixed_references[slot].items():
                    target = relative * parked
                    if not np.allclose(np.asarray(target), np.asarray(world), atol=1e-6, rtol=0):
                        raise ValueError(
                            f"parked slot {slot!r} in instance {iid}: physical prim "
                            f"{rel!r} moves but reset cannot move a static, kinematic "
                            "or world-constrained structure; regroup or rebuild"
                        )
                for key, relative in references[slot].items():
                    body_tables[iid][key] = _pose(relative * parked)
    return {"body_pose_version": 1, "instance_body_poses": body_tables, "instance_poses": locators}
