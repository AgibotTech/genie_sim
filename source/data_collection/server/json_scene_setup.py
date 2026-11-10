# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Validate a complete JSON object batch without importing the simulation SDK.

The returned dictionaries match ``CommandController._add_usd_object``. This
module only reads the filesystem and an optional stage-path predicate; callers
must finish validation before removing or authoring any scene objects.
"""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path, PurePosixPath


_REQUIRED = frozenset({
    "usd_path", "prim_path", "label_name", "position", "rotation", "scale",
    "object_color", "object_material", "object_mass",
})
_DEFAULTS = {
    "static_friction": 0.5,
    "dynamic_friction": 0.5,
    "model_type": "convexDecomposition",
    "add_rigid_body": True,
}
_ALLOWED = _REQUIRED | _DEFAULTS.keys()


def _validate_prim_path(value):
    if not isinstance(value, str) or not value.startswith("/World/Objects/"):
        raise ValueError("prim_path must name a descendant of /World/Objects")
    path = PurePosixPath(value)
    if str(path) != value or not all(part.isidentifier() for part in path.parts[1:]):
        raise ValueError(f"Invalid or noncanonical prim_path: {value!r}")
    return value


def _validate_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    return value


def _validate_array(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers")
    return [_validate_number(item, name) for item in value]


def _resolve_asset(value, root):
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("usd_path must be a nonempty file path")
    try:
        path = Path(value)
        path = (path if path.is_absolute() else root / path).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid asset file: {value!r}") from exc
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Asset must be a regular file inside assets_root: {value!r}")
    return str(path)


def validate_scene_objects(objects, assets_root, owned_paths=(), existing_path=None):
    """Return validated object kwargs, or reject the batch before any mutation.

    ``objects`` is a JSON list with nine required object arguments and the four
    existing optional arguments. Asset paths resolve inside ``assets_root``;
    prim paths must be canonical descendants of ``/World/Objects``. Existing
    prims may only be replaced when their exact paths are in ``owned_paths``.
    The optional ``existing_path(path)`` predicate performs read-only stage
    lookups after all object values and batch overlaps have been validated.

    Position, quaternion, scale, color, mass, and friction values retain their
    meanings and magnitudes. Returned arrays are independent copies. Previous
    owned paths are checked even for an empty replacement batch, so a caller
    cannot delete an unsafe path merely by preparing an empty scene.
    """
    if not isinstance(objects, list):
        raise TypeError("objects must be a list")
    try:
        root = Path(assets_root).resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("assets_root must be an existing directory") from exc
    if not root.is_dir():
        raise ValueError("assets_root must be an existing directory")
    if isinstance(owned_paths, (str, bytes, dict)):
        raise TypeError("owned_paths must be an iterable of prim paths")
    owned = {_validate_prim_path(path) for path in owned_paths}
    if existing_path is not None and not callable(existing_path):
        raise TypeError("existing_path must be callable")

    normalized = []
    paths = set()
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise TypeError(f"objects[{index}] must be a dictionary")
        missing = _REQUIRED - item.keys()
        unknown = item.keys() - _ALLOWED
        if missing or unknown:
            raise ValueError(f"objects[{index}] has missing keys {missing!r} or unknown keys {unknown!r}")
        result = {**_DEFAULTS, **item}
        result["prim_path"] = _validate_prim_path(result["prim_path"])
        if result["prim_path"] in paths:
            raise ValueError(f"Duplicate prim_path: {result['prim_path']!r}")
        paths.add(result["prim_path"])
        result["usd_path"] = _resolve_asset(result["usd_path"], root)
        for name in ("label_name", "object_material", "model_type"):
            if not isinstance(result[name], str):
                raise TypeError(f"{name} must be a string")
        for name, length in (("position", 3), ("rotation", 4), ("scale", 3), ("object_color", 3)):
            result[name] = _validate_array(result[name], length, name)
        if not any(result["rotation"]):
            raise ValueError("rotation must be a nonzero quaternion")
        if any(value <= 0 for value in result["scale"]):
            raise ValueError("scale components must be positive")
        for name in ("object_mass", "static_friction", "dynamic_friction"):
            result[name] = _validate_number(result[name], name)
        if result["object_mass"] <= 0:
            raise ValueError("object_mass must be positive")
        if result["static_friction"] < 0 or result["dynamic_friction"] < 0:
            raise ValueError("friction coefficients must be nonnegative")
        if not isinstance(result["add_rigid_body"], bool):
            raise TypeError("add_rigid_body must be a boolean")
        normalized.append(result)

    ordered_paths = sorted(paths)
    for parent, child in zip(ordered_paths, ordered_paths[1:]):
        if child.startswith(parent + "/"):
            raise ValueError(f"Overlapping object prim paths: {parent!r}, {child!r}")
    if existing_path is not None:
        for path in ordered_paths:
            if path not in owned and existing_path(path):
                raise ValueError(f"Object prim path already exists and is not owned: {path!r}")
    return normalized


def validate_no_joint_references(stage, prim_path):
    """Reject joints targeting a prospective object subtree without authoring it."""
    from pxr import Sdf, UsdPhysics

    root_path = Sdf.Path(prim_path)
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            targets = joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets()
            if any(path.HasPrefix(root_path) for path in targets):
                raise ValueError(f"Joint references the object subtree: {prim_path}")


def normalize_single_body_root(stage, prim_path, *, author=True):
    """Validate and remove a redundant identity child's rigid APIs in JSON setup.

    The caller subsequently authors the requested rigid body and mass on the
    logical object root. Keeping a native child body as well creates an empty
    outer actor: its pose free-falls independently of the child's collider and
    is then used incorrectly by IK and recording. Only the trivial single-body
    case is supported here; never flatten joints, body frames or tuned physics.
    ``author=False`` performs the same validation without modifying any layer.
    SDK imports stay local so batch-value validation remains SDK independent.
    """
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    if not isinstance(author, bool):
        raise TypeError("author must be a boolean")
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        raise ValueError(f"Missing object root: {prim_path}")
    subtree = list(Usd.PrimRange(root))
    children = [prim for prim in subtree
                if prim != root and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not children:
        return False

    def reject(reason):
        raise ValueError(f"Cannot promote nested rigid body at {prim_path}: {reason}")

    if len(children) != 1 or root.HasAPI(UsdPhysics.RigidBodyAPI):
        reject("expected exactly one child body and no root body")
    if UsdGeom.Xformable(root).GetResetXformStack():
        reject("object root must inherit its transform")
    body = children[0]
    body_path = body.GetPath()
    for prim in subtree:
        schemas = list(prim.GetAppliedSchemas())
        authored_schemas = prim.GetMetadata("apiSchemas")
        if authored_schemas:
            schemas.extend(authored_schemas.GetAppliedItems())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) or any("Deformable" in s for s in schemas):
            reject("articulation or deformable schema")
        if prim.IsA(UsdPhysics.Joint):
            reject("joint in object subtree")
        if (prim.HasAPI(UsdPhysics.CollisionAPI)
                and not prim.GetPath().HasPrefix(body_path)
                and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False):
            reject("collider outside the sole body subtree")
    validate_no_joint_references(stage, prim_path)

    current = body
    while current != root:
        xform = UsdGeom.Xformable(current)
        if xform and (xform.GetResetXformStack()
                      or not Gf.IsClose(xform.GetLocalTransformation(), Gf.Matrix4d(1), 1e-10)
                      or any(op.GetAttr().GetNumTimeSamples() for op in xform.GetOrderedXformOps())):
            reject("child body frame must be static, identity and inherit its transform")
        current = current.GetParent()
    rigid = UsdPhysics.RigidBodyAPI(body)
    if (rigid.GetRigidBodyEnabledAttr().Get() is False
            or rigid.GetKinematicEnabledAttr().Get() is True):
        reject("disabled or kinematic body")
    allowed = {"physics:rigidBodyEnabled", "physics:kinematicEnabled"}
    for attr in body.GetAuthoredAttributes():
        if attr.GetName().startswith(("physics:", "physx")):
            if attr.GetName() not in allowed or attr.GetNumTimeSamples():
                reject("authored body physics properties")
    if any(rel.GetName().startswith(("physics:", "physx"))
           for rel in body.GetAuthoredRelationships()):
        reject("authored body physics relationships")
    schemas = list(body.GetAppliedSchemas())
    authored_schemas = body.GetMetadata("apiSchemas")
    if authored_schemas:
        schemas.extend(authored_schemas.GetAppliedItems())
    if any(schema.startswith(("Physics", "Physx"))
           and schema not in {"PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"}
           for schema in schemas):
        reject("additional body physics API")
    if author:
        body.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if body.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            body.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
    return True
