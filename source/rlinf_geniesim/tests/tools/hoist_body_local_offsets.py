#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Move a scene's per-instance offset from INSIDE a rigid body onto its root.

Some benchmark scenes place an object by translating the collider *within* the
rigid body instead of moving the body. The two are the same arrangement -- the
body's frame is not observable, and PhysX derives mass properties from the
shapes either way -- but only one of them is a POSITION as far as the reset
tables are concerned: ``instance_body_poses`` records rigid-body root poses, so
an offset authored below the root cannot be carried by it. Two instances that
differ only there therefore cannot share a superset slot, and each one needs
its own copy of the geometry.

pour_workpiece is the real case: its 20 instances scatter accessories by
translating ``entity/Aligned/visual`` by one or two centimetres, 33 distinct
values over 72 objects. That turned 9 slots into 50 -- 50 free bodies, which
``build_superset_stage_manifest`` rejects outright (est. 229 ms/physics-step).
Hoisting the offsets gives the same scene back at 11 slots (est. 34 ms).

The rewrite is only applied where it is provably equivalent:

  * every prim between the body root and the offset prim has an identity local
    transform (otherwise the offset does not commute up to the root);
  * every Gprim, collider and physics-carrying prim under the body lies within
    the offset prim's subtree, so nothing is left behind when the root moves
    (materials and scopes are exempt -- a transform on them changes nothing);
  * the body authors no explicit centre of mass or inertia, which are stated in
    the body's own frame and would not move with the shapes;
  * no joint or USD relationship/connection consumes a moved frame, the offset
    prim whose op order is cleared, or a transform op that is rewritten; no
    affected transform is time-sampled or resets the transform stack.

Both modes verify default-time composed world transforms before the original
scene is changed. The guards above cover semantics that this pose check alone
cannot prove.

    hoist_body_local_offsets.py pour_workpiece            # report only
    hoist_body_local_offsets.py --apply pour_workpiece    # rewrite in place
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3]
BENCH = SOURCE / "geniesim_benchmark" / "src" / "geniesim_benchmark" / "benchmark" / "config" / "llm_task"

_INERTIAL = ("physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes")


def _is_identity(matrix, tol=1e-12):
    from pxr import Gf

    return Gf.IsClose(matrix, Gf.Matrix4d(1), tol)


def _local(prim, cache):
    """A prim's own local transform (world of prim, world of parent removed)."""
    from pxr import Gf

    parent = prim.GetParent()
    world = cache.GetLocalToWorldTransform(prim)
    if not parent or parent.IsPseudoRoot():
        return world
    return world * Gf.Matrix4d(cache.GetLocalToWorldTransform(parent)).GetInverse()


def _transform_guard(body):
    """Reject time-varying or reset stacks omitted by a default-time rewrite."""
    from pxr import Usd, UsdGeom

    prims = list(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
    parent = body.GetParent()
    while parent and not parent.IsPseudoRoot():
        prims.append(parent)
        parent = parent.GetParent()
    for prim in prims:
        xform = UsdGeom.Xformable(prim)
        if not xform:
            continue
        if prim.GetPath().HasPrefix(body.GetPath()) and xform.GetResetXformStack():
            return f"{prim.GetPath()} uses resetXformStack"
        if any(attr.GetName().startswith("xformOp:") and attr.GetNumTimeSamples() for attr in prim.GetAttributes()):
            return f"{prim.GetPath()}'s transform has time samples"
    return ""


def _reference_guard(body, target):
    """Reject references and incoming connections affected by the rewrite."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = body.GetStage()
    affected = {body.GetPath(), target.GetPath()}
    prim = target.GetParent()
    while prim and prim != body:
        affected.add(prim.GetPath())
        prim = prim.GetParent()
    # Hoisting also moves otherwise-identity sibling frames. They are absent
    # from `moved`, but a relationship to one makes its pose observable.
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if prim.GetPath().HasPrefix(target.GetPath()):
            continue
        if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader):
            continue
        if UsdGeom.Xformable(prim):
            affected.add(prim.GetPath())
    for prim in (body, target):
        affected.add(prim.GetPath().AppendProperty("xformOpOrder"))
        affected.update(op.GetAttr().GetPath() for op in UsdGeom.Xformable(prim).GetOrderedXformOps())
    # _write_local authors all three plain body ops, even if an op was not
    # previously present in xformOpOrder (or did not exist at all).
    affected.update(body.GetPath().AppendProperty("xformOp:" + name) for name in ("translate", "orient", "scale"))

    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            if any(body.GetPath() in rel.GetTargets() for rel in (joint.GetBody0Rel(), joint.GetBody1Rel())):
                return f"{prim.GetPath()} joint references {body.GetPath()}"
        for prop in prim.GetProperties():
            if isinstance(prop, Usd.Relationship):
                targets = prop.GetTargets()
            elif isinstance(prop, Usd.Attribute):
                targets = prop.GetConnections()
                if targets and prop.GetPath() in affected:
                    return f"{prop.GetPath()} has a connection referencing {targets[0]}, cannot rewrite"
            else:
                continue
            for path in targets:
                if path in affected:
                    return f"{prop.GetPath()} references {path}, which will move or be rewritten"
    return ""


def _hoistable(body, cache):
    """The prim whose offset can move onto ``body``, or ``(None, reason)``."""
    from pxr import Usd, UsdGeom, UsdPhysics

    for name in _INERTIAL:
        attr = body.GetAttribute(name)
        if attr and attr.HasAuthoredValue():
            return None, f"{body.GetPath()} explicitly sets {name}"
    if body.IsA(UsdGeom.Gprim) or body.HasAPI(UsdPhysics.CollisionAPI):
        return None, f"{body.GetPath()} is outside the offset subtree and has its own geometry or collider"
    moved = [
        p
        for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
        if p != body and UsdGeom.Xformable(p) and not _is_identity(_local(p, cache))
    ]
    if not moved:
        return None, ""
    why = _transform_guard(body)
    if why:
        return None, why
    if len(moved) > 1:
        # This also covers "a prim BETWEEN the root and the offset carries its
        # own transform", which would stop the offset commuting up to the root:
        # such a prim is itself one of `moved`, so the count catches it. An
        # extra walk up the chain was unreachable code.
        return None, f"{body.GetPath()}: body has {len(moved)} offsets inside it, cannot normalize to one"
    target = moved[0]
    # Nothing whose placement matters may be left behind when the root moves.
    # Materials and shaders are exempt: a physical material carries friction
    # and density, which do not depend on where the prim sits, and moving a
    # Material prim changes nothing at all. What does depend on placement is
    # geometry, colliders, and any nested rigid body (whose pose is recorded
    # separately, so it must not silently ride along).
    from pxr import UsdShade

    for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if p == body or p.GetPath().HasPrefix(target.GetPath()):
            continue
        if p.IsA(UsdShade.Material) or p.IsA(UsdShade.Shader):
            continue
        if p.IsA(UsdGeom.Gprim) or p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI):
            return None, (
                f"{p.GetPath()} is outside the offset subtree; moving with the " f"rigid-body root would misplace it"
            )
    why = _reference_guard(body, target)
    if why:
        return None, why
    return target, ""


def _precisions(prim):
    """``(translate, orient, scale)`` precisions to use when rewriting ``prim``."""
    from pxr import Sdf

    return (
        _precision(prim, "xformOp:translate", Sdf.ValueTypeNames.Float3),
        _precision(prim, "xformOp:orient", Sdf.ValueTypeNames.Quatf),
        _precision(prim, "xformOp:scale", Sdf.ValueTypeNames.Float3),
    )


def _write_local(xformable, new_local, precisions):
    """Author translate/orient/scale on ``xformable`` to reproduce ``new_local``.

    Plain ops in the conventional order -- NOT a single ``xformOp:transform``,
    and no op suffixes. The render side writes poses by setting
    ``xformOp:translate`` / ``xformOp:orient`` on these prims by name; a matrix
    op or a suffixed op silently swallows those writes.
    """
    from pxr import Gf, UsdGeom

    t = Gf.Transform(new_local)
    q = t.GetRotation().GetQuat()
    pt, po, ps = precisions
    xformable.AddTranslateOp(pt).Set(t.GetTranslation())
    if po == UsdGeom.XformOp.PrecisionFloat:
        xformable.AddOrientOp(po).Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*q.GetImaginary())))
    else:
        xformable.AddOrientOp(po).Set(Gf.Quatd(float(q.GetReal()), Gf.Vec3d(*q.GetImaginary())))
    xformable.AddScaleOp(ps).Set(t.GetScale())


def _precision_loss(body, new_local, tol=1e-9):
    """Why ``new_local`` cannot be written onto ``body`` exactly, or ``""``.

    The new transform goes back into ``xformOp:translate`` / ``orient`` /
    ``scale``, and USD will not let a stronger layer widen an op the asset
    already declared at single precision. Where the offset is a pure
    translation -- pour_workpiece's whole case -- the rotation coming out is
    the asset's own authored value and survives the round trip. Where the
    offset ROTATES, the result composes two rotations and generally has no
    float32 representation: bimanual_chip_handover lands ~1e-7 away.

    1e-7 does not matter physically. It matters here because the superset slot
    signature rounds at the 6th decimal, so the same layout can fall on either
    side of it and be split into two variants -- the rewrite would cost slots
    instead of saving them.

    Decided by actually writing the values at those precisions into a scratch
    prim and reading the composed local transform back, rather than by
    modelling the rounding: the same ``_write_local`` runs here and in
    ``_author``, so this cannot conclude something the rewrite then contradicts.
    """
    from pxr import Gf, Usd, UsdGeom

    tmp = Usd.Stage.CreateInMemory()
    probe = UsdGeom.Xform.Define(tmp, "/probe")
    _write_local(probe, new_local, _precisions(body))
    if Gf.IsClose(probe.GetLocalTransformation(), new_local, tol):
        return ""
    return (
        f"{body.GetPath()}: the pose hoisting would need to rewrite doesn't fit in the precision the asset "
        f"declares (offset has rotation, and xformOp:orient is single precision?), there would be ~1e-7 error"
    )


def _plan(scene, report):
    """``[(body_path, target_path, offset, new_body_local)]`` for one scene.

    Both matrices are kept: the rewrite needs the body's new local transform,
    and the verification needs the offset on its own to state what the body
    root is allowed to move BY.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(scene), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ValueError(f"cannot open {scene}")
    cache = UsdGeom.XformCache()
    out = []
    objects = stage.GetPrimAtPath("/World/Objects")
    if not objects:
        # Not every scene under llm_task lays its foreground out this way, and
        # a bulk run over the benchmark must not lose its results to one of
        # them: GetChildren() on a null prim raises "Accessed invalid null prim".
        report.append(f"    skipping {scene}: no /World/Objects")
        return out
    for obj in objects.GetChildren():
        for body in Usd.PrimRange(obj, Usd.TraverseInstanceProxies()):
            if not (
                body.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get()
            ):
                continue
            target, why = _hoistable(body, cache)
            if why:
                report.append(f"    skipping {why}")
            if target is None:
                continue
            offset = _local(target, cache)
            new_local = offset * _local(body, cache)
            why = _precision_loss(body, new_local)
            if why:
                report.append(f"    skipping {why}")
                continue
            out.append((body.GetPath(), target.GetPath(), offset, new_local))
    return out


def _precision(prim, op_name, single_type):
    """Match the precision the asset already declares for ``op_name``.

    USD refuses to add an op at a precision that contradicts the existing
    typeName, and a consumer reading e.g. ``quatf xformOp:orient`` should not
    find a ``quatd`` there after this rewrite. Default to double where the op
    does not exist yet: these values are COMPUTED per instance, and the
    signature that decides slot sharing rounds at the 6th decimal, so float32
    noise around unity is worth avoiding where nothing forces it.
    """
    from pxr import UsdGeom

    attr = prim.GetAttribute(op_name)
    if attr and attr.GetTypeName() == single_type:
        return UsdGeom.XformOp.PrecisionFloat
    return UsdGeom.XformOp.PrecisionDouble


def _author(scene, plan, out_path):
    """Write ``scene`` with each plan entry applied, to ``out_path``."""
    from pxr import Gf, Sdf, Usd, UsdGeom

    shutil.copyfile(scene, out_path)
    stage = Usd.Stage.Open(str(out_path), load=Usd.Stage.LoadAll)
    layer = stage.GetRootLayer()
    for body_path, target_path, _offset, new_local in plan:
        for path in (body_path, target_path):
            spec = layer.GetPrimAtPath(path)
            if spec is None:
                # The scene may author the offset without an explicit spec for
                # the body; Usd.Stage.OverridePrim creates one in this layer.
                stage.OverridePrim(path)
        for op in UsdGeom.Xformable(stage.GetPrimAtPath(target_path)).GetOrderedXformOps():
            attr = layer.GetAttributeAtPath(op.GetAttr().GetPath())
            if attr is not None:
                layer.GetPrimAtPath(target_path).RemoveProperty(attr)
        UsdGeom.Xformable(stage.GetPrimAtPath(target_path)).SetXformOpOrder([])
        body = UsdGeom.Xformable(stage.GetPrimAtPath(body_path))
        body.ClearXformOpOrder()
        for op in list(body.GetOrderedXformOps()):
            spec = layer.GetAttributeAtPath(op.GetAttr().GetPath())
            if spec is not None:
                layer.GetPrimAtPath(body_path).RemoveProperty(spec)
        # Precision follows whatever the asset already declares for each op, so
        # the value type never changes underneath a consumer. `_plan` has
        # already refused any body where that costs accuracy.
        _write_local(body, new_local, _precisions(stage.GetPrimAtPath(body_path)))
    layer.Save()
    return out_path


def _verify(before, after, plan):
    """Nothing observable may move, and each root must move by exactly its offset.

    "Observable" is geometry, colliders and rigid bodies -- what physics and
    rendering actually place. A bare grouping Xform between the body root and
    the collider (``entity/Aligned`` here) DOES move, by construction: the
    offset leaves it and lands on its ancestor. Nothing reads its pose, and its
    own child ends up exactly where it was, which is what the first check below
    proves prim by prim.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    a = Usd.Stage.Open(str(before), load=Usd.Stage.LoadAll)
    b = Usd.Stage.Open(str(after), load=Usd.Stage.LoadAll)
    ca, cb = UsdGeom.XformCache(), UsdGeom.XformCache()
    roots = {p: off for p, _, off, _n in plan}
    bad, checked = [], 0
    for prim in Usd.PrimRange.Stage(a, Usd.TraverseInstanceProxies()):
        other = b.GetPrimAtPath(prim.GetPath())
        if not other:
            bad.append(f"{prim.GetPath()} disappeared after rewriting")
            continue
        if not UsdGeom.Xformable(prim):
            continue
        wa = ca.GetLocalToWorldTransform(prim)
        wb = cb.GetLocalToWorldTransform(other)
        if prim.GetPath() in roots:
            # local' = offset * local  =>  world' = offset * world
            if not Gf.IsClose(roots[prim.GetPath()] * wa, wb, 1e-9):
                bad.append(f"{prim.GetPath()} rigid-body root moved by a different amount than planned")
            checked += 1
            continue
        if not (
            prim.IsA(UsdGeom.Gprim) or prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            continue
        checked += 1
        if not Gf.IsClose(wa, wb, 1e-9):
            bad.append(f"{prim.GetPath()} world transform was changed")
    if checked < len(plan) * 2:
        bad.append(f"only checked {checked} prims, fewer than the {len(plan)} edits should have touched")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--apply", action="store_true", help="rewrite the benchmark data (report only by default)")
    ap.add_argument("--bench-root", default=str(BENCH))
    args = ap.parse_args(argv)

    total, failed = 0, 0
    for task in args.tasks:
        base = Path(args.bench_root) / task
        for inst in sorted((p for p in base.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
            scene = inst / "scene.usda"
            report: list = []
            plan = _plan(scene, report)
            for line in report:
                print(line)
            if not plan:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "scene.usda"
                _author(scene, plan, out)
                bad = _verify(scene, out, plan)
                if bad:
                    failed += 1
                    print(f"  ❌ {task}/{inst.name}: {len(bad)} mismatches")
                    for line in bad[:3]:
                        print(f"       {line}")
                    continue
                total += len(plan)
                print(
                    f"  ✅ {task}/{inst.name}: hoisted {len(plan)} offsets"
                    + ("(written)" if args.apply else "(not written, add --apply)")
                )
                if args.apply:
                    shutil.copyfile(out, scene)
    print(f"\n{total} offsets total can be hoisted, {failed} instances failed verification")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
