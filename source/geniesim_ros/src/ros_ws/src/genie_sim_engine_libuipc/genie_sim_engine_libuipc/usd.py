# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Package-local USD base-pose and weld policies, extracted from the SDK bridge."""

from __future__ import annotations
import math
import numpy as np
from pathlib import Path

WELD_ANCHOR_ANNOT = "libuipc:weldAnchorIsWorld"


def topmost_rigid_body_paths(stage, root_path) -> list[str]:
    """Find URDF roots, excluding rigid descendants of another robot body."""
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path) if stage is not None else None
    if not root or not root.IsValid():
        return []
    paths = []
    for prim in Usd.PrimRange(root, Usd.PrimAllPrimsPredicate):
        if prim == root or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        ancestor = prim.GetParent()
        topmost = True
        while ancestor and ancestor.IsValid() and ancestor.GetPath() != root_path:
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                topmost = False
                break
            ancestor = ancestor.GetParent()
        if topmost:
            paths.append(str(prim.GetPath()))
    return paths


def classify_world_weld(stage, prim, robot_root_path, urdf_root_link_paths):
    """Classify only a world-side fixed joint whose child is a true robot root.

    A non-rigid fixed frame inside an arm is still part of that rigid arm;
    its joint to a nested hand must never be repaired as a world anchor.
    Newton recovers identity world frames from the child, but a nonidentity
    localPose0 is interpreted as an anchor and can discard/double placement.
    """
    from pxr import UsdPhysics

    if not prim.IsA(UsdPhysics.FixedJoint):
        return None
    joint = UsdPhysics.FixedJoint(prim)
    body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
    body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
    if len(body0) > 1 or len(body1) != 1 or body1[0] not in urdf_root_link_paths:
        return None
    parent = stage.GetPrimAtPath(body0[0]) if body0 else None
    ancestor = parent
    while ancestor and ancestor.IsValid():
        # USD endpoints may name a fixed frame rather than its rigid owner,
        # including on flat assets whose arm and hand are sibling bodies.
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            return None
        if ancestor.GetPath() == robot_root_path:
            break
        ancestor = ancestor.GetParent()
    world_side = "empty" if not body0 else "non-rigid-prim" if parent and parent.IsValid() else "dangling"
    p0 = joint.GetLocalPos0Attr().Get()
    r0 = joint.GetLocalRot0Attr().Get()
    identity = (p0 is None or all(abs(float(v)) <= 1e-6 for v in p0)) and (
        r0 is None or abs(float(r0.GetReal())) >= 1.0 - 1e-6
    )
    annotation = prim.GetAttribute(WELD_ANCHOR_ANNOT)
    anchored = bool(annotation and annotation.Get())
    return {"body0": body0, "body1": body1, "world_side": world_side, "misplaces": not identity and not anchored}


class USDPolicies:

    def _repair_root_weld_anchor(self) -> int:
        """Preserve the composed root-body pose in pinned imports, session only."""
        if self._stage is None or not self._pin_base_to_world:
            return 0
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        root_path = Sdf.Path(f"/{self._robot_prefix_str}")
        root = self._stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            return 0
        roots = topmost_rigid_body_paths(self._stage, root_path)
        repaired = 0
        for prim in Usd.PrimRange(root):
            record = classify_world_weld(self._stage, prim, root_path, roots)
            if record is None or not record["misplaces"]:
                continue
            child = self._stage.GetPrimAtPath(record["body1"][0])
            transform = Gf.Transform(UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            translation, rotation = transform.GetTranslation(), transform.GetRotation().GetQuat()
            with Usd.EditContext(self._stage, self._stage.GetSessionLayer()):
                joint = UsdPhysics.Joint(prim)
                joint.GetBody0Rel().SetTargets([])
                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*translation))
                joint.CreateLocalRot0Attr().Set(Gf.Quatf(rotation))
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))
                prim.CreateAttribute(WELD_ANCHOR_ANNOT, Sdf.ValueTypeNames.Bool).Set(True)
            repaired += 1
            self._logger.info(
                f"[libuipc] root-weld repair: {prim.GetPath()} anchored at {record['body1'][0]}'s world pose"
            )
        return repaired

    def _apply_init_base_pose(self) -> None:
        if self._stage is None:
            return
        base_pose = ((self._scene_cfg or {}).get("robot") or {}).get("init_base_pose")
        if not isinstance(base_pose, dict):
            return
        try:
            x = float(base_pose.get("x", 0.0))
            y = float(base_pose.get("y", 0.0))
            z = float(base_pose.get("z", 0.0))
            theta = float(base_pose.get("theta", 0.0))
        except (TypeError, ValueError) as exc:
            self._logger.warn(f"[libuipc] init_base_pose: numeric parse failed ({exc!r}); skipping")
            return
        try:
            import math
            from pxr import Gf, Sdf, UsdGeom
        except Exception as exc:
            self._logger.warn(f"[libuipc] init_base_pose: pxr import failed ({exc!r}); skipping")
            return
        root_path = Sdf.Path(f"/{self._robot_prefix_str}")
        root_prim = self._stage.GetPrimAtPath(root_path)
        if not root_prim or not root_prim.IsValid():
            self._logger.warn(f"[libuipc] init_base_pose: robot root {root_path} not on stage; skipping")
            return
        xformable = UsdGeom.Xformable(root_prim)
        if not xformable:
            self._logger.warn(f"[libuipc] init_base_pose: {root_path} is not Xformable; skipping")
            return
        t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "init_base")
        t_op.Set(Gf.Vec3d(x, y, z))
        half = 0.5 * theta
        quat = Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, 0.0, math.sin(half)))
        o_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "init_base")
        o_op.Set(quat)
        xformable.SetXformOpOrder([t_op, o_op])
        self._logger.info(
            f"[libuipc] init_base_pose applied to {root_path}: xyz=({x:.4f}, {y:.4f}, {z:.4f}) yaw={theta:.4f} rad"
        )

    def _deactivate_root_joint(self) -> int:
        if self._stage is None:
            return 0
        try:
            from pxr import Sdf, Usd, UsdPhysics
        except Exception as exc:
            self._logger.warn(
                f"[libuipc] pin_base_to_world=False: pxr import failed ({exc}); cannot locate URDF world-weld joint."
            )
            return 0
        robot_root_path = Sdf.Path(f"/{self._robot_prefix_str}")
        root_prim = self._stage.GetPrimAtPath(robot_root_path)
        if not root_prim or not root_prim.IsValid():
            self._logger.warn(
                f"[libuipc] pin_base_to_world=False: robot root prim {robot_root_path} not found; cannot identify world-weld joint."
            )
            return 0
        urdf_root_link_paths = []
        for desc in Usd.PrimRange(root_prim):
            if desc == root_prim:
                continue
            if not desc.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            ancestor = desc.GetParent()
            is_topmost = True
            while ancestor and ancestor.IsValid() and (ancestor.GetPath() != robot_root_path):
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    is_topmost = False
                    break
                ancestor = ancestor.GetParent()
            if is_topmost:
                urdf_root_link_paths.append(desc.GetPath())
        if not urdf_root_link_paths:
            self._logger.warn(
                f"[libuipc] pin_base_to_world=False: no RigidBodyAPI descendant under {robot_root_path}; cannot identify URDF root link. Base will be free by Newton's default in that case."
            )
            return 0
        found = 0
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdPhysics.FixedJoint):
                continue
            joint = UsdPhysics.FixedJoint(prim)
            b0 = joint.GetBody0Rel().GetTargets()
            b1 = joint.GetBody1Rel().GetTargets()
            if not b0 or not b1:
                continue
            if b0[0] != robot_root_path:
                continue
            if b1[0] not in urdf_root_link_paths:
                continue
            prim.SetActive(False)
            found += 1
            self._logger.info(
                f"[libuipc] pin_base_to_world=False → deactivated world-weld {prim.GetPath()} (body0={b0[0]}, body1={b1[0]}). Newton will add a FREE base joint; pair with the plain rigid-substep regime (real gravity + contacts)."
            )
        if found == 0:
            self._logger.warn(
                f"[libuipc] pin_base_to_world=False: no world-weld FixedJoint detected under {robot_root_path} (body0=root, body1=URDF root link). Likely a ghost-root URDF — see diagnose_urdf.py. Base will be free by Newton's default."
            )
        return found

    def _replace_joints_with_fixed(self, *, name_substring: str, label: str) -> int:
        if self._stage is None:
            return 0
        try:
            from pxr import Gf, Sdf, UsdPhysics
        except Exception as exc:
            self._logger.warn(f"[libuipc] {label}: USD import failed: {exc}")
            return 0
        DEG2RAD = math.pi / 180.0
        init_rad_by_name: Dict[str, float] = {}
        ip = getattr(self, "_init_joint_pos", None) or {}
        if isinstance(ip, dict):
            for nm, raw in ip.items():
                try:
                    v = float(raw.value if hasattr(raw, "value") else raw)
                except (TypeError, ValueError):
                    continue
                init_rad_by_name[nm] = v * DEG2RAD
        _AXIS_VEC = {"X": Gf.Vec3f(1.0, 0.0, 0.0), "Y": Gf.Vec3f(0.0, 1.0, 0.0), "Z": Gf.Vec3f(0.0, 0.0, 1.0)}
        n = 0
        n_baked = 0
        sample = []
        baked_sample = []
        try:
            candidates = []
            for prim in self._stage.Traverse():
                if not (prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.Joint)):
                    continue
                if name_substring not in prim.GetName():
                    continue
                if prim.GetTypeName() == "PhysicsFixedJoint":
                    continue
                if not prim.IsActive():
                    continue
                candidates.append(prim)
            for prim in candidates:
                joint_name = prim.GetName()
                joint = UsdPhysics.Joint(prim)
                body0_targets = list(joint.GetBody0Rel().GetTargets())
                body1_targets = list(joint.GetBody1Rel().GetTargets())
                if not body0_targets or not body1_targets:
                    prim.SetActive(False)
                    continue
                lp0 = joint.GetLocalPos0Attr().Get()
                lr0 = joint.GetLocalRot0Attr().Get()
                lp1 = joint.GetLocalPos1Attr().Get()
                lr1 = joint.GetLocalRot1Attr().Get()
                init_rad = init_rad_by_name.get(joint_name, 0.0)
                if init_rad != 0.0 and prim.GetTypeName() == "PhysicsRevoluteJoint":
                    rev = UsdPhysics.RevoluteJoint(prim)
                    axis_token = rev.GetAxisAttr().Get()
                    axis = _AXIS_VEC.get(axis_token) if axis_token else None
                    if axis is not None:
                        half = init_rad * 0.5
                        bake_q = Gf.Quatf(math.cos(half), axis * math.sin(half))
                        lr0 = Gf.Quatf(lr0) * bake_q if lr0 is not None else bake_q
                        n_baked += 1
                        if len(baked_sample) < 4:
                            baked_sample.append(f"{joint_name}@{math.degrees(init_rad):+.1f}°")
                src_path = prim.GetPath()
                new_path = Sdf.Path(f"{src_path.GetParentPath().pathString}/{src_path.name}__fixed")
                new_joint = UsdPhysics.FixedJoint.Define(self._stage, new_path)
                new_joint.GetBody0Rel().SetTargets(body0_targets)
                new_joint.GetBody1Rel().SetTargets(body1_targets)
                if lp0 is not None:
                    new_joint.GetLocalPos0Attr().Set(lp0)
                if lr0 is not None:
                    new_joint.GetLocalRot0Attr().Set(lr0)
                if lp1 is not None:
                    new_joint.GetLocalPos1Attr().Set(lp1)
                if lr1 is not None:
                    new_joint.GetLocalRot1Attr().Set(lr1)
                prim.SetActive(False)
                n += 1
                self._static_joint_q[joint_name] = init_rad
                if len(sample) < 4:
                    sample.append(src_path.name)
            if n > 0:
                msg = f"[libuipc] {label}=True: replaced {n} joint(s) containing {name_substring!r} with UsdPhysics.FixedJoint (bodies retained for TF; {n} DOF(s) removed from M); sample={sample}"
                if n_baked > 0:
                    msg += f"; baked init pose into {n_baked} FixedJoint local frame(s) (sample={baked_sample}) — published synthetically via /joint_states"
                self._logger.info(msg)
            else:
                self._logger.info(
                    f"[libuipc] {label}=True: no active joints containing {name_substring!r} on the stage (no-op)"
                )
        except Exception as exc:
            self._logger.warn(f"[libuipc] {label} policy failed: {exc}")
        return n


def _strip_disabled_physics_schemas(stage) -> None:
    """De-apply ``PhysicsRigidBodyAPI`` (and ``PhysicsMassAPI``) on prims
    whose composed ``physics:rigidBodyEnabled`` is ``False``, and clear the
    ``physics:approximation`` opinion on disabled colliders so Newton's GL
    viewer renders their actual mesh instead of the convex-decomposition
    proxy.

    Why this is necessary:

    1.  Newton's ``ModelBuilder.add_usd`` visual-loader pass
        (``newton/_src/utils/import_usd.py:_load_visual_shapes_impl``
        line 788) early-returns on any prim that has
        ``UsdPhysics.RigidBodyAPI`` applied, *regardless* of whether the
        rigid body is enabled.  Many of our scene props (e.g. the
        ``GDK/demo.usda`` shelves) compose ``RigidBodyAPI`` from the leaf
        USD and turn it off in the scene's per-instance ``over``;
        without the strip the visual is missing entirely.

    2.  When a mesh has ``PhysicsCollisionAPI`` with
        ``collisionEnabled=False`` *and* an ``physics:approximation``
        opinion (``convexDecomposition`` / ``convexHull`` / etc.), Newton
        still adds the shape to the builder (``COLLIDE_SHAPES`` flag
        cleared) and the GL viewer renders the *approximation* — which
        looks like the convex hull rather than the underlying mesh.
        Clearing the approximation token forces the viewer to render the
        mesh as authored.

    We deliberately do NOT strip ``PhysicsCollisionAPI`` itself: static
    props (like ``market_00``) that are NOT under a ``RigidBodyAPI``
    parent rely on the collision-shape pipeline to register their mesh
    as a (non-colliding) builder shape so the GL viewer renders it at
    all — Newton's visual-loader pass is only invoked from rigid-body
    parsing, so removing the schema would make those static props
    disappear from the viewport.

    The fix only touches the live stage in this process; on-disk USDs
    are unaffected, and other consumers (IsaacSim, OVRtx) read the
    canonical layers as authored.
    """
    try:
        from pxr import Sdf, UsdPhysics  # noqa: PLC0415
    except ImportError:
        return

    _RB_SCHEMAS = ("PhysicsRigidBodyAPI", "PhysicsMassAPI")
    stripped_rb = cleared_approx = 0
    for prim in stage.Traverse():
        applied = list(prim.GetAppliedSchemas())
        remove: list[str] = []

        rba = UsdPhysics.RigidBodyAPI(prim)
        if rba and rba.GetRigidBodyEnabledAttr().Get() is False:
            remove.extend(s for s in _RB_SCHEMAS if s in applied)

        col = UsdPhysics.CollisionAPI(prim)
        if col and col.GetCollisionEnabledAttr().Get() is False:
            approx_attr = prim.GetAttribute("physics:approximation")
            if approx_attr and approx_attr.HasAuthoredValue():
                approx_attr.Block()
                cleared_approx += 1

        if remove:
            cleaned = [s for s in applied if s not in remove]
            prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(cleaned))
            stripped_rb += sum(1 for s in remove if s in _RB_SCHEMAS)

    if stripped_rb or cleared_approx:
        print(
            f"[libuipc] reauthored disabled-physics opinions: "
            f"{stripped_rb} RigidBody/Mass schemas removed, "
            f"{cleared_approx} approximation tokens cleared "
            f"(restores visuals on rigidBodyEnabled=0 / collisionEnabled=0 props)",
            flush=True,
        )
