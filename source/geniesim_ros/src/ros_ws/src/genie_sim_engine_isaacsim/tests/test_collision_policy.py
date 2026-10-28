# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise collider edits through the instanced geometry used by AS3."""

import pytest

from genie_sim_engine_isaacsim.assemble._robot_impl import _apply_post_transformer_collision_policy


def test_instanced_colliders_are_editable_and_policy_survives_reopen(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    geometry = Usd.Stage.CreateNew(str(tmp_path / "geometry.usda"))
    geometry.SetDefaultPrim(UsdGeom.Xform.Define(geometry, "/geometry").GetPrim())
    mesh = UsdGeom.Mesh.Define(geometry, "/geometry/mesh").GetPrim()
    UsdPhysics.CollisionAPI.Apply(mesh)
    UsdPhysics.MeshCollisionAPI.Apply(mesh)
    geometry.GetRootLayer().Save()
    stage = Usd.Stage.CreateNew(str(tmp_path / "robot.usda"))
    root = UsdGeom.Xform.Define(stage, "/robot").GetPrim()
    stage.SetDefaultPrim(root)
    for link in ("gripper_l_base_link", "wheel_link", "arm_l_link1"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/robot/{link}").GetPrim())
        wrapper = UsdGeom.Xform.Define(stage, f"/robot/{link}/collider").GetPrim()
        wrapper.GetReferences().AddReference(str(tmp_path / "geometry.usda"))
        wrapper.SetInstanceable(True)
    material = UsdShade.Material.Define(stage, "/robot/Material").GetPrim()
    material.SetInstanceable(True)
    stage.GetRootLayer().Save()

    counts = _apply_post_transformer_collision_policy(str(tmp_path / "robot.usda"))
    assert counts["gripper_authored"] == 1
    assert counts["wheel_convex_hull"] == 1
    assert counts["stripped"] == 1
    assert counts["skipped_no_visual"] == 0
    stage.Reload()
    for link, approximation in (("gripper_l_base_link", "sdf"), ("wheel_link", "convexHull")):
        mesh = stage.GetPrimAtPath(f"/robot/{link}/collider/mesh")
        assert not mesh.IsInstanceProxy()
        assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == approximation
    assert (
        not UsdPhysics.CollisionAPI(stage.GetPrimAtPath("/robot/arm_l_link1/collider/mesh"))
        .GetCollisionEnabledAttr()
        .Get()
    )
    assert stage.GetPrimAtPath("/robot/Material").IsInstanceable()


def test_wuji_palm_colliders_remain_enabled_without_enabling_arm_or_body(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    robot_path = str(tmp_path / "robot.usda")
    stage = Usd.Stage.CreateNew(robot_path)
    root = UsdGeom.Xform.Define(stage, "/robot").GetPrim()
    stage.SetDefaultPrim(root)
    palm_paths = []
    noncontact_paths = []
    for side, mesh_side in (("l", "left"), ("r", "right")):
        body = UsdGeom.Xform.Define(stage, f"/robot/Geometry/hand_{side}_base_link").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        for index in range(1, 4):
            path = f"{body.GetPath()}/{mesh_side}_palm_link_collision{index:02d}"
            collider = UsdGeom.Mesh.Define(stage, path).GetPrim()
            UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr().Set(True)
            palm_paths.append(path)
    for link in ("arm_r_link7", "body_link"):
        body = UsdGeom.Xform.Define(stage, f"/robot/Geometry/{link}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        path = f"{body.GetPath()}/collider"
        collider = UsdGeom.Mesh.Define(stage, path).GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr().Set(True)
        noncontact_paths.append(path)
    stage.GetRootLayer().Save()

    counts = _apply_post_transformer_collision_policy(robot_path)

    assert counts["gripper_by_geometry"] == 2
    assert counts["gripper_authored"] == 2
    assert counts["stripped"] == 2
    stage.Reload()
    for path in palm_paths:
        collider = stage.GetPrimAtPath(path)
        assert UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get()
        assert UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() == "sdf"
    for path in noncontact_paths:
        assert not UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get()
