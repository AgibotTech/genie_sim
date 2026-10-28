# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Run with a USD-enabled interpreter (or inside the Isaac app smoke test)."""

import pytest


@pytest.fixture(scope="module", autouse=True)
def _usd_runtime():
    pytest.importorskip("pxr")
    global Sdf, Usd, UsdGeom, UsdPhysics
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics


from genie_sim_engine_isaacsim.assemble.physics_variant import select_physics_variant, validate_urdf_joints


@pytest.fixture
def robot(tmp_path):
    common = Usd.Stage.CreateNew(str(tmp_path / "physics.usda"))
    common.SetDefaultPrim(UsdGeom.Xform.Define(common, "/robot").GetPrim())
    body = UsdGeom.Xform.Define(common, "/robot/Geometry/base_link/base_link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.RevoluteJoint.Define(common, "/robot/Physics/idx01_arm_l_joint1")
    common.GetRootLayer().Save()
    physx = Sdf.Layer.CreateNew(str(tmp_path / "physx.usda"))
    physx.subLayerPaths = ["./physics.usda"]
    physx.Save()
    stage = Usd.Stage.CreateNew(str(tmp_path / "robot.usda"))
    root = UsdGeom.Xform.Define(stage, "/robot").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stage, "/robot/Geometry/base_link")
    variants = root.GetVariantSets().AddVariantSet("Physics")
    for name in ("none", "physx", "mujoco"):
        variants.AddVariant(name)
        variants.SetVariantSelection(name)
        if name != "none":
            with variants.GetVariantEditContext():
                root.GetPayloads().AddPayload("./physx.usda", "/robot")
    variants.ClearVariantSelection()
    stage.GetRootLayer().Save()
    return stage


def _urdf(tmp_path, joints):
    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot name="robot">' + "".join(f'<joint name="{j}" type="revolute"/>' for j in joints) + "</robot>"
    )
    return path


@pytest.mark.parametrize("previous", [None, "none", "mujoco"])
def test_unselected_or_inactive_variant_is_composed_and_persisted(robot, previous, tmp_path):
    variant = robot.GetDefaultPrim().GetVariantSet("Physics")
    if previous:
        variant.SetVariantSelection(previous)
    assert select_physics_variant(robot) == "physx"
    assert validate_urdf_joints(robot, _urdf(tmp_path, ["arm_l_joint1"])) == 1
    robot.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(tmp_path / "robot.usda"))
    assert reopened.GetDefaultPrim().GetVariantSet("Physics").GetVariantSelection() == "physx"
    assert reopened.GetPrimAtPath("/robot/Geometry/base_link/base_link").HasAPI(UsdPhysics.RigidBodyAPI)


def test_runtime_reference_selection_does_not_change_source(robot, tmp_path):
    scene = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(scene, "/genie").GetPrim()
    root.GetReferences().AddReference(str(tmp_path / "robot.usda"))
    select_physics_variant(scene, "/genie")
    assert scene.GetPrimAtPath("/genie/Physics/idx01_arm_l_joint1").IsA(UsdPhysics.RevoluteJoint)
    assert robot.GetDefaultPrim().GetVariantSet("Physics").GetVariantSelection() == ""


def test_flat_legacy_asset_needs_no_variant():
    stage = Usd.Stage.CreateInMemory()
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/robot").GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(stage.GetDefaultPrim())
    assert select_physics_variant(stage) is None


@pytest.mark.parametrize("previous", [None, "None"])
def test_legacy_physx_variant_is_selected_without_editing_source(tmp_path, previous):
    path = tmp_path / "legacy_robot.usda"
    source = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(source, "/robot").GetPrim()
    source.SetDefaultPrim(root)
    variants = root.GetVariantSets().AddVariantSet("Physics")
    variants.AddVariant("None")
    variants.AddVariant("PhysX")
    variants.SetVariantSelection("PhysX")
    with variants.GetVariantEditContext():
        body = UsdGeom.Xform.Define(source, "/robot/base_link").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
    if previous is None:
        variants.ClearVariantSelection()
    else:
        variants.SetVariantSelection(previous)
    source.GetRootLayer().Save()
    original = path.read_bytes()

    scene = Usd.Stage.CreateInMemory()
    scene_root = UsdGeom.Xform.Define(scene, "/genie").GetPrim()
    scene_root.GetReferences().AddReference(str(path))
    assert select_physics_variant(scene, "/genie") == "PhysX"
    assert scene.GetPrimAtPath("/genie/base_link").HasAPI(UsdPhysics.RigidBodyAPI)
    assert variants.GetVariantSelection() == (previous or "")
    assert path.read_bytes() == original


def test_unknown_variant_set_fails_instead_of_guessing():
    # Use an asset with only an unrelated physics variant.
    stage = Usd.Stage.CreateInMemory()
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/robot").GetPrim())
    stage.GetDefaultPrim().GetVariantSets().AddVariantSet("Physics").AddVariant("other")
    with pytest.raises(RuntimeError, match="requires 'physx' or 'PhysX'"):
        select_physics_variant(stage)


@pytest.mark.parametrize("variant_name", ["physx", "PhysX"])
def test_missing_payload_is_an_error(variant_name):
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/robot").GetPrim()
    stage.SetDefaultPrim(root)
    variants = root.GetVariantSets().AddVariantSet("Physics")
    variants.AddVariant(variant_name)
    with pytest.raises(RuntimeError, match="zero rigid bodies"):
        select_physics_variant(stage)


def test_missing_joint_inventory_is_an_error(robot, tmp_path):
    select_physics_variant(robot)
    with pytest.raises(RuntimeError, match="missing=.*arm_l_joint2"):
        validate_urdf_joints(robot, _urdf(tmp_path, ["arm_l_joint1", "arm_l_joint2"]))


def test_duplicate_joint_names_are_not_counted_as_success(robot, tmp_path):
    select_physics_variant(robot)
    UsdPhysics.RevoluteJoint.Define(robot, "/robot/duplicate/idx02_arm_l_joint1")
    with pytest.raises(RuntimeError, match="unexpected=.*arm_l_joint1"):
        validate_urdf_joints(robot, _urdf(tmp_path, ["arm_l_joint1"]))


def test_indexed_urdf_names_match_indexed_usd_names(robot, tmp_path):
    select_physics_variant(robot)
    assert validate_urdf_joints(robot, _urdf(tmp_path, ["idx01_arm_l_joint1"])) == 1


_ANNOTATION_URDF = """<robot name="test">
<link name="gripper_base_link"/><link name="gripper_pad"/>
<joint name="gripper_joint" type="revolute">
  <parent link="gripper_base_link"/><child link="gripper_pad"/>
</joint></robot>"""


def test_missing_derived_pad_is_a_hard_error(robot):
    from genie_sim_engine_isaacsim.assemble._robot_impl import _author_gripper_annotations

    select_physics_variant(robot)
    with pytest.raises(RuntimeError, match="pads=0/1"):
        _author_gripper_annotations(robot, _ANNOTATION_URDF)


def test_complete_derived_annotation_inventory_is_authored(robot):
    from genie_sim_engine_isaacsim.assemble._robot_impl import _author_gripper_annotations

    select_physics_variant(robot)
    pad = UsdGeom.Xform.Define(robot, "/robot/Geometry/gripper_pad").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(pad)
    assert _author_gripper_annotations(robot, _ANNOTATION_URDF) == {"pads_authored": 1, "couplers_authored": 0}
    assert pad.GetAttribute("newton:isGripperPad").Get() is True


@pytest.mark.parametrize("selection", ["", "none", "mujoco", "physx"])
def test_manifest_maps_link_owners_instead_of_same_named_meshes(tmp_path, selection):
    from genie_sim_engine_isaacsim.assemble.urdf_to_usd_stage import _extract_robot_body_paths

    physics = Usd.Stage.CreateNew(str(tmp_path / "physics.usda"))
    physics.SetDefaultPrim(UsdGeom.Xform.Define(physics, "/robot").GetPrim())
    for name, path in {
        "base_link": "/robot/Geometry/base_link",
        "body_link1": "/robot/Geometry/base_link/body_link1",
    }.items():
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(physics, path).GetPrim())
    physics.GetRootLayer().Save()

    robot_path = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(robot_path))
    root = UsdGeom.Xform.Define(stage, "/robot").GetPrim()
    stage.SetDefaultPrim(root)
    # Visual wrappers collide with both the root name and articulated links.
    for path in (
        "/robot/Geometry/base_link/base_link",
        "/robot/Geometry/base_link/body_link1/body_link1",
        "/robot/Geometry/base_link/arm_l/gripper/base_link",
        "/robot/Geometry/base_link/arm_r/gripper/base_link",
    ):
        UsdGeom.Xform.Define(stage, path)
    variants = root.GetVariantSets().AddVariantSet("Physics")
    for name in ("none", "mujoco", "physx"):
        variants.AddVariant(name)
        variants.SetVariantSelection(name)
        if name != "none":
            with variants.GetVariantEditContext():
                root.GetPayloads().AddPayload("./physics.usda", "/robot")
    variants.ClearVariantSelection()
    if selection:
        variants.SetVariantSelection(selection)
    stage.GetRootLayer().Save()
    original = robot_path.read_bytes()
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="robot"><link name="base_link"/><link name="body_link1"/></robot>')

    assert _extract_robot_body_paths(urdf_path=urdf, robot_usda=robot_path, robot_prefix="genie") == {
        "base_link": "/genie/Geometry/base_link",
        "body_link1": "/genie/Geometry/base_link/body_link1",
    }
    assert robot_path.read_bytes() == original
    assert variants.GetVariantSelection() == selection


def test_flat_manifest_body_mapping_is_preserved(tmp_path):
    from genie_sim_engine_isaacsim.assemble.urdf_to_usd_stage import _extract_robot_body_paths

    path = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/robot").GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/robot/base_link").GetPrim())
    UsdGeom.Xform.Define(stage, "/robot/base_link/base_link")
    stage.GetRootLayer().Save()
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="robot"><link name="base_link"/></robot>')
    assert _extract_robot_body_paths(urdf_path=urdf, robot_usda=path) == {"base_link": "/genie/base_link"}
