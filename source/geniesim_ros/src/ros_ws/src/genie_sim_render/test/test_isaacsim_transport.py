# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise ROS graph and USD transport without starting Kit during import."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "isaacsim_render.py"


def _extract(names, namespace, class_name="IsaacSimRenderNode"):
    tree = ast.parse(SCRIPT.read_text())
    nodes = list(tree.body)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            nodes.extend(node.body)
    functions = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    # Annotations are deferred just as they are in the original script.
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SCRIPT), "exec"), namespace)
    return namespace


def test_manifest_load_preserves_zero_principal_point(tmp_path):
    from genie_sim_engine_core.schema import CameraManifest, Manifest

    scene = tmp_path / "scene.usda"
    scene.touch()
    path = tmp_path / "manifest.json"
    Manifest(
        base_path=tmp_path,
        scene_usda=scene,
        cameras=[CameraManifest(path="head/Camera", topic="/camera/rgb", width=640, height=400, cx=0.0, cy=0.0)],
    ).write(path)
    namespace = _extract(["load"], {"Path": Path, "CameraConfig": SimpleNamespace}, class_name="Manifest")
    loaded = namespace["load"].__func__(SimpleNamespace, path)
    assert loaded.cameras[0].cx == 0.0
    assert loaded.cameras[0].cy == 0.0


@pytest.mark.parametrize("depth_topic", ["", "/camera/depth"])
def test_camera_graph_preserves_manifest_depth_and_frame_id(depth_topic):
    values = {}
    edits = []
    controller = SimpleNamespace(
        Keys=SimpleNamespace(CREATE_NODES="nodes", CONNECT="edges"),
        edit=lambda path, changes: edits.append(changes),
        set=lambda path, value: values.update({path: value}),
    )
    namespace = _extract(["_build_camera_omnigraph"], {"og": SimpleNamespace(Controller=controller)})
    camera = SimpleNamespace(
        width=640,
        height=400,
        path="head/Camera",
        frame_id="head_optical",
        topic="/camera/rgb",
        depth_topic=depth_topic,
        fx=305.64,
        fy=301.2,
        cx=318.8,
        cy=202.5,
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
        k3=0.0,
    )
    namespace["_build_camera_omnigraph"]("/Graph", [camera], ["/World/Camera"], 0)
    assert values["/Graph/CamHelper0.inputs:frameId"] == "head_optical"
    assert values["/Graph/CamInfo0.inputs:frameId"] == "head_optical"
    assert values["/Graph/CamHelper0.inputs:topicName"] == "/camera/rgb/image_raw"
    assert values["/Graph/CamInfo0.inputs:k"] == [305.64, 0.0, 318.8, 0.0, 301.2, 202.5, 0.0, 0.0, 1.0]
    assert ("Time.outputs:simulationTime", "CamInfo0.inputs:timeStamp") in edits[0]["edges"]
    if depth_topic:
        assert values["/Graph/DepthHelper0.inputs:type"] == "depth"
        assert values["/Graph/DepthHelper0.inputs:topicName"] == "/camera/depth/image_raw"
        assert values["/Graph/DepthHelper0.inputs:frameId"] == "head_optical"
        assert ("CreateRP0.outputs:renderProductPath", "DepthHelper0.inputs:renderProductPath") in edits[0]["edges"]
    else:
        assert not any(name == "DepthHelper0" for name, _ in edits[0]["nodes"])


def test_scene_world_pose_does_not_compose_parent_transform_twice():
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Parent").AddTranslateOp().Set(Gf.Vec3d(10, 0, 0))
    body = UsdGeom.Xform.Define(stage, "/World/Parent/Body").GetPrim()
    namespace = _extract(["_apply_xform", "_on_tf_scene"], {"Gf": Gf, "Sdf": Sdf, "UsdGeom": UsdGeom})
    transform = SimpleNamespace(
        translation=SimpleNamespace(x=2, y=3, z=4), rotation=SimpleNamespace(w=1, x=0, y=0, z=0)
    )
    message = SimpleNamespace(transforms=[SimpleNamespace(child_frame_id="/World/Parent/Body", transform=transform)])
    node = SimpleNamespace(_get_prim=lambda path: stage.GetPrimAtPath(path))
    namespace["_on_tf_scene"](node, message)
    world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert tuple(world.ExtractTranslation()) == (2, 3, 4)


def test_calibration_authors_an_off_centre_non_square_lens():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Camera.Define(stage, "/Camera").GetPrim()
    camera = SimpleNamespace(
        width=640, height=400, fx=300.0, fy=240.0, cx=290.0, cy=180.0, k1=0.02, k2=0.0, k3=0.0, p1=0.0, p2=0.0
    )
    namespace = _extract(["_set_camera_intrinsics"], {})
    namespace["_set_camera_intrinsics"](prim, camera)
    assert (
        "OmniLensDistortionOpenCvPinholeAPI"
        in stage.GetRootLayer().GetPrimAtPath("/Camera").GetInfo("apiSchemas").prependedItems
    )
    for key in ("fx", "fy", "cx", "cy", "k1"):
        assert prim.GetAttribute(f"omni:lensdistortion:opencvPinhole:{key}").Get() == pytest.approx(
            getattr(camera, key)
        )
    assert prim.GetAttribute("verticalAperture").Get() == pytest.approx(400 / 240)
    assert prim.GetAttribute("horizontalApertureOffset").Get() == pytest.approx(-30 / 300)
    assert prim.GetAttribute("verticalApertureOffset").Get() == pytest.approx(-20 / 240)
