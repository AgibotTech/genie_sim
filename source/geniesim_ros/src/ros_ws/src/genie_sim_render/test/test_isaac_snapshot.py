# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Snapshot USD/protocol tests. No SimulationApp, renderer, services or GPU."""

from importlib import import_module
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
scene_module = import_module("_isaac_snapshot_scene")
worker = import_module("isaacsim_snapshot_render")
pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


def _mesh(hidden=True):
    return dict(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        indices=np.array([0, 1, 2], np.int32),
        color=[0.8, 0.1, 0.2],
        hidden=hidden,
    )


def _instance(mesh="robot/mesh", count=1):
    return dict(
        mesh=mesh,
        xforms=np.tile(np.array([1, 2, 3, 0, 0, 0, 1], np.float32), (count, 1)),
        scales=np.ones((count, 3)),
        hidden=False,
    )


def _camera(name="head"):
    return dict(name=name, eye=[2, -2, 1.8], target=[0.5, 0, 1], up=[0, 0, 1], fov_degrees=65)


def _message(**kwargs):
    return dict(
        op="render",
        sequence=9,
        width=4,
        height=3,
        cameras=[_camera()],
        scene=dict(meshes={"robot/mesh": _mesh()}, instances={"robot": _instance()}),
        **kwargs,
    )


def test_hidden_prototype_has_visible_instances_and_standalone_food():
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    scene.apply(
        dict(meshes={"robot/mesh": _mesh(), "/chef_noodles": _mesh(hidden=False)}, instances={"robot": _instance()})
    )
    stage = scene.stage
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.Imageable(stage.GetPrimAtPath(scene.mesh_path("robot/mesh"))).ComputeVisibility() == "invisible"
    prim = stage.GetPrimAtPath(scene.instance_path("robot") + "/Geometry")
    assert UsdGeom.Imageable(prim).ComputeVisibility() == "inherited"
    assert len(UsdGeom.Mesh(prim).GetPointsAttr().Get()) == 3
    assert UsdGeom.Imageable(stage.GetPrimAtPath(scene.direct_path("robot/mesh"))).ComputeVisibility() == "invisible"
    assert UsdGeom.Imageable(stage.GetPrimAtPath(scene.direct_path("/chef_noodles"))).ComputeVisibility() == "inherited"
    assert not any(prim.HasAPI("PhysicsRigidBodyAPI") for prim in stage.Traverse())


def test_mesh_delta_updates_visible_reference_and_complete_instances_remove_old():
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    scene.apply(dict(meshes={"robot/mesh": _mesh()}, instances={"robot": _instance(count=2)}))
    moved = _mesh()["points"] + [0, 0, 0.5]
    scene.apply(dict(meshes={"robot/mesh": dict(points=moved)}, instances={"robot": _instance()}))
    prim = scene.stage.GetPrimAtPath(scene.instance_path("robot") + "/Geometry")
    np.testing.assert_allclose(np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()), moved)
    assert not scene.stage.GetPrimAtPath(scene.instance_path("robot", 1))
    scene.apply(dict(meshes={}, instances={}))
    assert not scene.stage.GetPrimAtPath(scene.instance_path("robot"))


def test_world_transform_and_per_instance_material():
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    batch = _instance()
    batch["xforms"][0] = [1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    batch["scales"][0] = [2, 3, 4]
    batch["colors"] = [[0.1, 0.7, 0.3]]
    batch["materials"] = [[0.25, 0.8, 0, 0]]
    batch["opacities"] = [[0.35]]
    scene.apply(dict(meshes={"robot/mesh": _mesh()}, instances={"robot": batch}))
    prim = scene.stage.GetPrimAtPath(scene.instance_path("robot") + "/Geometry")
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    np.testing.assert_allclose(matrix.Transform(Gf.Vec3d(1, 0, 0)), [1, 4, 3], atol=1e-6)
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    shader = UsdShade.Shader(scene.stage.GetPrimAtPath(str(material.GetPath()) + "/Surface"))
    np.testing.assert_allclose(shader.GetInput("diffuseColor").Get(), [0.1, 0.7, 0.3])
    assert shader.GetInput("roughness").Get() == pytest.approx(0.25)
    assert shader.GetInput("metallic").Get() == pytest.approx(0.8)
    assert shader.GetInput("opacity").Get() == pytest.approx(0.35)


def test_existing_ground_is_not_duplicated_and_hidden_batch_stays_hidden():
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    scene.apply(dict(meshes={"plane": _mesh()}, instances={"ground": _instance("plane")}))
    assert not scene.stage.GetPrimAtPath("/World/FallbackGround")
    hidden = _instance("plane")
    hidden["hidden"] = True
    scene.apply(dict(meshes={}, instances={"ground": hidden}))
    prim = scene.stage.GetPrimAtPath(scene.instance_path("ground") + "/Geometry")
    assert UsdGeom.Imageable(prim).ComputeVisibility() == "invisible"
    assert scene.stage.GetPrimAtPath("/World/FallbackGround")


def test_hidden_uninitialized_batch_waits_for_real_visible_transform():
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    hidden = _instance()
    hidden["hidden"] = True
    hidden["xforms"][:] = 0
    # Deliberately visible prototype: a hidden instance must still prevent
    # the prototype from appearing as a standalone mesh at the origin.
    scene.apply(dict(meshes={"robot/mesh": _mesh(hidden=False)}, instances={"robot": hidden}))
    assert not scene.stage.GetPrimAtPath(scene.instance_path("robot"))
    direct = scene.stage.GetPrimAtPath(scene.direct_path("robot/mesh"))
    assert UsdGeom.Imageable(direct).ComputeVisibility() == "invisible"
    scene.apply(dict(meshes={}, instances={"robot": _instance()}))
    prim = scene.stage.GetPrimAtPath(scene.instance_path("robot") + "/Geometry")
    assert UsdGeom.Imageable(prim).ComputeVisibility() == "inherited"
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    np.testing.assert_allclose(matrix.ExtractTranslation(), [1, 2, 3])
    scene.apply(dict(meshes={}, instances={"robot": hidden}))
    assert UsdGeom.Imageable(prim).ComputeVisibility() == "invisible"
    np.testing.assert_allclose(UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation(), [1, 2, 3])


def test_camera_looks_at_target_with_requested_vertical_fov_and_roll():
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    config = scene_module.validate_cameras([_camera()], 640, 480)[0]
    path = scene.camera(config, 640, 480)
    cam = UsdGeom.Camera(scene.stage.GetPrimAtPath(path))
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(cam.GetPrim())
    front = np.asarray(matrix.TransformDir(Gf.Vec3d(0, 0, -1)))
    expected = config["target"] - config["eye"]
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(front, expected, atol=1e-6)
    assert cam.GetVerticalApertureAttr().Get() == pytest.approx(2 * np.tan(np.radians(65) / 2))
    assert cam.GetHorizontalApertureAttr().Get() / cam.GetVerticalApertureAttr().Get() == pytest.approx(640 / 480)


@pytest.mark.parametrize(
    "bad", [dict(indices=np.array([0, 1, 99])), dict(points=[[np.nan, 0, 0]]), dict(indices=np.array([0.0, 1.0, 2.0]))]
)
def test_bad_mesh_is_rejected(bad):
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    with pytest.raises(ValueError):
        scene.apply(dict(meshes={"mesh": {**_mesh(), **bad}}, instances={}))


def test_bad_camera_and_zero_quaternion_are_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        scene_module.validate_cameras([{**_camera(), "target": _camera()["eye"]}], 4, 3)
    with pytest.raises(ValueError, match="positive integer"):
        scene_module.validate_cameras([_camera()], True, 3)
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    batch = _instance()
    batch["xforms"][0, 3:] = 0
    with pytest.raises(ValueError, match="zero quaternion"):
        scene.apply(dict(meshes={"robot/mesh": _mesh()}, instances={"robot": batch}))


class FakeReplicator:
    def __init__(self, *, attach_fails=False, wrong_size=False):
        self.events = []
        self.attach_fails = attach_fails
        self.wrong_size = wrong_size
        self.create = SimpleNamespace(render_product=self._product)
        self.AnnotatorRegistry = SimpleNamespace(get_annotator=self._annotator)
        self.orchestrator = SimpleNamespace(step=lambda **kw: self.events.append(("step", kw)))

    def _product(self, path, size, **kwargs):
        self.events.append(("create", path, size))
        return SimpleNamespace(destroy=lambda: self.events.append(("destroy", path)))

    def _annotator(self, _):
        def attach(product):
            self.events.append(("attach",))
            if self.attach_fails:
                raise RuntimeError("attach failed")

        def get_data(**kwargs):
            self.events.append(("read", kwargs))
            return np.full((3, 5 if self.wrong_size else 4, 4), 123, np.uint8)

        return SimpleNamespace(attach=attach, get_data=get_data, detach=lambda _: self.events.append(("detach",)))


def test_renderer_waits_for_render_then_copies_real_camera_arrays():
    rep = FakeReplicator()
    renderer = worker.SnapshotRenderer(None, stage=Usd.Stage.CreateInMemory(), replicator=rep)
    request = _message()
    request["cameras"] = [_camera("head"), _camera("hand_left"), _camera("hand_right"), _camera("observer")]
    response = renderer.render(request)
    assert response["sequence"] == 9
    assert response["renderer"] == "isaacsim"
    assert set(response["images"]) == {"head", "hand_left", "hand_right", "observer"}
    assert all(image.shape == (3, 4, 3) and image.dtype == np.uint8 for image in response["images"].values())
    assert next(event for event in rep.events if event[0] == "step")[1] == dict(
        rt_subframes=2, pause_timeline=True, delta_time=0.0, wait_for_render=True
    )
    assert [event[0] for event in rep.events].index("step") < [event[0] for event in rep.events].index("read")
    renderer.close()
    assert sum(e[0] == "destroy" for e in rep.events) == 4


def test_annotator_attach_failure_still_destroys_its_render_product():
    rep = FakeReplicator(attach_fails=True)
    renderer = worker.SnapshotRenderer(None, stage=Usd.Stage.CreateInMemory(), replicator=rep)
    with pytest.raises(RuntimeError, match="attach failed"):
        renderer.render(_message())
    renderer.close()
    assert [event[0] for event in rep.events][-2:] == ["detach", "destroy"]


def test_renderer_refuses_wrong_image_dimensions_without_resizing():
    rep = FakeReplicator(wrong_size=True)
    renderer = worker.SnapshotRenderer(None, stage=Usd.Stage.CreateInMemory(), replicator=rep)
    try:
        with pytest.raises(RuntimeError, match="expected uint8"):
            renderer.render(_message())
    finally:
        renderer.close()


class FakeConnection:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []
        self.closed = False

    def recv(self):
        try:
            return next(self.messages)
        except StopIteration:
            raise EOFError

    def send(self, message):
        self.sent.append(message)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("fail", ["construction", "render", None])
def test_worker_pipe_protocol_and_cleanup_after_construction_or_render_failure(fail):
    connection = FakeConnection([dict(op="render", sequence=42), dict(op="close")])
    cleanup = []

    def factory(resources):
        resources.callback(lambda: cleanup.append("app"))
        if fail == "construction":
            raise RuntimeError("construction failed")
        resources.callback(lambda: cleanup.append("renderer"))

        def render(message):
            if fail == "render":
                raise RuntimeError("render failed")
            return dict(sequence=message["sequence"], images={}, renderer="isaacsim")

        return SimpleNamespace(render=render)

    assert worker.serve(connection, factory) == (1 if fail else 0)
    assert connection.closed
    assert cleanup == (["app"] if fail == "construction" else ["renderer", "app"])
    if fail:
        assert "error" in connection.sent[-1]
    else:
        assert connection.sent == [
            dict(ready=True, renderer="isaacsim"),
            dict(sequence=42, images={}, renderer="isaacsim"),
        ]


def test_parent_eof_closes_worker_resources():
    connection = FakeConnection([])
    cleanup = []

    def factory(resources):
        resources.callback(lambda: cleanup.append(True))
        return SimpleNamespace()

    assert worker.serve(connection, factory) == 0
    assert cleanup == [True] and connection.closed


@pytest.mark.parametrize("failure", ["construction", "render"])
def test_original_error_is_logged_and_sent_before_app_cleanup(failure, monkeypatch):
    events = []
    connection = FakeConnection([dict(op="render", sequence=77)])
    original_send = connection.send

    def send(message):
        events.append("error" if "error" in message else "ready")
        original_send(message)

    connection.send = send
    monkeypatch.setattr(worker.traceback, "print_exc", lambda **_: events.append("traceback"))

    def factory(resources):
        # Real SimulationApp.close may terminate the process immediately.
        resources.callback(lambda: events.append("app.close"))
        if failure == "construction":
            raise RuntimeError("original construction error")

        def render(_):
            raise RuntimeError("original render error")

        return SimpleNamespace(render=render)

    assert worker.serve(connection, factory) == 1
    assert events.index("traceback") < events.index("app.close")
    assert events.index("error") < events.index("app.close")
    assert connection.sent[-1]["error"] == f"RuntimeError: original {failure} error"
    assert connection.closed


def test_real_private_pipe_transports_sequence_and_numpy_rgb():
    from multiprocessing import Pipe
    from threading import Thread

    parent, child = Pipe(duplex=True)
    closed = []

    def factory(resources):
        resources.callback(lambda: closed.append(True))
        return SimpleNamespace(
            render=lambda message: dict(
                sequence=message["sequence"], renderer="isaacsim", images={"head": np.full((3, 4, 3), 123, np.uint8)}
            )
        )

    thread = Thread(target=worker.serve, args=(child, factory), daemon=True)
    thread.start()
    try:
        assert parent.poll(2)
        assert parent.recv() == dict(ready=True, renderer="isaacsim")
        parent.send(dict(op="render", sequence=101))
        assert parent.poll(2)
        reply = parent.recv()
        assert reply["sequence"] == 101
        assert reply["images"]["head"].shape == (3, 4, 3)
        assert reply["images"]["head"].dtype == np.uint8
        parent.send(dict(op="close"))
    finally:
        parent.close()
        thread.join(2)
    assert not thread.is_alive()
    assert closed == [True]


def test_actual_factory_closes_app_when_scene_construction_fails(monkeypatch):
    from contextlib import ExitStack

    closed = []
    common = import_module("_isaac_render_common")
    monkeypatch.setattr(common, "preload_warp_before_kit", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "isaacsim",
        SimpleNamespace(SimulationApp=lambda config: SimpleNamespace(close=lambda: closed.append(config))),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("scene construction failed")

    monkeypatch.setattr(worker, "SnapshotRenderer", fail)
    with pytest.raises(RuntimeError, match="scene construction failed"):
        with ExitStack() as resources:
            worker._create_renderer(resources)
    from genie_sim_engine_isaacsim.kit.launch_config import prepare_launch_config

    assert closed == [prepare_launch_config(dict(headless=True, renderer="RaytracedLighting"))]


def test_snapshot_factory_applies_sdk62_launch_contract(monkeypatch):
    from contextlib import ExitStack
    from genie_sim_engine_isaacsim.kit import launch_config

    configs = []
    monkeypatch.setattr(import_module("_isaac_render_common"), "preload_warp_before_kit", lambda: None)
    monkeypatch.setattr(launch_config.metadata, "version", lambda name: "6.2.0.0")
    monkeypatch.delenv("ISAAC_PATH", raising=False)

    def app(config):
        configs.append(config)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(SimulationApp=app))
    monkeypatch.setattr(worker, "SnapshotRenderer", lambda app: SimpleNamespace(close=lambda: None))
    with ExitStack() as resources:
        worker._create_renderer(resources)
    assert configs == [
        dict(headless=True, renderer="RaytracedLighting", extra_args=["--/rtx/hydra/supportMultiTickRate=false"])
    ]


@pytest.fixture
def authored_robot_asset(tmp_path):
    """A USD composition fixture, not an Isaac-rendered robot or model asset."""
    from PIL import Image

    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    texture = texture_dir / "coat.png"
    Image.fromarray(np.array([[[255, 20, 30], [20, 255, 30]]], np.uint8)).save(texture)
    asset = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = UsdGeom.Xform.Define(stage, "/genie")
    root.AddTranslateOp().Set(Gf.Vec3d(10, 20, 30))
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/genie/Looks")
    material = UsdShade.Material.Define(stage, "/genie/Looks/Coat")
    surface = UsdShade.Shader.Define(stage, "/genie/Looks/Coat/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    reader = UsdShade.Shader.Define(stage, "/genie/Looks/Coat/UVReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, "/genie/Looks/Coat/ColorTexture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("textures/coat.png"))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    gloss = UsdShade.Material.Define(stage, "/genie/Looks/Gloss")
    mdl = UsdShade.Shader.Define(stage, "/genie/Looks/Gloss/MDL")
    mdl.SetSourceAsset(Sdf.AssetPath("materials/finish.mdl"), "mdl")
    mdl.SetSourceAssetSubIdentifier("AuthoredGloss", "mdl")
    mdl.CreateOutput("out", Sdf.ValueTypeNames.Token)
    gloss.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
    for name in ("base_link", "arm_link"):
        body = UsdGeom.Xform.Define(stage, "/genie/" + name)
        body.AddTranslateOp().Set(Gf.Vec3d(7, 8, 9))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        visual = UsdGeom.Xform.Define(stage, f"/genie/{name}/Visual")
        visual.AddTranslateOp().Set(Gf.Vec3d(0.25, -0.1, 0.2))
        mesh = UsdGeom.Mesh.Define(stage, f"/genie/{name}/Visual/Painted")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
            [(0, 0), (1, 0), (0, 1)]
        )
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        subset = UsdGeom.Subset.Define(stage, f"/genie/{name}/Visual/Painted/GlossFace")
        subset.CreateElementTypeAttr("face")
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr([0])
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(gloss)
    stage.GetRootLayer().Save()
    return asset


def _robot_poses():
    return {
        "base_link": [1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)],
        "arm_link": [4, 5, 6, 0, 0, 0, 1],
    }


def _authored_scene(asset, *, poses=None, descriptor=True):
    result = dict(meshes={}, instances={}, body_poses=_robot_poses() if poses is None else poses)
    if descriptor:
        result["authored_robot"] = dict(asset_path=str(asset))
    return result


def _painted_prim(stage, body="base_link"):
    prim = stage.GetPrimAtPath(f"/World/AuthoredRobot/{body}/Visual/Painted")
    assert prim and prim.IsA(UsdGeom.Mesh), "authored visual mesh was not referenced"
    return prim


def _texture_shader_for_bound_material(prim):
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    assert material, "visible geometry has no bound material"
    surface, _, _ = material.ComputeSurfaceSource()
    assert surface and surface.GetIdAttr().Get() == "UsdPreviewSurface"
    source = surface.GetInput("diffuseColor").GetConnectedSource()
    assert source, "texture connection was replaced with a constant RGB"
    texture = UsdShade.Shader(source[0].GetPrim())
    assert texture.GetIdAttr().Get() == "UsdUVTexture"
    return texture


def test_authored_whole_asset_keeps_sibling_looks_subsets_uv_and_relative_texture(authored_robot_asset):
    original_bytes = authored_robot_asset.read_bytes()
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    scene.apply(_authored_scene(authored_robot_asset))
    prim = _painted_prim(scene.stage)
    bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    assert str(bound.GetPath()) == "/World/AuthoredRobot/Looks/Coat"
    texture = _texture_shader_for_bound_material(prim)
    file = texture.GetInput("file").Get()
    assert file.path == "textures/coat.png"
    assert Path(file.resolvedPath) == authored_robot_asset.parent / "textures/coat.png"
    np.testing.assert_array_equal(UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get(), [[0, 0], [1, 0], [0, 1]])
    subset = scene.stage.GetPrimAtPath(str(prim.GetPath()) + "/GlossFace")
    assert UsdGeom.Subset(subset).GetIndicesAttr().Get() == [0]
    gloss, _ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()
    assert str(gloss.GetPath()) == "/World/AuthoredRobot/Looks/Gloss"
    mdl, _, _ = gloss.ComputeSurfaceSource("mdl")
    assert mdl.GetSourceAsset("mdl").path == "materials/finish.mdl"
    assert mdl.GetSourceAssetSubIdentifier("mdl") == "AuthoredGloss"
    assert UsdGeom.Imageable(prim).ComputeVisibility() == "inherited"
    assert not any(
        "Physics" in schema or "Physx" in schema for p in scene.stage.Traverse() for schema in p.GetAppliedSchemas()
    )
    assert authored_robot_asset.read_bytes() == original_bytes


def test_authored_world_pose_replaces_source_pose_and_preserves_visual_offset(authored_robot_asset):
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    scene.apply(_authored_scene(authored_robot_asset))
    body = scene.stage.GetPrimAtPath("/World/AuthoredRobot/base_link")
    assert UsdGeom.Xformable(body).GetResetXformStack()
    cache = UsdGeom.XformCache()
    np.testing.assert_allclose(cache.GetLocalToWorldTransform(body).ExtractTranslation(), [1, 2, 3], atol=1e-6)
    np.testing.assert_allclose(
        cache.GetLocalToWorldTransform(_painted_prim(scene.stage)).Transform(Gf.Vec3d(0)), [1.1, 2.25, 3.2], atol=1e-6
    )
    updated = _robot_poses()
    updated["base_link"] = [9, 8, 7, 0, 0, 0, 1]
    scene.apply(_authored_scene(authored_robot_asset, poses=updated, descriptor=False))
    np.testing.assert_allclose(
        UsdGeom.XformCache().GetLocalToWorldTransform(_painted_prim(scene.stage)).Transform(Gf.Vec3d(0)),
        [9.25, 7.9, 7.2],
        atol=1e-6,
    )
    _texture_shader_for_bound_material(_painted_prim(scene.stage))


def test_nested_authored_links_each_follow_world_pose_without_parent_pose_doubling(authored_robot_asset):
    source = Usd.Stage.Open(str(authored_robot_asset))
    edit = Sdf.BatchNamespaceEdit()
    edit.Add("/genie/arm_link", "/genie/base_link/arm_link")
    assert source.GetRootLayer().Apply(edit)
    source.GetRootLayer().Save()
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    scene.apply(_authored_scene(authored_robot_asset))
    arm = scene.stage.GetPrimAtPath("/World/AuthoredRobot/base_link/arm_link")
    assert UsdGeom.Xformable(arm).GetResetXformStack()
    cache = UsdGeom.XformCache()
    np.testing.assert_allclose(cache.GetLocalToWorldTransform(arm).ExtractTranslation(), [4, 5, 6], atol=1e-6)
    mesh = scene.stage.GetPrimAtPath(str(arm.GetPath()) + "/Visual/Painted")
    np.testing.assert_allclose(cache.GetLocalToWorldTransform(mesh).Transform(Gf.Vec3d(0)), [4.25, 4.9, 6.2], atol=1e-6)
    _texture_shader_for_bound_material(mesh)
    poses = _robot_poses()
    del poses["arm_link"]
    with pytest.raises(ValueError, match="missing"):
        scene.apply(_authored_scene(authored_robot_asset, poses=poses, descriptor=False))


def test_explicit_authored_root_works_without_default_prim(authored_robot_asset):
    source = Usd.Stage.Open(str(authored_robot_asset))
    source.ClearDefaultPrim()
    source.GetRootLayer().Save()
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    with pytest.raises(ValueError, match="root/default"):
        scene.apply(_authored_scene(authored_robot_asset))
    request = _authored_scene(authored_robot_asset)
    request["authored_robot"]["root_prim"] = "/genie"
    scene.apply(request)
    _texture_shader_for_bound_material(_painted_prim(scene.stage))


def test_legacy_authored_binding_relationship_gets_api_without_rebinding_or_source_edits(authored_robot_asset):
    source = Usd.Stage.Open(str(authored_robot_asset))
    for prim in source.Traverse():
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            prim.RemoveAPI(UsdShade.MaterialBindingAPI)
    source.GetRootLayer().Save()
    before = authored_robot_asset.read_bytes()
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    scene.apply(_authored_scene(authored_robot_asset))
    prim = _painted_prim(scene.stage)
    assert prim.HasAPI(UsdShade.MaterialBindingAPI)
    subset = scene.stage.GetPrimAtPath(str(prim.GetPath()) + "/GlossFace")
    assert subset.HasAPI(UsdShade.MaterialBindingAPI)
    _texture_shader_for_bound_material(prim)
    assert authored_robot_asset.read_bytes() == before


def test_ambiguous_authored_body_leaf_is_rejected(authored_robot_asset):
    source = Usd.Stage.Open(str(authored_robot_asset))
    for side in ("left", "right"):
        body = UsdGeom.Xform.Define(source, f"/genie/{side}/ambiguous_link")
        mesh = UsdGeom.Mesh.Define(source, str(body.GetPath()) + "/Mesh")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    source.GetRootLayer().Save()
    request = _authored_scene(authored_robot_asset)
    request["body_poses"]["ambiguous_link"] = [0, 0, 0, 0, 0, 0, 1]
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    with pytest.raises(ValueError, match="ambiguous"):
        scene.apply(request)


@pytest.mark.parametrize("failure", ["missing_asset", "invalid_root", "missing_pose", "changed_asset"])
def test_authored_asset_errors_do_not_silently_use_flat_robot(authored_robot_asset, tmp_path, failure):
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    snapshot = _authored_scene(authored_robot_asset)
    if failure == "missing_asset":
        snapshot["authored_robot"]["asset_path"] = str(tmp_path / "missing.usda")
    elif failure == "invalid_root":
        snapshot["authored_robot"]["root_prim"] = "/not_a_robot"
    elif failure == "missing_pose":
        del snapshot["body_poses"]["arm_link"]
    else:
        scene.apply(snapshot)
        other = tmp_path / "different.usda"
        other.write_bytes(authored_robot_asset.read_bytes())
        snapshot["authored_robot"]["asset_path"] = str(other)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        scene.apply(snapshot)


def test_authored_asset_is_shared_by_sensor_and_observer_render_requests(authored_robot_asset):
    rep = FakeReplicator()
    stage = Usd.Stage.CreateInMemory()
    renderer = worker.SnapshotRenderer(None, stage=stage, replicator=rep)
    seen = []

    def capture_barrier(**kwargs):
        prim = _painted_prim(stage)
        texture = _texture_shader_for_bound_material(prim)
        seen.append(
            (
                str(texture.GetPath()),
                np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()),
            )
        )
        rep.events.append(("step", kwargs))

    rep.orchestrator.step = capture_barrier
    request = _message()
    request["scene"] = _authored_scene(authored_robot_asset)
    request["cameras"] = [_camera(name) for name in ("head", "hand_left", "hand_right")]
    try:
        sensors = renderer.render(request)
        request["sequence"] += 1
        request["cameras"] = [_camera("observer")]
        moved = _robot_poses()
        moved["base_link"] = [9, 8, 7, 0, 0, 0, 1]
        request["scene"] = _authored_scene(authored_robot_asset, poses=moved, descriptor=False)
        observer = renderer.render(request)
    finally:
        renderer.close()
    assert set(sensors["images"]) == {"head", "hand_left", "hand_right"}
    assert set(observer["images"]) == {"observer"}
    assert sensors["renderer"] == observer["renderer"] == "isaacsim"
    assert seen[0][0] == seen[1][0]
    np.testing.assert_allclose(seen[0][1], [1.1, 2.25, 3.2], atol=1e-6)
    np.testing.assert_allclose(seen[1][1], [9.25, 7.9, 7.2], atol=1e-6)
    assert sum(event[0] == "step" for event in rep.events) == 2


def test_snapshot_texture_file_and_st_survive_instance_color_overrides(tmp_path):
    from PIL import Image

    path = tmp_path / "owned_by_user.png"
    Image.fromarray(np.array([[[20, 50, 200], [200, 50, 20]]], np.uint8)).save(path)
    before = path.read_bytes()
    rep = FakeReplicator()
    renderer = worker.SnapshotRenderer(None, stage=Usd.Stage.CreateInMemory(), replicator=rep)
    request = _message()
    request["scene"]["meshes"]["robot/mesh"].update(
        uvs=np.array([[0, 0], [1, 0], [0, 1]], np.float32), texture=str(path)
    )
    request["scene"]["instances"]["robot"]["colors"] = [[0.1, 0.7, 0.3]]
    request["scene"]["instances"]["robot"]["materials"] = [[0.2, 0.4, 0, 1]]
    try:
        renderer.render(request)
        prim = renderer.scene.stage.GetPrimAtPath(renderer.scene.instance_path("robot") + "/Geometry")
        texture = _texture_shader_for_bound_material(prim)
        assert Path(texture.GetInput("file").Get().resolvedPath) == path
        uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        assert uv.GetInterpolation() == "vertex"
        np.testing.assert_array_equal(uv.Get(), [[0, 0], [1, 0], [0, 1]])
    finally:
        renderer.close()
    assert path.read_bytes() == before


@pytest.mark.parametrize("channels", [3, 4])
def test_snapshot_array_texture_writes_real_png_and_removes_only_owned_file(channels):
    from PIL import Image

    rep = FakeReplicator()
    renderer = worker.SnapshotRenderer(None, stage=Usd.Stage.CreateInMemory(), replicator=rep)
    pixels = np.arange(2 * 3 * channels, dtype=np.uint8).reshape(2, 3, channels)[:, ::-1]
    request = _message()
    request["scene"]["meshes"]["robot/mesh"].update(uvs=np.array([[0, 0], [1, 0], [0, 1]], np.float32), texture=pixels)
    texture_path = None
    try:
        renderer.render(request)
        prim = renderer.scene.stage.GetPrimAtPath(renderer.scene.instance_path("robot") + "/Geometry")
        texture = _texture_shader_for_bound_material(prim)
        texture_path = Path(texture.GetInput("file").Get().resolvedPath)
        assert texture_path.is_file()
        with Image.open(texture_path) as image:
            assert image.format == "PNG"
            np.testing.assert_array_equal(np.asarray(image), pixels)
    finally:
        renderer.close()
    assert texture_path is not None and not texture_path.exists()


def test_texture_material_cache_does_not_alias_distinct_textures_or_lose_them_on_mesh_delta():
    rep = FakeReplicator()
    renderer = worker.SnapshotRenderer(None, stage=Usd.Stage.CreateInMemory(), replicator=rep)
    request = _message()
    uvs = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    request["scene"] = dict(
        meshes={
            name: {**_mesh(), "uvs": uvs, "texture": np.full((2, 2, 3), value, np.uint8)}
            for name, value in (("mesh_a", 20), ("mesh_b", 200))
        },
        instances={"a": _instance("mesh_a"), "b": _instance("mesh_b")},
    )

    def bindings():
        rows = []
        for name in ("a", "b"):
            prim = renderer.scene.stage.GetPrimAtPath(renderer.scene.instance_path(name) + "/Geometry")
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            texture = _texture_shader_for_bound_material(prim)
            rows.append((str(material.GetPath()), texture.GetInput("file").Get().resolvedPath))
        return rows

    try:
        renderer.render(request)
        before = bindings()
        assert before[0][0] != before[1][0]
        assert before[0][1] != before[1][1]
        request["sequence"] += 1
        request["scene"]["meshes"] = {"mesh_a": {"points": _mesh()["points"] + [0, 0, 0.1]}}
        renderer.render(request)
        assert bindings() == before
    finally:
        renderer.close()


@pytest.fixture
def authored_background_asset(tmp_path):
    """Real USD composition with an offset pot; no Kit or rendered-image fixture."""
    from PIL import Image

    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    Image.fromarray(np.full((2, 2, 3), [20, 70, 190], np.uint8)).save(texture_dir / "pot.png")
    asset = tmp_path / "background.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = UsdGeom.Xform.Define(stage, "/World")
    root.AddTranslateOp().Set(Gf.Vec3d(2, -3, 0.2))
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/background/Table")
    UsdLux.DomeLight.Define(stage, "/World/light/Dome").CreateIntensityAttr(1234)
    UsdGeom.Camera.Define(stage, "/World/NativeCamera")
    UsdPhysics.Scene.Define(stage, "/World/Physics")
    UsdPhysics.FixedJoint.Define(stage, "/World/FixedJoint")
    objects = UsdGeom.Xform.Define(stage, "/World/objects")
    objects.AddRotateZOp().Set(15)
    pot = UsdGeom.Xform.Define(stage, "/World/objects/Pot")
    pot.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.3, 0.7))
    pot.AddRotateZOp().Set(-20)
    pot.AddScaleOp().Set(Gf.Vec3f(0.8, 1.2, 0.6))
    entity = UsdGeom.Xform.Define(stage, "/World/objects/Pot/entity")
    entity.AddRotateXOp().Set(90)
    UsdPhysics.RigidBodyAPI.Apply(entity.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/objects/Pot/entity/Visual")
    mesh.CreatePointsAttr([(0, 0, 0), (0.1, 0, 0), (0, 0.2, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(
        [(0, 0), (1, 0), (0, 1)]
    )
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    material = UsdShade.Material.Define(stage, "/World/objects/Pot/entity/Looks/Paint")
    surface = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    texture = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("textures/pot.png"))
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    UsdGeom.Cube.Define(stage, "/World/objects/Food")
    stage.GetRootLayer().Save()
    return asset


def _background_snapshot(asset, *, pose=None, descriptor=True):
    reference = [1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    result = dict(meshes={}, instances={}, visual_body_poses={"chef_wok": list(reference) if pose is None else pose})
    if descriptor:
        result["authored_scene"] = dict(
            asset_path=str(asset),
            root_prim="/World",
            hide_prims=["/World/objects"],
            use_asset_lights=True,
            body_visuals=[dict(body="chef_wok", source_prim="/World/objects/Pot", reference_pose=reference)],
        )
    return result


def _dynamic_visual(stage):
    meshes = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
        and not prim.GetPath().HasPrefix(Sdf.Path("/World/AuthoredScene"))
        and not prim.GetPath().HasPrefix(Sdf.Path("/World/Prototypes"))
        and not prim.GetPath().HasPrefix(Sdf.Path("/World/Direct"))
    ]
    assert len(meshes) == 1
    return meshes[0]


def _pose_matrix(row):
    matrix = Gf.Matrix4d(1)
    matrix.SetRotate(Gf.Quatd(float(row[6]), Gf.Vec3d(*[float(x) for x in row[3:6]])))
    matrix.SetTranslateOnly(Gf.Vec3d(*[float(x) for x in row[:3]]))
    return matrix


def test_authored_background_preserves_materials_lights_hides_static_objects_and_disables_physics(
    authored_background_asset,
):
    original = authored_background_asset.read_bytes()
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    scene.apply(dict(meshes={}, instances={}))
    assert scene.stage.GetPrimAtPath("/World/FallbackGround")
    scene.apply(_background_snapshot(authored_background_asset))
    assert scene.stage.GetPrimAtPath("/World/AuthoredScene/background/Table")
    assert not scene.stage.GetPrimAtPath("/World/AuthoredScene/objects").IsActive()
    assert scene.stage.GetPrimAtPath("/World/AuthoredScene/light/Dome").IsActive()
    assert not scene.stage.GetPrimAtPath("/World/Light").IsActive()
    assert not scene.stage.GetPrimAtPath("/World/Sun").IsActive()
    assert not scene.stage.GetPrimAtPath("/World/FallbackGround")
    for path in (
        "/World/AuthoredScene/NativeCamera",
        "/World/AuthoredScene/Physics",
        "/World/AuthoredScene/FixedJoint",
    ):
        assert not scene.stage.GetPrimAtPath(path).IsActive()
    assert not any(
        schema.startswith(("Physics", "Physx", "OmniPhysics", "Newton"))
        for prim in scene.stage.Traverse()
        for schema in prim.GetAppliedSchemas()
    )
    prim = _dynamic_visual(scene.stage)
    texture = _texture_shader_for_bound_material(prim)
    assert Path(texture.GetInput("file").Get().resolvedPath) == authored_background_asset.parent / "textures/pot.png"
    assert UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").GetInterpolation() == "faceVarying"
    assert authored_background_asset.read_bytes() == original


def test_dynamic_authored_visual_preserves_source_world_offset_then_follows_rigid_pose(authored_background_asset):
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    snapshot = _background_snapshot(authored_background_asset)
    source = Usd.Stage.Open(str(authored_background_asset))
    original = source.GetPrimAtPath("/World/objects/Pot/entity/Visual")
    original_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(original)
    points = [Gf.Vec3d(0), Gf.Vec3d(0.1, 0, 0), Gf.Vec3d(0, 0.2, 0)]
    scene.apply(snapshot)
    first = UsdGeom.XformCache().GetLocalToWorldTransform(_dynamic_visual(scene.stage))
    np.testing.assert_allclose(first, original_matrix, atol=1e-10)
    moved = [4, -2, 6, 0, 0, -0.5, np.sqrt(0.75)]
    scene.apply(_background_snapshot(authored_background_asset, pose=moved, descriptor=False))
    actual = UsdGeom.XformCache().GetLocalToWorldTransform(_dynamic_visual(scene.stage))
    reference = snapshot["authored_scene"]["body_visuals"][0]["reference_pose"]
    expected = original_matrix * _pose_matrix(reference).GetInverse() * _pose_matrix(moved)
    for point in points:
        np.testing.assert_allclose(actual.Transform(point), expected.Transform(point), atol=1e-10)
    _texture_shader_for_bound_material(_dynamic_visual(scene.stage))


@pytest.mark.parametrize(
    "problem",
    [
        "asset",
        "root",
        "hide",
        "body_source",
        "reference",
        "missing",
        "nan",
        "zero",
        "changed",
        "changed_asset",
        "duplicate",
    ],
)
def test_authored_background_rejects_invalid_identity_or_missing_fresh_driver(
    authored_background_asset, tmp_path, problem
):
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    request = _background_snapshot(authored_background_asset)
    descriptor = request["authored_scene"]
    if problem == "asset":
        descriptor["asset_path"] = str(tmp_path / "missing.usda")
    elif problem == "root":
        descriptor["root_prim"] = "/missing"
    elif problem == "hide":
        descriptor["hide_prims"] = ["/World/not_present"]
    elif problem == "body_source":
        descriptor["body_visuals"][0]["source_prim"] = "/World/not_present"
    elif problem == "reference":
        descriptor["body_visuals"][0]["reference_pose"] = [0] * 7
    elif problem == "missing":
        request["visual_body_poses"] = {}
    elif problem == "nan":
        request["visual_body_poses"]["chef_wok"][0] = float("nan")
    elif problem == "zero":
        request["visual_body_poses"]["chef_wok"] = [0] * 7
    elif problem == "changed":
        scene.apply(request)
        descriptor["hide_prims"] = []
    elif problem == "changed_asset":
        scene.apply(request)
        other = tmp_path / "other.usda"
        other.write_bytes(authored_background_asset.read_bytes())
        descriptor["asset_path"] = str(other)
    else:
        descriptor["body_visuals"].append(dict(descriptor["body_visuals"][0]))
    with pytest.raises((ValueError, FileNotFoundError)):
        scene.apply(request)


def test_authored_background_requires_new_driver_on_every_capture_and_can_keep_worker_lights(authored_background_asset):
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    request = _background_snapshot(authored_background_asset)
    request["authored_scene"]["use_asset_lights"] = False
    scene.apply(request)
    assert scene.stage.GetPrimAtPath("/World/Light").IsActive()
    assert scene.stage.GetPrimAtPath("/World/Sun").IsActive()
    assert not scene.stage.GetPrimAtPath("/World/AuthoredScene/light/Dome").IsActive()
    assert not scene.stage.GetPrimAtPath("/World/FallbackGround")
    with pytest.raises(ValueError, match="visual_body_poses"):
        scene.apply(dict(meshes={}, instances={}))


def test_authored_background_and_dynamic_pot_are_shared_by_sensor_and_observer_requests(authored_background_asset):
    rep = FakeReplicator()
    renderer = worker.SnapshotRenderer(None, stage=Usd.Stage.CreateInMemory(), replicator=rep)
    captured = []

    def barrier(**kwargs):
        prim = _dynamic_visual(renderer.scene.stage)
        captured.append((UsdGeom.XformCache().GetLocalToWorldTransform(prim), _texture_shader_for_bound_material(prim)))

    rep.orchestrator.step = barrier
    request = _message()
    request["scene"] = _background_snapshot(authored_background_asset)
    request["cameras"] = [_camera(name) for name in ("head", "hand_left", "hand_right")]
    try:
        sensors = renderer.render(request)
        request["sequence"] += 1
        request["scene"] = _background_snapshot(authored_background_asset, pose=[1, 2, 4, 0, 0, 0, 1], descriptor=False)
        request["cameras"] = [_camera("observer")]
        observer = renderer.render(request)
    finally:
        renderer.close()
    assert set(sensors["images"]) == {"head", "hand_left", "hand_right"}
    assert set(observer["images"]) == {"observer"}
    assert len(captured) == 2
    assert not np.allclose(captured[0][0], captured[1][0])
    assert captured[0][1].GetPath() == captured[1][1].GetPath()


def test_mesh_opacity_reaches_standalone_material():
    scene = scene_module.SnapshotScene(Usd.Stage.CreateInMemory())
    scene.apply(dict(meshes={"food": {**_mesh(hidden=False), "opacity": 0.45}}, instances={}))
    prim = scene.stage.GetPrimAtPath(scene.direct_path("food"))
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    shader = UsdShade.Shader(scene.stage.GetPrimAtPath(str(material.GetPath()) + "/Surface"))
    assert shader.GetInput("opacity").Get() == pytest.approx(0.45)
