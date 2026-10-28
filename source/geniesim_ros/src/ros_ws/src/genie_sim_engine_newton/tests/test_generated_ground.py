# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""The generated USD floor and Newton plane must not draw coplanar surfaces."""

import importlib.util
import inspect
from pathlib import Path

import pytest

pytest.importorskip("pxr")
pytest.importorskip("warp")
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

_MODEL = Path(__file__).resolve().parents[1] / "genie_sim_engine_newton/engine/newton/setup/model.py"
_SPEC = importlib.util.spec_from_file_location("_ground_model_under_test", _MODEL)
model_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(model_module)


@pytest.fixture
def stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    floor = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
    floor.CreatePointsAttr([Gf.Vec3f(x, y, 0) for x, y in [(-25, -25), (25, -25), (25, 25), (-25, 25)]])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdGeom.Cube.Define(stage, "/World/Prop").AddTranslateOp().Set(Gf.Vec3d(0, 0, 2))
    return stage


@pytest.mark.parametrize("scene_cfg", [{}, {"scene": None}, {"scene": ""}, {"scene": " \t"}])
def test_generated_floor_is_excluded_without_changing_usd(stage, scene_cfg):
    before = stage.GetRootLayer().ExportToString()
    assert model_module._generated_ground_visual_paths(stage, scene_cfg) == ["/World/GroundPlane"]
    assert stage.GetRootLayer().ExportToString() == before
    assert stage.GetSessionLayer().empty


def test_authored_world_floor_is_preserved(stage):
    assert model_module._generated_ground_visual_paths(stage, {"scene": "scenes/authored.usda"}) == []


@pytest.mark.parametrize("schema", [UsdPhysics.CollisionAPI, UsdPhysics.RigidBodyAPI])
def test_physics_bearing_floor_is_preserved(stage, schema):
    schema.Apply(stage.GetPrimAtPath("/World/GroundPlane"))
    assert model_module._generated_ground_visual_paths(stage, {}) == []


def test_absent_generated_floor_is_ignored(stage):
    stage.RemovePrim("/World/GroundPlane")
    assert model_module._generated_ground_visual_paths(stage, {}) == []


def test_real_newton_import_retains_one_floor_and_other_visuals(stage):
    newton = pytest.importorskip("newton")
    if "load_static_visual_shapes" not in inspect.signature(newton.ModelBuilder.add_usd).parameters:
        pytest.skip("Static USD visual import requires Newton 1.6 (Isaac Sim 6.2)")
    before = stage.GetRootLayer().ExportToString()
    duplicate = newton.ModelBuilder()
    duplicate.add_usd(stage)
    duplicate.add_ground_plane()
    assert "/World/GroundPlane" in duplicate.shape_label
    assert "ground_plane" in duplicate.shape_label

    builder = newton.ModelBuilder()
    builder.add_usd(stage, ignore_paths=model_module._generated_ground_visual_paths(stage, {}))
    builder.add_ground_plane()
    assert builder.shape_label == ["/World/Prop", "ground_plane"]
    floor_index = builder.shape_label.index("ground_plane")
    flags = builder.shape_flags[floor_index]
    assert flags & newton.ShapeFlags.VISIBLE
    assert flags & newton.ShapeFlags.COLLIDE_SHAPES
    assert builder.shape_type[floor_index] == newton.GeoType.PLANE
    assert stage.GetRootLayer().ExportToString() == before
