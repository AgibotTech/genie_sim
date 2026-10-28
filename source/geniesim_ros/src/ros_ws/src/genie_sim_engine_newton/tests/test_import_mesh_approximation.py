# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Authored ``convexDecomposition`` must not reach Newton >= 1.4's import-time CoACD."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

_PATH = Path(__file__).resolve().parents[1] / "genie_sim_engine_newton/engine/newton/setup/model.py"
_SPEC = importlib.util.spec_from_file_location("_mesh_approximation_model_under_test", _PATH)
model_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(model_module)

PAD = "/genie/gripper_l_inner_link4/visuals/inner_link4/mesh"
CUP = "/World/cup/entity/mesh"
TAKEN = "/World/shelf/entity/mesh"
HULL = "/World/box/entity/mesh"
OFF = "/World/lid/entity/mesh"


def _collider(stage, path, approximation=None, enabled=True):
    mesh = UsdGeom.Mesh.Define(stage, path)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    if not enabled:
        UsdPhysics.CollisionAPI(mesh.GetPrim()).CreateCollisionEnabledAttr(False)
    if approximation is not None:
        mesh.GetPrim().CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set(approximation)
    return mesh.GetPrim()


@pytest.fixture
def stage():
    stage = Usd.Stage.CreateInMemory()
    for body in ("/genie/gripper_l_inner_link4", "/World/cup", "/World/shelf", "/World/box", "/World/lid"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, body).GetPrim())
    _collider(stage, PAD, "convexDecomposition")
    _collider(stage, CUP, "convexDecomposition")
    # Already taken over by the scene-collider scan, which blocks the opinion.
    _collider(stage, TAKEN, "convexDecomposition").GetAttribute("physics:approximation").Block()
    _collider(stage, HULL, "convexHull")
    _collider(stage, OFF, "convexDecomposition", enabled=False)
    return stage


def _approx(stage, path):
    attr = stage.GetPrimAtPath(path).GetAttribute("physics:approximation")
    return attr.Get() if attr.HasAuthoredValue() else None


def test_convex_decomposition_imports_become_one_hull(stage):
    assert model_module._pin_convex_decomposition_imports(stage) == 2

    assert _approx(stage, PAD) == "convexHull"
    assert _approx(stage, CUP) == "convexHull"


def test_other_opinions_are_left_alone(stage):
    model_module._pin_convex_decomposition_imports(stage)

    assert _approx(stage, TAKEN) is None
    assert _approx(stage, HULL) == "convexHull"
    assert _approx(stage, OFF) == "convexDecomposition"
