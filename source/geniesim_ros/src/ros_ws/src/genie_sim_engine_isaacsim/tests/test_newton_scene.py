# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Schema checks run inside the explicit Isaac 6.2 smoke app."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="module", autouse=True)
def _isaac_runtime():
    if "omni.kit.app" not in sys.modules:
        pytest.skip("requires an initialized Isaac app")
    global Usd, UsdPhysics
    from pxr import Usd, UsdPhysics


from genie_sim_engine_isaacsim.kit.newton_config import configure_physics_scene, SCENE_ITERATIONS


@pytest.fixture
def scene():
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    return stage


def test_scene_owns_solver_and_iteration_configuration(scene):
    configure_physics_scene(scene, 23, SimpleNamespace(info=Mock()))
    prim = scene.GetPrimAtPath("/physicsScene")
    assert prim.HasAPI("MjcSceneAPI")
    assert prim.GetAttribute("newton:solver").Get() == "mujoco"
    assert prim.GetAttribute(SCENE_ITERATIONS).Get() == 23


def test_zero_iterations_preserves_the_scene_value(scene):
    configure_physics_scene(scene, 23, SimpleNamespace(info=Mock()))
    configure_physics_scene(scene, 0, SimpleNamespace(info=Mock()))
    assert scene.GetPrimAtPath("/physicsScene").GetAttribute(SCENE_ITERATIONS).Get() == 23


def test_ambiguous_scene_does_not_pick_arbitrarily():
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/World/a")
    UsdPhysics.Scene.Define(stage, "/World/b")
    with pytest.raises(RuntimeError, match="identifiable physics scene"):
        configure_physics_scene(stage, 23, SimpleNamespace(info=Mock()))


def test_legacy_solver_token_uses_the_public_api_interpretation(scene):
    configure_physics_scene(scene, 23, SimpleNamespace(info=Mock()))
    scene.GetPrimAtPath("/physicsScene").GetAttribute("newton:solver").Set("mjcwarp")
    configure_physics_scene(scene, 0, SimpleNamespace(info=Mock()))
    assert scene.GetPrimAtPath("/physicsScene").HasAPI("MjcSceneAPI")


def test_another_active_newton_scene_is_not_silently_selected(scene):
    from isaacsim.physics.newton import switch_newton_solver

    other = UsdPhysics.Scene.Define(scene, "/World/other").GetPrim()
    switch_newton_solver("mujoco", other)
    with pytest.raises(RuntimeError, match="Multiple Newton physics scenes"):
        configure_physics_scene(scene, 23, SimpleNamespace(info=Mock()))
