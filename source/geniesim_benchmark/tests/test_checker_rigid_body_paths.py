# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise task checkers with moving rigid bodies and stationary containers.

Set GENIESIM_ASSETS_PATH to also use the actual composed task scenes. Without
assets, a small USD hierarchy reproduces the same parent/body pose contract.
"""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from geniesim_benchmark.plugins.ader.action.action_parsing import parse_action
from geniesim_benchmark.plugins.ader.action.common_actions import quat_wxyz_to_rotation_matrix

PACKAGE = Path(__file__).resolve().parents[1] / "src/geniesim_benchmark"
LAYOUTS = PACKAGE / "benchmark/config/llm_task"
SCENARIOS = sorted(LAYOUTS.glob("keyboard_keycap_swap_*/*/problems.json")) + sorted(
    (LAYOUTS / "transport_cartons_to_table_wbc").glob("*/problems.json")
)


def api_for_stage(stage):
    # Import the production methods without requiring the Isaac/Kit runtime.
    path = PACKAGE / "app/controllers/api_core.py"
    tree = ast.parse(path.read_text())
    core = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "APICore")
    names = {"get_obj_world_pose", "get_obj_world_pose_matrix"}
    methods = [n for n in core.body if isinstance(n, ast.FunctionDef) and n.name in names]

    def pose(path):
        prim = stage.GetPrimAtPath(path)
        assert prim.IsValid(), path
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        q = matrix.ExtractRotationQuat()
        return np.array(matrix.ExtractTranslation()), np.array([q.GetReal(), *q.GetImaginary()])

    namespace = {
        "np": np,
        "get_current_stage": lambda: stage,
        "get_world_pose": pose,
        "get_rotation_matrix_from_quaternion": quat_wxyz_to_rotation_matrix,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    cls = type("PoseAPI", (), {name: namespace[name] for name in names})
    cls._env_path = lambda self, path: path.replace("/Workspace/", "/World/")
    return cls()


def set_world_pose(stage, path, position, quaternion=(1, 0, 0, 0)):
    prim = stage.GetPrimAtPath(path)
    parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world = Gf.Matrix4d(1)
    world.SetRotate(Gf.Rotation(Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*quaternion[1:]))))
    world.SetTranslateOnly(Gf.Vec3d(*position))
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(world * parent.GetInverse())


@pytest.mark.parametrize("problems", SCENARIOS, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_only_the_moving_body_determines_task_credit(problems):
    problem = json.loads(problems.read_text())["problem1"]
    keycap = problems.parent.parent.name.startswith("keyboard_keycap_swap_")
    checker_name = "PlaceOnRivet" if keycap else "InBBox"
    actions = problem["Acts"][0]["ActionList"]
    checker_config = actions[2]["ActionSetWaitAny"][0]
    assert checker_name in checker_config
    object_name = "replacement_keycap_000" if keycap else "benchmark_luggage_cart_002"
    root = "/World/Objects/" + object_name
    body = root + "/entity/body"
    assets = os.environ.get("GENIESIM_ASSETS_PATH")
    if assets:
        layer = Sdf.Layer.CreateAnonymous("checker-regression.usda")
        layer.ImportFromString(
            problems.with_name("scene.usda").read_text().replace("@/geniesim_assets/", "@" + assets.rstrip("/") + "/")
        )
        stage = Usd.Stage.Open(layer)
    else:
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, root + "/entity")
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, body).GetPrim())
        UsdGeom.Xform.Define(stage, "/World/Objects/keyboard_000/entity")
    assert not stage.GetPrimAtPath(root + "/entity").HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath(body).HasAPI(UsdPhysics.RigidBodyAPI)
    api = api_for_stage(stage)
    clock = SimpleNamespace(current_time=0.0)
    api.ui_builder = SimpleNamespace(my_world=clock)
    env = SimpleNamespace(
        api_core=api,
        init_task_config={"sub_task_name": problems.parent.parent.name},
        task=SimpleNamespace(update_progress=lambda *args: None),
    )
    checker = parse_action(checker_config, [], env)
    if keycap:
        goal = api.get_obj_world_pose("/World/Objects/keyboard_000")[0] + checker.target_rel_pos
        quaternion = checker.target_quat
        instructions = json.loads(problems.with_name("instructions.json").read_text())
        assert instructions["subtask_steps"][2]["checker"] == checker_config
        assert [row["points"] for row in problem["ScoreCheckers"][0]["PolicyDispatchBudget"]["steps"]] == [5, 20, 25]
        ticks = checker.still_steps + 1
    else:
        goal, quaternion, ticks = checker.bbox_center, (1, 0, 0, 0), 3
    far = goal + np.array([10.0, 10.0, 10.0])

    # A stationary container at the goal cannot earn credit for a distant body.
    set_world_pose(stage, root + "/entity", goal, quaternion)
    set_world_pose(stage, body, far, quaternion)
    checker.start()
    for _ in range(ticks):
        clock.current_time += 0.1
        checker.update(0.1)
    assert not checker.is_finished()

    # Move only the physical body to the goal; keep its container far away.
    set_world_pose(stage, root + "/entity", far, quaternion)
    set_world_pose(stage, body, goal, quaternion)
    for _ in range(ticks):
        clock.current_time += 0.1
        checker.update(0.1)
    assert checker.is_finished()
    assert checker.progress_info["SCORE"] == 1
    np.testing.assert_allclose(api.get_obj_world_pose(root)[0], far)
