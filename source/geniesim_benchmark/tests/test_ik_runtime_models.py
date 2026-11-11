# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Optional checks against installed assets: set GENIESIM_IK_TEST_ASSETS."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from geniesim_benchmark.utils.name_utils import ROBOT_CONFIGS
from test_hard_ik import action_for


@pytest.fixture
def assets():
    value = os.environ.get("GENIESIM_IK_TEST_ASSETS")
    if not value:
        pytest.skip("Set GENIESIM_IK_TEST_ASSETS to verify actual robot assets")
    return Path(value)


@pytest.mark.parametrize("robot", ROBOT_CONFIGS)
def test_usd_model_ik_reaches_observed_link_frame(assets, robot):
    from pxr import Usd, UsdGeom, UsdPhysics
    from geniesim_benchmark.utils.usd_ik import UsdIKFKSolver

    cfgdir = Path(__file__).resolve().parents[1] / "src/geniesim_benchmark/app/robot_cfg"
    cfg = json.loads((cfgdir / f"{robot}.json").read_text())["robot"]
    path = assets / cfg["robot_usd"]
    if not path.exists():
        pytest.skip(f"Asset is unavailable: {cfg['robot_usd']}")
    solver = UsdIKFKSolver(robot, path)
    # USD-authored rest transforms are an independent reference for FK.
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    links = {prim.GetName(): prim for prim in stage.Traverse()}
    base = np.array(cache.GetLocalToWorldTransform(links[solver.frame])).T
    # Some files author a slightly nonzero rest pose. Recover the actual
    # joint angle from the independent body transforms and joint frames.
    measured = {}
    for prim in stage.Traverse():
        if prim.GetName() not in solver.joint_names or not prim.IsA(UsdPhysics.Joint):
            continue
        parent = prim.GetRelationship("physics:body0").GetTargets()[0]
        child = prim.GetRelationship("physics:body1").GetTargets()[0]
        frames = []
        for index in (0, 1):
            quat = prim.GetAttribute(f"physics:localRot{index}").Get()
            matrix = np.eye(4)
            matrix[:3, :3] = R.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()
            matrix[:3, 3] = prim.GetAttribute(f"physics:localPos{index}").Get()
            frames.append(matrix)
        parent_world = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent))).T
        child_world = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(child))).T
        motion = np.linalg.inv(parent_world @ frames[0]) @ child_world @ frames[1]
        axis = "XYZ".index(prim.GetAttribute("physics:axis").Get())
        a, b = (axis + 1) % 3, (axis + 2) % 3
        measured[prim.GetName()] = np.arctan2(motion[b, a] - motion[a, b], motion[a, a] + motion[b, b])
    poses = solver.compute_eef([measured[name] for name in solver.joint_names])
    for side, suffix in (("left", "fl"), ("right", "fr")):
        end = f"arm_{suffix}_end_link" if robot == "agilex" else f"arm_{side[0]}_end_link"
        expected = np.linalg.inv(base) @ np.array(cache.GetLocalToWorldTransform(links[end])).T
        # Authored body poses need not exactly close every joint constraint:
        # the G1 rest asset differs by 0.17 mm / 1 mrad before physics resolves it.
        np.testing.assert_allclose(poses[side][:3], expected[:3, 3], atol=5e-4)
        actual_rotation = R.from_quat(np.array(poses[side])[[4, 5, 6, 3]])
        assert (R.from_matrix(expected[:3, :3]) * actual_rotation.inv()).magnitude() < 2e-3
    q = (solver.lower + solver.upper) / 2
    target = q + 0.02
    actions = solver.eef_actions_to_joint([action_for(solver, target)], q, [])
    for side in ("left", "right"):
        np.testing.assert_allclose(
            solver.compute_eef(actions[0][:-2])[side], solver.compute_eef(target)[side], atol=1e-5
        )


def test_native_model_uses_actual_fk_and_never_mutates_simulation(assets):
    mujoco = pytest.importorskip("mujoco")
    from geniesim_benchmark.backends.mujoco.rubik.kinematics import NativeIKFKSolver
    from geniesim_benchmark.backends.mujoco.rubik.policy import parse_actions

    root = assets / "mujoco/rubik_cube"
    model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = json.loads((root / "initial_state.json").read_text())["qpos"]
    mujoco.mj_kinematics(model, data)
    arm_joints = np.array([model.joint(f"Joint{i}_{side}").id for side in ("L", "R") for i in range(1, 8)])
    arm_q = model.jnt_qposadr[arm_joints]
    solver = NativeIKFKSolver(model, data, arm_q, model.jnt_range[arm_joints])
    before = data.qpos.copy()
    q = before[arm_q]
    observed = solver.compute_eef(q)
    base = model.body("arm_base_link").id
    rotation = data.xmat[base].reshape(3, 3)
    for side, letter in (("left", "l"), ("right", "r")):
        end = model.body(f"arm_{letter}_end_link").id
        np.testing.assert_allclose(observed[side][:3], rotation.T @ (data.xpos[end] - data.xpos[base]), atol=1e-10)
    # Shift the target so this also exercises actual numerical convergence.
    target = np.clip(q + 0.005, solver.lower, solver.upper)
    poses = solver.compute_eef(target)
    result = {
        side
        + "_arm": {
            "kind": "EEF_ABS",
            "base_link": "arm_base_link",
            "values": [np.array(poses[side])[[0, 1, 2, 4, 5, 6, 3]].tolist()],
        }
        for side in ("left", "right")
    }
    result.update(left_effector=[[0.1] * 20], right_effector=[[0.2] * 20])
    contract = {"bounds": np.r_[model.jnt_range[arm_joints], [[-1, 1]] * 40]}
    actual, done = parse_actions(
        {"result": result}, contract, 10, measured_state={"arm_joint_states": q.tolist()}, ik_solver=solver
    )
    for side in ("left", "right"):
        np.testing.assert_allclose(solver.compute_eef(actual[0, :14])[side], poses[side], atol=1e-5)
    np.testing.assert_equal(data.qpos, before)
    assert not done


@pytest.mark.parametrize("backend", ["auto", "ik_solver", "hard", "ik_sdk", "curobo"])
def test_backend_aliases_preserve_the_runtime_g2op_model(assets, backend):
    from geniesim_benchmark.utils.ikfk_utils import get_shared_ikfk_solver
    from geniesim_benchmark.utils.usd_ik import UsdIKFKSolver

    path = assets / "robot/G2_omnipicker/robot_fix.usda"
    solver = get_shared_ikfk_solver([], [], [], "G2_omnipicker", backend, robot_usd_path=path)
    expected = get_shared_ikfk_solver([], [], [], "G2_omnipicker", "auto", robot_usd_path=path)
    assert isinstance(solver, UsdIKFKSolver)
    assert solver is expected
