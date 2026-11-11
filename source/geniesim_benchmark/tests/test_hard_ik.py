# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""SDK IK/FK against the shipped embodiment models, without a simulator."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from geniesim_benchmark.utils.name_utils import ROBOT_CONFIGS

PACKAGE = Path(__file__).resolve().parents[1] / "src/geniesim_benchmark"


def solver_for(robot):
    from geniesim_benchmark.utils.hard_ik import HardIKFKSolver

    return HardIKFKSolver(robot_cfg=robot)


def action_for(solver, joints):
    poses = solver.compute_eef(joints)
    return np.concatenate(
        [np.r_[poses[s][:3], R.from_quat(np.array(poses[s])[[4, 5, 6, 3]]).as_euler("xyz")] for s in ("left", "right")]
        + [[0.25, 0.75]]
    )


@pytest.mark.parametrize("robot", ROBOT_CONFIGS)
def test_every_embodiment_reaches_fk_targets_within_limits(robot):
    solver = solver_for(robot)
    rng = np.random.default_rng(23)
    for _ in range(3):
        target = rng.uniform(solver.lower * 0.2 + solver.upper * 0.8, solver.lower * 0.1 + solver.upper * 0.9)
        seed = np.clip(target + rng.uniform(-0.08, 0.08, target.size), solver.lower, solver.upper)
        action = action_for(solver, target)
        result = np.asarray(solver.eef_actions_to_joint([action], seed, [])[0])
        assert result.size == target.size + 2
        assert np.all(result[:-2] >= solver.lower)
        assert np.all(result[:-2] <= solver.upper)
        np.testing.assert_equal(result[-2:], action[-2:])
        actual = solver.compute_eef(result[:-2])
        expected = solver.compute_eef(target)
        for side in ("left", "right"):
            assert np.linalg.norm(np.subtract(actual[side][:3], expected[side][:3])) <= 1e-5
            a = R.from_quat(np.array(actual[side])[[4, 5, 6, 3]])
            b = R.from_quat(np.array(expected[side])[[4, 5, 6, 3]])
            assert (a * b.inv()).magnitude() <= 1e-4


def test_unreachable_pose_is_rejected_without_returning_approximate_motion():
    from geniesim_benchmark.utils.hard_ik import IKConvergenceError

    solver = solver_for("G2_crsB_wuji")
    seed = (solver.lower + solver.upper) / 2
    target = action_for(solver, seed)
    target[0] = 10
    with pytest.raises(IKConvergenceError, match="left.*position"):
        solver.eef_actions_to_joint([target], seed, [])
    # A failed request must not poison a later request on the shared solver.
    result = solver.eef_actions_to_joint([action_for(solver, seed)], seed, [])[0]
    np.testing.assert_allclose(result[:-2], seed, atol=1e-7)


def policy_method(name):
    path = PACKAGE / "benchmark/policy/corobotpolicy.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CoRobotPolicy")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"np": np, "R": R, "logger": SimpleNamespace(info=lambda *a: None)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    return ns[name], ns


def test_eef_without_solver_cannot_be_reinterpreted_as_joints():
    method, ns = policy_method("_post_process_action")
    ns["process_action"] = lambda *a, **kw: pytest.fail("EEF reached joint path")
    policy = SimpleNamespace(_ikfk_solver=None)
    with pytest.raises(RuntimeError, match="IK"):
        method(policy, {"kind": "EEF_ABS", "arm": np.zeros(14), "gripper": [0, 0]}, np.zeros(14))


@pytest.mark.parametrize("robot", ROBOT_CONFIGS)
def test_all_backend_names_use_one_shared_sdk_solver(robot):
    from geniesim_benchmark.utils.ikfk_utils import get_shared_ikfk_solver
    from geniesim_benchmark.utils.hard_ik import HardIKFKSolver

    expected = get_shared_ikfk_solver([], [], [], robot, "auto")
    assert isinstance(expected, HardIKFKSolver)
    for backend in ("ik_solver", "hard", "ik_sdk", "curobo"):
        assert get_shared_ikfk_solver([], [], [], robot, backend) is expected


def test_wuji_eef_keeps_all_forty_finger_targets():
    method, ns = policy_method("_post_process_action")
    from geniesim_benchmark.utils.infer_post_process import process_gripper_action_passthrough

    solver = solver_for("G2_crsB_wuji")
    q = (solver.lower + solver.upper) / 2
    poses = solver.compute_eef(q)
    arm = np.concatenate([np.array(poses[s])[[0, 1, 2, 4, 5, 6, 3]] for s in ("left", "right")])
    policy = SimpleNamespace(
        _ikfk_solver=solver,
        _arm_dim=14,
        _gripper_dim=40,
        _robot_config={"gripper_offset": 0},
        _process_gripper_action=process_gripper_action_passthrough,
    )
    fingers = np.linspace(-0.1, 0.2, 40)
    result = method(policy, {"kind": "EEF_ABS", "arm": arm, "gripper": fingers, "eef_frame": "arm_base_link"}, q)
    np.testing.assert_allclose(result["gripper"], fingers)


def test_hard_ik_is_the_default_for_g2op_too():
    from geniesim_benchmark.utils.hard_ik import HardIKFKSolver
    from geniesim_benchmark.utils.ikfk_utils import get_shared_ikfk_solver

    assert isinstance(get_shared_ikfk_solver([], [], [], "G2_omnipicker"), HardIKFKSolver)


@pytest.mark.parametrize("robot", ["G2", "G2_omnipicker", "G1_omnipicker", "arx"])
def test_legacy_constructor_uses_the_same_sdk_implementation(robot):
    from geniesim_benchmark.utils.ikfk_utils import IKFKSolver
    from geniesim_benchmark.utils.hard_ik import HardIKFKSolver

    assert IKFKSolver is HardIKFKSolver
    solver = IKFKSolver([], [], [], robot)
    q = (solver.lower + solver.upper) / 2
    for side, pose in solver.compute_eef(q).items():
        expected = solver_for("G2_omnipicker" if robot == "G2" else robot).compute_eef(q)[side]
        np.testing.assert_allclose(pose, expected, atol=1e-12)


def test_converter_uses_shared_solver_public_ik_fk_methods():
    from geniesim_benchmark.utils.g2_ikfk_converter import joints_to_eef, eef_to_joints

    solver = solver_for("G2_omnipicker")
    seed = (solver.lower + solver.upper) / 2
    target = seed + 0.02
    left, right = joints_to_eef(solver, target[:7], target[7:])
    solved = np.concatenate(eef_to_joints(solver, left, right, seed))
    actual = solver.compute_eef(solved)
    expected = solver.compute_eef(target)
    for side in ("left", "right"):
        np.testing.assert_allclose(actual[side], expected[side], atol=1e-5)


def test_hard_ik_rejects_unknown_robot_and_backend():
    from geniesim_benchmark.utils.ikfk_utils import get_shared_ikfk_solver

    with pytest.raises(ValueError):
        get_shared_ikfk_solver([], [], [], "unknown_robot")
    with pytest.raises(ValueError):
        get_shared_ikfk_solver([], [], [], "G2_crsB_wuji", "unknown_solver")


@pytest.mark.parametrize("robot", ROBOT_CONFIGS)
def test_joint_control_needs_no_solver_and_keeps_all_channels(robot):
    from geniesim_benchmark.utils.infer_post_process import process_action

    method, ns = policy_method("_post_process_action")
    ns["process_action"] = process_action
    cfg = ROBOT_CONFIGS[robot]
    arm = np.linspace(0.1, 0.2, len(cfg["arm_joints"]))
    fingers = np.linspace(0, 0.2, len(cfg["gripper_joints"]))
    policy = SimpleNamespace(
        _ikfk_solver=None,
        _arm_dim=len(arm),
        _gripper_dim=len(fingers),
        _robot_config=cfg,
        _process_gripper_action=cfg["process_gripper_action"],
    )
    result = method(policy, {"kind": "JOINT_ABS", "arm": arm, "gripper": fingers}, np.zeros_like(arm))
    np.testing.assert_allclose(result["arm"], arm)
    np.testing.assert_allclose(result["gripper"], cfg["process_gripper_action"](fingers, cfg))


def test_arx_eef_in_base_link_requires_no_nonexistent_arm_base_link():
    solver = solver_for("arx")
    q = (solver.lower + solver.upper) / 2
    poses = solver.compute_eef(q)
    arm = np.concatenate([np.array(poses[s])[[0, 1, 2, 4, 5, 6, 3]] for s in ("left", "right")])
    method, ns = policy_method("_post_process_action")
    cfg = ROBOT_CONFIGS["arx"]
    policy = SimpleNamespace(
        _ikfk_solver=solver,
        _arm_dim=12,
        _gripper_dim=2,
        _robot_config=cfg,
        _process_gripper_action=cfg["process_gripper_action"],
    )
    result = method(policy, {"kind": "EEF_ABS", "arm": arm, "gripper": [0, 0]}, q)
    np.testing.assert_allclose(result["arm"], q, atol=1e-6)


def test_eef_rejects_conflicting_per_arm_frames():
    method, _ = policy_method("_parse_result")
    with pytest.raises(ValueError, match="frames must match"):
        method(
            {
                "left_arm": {"kind": "EEF_ABS", "base_link": "base_link"},
                "right_arm": {"kind": "EEF_ABS", "base_link": "arm_base_link"},
            }
        )


def test_newton_wuji_accepts_eef_and_preserves_fingers():
    from geniesim_benchmark.newton.corobot import parse_result, WUJI_GROUPS

    solver = solver_for("G2_crsB_wuji")
    q = (solver.lower + solver.upper) / 2
    fk = solver.compute_eef(q)
    reply = {
        "result": {
            s
            + "_arm": {
                "kind": "EEF_ABS",
                "base_link": "arm_base_link",
                "values": [np.array(fk[s])[[0, 1, 2, 4, 5, 6, 3]].tolist()],
            }
            for s in ("left", "right")
        }
    }
    reply["result"].update(left_effector=[[0.1] * 20], right_effector=[[0.2] * 20])
    names = WUJI_GROUPS.all_joints
    initial = dict.fromkeys(names, 0.0)
    initial.update(zip(solver.joint_names, q))
    chunk = parse_result(
        reply, controllable_joints=names, held_positions=initial, initial_positions=initial, ik_solver=solver
    )
    assert chunk.targets[0]["hand_l_thumb_joint1"] == 0.1
    np.testing.assert_allclose([chunk.targets[0][name] for name in solver.joint_names], q, atol=1e-6)


def test_rubik_wuji_accepts_eef_and_preserves_fingers():
    from geniesim_benchmark.backends.mujoco.rubik.policy import parse_actions

    solver = solver_for("G2_crsB_wuji")
    q = (solver.lower + solver.upper) / 2
    fk = solver.compute_eef(q)
    reply = {
        "result": {
            s
            + "_arm": {
                "kind": "EEF_ABS",
                "base_link": "arm_base_link",
                "values": [np.array(fk[s])[[0, 1, 2, 4, 5, 6, 3]].tolist()],
            }
            for s in ("left", "right")
        }
    }
    reply["result"].update(left_effector=[[0.1] * 20], right_effector=[[0.2] * 20])
    contract = {"bounds": np.r_[np.c_[solver.lower, solver.upper], [[-1, 1]] * 40].tolist()}
    actions, done = parse_actions(
        reply, contract, 10, measured_state={"arm_joint_states": q.tolist()}, ik_solver=solver
    )
    assert not done
    np.testing.assert_allclose(actions[0, :14], q, atol=1e-6)
    np.testing.assert_allclose(actions[0, 14:34], 0.1)
    np.testing.assert_allclose(actions[0, 34:], 0.2)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_ik_input_is_rejected(value):
    solver = solver_for("arx")
    seed = (solver.lower + solver.upper) / 2
    action = action_for(solver, seed)
    action[0] = value
    with pytest.raises(ValueError, match="finite"):
        solver.eef_actions_to_joint([action], seed, [])


@pytest.mark.parametrize("robot", [name for name in ROBOT_CONFIGS if name != "arx"])
def test_base_link_targets_with_rotated_translated_torso_reach_same_pose(robot):
    from geniesim_benchmark.utils.eef_control import eef_poses_to_joints

    solver = solver_for(robot)
    seed = (solver.lower + solver.upper) / 2
    goal = seed + 0.02
    poses = solver.compute_eef(goal)
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
    transform[:3, 3] = [0.4, -0.1, 0.8]
    base_targets, arm_targets = [], []
    for side in ("left", "right"):
        pose = np.array(poses[side])
        rotation = R.from_quat(pose[[4, 5, 6, 3]])
        position = transform[:3, :3] @ pose[:3] + transform[:3, 3]
        base_targets.append([np.r_[position, (R.from_matrix(transform[:3, :3]) * rotation).as_quat()]])
        arm_targets.append([np.r_[pose[:3], rotation.as_quat()]])
    # Both representations of the same physical target must recover it.
    # A nonzero joint displacement also exercises SDK IK, beyond seed acceptance.
    for frame, targets in (("base_link", base_targets), ("arm_base_link", arm_targets)):
        actual = eef_poses_to_joints(solver, *targets, seed, frame, transform)
        observed = solver.compute_eef(actual[0])
        for side in ("left", "right"):
            assert np.linalg.norm(np.subtract(observed[side][:3], poses[side][:3])) <= 1e-5
            a = R.from_quat(np.array(observed[side])[[4, 5, 6, 3]])
            b = R.from_quat(np.array(poses[side])[[4, 5, 6, 3]])
            assert (a * b.inv()).magnitude() <= 1e-4


@pytest.mark.parametrize("per_arm", [False, True])
@pytest.mark.parametrize("frame", ["base_link", "arm_base_link"])
def test_result_frame_declaration_reaches_policy_action(per_arm, frame):
    parse, _ = policy_method("_parse_result")
    reply = {side + "_arm": {"kind": "EEF_ABS", "values": [[0, 0, 1, 0, 0, 0, 1]]} for side in ("left", "right")}
    reply.update(left_effector=[[0]], right_effector=[[0]])
    if per_arm:
        for side in ("left", "right"):
            reply[side + "_arm"]["base_link"] = frame
        reply["base_link"] = "base_link" if frame == "arm_base_link" else "arm_base_link"
    else:
        reply["base_link"] = frame
    assert parse(reply)[0]["eef_frame"] == frame


def test_base_link_target_without_required_transform_is_rejected():
    from geniesim_benchmark.utils.eef_control import eef_poses_to_joints

    solver = solver_for("G2_omnipicker")
    q = (solver.lower + solver.upper) / 2
    poses = solver.compute_eef(q)
    targets = [[np.array(poses[side])[[0, 1, 2, 4, 5, 6, 3]]] for side in ("left", "right")]
    with pytest.raises(RuntimeError, match="arm_base_transform"):
        eef_poses_to_joints(solver, *targets, q, "base_link")


def test_live_link_poses_recover_the_torso_transform_with_base_rotation():
    from geniesim_benchmark.utils.ikfk_utils import arm_base_transform_from_poses

    world_base = np.eye(4)
    world_base[:3, :3] = R.from_euler("xyz", [-0.1, 0.2, 1.3]).as_matrix()
    world_base[:3, 3] = [3.0, -2.0, 0.05]
    base_arm = np.eye(4)
    base_arm[:3, :3] = R.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
    base_arm[:3, 3] = [0.1, 0.0, 1.2]
    world_arm = world_base @ base_arm
    quaternions = [R.from_matrix(t[:3, :3]).as_quat()[[3, 0, 1, 2]] for t in (world_base, world_arm)]
    actual = arm_base_transform_from_poses(world_base[:3, 3], quaternions[0], world_arm[:3, 3], quaternions[1])
    np.testing.assert_allclose(actual, base_arm, atol=1e-12)


def test_eef_chunk_uses_previous_solution_and_rejects_bad_quaternions():
    from geniesim_benchmark.utils.eef_control import eef_poses_to_joints

    solver = solver_for("arx")
    seed = (solver.lower + solver.upper) / 2
    q = seed + 0.01
    frames = [solver.compute_eef(row) for row in (seed, q)]
    poses = [np.array([f[side] for f in frames])[:, [0, 1, 2, 4, 5, 6, 3]] for side in ("left", "right")]
    actual = eef_poses_to_joints(solver, *poses, seed, "base_link")
    for row, target in zip(actual, (seed, q)):
        for side in ("left", "right"):
            np.testing.assert_allclose(solver.compute_eef(row)[side], solver.compute_eef(target)[side], atol=1e-5)
    poses[0][0, 3:] = 0
    with pytest.raises(ValueError, match="quaternion"):
        eef_poses_to_joints(solver, *poses, seed, "base_link")


def test_legacy_curobo_import_also_uses_hard_ik():
    from geniesim_benchmark.utils.curobo_ikfk_utils import CuroboIKFKSolver
    from geniesim_benchmark.utils.hard_ik import HardIKFKSolver

    assert isinstance(CuroboIKFKSolver([], [], [], "G1_omnipicker"), HardIKFKSolver)


def test_hard_ik_import_defers_isaac_assets_and_native_sdk():
    import subprocess
    import sys

    code = """
import importlib.abc, sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'geniesim_assets', 'ik_solver', 'isaacsim', 'omni', 'torch', 'curobo'}:
            raise ImportError('Forbidden runtime dependency: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
from geniesim_benchmark.utils.hard_ik import HardIKFKSolver
from geniesim_benchmark.utils.eef_control import eef_poses_to_joints
if importlib.util.find_spec("mujoco") is not None:
    from geniesim_benchmark.backends.mujoco.rubik.kinematics import NativeIKFKSolver
HardIKFKSolver(robot_cfg='arx')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_hard_ik_and_fk_both_execute_through_the_g2op_wheel(monkeypatch):
    import ik_solver

    calls = {"fk": 0, "ik": 0}
    original_fk, original_solve = ik_solver.Solver.compute_fk, ik_solver.Solver.solve

    def fk(self, joints):
        calls["fk"] += 1
        return original_fk(self, joints)

    def solve(self):
        calls["ik"] += 1
        return original_solve(self)

    monkeypatch.setattr(ik_solver.Solver, "compute_fk", fk)
    monkeypatch.setattr(ik_solver.Solver, "solve", solve)
    solver = solver_for("arx")
    seed = (solver.lower + solver.upper) / 2
    target = action_for(solver, seed + 0.02)
    solver.eef_actions_to_joint([target], seed, [])
    assert calls["fk"] > 0, "FK must use ik_solver.compute_fk"
    assert calls["ik"] > 0, "IK must use ik_solver.solve"


def test_sdk_approximate_solution_is_rejected(monkeypatch):
    import ik_solver
    from geniesim_benchmark.utils.hard_ik import IKConvergenceError

    solver = solver_for("arx")
    seed = (solver.lower + solver.upper) / 2
    target = action_for(solver, seed + 0.05)
    monkeypatch.setattr(ik_solver.Solver, "solve", lambda self: self.get_current_state())
    with pytest.raises(IKConvergenceError, match="no command returned"):
        solver.eef_actions_to_joint([target], seed, [])


def test_shared_sdk_solver_keeps_parallel_targets_isolated():
    from concurrent.futures import ThreadPoolExecutor

    solver = solver_for("G2_crsB_90d")
    base = (solver.lower + solver.upper) / 2

    def run(index):
        target = base + index * 0.005
        seed = target + 0.01
        result = solver.eef_actions_to_joint([action_for(solver, target)], seed, [])[0]
        for side in ("left", "right"):
            np.testing.assert_allclose(
                solver.compute_eef(result[:-2])[side], solver.compute_eef(target)[side], atol=1e-5
            )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(12)))
