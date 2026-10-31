# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Analytic robot geometry and lifecycle checks; no simulator/model calls."""

import copy
import math
from dataclasses import replace

import pytest

from agent2act.adapters.kinematics.numerical import NumericalIK
from agent2act.adapters.kinematics.urdf import UrdfKinematics
from agent2act.adapters.motion.urdf_joint import UrdfJointMotion
from agent2act.bootstrap import ApplicationFactory, _joint_motion, default_registry
from agent2act.configuration import parse_settings
from agent2act.domain.errors import ConfigurationError, LocalIKConvergenceError
from agent2act.domain.models import ContractError, Decision
from agent2act.domain.timing import MotionTiming
from agent2act.domain.validation import ActionValidator


@pytest.fixture
def model(tmp_path, spec, observation):
    path = tmp_path / "robot.urdf"
    path.write_text(f'''<robot>
    <joint name="{spec.left_joints[0]}" type="prismatic">
    <parent link="base_link"/><child link="left"/><axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" velocity="0.1"/></joint>
    <joint name="{spec.right_joints[0]}" type="prismatic">
    <parent link="base_link"/><child link="right"/><axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" velocity="0.2"/></joint></robot>''')
    options = dict(path=str(path), frame="base_link", left_link="left", right_link="right")
    fk = UrdfKinematics(spec, options)
    obs = replace(
        observation,
        end_pose={
            "base_link": {
                side + "_arm": dict(position=p[:3], orientation=p[3:])
                for side, p in fk.poses(observation.joints, []).items()
            }
        },
    )
    fk.close()
    return options, obs


def intent(obs, **waypoint):
    return Decision(
        "motion", obs.instruction, "analytic test", motion=dict(frame="base_link", waypoints=[waypoint])
    )


def test_solves_and_times_actual_joint_path(spec, model):
    options, obs = model
    compiler = _joint_motion(spec, MotionTiming(), options)
    target = [0.1, 0, 0, 0, 0, 0, 1]
    result = compiler.generate(obs, intent(obs, left_pose=target))
    assert result.kind == "JOINT_ABS"
    assert result.left_arm[-1][0] == pytest.approx(0.1, abs=1e-5)
    assert result.right_arm == [[0]] * result.steps
    assert result.steps > result.metadata["local_ik"]["source_steps"]
    rows = [obs.joints[0]] + [row[0] for row in result.left_arm]
    assert max(abs(b - a) for a, b in zip(rows, rows[1:])) <= 0.1 * spec.dt_s + 1e-9
    assert result.metadata["local_ik"]["generated_endpoint_fk"]["left"] == pytest.approx(target, abs=1e-5)
    path = result.metadata["cartesian_path"]
    assert len(path["left"]) == len(path["right"]) == result.steps
    assert path["left"][-1] == result.metadata["local_ik"]["generated_endpoint_fk"]["left"]
    # Each row's FK matches its own joint command, so any issued prefix has a Cartesian endpoint.
    assert [pose[0] for pose in path["left"]] == pytest.approx([row[0] for row in result.left_arm], abs=1e-9)
    ActionValidator(compiler.spec).validate(result, obs)
    compiler.reset()
    compiler.close()
    compiler.close()
    with pytest.raises(RuntimeError, match="closed"):
        compiler.generate(obs, intent(obs))


@pytest.mark.parametrize("duration,steps", [(None, 10), (0.15, 3)])
def test_disabled_limits_skip_urdf_joint_fk_and_gripper_retiming(spec, model, duration, steps):
    options, obs = model
    spec.enforce_speed_limits = False
    spec.max_joint_speed = spec.max_eef_speed = spec.max_eef_angular_speed = spec.max_gripper_speed = 1e-6
    compiler = _joint_motion(spec, MotionTiming(speed_fraction=0.25), options)
    try:
        wp = {"left_pose": [0.2, 0, 0, 0, 0, 0, 1], "grippers": [0.9, 0.6]}
        if duration is not None:
            wp["duration_s"] = duration
        chunk = compiler.generate(obs, intent(obs, **wp))
        assert chunk.steps == steps
        assert chunk.left_arm[-1][0] == pytest.approx(0.2, abs=1e-5)
        assert chunk.grippers[-1] == pytest.approx(wp["grippers"])
        assert compiler.describe()["compiler"]["speed_limits_enforced"] is False
        ActionValidator(compiler.spec).validate(chunk, obs)
        with pytest.raises(LocalIKConvergenceError):
            compiler.generate(obs, intent(obs, left_pose=[0.6, 0, 0, 0, 0, 0, 1]))
        compiler.spec.max_chunk_steps = 2
        with pytest.raises(ContractError, match="max_chunk_steps|duration out of range"):
            compiler.generate(obs, intent(obs, **wp))
    finally:
        compiler.close()


@pytest.mark.parametrize("overrun,accepted", [(1e-6, True), (2e-5, False)])
def test_disabled_recovery_speed_checks_preserve_ik_accuracy(spec, model, overrun, accepted):
    options, obs = model
    spec.enforce_speed_limits = False
    spec.max_joint_speed = spec.max_eef_speed = 1e-7
    compiler = _joint_motion(spec, MotionTiming(), options)
    obs.joints[0] = 0.5 + overrun
    obs.end_pose["base_link"]["left_arm"]["position"][0] = obs.joints[0]
    before = copy.deepcopy(obs)
    try:
        if accepted:
            chunk = compiler.generate(obs, intent(obs))
            assert chunk.left_arm[0][0] == 0.5
            ActionValidator(compiler.spec).validate(chunk, obs)
        else:
            with pytest.raises(ContractError, match="start is outside joint bounds"):
                compiler.generate(obs, intent(obs))
        assert obs == before
    finally:
        compiler.close()


def test_gripper_only_holds_joints_exactly_and_preserves_command_reference(spec, model):
    options, obs = model
    obs.issued_gripper_targets = [0.8, 0.6]
    compiler = _joint_motion(spec, MotionTiming(target_hold_s=0.2), options)
    before = copy.deepcopy(obs)
    result = compiler.generate(obs, intent(obs, grippers=[0.9, 0.6]))
    assert result.left_arm == [[0]] * result.steps
    assert result.grippers[-1] == [0.9, 0.6]
    assert result.grippers[0][0] >= 0.8
    assert result.metadata["target_hold_steps"] == 4
    assert obs == before


@pytest.mark.parametrize("target", [[0, 0.1, 0, 0, 0, 0, 1], [0.51, 0, 0, 0, 0, 0, 1]])
def test_unreachable_target_or_joint_bound_rejected(spec, model, target):
    options, obs = model
    compiler = _joint_motion(spec, MotionTiming(), options)
    with pytest.raises(LocalIKConvergenceError, match="did not converge"):
        compiler.generate(obs, intent(obs, left_pose=target))


def test_capacity_is_not_increased_for_joint_timing(spec, model):
    options, obs = model
    spec.max_chunk_steps = 20
    compiler = _joint_motion(spec, MotionTiming(), options)
    with pytest.raises(ContractError, match="max_chunk_steps after joint timing"):
        compiler.generate(obs, intent(obs, left_pose=[0.1, 0, 0, 0, 0, 0, 1]))


def test_bounds_intersect_and_start_outside_is_rejected(spec, model):
    options, obs = model
    spec.joint_limits = [[-0.05, 0.05], [-0.1, 0.1]]
    compiler = _joint_motion(spec, MotionTiming(), options)
    assert compiler.spec.joint_limits == spec.joint_limits
    with pytest.raises(ContractError, match="did not converge"):
        compiler.generate(obs, intent(obs, left_pose=[0.08, 0, 0, 0, 0, 0, 1]))
    obs.joints[0] = 0.06
    obs.end_pose["base_link"]["left_arm"]["position"][0] = 0.06
    with pytest.raises(ContractError, match="start is outside"):
        compiler.generate(obs, intent(obs))


def test_geometry_mismatch_and_unknown_constraints_rejected(spec, model):
    options, obs = model
    compiler = _joint_motion(spec, MotionTiming(), options)
    with pytest.raises(ContractError, match="unknown waypoint"):
        compiler.generate(obs, intent(obs, collision_free=True))
    obs.end_pose["base_link"]["left_arm"]["position"][0] = 0.1
    with pytest.raises(ContractError, match="FK observation mismatch"):
        compiler.generate(obs, intent(obs))


def test_rotation_ik_handles_quaternion_sign(tmp_path, spec):
    p = tmp_path / "rot.urdf"
    p.write_text(f'''<robot><joint name="{spec.left_joints[0]}" type="revolute"><parent link="base_link"/>
    <child link="left"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1"/></joint>
    <joint name="{spec.right_joints[0]}" type="revolute"><parent link="base_link"/><child link="right"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1"/></joint></robot>''')
    fk = UrdfKinematics(spec, dict(path=str(p), frame="base_link", left_link="left", right_link="right"))
    ik = NumericalIK(fk, [[-1, 1], [-1, 1]])
    target = [0, 0, 0, 0, 0, -math.sin(0.15), -math.cos(0.15)]
    result = ik.solve(target, [0, 0], [], "left")
    assert result == pytest.approx([0.3, 0], abs=1e-4)


def test_constructor_failure_closes_created_components(spec, model, monkeypatch):
    options, _ = model
    closed = []
    original = UrdfKinematics.close

    def close(self):
        closed.append(self)
        original(self)

    monkeypatch.setattr(UrdfKinematics, "close", close)
    monkeypatch.setattr(
        UrdfKinematics, "joint_constraints", lambda *_: (_ for _ in ()).throw(ContractError("missing limits"))
    )
    with pytest.raises(ContractError, match="missing limits"):
        _joint_motion(spec, MotionTiming(), options)
    assert len(closed) == 1 and closed[0].closed


def test_registry_selection_and_later_failure_cleanup(config, spec, model):
    options, _ = model
    config["motion_backend"] = dict(type="test_joint", **options)
    config["kinematics"] = dict(type="missing")
    registry = default_registry()
    created = []

    def build(s, t, o):
        compiler = _joint_motion(s, t, o)
        created.append(compiler)
        return compiler

    registry.motions.register("test_joint", build)
    with pytest.raises(ConfigurationError, match="unknown kinematics"):
        ApplicationFactory(parse_settings(config), registry).create_session()
    assert len(created) == 1 and created[0].closed and created[0].fk.closed


def test_default_motion_rejects_unknown_backend_options(config):
    config["motion_backend"] = dict(type="local_interpolation", unused=True)
    with pytest.raises(ConfigurationError, match="accepts no backend options"):
        ApplicationFactory(parse_settings(config)).create_session()


def test_fk_construction_failure_closes_injected_executor(spec):
    class Compiler:
        closed = False

        def close(self):
            self.closed = True

    compiler = Compiler()
    with pytest.raises(ContractError, match="requires path"):
        UrdfJointMotion(spec, MotionTiming(), {}, compiler)
    assert compiler.closed


def test_body_hold_and_full_tail_rejection(spec, model):
    options, obs = model
    spec.body_joints = ["uncontrolled_body"]
    spec.body_limits = [[-1, 1]]
    spec.body_speed_limits = [0.1]
    obs.body = [0.2]
    compiler = _joint_motion(spec, MotionTiming(), options)
    result = compiler.generate(obs, intent(obs, left_pose=[0.02, 0, 0, 0, 0, 0, 1]))
    assert result.body == [[0.2]] * result.steps
    request = intent(obs)
    request.motion["waypoints"] = [
        {"left_pose": [0.02, 0, 0, 0, 0, 0, 1]},
        {"left_pose": [0.02, 0.1, 0, 0, 0, 0, 1]},
    ]
    with pytest.raises(ContractError, match="did not converge"):
        compiler.generate(obs, request)


def test_saturated_joint_does_not_starve_remaining_freedom():
    from types import SimpleNamespace

    class Geometry:
        spec = SimpleNamespace(left_joints=["dominant", "fine"], right_joints=[])
        chains = {"left": [], "right": []}
        frame = "base_link"

        def poses(self, q, body):
            return {"left": [q[0] + 0.01 * q[1], 0, 0, 0, 0, 0, 1]}

    ik = NumericalIK(Geometry(), [[-0.1, 0.1], [-1, 1]])
    q = ik.solve([0.101, 0, 0, 0, 0, 0, 1], [0.1, 0], [], "left")
    assert q[0] <= 0.1
    assert abs(q[0] + 0.01 * q[1] - 0.101) <= 1e-5
    # The active set must release a bound for an inward target too.
    q = ik.solve([0.09, 0, 0, 0, 0, 0, 1], [0.1, 0], [], "left")
    assert q[0] < 0.1


def test_failed_search_reports_geometry_without_issuing(spec, model, monkeypatch):
    options, obs = model
    captured = []
    monkeypatch.setattr(
        "agent2act.adapters.kinematics.numerical.record_artifact",
        lambda kind, data: captured.append((kind, data)),
    )
    compiler = _joint_motion(spec, MotionTiming(), options)
    with pytest.raises(ContractError, match="not proof of global infeasibility"):
        compiler.generate(obs, intent(obs, left_pose=[0, 0.1, 0, 0, 0, 0, 1]))
    kind, data = captured[-1]
    assert kind == "ik-search-failure"
    assert data["frame"] == "base_link"
    assert data["remaining_translation_m"][1] > 0
    assert data["joint_margins"][0]["lower_margin"] >= 0


@pytest.mark.parametrize("side,sign", [(0, -1), (0, 1), (1, -1), (1, 1)])
def test_tiny_measured_overrun_gets_explicit_valid_first_command(spec, model, side, sign):
    options, obs = model
    compiler = _joint_motion(spec, MotionTiming(), options)
    obs.joints[side] = sign * (0.5 + 1e-6)
    obs.end_pose["base_link"] = {
        arm + "_arm": dict(position=p[:3], orientation=p[3:])
        for arm, p in compiler.fk.poses(obs.joints, obs.body).items()
    }
    obs.issued_gripper_targets = [0.8, 0.6]
    before = copy.deepcopy(obs)
    result = compiler.generate(obs, intent(obs, grippers=[0.9, 0.6]))
    ActionValidator(compiler.spec).validate(result, obs)
    recovery = result.metadata["joint_bound_recovery"]
    assert recovery["observed_joints"] == obs.joints
    assert recovery["first_command"][side] == sign * 0.5
    assert result.grippers[0] == [0.8, 0.6]
    assert result.grippers[-1] == [0.9, 0.6]
    assert result.steps == result.metadata["motion_steps"] + result.metadata["target_hold_steps"]
    assert result.metadata["local_ik"]["source_steps"] == 1
    assert all(-0.5 <= q <= 0.5 for row in result.left_arm + result.right_arm for q in row)
    assert compiler.spec.joint_limits == [[-0.5, 0.5], [-0.5, 0.5]]
    assert obs == before
    compiler.close()


@pytest.mark.parametrize("limiter", ["pose", "joint_speed", "eef_speed"])
def test_measured_overrun_recovery_cannot_expand_precision_or_speed_limits(spec, model, limiter):
    options, obs = model
    overrun = 2e-5 if limiter == "pose" else 1e-6
    if limiter == "joint_speed":
        spec.max_joint_speed = 1e-7
    if limiter == "eef_speed":
        spec.max_eef_speed = 1e-7
    compiler = _joint_motion(spec, MotionTiming(), options)
    obs.joints[0] = 0.5 + overrun
    obs.end_pose["base_link"]["left_arm"]["position"][0] = obs.joints[0]
    before = copy.deepcopy(obs)
    with pytest.raises(ContractError, match="start is outside joint bounds"):
        compiler.generate(obs, intent(obs))
    assert obs == before
    compiler.close()


def test_bound_recovery_does_not_mask_fk_mismatch_or_enlarge_chunk_budget(spec, model):
    options, obs = model
    spec.max_chunk_steps = 1
    compiler = _joint_motion(spec, MotionTiming(), options)
    obs.joints[0] = 0.500001
    with pytest.raises(ContractError, match="FK observation mismatch"):
        compiler.generate(obs, intent(obs))
    obs.end_pose["base_link"]["left_arm"]["position"][0] = obs.joints[0]
    with pytest.raises(ContractError, match="max_chunk_steps after joint timing"):
        compiler.generate(obs, intent(obs))
    compiler.close()


def test_bound_recovery_uses_redundancy_without_relaxing_pose_accuracy(tmp_path, spec, observation):
    spec = replace(spec, left_joints=["left_main", "left_aux"], joint_limits=None)
    path = tmp_path / "redundant.urdf"
    path.write_text(f'''<robot>
      <joint name="left_main" type="revolute"><parent link="base_link"/><child link="elbow"/>
        <axis xyz="0 0 1"/><limit lower="-0.5" upper="0.5" velocity="1"/></joint>
      <joint name="left_aux" type="revolute"><parent link="elbow"/><child link="left"/>
        <axis xyz="0 0 1"/><limit lower="-0.5" upper="0.5" velocity="1"/></joint>
      <joint name="{spec.right_joints[0]}" type="prismatic"><parent link="base_link"/><child link="right"/>
        <axis xyz="0 1 0"/><limit lower="-0.5" upper="0.5" velocity="1"/></joint>
    </robot>''')
    options = dict(path=str(path), frame="base_link", left_link="left", right_link="right")
    compiler = _joint_motion(spec, MotionTiming(), options)
    obs = replace(observation, joints=[0.5002, 0.0, 0.0])
    original = compiler.fk.poses(obs.joints, obs.body)
    obs.end_pose = {
        "base_link": {side + "_arm": dict(position=p[:3], orientation=p[3:]) for side, p in original.items()}
    }
    before = copy.deepcopy(obs)
    result = compiler.generate(obs, intent(obs))
    recovery = result.metadata["joint_bound_recovery"]
    assert recovery["projected_seed"] == [0.5, 0.0, 0.0]
    assert recovery["first_command"][0] <= 0.5
    assert recovery["first_command"][1] > 0.0
    assert recovery["eef_changes"]["left"]["rotation_rad"] <= compiler.ik.rotation_tolerance_rad
    assert compiler.ik.rotation_tolerance_rad == 1e-4
    assert compiler.ik.position_tolerance_m == 1e-5
    ActionValidator(compiler.spec).validate(result, obs)
    assert result.grippers[0] == obs.grippers
    assert obs == before
    compiler.close()


@pytest.mark.parametrize("profile", ["quintic", "smooth_cruise"])
def test_reachable_joint_arc_does_not_require_cartesian_chord_ik(tmp_path, spec, observation, profile):
    """A one-link arm can follow its circle, but cannot follow the chord inside it."""
    path = tmp_path / "arc.urdf"
    path.write_text(f'''<robot>
      <joint name="{spec.left_joints[0]}" type="revolute">
        <parent link="base_link"/><child link="elbow"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" velocity="1"/>
      </joint>
      <joint name="tip" type="fixed"><parent link="elbow"/><child link="left"/>
        <origin xyz="1 0 0"/>
      </joint>
      <joint name="{spec.right_joints[0]}" type="prismatic">
        <parent link="base_link"/><child link="right"/><axis xyz="0 1 0"/>
        <limit lower="-1" upper="1" velocity="1"/>
      </joint>
    </robot>''')
    options = dict(path=str(path), frame="base_link", left_link="left", right_link="right")
    compiler = _joint_motion(spec, MotionTiming(profile=profile), options)
    try:
        poses = compiler.fk.poses(observation.joints, observation.body)
        obs = replace(
            observation,
            end_pose={
                "base_link": {
                    side + "_arm": dict(position=p[:3], orientation=p[3:]) for side, p in poses.items()
                }
            },
        )
        target = compiler.fk.poses([0.4, 0], obs.body)["left"]
        # Establish that the Cartesian chord itself is geometrically impossible.
        chord = [(a + b) / 2 for a, b in zip(poses["left"][:3], target[:3])]
        assert math.hypot(*chord) < 0.99
        result = compiler.generate(obs, intent(obs, left_pose=target))
        assert result.left_arm[-1][0] == pytest.approx(0.4, abs=1e-4)
        assert all(-1 <= row[0] <= 1 for row in result.left_arm)
        assert all(math.hypot(*p[:3]) == pytest.approx(1) for p in result.metadata["cartesian_path"]["left"])
        assert result.metadata["cartesian_path"]["left"][-1] == pytest.approx(target, abs=1e-4)
        ActionValidator(compiler.spec).validate(result, obs)
    finally:
        compiler.close()


@pytest.mark.parametrize("residual,accepted", [(0.005, True), (0.011, False)])
def test_endpoint_ik_acceptance_is_not_relaxed(spec, model, residual, accepted):
    options, obs = model
    compiler = _joint_motion(
        spec,
        MotionTiming(),
        {
            **options,
            "accept_translation_m": 0.01,
            "accept_rotation_rad": 0.02,
        },
    )
    try:
        request = intent(obs, left_pose=[0.05, residual, 0, 0, 0, 0, 1])
        if not accepted:
            with pytest.raises(LocalIKConvergenceError):
                compiler.generate(obs, request)
        else:
            chunk = compiler.generate(obs, request)
            endpoint = chunk.metadata["local_ik"]["generated_endpoint_fk"]["left"]
            assert math.dist(endpoint[:3], request.motion["waypoints"][0]["left_pose"][:3]) <= 0.01
    finally:
        compiler.close()


def test_joint_interpolation_visits_every_waypoint_in_order(spec, model):
    options, obs = model
    compiler = _joint_motion(spec, MotionTiming(), options)
    try:
        request = intent(obs)
        request.motion["waypoints"] = [
            {"left_pose": [0.05, 0, 0, 0, 0, 0, 1], "grippers": [0.6, 0.5], "duration_s": 2},
            {"left_pose": [-0.05, 0, 0, 0, 0, 0, 1], "grippers": [0.7, 0.5]},
        ]
        chunk = compiler.generate(obs, request)
        positions = [row[0] for row in chunk.left_arm]
        turn = positions.index(max(positions))
        assert max(positions) == pytest.approx(0.05, abs=1e-5)
        assert positions[-1] == pytest.approx(-0.05, abs=1e-5)
        assert turn + 1 >= 2 / spec.dt_s
        assert chunk.grippers[turn] == [0.6, 0.5]
        assert chunk.grippers[-1] == [0.7, 0.5]
        assert all(a <= b for a, b in zip(positions[:turn], positions[1 : turn + 1]))
        assert all(a >= b for a, b in zip(positions[turn:], positions[turn + 1 :]))
        ActionValidator(compiler.spec).validate(chunk, obs)
    finally:
        compiler.close()
