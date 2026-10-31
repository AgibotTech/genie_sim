# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Waist yaw motion, its issued prefix, and model-facing waist context; no simulator or model calls."""

import math
from dataclasses import replace

import pytest

from agent2act.adapters.kinematics.urdf import UrdfKinematics
from agent2act.adapters.planners.motion_context import compact_motion_state, motion_only_state
from agent2act.adapters.planners.prompts import FINAL_HOLD_INSTRUCTIONS, WAIST_INSTRUCTIONS
from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.application.routing import ExecutionRouter
from agent2act.application.runtime import AgentRuntime
from agent2act.application.settings import ExecutionConfig
from agent2act.bootstrap import _joint_motion
from agent2act.domain.models import ActionChunk, ContractError, Decision
from agent2act.domain.timing import MotionTiming
from agent2act.domain.validation import ActionValidator
from agent2act.execution.motion import MotionCompiler


@pytest.fixture
def waist_model(tmp_path, spec, observation):
    spec = replace(
        spec,
        body_joints=["pitch", "waist_yaw"],
        body_limits=[[-1, 1], [-3, 3]],
        body_speed_limits=[1, 1],
        max_eef_speed=2.0,
    )
    path = tmp_path / "robot.urdf"
    path.write_text(f'''<robot>
    <joint name="pitch" type="revolute"><parent link="base_link"/><child link="hip"/>
    <axis xyz="0 1 0"/><limit lower="-1" upper="1" velocity="1"/></joint>
    <joint name="waist_yaw" type="revolute"><parent link="hip"/><child link="torso"/>
    <axis xyz="0 0 1"/><limit lower="-3" upper="3" velocity="1"/></joint>
    <joint name="{spec.left_joints[0]}" type="prismatic">
    <origin xyz="0.3 0.1 0"/><parent link="torso"/><child link="left"/><axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" velocity="0.5"/></joint>
    <joint name="{spec.right_joints[0]}" type="prismatic">
    <origin xyz="0.3 -0.1 0"/><parent link="torso"/><child link="right"/><axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" velocity="0.5"/></joint></robot>''')
    options = dict(path=str(path), frame="base_link", left_link="left", right_link="right")
    body = [0.0, 0.0]
    fk = UrdfKinematics(spec, options)
    obs = replace(
        observation,
        body=body,
        end_pose={
            "base_link": {
                side + "_arm": dict(position=p[:3], orientation=p[3:])
                for side, p in fk.poses(observation.joints, body).items()
            }
        },
    )
    fk.close()
    return spec, {**options, "waist_yaw_joint": "waist_yaw"}, obs


def decision(obs, *waypoints):
    return Decision(
        "motion", obs.instruction, "waist test", motion=dict(frame="base_link", waypoints=list(waypoints))
    )


def test_waist_turn_carries_omitted_arms_and_respects_speed(waist_model):
    spec, options, obs = waist_model
    compiler = _joint_motion(spec, MotionTiming(), options)
    control = compiler.describe()["waist_control"]
    assert control["joint"] == "waist_yaw" and control["state_index"] == 1
    assert control["limits_rad"] == [-3, 3]
    result = compiler.generate(obs, decision(obs, {"waist_yaw": -math.pi / 2}))
    assert result.kind == "JOINT_ABS"
    assert result.body[-1] == pytest.approx([0.0, -math.pi / 2])
    assert all(row[0] == 0.0 for row in result.body)
    # Omitted arms keep their joints and turn rigidly with the torso.
    assert result.left_arm == [obs.joints[:1]] * result.steps
    assert result.right_arm == [obs.joints[1:]] * result.steps
    yaws = [0.0] + [row[1] for row in result.body]
    assert max(abs(b - a) for a, b in zip(yaws, yaws[1:])) <= spec.dt_s + 1e-9
    left_end = result.metadata["cartesian_path"]["left"][-1]
    assert left_end[:2] == pytest.approx([0.1, -0.3], abs=1e-6)
    assert result.metadata["waist_motion"]["target_rad"] == pytest.approx(-math.pi / 2)
    ActionValidator(compiler.spec).validate(result, obs)
    compiler.close()


def test_disabled_speed_limits_do_not_retime_waist_turn(waist_model):
    original, options, obs = waist_model
    spec = replace(original, enforce_speed_limits=False, body_speed_limits=[1e-6, 1e-6])
    compiler = _joint_motion(spec, MotionTiming(), options)
    try:
        result = compiler.generate(obs, decision(obs, {"waist_yaw": -math.pi / 2}))
        assert result.steps == 10
        assert result.body[-1] == pytest.approx([0, -math.pi / 2])
        assert result.left_arm == [obs.joints[:1]] * 10
        ActionValidator(compiler.spec).validate(result, obs)
    finally:
        compiler.close()


def test_operating_waist_speed_prevents_one_step_turn_without_changing_robot_limits(waist_model):
    original, options, obs = waist_model
    spec = replace(original, body_speed_limits=[1, 1000])
    compiler = _joint_motion(spec, MotionTiming(), {**options, "waist_speed_rad_s": 1.0})
    try:
        result = compiler.generate(obs, decision(obs, {"waist_yaw": -1.2}))
        values = [obs.body[1]] + [row[1] for row in result.body]
        assert result.steps > 1
        assert max(abs(b - a) / spec.dt_s for a, b in zip(values, values[1:])) <= 1.0 + 1e-9
        assert result.body[-1][1] == pytest.approx(-1.2)
        assert compiler.spec.body_speed_limits == spec.body_speed_limits == [1, 1000]
        assert compiler.describe()["waist_control"]["speed_rad_s"] == 1.0
        ActionValidator(compiler.spec).validate(result, obs)
    finally:
        compiler.close()


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), 1001])
def test_operating_waist_speed_rejects_invalid_values(waist_model, value):
    spec, options, _ = waist_model
    with pytest.raises(ContractError, match="waist_speed_rad_s"):
        _joint_motion(spec, MotionTiming(), {**options, "waist_speed_rad_s": value})


def test_operating_waist_speed_failure_cleans_up_owned_compiler(waist_model, monkeypatch):
    spec, options, _ = waist_model
    closed = []
    original = MotionCompiler.close

    def close(compiler):
        closed.append(compiler)
        original(compiler)

    monkeypatch.setattr(MotionCompiler, "close", close)
    with pytest.raises(ContractError, match="waist_speed_rad_s"):
        _joint_motion(spec, MotionTiming(), {**options, "waist_speed_rad_s": 0})
    assert len(closed) == 1


def test_operating_waist_speed_scales_with_timing_and_requires_configured_waist(waist_model):
    spec, options, obs = waist_model
    compiler = _joint_motion(spec, MotionTiming(speed_fraction=0.5), {**options, "waist_speed_rad_s": 0.5})
    try:
        chunk = compiler.generate(obs, decision(obs, {"waist_yaw": 0.5}))
        values = [0.0] + [row[1] for row in chunk.body]
        assert max(abs(b - a) / spec.dt_s for a, b in zip(values, values[1:])) <= 0.25 + 1e-9
        assert compiler.describe()["waist_control"]["speed_rad_s"] == 0.25
    finally:
        compiler.close()
    without_waist = dict(options)
    without_waist.pop("waist_yaw_joint")
    with pytest.raises(ContractError, match="waist_speed_rad_s requires"):
        _joint_motion(spec, MotionTiming(), {**without_waist, "waist_speed_rad_s": 0.5})


def test_waist_waypoint_solves_arm_target_at_new_waist_angle(waist_model):
    spec, options, obs = waist_model
    compiler = _joint_motion(spec, MotionTiming(), options)
    # At yaw=+pi/2 the left arm slides along base_link -y; x=0.1 needs joint 0.1 beyond the 0.3 mount.
    target = [-0.1, 0.4, 0, 0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    result = compiler.generate(obs, decision(obs, {"waist_yaw": math.pi / 2}, {"left_pose": target}))
    assert result.body[-1][1] == pytest.approx(math.pi / 2)
    assert result.left_arm[-1][0] == pytest.approx(0.1, abs=1e-4)
    assert result.metadata["local_ik"]["generated_endpoint_fk"]["left"][:3] == pytest.approx(
        target[:3], abs=1e-4
    )
    compiler.close()


def test_waist_limits_and_unknown_fields_rejected(waist_model, spec, observation):
    spec, options, obs = waist_model
    compiler = _joint_motion(spec, MotionTiming(), options)
    with pytest.raises(ContractError, match="outside the joint limits"):
        compiler.generate(obs, decision(obs, {"waist_yaw": 3.5}))
    with pytest.raises(ContractError, match="unknown waypoint fields"):
        compiler.generate(obs, decision(obs, {"waist_yaw": 0.1, "head_pitch": 0.2}))
    compiler.close()
    unconfigured = _joint_motion(
        spec, MotionTiming(), {k: v for k, v in options.items() if k != "waist_yaw_joint"}
    )
    assert "waist_control" not in unconfigured.describe()
    with pytest.raises(ContractError, match="unknown waypoint fields"):
        unconfigured.generate(obs, decision(obs, {"waist_yaw": 0.1}))
    unconfigured.close()
    with pytest.raises(ContractError, match="waist_yaw_joint"):
        _joint_motion(spec, MotionTiming(), {**options, "waist_yaw_joint": "missing"})


class _Planner:
    def __init__(self, obs):
        self.obs = obs

    def decide(self, observation, context):
        return decision(observation, {"waist_yaw": 1.0})

    def reset(self):
        pass

    def close(self):
        pass


class _Executor:
    mode = "motion"

    def __init__(self, body_rows):
        self.body_rows = body_rows

    def describe(self):
        return {"mode": "motion"}

    def generate(self, observation, request):
        n = len(self.body_rows)
        return ActionChunk(
            "JOINT_ABS",
            "base_link",
            0.05,
            [observation.joints[:1]] * n,
            [observation.joints[1:]] * n,
            [list(observation.issued_gripper_targets or observation.grippers)] * n,
            {},
            [list(row) for row in self.body_rows],
        )

    def reset(self):
        pass

    def close(self):
        pass


@pytest.mark.parametrize("moving, expected", [(True, 30), (False, 8)])
def test_body_motion_issues_longer_prefix(waist_model, moving, expected):
    spec, _, obs = waist_model
    rows = [[0.0, 0.02 * (i + 1) if moving else 0.0] for i in range(40)]
    config = ExecutionConfig(mode="motion", allowed_modes=["motion"], execute_steps=8, body_execute_steps=30)
    runtime = AgentRuntime(
        spec, _Planner(obs), ExecutionRouter([_Executor(rows)], config), lambda event: None
    )
    chunk = runtime.step(obs)
    assert chunk.steps == expected
    assert runtime.router.describe()["body_execute_steps"] == 30
    with pytest.raises(ContractError, match="body_execute_steps"):
        ExecutionConfig(body_execute_steps=0)


def _state(waist):
    motion = {"compiler": {}, **({"waist_control": {"state_index": 1}} if waist else {})}
    return {
        "instruction": "turn",
        "states": {
            "end_pose": {},
            "gripper_states": [0, 0],
            "waist_joint_states": [0, 0.5],
            "head_joint_states": [0],
        },
        "context": {
            "sequence": 1,
            "execution": {"mode": "motion", "body_execute_steps": 120, "executors": {"motion": motion}},
        },
    }


@pytest.mark.parametrize("waist", [True, False])
def test_waist_context_only_when_compiler_accepts_it(waist):
    for state in (motion_only_state(_state(waist)), motion_only_state(compact_motion_state(_state(waist)))):
        assert ("waist_joint_states" in state["states"]) is waist
        assert "head_joint_states" not in state["states"]
        execution = state["context"]["execution"]
        assert ("waist_control" in execution) is waist
        assert execution.get("body_execute_steps") == (120 if waist else None)


def test_waist_instructions_follow_request_context():
    planner = ResponsesPlanner(model="test-model", api_key_env=None)
    assert planner._context_instructions() == ""
    planner._waist_control_active = True
    text = planner._context_instructions()
    assert WAIST_INSTRUCTIONS in text
    assert WAIST_INSTRUCTIONS not in planner._context_instructions(waist=False)
    assert FINAL_HOLD_INSTRUCTIONS not in text
    planner._final_hold_active = True
    assert FINAL_HOLD_INSTRUCTIONS in planner._context_instructions()
    planner.reset()
    assert planner._context_instructions() == ""


def test_ik_acceptance_options_are_validated_and_applied(waist_model):
    spec, options, obs = waist_model
    compiler = _joint_motion(
        spec, MotionTiming(), {**options, "accept_translation_m": 0.005, "accept_rotation_rad": 0.02}
    )
    assert compiler.describe()["compiler"]["ik_acceptance"] == {
        "accept_translation_m": 0.005,
        "accept_rotation_rad": 0.02,
    }
    assert compiler.ik.accept_translation_m == 0.005
    compiler.close()
    with pytest.raises(ContractError, match="set together"):
        _joint_motion(spec, MotionTiming(), {**options, "accept_translation_m": 0.005})
    with pytest.raises(ContractError, match="no larger than"):
        _joint_motion(
            spec, MotionTiming(), {**options, "accept_translation_m": 0.5, "accept_rotation_rad": 0.02}
        )


def test_waist_settle_hold_only_after_body_motion(waist_model):
    spec, options, obs = waist_model
    compiler = _joint_motion(spec, MotionTiming(), {**options, "waist_settle_s": 0.5})
    assert compiler.describe()["waist_control"]["settle_hold_s"] == 0.5
    turned = compiler.generate(obs, decision(obs, {"waist_yaw": 0.3}))
    assert turned.metadata["target_hold_steps"] == 10
    assert turned.body[-1][1] == pytest.approx(0.3) and turned.body[-11][1] == pytest.approx(0.3)
    held = compiler.generate(obs, decision(obs, {"waist_yaw": 0.0, "grippers": [0.5, 0.5]}))
    assert held.metadata["target_hold_steps"] == 0
    compiler.close()
    with pytest.raises(ContractError, match="waist_settle_s"):
        _joint_motion(spec, MotionTiming(), {**options, "waist_settle_s": 11})


def test_issued_body_targets_continue_across_chunks(waist_model):
    spec, options, obs = waist_model
    compiler = _joint_motion(spec, MotionTiming(), options)
    lagging = replace(obs, issued_body_targets=[0.0, 0.4])
    held = compiler.generate(lagging, decision(lagging, {"grippers": [0.5, 0.5]}))
    # Non-waist motion keeps commanding the issued waist, not the lagging measurement.
    assert held.body == [[0.0, 0.4]] * held.steps
    turned = compiler.generate(lagging, decision(lagging, {"waist_yaw": 0.5}))
    assert turned.metadata["waist_motion"]["start_rad"] == 0.4
    assert turned.metadata["waist_motion"]["measured_start_rad"] == 0.0
    ActionValidator(compiler.spec).validate(turned, lagging)
    compiler.close()
    from agent2act.domain.actions import hold_chunk

    assert hold_chunk(lagging, spec).body == [[0.0, 0.4]]
    with pytest.raises(ContractError, match="body joint speed"):
        ActionValidator(spec).validate(held, replace(lagging, issued_body_targets=None))


def test_ik_failure_hint_mentions_the_waist_only_with_waist_control(waist_model):
    spec, options, obs = waist_model
    # The prismatic left joint stops at 0.5, so a target 1.2 m ahead saturates it.
    target = [1.2, 0.1, 0, 0, 0, 0, 1]
    without = {k: v for k, v in options.items() if k != "waist_yaw_joint"}
    for compiler_options, mentions_waist in ((options, True), (without, False)):
        compiler = _joint_motion(spec, MotionTiming(), compiler_options)
        with pytest.raises(ContractError, match="local IK did not converge") as info:
            compiler.generate(obs, decision(obs, {"left_pose": target}))
        assert ("waist" in str(info.value)) is mentions_waist
        compiler.close()


def test_unlimited_waist_turn_is_not_slowed_by_carried_arm_speed(waist_model):
    spec, options, obs = waist_model
    fast = replace(spec, body_speed_limits=[1, 1000])
    compiler = _joint_motion(fast, MotionTiming(), options)
    result = compiler.generate(obs, decision(obs, {"waist_yaw": 1.5}))
    # The carried arms sweep 0.45 m in one step, far above max_eef_speed, but only the waist moved.
    assert result.metadata["motion_steps"] == 1
    assert result.body[-1][1] == pytest.approx(1.5)
    compiler.close()
