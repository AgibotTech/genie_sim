# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import json
from pathlib import Path

import pytest

from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.bootstrap import ApplicationFactory, default_registry
from agent2act.configuration import PluginSettings, load_config
from agent2act.domain.models import ContractError, Decision
from agent2act.domain.timing import MotionTiming
from agent2act.domain.validation import ActionValidator
from agent2act.execution.fixed_steps import FixedStepsMotion


def target_decision(observation):
    left = list(observation.eef("left", "base_link"))
    right = list(observation.eef("right", "base_link"))
    left[0] += 1.0
    left[3:] = [1, 0, 0, 0]
    return Decision.from_dict(
        {
            "mode": "motion",
            "instruction": "Move",
            "reason": "Test target",
            "motion": {
                "frame": "base_link",
                "waypoints": [{"left_pose": left, "right_pose": right, "grippers": [0, 0]}],
            },
        }
    )


def test_fixed_horizon_endpoint_and_default_speed_validation(spec, observation):
    compiler = FixedStepsMotion(spec, MotionTiming())
    decision = target_decision(observation)
    chunk = compiler.generate(observation, decision)
    assert chunk.steps == 30
    assert chunk.left_arm[-1] == pytest.approx(decision.motion["waypoints"][0]["left_pose"])
    assert chunk.left_arm[14][0] == pytest.approx(observation.eef("left", "base_link")[0] + 0.5)
    assert chunk.metadata["target_hold_steps"] == 0
    with pytest.raises(ContractError, match="speed"):
        ActionValidator(spec).validate(chunk, observation)
    spec.enforce_speed_limits = False
    ActionValidator(spec).validate(chunk, observation)
    chunk.grippers[-1][0] = spec.gripper_bounds[0][1] + 1
    with pytest.raises(ContractError, match="gripper bounds"):
        ActionValidator(spec).validate(chunk, observation)
    compiler.reset()
    compiler.close()
    compiler.close()


@pytest.mark.parametrize("change", ["duration", "orientation", "extra_waypoint"])
def test_reject_unsupported_intents(spec, observation, change):
    decision = target_decision(observation)
    waypoint = decision.motion["waypoints"][0]
    if change == "duration":
        waypoint["duration_s"] = 2
    elif change == "orientation":
        waypoint["left_pose"] = waypoint["left_pose"][:3]
    else:
        decision.motion["waypoints"].append(copy.deepcopy(waypoint))
    with pytest.raises(ContractError):
        FixedStepsMotion(spec, MotionTiming()).generate(observation, decision)


def test_fixed_prompt_does_not_silently_repair_orientation(spec, observation):
    calls = []
    payload = {
        "mode": "motion",
        "instruction": "Move",
        "reason": "Test",
        "motion": {
            "frame": "base_link",
            "waypoints": [{"left_pose": [0, 0, 0]}],
        },
    }

    def post(body):
        calls.append(body)
        return {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(payload)}]}]
        }

    compiler = FixedStepsMotion(spec, MotionTiming())
    planner = ResponsesPlanner(model="test", post=post)
    context = {"execution": {"mode": "motion", "executors": {"motion": compiler.describe()}}}
    try:
        decision = planner.decide(observation, context)
        assert decision.motion["waypoints"][0]["left_pose"] == [0, 0, 0]
        protocol = calls[0]["input"][-1]["content"][0]["text"]
        assert "Do not omit orientation" in protocol
        assert "speed-limited" not in protocol
        with pytest.raises(ContractError):
            compiler.generate(observation, decision)
        planner.reset()
        assert planner._fixed_motion_steps is None
    finally:
        planner.close()


def test_failed_motion_construction_closes_planner():
    settings = load_config(Path(__file__).resolve().parents[2] / "examples/g2_llm_recipe.json")
    closed = []

    class Planner:
        def close(self):
            closed.append(True)

    registry = default_registry()
    registry.planners.register("test_cleanup", lambda options: Planner())
    settings = settings.model_copy(
        update={
            "planner": PluginSettings(type="test_cleanup"),
            "motion_backend": PluginSettings(type="fixed_steps", steps=0),
        }
    )
    with pytest.raises(Exception, match="fixed steps"):
        ApplicationFactory(settings, registry).create_session()
    assert closed == [True]


def test_retiming_respects_speeds_and_prefix(spec, observation):
    spec.max_eef_speed = 0.1
    spec.max_chunk_steps = 512
    decision = target_decision(observation)
    compiler = FixedStepsMotion(spec, MotionTiming(), retime_to_limits=True)
    chunk = compiler.generate(observation, decision)
    assert chunk.steps > 30
    ActionValidator(spec).validate(chunk, observation)
    prefix = chunk.prefix(30)
    assert prefix.left_arm[-1][0] - observation.eef("left", "base_link")[0] <= 0.1 * 30 * spec.dt_s + 1e-9
    assert chunk.left_arm[-1] == pytest.approx(decision.motion["waypoints"][0]["left_pose"])
    spec.max_chunk_steps = 30
    with pytest.raises(ContractError, match="max_chunk_steps"):
        compiler.generate(observation, decision)


def test_disabled_speed_limits_override_retiming_request(spec, observation):
    spec.enforce_speed_limits = False
    spec.max_eef_speed = spec.max_eef_angular_speed = spec.max_gripper_speed = 1e-6
    compiler = FixedStepsMotion(spec, MotionTiming(), steps=10, retime_to_limits=True)
    decision = target_decision(observation)
    chunk = compiler.generate(observation, decision)
    assert chunk.steps == 10
    assert chunk.left_arm[-1] == pytest.approx(decision.motion["waypoints"][0]["left_pose"])
    assert compiler.describe()["compiler"]["retime_to_limits"] is False
    assert chunk.metadata["retime_to_limits"] is False
    ActionValidator(spec).validate(chunk, observation)
