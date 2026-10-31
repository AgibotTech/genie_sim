# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import json

import pytest

from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.domain.models import ContractError


def test_compact_motion_context_reports_disabled_speed_limits_and_default_steps():
    from agent2act.adapters.planners.motion_context import compact_motion_state

    state = {
        "context": {
            "execution": {
                "mode": "motion",
                "executors": {
                    "motion": {
                        "compiler": {
                            "speed_limits_enforced": False,
                            "operating_timing": {"default_steps": 10},
                        }
                    }
                },
            }
        },
        "states": {},
    }
    execution = compact_motion_state(state)["context"]["execution"]
    assert execution["speed_limits_enforced"] is False
    assert execution["default_steps"] == 10
    assert "Speed limits are disabled" in execution["timing"]


@pytest.mark.parametrize("planner_type", [ResponsesPlanner, EmbodiedPlanner])
def test_compact_input_preserves_commands_rejections_and_review(planner_type, observation, motion_decision):
    calls = []

    def post(body):
        calls.append(body)
        return {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(motion_decision)}]}
            ]
        }

    context = {
        "robot": {"max_eef_speed": 0.25, "left_joints": ["joint"], "gripper_bounds": [[-0.785, 0]] * 2},
        "execution": {"mode": "motion", "policy_execution": "irrelevant"},
        "gripper_commands": [-0.4, 0],
        "observed_grippers_in_command_units": [-0.5, 0],
        "recent_events": [
            {"status": "issued", "reason": "old hypothesis"},
            {"status": "issued", "issued_endpoint": {"grippers": [-0.4, 0]}, "reason": "close"},
            {"status": "rejected", "error": "motion exceeds max_chunk_steps"},
        ],
        "proposal": None,
    }
    original = copy.deepcopy(context)
    original_obs = copy.deepcopy(observation)
    planner = planner_type(
        model="test",
        post=post,
        compact_motion_context=True,
        gripper_open_closed={"left": [-0.785, 0], "right": [-0.785, 0]},
    )
    try:
        planner.decide(observation, context)
        state = json.loads(calls[0]["input"][0]["content"][0]["text"])
        assert "max_eef_speed" not in state["context"]["robot"]
        assert "policy_execution" not in state["context"]["execution"]
        assert "arm_joint_states" not in state["states"]
        assert state["context"]["recent_events"] == original["recent_events"][1:]
        assert state["context"]["gripper_commands"] == [-0.4, 0]
        assert state["context"]["observed_grippers_in_command_units"] == [-0.5, 0]
        assert state["gripper_command_reference"]["left"] == {
            "fully_open_command": -0.785,
            "fully_closed_command": 0,
        }
        assert context == original
        assert observation == original_obs
        context["proposal"] = {"id": "review"}
        planner.decide(observation, context)
        review = json.loads(calls[1]["input"][0]["content"][0]["text"])
        assert review["context"] == context
        assert "arm_joint_states" in review["states"]
    finally:
        planner.close()


@pytest.mark.parametrize(
    "options",
    [
        {"compact_motion_context": 1},
        {"concise_tracking_feedback": 1},
        {"gripper_open_closed": {"left": [-0.785, 0]}},
        {"gripper_open_closed": {"left": [0, 0], "right": [-0.785, 0]}},
    ],
)
def test_invalid_presentation_options(options):
    with pytest.raises(ContractError):
        ResponsesPlanner(model="test", **options)


@pytest.mark.parametrize("fixed_steps", [False, True])
def test_current_only_removes_all_history_but_keeps_retry_feedback(observation, motion_decision, fixed_steps):
    import io

    from PIL import Image

    observation.images = {}
    for camera, color in zip(("head", "hand_left", "hand_right"), ("red", "green", "blue"), strict=True):
        data = io.BytesIO()
        Image.new("RGB", (16, 16), color).save(data, "JPEG")
        observation.images[camera] = {"encoding": "JPEG", "image_data": data.getvalue()}
    observation.history = [{"images": observation.images}]
    calls = []

    def post(body):
        calls.append(body)
        return {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(motion_decision)}]}
            ]
        }

    context = {
        "sequence": 2,
        "planning_attempt": 1,
        "execution": {"mode": "motion"},
        "gripper_commands": [-0.4, 0],
        "recent_events": [
            {"sequence": 1, "status": "issued", "reason": "OLD_ACTION"},
            {"sequence": 1, "status": "rejected", "error": "OLD_ERROR"},
            {"sequence": 2, "status": "rejected", "error": "CURRENT_ERROR"},
        ],
        "tracking_feedback": {"previous_sequence": 1},
    }
    if fixed_steps:
        context["execution"]["executors"] = {
            "motion": {
                "compiler": {
                    "fixed_steps": 10,
                    "retime_to_limits": True,
                    "timing": "linear position/gripper interpolation and quaternion SLERP; no extra hold",
                    "speed_limits_enforced": True,
                }
            }
        }
    original = copy.deepcopy(context)
    planner = ResponsesPlanner(model="test", post=post, current_observation_only=True, history_frames=1)
    try:
        for _ in range(2):
            planner.decide(observation, context)
            parts = calls[-1]["input"][0]["content"]
            assert sum(p["type"] == "input_image" for p in parts) == 3
            for index, camera in enumerate(("head", "hand_left", "hand_right")):
                assert parts[1 + index * 2] == {"type": "input_text", "text": f"current camera={camera}"}
                assert parts[2 + index * 2]["type"] == "input_image"
            assert len({p["image_url"] for p in parts if p["type"] == "input_image"}) == 3
            text = json.dumps(parts)
            assert "previous_robot_observation" not in text
            assert "OLD_ACTION" not in text and "OLD_ERROR" not in text
            state = json.loads(parts[0]["text"])
            assert "recent_events" not in state["context"]
            assert "tracking_feedback" not in state["context"]
            assert state["context"]["gripper_commands"] == [-0.4, 0]
            assert state["context"]["current_rejections"][0]["error"] == "CURRENT_ERROR"
        assert context == original
        assert planner._last_visual == []
        context["proposal"] = {"id": "review"}
        planner.decide(observation, context)
        state = json.loads(calls[-1]["input"][0]["content"][0]["text"])
        assert state["context"] == context
    finally:
        planner.close()


def test_current_only_can_include_latest_tracking_summary(observation, motion_decision):
    calls = []

    def post(body):
        calls.append(body)
        return {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(motion_decision)}]}
            ]
        }

    context = {
        "execution": {"mode": "motion"},
        "recent_events": [{"status": "issued", "reason": "OLD_ACTION"}],
        "tracking_feedback": {
            "previous_sequence": 29,
            "frame": "base_link",
            "left": {
                "commanded_delta_m": [0, 0.035, 0],
                "observed_delta_m": [0.0053, 0.0246, -0.0051],
                "remaining_delta_m": [-0.0053, 0.0104, 0.0051],
                "rotation_error_rad": 0.00184,
            },
        },
    }
    original = copy.deepcopy(context)
    planner = ResponsesPlanner(
        model="test", post=post, current_observation_only=True, concise_tracking_feedback=True
    )
    try:
        planner.decide(observation, context)
        state = json.loads(calls[-1]["input"][0]["content"][0]["text"])
        summary = state["context"]["tracking_feedback"]
        assert isinstance(summary, str)
        assert "commanded [0.0, 35.0, 0.0]" in summary
        assert "observed [5.3, 24.6, -5.1]" in summary
        assert "remaining [-5.3, 10.4, 5.1]" in summary
        assert "not object alignment or grasp success" not in summary
        assert "Robot tracking only" not in summary
        assert "recent_events" not in state["context"]
        assert context == original
        context["tracking_feedback"] = {}
        planner.decide(observation, context)
        state = json.loads(calls[-1]["input"][0]["content"][0]["text"])
        assert "tracking_feedback" not in state["context"]
    finally:
        planner.close()
