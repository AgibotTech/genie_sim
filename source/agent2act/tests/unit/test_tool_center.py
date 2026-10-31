# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import json
import math

import pytest

from agent2act.adapters.planners.chat_completions import ChatCompletionsPlanner
from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.adapters.planners.tool_center import ToolCenter
from agent2act.domain.models import ContractError

OFFSETS = {"left": [0, 0, 0.14845], "right": [0, 0, 0.14845]}
CONTEXT = {"execution": {"mode": "motion", "motion_frame": "base_link"}}


@pytest.mark.parametrize("arm", ["left", "right"])
def test_rotation_and_inverse(arm):
    center = ToolCenter(OFFSETS)
    wrist = [0.2, -0.3, 0.4, 0, math.sqrt(0.5), 0, math.sqrt(0.5)]
    tcp = center.convert(wrist, arm)
    assert tcp[:3] == pytest.approx([0.34845, -0.3, 0.4])
    assert center.convert(tcp, arm, inverse=True) == pytest.approx(wrist)


@pytest.mark.parametrize("planner_type", [ResponsesPlanner, EmbodiedPlanner, ChatCompletionsPlanner])
@pytest.mark.parametrize("compact", [False, True])
def test_model_center_target_to_wrist_and_history(planner_type, compact, observation):
    calls = []
    q = [0, math.sqrt(0.5), 0, math.sqrt(0.5)]
    target = [0.5, 0.2, 0.6] + q

    def post(body):
        calls.append(copy.deepcopy(body))
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "mode": "motion",
                                    "instruction": "approach",
                                    "reason": "visible target",
                                    "motion": {
                                        "frame": "base_link",
                                        "waypoints": [{"left_pose": target, "grippers": [0, 0]}],
                                    },
                                }
                            ),
                        }
                    ],
                }
            ]
        }

        if planner_type is ChatCompletionsPlanner:
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": response["output"][0]["content"][0]["text"]},
                    }
                ]
            }
        return response

    original = copy.deepcopy(observation)
    context = copy.deepcopy(CONTEXT)
    context["sequence"] = 1
    before_context = copy.deepcopy(context)
    planner = planner_type(
        model="test", post=post, gripper_center_offset_m=OFFSETS, compact_motion_context=compact
    )
    try:
        decision = planner.decide(observation, context)
        parts = (
            calls[0]["messages"][1]["content"]
            if planner_type is ChatCompletionsPlanner
            else calls[0]["input"][0]["content"]
        )
        state = json.loads(parts[0]["text"])
        assert "gripper center" in state["pose_reference"]
        wrist = observation.eef("left", "base_link")
        expected = ToolCenter(OFFSETS).convert(wrist, "left")
        assert state["states"]["end_pose"]["base_link"]["left_arm"]["position"] == pytest.approx(expected[:3])
        assert decision.motion["waypoints"][0]["left_pose"][:3] == pytest.approx([0.35155, 0.2, 0.6])
        assert "compose_grasp_pose" not in str(calls[0])
        assert "No wrist-to-grasp-center calibration" not in str(calls[0])
        assert observation == original
        assert context == before_context
    finally:
        planner.close()


def test_feedback_rotation_moves_center_without_wrist_translation():
    before = [0, 0, 0, 0, 0, 0, 1]
    after = [0, 0, 0, 0, math.sqrt(0.5), 0, math.sqrt(0.5)]
    measured = {
        "left": {
            "previous_pose": before,
            "observed_pose": after,
            "translation_delta_m": [0, 0, 0],
            "rotation_delta_rad": math.pi / 2,
        }
    }
    source = {
        "tracking_feedback": {
            "kind": "EEF_ABS",
            "generated_endpoint": {"left_arm": after},
            "left": {
                "previous_pose": before,
                "observed_pose": after,
                "issued_pose": after,
                "rotation_error_rad": 0,
            },
            "measured_eef_motion": measured,
            "summary": "obsolete wrist summary",
        },
        "recent_events": [
            {
                "kind": "EEF_ABS",
                "issued_endpoint": {"left_arm": after},
                "observed_effect": {"measured_eef_motion": measured},
            }
        ],
    }
    original = copy.deepcopy(source)
    result = ToolCenter(OFFSETS).context(source)
    expected = [0.14845, 0, -0.14845]
    assert result["tracking_feedback"]["generated_endpoint"]["left_arm"][:3] == pytest.approx([0.14845, 0, 0])
    assert result["tracking_feedback"]["left"]["observed_delta_m"] == pytest.approx(expected)
    assert result["tracking_feedback"]["left"]["translation_error_m"] == pytest.approx(0)
    assert result["tracking_feedback"]["left"]["along_track_fraction"] == pytest.approx(1)
    assert "summary" not in result["tracking_feedback"]
    assert result["recent_events"][0]["observed_effect"]["measured_eef_motion"]["left"][
        "translation_delta_m"
    ] == pytest.approx(expected)
    assert source == original


@pytest.mark.parametrize(
    "offsets", [{}, {"left": [0, 0, 1]}, {"left": [0, 0, float("nan")], "right": [0, 0, 1]}]
)
def test_invalid_offsets(offsets):
    with pytest.raises(ContractError):
        ResponsesPlanner(model="test", gripper_center_offset_m=offsets)


def test_conflicting_tool_and_unsupported_review(observation):
    with pytest.raises(ContractError, match="not both"):
        EmbodiedPlanner(model="test", wrist_to_center_m=OFFSETS, gripper_center_offset_m=OFFSETS)
    planner = ResponsesPlanner(
        model="test", gripper_center_offset_m=OFFSETS, post=lambda _: pytest.fail("no request")
    )
    try:
        with pytest.raises(ContractError, match="motion-only"):
            planner.decide(observation, {"execution": {"mode": "auto"}})
    finally:
        planner.close()


def test_runtime_feedback_and_hold_use_original_execution_frame(config, request_payload):
    from agent2act.bootstrap import ApplicationFactory, default_registry
    from agent2act.configuration import parse_settings

    calls = []

    def post(body):
        state = json.loads(body["input"][0]["content"][0]["text"])
        calls.append(state)
        center = state["states"]["end_pose"]["base_link"]["left_arm"]["position"]
        # Three components exercises holding observed orientation; the other side is omitted.
        decision = {
            "mode": "motion",
            "instruction": "hold",
            "reason": "check observation",
            "motion": {"frame": "base_link", "waypoints": [{"left_pose": center}]},
        }
        return {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(decision)}]}
            ]
        }

    registry = default_registry()
    registry.planners.register(
        "offline_center", lambda _: ResponsesPlanner(model="test", post=post, gripper_center_offset_m=OFFSETS)
    )
    config["planner"] = {"type": "offline_center"}
    config["execution"].update(mode="motion", max_motion_continuations=0)
    with ApplicationFactory(parse_settings(config), registry).create_session() as session:
        session.handle(request_payload)
        issued = session.runtime._last_issued
        for arm in ("left", "right"):
            original = request_payload["params"]["states"]["end_pose"]["base_link"][arm + "_arm"]
            assert getattr(issued, arm + "_arm")[-1] == pytest.approx(
                original["position"] + original["orientation"]
            )
        session.handle(request_payload)
        feedback = calls[-1]["context"]["tracking_feedback"]["left"]
        assert feedback["commanded_delta_m"] == pytest.approx([0, 0, 0])
        assert feedback["observed_delta_m"] == pytest.approx([0, 0, 0])
        assert feedback["along_track_fraction"] is None
        assert feedback["observed_pose"][2] == pytest.approx(issued.left_arm[-1][2] + 0.14845)
        assert session.runtime.memory[0]["issued_endpoint"]["left_arm"] == issued.left_arm[-1]


def test_missing_previous_orientation_does_not_relabel_wrist_delta():
    result = ToolCenter(OFFSETS).context(
        {
            "tracking_feedback": {
                "left": {
                    "observed_pose": [0, 0, 0, 0, 0, 0, 1],
                    "issued_pose": [0, 0, 0, 0, 0, 0, 1],
                    "observed_delta_m": [1, 2, 3],
                }
            }
        }
    )
    assert "observed_delta_m" not in result["tracking_feedback"]["left"]
