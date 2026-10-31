# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Request-scoped prompts preserve execution contracts across planner modes."""

import json
from copy import deepcopy

import pytest

from agent2act.adapters.planners.chat_completions import ChatCompletionsPlanner
from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.prompts import INSTRUCTIONS
from agent2act.adapters.planners.responses import ResponsesPlanner


@pytest.mark.parametrize("planner_type", [ResponsesPlanner, ChatCompletionsPlanner, EmbodiedPlanner])
@pytest.mark.parametrize("compact", [False, True])
def test_motion_wire_context_excludes_routing(planner_type, compact, observation, motion_decision):
    calls = []

    def post(body):
        calls.append(body)
        text = json.dumps(motion_decision)
        if "messages" in body:
            return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}

    context = {
        "sequence": 2,
        "planning_attempt": 1,
        "robot": {"dt_s": 0.02, "gripper_bounds": [[-1, 0]] * 2, "supports_eef": True},
        "execution": {
            "mode": "motion",
            "allowed_modes": ["motion"],
            "execute_steps": 20,
            "policy_execute_steps": 8,
            "max_policy_chunks": 5,
            "proposal_review": False,
            "policy_execution": "UNRELATED",
            "motion_frame": "base_link",
            "executors": {
                "motion": {"mode": "motion", "compiler": {"timing": "local timing"}},
                "policy": {"backend": "UNRELATED"},
            },
        },
        "policy_review": None,
        "proposal": None,
        "proposal_error": None,
        "previous_assessment": None,
        "gripper_commands": [-0.5, 0],
        "recent_events": [
            {
                "sequence": 1,
                "status": "issued",
                "mode": "motion",
                "backend": None,
                "policy_run": None,
                "assessment": None,
                "reviewed_proposal_id": None,
                "issued_endpoint": {"grippers": [-0.5, 0]},
            },
            {"sequence": 2, "status": "rejected", "mode": "motion", "error": "invalid pose"},
        ],
    }
    original = deepcopy(context)
    with_planner = planner_type(model="test", post=post, compact_motion_context=compact)
    try:
        with_planner.decide(observation, context)
        body = calls[0]
        parts = body["messages"][1]["content"] if "messages" in body else body["input"][0]["content"]
        state = json.loads(parts[0]["text"])
        wire = state["context"]
        assert wire["execution"]["motion_frame"] == "base_link"
        assert wire["execution"]["execute_steps"] == 20
        assert wire["robot"]["gripper_bounds"] == [[-1, 0]] * 2
        assert wire["gripper_commands"] == [-0.5, 0]
        assert wire["recent_events"][-1]["error"] == "invalid pose"
        assert wire["recent_events"][0]["issued_endpoint"] == {"grippers": [-0.5, 0]}
        for forbidden in (
            "mode",
            "allowed_modes",
            "policy_execute_steps",
            "max_policy_chunks",
            "proposal_review",
            "policy_execution",
            "policy_review",
            "proposal",
            "proposal_error",
            "previous_assessment",
            "backend",
            "policy_run",
            "assessment",
            "reviewed_proposal_id",
            "supports_eef",
        ):
            assert f'"{forbidden}"' not in json.dumps(state)
        assert "UNRELATED" not in json.dumps(body)
        assert "arm_joint_states" not in state["states"]
        assert context == original
    finally:
        with_planner.close()


@pytest.mark.parametrize("planner_type", [ResponsesPlanner, ChatCompletionsPlanner])
def test_motion_request_and_switch_to_review(planner_type, observation, motion_decision):
    calls = []
    minimal = {**motion_decision, "success_condition": "Observe the target", "status": "continue"}

    def post(body):
        calls.append(body)
        text = json.dumps(minimal)
        if "messages" in body:
            return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}

    planner = planner_type(model="test", post=post)
    try:
        for context in (
            {"execution": {"mode": "motion"}},
            {"execution": {"mode": "auto"}},
            {"execution": {"mode": "auto", "proposal_review": True}},
            {"execution": {"mode": "motion"}},
        ):
            decision = planner.decide(observation, context)
            assert decision.backend is None
            assert decision.assessment is None
            assert decision.policy_chunks == 1
        prompts = [
            body["input"][-1]["content"][0]["text"] if "input" in body else body["messages"][0]["content"]
            for body in calls
        ]
        assert len(prompts[0]) < len(INSTRUCTIONS) / 4
        assert "CANDIDATE REVIEW" not in prompts[0]
        assert "policy_chunks" not in prompts[0]
        assert "CANDIDATE REVIEW" not in prompts[1]
        assert "CANDIDATE REVIEW" in prompts[2]
        assert prompts[3] == prompts[0]
        planner.reset()
        assert planner._prompt_profile == "review"
    finally:
        planner.close()


def test_motion_prompts_lift_ten_centimetres_without_small_orientation_rule():
    from agent2act.adapters.planners.motion_context import fixed_motion_instructions
    from agent2act.adapters.planners.prompts import planner_instructions

    for text in (planner_instructions("motion"), fixed_motion_instructions(False, grasp_geometry=True)):
        flat = " ".join(text.split())
        assert "command TCP z 0.10 m above the measured grasp height" in flat
        assert "prefer small orientation" not in flat
