# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import json

import pytest

from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.motion_context import confirmation_wait_instructions
from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.bootstrap import build_session


def hold_config(config, duration=2.0):
    config["robot"]["dt_s"] = 1 / 30
    config["execution"].update(execute_steps=30, max_motion_continuations=1)
    decision = config["planner"]["decisions"][0]
    decision.update(instruction="Hold for confirmation", success_condition="Observed check passes")
    decision["motion"]["waypoints"] = [{"duration_s": duration}]
    return decision


def test_two_window_hold_uses_one_decision_and_preserves_commands(config, request_payload):
    hold_config(config)
    events = []
    with build_session(config, event_sink=events.append) as session:
        session.handle(request_payload)
        first = copy.deepcopy(session.runtime._last_issued)
        session.handle(request_payload)  # Only one scripted decision is available.
        second = session.runtime._last_issued
        assert first.steps == second.steps == 30
        for key in ("left_arm", "right_arm", "grippers", "body"):
            assert getattr(first, key) == getattr(second, key)
        assert session.runtime._pending_motion is None
        assert session.runtime.planner.index == 1
    assert len([e for e in events if e.get("status") == "motion_continued"]) == 1


def test_hold_budget_requires_replanning_after_two_windows(config, request_payload):
    decision = hold_config(config)
    config["planner"]["decisions"].append(copy.deepcopy(decision))
    events = []
    with build_session(config, event_sink=events.append) as session:
        for _ in range(3):
            session.handle(request_payload)
        assert session.runtime.planner.index == 2
    assert len([e for e in events if e.get("status") == "motion_continued"]) == 1


def test_hold_tracking_failure_returns_to_planner(config, request_payload):
    decision = hold_config(config)
    config["planner"]["decisions"].append(copy.deepcopy(decision))
    events = []
    with build_session(config, event_sink=events.append) as session:
        session.handle(request_payload)
        request_payload["params"]["states"]["gripper_states"] = [0.0, 0.0]
        session.handle(request_payload)
    assert not any(e.get("status") == "motion_continued" for e in events)


@pytest.mark.parametrize("planner_type", [ResponsesPlanner, EmbodiedPlanner])
def test_hold_guidance_uses_current_window_and_resets(planner_type, observation, motion_decision):
    calls = []

    def post(body):
        calls.append(body)
        return {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(motion_decision)}]}
            ]
        }

    context = {
        "robot": {"dt_s": 1 / 30, "max_chunk_steps": 256},
        "execution": {"mode": "motion", "execute_steps": 30, "max_motion_continuations": 1},
    }
    planner = planner_type(model="offline", post=post)
    try:
        planner.decide(observation, context)
        prompt = json.dumps(calls[-1])
        assert "CONFIRMATION WAIT" in prompt
        assert "duration_s=2" in prompt
        planner.reset()
        assert "CONFIRMATION WAIT" not in planner._context_instructions()
        planner.decide(observation, {"execution": {"mode": "motion"}})
        assert "CONFIRMATION WAIT" not in json.dumps(calls[-1])
    finally:
        planner.close()


def test_fixed_steps_does_not_receive_duration_hold_guidance():
    assert (
        confirmation_wait_instructions(
            {
                "robot": {"dt_s": 1 / 30},
                "execution": {
                    "execute_steps": 30,
                    "executors": {"motion": {"compiler": {"fixed_steps": 30}}},
                },
            }
        )
        == ""
    )
