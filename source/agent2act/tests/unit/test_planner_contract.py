# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Planner authority, execution boundaries and observation contracts."""

import copy
import json

import pytest

from agent2act.adapters.planners.chat_completions import ChatCompletionsPlanner
from agent2act.application.settings import ExecutionConfig
from agent2act.bootstrap import ApplicationFactory, build_session, default_registry
from agent2act.configuration import parse_settings
from agent2act.domain.errors import ContractError
from agent2act.execution.interpolation import LocalInterpolation


def reply(decision):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(decision)}}]}


def test_selected_frame_is_presented_without_relabeling(observation, motion_decision):
    # Measured in the final trace: arm_base_link.z is base_link.y, not height.
    observation.end_pose["arm_base_link"] = copy.deepcopy(observation.end_pose["base_link"])
    observation.end_pose["arm_base_link"]["left_arm"]["position"] = [0.41449, -0.09285, 0.25651]
    observation.end_pose["base_link"]["left_arm"]["position"] = [0.60006, 0.25651, 1.03305]
    calls = []
    planner = ChatCompletionsPlanner(
        model="test", post=lambda body: calls.append(body) or reply(motion_decision), json_mode=True
    )
    try:
        planner.decide(observation, {"execution": {"motion_frame": "base_link"}})
        state = json.loads(calls[0]["messages"][1]["content"][0]["text"])
        assert list(state["states"]["end_pose"]) == ["base_link"]
        assert state["states"]["end_pose"]["base_link"]["left_arm"]["position"][2] == 1.03305
        assert observation.end_pose["arm_base_link"]["left_arm"]["position"][2] == 0.25651
        assert calls[0]["response_format"] == {"type": "json_object"}
        del observation.end_pose["base_link"]
        with pytest.raises(ContractError, match="missing left end_pose"):
            planner.decide(observation, {"execution": {"motion_frame": "base_link"}})
        assert len(calls) == 1  # No guessing when the selected frame is absent.
    finally:
        planner.close()


def test_frame_constraint_rejects_custom_planner_before_action(config, request_payload):
    config["execution"].update(motion_frame="base_link", motion_frame_axes=["forward", "left", "up"])
    valid = copy.deepcopy(config["planner"]["decisions"][0])
    invalid = copy.deepcopy(valid)
    invalid["motion"]["frame"] = "arm_base_link"
    config["planner"]["decisions"] = [invalid, valid]
    events = []
    with build_session(config, event_sink=events.append) as session:
        result = session.handle(request_payload)
        assert result["result"]["left_arm"]["base_link"] == "base_link"
    rejected = next(e for e in events if e.get("status") == "rejected")
    assert "do not relabel" in rejected["error"]
    assert len([e for e in events if e.get("status") == "issued"]) == 1


def test_multi_waypoint_prefix_is_reported_and_next_observation_replans(
    config, request_payload, observation, spec
):
    decision = copy.deepcopy(config["planner"]["decisions"][0])
    first = observation.eef("left", "base_link")
    first[0] += 0.01
    last = [first[0] + 0.01, *first[1:]]
    decision["motion"]["waypoints"] = [
        {"duration_s": 0.5, "left_pose": first},
        {"duration_s": 0.5, "left_pose": last},
    ]
    expected = LocalInterpolation(spec).generate(observation, decision["motion"])
    config["execution"].update(execute_steps=8, max_motion_continuations=2)
    config["planner"]["decisions"] = [decision, {**decision, "status": "complete", "motion": None}]
    events = []
    with build_session(config, event_sink=events.append) as session:
        result = session.handle(request_payload)["result"]["agent2act"]
        assert result["generated_steps"] == 20
        assert result["issued_steps"] == 8
        assert result["discarded_steps"] == 12
        issued = session.runtime._last_issued
        assert issued.left_arm == expected.left_arm[:8]
        assert issued.left_arm[-1] != last
        assert session.runtime._pending_motion is None
        assert session.handle(request_payload)["result"]["agent2act"]["planner_status"] == "complete"
    assert not any(e.get("status") == "rejected" for e in events)


def test_short_stages_that_fit_budget_remain_supported(config, request_payload):
    config["planner"]["decisions"][0]["motion"]["waypoints"] = [{"duration_s": 0.05}, {"duration_s": 0.05}]
    with build_session(config) as session:
        result = session.handle(request_payload)["result"]["agent2act"]
    assert result["generated_steps"] == result["issued_steps"] == 2


def test_planner_can_reaffirm_complete_hold_or_resume_without_forced_movement(config, request_payload):
    decision = config["planner"]["decisions"][0]
    complete = {**decision, "status": "complete", "motion": None}
    hold = {**decision, "motion": {"frame": "base_link", "waypoints": [{"duration_s": 0.1}]}}
    config["execution"]["max_replans"] = 0
    config["planner"]["decisions"] = [complete, complete, hold, decision]
    events = []
    with build_session(config, event_sink=events.append) as session:
        for _ in range(2):
            result = session.handle(request_payload)["result"]["agent2act"]
            assert result["status"] == "planner_complete_unverified"
            assert session.runtime.episode_key is not None
        result = session.handle(request_payload)["result"]["agent2act"]
        assert result["planner_status"] == "continue"
        assert result["issued_steps"] == 2
        assert session.handle(request_payload)["result"]["agent2act"]["planner_status"] == "continue"
        request_payload["params"]["episode_done"] = True
        assert session.handle(request_payload)["result"]["agent2act"]["status"] == "episode_done"
        assert session.runtime.episode_key is None
        assert session.runtime.memory == []
    assert not any(e.get("status") == "rejected" for e in events)


def test_previous_claim_tracking_and_scaled_gripper_commands_reach_model(config, request_payload):
    calls = []
    complete = {**config["planner"]["decisions"][0], "status": "complete", "motion": None}
    config["robot"].update(gripper_observation_scale=[0.5, 0.25])
    request_payload["params"]["states"]["gripper_states"] = [1, 1]
    registry = default_registry()
    registry.planners.register(
        "offline",
        lambda _: ChatCompletionsPlanner(
            model="test", post=lambda body: calls.append(body) or reply(complete)
        ),
    )
    config["planner"] = {"type": "offline"}
    config["execution"]["max_replans"] = 0
    with ApplicationFactory(parse_settings(config), registry).create_session() as session:
        session.handle(request_payload)
        session.handle(request_payload)
    state = json.loads(calls[1]["messages"][1]["content"][0]["text"])
    context = state["context"]
    assert "completion_feedback" not in context
    previous = next(e for e in context["recent_events"] if e["status"] == "issued")
    assert previous["planner_status"] == "complete"
    assert previous["issued_steps"] == 1
    assert "frame" in previous and "kind" in previous
    assert "NOT distance to an object" in state["context"]["tracking_feedback"]["meaning"]
    assert state["context"]["gripper_commands"] == [0.5, 0.25]


def test_malformed_json_reports_location_and_single_gripper_is_not_guessed(config, request_payload):
    valid = copy.deepcopy(config["planner"]["decisions"][0])
    bad = copy.deepcopy(valid)
    bad["motion"]["waypoints"][0]["grippers"] = [0.0]
    answers = iter(
        [
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"motion": {"waypoints": [{} }}]}'}}
                ]
            },
            reply(bad),
            reply(valid),
        ]
    )
    registry = default_registry()
    registry.planners.register(
        "offline", lambda _: ChatCompletionsPlanner(model="test", post=lambda _: next(answers))
    )
    config["planner"] = {"type": "offline"}
    config["execution"]["max_replans"] = 2
    events = []
    with ApplicationFactory(parse_settings(config), registry).create_session(
        event_sink=events.append
    ) as session:
        session.handle(request_payload)
    errors = [e["error"] for e in events if e.get("status") == "rejected"]
    assert "line 1 column" in errors[0]
    assert "grippers must have 2 elements" in errors[1]
    assert len([e for e in events if e.get("status") == "issued"]) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"motion_frame": "world"},
        {"motion_frame_axes": ["forward", "left", "up"]},
        {"motion_frame": "base_link", "motion_frame_axes": ["forward", "left"]},
    ],
)
def test_invalid_frame_settings_rejected(options):
    with pytest.raises(ContractError):
        ExecutionConfig(**options)


def test_invalid_json_mode_rejected():
    with pytest.raises(ContractError, match="json_mode"):
        ChatCompletionsPlanner(model="test", json_mode="true")
