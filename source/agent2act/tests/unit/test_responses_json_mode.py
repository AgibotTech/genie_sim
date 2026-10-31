# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import urllib.error
from email.message import Message

import pytest

from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.bootstrap import build_session
from agent2act.domain.errors import ContractError, PlanningExhaustedError


def test_gateway_instruction_replacement_preserves_input_contract(observation, motion_decision):
    from agent2act.adapters.planners.motion_context import fixed_motion_instructions

    calls = []

    def post(body):
        assert json.dumps(body).count(fixed_motion_instructions(False).splitlines()[0]) == 1
        assert body["instructions"] == "Follow the robot control contract in the developer message in input."
        body = dict(body)
        body["instructions"] = "You are a coding agent."
        calls.append(body)
        developer = [item for item in body["input"] if item["role"] == "developer"]
        assert len(developer) == 1
        assert developer[0]["content"] == [{"type": "input_text", "text": fixed_motion_instructions(False)}]
        return {
            "instructions": body["instructions"],
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(motion_decision)}]}
            ],
        }

    context = {
        "execution": {
            "mode": "motion",
            "allowed_modes": ["motion"],
            "executors": {"motion": {"compiler": {"fixed_steps": 10, "retime_to_limits": True}}},
        }
    }
    planner = ResponsesPlanner(model="test", post=post)
    try:
        assert planner.decide(observation, context).mode == "motion"
    finally:
        planner.close()
    assert len(calls) == 1


@pytest.mark.parametrize("options", [{}, {"json_mode": False}, {"json_mode": True}])
def test_json_mode_registered_request(config, request_payload, monkeypatch, options):
    decision = config["planner"]["decisions"][0]
    calls = []

    def post(self, body):
        calls.append(body)
        return {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(decision)}]}
            ],
        }

    monkeypatch.setattr(ResponsesPlanner, "_post", post)
    config["planner"] = {"type": "responses", "model": "test", **options}
    with build_session(config) as session:
        session.handle(request_payload)
    assert len(calls) == 1
    if options.get("json_mode"):
        assert calls[0]["text"] == {"format": {"type": "json_object"}}
        parts = calls[0]["input"][0]["content"]
        assert json.loads(parts[0]["text"])["instruction"]
        assert {"type": "input_text", "text": "Return one valid JSON decision object."} in parts
    else:
        assert "text" not in calls[0]
    assert "response_format" not in calls[0]


@pytest.mark.parametrize("value", ["true", 1, 0, None, {}, []])
def test_json_mode_requires_boolean(value):
    with pytest.raises(ContractError, match="json_mode must be boolean"):
        ResponsesPlanner(model="test", json_mode=value)


def test_extra_brace_exhausts_replanning_without_issuing(config, request_payload, monkeypatch):
    decision = config["planner"]["decisions"][0]
    # Same extra waypoint closing brace as the recorded failed run.
    text = json.dumps(decision, separators=(",", ":"))
    text = text.replace("}]}", "}}]}")
    calls = []

    def post(self, body):
        calls.append(body)
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}

    monkeypatch.setattr(ResponsesPlanner, "_post", post)
    config["planner"] = {"type": "responses", "model": "test", "json_mode": True}
    config["execution"]["max_replans"] = 1
    events = []
    with build_session(config, event_sink=events.append) as session:
        with pytest.raises(PlanningExhaustedError, match="valid JSON decision object"):
            session.handle(request_payload)
    assert len(calls) == 2
    assert all(body["text"]["format"]["type"] == "json_object" for body in calls)
    assert not any(event.get("status") == "issued" for event in events)


def test_unsupported_json_mode_http_400_retries_without_changing_format(observation, monkeypatch):
    calls = []

    def send(self, request, event):
        calls.append(json.loads(request.data))
        raise urllib.error.HTTPError(request.full_url, 400, "unsupported format", Message(), None)

    monkeypatch.setattr(ResponsesPlanner, "_send_http", send)
    planner = ResponsesPlanner(
        model="test",
        api_key_env=None,
        json_mode=True,
        retry_backoff_s=0.001,
        retry_max_delay_s=0.001,
    )
    try:
        with pytest.raises(RuntimeError, match="LLM HTTP status 400"):
            planner.decide(observation, {})
    finally:
        planner.close()
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    assert calls[0]["text"] == {"format": {"type": "json_object"}}
