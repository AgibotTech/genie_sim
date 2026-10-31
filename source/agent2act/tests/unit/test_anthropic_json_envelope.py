# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
from dataclasses import asdict

import pytest

from agent2act.adapters.planners.anthropic import AnthropicPlanner
from agent2act.domain.errors import ContractError, PlanningExhaustedError


def response(text):
    return {
        "type": "message",
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "not a decision", "signature": "offline"},
            {"type": "text", "text": text},
        ],
    }


@pytest.mark.parametrize(
    "wrap",
    [
        "{}",
        "  {}\n",
        "```json\n{}\n```",
        "```JSON\r\n{}\r\n```",
        "```\n{}\n```",
        "Here is the requested decision:\n```json\n{}\n```\nDone.",
        "Here is the requested decision:\n{}\nDone.",
    ],
)
def test_single_decision_envelope_preserves_decision_and_request(observation, motion_decision, wrap):
    calls = []

    def post(body):
        calls.append(body)
        return response(wrap.format(json.dumps(motion_decision)))

    planner = AnthropicPlanner(model="offline", api_key_env=None, post=post)
    try:
        result = planner.decide(observation, {})
    finally:
        planner.close()
    plain = AnthropicPlanner(
        model="offline", api_key_env=None, post=lambda _: response(json.dumps(motion_decision))
    )
    try:
        assert asdict(result) == asdict(plain.decide(observation, {}))
    finally:
        plain.close()
    assert len(calls) == 1
    assert calls[0]["model"] == "offline"


@pytest.mark.parametrize(
    "wrap",
    [
        "{0}\n{0}",
        "```json\n{0}\n```\n```json\n{0}\n```",
        "Example {0}\n```json\n{0}\n```",
        "[\n{0}\n]",
        'Broken outer {{"decision":{0}',
        "```python\n{0}\n```",
        "```json\n{0}",
        "null\n{0}",
        '{0}\n"another value"',
        '"quoted {0} text"',
    ],
)
def test_ambiguous_or_incomplete_envelope_is_rejected(observation, motion_decision, wrap):
    planner = AnthropicPlanner(
        model="offline", api_key_env=None, post=lambda _: response(wrap.format(json.dumps(motion_decision)))
    )
    try:
        with pytest.raises(ContractError):
            planner.decide(observation, {})
    finally:
        planner.close()


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"mode":"motion",}\n```',
        'Here it is: {"mode":"motion", "motion":{}}',
        '```json\n{"mode":"motion","mode":"policy"}\n```',
        '```json\n{"mode":"invalid","instruction":"test","reason":"test"}\n```',
    ],
)
def test_unwrapping_never_repairs_syntax_or_bypasses_decision_validation(observation, text):
    planner = AnthropicPlanner(model="offline", api_key_env=None, post=lambda _: response(text))
    try:
        with pytest.raises(ContractError):
            planner.decide(observation, {})
    finally:
        planner.close()


def test_output_budget_exhaustion_cannot_be_unwrapped(observation, motion_decision):
    reply = response("```json\n" + json.dumps(motion_decision) + "\n```")
    reply["stop_reason"] = "max_tokens"
    planner = AnthropicPlanner(model="offline", api_key_env=None, post=lambda _: reply)
    try:
        with pytest.raises(ContractError, match="output budget exhausted"):
            planner.decide(observation, {})
    finally:
        planner.close()


def test_ambiguous_reply_exhausts_without_issuing(config, request_payload, monkeypatch):
    from agent2act.bootstrap import build_session

    text = json.dumps(config["planner"]["decisions"][0])
    monkeypatch.setattr(AnthropicPlanner, "_post", lambda self, body: response(text + "\n" + text))
    config["planner"] = {"type": "anthropic", "model": "offline", "api_key_env": None}
    config["execution"]["max_replans"] = 1
    events = []
    with build_session(config, event_sink=events.append) as session:
        with pytest.raises(PlanningExhaustedError):
            session.handle(request_payload)
    assert not any(e.get("status") == "issued" for e in events)


def test_valid_json_with_braces_and_backticks_in_text_is_unchanged(observation, motion_decision):
    motion_decision["reason"] = 'Keep {the target} and [the grippers] still; "```json" is text.'
    planner = AnthropicPlanner(
        model="offline", api_key_env=None, post=lambda _: response(json.dumps(motion_decision))
    )
    try:
        assert planner.decide(observation, {}).reason == motion_decision["reason"]
    finally:
        planner.close()


def test_wrapped_decision_still_obeys_robot_limits(config, request_payload, monkeypatch):
    from agent2act.bootstrap import build_session

    decision = {
        "mode": "motion",
        "instruction": "Move the arm",
        "reason": "offline invalid-target test",
        "motion": {"frame": "base_link", "waypoints": [{"duration_s": 1, "grippers": [999, 0]}]},
    }
    text = "```json\n" + json.dumps(decision) + "\n```"
    monkeypatch.setattr(AnthropicPlanner, "_post", lambda self, body: response(text))
    config["planner"] = {"type": "anthropic", "model": "offline", "api_key_env": None}
    config["execution"]["max_replans"] = 0
    events = []
    with build_session(config, event_sink=events.append) as session:
        with pytest.raises(PlanningExhaustedError, match="max_chunk_steps|bounds|limit"):
            session.handle(request_payload)
    assert not any(e.get("status") == "issued" for e in events)
