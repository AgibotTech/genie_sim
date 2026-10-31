# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import io
import json
from copy import deepcopy

import pytest
from PIL import Image

from agent2act.adapters.planners.anthropic import AnthropicPlanner
from agent2act.adapters.telemetry import capture_artifacts
from agent2act.bootstrap import ApplicationFactory, default_registry
from agent2act.configuration import parse_settings
from agent2act.domain.errors import ConfigurationError, ContractError


def reply(decision):
    return {
        "type": "message",
        "model": "offline",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": json.dumps(decision)}],
    }


def tool_reply(name="read_depth_pixels", stop="tool_use"):
    return {
        "type": "message",
        "stop_reason": stop,
        "content": [
            {"type": "thinking", "thinking": "", "signature": "offline-signature"},
            {
                "type": "tool_use",
                "id": "depth_1",
                "name": name,
                "input": {"camera": "head", "pixels": [{"label": "test", "u": 0, "v": 0}]},
            },
        ],
    }


def test_multimodal_tools_signed_thinking_and_wire_traces(observation, motion_decision):
    image = io.BytesIO()
    Image.new("RGB", (32, 16), "red").save(image, "JPEG")
    observation.images = {"head": {"encoding": "JPEG", "image_data": image.getvalue()}}
    calls, artifacts = [], []
    first = tool_reply()

    def post(body):
        calls.append(deepcopy(body))
        return deepcopy(first) if len(calls) == 1 else reply(motion_decision)

    planner = AnthropicPlanner(
        model="offline",
        api_key_env=None,
        reasoning_effort="high",
        use_depth=True,
        send_raw_depth_image=False,
        enable_raw_depth_query=True,
        post=post,
    )
    try:
        with capture_artifacts(lambda kind, data: artifacts.append((kind, deepcopy(data)))):
            assert planner.decide(observation, {"mode": "motion"}).mode == "motion"
        body = calls[0]
        assert "input" not in body and "reasoning" not in body and "store" not in body
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "high"}
        assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
        assert body["tools"][0]["name"] == "read_depth_pixels"
        assert "input_schema" in body["tools"][0]
        assert "For metric depth use read_depth_pixels" in body["system"]
        parts = body["messages"][0]["content"]
        source = next(p["source"] for p in parts if p["type"] == "image")
        assert source["type"] == "base64" and source["media_type"] == "image/jpeg"
        continuation = calls[1]["messages"]
        assert continuation[-2] == {"role": "assistant", "content": first["content"]}
        result = continuation[-1]["content"][0]
        assert result["type"] == "tool_result" and result["tool_use_id"] == "depth_1"
        assert json.loads(result["content"])["motion_executed"] is False
        assert next(data for kind, data in artifacts if kind == "llm-request") == body
        assert next(data for kind, data in artifacts if kind == "llm-response") == reply(motion_decision)
        assert len(planner._geometry_memory) == 1
        planner.reset()
        assert planner._geometry_memory == []
    finally:
        planner.close()
        planner.close()
    with pytest.raises(RuntimeError, match="closed"):
        planner.decide(observation, {})


def test_two_round_budget_then_disable_tools(observation, motion_decision):
    calls = []

    def post(body):
        calls.append(deepcopy(body))
        return tool_reply() if len(calls) < 3 else reply(motion_decision)

    planner = AnthropicPlanner(model="offline", use_depth=True, enable_raw_depth_query=True, post=post)
    try:
        planner.decide(observation, {})
        assert len(calls) == 3
        assert calls[-1]["tool_choice"] == {"type": "none"}
    finally:
        planner.close()


def test_gateway_two_calls_in_one_round_preserves_thinking_and_total_budget(observation, motion_decision):
    first = tool_reply()
    second = deepcopy(first["content"][-1])
    second["id"] = "depth_2"
    second["input"]["camera"] = "hand_right"
    first["content"].append(second)
    calls = []

    def post(body):
        calls.append(deepcopy(body))
        return deepcopy(first) if len(calls) == 1 else reply(motion_decision)

    planner = AnthropicPlanner(model="offline", use_depth=True, enable_raw_depth_query=True, post=post)
    try:
        assert planner.decide(observation, {}).mode == "motion"
        assert len(calls) == 2
        assert calls[-1]["tool_choice"] == {"type": "none"}
        assert calls[-1]["messages"][-2]["content"] == first["content"]
        results = calls[-1]["messages"][-1]["content"]
        assert [r["tool_use_id"] for r in results] == ["depth_1", "depth_2"]
        assert all(json.loads(r["content"])["motion_executed"] is False for r in results)
        assert len(planner._geometry_memory) == 2
    finally:
        planner.close()


@pytest.mark.parametrize("count,duplicate", [(3, False), (2, True)])
def test_multi_call_invalid_batch_rejected_before_queries(observation, count, duplicate):
    response = tool_reply()
    for index in range(1, count):
        call = deepcopy(response["content"][-1])
        call["id"] = "depth_1" if duplicate else f"depth_{index + 1}"
        response["content"].append(call)
    planner = AnthropicPlanner(
        model="offline", use_depth=True, enable_raw_depth_query=True, post=lambda _: response
    )
    try:
        with pytest.raises(ContractError, match="budget exceeded|duplicate"):
            planner.decide(observation, {})
        assert planner._geometry_memory == []
    finally:
        planner.close()


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"type": "message", "content": None},
        {"type": "message", "content": [None]},
        {"type": "message", "stop_reason": "max_tokens", "content": [{"type": "text", "text": "{}"}]},
        {"type": "message", "stop_reason": "end_turn", "content": [{"type": "text", "text": None}]},
        {"type": "message", "stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": "{}"}]},
        tool_reply("unknown"),
        tool_reply(stop="max_tokens"),
    ],
)
def test_invalid_response_never_becomes_decision(observation, response):
    planner = AnthropicPlanner(
        model="offline", use_depth=True, enable_raw_depth_query=True, post=lambda _: response
    )
    try:
        with pytest.raises(ContractError):
            planner.decide(observation, {})
    finally:
        planner.close()


def test_third_tool_round_rejected(observation):
    planner = AnthropicPlanner(
        model="offline", use_depth=True, enable_raw_depth_query=True, post=lambda _: tool_reply()
    )
    try:
        with pytest.raises(ContractError, match="budget exceeded"):
            planner.decide(observation, {})
    finally:
        planner.close()


def test_thinking_only_truncation_reports_budget_without_issuing_a_decision(observation):
    calls = []
    response = {
        "type": "message",
        "stop_reason": "max_tokens",
        "content": [{"type": "thinking", "thinking": "", "signature": "offline"}],
        "usage": {"output_tokens": 16384, "output_tokens_details": {"thinking_tokens": 16384}},
    }
    planner = AnthropicPlanner(
        model="offline",
        reasoning_effort="high",
        max_output_tokens=16384,
        post=lambda body: calls.append(body) or response,
    )
    try:
        with pytest.raises(ContractError, match="output budget exhausted") as error:
            planner.decide(observation, {})
        assert "stop_reason=max_tokens" in str(error.value)
        assert "output_tokens=16384" in str(error.value)
        assert "thinking_tokens=16384" in str(error.value)
        assert len(calls) == 1
        assert calls[0]["output_config"] == {"effort": "high"}
        assert planner._geometry_memory == []
    finally:
        planner.close()


@pytest.mark.parametrize(
    "options",
    [
        {"json_mode": True},
        {"stream": True},
        {"reasoning_summary": "auto"},
        {"reasoning_effort": "xhigh"},
        {"invalid_encrypted_content_retries": 1},
        {"invalid_encrypted_content_replans": 1},
    ],
)
def test_unsupported_options_rejected(options):
    with pytest.raises(ContractError):
        AnthropicPlanner(model="offline", **options)


def test_registered_planner_closed_on_construction_failure(config, monkeypatch):
    registry = default_registry()
    planner = registry.planners.resolve("anthropic")({"model": "offline", "api_key_env": None})
    config["planner"] = {"type": "anthropic", "model": "offline"}
    monkeypatch.setattr(registry.planners, "resolve", lambda _: lambda options: planner)

    def fail(*args):
        raise ContractError("construction failed")

    monkeypatch.setattr("agent2act.bootstrap.MotionCompiler", fail)
    with pytest.raises(ConfigurationError):
        ApplicationFactory(parse_settings(config), registry).create_session()
    assert planner._closed.is_set()
