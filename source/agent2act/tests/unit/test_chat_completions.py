# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import io
import json

import pytest
from PIL import Image

from agent2act.adapters.planners.chat_completions import ChatCompletionsPlanner
from agent2act.adapters.planners.prompts import INSTRUCTIONS
from agent2act.adapters.telemetry import capture_artifacts
from agent2act.bootstrap import ApplicationFactory, default_registry
from agent2act.configuration import parse_settings
from agent2act.domain.errors import ConfigurationError, ContractError
from agent2act.service.trace import RequestTrace


def reply(decision):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(decision)}}]}


def test_multimodal_history_depth_and_actual_wire_traces(observation, motion_decision, tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (800, 400), "red").save(buf, "JPEG")
    observation.images = {"head": {"encoding": "JPEG", "image_data": buf.getvalue()}}
    observation.history = [{"images": observation.images}]
    calls = []
    planner = ChatCompletionsPlanner(
        model="qwen3-vl-plus",
        enable_thinking=False,
        image_max_side=640,
        use_depth=True,
        max_output_tokens=1234,
        post=lambda body: calls.append(body) or reply(motion_decision),
    )
    with capture_artifacts(RequestTrace(tmp_path)):
        decision = planner.decide(observation, {"mode": "motion"})
    assert decision.mode == "motion"
    body = calls[0]
    assert set(body) == {"model", "messages", "max_tokens", "stream", "enable_thinking"}
    assert body["max_tokens"] == 1234
    assert body["stream"] is False and body["enable_thinking"] is False
    assert body["messages"][0] == {"role": "system", "content": INSTRUCTIONS}
    parts = body["messages"][1]["content"]
    state = json.loads(parts[0]["text"])
    assert state["context"] == {"mode": "motion"}
    assert len([p for p in parts if p["type"] == "image_url"]) == 2
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "previous chunk history" in parts[1]["text"]
    assert "current" in parts[4]["text"]
    assert json.loads(parts[-1]["text"])["depth"]["available"] is False
    artifact = json.loads(next(tmp_path.glob("*-llm-request.json")).read_text())
    ref = artifact["messages"][1]["content"][2]["image_url"]["url"]
    with Image.open(tmp_path / ref["$artifact"]) as image:
        assert image.size == (640, 320)
    assert json.loads(next(tmp_path.glob("*-llm-response.json")).read_text()) == reply(motion_decision)
    planner.reset()
    planner.close()
    planner.close()
    with pytest.raises(RuntimeError, match="closed"):
        planner.decide(observation, {})


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"error": {"message": "failed"}},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
        {"choices": [{"finish_reason": "content_filter"}]},
        {"choices": [{"finish_reason": "tool_calls"}]},
        {"choices": [{"finish_reason": "stop", "message": {"refusal": "refused"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "```json\n{}\n```"}}]},
    ],
)
def test_invalid_responses_are_rejected(observation, response):
    planner = ChatCompletionsPlanner(model="test", post=lambda _: response)
    with pytest.raises(ContractError):
        planner.decide(observation, {})
    planner.close()


def test_thinking_text_is_not_parsed_as_decision(observation, motion_decision):
    response = reply(motion_decision)
    response["choices"][0]["message"]["reasoning_content"] = "not a JSON decision"
    planner = ChatCompletionsPlanner(model="test", post=lambda _: response)
    assert planner.decide(observation, {}).mode == "motion"
    planner.close()


@pytest.mark.parametrize("options", [{"reasoning_effort": "low"}, {"enable_thinking": "false"}])
def test_unsupported_options_rejected(options):
    with pytest.raises(ContractError):
        ChatCompletionsPlanner(model="test", **options)


def test_registered_planner_is_closed_on_construction_failure(config, monkeypatch):
    registry = default_registry()
    factory = registry.planners.resolve("chat_completions")
    planner = factory({"model": "qwen3-vl-plus", "api_key_env": None})
    config["planner"] = {"type": "chat_completions", "model": "qwen3-vl-plus"}

    def resolve(name):
        assert name == "chat_completions"
        return lambda options: planner

    def fail(spec):
        raise ContractError("construction failed")

    monkeypatch.setattr(registry.planners, "resolve", resolve)
    monkeypatch.setattr("agent2act.bootstrap.MotionCompiler", fail)
    with pytest.raises(ConfigurationError):
        ApplicationFactory(parse_settings(config), registry).create_session()
    assert planner._closed.is_set()
