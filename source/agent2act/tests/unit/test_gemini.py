# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import io
import json
import struct
from copy import deepcopy

import pytest
from PIL import Image

from agent2act.adapters.planners.gemini import GeminiPlanner
from agent2act.adapters.telemetry import capture_artifacts
from agent2act.bootstrap import ApplicationFactory, default_registry
from agent2act.configuration import parse_settings
from agent2act.domain.errors import ConfigurationError, ContractError


def reply(decision):
    return {
        "modelVersion": "offline",
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [{"text": json.dumps(decision)}],
                },
            }
        ],
    }


def tool_reply(call_id="depth1"):
    return {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Public summary", "thought": True},
                        {
                            "thoughtSignature": "opaque-signed-data",
                            "functionCall": {
                                "id": call_id,
                                "name": "read_depth_pixels",
                                "args": {"camera": "head", "pixels": [{"label": "test", "u": 0, "v": 0}]},
                            },
                        },
                    ],
                },
            }
        ],
    }


def sensor(observation):
    image = io.BytesIO()
    Image.new("RGB", (3, 2), "red").save(image, "JPEG")
    observation.images = {
        "head": {
            "encoding": "JPEG",
            "image_data": image.getvalue(),
            "width": 3,
            "height": 2,
            "snapshot_id": "current",
        },
        "head_depth": {
            "encoding": "RAW_UINT16",
            "width": 3,
            "height": 2,
            "image_data": struct.pack("<6H", *([1000] * 6)),
            "snapshot_id": "current",
        },
    }
    return observation


def test_native_images_depth_signature_summary_and_lifecycle(observation, motion_decision):
    calls, artifacts = [], []
    first = tool_reply()

    def post(body):
        calls.append(deepcopy(body))
        return deepcopy(first) if len(calls) == 1 else reply(motion_decision)

    planner = GeminiPlanner(
        model="offline",
        api_key_env=None,
        post=post,
        json_mode=True,
        reasoning_effort="high",
        reasoning_summary="auto",
        use_depth=True,
        enable_raw_depth_query=True,
        send_raw_depth_image=False,
    )
    try:
        with capture_artifacts(lambda kind, value: artifacts.append((kind, deepcopy(value)))):
            assert planner.decide(sensor(observation), {}).mode == "motion"
        body = calls[0]
        assert set(body) == {"systemInstruction", "contents", "generationConfig", "tools", "toolConfig"}
        assert body["generationConfig"]["thinkingConfig"] == {
            "thinkingLevel": "HIGH",
            "includeThoughts": True,
        }
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        declaration = body["tools"][0]["functionDeclarations"][0]
        assert declaration["name"] == "read_depth_pixels"
        assert declaration["parametersJsonSchema"]["additionalProperties"] is False
        assert "For metric depth use read_depth_pixels" in body["systemInstruction"]["parts"][0]["text"]
        assert any(
            p.get("inlineData", {}).get("mimeType") == "image/jpeg" for p in body["contents"][0]["parts"]
        )
        assert calls[1]["contents"][-2] == first["candidates"][0]["content"]
        result = calls[1]["contents"][-1]["parts"][0]["functionResponse"]
        assert result["id"] == "depth1" and result["name"] == "read_depth_pixels"
        assert result["response"]["pixels"][0]["depth_mm"] == 1000
        summaries = [v for k, v in artifacts if k == "reasoning-summary"]
        assert summaries[0]["summary"] == ["Public summary"]
        assert summaries[1]["status"] == "not_provided"
        assert next(v for k, v in artifacts if k == "model-identity")["model_identity_status"] == "match"
        assert len(planner._geometry_memory) == 1
        planner.reset()
        assert planner._geometry_memory == []
    finally:
        planner.close()
        planner.close()
    with pytest.raises(RuntimeError, match="closed"):
        planner.decide(observation, {})


@pytest.mark.parametrize("parallel", [False, True])
def test_two_call_budget_and_idless_continuation(observation, motion_decision, parallel):
    calls = []
    first = tool_reply()
    first["candidates"][0]["content"]["parts"][1]["functionCall"].pop("id")
    if parallel:
        first["candidates"][0]["content"]["parts"].append(
            deepcopy(first["candidates"][0]["content"]["parts"][1])
        )

    def post(body):
        calls.append(deepcopy(body))
        return deepcopy(first) if len(calls) < (2 if parallel else 3) else reply(motion_decision)

    planner = GeminiPlanner(model="offline", use_depth=True, enable_raw_depth_query=True, post=post)
    try:
        planner.decide(observation, {})
        assert calls[-1]["toolConfig"] == {"functionCallingConfig": {"mode": "NONE"}}
        assert len(planner._geometry_memory) == 2
        assert "id" not in calls[1]["contents"][-1]["parts"][0]["functionResponse"]
    finally:
        planner.close()


@pytest.mark.parametrize(
    "damage", ["unknown", "duplicate", "budget", "args", "truncated", "blocked", "thought_call"]
)
def test_reject_entire_invalid_tool_batch_before_execution(observation, damage):
    response = tool_reply()
    candidate = response["candidates"][0]
    part = candidate["content"]["parts"][1]
    if damage == "unknown":
        part["functionCall"]["name"] = "move_robot"
    elif damage == "duplicate":
        candidate["content"]["parts"].append(deepcopy(part))
    elif damage == "budget":
        for i in range(2):
            extra = deepcopy(part)
            extra["functionCall"]["id"] = str(i)
            candidate["content"]["parts"].append(extra)
    elif damage == "args":
        part["functionCall"]["args"] = "{}"
    elif damage == "truncated":
        candidate["finishReason"] = "MAX_TOKENS"
    elif damage == "blocked":
        response["promptFeedback"] = {"blockReason": "SAFETY"}
    elif damage == "thought_call":
        part["thought"] = True
    planner = GeminiPlanner(
        model="offline", use_depth=True, enable_raw_depth_query=True, post=lambda _: response
    )
    try:
        with pytest.raises(ContractError):
            planner.decide(observation, {})
        assert planner._geometry_memory == []
    finally:
        planner.close()


def test_third_call_rejected(observation):
    planner = GeminiPlanner(
        model="offline", use_depth=True, enable_raw_depth_query=True, post=lambda _: tool_reply()
    )
    try:
        with pytest.raises(ContractError, match="budget exceeded"):
            planner.decide(observation, {})
        assert len(planner._geometry_memory) == 2
    finally:
        planner.close()


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"candidates": []},
        {"candidates": [None]},
        {"candidates": [{"finishReason": "SAFETY"}]},
        {"candidates": [{"finishReason": "CONTINUATION"}]},
        {"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [None]}}]},
        {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"role": "model", "parts": [{"thought": True, "text": "{}"}]},
                }
            ]
        },
    ],
)
def test_invalid_final_response(observation, response):
    planner = GeminiPlanner(model="offline", post=lambda _: response)
    try:
        with pytest.raises(ContractError):
            planner.decide(observation, {})
    finally:
        planner.close()


@pytest.mark.parametrize(
    "options",
    [
        {"stream": True},
        {"reasoning_effort": "max"},
        {"reasoning_summary": "detailed"},
        {"invalid_encrypted_content_retries": 1},
        {"invalid_encrypted_content_replans": 1},
    ],
)
def test_unsupported_options(options):
    with pytest.raises(ContractError):
        GeminiPlanner(model="offline", **options)


def test_construction_failure_closes_registered_planner(config, monkeypatch):
    registry = default_registry()
    planner = registry.planners.resolve("gemini")({"model": "offline", "api_key_env": None})
    config["planner"] = {"type": "gemini", "model": "offline"}
    monkeypatch.setattr(registry.planners, "resolve", lambda _: lambda options: planner)

    def fail(*args):
        raise ContractError("construction failed")

    monkeypatch.setattr("agent2act.bootstrap.MotionCompiler", fail)
    with pytest.raises(ConfigurationError):
        ApplicationFactory(parse_settings(config), registry).create_session()
    assert planner._closed.is_set()
