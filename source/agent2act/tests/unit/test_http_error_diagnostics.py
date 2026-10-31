# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import io
import json
import logging
import urllib.error
import urllib.request
from email.message import Message

import pytest

from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.adapters.telemetry import capture_artifacts


@pytest.mark.parametrize("status", [400, 429])
def test_http_error_trace_redacts_key_and_preserves_failure(status, monkeypatch, caplog):
    secret = "OFFLINE_ONLY_SECRET"
    raw = json.dumps(
        {
            "error": {
                "type": "invalid_request_error",
                "code": "bad_item",
                "param": "input[3]",
                "message": "Rejected Bearer " + secret,
                "extra": secret,
            },
            "unexpected": secret,
        }
    ).encode()
    stream = io.BytesIO(raw)
    calls = []
    artifacts = []

    def send(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, status, "error", Message(), stream)

    monkeypatch.setenv("OFFLINE_TEST_KEY", secret)
    monkeypatch.setattr(urllib.request, "urlopen", send)
    planner = ResponsesPlanner(model="test", api_key_env="OFFLINE_TEST_KEY", max_retries=0)
    with (
        capture_artifacts(lambda kind, value: artifacts.append((kind, value))),
        caplog.at_level(logging.INFO),
    ):
        with pytest.raises(RuntimeError, match=f"LLM HTTP status {status}"):
            planner._post({"model": "test", "input": []})
    planner.close()
    assert len(calls) == 1 and stream.closed
    kind, artifact = artifacts[0]
    assert kind == "llm-http-error"
    assert artifact["http_status"] == status
    assert artifact["error"] == {
        "type": "invalid_request_error",
        "code": "bad_item",
        "param": "input[3]",
        "message": "Rejected Bearer [REDACTED]",
    }
    assert secret not in json.dumps(artifacts)
    assert "Rejected Bearer" not in caplog.text


@pytest.mark.parametrize(
    "planner_type,headers",
    [
        ("gemini", {"x-goog-api-key"}),
        ("anthropic", {"authorization", "x-api-key"}),
        ("responses", {"authorization"}),
    ],
)
def test_each_provider_credential_header_is_redacted(planner_type, headers, monkeypatch):
    """Gemini authenticates with x-goog-api-key and sends no Authorization at all."""
    from agent2act.adapters.planners.anthropic import AnthropicPlanner
    from agent2act.adapters.planners.gemini import GeminiPlanner

    secret = "OFFLINE_ONLY_SECRET"
    raw = json.dumps({"error": {"code": "bad_key", "message": f"API key not valid: {secret}"}}).encode()
    requests, artifacts = [], []

    def send(request, timeout):
        requests.append(request)
        raise urllib.error.HTTPError(request.full_url, 400, "error", Message(), io.BytesIO(raw))

    monkeypatch.setenv("OFFLINE_TEST_KEY", secret)
    monkeypatch.setattr(urllib.request, "urlopen", send)
    planner_class = {"gemini": GeminiPlanner, "anthropic": AnthropicPlanner}.get(
        planner_type, ResponsesPlanner
    )
    planner = planner_class(model="test", api_key_env="OFFLINE_TEST_KEY", max_retries=0)
    with capture_artifacts(lambda kind, value: artifacts.append((kind, value))):
        with pytest.raises(RuntimeError, match="LLM HTTP status 400"):
            planner._post({"model": "test", "input": []})
    planner.close()

    # The credential is sent under this provider's own header; Gemini sends no Authorization.
    sent = {name.lower() for name, value in requests[0].header_items() if secret in value}
    assert sent == headers

    kind, artifact = artifacts[0]
    assert kind == "llm-http-error"
    assert artifact["error"]["code"] == "bad_key"
    assert artifact["error"]["message"] == "API key not valid: [REDACTED]"
    assert secret not in json.dumps(artifacts)


@pytest.mark.parametrize("raw", [b"<html>gateway failure</html>", b"x" * (16 * 1024 + 2), b"[]"])
def test_unstructured_or_oversized_error_does_not_replace_http_failure(raw, monkeypatch):
    artifacts = []

    def send(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, "error", Message(), io.BytesIO(raw))

    monkeypatch.setattr(urllib.request, "urlopen", send)
    planner = ResponsesPlanner(model="test", api_key_env=None, max_retries=0)
    with capture_artifacts(lambda kind, value: artifacts.append(value)):
        with pytest.raises(RuntimeError, match="LLM HTTP status 400"):
            planner._post({})
    planner.close()
    assert len(artifacts) == 1
    assert "error" not in artifacts[0]
    assert artifacts[0]["body_truncated"] == (len(raw) > 16 * 1024)


def test_error_body_read_failure_keeps_original_http_status(monkeypatch):
    class BrokenBody(io.BytesIO):
        def read(self, size=-1):
            raise TimeoutError("diagnostic read failed")

    def send(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, "error", Message(), BrokenBody())

    artifacts = []
    monkeypatch.setattr(urllib.request, "urlopen", send)
    planner = ResponsesPlanner(model="test", api_key_env=None, max_retries=0)
    with capture_artifacts(lambda kind, value: artifacts.append(value)):
        with pytest.raises(RuntimeError, match="LLM HTTP status 400"):
            planner._post({})
    planner.close()
    assert artifacts[0]["diagnostic_error_type"] == "TimeoutError"


def test_diagnostic_capture_preserves_transient_retry_payload(monkeypatch):
    calls = []

    def send(request, timeout):
        calls.append(request.data)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url, 429, "error", Message(), io.BytesIO(b'{"error":{"code":"rate_limit"}}')
            )
        return io.BytesIO(b'{"output":[]}')

    monkeypatch.setattr(urllib.request, "urlopen", send)
    planner = ResponsesPlanner(
        model="test", api_key_env=None, max_retries=1, retry_backoff_s=0.001, retry_max_delay_s=0.001
    )
    artifacts = []
    with capture_artifacts(lambda kind, value: artifacts.append(value)):
        assert planner._post({"input": [{"type": "reasoning", "encrypted_content": "opaque"}]}) == {
            "output": []
        }
    planner.close()
    assert len(calls) == 2 and calls[0] == calls[1]
    assert len(artifacts) == 1 and artifacts[0]["error"]["code"] == "rate_limit"


@pytest.mark.parametrize("second_fails", [False, True])
def test_encrypted_content_recovery_is_bounded_and_preserves_tools(monkeypatch, second_fails):
    calls = []

    def send(request, timeout):
        calls.append(json.loads(request.data))
        if len(calls) == 1 or second_fails:
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "bad",
                Message(),
                io.BytesIO(b'{"error":{"code":"invalid_encrypted_content"}}'),
            )
        return io.BytesIO(b'{"output":[]}')

    monkeypatch.setattr(urllib.request, "urlopen", send)
    planner = ResponsesPlanner(model="test", api_key_env=None, max_retries=0)
    body = {
        "input": [
            {"role": "user", "content": "observation"},
            {"type": "reasoning", "encrypted_content": "opaque"},
            {"type": "function_call", "call_id": "c", "name": "sensor", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": "measured"},
        ]
    }
    try:
        if second_fails:
            with pytest.raises(RuntimeError, match="400"):
                planner._post(body)
        else:
            assert planner._post(body) == {"output": []}
        assert len(calls) == 2
        assert calls[0] == body
        assert calls[1]["input"] == [body["input"][i] for i in (0, 2, 3)]
        assert len(body["input"]) == 4
    finally:
        planner.close()
