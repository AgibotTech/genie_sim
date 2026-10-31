# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import io
import json
import logging
import urllib.error
import urllib.request
from copy import deepcopy
from email.message import Message

import pytest

from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.responses import InvalidEncryptedContentError, ResponsesPlanner
from agent2act.adapters.telemetry import capture_artifacts
from agent2act.domain.errors import ContractError, TransientBackendError

REASONING = {"type": "reasoning", "id": "opaque_item", "encrypted_content": "opaque", "summary": []}


def tool_response(call_id):
    return {
        "output": [
            REASONING,
            {
                "type": "function_call",
                "name": "compose_grasp_pose",
                "call_id": call_id,
                "arguments": json.dumps(
                    {"arm": "left", "center_m": [0.5, 0.1, 0.8], "orientation_xyzw": [1, 0, 0, 0]}
                ),
            },
        ]
    }


def http_error(request, status=400, code="invalid_encrypted_content"):
    return urllib.error.HTTPError(
        request.full_url,
        status,
        "offline error",
        Message(),
        io.BytesIO(json.dumps({"error": {"type": "invalid_request_error", "code": code}}).encode()),
    )


@pytest.mark.parametrize("option", ["invalid_encrypted_content_retries", "invalid_encrypted_content_replans"])
@pytest.mark.parametrize("value", [True, -1, 100, 1.5, "3"])
def test_recovery_limits_are_validated(option, value):
    with pytest.raises(ContractError, match=option):
        EmbodiedPlanner(model="offline", **{option: value})


def test_planning_recovery_can_restart_ninety_nine_times_but_remains_bounded(observation):
    calls = []
    artifacts = []

    def post(body):
        calls.append(body)
        if len(calls) % 2:
            return tool_response(str(len(calls)))
        raise InvalidEncryptedContentError("LLM HTTP status 400: invalid_encrypted_content")

    planner = EmbodiedPlanner(
        model="offline",
        post=post,
        invalid_encrypted_content_replans=99,
        wrist_to_center_m={"left": [0, 0, 0.15], "right": [0, 0, 0.15]},
    )
    try:
        with capture_artifacts(lambda kind, value: artifacts.append((kind, value))):
            with pytest.raises(InvalidEncryptedContentError):
                planner._decision_response(planner._request_body([]), observation)
        assert len(calls) == 200
        assert all(body == calls[0] for body in calls[::2])
        recovery = [value for kind, value in artifacts if kind == "llm-planning-recovery"]
        assert len(recovery) == 100
        assert sum(value["stage"] == "restart_current_planning" for value in recovery) == 99
        assert recovery[-1]["stage"] == "exhausted"
        assert planner._geometry_memory == []
    finally:
        planner.close()


def test_same_request_retry_limit_remains_ten():
    with pytest.raises(ContractError, match="invalid_encrypted_content_retries"):
        EmbodiedPlanner(model="offline", invalid_encrypted_content_retries=11)


def test_encrypted_error_retries_identical_request_without_spending_transient_retries(monkeypatch, caplog):
    requests = []

    def send(request, timeout):
        requests.append(request.data)
        if len(requests) <= 2:
            raise http_error(request)
        if len(requests) == 3:
            raise http_error(request, 500, "server_error")
        return io.BytesIO(b'{"output":[]}')

    monkeypatch.setattr(urllib.request, "urlopen", send)
    planner = ResponsesPlanner(
        model="offline",
        api_key_env=None,
        max_retries=1,
        invalid_encrypted_content_retries=2,
        retry_backoff_s=0.001,
        retry_max_delay_s=0.001,
    )
    try:
        with caplog.at_level(logging.INFO):
            assert planner._post({"input": [REASONING]}) == {"output": []}
        assert len(requests) == 4 and len(set(requests)) == 1
        assert "llm_request_retry" in caplog.text
    finally:
        planner.close()


@pytest.mark.parametrize(
    "status,code,items",
    [
        (400, "bad_item", [REASONING]),
        (401, "invalid_encrypted_content", [REASONING]),
        (400, "invalid_encrypted_content", []),
    ],
)
def test_other_errors_or_no_reasoning_do_not_trigger_recovery(monkeypatch, status, code, items):
    calls = []

    def send(request, timeout):
        calls.append(request.data)
        raise http_error(request, status, code)

    monkeypatch.setattr(urllib.request, "urlopen", send)
    planner = ResponsesPlanner(
        model="offline",
        api_key_env=None,
        invalid_encrypted_content_retries=3,
        retry_backoff_s=0.001,
        retry_max_delay_s=0.001,
    )
    try:
        with pytest.raises(RuntimeError, match=str(status)):
            planner._post({"input": items})
        assert len(calls) == (3 if status == 400 and code == "bad_item" else 1)
        assert len(set(calls)) == 1
    finally:
        planner.close()


def test_same_request_retries_are_bounded_and_cancelled(monkeypatch):
    calls = []

    def send(request, timeout):
        calls.append(request.data)
        raise http_error(request)

    monkeypatch.setattr(urllib.request, "urlopen", send)
    planner = ResponsesPlanner(
        model="offline",
        api_key_env=None,
        max_retries=99,
        invalid_encrypted_content_retries=2,
        retry_backoff_s=0.001,
        retry_max_delay_s=0.001,
    )
    try:
        with pytest.raises(InvalidEncryptedContentError):
            planner._post({"input": [REASONING]})
        assert len(calls) == 3
        monkeypatch.setattr(planner._closed, "wait", lambda delay: True)
        with pytest.raises(RuntimeError, match="closed"):
            planner._post({"input": [REASONING]})
        assert len(calls) == 4
    finally:
        planner.close()


def test_replan_only_configuration_never_strips_reasoning(monkeypatch):
    calls = []

    def send(request, timeout):
        calls.append(json.loads(request.data))
        raise http_error(request)

    monkeypatch.setattr(urllib.request, "urlopen", send)
    planner = EmbodiedPlanner(model="offline", api_key_env=None, invalid_encrypted_content_replans=1)
    body = {"input": [REASONING]}
    try:
        with pytest.raises(InvalidEncryptedContentError):
            planner._post(body)
        assert calls == [body]
    finally:
        planner.close()


def test_default_continuation_fallback_shares_original_deadline(monkeypatch):
    clock = [100.0]
    calls = []
    monkeypatch.setattr("agent2act.adapters.planners.responses.time.monotonic", lambda: clock[0])

    def send(request, timeout):
        calls.append(timeout)
        clock[0] += 2
        raise http_error(request)

    monkeypatch.setattr(urllib.request, "urlopen", send)
    planner = ResponsesPlanner(model="offline", api_key_env=None, retry_budget_s=1)
    try:
        with pytest.raises(TransientBackendError, match="budget exhausted"):
            planner._post({"input": [REASONING]})
        assert calls == [1]
    finally:
        planner.close()


def test_transient_retry_after_default_fallback_preserves_tool_exchange(monkeypatch):
    calls = []
    body = {
        "input": [
            REASONING,
            {"type": "function_call", "call_id": "c", "name": "sensor", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": "measured"},
        ]
    }

    def send(request, timeout):
        calls.append(json.loads(request.data))
        if len(calls) == 1:
            raise http_error(request)
        if len(calls) == 2:
            raise http_error(request, 500, "server_error")
        return io.BytesIO(b'{"output":[]}')

    monkeypatch.setattr(urllib.request, "urlopen", send)
    planner = ResponsesPlanner(
        model="offline",
        api_key_env=None,
        max_retries=1,
        retry_backoff_s=0.001,
        retry_max_delay_s=0.001,
    )
    try:
        assert planner._post(body) == {"output": []}
        assert calls == [body, {"input": body["input"][1:]}, {"input": body["input"][1:]}]
        assert body["input"][0] == REASONING
    finally:
        planner.close()


@pytest.mark.parametrize("success", [True, False])
def test_replanning_restarts_tool_exchange_preserves_history_and_rolls_back_queries(observation, success):
    before = deepcopy(observation)
    requests, artifacts = [], []
    old_memory = [{"sequence": 9, "name": "earlier_measurement"}]

    def post(body):
        requests.append(deepcopy(body))
        if len(requests) in (1, 3):
            return tool_response("abandoned" if len(requests) == 1 else "fresh")
        if len(requests) == 4 and success:
            return {"output": []}
        raise InvalidEncryptedContentError("LLM HTTP status 400: invalid_encrypted_content")

    planner = EmbodiedPlanner(
        model="offline",
        post=post,
        invalid_encrypted_content_replans=1,
        wrist_to_center_m={"left": [0, 0, 0.15], "right": [0, 0, 0.15]},
    )
    planner._geometry_memory = deepcopy(old_memory)
    body = planner._request_body([{"type": "input_text", "text": "current observation and issued history"}])
    try:
        with capture_artifacts(lambda kind, value: artifacts.append((kind, deepcopy(value)))):
            if success:
                assert planner._decision_response(body, observation) == {"output": []}
            else:
                with pytest.raises(InvalidEncryptedContentError):
                    planner._decision_response(body, observation)
        assert len(requests) == 4
        assert requests[0] == requests[2]
        assert requests[1]["input"][-1]["call_id"] == "abandoned"
        assert requests[3]["input"][-1]["call_id"] == "fresh"
        assert body == planner._request_body(
            [{"type": "input_text", "text": "current observation and issued history"}]
        )
        assert observation == before
        assert planner._geometry_memory[:1] == old_memory
        assert len(planner._geometry_memory) == (2 if success else 1)
        recovery = [x for kind, x in artifacts if kind == "llm-planning-recovery"]
        assert recovery[0]["stage"] == "restart_current_planning"
        assert all(x["action_issued"] is False for x in recovery)
        if not success:
            assert recovery[-1]["stage"] == "exhausted"
        assert planner._recovery_deadline is None
        assert "encrypted_content_recovery_attempt" not in planner._call_context
    finally:
        planner.close()


def test_replanning_and_all_tool_http_calls_share_budget(monkeypatch, observation):
    clock = [100.0]
    calls = []
    monkeypatch.setattr("agent2act.adapters.planners.responses.time.monotonic", lambda: clock[0])

    def send(request, timeout):
        calls.append((json.loads(request.data), timeout))
        if len(calls) == 1:
            clock[0] += 2
            return io.BytesIO(json.dumps(tool_response("first")).encode())
        clock[0] += 2
        raise http_error(request)

    monkeypatch.setattr(urllib.request, "urlopen", send)
    planner = EmbodiedPlanner(
        model="offline",
        api_key_env=None,
        timeout_s=300,
        retry_budget_s=3,
        invalid_encrypted_content_replans=3,
        wrist_to_center_m={"left": [0, 0, 0.15], "right": [0, 0, 0.15]},
    )
    try:
        with pytest.raises(TransientBackendError, match="budget exhausted"):
            planner._decision_response(planner._request_body([]), observation)
        assert [timeout for _, timeout in calls] == [3, 1]
        assert planner._recovery_deadline is None
    finally:
        planner.close()
