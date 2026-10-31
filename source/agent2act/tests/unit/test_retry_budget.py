# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import io
import json
import urllib.error
import urllib.request
from email.message import Message

import pytest

from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.domain.errors import ContractError, TransientBackendError


@pytest.mark.parametrize("value", [True, False, 0, -1, float("nan"), float("inf"), "1200"])
def test_invalid_retry_budget(value):
    with pytest.raises(ContractError, match="retry_budget_s"):
        ResponsesPlanner(model="offline", retry_budget_s=value)


def setup_clock(monkeypatch, planner):
    clock = [100.0]
    monkeypatch.setattr("agent2act.adapters.planners.responses.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("agent2act.adapters.planners.responses.random.uniform", lambda a, b: 1.0)

    def wait(delay):
        clock[0] += delay
        return False

    monkeypatch.setattr(planner._closed, "wait", wait)
    return clock


def test_extended_rate_limit_retries_reach_ninth_attempt(monkeypatch):
    planner = ResponsesPlanner(
        model="offline",
        api_key_env=None,
        max_retries=8,
        retry_backoff_s=1,
        retry_max_delay_s=30,
    )
    clock = setup_clock(monkeypatch, planner)
    calls = []

    def send(request, timeout):
        calls.append(request.data)
        if len(calls) <= 8:
            raise urllib.error.HTTPError(request.full_url, 429, "limited", Message(), io.BytesIO(b"{}"))
        return io.BytesIO(b'{"output":[]}')

    monkeypatch.setattr(urllib.request, "urlopen", send)
    assert planner._post({"input": []}) == {"output": []}
    planner.close()
    assert len(calls) == 9
    assert all(data == calls[0] for data in calls)
    assert clock[0] == 221


def test_retry_count_above_ninety_nine_is_rejected():
    for count in (30, 31, 99):
        ResponsesPlanner(model="offline", max_retries=count).close()
    with pytest.raises(ContractError, match="retry settings"):
        ResponsesPlanner(model="offline", max_retries=100)


@pytest.mark.parametrize("last_attempt_succeeds", [True, False])
def test_hundred_gateway_attempts_preserve_request_and_budget(monkeypatch, last_attempt_succeeds):
    planner = ResponsesPlanner(
        model="offline",
        api_key_env=None,
        max_retries=99,
        retry_max_delay_s=10,
        retry_budget_s=1800,
    )
    clock = setup_clock(monkeypatch, planner)
    calls = []

    def send(request, timeout):
        calls.append(request.data)
        clock[0] += 2
        if len(calls) == 100 and last_attempt_succeeds:
            return io.BytesIO(b'{"output":[]}')
        raise urllib.error.HTTPError(request.full_url, 500, "gateway", Message(), io.BytesIO(b"{}"))

    monkeypatch.setattr(urllib.request, "urlopen", send)
    try:
        if last_attempt_succeeds:
            assert planner._post({"input": []}) == {"output": []}
        else:
            with pytest.raises(TransientBackendError, match="500"):
                planner._post({"input": []})
    finally:
        planner.close()
    assert len(calls) == 100
    assert all(data == calls[0] for data in calls)
    assert clock[0] == 1265


def test_fast_rate_limits_can_retry_within_shared_budget(monkeypatch):
    planner = ResponsesPlanner(
        model="offline",
        api_key_env=None,
        timeout_s=600,
        max_retries=5,
        retry_backoff_s=60,
        retry_max_delay_s=60,
        retry_budget_s=1200,
    )
    clock = setup_clock(monkeypatch, planner)
    calls = []

    def send(request, timeout):
        calls.append((request.data, timeout))
        clock[0] += 50
        if len(calls) < 4:
            raise urllib.error.HTTPError(request.full_url, 429, "limited", Message(), io.BytesIO(b"{}"))
        return io.BytesIO(b'{"output":[]}')

    monkeypatch.setattr(urllib.request, "urlopen", send)
    assert planner._post({"input": [{"encrypted_content": "opaque"}]}) == {"output": []}
    planner.close()
    assert len(calls) == 4
    assert all(data == calls[0][0] for data, _ in calls)
    assert all(timeout == 600 for _, timeout in calls)
    assert clock[0] == 480


def test_timeouts_share_budget_and_shorten_last_attempt(monkeypatch):
    planner = ResponsesPlanner(
        model="offline",
        api_key_env=None,
        timeout_s=600,
        max_retries=5,
        retry_backoff_s=60,
        retry_max_delay_s=60,
        retry_budget_s=1200,
    )
    clock = setup_clock(monkeypatch, planner)
    timeouts = []

    def send(request, timeout):
        timeouts.append(timeout)
        clock[0] += timeout
        raise TimeoutError("offline timeout")

    monkeypatch.setattr(urllib.request, "urlopen", send)
    with pytest.raises(TransientBackendError):
        planner._post({})
    planner.close()
    assert timeouts == [600, 540]
    assert clock[0] == 1300


def test_budget_rejects_backoff_that_cannot_fit(monkeypatch):
    planner = ResponsesPlanner(
        model="offline",
        api_key_env=None,
        max_retries=5,
        retry_backoff_s=60,
        retry_max_delay_s=60,
        retry_budget_s=60,
    )
    clock = setup_clock(monkeypatch, planner)
    calls = []

    def send(request, timeout):
        calls.append(timeout)
        clock[0] += 10
        raise urllib.error.HTTPError(request.full_url, 429, "limited", Message(), io.BytesIO(b"{}"))

    monkeypatch.setattr(urllib.request, "urlopen", send)
    with pytest.raises(TransientBackendError, match="429"):
        planner._post({})
    planner.close()
    assert calls == [60] and clock[0] == 110


def test_overslept_backoff_never_starts_another_attempt(monkeypatch):
    planner = ResponsesPlanner(model="offline", api_key_env=None, max_retries=5, retry_budget_s=10)
    clock = setup_clock(monkeypatch, planner)
    calls = []

    def send(request, timeout):
        calls.append(timeout)
        raise TimeoutError("offline timeout")

    def wait(delay):
        clock[0] += 11
        return False

    monkeypatch.setattr(urllib.request, "urlopen", send)
    monkeypatch.setattr(planner._closed, "wait", wait)
    with pytest.raises(TransientBackendError, match="budget exhausted"):
        planner._post({})
    planner.close()
    assert calls == [10]


def test_budget_does_not_retry_permanent_errors(monkeypatch):
    planner = ResponsesPlanner(model="offline", api_key_env=None, max_retries=5, retry_budget_s=1200)
    setup_clock(monkeypatch, planner)
    calls = []

    def send(request, timeout):
        calls.append(json.loads(request.data))
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", Message(), io.BytesIO(b"{}"))

    monkeypatch.setattr(urllib.request, "urlopen", send)
    with pytest.raises(RuntimeError, match="401") as info:
        planner._post({"input": []})
    planner.close()
    assert not isinstance(info.value, TransientBackendError)
    assert len(calls) == 1


def test_gateway_origin_errors_are_retryable():
    from agent2act.adapters.planners.responses import _RETRYABLE_HTTP

    assert {400, 502, 520, 530} <= _RETRYABLE_HTTP
    assert not {401, 403, 404} & _RETRYABLE_HTTP
