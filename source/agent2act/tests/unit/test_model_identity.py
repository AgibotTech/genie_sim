# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Identity telemetry never silently substitutes models or changes actions."""

import json
import logging
from copy import deepcopy

import msgpack
import pytest

from agent2act.adapters.planners.chat_completions import ChatCompletionsPlanner
from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.adapters.telemetry import capture_artifacts
from agent2act.bootstrap import ApplicationFactory, default_registry
from agent2act.configuration import parse_settings
from agent2act.domain.errors import ContractError, ModelIdentityError
from agent2act.service.server import AgentServer


@pytest.mark.parametrize("planner_type", [ResponsesPlanner, ChatCompletionsPlanner])
def test_model_identity_is_per_response_and_not_model_input(
    planner_type, observation, motion_decision, caplog
):
    calls, artifacts = [], []
    reports = iter(["requested", "other", None, "requested"])

    def post(body):
        calls.append(body)
        result = {"model": next(reports)}
        text = json.dumps(motion_decision)
        if planner_type is ChatCompletionsPlanner:
            result["choices"] = [{"finish_reason": "stop", "message": {"content": text}}]
        else:
            result["output"] = [{"type": "message", "content": [{"type": "output_text", "text": text}]}]
        return result

    planner = planner_type(model="requested", post=post, current_observation_only=True, history_frames=0)
    with (
        caplog.at_level(logging.INFO),
        capture_artifacts(lambda kind, value: artifacts.append((kind, value))),
    ):
        try:
            for sequence in range(1, 5):
                if sequence == 4:
                    planner.reset()
                result = planner.decide(observation, {"sequence": sequence, "execution": {"mode": "motion"}})
                assert result.instruction == motion_decision["instruction"]
        finally:
            planner.close()
    identities = [v for k, v in artifacts if k == "model-identity"]
    assert [v["model_identity_status"] for v in identities] == ["match", "mismatch", "unreported", "match"]
    assert [v["fixed_model_comparison_eligible"] for v in identities] == [True, False, False, True]
    assert [v["sequence"] for v in identities] == [1, 2, 3, 4]
    events = [r for r in caplog.records if r.message == "llm_model_identity"]
    assert [r.levelno for r in events] == [logging.INFO, logging.WARNING, logging.WARNING, logging.INFO]
    assert all(body["model"] == "requested" for body in calls)
    assert all("model_identity_status" not in json.dumps(body) for body in calls)
    assert len(calls) == 4


@pytest.mark.parametrize("reports", [["other", "requested"], [None, "other", "requested"]])
@pytest.mark.parametrize("planner_type", [ResponsesPlanner, ChatCompletionsPlanner])
def test_strict_identity_retries_identical_input_before_parsing(
    planner_type, reports, observation, motion_decision
):
    calls, artifacts = [], []

    def post(body):
        calls.append(body)
        model = reports[len(calls) - 1]
        text = json.dumps(motion_decision) if model == "requested" else "not a decision"
        if planner_type is ChatCompletionsPlanner:
            return {"model": model, "choices": [{"finish_reason": "stop", "message": {"content": text}}]}
        return {
            "model": model,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        }

    planner = planner_type(model="requested", post=post, model_identity_policy="require_match")
    try:
        with capture_artifacts(lambda k, v: artifacts.append((k, v))):
            result = planner.decide(observation, {"sequence": 1, "execution": {"mode": "motion"}})
        assert result.instruction == motion_decision["instruction"]
        assert len(calls) == len(reports) and all(body == calls[0] for body in calls)
        identities = [v for k, v in artifacts if k == "model-identity"]
        assert [i["model_identity_attempt"] for i in identities] == list(range(1, len(reports) + 1))
        assert len([k for k, _ in artifacts if k == "model_identity_retry"]) == len(reports) - 1
        assert not any(k == "model_identity_exhausted" for k, _ in artifacts)
    finally:
        planner.close()


@pytest.mark.parametrize("reports", ["other", None])
def test_exhausted_identity_is_backend_error_not_action_replan(reports, observation):
    calls = []
    planner = ResponsesPlanner(
        model="requested",
        model_identity_policy="require_match",
        post=lambda body: calls.append(body) or {"model": reports},
    )
    try:
        with pytest.raises(ModelIdentityError) as info:
            planner.decide(observation, {})
        assert not isinstance(info.value, ContractError)
        assert len(calls) == 3
        assert planner._identity_deadline is None
    finally:
        planner.close()


def test_identity_retries_share_http_time_budget(monkeypatch, observation, motion_decision):
    clock = [100.0]
    monkeypatch.setattr("agent2act.adapters.planners.responses.time.monotonic", lambda: clock[0])
    planner = ResponsesPlanner(
        model="requested",
        api_key_env=None,
        timeout_s=20,
        retry_budget_s=10,
        model_identity_policy="require_match",
    )
    timeouts = []

    def send(request, event):
        timeouts.append(event["timeout_s"])
        clock[0] += 6 if len(timeouts) == 1 else 1
        return {
            "model": "other" if len(timeouts) == 1 else "requested",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(motion_decision)}]}
            ],
        }

    monkeypatch.setattr(planner, "_send_http", send)
    try:
        planner.decide(observation, {})
        assert timeouts == [10, 4]
    finally:
        planner.close()


@pytest.mark.parametrize(
    "options",
    [
        {"model_identity_policy": "fallback"},
        {"model_identity_max_retries": True},
        {"model_identity_max_retries": -1},
        {"model_identity_max_retries": 6},
    ],
)
def test_invalid_identity_options(options):
    with pytest.raises(ContractError):
        ResponsesPlanner(model="test", **options)


def test_tool_planner_rejects_strict_mode_instead_of_reusing_unverified_tool_history():
    with pytest.raises(ContractError, match="without geometry tool rounds"):
        EmbodiedPlanner(model="test", model_identity_policy="require_match")


def test_identity_budget_stops_before_another_request(monkeypatch, observation):
    clock, calls = [100.0], []
    monkeypatch.setattr("agent2act.adapters.planners.responses.time.monotonic", lambda: clock[0])

    def post(body):
        calls.append(body)
        clock[0] += 6
        return {"model": "wrong"}

    planner = ResponsesPlanner(
        model="test", post=post, model_identity_policy="require_match", retry_budget_s=5
    )
    try:
        with pytest.raises(ModelIdentityError):
            planner.decide(observation, {})
        assert len(calls) == 1
    finally:
        planner.close()


def test_identity_exhaustion_interrupts_episode_without_actions_or_replanning(
    config, request_payload, tmp_path
):
    calls = []
    returned_model = ["wrong"]
    decision = config["planner"]["decisions"][0]

    def post(body):
        calls.append(body)
        return {
            "model": returned_model[0],
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(decision)}]}
            ],
        }

    registry = default_registry()
    registry.planners.register(
        "offline_identity",
        lambda options: ResponsesPlanner(
            model="test",
            post=post,
            model_identity_policy="require_match",
        ),
    )
    config["planner"] = {"type": "offline_identity"}
    config["execution"]["max_replans"] = 4
    config["logging"] = {"trace_dir": str(tmp_path / "run.trace")}
    # _dispatch has no socket dependency: exercise wire error mapping without opening a port.
    server = object.__new__(AgentServer)
    with ApplicationFactory(parse_settings(config), registry).create_session() as session:
        reply = server._dispatch(session, msgpack.packb(request_payload, use_bin_type=True))
        assert reply["error_code"] == "model_identity_exhausted" and "result" not in reply
        assert len(calls) == 3
        assert not list(tmp_path.rglob("*-simulator-response.json"))
        assert list(tmp_path.rglob("*-episode-interrupted.json"))
        returned_model[0] = "test"
        reply = server._dispatch(session, msgpack.packb(request_payload, use_bin_type=True))
        assert reply["error_code"] == "model_identity_exhausted" and len(calls) == 3
        fresh = deepcopy(request_payload)
        fresh["params"]["episode_idx"] += 1
        reply = server._dispatch(session, msgpack.packb(fresh, use_bin_type=True))
        assert "result" in reply and len(calls) == 4
    events = [json.loads(line) for line in (tmp_path / "run.action-events.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["kind"] == "episode-interrupted"]) == 1
    assert len([e for e in events if e["data"].get("status") == "issued"]) == 1
