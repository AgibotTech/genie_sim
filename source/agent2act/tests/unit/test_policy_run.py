# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Supervisor grants are bounded; every policy invocation sees fresh state."""

from copy import deepcopy

import pytest

from agent2act.application.routing import ExecutionRouter
from agent2act.application.runtime import AgentRuntime
from agent2act.application.settings import ExecutionConfig
from agent2act.configuration import parse_settings
from agent2act.domain.errors import ConfigurationError
from agent2act.domain.models import ActionChunk, ContractError, Decision
from agent2act.execution.motion import MotionCompiler
from agent2act.execution.policy import PolicyExecutor


class Supervisor:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.contexts = []

    def decide(self, observation, context):
        self.contexts.append(deepcopy(context))
        return next(self.decisions)

    def reset(self):
        pass

    def close(self):
        pass


class FreshPolicy:
    def __init__(self, spec):
        self.spec = spec
        self.calls = []
        self.fail = None

    def generate(self, observation, instruction):
        self.calls.append((deepcopy(observation), instruction))
        if self.fail:
            raise self.fail
        # Last row intentionally outside the prefix; it must never be cached/replayed.
        return ActionChunk(
            "JOINT_ABS",
            "base_link",
            self.spec.dt_s,
            [[observation.joints[0] + 0.005 * i] for i in range(1, 6)],
            [[observation.joints[1]] for _ in range(5)],
            [list(observation.grippers) for _ in range(5)],
        )

    def describe(self):
        return {"skills": ["test manipulation"]}

    def reset(self):
        pass

    def close(self):
        pass


def policy(chunks=3, backend=None):
    return Decision(
        "policy",
        "pick the test block",
        "supported local subtask",
        backend=backend,
        policy_chunks=chunks,
        success_condition="block visibly leaves table",
    )


def motion():
    return Decision(
        "motion",
        "hold for review",
        "inspect scene",
        motion={"frame": "base_link", "waypoints": [{"duration_s": 0.2}]},
    )


def runtime(spec, decisions, **options):
    planner, backend, events = Supervisor(decisions), FreshPolicy(spec), []
    settings = ExecutionConfig(execute_steps=4, policy_execute_steps=2, max_policy_chunks=3, **options)
    executor = PolicyExecutor({"local": backend}, "local")
    router = ExecutionRouter([MotionCompiler(spec), executor], settings)
    return AgentRuntime(spec, planner, router, events.append), planner, backend, events


def test_fresh_observations_stable_intent_budget_and_switch(spec, observation):
    agent, planner, backend, events = runtime(spec, [policy(), motion()], memory_size=1)
    try:
        for i in range(3):
            observation.joints[0] = i * 0.002  # Deliberately different from issued endpoint.
            observation.images = {"frame": i}
            chunk = agent.step(deepcopy(observation))
            assert chunk.steps == 2
            assert chunk.left_arm[0][0] == pytest.approx(i * 0.002 + 0.005)
            assert chunk.metadata["discarded_steps"] == 3
            assert chunk.metadata["policy_run"]["issued_chunks"] == i + 1
        assert len(planner.contexts) == 1
        assert [o.images["frame"] for o, _ in backend.calls] == [0, 1, 2]
        assert {text for _, text in backend.calls} == {"pick the test block"}
        assert agent._policy_run.decision.backend == "local"
        chunk = agent.step(observation)
        assert chunk.steps == 4 and chunk.kind == "EEF_ABS"
        review = planner.contexts[-1]["policy_review"]
        assert review["issued_chunks"] == 3 and review["issued_steps"] == 6
        assert review["issued_duration_s"] == pytest.approx(0.3)
        assert review["reason"] == "budget_exhausted"
        assert review["task_success"] == "unverified"
        assert "joint_error_rad" in planner.contexts[-1]["tracking_feedback"]
        assert len(backend.calls) == 3
        assert len([e for e in events if e["status"] == "policy_continued"]) == 2
        assert agent._policy_run is None
    finally:
        agent.close()


@pytest.mark.parametrize("change", ["reset", "done", "instruction", "episode"])
def test_grant_cannot_cross_task_boundaries(spec, observation, change):
    agent, planner, backend, _ = runtime(spec, [policy(), motion()])
    try:
        agent.step(observation)
        observation = deepcopy(observation)
        if change == "reset":
            agent.reset()
        elif change == "done":
            observation.done = True
        elif change == "instruction":
            observation.instruction = "different task"
        else:
            observation.episode_key = ("different", 2)
        agent.step(observation)
        assert agent._policy_run is None
        assert len(backend.calls) == 1
        assert len(planner.contexts) == (1 if change == "done" else 2)
    finally:
        agent.close()


def test_rejection_revokes_grant_and_replans_from_fresh_feedback(spec, observation):
    agent, planner, backend, _ = runtime(spec, [policy(), motion()])
    try:
        agent.step(observation)
        backend.fail = ContractError("unsupported constraint")
        chunk = agent.step(observation)
        assert chunk.kind == "EEF_ABS"
        assert agent._policy_run is None
        assert len(planner.contexts) == 2
        review = planner.contexts[-1]["policy_review"]
        assert review["issued_chunks"] == 1
        assert "unsupported constraint" in review["reason"]
    finally:
        agent.close()


def test_transport_failure_revokes_grant_without_silent_fallback(spec, observation):
    agent, planner, backend, _ = runtime(spec, [policy(), motion()])
    try:
        agent.step(observation)
        backend.fail = TimeoutError("offline timeout")
        with pytest.raises(TimeoutError):
            agent.step(observation)
        assert agent._policy_run is None
        assert len(planner.contexts) == 1
        agent.step(observation)
        assert "TimeoutError" in planner.contexts[-1]["policy_review"]["reason"]
        assert len(backend.calls) == 2
    finally:
        agent.close()


def test_full_policy_suffix_is_validated_before_prefix(spec, observation):
    agent, planner, backend, _ = runtime(spec, [policy(), motion()])
    original = backend.generate

    def invalid(obs, instruction):
        chunk = original(obs, instruction)
        chunk.left_arm[-1][0] += 0.5  # Inside joint bounds but over speed, outside issued prefix.
        return chunk

    backend.generate = invalid
    try:
        assert agent.step(observation).kind == "EEF_ABS"
        assert agent._policy_run is None
        assert "joint speed" in planner.contexts[-1]["policy_review"]["reason"]
    finally:
        agent.close()


@pytest.mark.parametrize(
    "mode,decisions",
    [
        ("motion", [policy(), motion()]),
        ("policy", [motion(), policy()]),
    ],
)
def test_fixed_mode_is_not_bypassed(spec, observation, mode, decisions):
    agent, planner, _, _ = runtime(spec, decisions, mode=mode)
    try:
        assert agent.step(observation).metadata["mode"] == mode
        assert len(planner.contexts) == 2
        assert "forbidden" in planner.contexts[-1]["recent_events"][-1]["error"]
    finally:
        agent.close()


def test_excess_budget_and_unknown_backend_are_rejected(spec, observation):
    agent, planner, backend, _ = runtime(spec, [policy(4), policy(3, "missing"), policy(1)])
    try:
        agent.step(observation)
        assert len(planner.contexts) == 3 and len(backend.calls) == 1
        assert any("max_policy_chunks" in e.get("error", "") for e in planner.contexts[1]["recent_events"])
        assert "unknown policy backend" in planner.contexts[-1]["policy_review"]["reason"]
    finally:
        agent.close()


@pytest.mark.parametrize("value", [0, 33, True, 1.5, "3", None])
def test_invalid_decision_budget(value):
    with pytest.raises(ContractError, match="policy_chunks"):
        Decision.from_dict(
            {"mode": "policy", "instruction": "pick", "reason": "test", "policy_chunks": value}
        )


def test_budget_only_applies_to_continuing_policy():
    for data in [vars(motion()) | {"policy_chunks": 2}, vars(policy()) | {"status": "complete"}]:
        with pytest.raises(ContractError, match="continuing policy"):
            Decision.from_dict(data)


def test_legacy_policy_decision_still_calls_supervisor_each_time(spec, observation):
    decision = Decision.from_dict({"mode": "policy", "instruction": "pick", "reason": "test"})
    agent, planner, backend, _ = runtime(spec, [decision, decision])
    try:
        agent.config.max_policy_chunks = 1
        agent.config.policy_execute_steps = None
        assert agent.step(observation).steps == 4  # Uses shared legacy prefix.
        assert agent.step(observation).steps == 4
        assert len(planner.contexts) == len(backend.calls) == 2
    finally:
        agent.close()


def test_explicit_renewal_starts_a_new_finite_budget(spec, observation):
    agent, planner, backend, _ = runtime(spec, [policy(2), policy(2), motion()])
    try:
        for _ in range(5):
            agent.step(observation)
        assert len(planner.contexts) == 3 and len(backend.calls) == 4
        assert [c["policy_review"]["issued_chunks"] for c in planner.contexts[1:]] == [2, 2]
        assert planner.contexts[1]["policy_review"]["started_sequence"] == 1
        assert planner.contexts[2]["policy_review"]["started_sequence"] == 3
    finally:
        agent.close()


def test_replanning_exhaustion_does_not_preserve_authorization(spec, observation):
    from agent2act.domain.errors import PlanningExhaustedError

    agent, planner, backend, _ = runtime(spec, [policy(), motion()], max_replans=0)
    try:
        agent.step(observation)
        backend.fail = ContractError("invalid action")
        with pytest.raises(PlanningExhaustedError):
            agent.step(observation)
        assert agent._policy_run is None
        agent.step(observation)
        assert len(planner.contexts) == 2
        assert planner.contexts[-1]["policy_review"]["issued_chunks"] == 1
    finally:
        agent.close()


@pytest.mark.parametrize(
    "option",
    [
        {"max_policy_chunks": 0},
        {"max_policy_chunks": True},
        {"max_policy_chunks": 33},
        {"policy_execute_steps": 0},
        {"policy_execute_steps": True},
        {"policy_execute_steps": 4097},
    ],
)
def test_invalid_config(option):
    with pytest.raises(ContractError):
        ExecutionConfig(**option)


def test_policy_prefix_must_fit_robot(config):
    config["execution"]["policy_execute_steps"] = 257
    with pytest.raises(ConfigurationError, match="policy_execute_steps"):
        parse_settings(config)
