# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""A declared policy training scope is a runtime constraint, not prompt advice."""

import pytest

from agent2act.domain.errors import ContractError
from agent2act.domain.models import ActionChunk, Decision
from agent2act.execution.policy import PolicyExecutor


class CapabilityPolicy:
    def __init__(self, capabilities):
        self.capabilities = capabilities
        self.calls = []

    def describe(self):
        return {"capabilities": self.capabilities}

    def generate(self, observation, instruction):
        self.calls.append(instruction)
        return ActionChunk("JOINT_ABS", "base_link", 0.05, [[0]], [[0]], [[0, 0]])

    def reset(self):
        pass

    def close(self):
        pass


def capability(name="full_task", scope="original_task", limit=2):
    return {
        "name": name,
        "description": "A deliberately narrow trained behavior.",
        "instruction_scope": scope,
        "max_policy_chunks": limit,
    }


def policy(**changes):
    data = {
        "mode": "policy",
        "instruction": "Put the blue block in the bowl",
        "reason": "The trained behavior owns this bounded operation.",
        "backend": "narrow",
        "policy_capability": "full_task",
        "policy_chunks": 1,
    }
    data.update(changes)
    return Decision.from_dict(data)


def test_declared_capability_requires_explicit_selection(observation):
    backend = CapabilityPolicy([capability()])
    executor = PolicyExecutor({"narrow": backend}, "narrow")

    with pytest.raises(ContractError, match="must select"):
        executor.generate(observation, policy(policy_capability=None))
    assert backend.calls == []


def test_original_task_capability_rejects_llm_prompt_rewrite(observation):
    backend = CapabilityPolicy([capability()])
    executor = PolicyExecutor({"narrow": backend}, "narrow")

    with pytest.raises(ContractError, match="exactly equal"):
        executor.generate(observation, policy(instruction="Pick up the blue block"))
    assert backend.calls == []


def test_capability_budget_is_stricter_than_global_router_budget(observation):
    backend = CapabilityPolicy([capability(limit=2)])
    executor = PolicyExecutor({"narrow": backend}, "narrow")

    with pytest.raises(ContractError, match="capability limit 2"):
        executor.generate(observation, policy(policy_chunks=3))
    assert backend.calls == []


def test_original_task_capability_forwards_verbatim_instruction(observation):
    observation.instruction = "Put the blue block in the bowl"
    backend = CapabilityPolicy([capability()])
    executor = PolicyExecutor({"narrow": backend}, "narrow")

    executor.generate(observation, policy())
    assert backend.calls == [observation.instruction]


def test_legacy_backend_remains_compatible_without_capability_declaration(observation):
    backend = CapabilityPolicy([])
    executor = PolicyExecutor({"narrow": backend}, "narrow")

    executor.generate(observation, policy(policy_capability=None))
    assert backend.calls == ["Put the blue block in the bowl"]
