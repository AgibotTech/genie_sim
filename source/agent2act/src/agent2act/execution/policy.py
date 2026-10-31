# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""PolicyExecutor backend dispatch."""

from contextlib import ExitStack
from typing import Any

from agent2act.domain.errors import ContractError
from agent2act.domain.models import ActionChunk, Decision, Observation
from agent2act.domain.ports import PolicyBackend


class PolicyExecutor:
    mode = "policy"

    def __init__(self, backends: dict[str, PolicyBackend], default: str) -> None:
        if default not in backends:
            raise ContractError("policy default backend is not registered")
        self.backends, self.default = dict(backends), default

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "default": self.default,
            "backends": {k: b.describe() for k, b in self.backends.items()},
        }

    def reset(self) -> None:
        for b in self.backends.values():
            b.reset()

    @staticmethod
    def _capabilities(backend: PolicyBackend) -> dict[str, dict[str, Any]]:
        """Read the backend's optional training-coverage contract.

        Backends need not declare capabilities, but once declared every invocation must select one
        explicitly; a prose ``skills`` string from the LLM is not permission."""
        declared = backend.describe().get("capabilities", [])
        if not isinstance(declared, list):
            raise ContractError("policy backend capabilities must be a list")
        result: dict[str, dict[str, Any]] = {}
        for capability in declared:
            if not isinstance(capability, dict):
                raise ContractError("policy capability must be an object")
            required = {"name", "description", "instruction_scope", "max_policy_chunks"}
            if set(capability) != required:
                raise ContractError(f"policy capability requires exactly {sorted(required)}")
            name = capability["name"]
            if not isinstance(name, str) or not name.strip() or name in result:
                raise ContractError("policy capability names must be unique nonempty text")
            if not isinstance(capability["description"], str) or not capability["description"].strip():
                raise ContractError("policy capability description must be nonempty text")
            if capability["instruction_scope"] not in ("local_subtask", "original_task"):
                raise ContractError("policy capability instruction_scope is invalid")
            limit = capability["max_policy_chunks"]
            if type(limit) is not int or not 1 <= limit <= 32:
                raise ContractError("policy capability max_policy_chunks must be an integer in [1,32]")
            result[name] = capability
        return result

    def _check_capability(self, backend: PolicyBackend, observation: Observation, decision: Decision) -> None:
        capabilities = self._capabilities(backend)
        if not capabilities:
            if decision.policy_capability is not None:
                raise ContractError("selected policy backend does not declare policy capabilities")
            return
        if decision.policy_capability is None:
            raise ContractError("policy decision must select a declared policy_capability")
        capability = capabilities.get(decision.policy_capability)
        if capability is None:
            raise ContractError(f"unknown policy_capability: {decision.policy_capability}")
        if decision.policy_chunks > capability["max_policy_chunks"]:
            raise ContractError(
                f"policy_chunks exceeds selected capability limit {capability['max_policy_chunks']}"
            )
        if (
            capability["instruction_scope"] == "original_task"
            and decision.instruction != observation.instruction
        ):
            raise ContractError(
                "policy capability requires instruction_scope=original_task; "
                "instruction must exactly equal the environment task instruction"
            )

    def generate(self, observation: Observation, decision: Decision) -> ActionChunk:
        name = decision.backend or self.default
        if name not in self.backends:
            raise ContractError(f"unknown policy backend: {name}")
        self._check_capability(self.backends[name], observation, decision)
        chunk = self.backends[name].generate(observation, decision.instruction)
        chunk.metadata["backend"] = name
        return chunk

    def close(self) -> None:
        with ExitStack() as cleanup:
            for backend in self.backends.values():
                cleanup.callback(backend.close)
