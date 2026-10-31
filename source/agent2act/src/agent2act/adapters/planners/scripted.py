# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Deterministic fixture planner for diagnostics."""

from dataclasses import asdict

from agent2act.domain.models import ContractError, Decision


class ScriptedPlanner:
    """Deterministic fixture runner for protocol smoke tests; not an LLM substitute."""

    def __init__(self, decisions):
        self.decisions = [Decision.from_dict(d) for d in decisions]
        self.index = 0

    def reset(self):
        self.index = 0

    def decide(self, observation, context):
        if self.index >= len(self.decisions):
            raise ContractError("script exhausted; supply another decision or episode_done")
        decision = Decision.from_dict(asdict(self.decisions[self.index]))
        self.index += 1
        return decision

    def close(self) -> None:
        pass
