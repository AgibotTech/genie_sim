# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""A bounded grant to a policy, containing intent and accounting, never dense actions."""

from dataclasses import dataclass
from typing import Any

from agent2act.domain.models import Decision


@dataclass
class PolicyRun:
    decision: Decision
    started_sequence: int
    issued_chunks: int = 0
    issued_steps: int = 0
    issued_duration_s: float = 0.0

    @property
    def exhausted(self) -> bool:
        return self.issued_chunks >= self.decision.policy_chunks

    def summary(self, reason: str) -> dict[str, Any]:
        return {
            "started_sequence": self.started_sequence,
            "backend": self.decision.backend,
            "instruction": self.decision.instruction,
            "success_condition": self.decision.success_condition,
            "budget_chunks": self.decision.policy_chunks,
            "issued_chunks": self.issued_chunks,
            "issued_steps": self.issued_steps,
            "issued_duration_s": round(self.issued_duration_s, 6),
            "reason": reason,
            "task_success": "unverified",
            "playback_status": "unverified; counts describe issued commands",
        }
