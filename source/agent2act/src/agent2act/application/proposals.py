# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""A single-use, observation-bound candidate. No execution or model adapters."""

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from agent2act.domain.errors import ExpiredProposalError
from agent2act.domain.models import ActionChunk, ContractError, Decision, Observation


def fingerprint(obs: Observation) -> str:
    # Images contain bytes; bind their content without logging or copying payloads into context.
    value = asdict(obs)
    encoded = json.dumps(
        value, sort_keys=True, default=lambda v: hashlib.sha256(v).hexdigest(), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


class PolicyProposal:
    def __init__(
        self,
        obs: Observation,
        sequence: int,
        decision: Decision,
        chunk: ActionChunk,
        preview: dict[str, Any],
        execute_steps: int,
        max_age_s: float,
    ) -> None:
        self.id = uuid4().hex
        self._fingerprint = fingerprint(obs)
        self.sequence = sequence
        self.decision = deepcopy(decision)
        self._chunk = deepcopy(chunk)
        self._preview = deepcopy(preview)
        self.execute_steps = min(execute_steps, chunk.steps)
        self.created = time.monotonic()
        self.max_age_s = max_age_s
        self.consumed = False

    @property
    def executable_duration_s(self) -> float:
        """Duration of the reviewed prefix, not the generated suffix or observed playback."""
        return self.execute_steps * self._chunk.dt_s

    def describe(self) -> dict[str, Any]:
        return {
            "proposal_id": self.id,
            "sequence": self.sequence,
            "backend": self.decision.backend,
            "policy_capability": self.decision.policy_capability,
            "instruction": self.decision.instruction,
            "generated_steps": self._chunk.steps,
            "execute_steps": self.execute_steps,
            "executable_duration_s": self.executable_duration_s,
            "retiming": deepcopy(self._chunk.metadata.get("retiming")),
            "conditioning": deepcopy(self._chunk.metadata.get("conditioning")),
            **deepcopy(self._preview),
        }

    def check_current(self, obs: Observation, sequence: int) -> None:
        if self.consumed or sequence != self.sequence or fingerprint(obs) != self._fingerprint:
            raise ContractError("proposal already consumed or observation changed")
        if time.monotonic() - self.created > self.max_age_s:
            raise ExpiredProposalError("proposal expired during review; fresh observation required")

    def consume(self, obs: Observation, sequence: int, decision: Decision) -> ActionChunk:
        self.check_current(obs, sequence)
        if decision.proposal_id != self.id or any(
            getattr(decision, key) != getattr(self.decision, key)
            for key in ("backend", "policy_capability", "instruction")
        ):
            raise ContractError("accept must identify the exact reviewed proposal and its instruction")
        self.consumed = True
        chunk = deepcopy(self._chunk)
        chunk.metadata.update(proposal_id=self.id, proposal_age_s=time.monotonic() - self.created)
        return chunk
