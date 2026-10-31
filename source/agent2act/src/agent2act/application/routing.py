# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Enforce mode selection and dispatch through executor ports."""

from collections.abc import Iterable
from contextlib import ExitStack
from typing import Any

from agent2act.domain.models import ActionChunk, ContractError, Decision, Observation
from agent2act.domain.ports import ActionExecutor

from .settings import ExecutionConfig


class ExecutionRouter:
    def __init__(self, executors: Iterable[ActionExecutor], config: ExecutionConfig) -> None:
        self.executors: dict[str, ActionExecutor] = {}
        self.config = config
        for executor in executors:
            if executor.mode in self.executors:
                raise ContractError(f"duplicate executor {executor.mode}")
            self.executors[executor.mode] = executor
        required = config.allowed_modes if config.mode == "auto" else [config.mode]
        if any(m not in self.executors for m in required):
            raise ContractError("all enabled modes must have an installed executor/backend")

    def describe(self) -> dict[str, Any]:
        allowed = self.config.allowed_modes if self.config.mode == "auto" else [self.config.mode]
        return {
            "mode": self.config.mode,
            "allowed_modes": allowed,
            "execute_steps": self.config.execute_steps,
            "policy_execute_steps": self.step_limit("policy"),
            "body_execute_steps": self.config.body_execute_steps,
            "complete_hold_steps": self.config.complete_hold_steps,
            "max_policy_chunks": self.config.max_policy_chunks,
            "proposal_review": self.config.proposal_review,
            "policy_execution": (
                "policy_chunks includes the first chunk; reuse the same instruction/backend, "
                "infer fresh actions from each new observation, then return to planner for review. "
                "Budget exhaustion is not success; discarded action suffixes are never replayed."
            ),
            "motion_frame": self.config.motion_frame,
            "motion_frame_axes": self.config.motion_frame_axes,
            "multi_waypoint_truncation": "issue execute_steps prefix; discard remainder and replan",
            "max_motion_continuations": self.config.max_motion_continuations,
            "executors": {m: self.executors[m].describe() for m in allowed},
        }

    def check(self, decision: Decision) -> None:
        allowed = self.describe()["allowed_modes"]
        if decision.mode not in allowed:
            raise ContractError(f"selected mode {decision.mode} is forbidden; allowed={allowed}")
        if decision.policy_chunks > self.config.max_policy_chunks:
            raise ContractError(f"policy_chunks exceeds max_policy_chunks={self.config.max_policy_chunks}")
        if (
            decision.mode == "motion"
            and decision.motion is not None
            and self.config.motion_frame is not None
            and decision.motion.get("frame") != self.config.motion_frame
        ):
            raise ContractError(
                f"motion requires frame={self.config.motion_frame}; use observed poses in that frame, "
                "do not relabel coordinates from another frame"
            )

    def step_limit(self, mode: str) -> int:
        if mode == "policy" and self.config.policy_execute_steps is not None:
            return self.config.policy_execute_steps
        return self.config.execute_steps

    def generate(self, observation: Observation, decision: Decision) -> ActionChunk:
        self.check(decision)
        chunk = self.executors[decision.mode].generate(observation, decision)
        chunk.metadata["mode"] = decision.mode
        return chunk

    def reset(self) -> None:
        for executor in self.executors.values():
            executor.reset()

    def close(self) -> None:
        with ExitStack() as cleanup:
            for executor in self.executors.values():
                cleanup.callback(executor.close)
