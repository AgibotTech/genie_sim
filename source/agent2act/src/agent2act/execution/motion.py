# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Deterministic motion-intent compiler.

The planner chooses semantic Cartesian waypoints; this compiler converts them to dense simulator
actions, so the planner never emits raw control sequences or per-step speed limits."""

from typing import Any

from agent2act.domain.errors import ContractError
from agent2act.domain.models import ActionChunk, Decision, Observation, RobotSpec
from agent2act.domain.timing import MotionTiming

from .interpolation import LocalInterpolation


class MotionCompiler:
    mode = "motion"

    def __init__(self, spec: RobotSpec, timing: MotionTiming | None = None) -> None:
        self.backend = LocalInterpolation(spec, timing)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "input": "motion intent with one or more Cartesian waypoints",
            "compiler": self.backend.describe(),
        }

    def reset(self) -> None:
        self.backend.reset()

    def generate(self, observation: Observation, decision: Decision) -> ActionChunk:
        if decision.motion is None:
            raise ContractError("missing motion intent")
        chunk = self.backend.generate(observation, decision.motion)
        chunk.metadata["compiler"] = "local_interpolation"
        return chunk

    def close(self) -> None:
        self.backend.close()
