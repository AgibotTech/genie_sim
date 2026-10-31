# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Dependency contracts implemented by adapters; no concrete backend imports."""

from typing import Any, Protocol

from .models import ActionChunk, Decision, Observation


class ManagedComponent(Protocol):
    def reset(self) -> None: ...
    def close(self) -> None: ...


class Planner(ManagedComponent, Protocol):
    def decide(self, observation: Observation, context: dict[str, Any]) -> Decision: ...


class ActionExecutor(ManagedComponent, Protocol):
    mode: str

    def describe(self) -> dict[str, Any]: ...
    def generate(self, observation: Observation, decision: Decision) -> ActionChunk: ...


class PolicyBackend(ManagedComponent, Protocol):
    def describe(self) -> dict[str, Any]: ...
    def generate(self, observation: Observation, instruction: str) -> ActionChunk: ...


class Kinematics(ManagedComponent, Protocol):
    """Robot-only previews. Optional command_envelope covers every chunk step.

    The envelope reports frame, steps, and left/right translation_m and rotation_rad maxima relative
    to this observation's measured EEF poses. executable_command_envelope covers only
    min(steps, chunk.steps) for candidate review, with the same reference."""

    def preview(self, observation: Observation, chunk: ActionChunk, steps: int) -> dict[str, Any]: ...


class EventSink(Protocol):
    def __call__(self, event: dict[str, Any]) -> None: ...
