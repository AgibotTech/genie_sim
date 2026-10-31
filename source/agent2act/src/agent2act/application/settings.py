# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Execution semantics independent of deployment and transport."""

from dataclasses import dataclass, field
from typing import Any

from agent2act.domain.models import MODES, ContractError, number


@dataclass
class ExecutionConfig:
    mode: str = "auto"
    allowed_modes: list[str] = field(default_factory=lambda: list(MODES))
    execute_steps: int = 8
    policy_execute_steps: int | None = None
    # Longer issued prefix for chunks that move a body joint (e.g. a waist turn).
    body_execute_steps: int | None = None
    # Hold rows issued when the planner reports completion, so a simulator that
    # checks progress every N steps can score the episode without idle replans.
    complete_hold_steps: int | None = None
    max_policy_chunks: int = 1
    proposal_review: bool = False
    proposal_backend: str | None = None
    proposal_capability: str | None = None
    proposal_max_age_s: float = 300.0
    max_replans: int = 2
    terminal_on_planning_exhaustion: bool = False
    retry_ik_exhaustion: bool = False
    memory_size: int = 12
    need_depth: bool = False
    motion_frame: str | None = None
    motion_frame_axes: list[str] | None = None
    max_motion_continuations: int = 0
    continuation_translation_tolerance_m: float = 0.01
    continuation_rotation_tolerance_rad: float = 0.1
    continuation_gripper_tolerance: float = 0.05
    history: dict[str, Any] = field(
        default_factory=lambda: {"interval": 2, "last_n": 2, "resolution": {"head": [320, 240]}}
    )

    def __post_init__(self) -> None:
        if type(self.proposal_review) is not bool:
            raise ContractError("proposal_review must be boolean")
        if type(self.terminal_on_planning_exhaustion) is not bool:
            raise ContractError("terminal_on_planning_exhaustion must be boolean")
        if type(self.retry_ik_exhaustion) is not bool:
            raise ContractError("retry_ik_exhaustion must be boolean")
        if number(self.proposal_max_age_s, "proposal_max_age_s") <= 0:
            raise ContractError("proposal_max_age_s must be positive")
        for value in (self.proposal_backend, self.proposal_capability):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ContractError("proposal backend/capability must be nonempty text")
        if self.proposal_review and (not self.proposal_backend or not self.proposal_capability):
            raise ContractError("proposal review requires an explicit backend and capability")
        if self.proposal_review and self.max_motion_continuations:
            raise ContractError("proposal review does not support automatic motion continuation")
        if type(self.max_policy_chunks) is not int or not 1 <= self.max_policy_chunks <= 32:
            raise ContractError("max_policy_chunks must be an integer in [1,32]")
        if self.policy_execute_steps is not None and (
            type(self.policy_execute_steps) is not int or not 1 <= self.policy_execute_steps <= 4096
        ):
            raise ContractError("policy_execute_steps must be an integer in [1,4096] or null")
        if self.body_execute_steps is not None and (
            type(self.body_execute_steps) is not int or not 1 <= self.body_execute_steps <= 4096
        ):
            raise ContractError("body_execute_steps must be an integer in [1,4096] or null")
        if self.complete_hold_steps is not None and (
            type(self.complete_hold_steps) is not int or not 1 <= self.complete_hold_steps <= 4096
        ):
            raise ContractError("complete_hold_steps must be an integer in [1,4096] or null")
        if self.motion_frame not in (None, "base_link", "arm_base_link"):
            raise ContractError("invalid motion_frame")
        if self.motion_frame_axes is not None and (
            self.motion_frame is None
            or not isinstance(self.motion_frame_axes, list)
            or len(self.motion_frame_axes) != 3
            or any(not isinstance(axis, str) or not axis.strip() for axis in self.motion_frame_axes)
        ):
            raise ContractError("motion_frame_axes requires motion_frame and three axis descriptions")
        if self.mode not in ("auto", *MODES):
            raise ContractError("invalid execution mode")
        if (
            not isinstance(self.allowed_modes, list)
            or not self.allowed_modes
            or any(m not in MODES for m in self.allowed_modes)
        ):
            raise ContractError("invalid allowed_modes")
        if len(set(self.allowed_modes)) != len(self.allowed_modes):
            raise ContractError("allowed_modes must be unique")
        if self.mode != "auto" and self.mode not in self.allowed_modes:
            raise ContractError("fixed mode must appear in allowed_modes")
        for name, low, high in (("execute_steps", 1, 4096), ("max_replans", 0, 10), ("memory_size", 1, 100)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ContractError(f"invalid {name}")
        if type(self.max_motion_continuations) is not int or not 0 <= self.max_motion_continuations <= 4:
            raise ContractError("max_motion_continuations must be an integer in [0,4]")
        for name in (
            "continuation_translation_tolerance_m",
            "continuation_rotation_tolerance_rad",
            "continuation_gripper_tolerance",
        ):
            if number(getattr(self, name), name) <= 0:
                raise ContractError(f"{name} must be positive")
        if type(self.need_depth) is not bool or not isinstance(self.history, dict):
            raise ContractError("invalid observation request settings")

        if set(self.history) - {"interval", "last_n", "resolution"}:
            raise ContractError("unknown history capture setting")
        for key in ("interval", "last_n"):
            value = self.history.get(key, 0)
            if type(value) is not int or value < 0:
                raise ContractError(f"history.{key} must be a nonnegative integer")
        resolution = self.history.get("resolution", {})
        if not isinstance(resolution, dict):
            raise ContractError("history.resolution must be an object")
        for camera, size in resolution.items():
            if camera not in ("head", "hand_left", "hand_right"):
                raise ContractError(f"unsupported history camera: {camera}")
            if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
                raise ContractError("history resolution must be [positive width, positive height]")
