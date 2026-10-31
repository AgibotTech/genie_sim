# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Evidence-bearing model assessments, never environment success signals."""

import math
from dataclasses import dataclass
from typing import Any

from .errors import ContractError


@dataclass
class ArmCommandBounds:
    """Limits relative to measured wrist pose and the current gripper command."""

    translation_m: float
    rotation_rad: float
    gripper_change: float

    @classmethod
    def from_dict(cls, value: Any) -> "ArmCommandBounds":
        if not isinstance(value, dict):
            raise ContractError("arm command bounds must be an object")
        try:
            result = cls(**value)
        except TypeError as exc:
            raise ContractError(f"invalid arm command bounds: {exc}") from exc
        for key in ("translation_m", "rotation_rad", "gripper_change"):
            limit = getattr(result, key)
            if (
                isinstance(limit, bool)
                or not isinstance(limit, (int, float))
                or not math.isfinite(limit)
                or limit < 0
            ):
                raise ContractError(f"command bounds require finite nonnegative {key}")
        return result


@dataclass
class CommandBounds:
    """Command-space envelope; does not certify actual motion or contact."""

    frame: str
    left: ArmCommandBounds
    right: ArmCommandBounds

    @classmethod
    def from_dict(cls, value: Any) -> "CommandBounds":
        if not isinstance(value, dict):
            raise ContractError("command_bounds must be an object")
        try:
            result = cls(**value)
        except TypeError as exc:
            raise ContractError(f"invalid command_bounds: {exc}") from exc
        if result.frame not in ("base_link", "arm_base_link"):
            raise ContractError("unsupported command_bounds frame")
        result.left = ArmCommandBounds.from_dict(result.left)
        result.right = ArmCommandBounds.from_dict(result.right)
        return result


@dataclass
class ObservationRequest:
    """Model evidence for a view/probe action, not a verified safety certificate."""

    missing_evidence: str
    candidate_commitment: str
    expected_observation: str
    preserved_conditions: list[str]
    command_bounds: CommandBounds | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "ObservationRequest":
        if not isinstance(value, dict):
            raise ContractError("observation_request must be an object")
        try:
            result = cls(**value)
        except TypeError as exc:
            raise ContractError(f"invalid observation_request: {exc}") from exc
        if result.command_bounds is not None:
            result.command_bounds = CommandBounds.from_dict(result.command_bounds)
        for key in ("missing_evidence", "candidate_commitment", "expected_observation"):
            text = getattr(result, key)
            if not isinstance(text, str) or not text.strip():
                raise ContractError(f"observation_request requires evidence: {key}")
        if (
            not isinstance(result.preserved_conditions, list)
            or not result.preserved_conditions
            or any(not isinstance(v, str) or not v.strip() for v in result.preserved_conditions)
        ):
            raise ContractError("observation_request requires nonempty preserved_conditions")
        return result


@dataclass
class MotionAlternative:
    """Model justification for a shorter geometric action, not proof of equivalence."""

    expected_effect: str
    geometry_evidence: str
    path_evidence: str
    omitted_effects: str
    preserved_conditions: list[str]

    @classmethod
    def from_dict(cls, value: Any) -> "MotionAlternative":
        if not isinstance(value, dict):
            raise ContractError("motion_alternative must be an object")
        try:
            result = cls(**value)
        except TypeError as exc:
            raise ContractError(f"invalid motion_alternative: {exc}") from exc
        for key in ("expected_effect", "geometry_evidence", "path_evidence", "omitted_effects"):
            text = getattr(result, key)
            if not isinstance(text, str) or not text.strip():
                raise ContractError(f"motion_alternative requires evidence: {key}")
        if (
            not isinstance(result.preserved_conditions, list)
            or not result.preserved_conditions
            or any(not isinstance(v, str) or not v.strip() for v in result.preserved_conditions)
        ):
            raise ContractError("motion_alternative requires nonempty preserved_conditions")
        return result


@dataclass
class ReviewAssessment:
    execution_status: str
    execution_evidence: str
    intent_status: str
    intent_evidence: str
    current_subgoal: str
    expected_next_intent: str
    predicted_next_intent: str
    verified_completed: list[str]
    remaining: list[str]
    observation_request: ObservationRequest | None = None
    motion_alternative: MotionAlternative | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewAssessment":
        if not isinstance(value, dict):
            raise ContractError("review assessment must be an object")
        try:
            result = cls(**value)
        except TypeError as exc:
            raise ContractError(f"invalid review assessment: {exc}") from exc
        if result.execution_status not in ("not_started", "progressing", "failed", "uncertain", "recovered"):
            raise ContractError("invalid execution_status")
        if result.intent_status not in ("aligned", "misaligned", "uncertain"):
            raise ContractError("invalid intent_status")
        if result.observation_request is not None:
            result.observation_request = ObservationRequest.from_dict(result.observation_request)
            if result.intent_status != "uncertain":
                raise ContractError("observation_request requires uncertain intent")
        if result.motion_alternative is not None:
            result.motion_alternative = MotionAlternative.from_dict(result.motion_alternative)
            if result.intent_status != "aligned" or result.observation_request is not None:
                raise ContractError("motion_alternative requires aligned intent without observation_request")
        for key in (
            "execution_evidence",
            "intent_evidence",
            "current_subgoal",
            "expected_next_intent",
            "predicted_next_intent",
        ):
            text = getattr(result, key)
            if not isinstance(text, str) or not text.strip():
                raise ContractError(f"assessment requires evidence: {key}")
        for items in (result.verified_completed, result.remaining):
            if not isinstance(items, list) or any(not isinstance(v, str) or not v.strip() for v in items):
                raise ContractError("review progress must contain nonempty strings")
        return result

    @property
    def permits_correction(self) -> bool:
        return (
            self.execution_status == "failed"
            or self.intent_status == "misaligned"
            or self.observation_request is not None
            or self.motion_alternative is not None
        )
