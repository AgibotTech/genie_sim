# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Version-one internal contracts. All poses use metres and xyzw quaternions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import ContractError
from .review import ReviewAssessment

MODES = ("motion", "policy")
FRAMES = ("base_link", "arm_base_link")


def number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ContractError(f"{name} must be a finite number")
    return float(value)


def vector(value: Any, size: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ContractError(f"{name} must have {size} elements")
    return [number(v, name) for v in value]


def pose(value: Any, name: str = "pose") -> list[float]:
    result = vector(value, 7, name)
    norm = math.sqrt(sum(v * v for v in result[3:]))
    if abs(norm - 1.0) > 1e-3:
        raise ContractError(f"{name} quaternion must be unit length (xyzw)")
    return result


def normalized_pose(value: Any, name: str = "pose") -> list[float]:
    """Parse a pose and normalize its quaternion at the motion boundary."""
    result = vector(value, 7, name)
    norm = math.sqrt(sum(v * v for v in result[3:]))
    if norm <= 1e-12:
        raise ContractError(f"{name} quaternion must be nonzero (xyzw)")
    return result[:3] + [v / norm for v in result[3:]]


@dataclass
class RobotSpec:
    robot_type: str
    left_joints: list[str]
    right_joints: list[str]
    dt_s: float
    gripper_bounds: list[list[float]]
    # Bounds are in the CoRobot wire action convention, NOT physical finger angles.
    gripper_observation_scale: list[float] = field(default_factory=lambda: [1.0, 1.0])
    gripper_observation_offset: list[float] = field(default_factory=lambda: [0.0, 0.0])
    joint_limits: list[list[float]] | None = None
    max_joint_speed: float = 1.0
    max_eef_speed: float = 0.25
    max_eef_angular_speed: float = 1.0
    max_gripper_speed: float = 2.0
    supports_eef: bool = False
    enforce_speed_limits: bool = True
    max_chunk_steps: int = 256
    body_joints: list[str] = field(default_factory=list)
    body_limits: list[list[float]] = field(default_factory=list)
    body_speed_limits: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.robot_type, str)
            or not self.robot_type
            or not isinstance(self.left_joints, list)
            or not self.left_joints
            or not isinstance(self.right_joints, list)
            or not self.right_joints
        ):
            raise ContractError("robot type and joint names are required")
        if not isinstance(self.body_joints, list):
            raise ContractError("body_joints must be a list")
        names = self.left_joints + self.right_joints + self.body_joints
        if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ContractError("joint names must be nonempty and unique")
        for key in ("dt_s", "max_joint_speed", "max_eef_speed", "max_eef_angular_speed", "max_gripper_speed"):
            if number(getattr(self, key), key) <= 0:
                raise ContractError(f"{key} must be positive")
        if type(self.max_chunk_steps) is not int or self.max_chunk_steps < 1:
            raise ContractError("max_chunk_steps must be a positive integer")
        if type(self.enforce_speed_limits) is not bool:
            raise ContractError("enforce_speed_limits must be boolean")
        if type(self.supports_eef) is not bool:
            raise ContractError("supports_eef must be boolean")
        self.gripper_observation_scale = vector(self.gripper_observation_scale, 2, "gripper scale")
        self.gripper_observation_offset = vector(self.gripper_observation_offset, 2, "gripper offset")
        self._bounds(self.gripper_bounds, 2, "gripper_bounds")
        if self.joint_limits is not None:
            self._bounds(self.joint_limits, len(self.left_joints + self.right_joints), "joint_limits")
        self._bounds(self.body_limits, len(self.body_joints), "body_limits")
        speeds = vector(self.body_speed_limits, len(self.body_joints), "body_speed_limits")
        if any(v <= 0 for v in speeds):
            raise ContractError("body speed limits must be positive")

    @staticmethod
    def _bounds(values: Any, size: int, name: str) -> None:
        if not isinstance(values, list) or len(values) != size:
            raise ContractError(f"{name} must have {size} bounds")
        for bounds in values:
            low, high = vector(bounds, 2, name)
            if low >= high:
                raise ContractError(f"{name} requires lower < upper")

    def describe(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "per_step_limits": {
                "eef_translation_m": self.max_eef_speed * self.dt_s,
                "eef_rotation_rad": self.max_eef_angular_speed * self.dt_s,
                "joint_delta_rad": self.max_joint_speed * self.dt_s,
                "gripper_delta": self.max_gripper_speed * self.dt_s,
            },
        }


@dataclass
class Observation:
    robot_type: str
    joints: list[float]
    grippers: list[float]
    end_pose: dict[str, Any]
    images: dict[str, Any]
    instruction: str
    episode_key: tuple[str, int]
    done: bool
    state_groups: dict[str, list[float]]
    history: list[dict[str, Any]] = field(default_factory=list)
    # Runtime-owned command continuity; never a playback acknowledgement or sensor value.
    issued_gripper_targets: list[float] | None = None
    body: list[float] = field(default_factory=list)
    # Runtime-owned body command continuity (e.g. a slow waist drive still converging).
    issued_body_targets: list[float] | None = None

    def eef(self, arm: str, frame: str) -> list[float]:
        try:
            p = self.end_pose[frame][arm + "_arm"]
            return pose(p["position"] + p["orientation"], f"{arm} observed EEF")
        except (KeyError, TypeError) as exc:
            raise ContractError(f"missing {arm} end_pose in {frame}") from exc

    def command_grippers(self, spec: RobotSpec) -> list[float]:
        """Convert measured positions to command units, without inferring actuator targets."""
        return [
            v * s + b
            for v, s, b in zip(self.grippers, spec.gripper_observation_scale, spec.gripper_observation_offset)
        ]


@dataclass
class ActionChunk:
    kind: str
    frame: str
    dt_s: float
    left_arm: list[list[float]]
    right_arm: list[list[float]]
    grippers: list[list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)
    body: list[list[float]] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return len(self.left_arm)

    def prefix(self, steps: int) -> ActionChunk:
        return ActionChunk(
            self.kind,
            self.frame,
            self.dt_s,
            self.left_arm[:steps],
            self.right_arm[:steps],
            self.grippers[:steps],
            dict(self.metadata),
            self.body[:steps],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionChunk:
        if not isinstance(data, dict):
            raise ContractError("action must be an object")
        expected = {"kind", "frame", "dt_s", "left_arm", "right_arm", "grippers"}
        if set(data) - {"body"} != expected:
            raise ContractError(f"action requires exactly {sorted(expected)}")
        return cls(**data)


@dataclass
class Decision:
    mode: str
    instruction: str
    reason: str
    backend: str | None = None
    policy_capability: str | None = None
    motion: dict[str, Any] | None = None
    success_condition: str = "Verify the subtask from the next observation."
    status: str = "continue"
    policy_chunks: int = 1
    proposal_id: str | None = None
    assessment: ReviewAssessment | None = None

    @classmethod
    def from_dict(cls, data: Any) -> Decision:
        if not isinstance(data, dict):
            raise ContractError("decision must be an object")
        try:
            result = cls(**data)
        except TypeError as exc:
            raise ContractError(f"invalid decision fields: {exc}") from exc
        if result.assessment is not None:
            result.assessment = ReviewAssessment.from_dict(result.assessment)
            if result.assessment.observation_request is not None and (
                result.mode != "motion" or result.status != "continue"
            ):
                raise ContractError("observation_request requires continuing motion")
            if result.assessment.motion_alternative is not None and (
                result.mode != "motion" or result.status != "continue" or result.policy_chunks != 1
            ):
                raise ContractError("motion_alternative requires continuing motion with policy_chunks=1")
        if result.proposal_id is not None and (
            not isinstance(result.proposal_id, str) or not result.proposal_id
        ):
            raise ContractError("proposal_id must be nonempty text or null")
        if result.proposal_id is not None and (result.mode != "policy" or result.status != "continue"):
            raise ContractError("proposal_id requires continuing policy mode")
        if result.mode not in MODES or result.status not in ("continue", "complete", "abandon"):
            raise ContractError("invalid decision mode or status")
        if type(result.policy_chunks) is not int or not 1 <= result.policy_chunks <= 32:
            raise ContractError("policy_chunks must be an integer in [1,32]")
        if result.policy_chunks != 1 and (result.mode != "policy" or result.status != "continue"):
            raise ContractError("policy_chunks > 1 requires a continuing policy decision")
        for key in ("instruction", "reason", "success_condition"):
            if not isinstance(getattr(result, key), str) or not getattr(result, key).strip():
                raise ContractError(f"decision {key} must be nonempty text")
        if result.backend is not None and not isinstance(result.backend, str):
            raise ContractError("backend must be a string or null")
        if result.policy_capability is not None and (
            not isinstance(result.policy_capability, str) or not result.policy_capability.strip()
        ):
            raise ContractError("policy_capability must be nonempty text or null")
        if result.motion is not None and not isinstance(result.motion, dict):
            raise ContractError("motion must be an object or null")
        if result.mode == "motion" and result.backend is not None:
            raise ContractError(
                "mode=motion requires backend=null (JSON null, not a string); "
                "remove backend or set it to null, and keep mode=motion"
            )
        if result.mode != "policy" and result.policy_capability is not None:
            raise ContractError("policy_capability requires mode=policy")
        if result.status == "continue":
            if result.mode == "motion" and result.motion is None:
                raise ContractError("motion requires a motion intent")
            if result.mode == "policy" and result.motion is not None:
                raise ContractError("policy does not accept motion")
        return result
