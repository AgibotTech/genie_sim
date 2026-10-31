# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""All-command wrist envelopes, independent of robot geometry implementation."""

import math
from typing import Any

from .actions import gripper_start
from .geometry import quaternion_angle
from .models import ActionChunk, ContractError, Observation, RobotSpec, number, pose
from .review import CommandBounds


def wrist_envelope(
    observation: Observation, frame: str, left: list[list[float]], right: list[list[float]]
) -> dict[str, Any]:
    """Summarize every provided command, relative to the measured wrist poses."""
    if not left or len(left) != len(right):
        raise ContractError("wrist envelope requires equal nonempty trajectories")
    result: dict[str, Any] = {"frame": frame, "steps": len(left)}
    for arm, rows in (("left", left), ("right", right)):
        initial = observation.eef(arm, frame)
        poses = [pose(row) for row in rows]
        result[arm] = {
            "translation_m": max(math.dist(initial[:3], row[:3]) for row in poses),
            "rotation_rad": max(quaternion_angle(initial[3:], row[3:]) for row in poses),
        }
    return result


def validate_command_bounds(
    bounds: CommandBounds,
    observation: Observation,
    chunk: ActionChunk,
    spec: RobotSpec,
    envelope: dict[str, Any],
) -> None:
    """Reject incomplete geometry and every exceeded bound; never edit commands."""
    if not isinstance(envelope, dict):
        raise ContractError("command_bounds require all-command geometry")
    if chunk.frame != bounds.frame or envelope.get("frame") != bounds.frame:
        raise ContractError("command_bounds frame mismatch")
    if type(envelope.get("steps")) is not int or envelope["steps"] != chunk.steps:
        raise ContractError("command_bounds require all-command geometry")
    commands = gripper_start(observation, spec)[0]
    for index, arm in enumerate(("left", "right")):
        limits = getattr(bounds, arm)
        extrema = envelope.get(arm)
        if not isinstance(extrema, dict):
            raise ContractError("command_bounds require both wrist envelopes")
        for key in ("translation_m", "rotation_rad"):
            actual = number(extrema.get(key), f"command_bounds {arm}.{key}")
            if actual < 0 or actual > getattr(limits, key) + 1e-9:
                raise ContractError(f"command_bounds exceeded: {arm}.{key}={actual:.9g}")
        if any(abs(row[index] - commands[index]) > limits.gripper_change + 1e-9 for row in chunk.grippers):
            raise ContractError(f"command_bounds exceeded: {arm}.gripper_change")
