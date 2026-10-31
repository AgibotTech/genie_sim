# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Robot actions shared by all simulator adapters."""

from typing import Any

from .models import ActionChunk, ContractError, Observation, RobotSpec, vector


def gripper_start(observation: Observation, spec: RobotSpec) -> tuple[list[float], dict[str, Any]]:
    """Continue the issued setpoints, or initialize from bounded measured positions.

    Physics may overshoot a command bound, so the nearest valid endpoint is commanded."""
    observed = observation.command_grippers(spec)
    if observation.issued_gripper_targets is not None:
        return vector(observation.issued_gripper_targets, 2, "issued gripper targets"), {
            "gripper_start_reference": "last_issued_target; playback unverified",
            "gripper_start_observed": observed,
        }
    commands = [min(upper, max(lower, value)) for value, (lower, upper) in zip(observed, spec.gripper_bounds)]
    metadata = (
        {"gripper_start_correction": {"observed": observed, "command": commands}}
        if commands != observed
        else {}
    )
    return commands, metadata


def body_start(observation: Observation, spec: RobotSpec) -> list[float]:
    """Continue the issued body setpoints, or hold the measured body.

    A slow drive may still be converging; holding the measurement would cancel the remaining motion."""
    if observation.issued_body_targets is not None:
        return vector(observation.issued_body_targets, len(spec.body_joints), "issued body targets")
    return list(observation.body)


def hold_chunk(observation: Observation, spec: RobotSpec, steps: int = 1) -> ActionChunk:
    """Hold the current command for ``steps`` rows (a simulator may check progress periodically)."""
    if type(steps) is not int or steps < 1:
        raise ContractError("hold steps must be a positive integer")
    count = len(spec.left_joints)
    grippers, metadata = gripper_start(observation, spec)
    body = body_start(observation, spec)
    return ActionChunk(
        "JOINT_ABS",
        "base_link",
        spec.dt_s,
        [list(observation.joints[:count]) for _ in range(steps)],
        [list(observation.joints[count:]) for _ in range(steps)],
        [list(grippers) for _ in range(steps)],
        metadata,
        [list(body) for _ in range(steps)] if spec.body_joints else [],
    )
