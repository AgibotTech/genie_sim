# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Reject invalid chunks without silently clamping or replanning their actions."""

import math

from .geometry import quaternion_angle
from .models import FRAMES, ActionChunk, ContractError, Observation, RobotSpec, number, pose, vector

# Simulator float32 endpoints can differ from configured decimal bounds
# (e.g. -0.785 becomes -0.785000026). This tolerance does not alter actions.
_GRIPPER_BOUND_EPSILON = 1e-7


class ActionValidator:
    def __init__(self, spec: RobotSpec):
        self.spec = spec

    def validate(self, chunk: ActionChunk, obs: Observation) -> None:
        s = self.spec
        if chunk.kind not in ("JOINT_ABS", "EEF_ABS") or chunk.frame not in FRAMES:
            raise ContractError("unsupported action kind or frame")
        if not math.isclose(number(chunk.dt_s, "dt_s"), s.dt_s, rel_tol=0, abs_tol=1e-9):
            raise ContractError("action dt_s must match configured simulator action-step period")
        if not isinstance(chunk.left_arm, list) or not 1 <= len(chunk.left_arm) <= s.max_chunk_steps:
            raise ContractError("empty or oversized chunk")
        if not isinstance(chunk.right_arm, list) or not isinstance(chunk.grippers, list):
            raise ContractError("action sequences must be lists")
        if len(chunk.right_arm) != chunk.steps or len(chunk.grippers) != chunk.steps:
            raise ContractError("all action sequences must have equal horizons")
        if not isinstance(chunk.body, list) or (chunk.body and not s.body_joints):
            raise ContractError("body actions require configured body joints")
        if chunk.body and len(chunk.body) != chunk.steps:
            raise ContractError("body actions must have the same horizon")
        vector(obs.body, len(s.body_joints), "observed body")
        # Like grippers: setpoint continuity is checked against the last issued body command.
        previous_body = vector(
            obs.issued_body_targets if obs.issued_body_targets is not None else obs.body,
            len(s.body_joints),
            "issued body targets",
        )
        for i, values in enumerate(chunk.body):
            values = vector(values, len(s.body_joints), "body action")
            for value, previous_value, bounds, speed in zip(
                values, previous_body, s.body_limits, s.body_speed_limits
            ):
                if not bounds[0] - 1e-7 <= value <= bounds[1] + 1e-7:
                    raise ContractError(f"body joint limit exceeded at step {i}")
                if s.enforce_speed_limits and abs(value - previous_value) > speed * s.dt_s + 1e-9:
                    raise ContractError(f"body joint speed exceeded at step {i}")
            previous_body = values
        joint = chunk.kind == "JOINT_ABS"
        if not joint and not s.supports_eef:
            raise ContractError("robot profile does not enable simulator-side EEF IK")
        if joint:
            n = len(s.left_joints)
            previous = [obs.joints[:n], obs.joints[n:]]
        else:
            previous = [obs.eef("left", chunk.frame), obs.eef("right", chunk.frame)]
        # Limit setpoint changes across chunks. Contact may prevent the measured
        # fingers reaching their target; starting from that measurement releases
        # the existing closing command. Physical tracking remains separate feedback.
        last_grippers = (
            vector(obs.issued_gripper_targets, 2, "issued gripper targets")
            if obs.issued_gripper_targets is not None
            else obs.command_grippers(s)
        )
        for i, (left, right, grippers) in enumerate(zip(chunk.left_arm, chunk.right_arm, chunk.grippers)):
            rows = (
                [
                    vector(left, len(s.left_joints), "left joints"),
                    vector(right, len(s.right_joints), "right joints"),
                ]
                if joint
                else [pose(left), pose(right)]
            )
            if joint and s.joint_limits:
                for value, (lower, upper) in zip(rows[0] + rows[1], s.joint_limits):
                    if not lower <= value <= upper:
                        raise ContractError(f"joint limit exceeded at step {i}")
            for arm, row, prev in zip(("left", "right"), rows, previous):
                if not s.enforce_speed_limits:
                    continue
                if joint:
                    if any(abs(a - b) > s.max_joint_speed * s.dt_s + 1e-9 for a, b in zip(row, prev)):
                        raise ContractError(f"joint speed exceeded at step {i}")
                else:
                    if math.dist(row[:3], prev[:3]) > s.max_eef_speed * s.dt_s + 1e-9:
                        raise ContractError(
                            f"EEF linear speed exceeded at step {i}: arm={arm}, frame={chunk.frame}, "
                            f"distance_m={math.dist(row[:3], prev[:3]):.9f}, "
                            f"allowed_m={s.max_eef_speed * s.dt_s:.9f}, "
                            f"previous_xyz={prev[:3]}, target_xyz={row[:3]}"
                        )
                    if quaternion_angle(row[3:], prev[3:]) > s.max_eef_angular_speed * s.dt_s + 1e-9:
                        raise ContractError(f"EEF angular speed exceeded at step {i}")
            gs = vector(grippers, 2, "grippers")
            for g, last, (lower, upper) in zip(gs, last_grippers, s.gripper_bounds):
                if not lower - _GRIPPER_BOUND_EPSILON <= g <= upper + _GRIPPER_BOUND_EPSILON:
                    raise ContractError(
                        f"gripper bounds exceeded at step {i}: command {g!r}, allowed [{lower!r}, {upper!r}]. "
                        "Measured positions may overshoot; use a valid command or omit grippers to hold."
                    )
                if s.enforce_speed_limits and abs(g - last) > s.max_gripper_speed * s.dt_s + 1e-9:
                    raise ContractError(f"gripper speed exceeded at step {i}")
            previous, last_grippers = rows, gs
