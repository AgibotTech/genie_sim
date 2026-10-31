# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Fixed-horizon Cartesian interpolation for explicit timing experiments."""

import math
from typing import Any

from agent2act.domain.actions import gripper_start
from agent2act.domain.geometry import interpolate_pose, quaternion_angle
from agent2act.domain.models import (
    ActionChunk,
    ContractError,
    Decision,
    Observation,
    RobotSpec,
    normalized_pose,
    vector,
)
from agent2act.domain.timing import MotionTiming


class FixedStepsMotion:
    mode = "motion"

    def __init__(
        self, spec: RobotSpec, timing: MotionTiming, steps: int = 30, retime_to_limits: bool = False
    ) -> None:
        if type(retime_to_limits) is not bool:
            raise ContractError("retime_to_limits must be boolean")
        self.retime_to_limits = retime_to_limits
        if type(steps) is not int or not 1 <= steps <= spec.max_chunk_steps:
            raise ContractError("fixed steps must fit robot max_chunk_steps")
        if not spec.supports_eef:
            raise ContractError("fixed steps require EEF support")
        if timing.min_duration_s or timing.min_gripper_duration_s or timing.target_hold_s:
            raise ContractError("fixed steps require zero minimum durations and target hold")
        if timing.speed_fraction != 1 or timing.profile != "quintic":
            raise ContractError("fixed steps do not support timing profile or speed fraction overrides")
        self.spec, self.steps = spec, steps

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "input": "one complete Cartesian waypoint; no duration_s",
            "compiler": {
                "fixed_steps": self.steps,
                "retime_to_limits": self.retime_to_limits and self.spec.enforce_speed_limits,
                "timing": "linear position/gripper interpolation and quaternion SLERP; no extra hold",
                "speed_limits_enforced": self.spec.enforce_speed_limits,
            },
        }

    def generate(self, observation: Observation, decision: Decision) -> ActionChunk:
        motion = decision.motion
        if not isinstance(motion, dict) or set(motion) != {"frame", "waypoints"}:
            raise ContractError("fixed steps require frame and waypoints")
        points = motion["waypoints"]
        if not isinstance(points, list) or len(points) != 1:
            raise ContractError("fixed steps require exactly one waypoint")
        target = points[0]
        if not isinstance(target, dict) or set(target) != {"left_pose", "right_pose", "grippers"}:
            raise ContractError(
                "provide both full poses and grippers; fixed steps reject duration/other constraints"
            )
        # Validate complete poses without inventing omitted orientation.
        left = normalized_pose(target["left_pose"], "left_pose")
        right = normalized_pose(target["right_pose"], "right_pose")
        grippers = vector(target["grippers"], 2, "grippers")
        start_l = observation.eef("left", motion["frame"])
        start_r = observation.eef("right", motion["frame"])
        start_g, metadata = gripper_start(observation, self.spec)
        steps = self.steps
        if self.retime_to_limits and self.spec.enforce_speed_limits:
            duration = max(
                math.dist(start_l[:3], left[:3]) / self.spec.max_eef_speed,
                math.dist(start_r[:3], right[:3]) / self.spec.max_eef_speed,
                quaternion_angle(start_l[3:], left[3:]) / self.spec.max_eef_angular_speed,
                quaternion_angle(start_r[3:], right[3:]) / self.spec.max_eef_angular_speed,
                max(abs(a - b) for a, b in zip(start_g, grippers)) / self.spec.max_gripper_speed,
            )
            steps = max(steps, math.ceil(duration / self.spec.dt_s))
        if steps > self.spec.max_chunk_steps:
            raise ContractError("retimed motion exceeds robot max_chunk_steps; choose a nearer target")
        metadata.update(
            {
                "compiler": "fixed_steps",
                "motion_steps": steps,
                "target_hold_steps": 0,
                "retime_to_limits": self.retime_to_limits and self.spec.enforce_speed_limits,
            }
        )
        chunk = ActionChunk("EEF_ABS", motion["frame"], self.spec.dt_s, [], [], [], metadata)
        for i in range(1, steps + 1):
            fraction = i / steps
            chunk.left_arm.append(interpolate_pose(start_l, left, fraction))
            chunk.right_arm.append(interpolate_pose(start_r, right, fraction))
            chunk.grippers.append([a + fraction * (b - a) for a, b in zip(start_g, grippers)])
        return chunk

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass
