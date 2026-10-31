# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Local task-space interpolation backend."""

import math
from dataclasses import asdict
from typing import Any

from agent2act.domain.actions import gripper_start
from agent2act.domain.geometry import (
    interpolate_pose,
    quaternion_speed_bound,
    smooth_cruise,
    smoothstep,
)
from agent2act.domain.models import (
    FRAMES,
    ActionChunk,
    ContractError,
    Observation,
    RobotSpec,
    normalized_pose,
    number,
    vector,
)
from agent2act.domain.timing import MotionTiming


class LocalInterpolation:
    """Smooth timing and shortest-path quaternion SLERP, starting at observed EEFs."""

    def __init__(self, spec: RobotSpec, timing: MotionTiming | None = None) -> None:
        self.spec = spec
        self.timing = timing or MotionTiming()
        self._progress, self._peak_speed_factor = {
            "quintic": (smoothstep, 1.875),
            "smooth_cruise": (smooth_cruise, 1.25),
        }[self.timing.profile]

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "local_interpolation",
            "operating_timing": asdict(self.timing),
            "peak_speed_factor": self._peak_speed_factor,
            "speed_limits_enforced": self.spec.enforce_speed_limits,
            "collision_checking": False,
            "ik": "simulator",
            "timing": (
                "omit duration_s for automatic speed-limited timing; explicit duration_s is a minimum"
                if self.spec.enforce_speed_limits
                else f"moving targets without duration_s use {self.timing.default_steps} interpolation steps; "
                "speed limits do not extend timing; explicit duration_s and minimum durations apply"
            ),
        }

    def reset(self) -> None:
        pass

    def generate(self, observation: Observation, request: dict[str, Any]) -> ActionChunk:
        if not self.spec.supports_eef:
            raise ContractError("motion requires EEF-capable robot")
        if not isinstance(request, dict) or set(request) != {"frame", "waypoints"}:
            raise ContractError("motion requires exactly frame and waypoints")
        frame = request["frame"]
        if frame not in FRAMES:
            raise ContractError("unsupported motion frame")
        waypoints = request["waypoints"]
        if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= self.spec.max_chunk_steps:
            raise ContractError("motion needs a bounded, nonempty waypoint list")
        left, right = observation.eef("left", frame), observation.eef("right", frame)
        grippers, metadata = gripper_start(observation, self.spec)
        output = ActionChunk("EEF_ABS", frame, self.spec.dt_s, [], [], [], metadata)
        for wp in waypoints:
            if not isinstance(wp, dict) or set(wp) - {"duration_s", "left_pose", "right_pose", "grippers"}:
                raise ContractError("unknown waypoint fields; unsupported constraints cannot be ignored")
            duration = number(wp.get("duration_s", self.spec.dt_s), "waypoint duration_s")
            steps_float = duration / self.spec.dt_s
            if duration <= 0 or steps_float > self.spec.max_chunk_steps:
                raise ContractError("waypoint duration out of range")
            steps = max(
                1,
                round(steps_float)
                if math.isclose(steps_float, round(steps_float), rel_tol=0, abs_tol=1e-9)
                else math.ceil(steps_float),
            )
            if output.steps + steps > self.spec.max_chunk_steps:
                raise ContractError("motion exceeds max_chunk_steps")
            end_l = normalized_pose(wp["left_pose"], "left_pose") if "left_pose" in wp else left
            end_r = normalized_pose(wp["right_pose"], "right_pose") if "right_pose" in wp else right
            end_g = vector(wp["grippers"], 2, "waypoint grippers") if "grippers" in wp else grippers
            # Each profile's peak/average speed ratio bounds continuous speed.
            # Lengthen the segment for speed limits only when enforcement is enabled.
            arm_changed = end_l != left or end_r != right
            gripper_changed = end_g != grippers
            if (
                not self.spec.enforce_speed_limits
                and "duration_s" not in wp
                and (arm_changed or gripper_changed)
            ):
                steps = self.timing.default_steps
            minimum = max(
                self.timing.min_duration_s if arm_changed else 0,
                self.timing.min_gripper_duration_s if gripper_changed else 0,
            )
            required = max(steps, math.ceil(minimum / self.spec.dt_s))
            for start, end in ((left, end_l), (right, end_r)) if self.spec.enforce_speed_limits else ():
                required = max(
                    required,
                    math.ceil(
                        self._peak_speed_factor
                        * math.dist(start[:3], end[:3])
                        / (self.spec.max_eef_speed * self.timing.speed_fraction * self.spec.dt_s)
                    ),
                )
                required = max(
                    required,
                    math.ceil(
                        self._peak_speed_factor
                        * quaternion_speed_bound(start[3:], end[3:])
                        / (self.spec.max_eef_angular_speed * self.timing.speed_fraction * self.spec.dt_s)
                    ),
                )
            if self.spec.enforce_speed_limits and "grippers" in wp:
                required = max(
                    required,
                    math.ceil(
                        self._peak_speed_factor
                        * max(abs(a - b) for a, b in zip(grippers, end_g, strict=True))
                        / (self.spec.max_gripper_speed * self.timing.speed_fraction * self.spec.dt_s)
                    ),
                )
            steps = required
            if output.steps + steps > self.spec.max_chunk_steps:
                reason = "speed timing" if self.spec.enforce_speed_limits else "motion timing"
                raise ContractError(f"motion exceeds robot max_chunk_steps after {reason}")
            for step in range(1, steps + 1):
                f = self._progress(step / steps)
                output.left_arm.append(interpolate_pose(left, end_l, f))
                output.right_arm.append(interpolate_pose(right, end_r, f))
                output.grippers.append([a + f * (b - a) for a, b in zip(grippers, end_g)])
            left, right, grippers = end_l, end_r, end_g
        motion_steps = output.steps
        hold_steps = math.ceil(self.timing.target_hold_s / self.spec.dt_s)
        if motion_steps + hold_steps > self.spec.max_chunk_steps:
            raise ContractError("motion exceeds robot max_chunk_steps after target hold")
        for _ in range(hold_steps):
            output.left_arm.append(list(left))
            output.right_arm.append(list(right))
            output.grippers.append(list(grippers))
        output.metadata.update(
            {
                "motion_steps": motion_steps,
                "target_hold_steps": hold_steps,
                "timing_profile": self.timing.profile,
                "speed_limits_enforced": self.spec.enforce_speed_limits,
            }
        )
        return output

    def close(self) -> None:
        pass
