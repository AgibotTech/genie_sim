# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Adapt the existing /cmd_4ws contract for the teleop simulator bridge.

The shared ROS controller fills unused fields with zero. Select only steering
positions and drive velocities here; retain the original ROS wire format.
Watchdogs belong to this teleop adapter and use the caller's monotonic clock.
"""

import math

STEER_JOINTS = (
    "idx111_chassis_lwheel_front_joint1",
    "idx131_chassis_rwheel_front_joint1",
    "idx121_chassis_lwheel_rear_joint1",
    "idx141_chassis_rwheel_rear_joint1",
)
DRIVE_JOINTS = (
    "idx112_chassis_lwheel_front_joint2",
    "idx132_chassis_rwheel_front_joint2",
    "idx122_chassis_lwheel_rear_joint2",
    "idx142_chassis_rwheel_rear_joint2",
)


def wheel_values(names, positions, velocities):
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate wheel joint names")
    try:
        steer = [positions[names.index(name)] for name in STEER_JOINTS]
        drive = [velocities[names.index(name)] for name in DRIVE_JOINTS]
    except (ValueError, IndexError) as exc:
        raise ValueError("Complete steering positions and drive velocities are required") from exc
    if not all(math.isfinite(value) for value in steer + drive):
        raise ValueError("Wheel targets/feedback must be finite")
    return steer, drive


class ChassisCommandAdapter:
    def __init__(self, timeout_s=0.3):
        self.timeout_s = timeout_s
        self.feedback_stamp = None
        self.command_stamp = None
        self.twist_stamp = None
        self.moving = False
        self.steer = None
        self.command = None

    def update_feedback(self, names, positions, velocities, now):
        steer, _ = wheel_values(names, positions, velocities)
        self.feedback_stamp = now
        if self.steer is None:
            self.steer = steer

    def update_twist(self, vx, vy, wz, now):
        self.moving = all(math.isfinite(v) for v in (vx, vy, wz)) and any(abs(v) >= 1e-6 for v in (vx, vy, wz))
        self.twist_stamp = now

    def update_command(self, names, positions, velocities, now):
        # An invalid frame must stop previously latched motion immediately.
        self.command_stamp = None
        self.command = wheel_values(names, positions, velocities)
        self.command_stamp = now

    def snapshot(self, now, playback=False):
        # A bridge in arm-only mode must not acquire ownership of the chassis.
        if self.command is None or self.steer is None:
            return None
        fresh = all(
            stamp is not None and 0 <= now - stamp < self.timeout_s
            for stamp in (self.twist_stamp, self.feedback_stamp, self.command_stamp)
        )
        if fresh and self.moving and not playback:
            self.steer, drive = self.command
        else:
            # Hold the last steering target while stopping the wheels. Repeating
            # this frame also stops the original servo's latched final command.
            drive = [0.0] * 4
        names, positions, velocities = [], [], []
        for steer_name, drive_name, position, velocity in zip(STEER_JOINTS, DRIVE_JOINTS, self.steer, drive):
            names.extend((steer_name, drive_name))
            positions.extend((position, math.nan))
            velocities.extend((math.nan, velocity))
        return names, positions, velocities, []
