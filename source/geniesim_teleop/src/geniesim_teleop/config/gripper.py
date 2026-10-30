# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Robot-specific Pico gripper commands (cmd = 1 - index trigger)."""

from dataclasses import dataclass
from geniesim_teleop.config.task_config import robot_family


@dataclass(frozen=True)
class GripperControl:
    mode: str
    joint_names: tuple[str, str]

    def targets(self, left, right):
        commands = (left, right)
        if self.mode == "velocity":
            return [-5.0 * (1.0 - cmd) if cmd < 0.8 else 2.0 for cmd in commands]
        # The inner joints mimic these outer actuators in the omnipicker USD.
        # Its actual range is 0 (closed) .. 0.785 rad (open).
        return [0.785 * cmd for cmd in commands]


def gripper_control(robot_cfg):
    family = robot_family(robot_cfg)
    if family == "G2_crsB_90d":
        return GripperControl("velocity", ("idx32_gripper_l_outer_joint1", "idx72_gripper_r_outer_joint1"))
    if "omnipicker" in family:
        return GripperControl("position", ("idx41_gripper_l_outer_joint1", "idx81_gripper_r_outer_joint1"))
    raise ValueError(f"No Pico gripper controller for {robot_cfg}; use --arm_only with an external hand controller")
