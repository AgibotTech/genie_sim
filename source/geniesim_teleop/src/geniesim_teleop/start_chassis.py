# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Launch the ROS chassis servo with teleop-local configuration."""

import os
from pathlib import Path


def chassis_command(controller_share):
    return [
        "ros2",
        "run",
        "genie_sim_controllers",
        "genie_sim_chassis_servo_node",
        "--ros-args",
        "--params-file",
        str(Path(controller_share) / "config/4wd.yaml"),
        "--params-file",
        str(Path(__file__).resolve().parent / "config/chassis_servo.yaml"),
        "-r",
        "joint_states:=/teleop/chassis_state",
    ]


def main():
    from ament_index_python.packages import get_package_share_directory

    command = chassis_command(get_package_share_directory("genie_sim_controllers"))
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
