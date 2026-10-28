# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("genie_sim_controllers"),
        "config",
        "4wd.yaml",
    )

    servo_node = Node(
        package="genie_sim_controllers",
        executable="genie_sim_chassis_servo_node",
        name="servo_node",
        parameters=[config],
        output="screen",
    )

    return LaunchDescription([servo_node])
