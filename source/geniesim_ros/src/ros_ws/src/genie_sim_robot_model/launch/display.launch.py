# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
from pathlib import Path

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE_NAME = "genie_sim_robot_model"

CYAN = "\033[36m"
RED = "\033[31m"
RST = "\033[0m"


def _parse_xacro(xacro_file: Path, **kwargs) -> str | None:
    if not xacro_file.exists():
        print(f"{RED}xacro file not found: {xacro_file}{RST}")
        return None
    try:
        doc = xacro.process_file(str(xacro_file), mappings=kwargs)
        return doc.toprettyxml(indent="  ")
    except Exception as e:
        print(f"{RED}xacro processing failed: {e}{RST}")
        return None


def generate_launch_description():
    pkg_share = Path(get_package_share_directory(PACKAGE_NAME))
    xacro_file = pkg_share / "xacro" / "robot.xacro"

    declared_arguments = [
        DeclareLaunchArgument(
            "robot_model",
            default_value="agilex",
            description="Robot model (arx | agilex | franka | universal_robots | urdf | xacro_file)",
        ),
        DeclareLaunchArgument(
            "arm",
            default_value="",
            description="Arm variant (e.g. piper for agilex, x5 for arx, ur5 for universal_robots)",
        ),
        DeclareLaunchArgument(
            "body",
            default_value="",
            description="Body model (matches genie_sim_bringup convention; forwarded to xacro as 'body')",
        ),
        DeclareLaunchArgument(
            "gripper",
            default_value="",
            description="Gripper model (e.g. omnipicker, swiftpicker, no_gripper)",
        ),
        DeclareLaunchArgument(
            "inertia_source",
            default_value="",
            description="crsB only: 'default' (current sim values) or 'sysid' "
            "(identified set: link inertials plus joint damping and friction). "
            "Empty = xacro default.",
        ),
        DeclareLaunchArgument(
            "param_source",
            default_value="",
            description="arx acone/x5 only: 'default' (ARX stock drive gains and "
            "passive joint params) or 'sysid' (identified set; AC one carries "
            "the left-arm set on both arms). Empty = xacro default.",
        ),
        DeclareLaunchArgument(
            "rviz_config_file",
            default_value="",
            description="RViz config file (absolute path). Empty = default.",
        ),
        DeclareLaunchArgument(
            "debug",
            default_value="true",
            choices=["true", "false"],
            description="Launch joint_state_publisher_gui for interactive control",
        ),
    ]

    def launch_setup(context):
        robot_model = context.perform_substitution(LaunchConfiguration("robot_model"))
        arm = context.perform_substitution(LaunchConfiguration("arm"))
        body = context.perform_substitution(LaunchConfiguration("body"))
        gripper = context.perform_substitution(LaunchConfiguration("gripper"))
        inertia_source = context.perform_substitution(LaunchConfiguration("inertia_source"))
        param_source = context.perform_substitution(LaunchConfiguration("param_source"))
        rviz_cfg = context.perform_substitution(LaunchConfiguration("rviz_config_file"))
        debug = context.perform_substitution(LaunchConfiguration("debug"))

        xacro_args = {"robot_model": robot_model}
        if arm:
            xacro_args["arm"] = arm
        if body:
            xacro_args["body"] = body
        if gripper:
            xacro_args["gripper"] = gripper
        if inertia_source:
            xacro_args["inertia_source"] = inertia_source
        if param_source:
            xacro_args["param_source"] = param_source

        print(f"{CYAN}robot_model: {robot_model}{RST}")
        if arm:
            print(f"{CYAN}arm: {arm}{RST}")
        if body:
            print(f"{CYAN}body: {body}{RST}")
        if gripper:
            print(f"{CYAN}gripper: {gripper}{RST}")
        if inertia_source:
            print(f"{CYAN}inertia_source: {inertia_source}{RST}")
        if param_source:
            print(f"{CYAN}param_source: {param_source}{RST}")

        robot_description = _parse_xacro(xacro_file, **xacro_args)
        if not robot_description:
            return []

        publish_frequency = 125.0
        nodes = []

        nodes.append(
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_description": robot_description,
                        "publish_frequency": publish_frequency,
                        "frame_prefix": "",
                        "ignore_timestamp": True,
                    }
                ],
                remappings=[
                    ("/joint_states", "/joint_states"),
                    ("/robot_description", "/robot_description"),
                ],
            )
        )

        if debug.lower() == "true":
            nodes.append(
                Node(
                    package="joint_state_publisher_gui",
                    executable="joint_state_publisher_gui",
                    name="joint_state_publisher_gui",
                    output="both",
                    parameters=[{"rate": publish_frequency}],
                    remappings=[("/joint_states", "/joint_states")],
                )
            )

        if not rviz_cfg:
            rviz_cfg = str(pkg_share / "rviz" / "view_robot.rviz")
        rviz_path = Path(rviz_cfg)
        if not rviz_path.exists():
            print(f"{CYAN}rviz config not found: {rviz_path}, using default{RST}")
            rviz_cfg = str(pkg_share / "rviz" / "view_robot.rviz")

        print(f"{CYAN}rviz config: {rviz_cfg}{RST}")

        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="robot_state",
                output="both",
                arguments=["-d", rviz_cfg, "--ros-args", "--log-level", "info"],
            )
        )

        return nodes

    return LaunchDescription(
        [
            *declared_arguments,
            OpaqueFunction(function=launch_setup),
        ]
    )
