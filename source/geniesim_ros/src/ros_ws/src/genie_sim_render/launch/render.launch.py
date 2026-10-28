# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

MSG_COLOR = "\033[96m"
ERR_COLOR = "\033[91m"
RESET = "\033[0m"

PERSPECTIVE_FILE = PathJoinSubstitution([FindPackageShare("genie_sim_render"), "config", "cam_x4.perspective"])


def _find_ovrtx_bin():
    try:
        import ovrtx

        return os.path.join(os.path.dirname(ovrtx.__file__), "bin")
    except ImportError:
        return ""


def _launch_setup(context):
    render = context.perform_substitution(LaunchConfiguration("render")).strip()
    stage_manifest = context.perform_substitution(LaunchConfiguration("stage_manifest")).strip()
    scene_config = context.perform_substitution(LaunchConfiguration("scene_config")).strip()
    render_fps = float(context.perform_substitution(LaunchConfiguration("render_fps")))
    prim_paths = context.perform_substitution(LaunchConfiguration("prim_paths")).strip()
    ovrtx_root = context.perform_substitution(LaunchConfiguration("ovrtx_root")).strip()

    valid_renders = {"rmagine", "ovrtx"}
    if render not in valid_renders:
        print(f"{ERR_COLOR}unknown render:={render!r}, expected one of {sorted(valid_renders)}{RESET}")
        sys.exit(1)

    actions = []

    if render == "rmagine":
        if not scene_config:
            print(f"{ERR_COLOR}scene_config is required for render:=rmagine{RESET}")
            sys.exit(1)
        scene_config = str(Path(scene_config).resolve())
        print(f"{MSG_COLOR}render: {render}, scene_config: {scene_config}, render_fps: {render_fps}{RESET}")
        actions.append(
            Node(
                package="genie_sim_render_rmagine",
                executable="genie_sim_render_rmagine",
                name="render_rmagine",
                output="screen",
                parameters=[
                    {
                        "config": scene_config,
                        "base_path": str(Path.cwd()),
                    }
                ],
            )
        )

    elif render == "ovrtx":
        if not stage_manifest:
            print(
                f"{ERR_COLOR}stage_manifest is required for render:={render}. "
                f"Run genie_sim_engine's assemble_scene first to produce manifest.json.{RESET}"
            )
            sys.exit(1)
        stage_manifest = str(Path(stage_manifest).resolve())
        if not os.path.isfile(stage_manifest):
            print(f"{ERR_COLOR}stage_manifest not found: {stage_manifest}{RESET}")
            sys.exit(1)
        print(f"{MSG_COLOR}render: {render}, stage_manifest: {stage_manifest}, render_fps: {render_fps}{RESET}")

        actions.append(
            Node(
                package="genie_sim_render",
                executable="genie_sim_render_node",
                name="render_ovrtx",
                output="screen",
                parameters=[
                    {
                        "stage_manifest": stage_manifest,
                        "render_fps": render_fps,
                        "prim_paths": prim_paths,
                        "ovrtx_root": ovrtx_root,
                    }
                ],
                remappings=[
                    # Match the rviz plugin / view_robot.rviz / ovrtx visualizer
                    # default topic so the FreeCam follows RViz when this
                    # launch file is used standalone. bringup overrides this
                    # via make_render_ovrtx_node when the physics node name
                    # differs.
                    ("~/free_cam_pose", "/genie_sim_engine/viewer/camera_pose"),
                ],
            )
        )

    actions.append(
        TimerAction(
            period=4.0,
            actions=[
                ExecuteProcess(
                    cmd=["rqt", "--perspective-file", PERSPECTIVE_FILE],
                    output="screen",
                    name="rqt_cam_viewer",
                    condition=IfCondition(LaunchConfiguration("rqt")),
                ),
            ],
        )
    )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "render",
                default_value="ovrtx",
                description="Render backend: 'rmagine' or 'ovrtx'",
            ),
            DeclareLaunchArgument(
                "stage_manifest",
                default_value="",
                description=(
                    "Path to manifest.json produced by genie_sim_engine's assemble_scene. "
                    "Required for render:=ovrtx."
                ),
            ),
            DeclareLaunchArgument(
                "scene_config",
                default_value="",
                description="Path to scene config JSON (required only for render:=rmagine)",
            ),
            DeclareLaunchArgument(
                "render_fps",
                default_value="10.0",
                description="Target render frame rate",
            ),
            DeclareLaunchArgument(
                "prim_paths",
                default_value="",
                description="Comma-separated list of prim paths matching PoseArray poses (ovrtx only)",
            ),
            DeclareLaunchArgument(
                "ovrtx_root",
                default_value=_find_ovrtx_bin(),
                description="Path to ovrtx binary package root (ovrtx only)",
            ),
            DeclareLaunchArgument(
                "rqt",
                default_value="true",
                description="Launch rqt with cam_x4 perspective",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
