# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os


def execute_task(robot: RobotAPI):
    # objects_list: Dictionary mapping asset_id to object metadata
    objects_list = {
        "table_000": {"has_interactions": False},
        "benchmark_microwave_oven_003": {"has_interactions": False},
    }

    # runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "open the microwave door",
        "english_task_name": "open the microwave door",
        "init_scene_text": "The robot is in front of the table, there is a microwave on the table",
    }
    robot.client.set_task_task_basic_info(description_json)

    # Find microwave in scene using keywords
    microwave_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        if "microwave" in keywords or "kitchen_appliance" in keywords:
            microwave_id = obj_id
            break

    if microwave_id is None:
        print("Microwave not found in scene")
        return False

    # Get microwave position and size
    microwave_pos, microwave_rot = robot.get_object_pose(microwave_id)
    handle_pos, handle_rot = robot.get_object_pose(microwave_id, "part_01")
    grasp_orientation_local = np.array(
        [-0.5, -0.5, -0.5, 0.5]
    )  # orientation pointing towards microwave
    local_position = np.array([0.0, 0.02, 0.0])
    grasp_position, grasp_orientation = (
        get_world_position_and_orientation_from_relative(
            local_position, grasp_orientation_local, handle_pos, handle_rot
        )
    )
    move_config = MoveConfig(
        action_description={
            "zh": "抓取微波炉门把手",
            "en": "grasp the microwave door handle",
        },
        gripper_action="close",
    )
    target_poses = []
    for z in np.linspace(-0.03, 0.03, 10):
        for rotate_z in np.linspace(-10, 10, 10):
            target_pose = np.eye(4)
            target_pose[:3, 3] = grasp_position
            target_pose[:3, :3] = quat2mat_wxyz(grasp_orientation)
            target_pose[2, 3] += z
            target_pose = rotate_along_axis(target_pose, rotate_z, "z", use_local=False)
            target_poses.append(target_pose)
    target_poses = np.array(target_poses)
    move_success = robot.move_actor_to_target_pose(
        "gripper", target_poses, arm="left", move_config=move_config
    )
    if not move_success:
        print("Move actor to target pose failed")
        return False
    robot.remove_objs_from_obstacle(
        [
            robot.get_object_prim_path(microwave_id, "part_01"),
            robot.get_object_prim_path(microwave_id, "part_00"),
        ]
    )

    def check_open_success() -> bool:
        current_joint_info = robot.get_articulation_joint_info(
            robot.get_object_prim_path(microwave_id)
        )
        current_joint_angle = current_joint_info["current_angle_deg"]
        upper_limit_deg = current_joint_info["upper_limit_deg"]
        lower_limit_deg = current_joint_info["lower_limit_deg"]
        open_threshold_ratio = 0.3
        print(
            f"current_joint_angle: {current_joint_angle}, upper_limit_deg: {upper_limit_deg}, lower_limit_deg: {lower_limit_deg}"
        )
        print(f"open_threshold_ratio: {open_threshold_ratio}")
        if current_joint_angle <= upper_limit_deg - open_threshold_ratio * (
            upper_limit_deg - lower_limit_deg
        ):
            return True
        print(
            "current_joint_angle is greater than upper_limit_deg - open_threshold_ratio * (upper_limit_deg - lower_limit_deg):",
            current_joint_angle,
            ">",
            upper_limit_deg
            - open_threshold_ratio * (upper_limit_deg - lower_limit_deg),
        )
        return False

    open_success = robot.open(
        actor=microwave_id,
        open_config=OpenConfig(
            target_limit="min",
            solvable_threshold=0.4,
            action_description={"zh": "打开微波炉门", "en": "open the microwave door"},
            extra_checkers=[check_open_success],
        ),
        arm="left",
    )

    if not open_success:
        print("Open microwave door failed")
        return False
    return True
