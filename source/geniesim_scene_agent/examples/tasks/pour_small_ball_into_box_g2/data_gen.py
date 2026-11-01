# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    ## Initialize objects_list based on provided interaction data
    objects_list = {
        "benchmark_cup_112": {
            "has_interactions": True,
            "object_type": "cup",
            "interaction_type": "container"
        },
        "benchmark_storage_box_007": {
            "has_interactions": True,
            "object_type": "storage_box",
            "interaction_type": "container"
        },
        "benchmark_building_blocks_111": {
            "has_interactions": True,
            "object_type": "ball",
            "interaction_type": "small_object"
        }
    }

    ## Runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## MANDATORY: Set task description before first action
    description_json = {
        "task_name": "pick up the cup and pour the ball from the cup into the box",
        "english_task_name": "pick up the cup and pour the ball from the cup into the box",
        "init_scene_text": "The robot is in front of the table, there is a paper cup with a ball inside, and a storage box on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    ## 1. Find objects (Box, Cup, Ball)
    box_id = None
    cup_id = None
    ball_id = None

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        if "benchmark_storage_box_007" in usd or any("storage_box" in kw for kw in keywords):
            box_id = obj_id
        elif "benchmark_cup_112" in usd or any("paper_cup" in kw for kw in keywords):
            cup_id = obj_id
        elif "benchmark_building_blocks_111" in usd or any("ball" in kw for kw in keywords):
            ball_id = obj_id

    if not box_id:
        print("Storage box not found")
        return False
    if not cup_id:
        print("Cup not found")
        return False
    if not ball_id:
        print("Ball not found")
        return False

    print(f"Found box: {box_id}, cup: {cup_id}, ball: {ball_id}")

    ## 2. Get positions and sizes
    box_pos, box_rot = robot.get_object_pose(box_id)
    box_size = robot.get_object_size_in_world_frame(box_id)
    cup_pos, _ = robot.get_object_pose(cup_id)

    if box_pos is None or box_size is None or cup_pos is None:
        print("Failed to get object info")
        return False

    ## 3. Determine arm
    arm = robot.judge_arm_to_use(cup_pos)
    print(f"Using arm: {arm}")

    def check_ball_in_cup():
        """Check if ball is inside the cup"""
        ball_pos, ball_rot = robot.get_object_pose(ball_id)
        ball_size = robot.get_object_size_in_local_frame(ball_id)
        cup_pos, cup_rot = robot.get_object_pose(cup_id)
        cup_size = robot.get_object_size_in_local_frame(cup_id)
        if ball_pos is None or cup_pos is None:
            return False

        # Check containment ignoring z-axis (xy plane projection)
        # For a ball in an open box, xy containment is sufficient
        cup_contains_ball, _ = check_box_containment(
            cup_pos, cup_rot, cup_size,
            ball_pos, ball_rot, ball_size,
            ignore_axes=["z"]
        )
        return cup_contains_ball

    ## 4. Pick up the cup
    grasp_config = GraspConfig(
        set_grasp_along_axis="y",
        set_grasp_pose_center=True,
        grasp_lower_percentile=40,
        grasp_upper_percentile=60,
        unmovable_objects=[box_id], # Protect box
        action_description={"zh": "拿起杯子", "en": "picks up the cup"},
        extra_checkers=[check_ball_in_cup]
    )

    print("Picking up cup...")
    if not robot.pick_actor(cup_id, arm=arm, grasp_config=grasp_config):
        print("Failed to pick up cup")
        return False

    ## 5. Move the cup above the box
    ## Calculate target position: Above the box center
    target_x = box_pos[0]
    target_y = box_pos[1]
    box_top_z = box_pos[2] + box_size[2] / 2.0

    ## Add height to hold cup above box (avoid collision)
    hold_height = 0.15  # 15cm above box top
    target_z = box_top_z + hold_height

    target_position = np.array([target_x, target_y, target_z])

    ## Generate candidate poses for moving (small variations in xy and z)
    move_poses = []
    cup_pos, cup_rot = robot.get_object_pose(cup_id)
    cup_rotation_matrix = quat2mat_wxyz(cup_rot)
    arm_offset = 0.05 if arm == "left" else -0.05
    for dx in np.linspace(-0.01, 0.01, 3):
        for dy in np.linspace(-0.01, 0.01, 3):
            for dz in np.linspace(-0.02, 0.02, 5):
                pos = target_position + np.array([dx, dy + arm_offset, dz])
                pose = np.eye(4)
                pose[:3, 3] = pos
                pose[:3, :3] = cup_rotation_matrix
                move_poses.append(pose)

    move_poses = np.array(move_poses)

    move_config = MoveConfig(
        action_description={"zh": "移动杯子到盒子上方", "en": "moves the cup above the box"},
        extra_checkers=[check_ball_in_cup]
    )

    print(f"Moving cup to above box (target z: {target_z})...")
    if not robot.move_actor_to_target_pose(cup_id, move_poses, arm=arm, move_config=move_config):
        print("Failed to move cup above box")
        return False

    ## 6. Rotate the cup to pour (around World X-axis)
    ## Constraint: Only rotate around X axis
    rotate_config = RotateConfig(
        constraint_axises=["y", "z"],  # Constrain Y and Z, only X is free
        offset_and_constraint_in_goal_frame=False,  # World frame
        action_description={"zh": "倒入小球", "en": "pours the ball"}
    )

    ## Define checker function to verify ball is in box after rotation
    def check_ball_in_box():
        """Check if ball is inside the box"""
        robot.client.run_step(30)
        ball_pos, ball_rot = robot.get_object_pose(ball_id)
        ball_size = robot.get_object_size_in_local_frame(ball_id)
        box_size_local = robot.get_object_size_in_local_frame(box_id)
        box_pos, box_rot = robot.get_object_pose(box_id)
        if ball_pos is None or box_pos is None:
            return False
        # Check containment ignoring z-axis (xy plane projection)
        # For a ball in an open box, xy containment is sufficient
        box_contains_ball, _ = check_box_containment(
            box_pos, box_rot, box_size_local,
            ball_pos, ball_rot, ball_size,
            ignore_axes=["z"]
        )
        if not box_contains_ball:
            print("Ball not in box")
        if not ball_pos[2] < box_pos[2] + box_size[2] / 2.0:
            print("Ball not below box")
            print(ball_pos[2], box_pos[2] + box_size[2] / 2.0)
        return box_contains_ball and ball_pos[2] < box_pos[2] + box_size[2] / 2.0

    ## Add checker to rotate config
    rotate_config.extra_checkers = [check_ball_in_box]

    ## Generate target poses for rotation
    rotate_poses = []

    ## Generate poses with different tilt angles around X
    cup_pos, cup_rot = robot.get_object_pose(cup_id)
    cup_pose = make_pose(cup_pos, quat2mat_wxyz(cup_rot))
    for angle_deg in [100, 110, 120, 130]:
        angle_deg = angle_deg if arm == "left" else -angle_deg
        target_pose = rotate_along_axis(cup_pose, angle_deg, "x", use_local=False)
        rotate_poses.append(target_pose)

    rotate_poses = np.array(rotate_poses)

    print("Rotating cup to pour ball...")
    if not robot.rotate_actor(cup_id, rotate_poses, arm=arm, rotate_config=rotate_config):
        print("Failed to rotate/pour cup")
        success = False
    else:
        print("Pour action completed")

    # robot.back_to_origin(arm=arm)

    return success
