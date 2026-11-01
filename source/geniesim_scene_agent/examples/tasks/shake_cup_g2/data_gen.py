# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Extract asset information from interaction data.
    objects_list = {
        "benchmark_cup_043": {"has_interactions": True, "object_type": "cup"},
        "benchmark_cup_105": {"has_interactions": True, "object_type": "cup"},
        "benchmark_beer_mug_007": {"has_interactions": True, "object_type": "beer_mug"},
        "benchmark_beer_mug_001": {"has_interactions": True, "object_type": "beer_mug"},
        "benchmark_beer_mug_002": {"has_interactions": True, "object_type": "beer_mug"},
    }

    # Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## MANDATORY: Set task description before first action
    description_json = {
        "task_name": "pick up the cup on the table, move to chest position and shake horizontally by a small angle",
        "english_task_name": "pick up the cup on the table, move to chest position and shake horizontally by a small angle",
        "init_scene_text": "The robot is in front of the table with a randomly selected cup (cup or beer mug) placed on it"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Locate the cup by USD path or keywords.
    cup_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check whether this cup type is supported.
        if any(cup_usd in usd for cup_usd in objects_list.keys()):
            cup_id = obj_id
            print(f"Found cup: {cup_id}, usd: {usd}")
            break

        # Alternatively check keywords.
        elif any("cup" in kw.lower() for kw in keywords):
            cup_id = obj_id
            print(f"Found cup by keyword: {cup_id}")
            break

    if not cup_id:
        print("Cup not found")
        return False

    # Get cup information.
    cup_position, _ = robot.get_object_pose(cup_id)
    cup_size = robot.get_object_size_in_world_frame(cup_id)

    if cup_position is None or cup_size is None:
        print("Cannot get cup position/size")
        return False

    # 2. Choose the arm.
    arm = robot.judge_arm_to_use(cup_position)
    print(f"Using arm: {arm}")

    # 3. Grasp the cup.
    grasp_config = GraspConfig(
        set_grasp_along_axis="y",  # Grasp along the cup axis.
        set_grasp_pose_center=True,
        unmovable_objects=[],
        action_description={"zh": "抓起桌上的杯子", "en": "picks up the cup on the table"}  # REQUIRED
    )

    if not robot.pick_actor(cup_id, arm=arm, grasp_config=grasp_config):
        print("Failed to pick up cup")
        return False

    print("Successfully picked up cup")

    # 4. Move the gripper to a horizontal pose in front of the chest for shaking.
    # Use current gripper height as reference.
    gripper_pose = robot.get_ee_pose(arm)
    shake_z = gripper_pose[2, 3]

    # Build horizontal pose: rotate 90 degrees about world Z.
    # Left and right arms use different horizontal poses.
    if arm == "left":
        gripper_direction_quat = np.array([0, 0, 0.70711, -0.70711])  # wxyz
    else:
        gripper_direction_quat = np.array([0, 0, 0.70711, 0.70711])  # wxyz

    base_rot_mat = quat2mat_wxyz(gripper_direction_quat)

    # Generate chest-front horizontal candidate poses with jitter.
    shake_poses = []
    for z_off in np.linspace(-0.05, 0.10, 5):  # z offset.
        for y_off in np.linspace(-0.02, 0.02, 5):  # y offset.
            for angle in np.linspace(-15, 15, 5):  # Small angular jitter.
                pose = np.eye(4)
                pose[:3, :3] = base_rot_mat
                # Position in front of chest, varying z.
                pose[:3, 3] = [np.random.uniform(-0.1, 0.1), y_off, shake_z + z_off]
                # Small world-z rotation.
                pose = rotate_along_axis(pose, angle, "z", use_local=False)
                shake_poses.append(pose)

    shake_poses = np.array(shake_poses)
    print(f"Generated {len(shake_poses)} candidate poses for chest position")

    def bottle_in_hand_checker():
        """Ensure the cup remains held."""
        cup_pos, _ = robot.get_object_pose(cup_id)
        gripper_pos = robot.get_ee_pose(arm)[:3, 3]
        return np.linalg.norm(cup_pos - gripper_pos) < 0.08

    move_config = MoveConfig(
        extra_checkers=[bottle_in_hand_checker],
        action_description={"zh": "移动杯子到胸前位置", "en": "moves the cup to chest position"},
    )

    if not robot.move_actor_to_target_pose("gripper", shake_poses, arm=arm, move_config=move_config):
        print("Failed to move to chest position")
        success = False

    print("Successfully moved to chest position")

    # 5. Shake the cup horizontally over 5-10-degree ranges.
    # Convert degrees to radians.
    angle_ranges = []
    for angle_deg in np.linspace(5, 10, 5):
        angle_rad = angle_deg * np.pi / 180
        angle_ranges.append([-angle_rad, angle_rad])

    print(f"Shaking with angle ranges (radians): {angle_ranges}")

    shake_config = ShakeConfig(
        action_description={"zh": "水平小幅度摇晃杯子", "en": "shakes the cup horizontally by a small angle"},
        extra_checkers=[bottle_in_hand_checker],
    )

    # horizontal=True uses joint6 for horizontal shaking; False uses joint7 vertically.
    # Use horizontal=True to meet the horizontal-shaking requirement.
    if not robot.shake(
        actor=cup_id,
        arm=arm,
        horizontal=True,  # Horizontal shaking with joint6.
        angle_ranges=angle_ranges,
        times=3,  # Repeat each angular range three times.
        shake_config=shake_config
    ):
        print("Failed to shake the cup")
        success = False

    print("Successfully shook the cup")


    return success
