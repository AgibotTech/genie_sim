# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np

def execute_task(robot: RobotAPI):
    scene_info = robot.scene_info
    interactions = robot.interactions

    # 1. Find the chips bucket
    chips_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if "chips_bucket" in keywords or "snack" in keywords:
            chips_id = obj_id
            break

    if not chips_id:
        print("Chips bucket not found")
        return False

    # 2. Get initial pose
    init_pos, init_rot = robot.get_object_pose(chips_id)
    init_pos = np.array(init_pos)
    init_rot = np.array(init_rot)  # [w, x, y, z]

    # 3. Determine side and arms
    # Table center is roughly at (0, y, z) in world, Left side y > 0, Right side y < 0
    # Based on scene generation: Left y (0.15, 0.25), Right y (-0.25, -0.15)
    is_on_left = init_pos[1] > 0

    # First hand: Corresponding to initial position
    arm_a = "left" if is_on_left else "right"
    # Second hand: The other one
    arm_b = "right" if is_on_left else "left"

    print(f"Chips on {'Left' if is_on_left else 'Right'}. First hand: {arm_a}, Second hand: {arm_b}")

    # 4. Task Description
    description_json = {
        "task_name": "pick up chips bucket with one hand, pass to the other, and place on the opposite side",
        "english_task_name": "pick up chips bucket with one hand, pass to the other, and place on the opposite side",
        "init_scene_text": "A bucket of chips is on the table."
    }
    robot.client.set_task_task_basic_info(description_json)

    # 5. First Grasp (Hand A)
    # Randomly choose lower (0-30) or upper (70-100) percentile
    choose_lower = np.random.choice([True, False])
    UPPER_PART_LOWER_PERCENTILE = 70
    UPPER_PART_UPPER_PERCENTILE = 90
    LOWER_PART_LOWER_PERCENTILE = 10
    LOWER_PART_UPPER_PERCENTILE = 30
    grasp_a_config = GraspConfig(
        set_grasp_pose_center=True,
        set_grasp_along_axis='y',
        grasp_lower_percentile=LOWER_PART_LOWER_PERCENTILE if choose_lower else UPPER_PART_LOWER_PERCENTILE,
        grasp_upper_percentile=LOWER_PART_UPPER_PERCENTILE if choose_lower else UPPER_PART_UPPER_PERCENTILE,
        action_description={"zh": "抓起薯片桶", "en": "grasps the chips bucket"}
    )

    print(f"Hand {arm_a} grasping ({'lower' if choose_lower else 'upper'} part)...")
    success = robot.pick_actor(chips_id, arm=arm_a, grasp_config=grasp_a_config)
    if not success:
        return False

    # 6. Move to Handoff Pose (Hand A)
    gripper_pose = robot.get_ee_pose(arm_a)
    # Position: x(-0.1 ~ -0.05), y(0.0). Z set to a reasonable chest height (e.g., 0.3m)
    handoff_x = np.random.uniform(-0.1, -0.05)
    handoff_y = 0.0
    handoff_z = gripper_pose[2, 3] # Chest height approximation

    gripper_direction_quaternion = np.array([0,0,0.70711, -0.70711]) if is_on_left else np.array([0,0,0.70711, 0.70711])
    base_rot_mat = quat2mat_wxyz(gripper_direction_quaternion)
    # Generate candidates with small Z rotation (±5 deg)
    handoff_poses = []
    z_jitter = np.linspace(-0.05, 0.15, 5)
    y_jitter = np.linspace(-0.02, 0.02, 5) # Small y jitter
    angle_jitter = np.linspace(-15, 15, 5)

    for z_off in z_jitter:
        for y_off in y_jitter:
            for angle in angle_jitter:
                # Apply rotation around World Z

                pose = np.eye(4)
                pose[:3, :3] = base_rot_mat
                pose[:3, 3] = [handoff_x, handoff_y + y_off, handoff_z + z_off]
                pose = rotate_along_axis(pose, angle, "z", use_local=False)
                handoff_poses.append(pose)

    handoff_poses = np.array(handoff_poses)

    print(f"Hand {arm_a} moving to handoff pose...")
    move_config = MoveConfig(
        action_description={"zh": "移动到胸前位置", "en": "moves to chest position"}
    )
    success = robot.move_actor_to_target_pose("gripper", handoff_poses, arm=arm_a, move_config=move_config)
    if not success:
        return False

    # 7. Second Grasp (Hand B)
    # Grasp the other half
    def grasp_pose_filter(grasp_poses, obj_pose, grasp_widths):
        gripper_normal_direction_quat = np.array([0,0,0.70711, -0.70711]) if not is_on_left else np.array([0,0,0.70711, 0.70711])
        gripper_normal_direction_mat = quat2mat_wxyz(gripper_normal_direction_quat)
        gripper_poses_mat = grasp_poses[:, :3, :3]
        for i in range(grasp_poses.shape[0]):
            grasp_poses[i, :3, :3] = gripper_normal_direction_mat
        return grasp_poses, np.array([0.8] * grasp_poses.shape[0])

    grasp_b_config = GraspConfig(
        set_grasp_pose_center=True,
        set_grasp_along_axis='y',
        grasp_lower_percentile=UPPER_PART_LOWER_PERCENTILE if choose_lower else LOWER_PART_LOWER_PERCENTILE,
        grasp_upper_percentile=UPPER_PART_UPPER_PERCENTILE if choose_lower else LOWER_PART_UPPER_PERCENTILE,
        grasp_pose_filter=grasp_pose_filter,
        pick_up_distance=0.0,
        action_description={"zh": "接过薯片桶", "en": "receives the chips bucket"},
    )

    print(f"Hand {arm_b} grasping ({'upper' if choose_lower else 'lower'} part)...")
    success = robot.pick_actor(chips_id, arm=arm_b, grasp_config=grasp_b_config)
    if not success:
        return False
    # 8. Release and Reset Hand A
    print(f"Hand {arm_a} releasing and resetting...")
    robot.handover_ownership_of_attached_object(from_arm=arm_a)
    robot.back_to_origin(arm=arm_a)

    # 9. Place to Opposite Side (Hand B)
    # Target X: (-0.15, 0.1)
    # Target Y: Opposite side.
    # If was Left (y > 0), target Right y (-0.25, -0.15)
    # If was Right (y < 0), target Left y (0.15, 0.25)

    target_x_range = np.linspace(-0.15, 0.1, 10)
    if is_on_left:
        target_y_range = np.linspace(-0.25, -0.15, 10)
    else:
        target_y_range = np.linspace(0.15, 0.25, 10)

    margin = 0.01
    placement_z_base = init_pos[2] + margin

    # Preserve Initial Rotation (Yaw only)
    init_rot_mat = quat2mat_wxyz(init_rot)
    # Generate candidate poses
    place_poses = []
    z_jitter = np.linspace(0, 0.02, 2)
    angle_jitter = np.linspace(0, 360, 9)
    for tx in target_x_range:
        for ty in target_y_range:
            for tz_off in z_jitter:
                for angle in angle_jitter:
                    pose = np.eye(4)
                    pose[:3, :3] = init_rot_mat
                    pose = rotate_along_axis(pose, angle, "z", use_local=False)
                    pose[:3, 3] = [tx, ty, placement_z_base + tz_off]
                    place_poses.append(pose)

    place_poses = np.array(place_poses)

    print(f"Hand {arm_b} placing to opposite side...")
    def place_checker():
        robot.wait(0.5)
        cur_pose, cur_rot = robot.get_object_pose(chips_id)
        cur_rot_mat = quat2mat_wxyz(cur_rot)
        cur_pos = np.array(cur_pose)
        if np.abs(cur_pos[2] - init_pos[2]) > 0.02:
            print(f"Place failed height: {cur_pos[2]} - {init_pos[2]} > 0.02")
            return False
        return True

    place_config = PlaceConfig(
        extra_checkers=[place_checker],
        action_description={"zh": "将薯片放置到对侧", "en": "places the chips to the opposite side"}
    )
    success = robot.place_actor_to_target_pose(chips_id, target_pose=place_poses, arm=arm_b, place_config=place_config)

    return success
