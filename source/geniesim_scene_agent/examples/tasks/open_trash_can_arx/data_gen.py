# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Initialize objects_list based on interaction data
    objects_list = {
        "benchmark_trash_can_015": {"has_interactions": False, "object_type": "trash_can"},
        "table_000": {"has_interactions": False, "object_type": "table"}
    }

    # Runtime data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "move the gripper above the trash can button and press the foot pedal",
        "english_task_name": "move the gripper above the trash can button and press the foot pedal",
        "init_scene_text": "The robot is in front of the table, there is a trash can at the center of the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Find the table
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or any("table" in kw.lower() for kw in keywords):
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. Find the trash can on the table
    trash_can_id = None
    trash_can_info = None

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check if it's a trash can by USD or keywords
        if "benchmark_trash_can_015" in usd or any("trash_can" in kw.lower() for kw in keywords):
            trash_can_id = obj_id
            trash_can_info = obj_info
            break

    if not trash_can_id:
        print("Trash can not found")
        return False

    print(f"Found trash can: {trash_can_id}")

    # 3. Get trash can position and rotation
    trash_can_pos, trash_can_rot = robot.get_object_pose(trash_can_id)
    trash_can_origin_pos = trash_can_pos.copy()
    trash_can_origin_rot = trash_can_rot.copy()

    # Get part_00 position for trash can (foot pedal)
    trash_can_cover_part_pos, trash_can_cover_part_rot = robot.get_object_pose(trash_can_id, articulation_part="part_01")
    trash_can_size = robot.get_object_size_in_world_frame(trash_can_id)

    if trash_can_pos is None or trash_can_size is None:
        print("Cannot get trash can position or size")
        return False

    print(f"Trash can position: {trash_can_pos}")
    print(f"Trash can rotation: {trash_can_rot}")
    print(f"Trash can size: {trash_can_size}")

    # 4. Determine which arm to use based on trash can position
    arm = robot.judge_arm_to_use(trash_can_pos)
    print(f"Using arm: {arm}")

    # 5. Calculate target pose for moving above the button
    # First, calculate the button positions in trash can's local coordinate system
    # According to task description: x=1.0897~0.10113, y=-0.2479, z=0.02~-0.02

    # Get trash can rotation matrix
    trash_can_rot_matrix = quat2mat_wxyz(trash_can_rot)

    # Generate multiple button positions in local coordinate system
    x_candidates = np.linspace(0.1, 0.105, 5)  # 5 x positions
    y_value = -0.08847  # fixed y
    z_candidates = np.linspace(-0.02, 0.02, 5)      # 5 z positions

    # Create local to world transformation matrix
    local_to_world = np.eye(4)
    local_to_world[:3, :3] = trash_can_rot_matrix
    local_to_world[:3, 3] = trash_can_pos

    # Base orientation: gripper z-axis downward (world -z direction)
    # This means we want the gripper's local z-axis pointing downward
    # Standard gripper orientation: z-axis forward, y-axis up
    # For downward press, we need to rotate 90 degrees around x-axis
    base_downward_rotation = euler2mat([np.pi, 0, 0])  # Rotate 180 degrees around x

    # Generate target poses above the button
    target_poses = []

    for x_local in x_candidates:
        for z_local in z_candidates:
            # Local position of button
            local_pos = np.array([x_local, y_value, z_local])

            # Convert to world coordinates
            local_pos_homogeneous = np.append(local_pos, 1.0)
            world_pos_homogeneous = np.dot(local_to_world, local_pos_homogeneous)
            world_pos = world_pos_homogeneous[:3]

            # Target position: above the button (0.02m above)
            target_pos = world_pos.copy()
            target_pos[2] += 0.05  # 2cm above
            target_pose = np.eye(4)
            target_pose[:3, 3] = target_pos
            target_pose[:3, :3] = base_downward_rotation

            # Generate multiple orientations with small rotations
            # Start with base downward orientation
            for angle in np.linspace(180, 270, 12, endpoint=False):
                rotated_target_pose = rotate_along_axis(target_pose, angle, rot_axis="z", use_local=False)
                for angle_x in np.random.uniform(-10, 10, 10):
                    rotated_target_pose = rotate_along_axis(rotated_target_pose, angle_x, rot_axis="x", use_local=False)
                    for angle_y in np.random.uniform(-10, 10, 10):
                        rotated_target_pose = rotate_along_axis(rotated_target_pose, angle_y, rot_axis="y", use_local=False)
                target_poses.append(rotated_target_pose)

    target_poses = np.array(target_poses)
    print(f"Generated {len(target_poses)} candidate poses above button")

    # 6. First, close the gripper (empty grasp)
    robot.close_gripper(arm=arm)

    # 7. Define action description for moving above button
    move_config = MoveConfig(
        action_description={"zh": "将夹爪移动到垃圾桶按钮上方", "en": "moves the gripper above the trash can button"}
    )

    # Move gripper above the button
    move_success = robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=target_poses,
        arm=arm,
        move_config=move_config
    )

    if not move_success:
        print("Move gripper above button failed")
        return False

    # 8. Define checker function for button press
    def check_cover_opened():
        # Get current trash can part position
        current_trash_can_cover_part_pos, current_trash_can_cover_part_rot = robot.get_object_pose(trash_can_id, articulation_part="part_01")
        if current_trash_can_cover_part_pos is None or current_trash_can_cover_part_rot is None:
            return False

        # Check: local X-axis angle between before/after must be > 70 degrees
        rot_before = quat2mat_wxyz(trash_can_cover_part_rot)
        rot_after = quat2mat_wxyz(current_trash_can_cover_part_rot)
        x_axis_before = rot_before[:3, 0] / np.linalg.norm(rot_before[:3, 0])
        x_axis_after = rot_after[:3, 0] / np.linalg.norm(rot_after[:3, 0])
        x_axis_angle_deg = np.degrees(np.arccos(np.clip(np.dot(x_axis_before, x_axis_after), -1.0, 1.0)))
        if x_axis_angle_deg <= 70:
            print(f"Cover part X-axis angle = {x_axis_angle_deg:.1f} deg (required > 70 deg)")
            return False
        return True

    # 9. Press the button (move downward)
    gripper_pose = robot.get_ee_pose(arm=arm)
    gripper_pose[3,3] -= 0.045
    press_move_config = MoveConfig(
        extra_params={
            "from_current_pose": True,
            "goal_offset": [0, 0, -0.045, 1, 0, 0, 0],  # Move 0.045m downward
            "path_constraint": [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],  # Allow movement in z direction
            "offset_and_constraint_in_goal_frame": False,
        },
        extra_checkers=[check_cover_opened],
        action_description={"zh": "按压垃圾桶的脚按钮", "en": "presses the foot pedal of the trash can"},
        unmovable_objects=[trash_can_id],
    )

    # Remove part_00 from obstacle list (ignore collision with button part)
    prim_paths = [robot.get_object_prim_path(trash_can_id, articulation_part="part_00")]
    robot.client.remove_objs_from_obstacle(prim_paths)

    # Press the button (move downward from current position)
    press_move_success = robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=[gripper_pose],
        arm=arm,
        move_config=press_move_config,
    )

    if not press_move_success:
        print("Press button failed")
        return False


    print(f"\n=== Task summary ===")
    print("✅ SUCCESS: Trash can foot pedal pressed successfully")

    return success
