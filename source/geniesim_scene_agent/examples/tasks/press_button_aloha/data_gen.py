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
        # Add button assets if they have interactions
        "benchmark_button_009": {"has_interactions": True, "object_type": "button"},
        "benchmark_button_011": {"has_interactions": True, "object_type": "button"},
        "benchmark_button_013": {"has_interactions": True, "object_type": "button"},
        "benchmark_button_017": {"has_interactions": True, "object_type": "button"},
        "benchmark_button_018": {"has_interactions": True, "object_type": "button"},
        "benchmark_button_022": {"has_interactions": True, "object_type": "button"},
        "benchmark_button_023": {"has_interactions": True, "object_type": "button"},
        "benchmark_button_026": {"has_interactions": True, "object_type": "button"},
        "benchmark_button_027": {"has_interactions": True, "object_type": "button"},
        "benchmark_button_028": {"has_interactions": True, "object_type": "button"},
        "table_000": {"has_interactions": False, "object_type": "table"}
    }

    # Runtime data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "move the gripper above the button and press it",
        "english_task_name": "move the gripper above the button and press it",
        "init_scene_text": "The robot is in front of the table, there is a button on the table"
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

    # 2. Find the button on the table
    button_id = None
    button_info = None

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check if it's a button by USD or keywords
        is_button = False
        for button_usd in ["benchmark_button_009", "benchmark_button_011", "benchmark_button_013",
                          "benchmark_button_017", "benchmark_button_018", "benchmark_button_022",
                          "benchmark_button_023", "benchmark_button_026", "benchmark_button_027",
                          "benchmark_button_028"]:
            if button_usd in usd:
                is_button = True
                break

        if is_button or any("button" in kw.lower() for kw in keywords):
            button_id = obj_id
            button_info = obj_info
            break

    if not button_id:
        print("Button not found")
        return False

    print(f"Found button: {button_id}")

    # 3. Get button position and size
    button_pos, button_rot = robot.get_object_pose(button_id)
    button_origin_pos = button_pos.copy()
    button_origin_rot = button_rot.copy()
    button_to_be_pressed_part_pos, button_to_be_pressed_part_rot = robot.get_object_pose(button_id, articulation_part="part_00")
    button_size = robot.get_object_size_in_world_frame(button_id)

    if button_pos is None or button_size is None:
        print("Cannot get button position or size")
        return False

    print(f"Button position: {button_pos}")
    print(f"Button size: {button_size}")

    # 4. First close the gripper (empty grasp)
    # Determine which arm to use based on button position
    arm = robot.judge_arm_to_use(button_pos)
    print(f"Using arm: {arm}")


    # 5. Calculate target pose for moving above the button
    # Target: 0.02 meters above the button top surface
    button_top_z = button_pos[2] + 0.5 * button_size[2]

    target_height = button_top_z + 0.05  # 0.05m above

    # 6. Define action description for moving above button
    move_config = MoveConfig(
        action_description={"zh": "将夹爪移动到按钮上方", "en": "moves the gripper above the button"}
    )
    target_poses = []
    down_z_rotation = euler2mat([np.pi, 0, 0])
    for dx in np.linspace(-0.001, 0.001, 3):
        for dy in np.linspace(-0.001, 0.001, 3):
            position = np.array([button_pos[0] + dx, button_pos[1] + dy, target_height])
            target_pose = np.eye(4)
            target_pose[:3, 3] = position
            target_pose[:3, :3] = down_z_rotation
            for angle in np.linspace(180, 270, 12, endpoint=False):
                rotated_target_pose = rotate_along_axis(target_pose, angle, rot_axis="z", use_local=False)
                for angle_x in np.random.uniform(-10, 10, 10):
                    rotated_target_pose = rotate_along_axis(rotated_target_pose, angle_x, rot_axis="x", use_local=False)
                    for angle_y in np.random.uniform(-10, 10, 10):
                        rotated_target_pose = rotate_along_axis(rotated_target_pose, angle_y, rot_axis="y", use_local=False)

                target_poses.append(rotated_target_pose)
    target_poses = np.array(target_poses)
    robot.close_gripper(arm=arm)
    move_success = robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=target_poses,
        arm=arm,
        move_config=move_config
    )
    if not move_success:
        print("Move gripper above button failed")
        return False
    def check_button_pressed():
        button_pos, button_rot = robot.get_object_pose(button_id)
        if button_pos is None:
            return False
        if np.linalg.norm(button_pos - button_origin_pos) > 0.01:
            print("button base is moved")
            return False
        button_to_be_pressed_part_pos_after, button_to_be_pressed_part_rot_after = robot.get_object_pose(button_id, articulation_part="part_00")
        if button_to_be_pressed_part_pos_after[2] > button_to_be_pressed_part_pos[2] - 0.001:
            print(f"button is not pressed: after:{button_to_be_pressed_part_pos_after[2]}  before:{button_to_be_pressed_part_pos[2]}")
            return False
        return True
    press_move_config = MoveConfig(
        extra_params={
            "from_current_pose": True,
            "goal_offset": [0, 0, -0.05, 1, 0, 0, 0],
            "path_constraint": [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
            "offset_and_constraint_in_goal_frame": False,
        },
        extra_checkers=[check_button_pressed],
        action_description={"zh": "按压按钮", "en": "presses the button"},
        unmovable_objects=[button_id],
    )
    prim_paths = [robot.get_object_prim_path(button_id)]
    robot.client.remove_objs_from_obstacle(prim_paths)
    press_move_success = robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=target_poses,
        arm=arm,
        move_config=press_move_config,
    )
    if not press_move_success:
        print("Press button failed")
        return False



    print(f"\n=== Task summary ===")
    print("✅ SUCCESS: Gripper positioned above button and press simulated")

    return success
