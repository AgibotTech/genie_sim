# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Initialize objects_list
    objects_list = {
        "benchmark_building_blocks_073": {"has_interactions": False},
        "benchmark_building_blocks_085": {"has_interactions": False},
        "benchmark_building_blocks_089": {"has_interactions": False},
        "table_000": {"has_interactions": False}
    }

    scene_info = robot.scene_info

    # 1. Find Table
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        if obj_info.get("usd") == "table_000" or "table" in obj_info.get("keywords", []):
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    # Calculate table top surface Z height
    table_top_z = table_pos[2] + table_size[2] / 2.0

    # 2. Find Scattered Blocks (No demo stack in scene anymore)
    scattered_blocks = {}
    color_to_usd = {
        "red": "benchmark_building_blocks_073",
        "yellow": "benchmark_building_blocks_085",
        "purple": "benchmark_building_blocks_089"
    }
    color_to_zh = {
        "red": "红",
        "yellow": "黄",
        "purple": "紫"
    }

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Identify the block color based on USD and ensure it's a scattered block
        for color, target_usd in color_to_usd.items():
            if usd == target_usd and "scattered" in keywords:
                scattered_blocks[color] = {"id": obj_id, "usd": usd, "color": color}

    if len(scattered_blocks) != 3:
        print("Did not find exactly 3 scattered blocks")
        return False

    # 3. Determine Stacking Order (Randomly generate from bottom to top)
    colors = ["red", "yellow", "purple"]
    stack_order = list(np.random.permutation(colors))

    # 4. Set Task Description based on the random order
    bottom_zh = color_to_zh[stack_order[0]]
    middle_zh = color_to_zh[stack_order[1]]
    top_zh = color_to_zh[stack_order[2]]

    task_name = f"按照{bottom_zh}{middle_zh}{top_zh}从下到上的顺序堆叠积木"
    english_task_name = f"stack the blocks from bottom to top in the order of {stack_order[0]}, {stack_order[1]}, {stack_order[2]}"
    print(task_name)
    description_json = {
        "task_name": english_task_name,
        "english_task_name": english_task_name,
        "init_scene_text": "The robot is in front of a table with three scattered building blocks (red, yellow, purple)."
    }
    robot.client.set_task_task_basic_info(description_json)

    # 5. Map Scattered Blocks to Order
    ordered_blocks = []
    for color in stack_order:
        ordered_blocks.append(scattered_blocks[color])

    # 6. Define Target Stacking Position
    # Stack at the center of the table
    target_xy = table_pos[:2]

    # Block size and margin
    block_size = 0.04
    margin = 0.01

    # 7. Generate candidate upright rotations
    # Since there's no demo stack to copy orientation from, we generate upright Z-axis poses.
    # Assuming default block Y-axis is up, rotate 90 deg around X to make Z up.
    base_rotations = [
        euler2mat([np.pi/2, 0, 0]),   # Y -> Z
        euler2mat([-np.pi/2, 0, 0]),  # -Y -> Z
    ]
    z_rot_angles = np.linspace(0, 2*np.pi, 4, endpoint=False) # 8 orientations

    # 8. Execute Stacking
    placed_blocks = []
    last_arm = ""

    for i, block_info in enumerate(ordered_blocks):
        block_id = block_info["id"]
        color = block_info["color"]

        print(f"\n--- Stacking block {i+1}/3: {color} ({block_id}) ---")

        block_pos, _ = robot.get_object_pose(block_id)
        arm = robot.judge_arm_to_use(block_pos)

        # Calculate target Z height
        if i == 0:
            target_z = table_top_z + margin + block_size / 2.0
        else:
            prev_block_id = placed_blocks[-1]
            prev_pos, _ = robot.get_object_pose(prev_block_id)
            prev_top_z = prev_pos[2] + block_size / 2.0
            target_z = prev_top_z + margin + block_size / 2.0

        target_position = np.array([target_xy[0], target_xy[1], target_z])

        # Generate Candidate Poses (144 combinations)
        target_poses = []
        x_offsets = np.linspace(-0.002, 0.002, 3)
        y_offsets = np.linspace(-0.002, 0.002, 3)

        for dx in x_offsets:
            for dy in y_offsets:
                pos = target_position + np.array([dx, dy, 0])
                for base_rot in base_rotations:
                    for z_angle in z_rot_angles:
                        rot_z = euler2mat([0, 0, z_angle])
                        final_rot = np.dot(rot_z, base_rot)
                        pose = np.eye(4)
                        pose[:3, :3] = final_rot
                        pose[:3, 3] = pos
                        target_poses.append(pose)

        target_poses = np.array(target_poses)

        # Unmovable objects: Table + already placed blocks
        unmovable = [table_id] + placed_blocks

        # Checker function to verify successful placement
        def create_checker(b_id, exp_pos):
            def check():
                curr_pos, _ = robot.get_object_pose(b_id)
                if curr_pos is None: return False
                # Check XY distance from target center
                dist_xy = np.linalg.norm(curr_pos[:2] - exp_pos[:2])
                if dist_xy > 0.02: return False
                # Check Z height difference
                if abs(curr_pos[2] - exp_pos[2]) > 0.02: return False
                return True
            return check

        checker = create_checker(block_id, target_position)

        # Action Descriptions dynamically based on color and position
        position_zh = ["最底层", "中间层", "最顶层"][i]
        position_en = ["bottom", "middle", "top"][i]
        color_zh = color_to_zh[color]

        grasp_desc = {"zh": f"拿起{color_zh}积木", "en": f"picks up the {color} block"}
        place_desc = {"zh": f"将{color_zh}积木放置到塔的{position_zh}", "en": f"places the {color} block on the {position_en} of the tower"}

        grasp_config = GraspConfig(
            unmovable_objects=unmovable,
            action_description=grasp_desc
        )

        place_config = PlaceConfig(
            unmovable_objects=unmovable,
            extra_checkers=[checker],
            action_description=place_desc
        )

        # Return previous arm to origin if switching arms
        if last_arm != "" and last_arm != arm:
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        success = robot.pick_and_place_actor_to_target_pose(
            actor=block_id,
            target_pose=target_poses,
            arm=arm,
            grasp_config=grasp_config,
            place_config=place_config
        )

        if not success:
            print(f"Failed to stack block {block_id}")
            return False

        placed_blocks.append(block_id)
        print(f"Successfully stacked block {block_id}")

    # Return arm to origin at the end
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("Task completed successfully")
    return True
