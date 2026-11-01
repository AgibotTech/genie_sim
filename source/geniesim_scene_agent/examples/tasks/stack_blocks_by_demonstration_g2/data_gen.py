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
    table_top_z = table_pos[2] + table_size[2] / 2.0

    # 2. Find Scattered Blocks and Demo Stack Blocks
    scattered_blocks = []
    demo_blocks = {"bottom": None, "middle": None, "top": None}

    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        usd = obj_info.get("usd", "")

        # Check if it's a building block
        if "benchmark_building_blocks_" in usd:
            if "scattered" in keywords:
                scattered_blocks.append({"id": obj_id, "usd": usd, "keywords": keywords})
            elif "stacked_demo" in keywords:
                if "bottom" in keywords:
                    demo_blocks["bottom"] = {"id": obj_id, "usd": usd}
                elif "middle" in keywords:
                    demo_blocks["middle"] = {"id": obj_id, "usd": usd}
                elif "top" in keywords:
                    demo_blocks["top"] = {"id": obj_id, "usd": usd}

    print(f"Found {len(scattered_blocks)} scattered blocks")
    print(f"Demo stack: Bottom={demo_blocks['bottom']['usd'] if demo_blocks['bottom'] else 'None'}, "
          f"Middle={demo_blocks['middle']['usd'] if demo_blocks['middle'] else 'None'}, "
          f"Top={demo_blocks['top']['usd'] if demo_blocks['top'] else 'None'}")

    if len(scattered_blocks) != 3 or not all(demo_blocks.values()):
        print("Scene setup incomplete")
        return False

    # 3. Determine Stacking Order from Demo
    # The order is Bottom -> Middle -> Top
    target_order_usds = [
        demo_blocks["bottom"]["usd"],
        demo_blocks["middle"]["usd"],
        demo_blocks["top"]["usd"]
    ]

    # Get rotation from demo stack to match style
    demo_bottom_pos, demo_bottom_rot = robot.get_object_pose(demo_blocks["bottom"]["id"])
    # Use the rotation of the demo stack for the new stack
    target_rotation_matrix = quat2mat_wxyz(demo_bottom_rot)

    # 4. Map Scattered Blocks to Order
    ordered_blocks_to_stack = []
    for target_usd in target_order_usds:
        found = False
        for block in scattered_blocks:
            if block["usd"] == target_usd:
                ordered_blocks_to_stack.append(block)
                found = True
                break
        if not found:
            print(f"Could not find scattered block for {target_usd}")
            return False

    print(f"Stacking order determined: {[b['usd'] for b in ordered_blocks_to_stack]}")

    # 5. Set Task Description
    description_json = {
        "task_name": "stack the scattered blocks in the order shown in the demonstration",
        "english_task_name": "stack the scattered blocks in the order shown in the demonstration",
        "init_scene_text": "The robot is in front of a table with three scattered building blocks and a demonstration stack of three blocks."
    }
    robot.client.set_task_task_basic_info(description_json)

    # 6. Define Target Stacking Position
    # Choose a position on the table, e.g., center or a clear spot.
    # Let's use the table center (0, 0) relative to table, assuming table is at origin.
    # Or slightly offset to avoid collision if needed. Let's use (0, 0).
    target_stack_xy = np.array([0.0, 0.0])

    # Block size
    block_size = 0.04
    margin = 0.01

    # 7. Execute Stacking
    placed_blocks = []
    last_arm = ""
    for i, block_info in enumerate(ordered_blocks_to_stack):
        block_id = block_info["id"]
        print(f"\n--- Stacking block {i+1}/3: {block_id} ---")

        # Get block pose
        block_pos, block_rot = robot.get_object_pose(block_id)
        arm = robot.judge_arm_to_use(block_pos)

        # Calculate target height
        # Base height = table_top_z + margin + block_size/2
        # Subsequent heights = previous_block_top_z + margin + block_size/2
        if i == 0:
            target_z = table_top_z + margin + block_size / 2.0
        else:
            prev_block_id = placed_blocks[-1]
            prev_pos, _ = robot.get_object_pose(prev_block_id)
            # Top of previous block
            prev_top_z = prev_pos[2] + block_size / 2.0
            target_z = prev_top_z + margin + block_size / 2.0

        target_position = np.array([target_stack_xy[0], target_stack_xy[1], target_z])

        # Generate Candidate Poses
        target_poses = []
        # Position variations
        x_offsets = np.linspace(-0.002, 0.002, 3)
        y_offsets = np.linspace(-0.002, 0.002, 3)
        z_offsets = np.linspace(0, 0.005, 3)

        # Rotation variations (around Z axis, keeping the demo orientation style)
        # The demo orientation is already in target_rotation_matrix.
        # We can add small rotations around Z to provide candidates.
        base_rot = target_rotation_matrix

        for dx in x_offsets:
            for dy in y_offsets:
                for dz in z_offsets:
                    pos = target_position + np.array([dx, dy, dz])
                    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                        # Rotate around Z axis
                        rot_z = euler2mat([0, 0, angle])
                        final_rot = np.dot(rot_z, base_rot)

                        pose = np.eye(4)
                        pose[:3, :3] = final_rot
                        pose[:3, 3] = pos
                        target_poses.append(pose)

        target_poses = np.array(target_poses)

        # Unmovable objects: Table + already placed blocks
        unmovable = [table_id] + placed_blocks

        # Checker function
        def create_checker(b_id, exp_pos):
            def check():
                curr_pos, _ = robot.get_object_pose(b_id)
                if curr_pos is None: return False
                # Check XY distance
                dist = np.linalg.norm(curr_pos[:2] - exp_pos[:2])
                if dist > 0.02:
                    print(f"Block {b_id} is too far from expected position (dist={dist:.3f})")
                    return False
                # Check Z height
                if abs(curr_pos[2] - exp_pos[2]) > 0.02:
                    print(f"Block {b_id} is too far from expected height (dist={abs(curr_pos[2] - exp_pos[2]):.3f})")
                    return False
                return True
            return check

        checker = create_checker(block_id, target_position)

        # Action Descriptions
        if i == 0:
            grasp_desc = {"zh": "拿起最底层的积木", "en": "picks up the bottom block"}
            place_desc = {"zh": "将积木放置到桌子中心作为塔基", "en": "places the block at the center of the table as the base"}
        elif i == 1:
            grasp_desc = {"zh": "拿起中间的积木", "en": "picks up the middle block"}
            place_desc = {"zh": "将积木放置到塔的中间层", "en": "places the block on the middle layer of the tower"}
        else:
            grasp_desc = {"zh": "拿起最顶层的积木", "en": "picks up the top block"}
            place_desc = {"zh": "将积木放置到塔的顶部", "en": "places the block on top of the tower"}

        grasp_config = GraspConfig(
            unmovable_objects=unmovable,
            action_description=grasp_desc
        )

        place_config = PlaceConfig(
            unmovable_objects=unmovable,
            extra_checkers=[checker],
            action_description=place_desc
        )
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

        # Return arm to origin
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("Task completed successfully")
    return True
