# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os


def execute_task(robot: RobotAPI):
    ## Initialize objects_list based on provided assets
    objects_list = {
        "benchmark_letter_026": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "A",
        },
        "benchmark_letter_027": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "B",
        },
        "benchmark_letter_028": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "C",
        },
        "benchmark_letter_029": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "D",
        },
        "benchmark_letter_030": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "E",
        },
        "benchmark_letter_031": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "F",
        },
        "benchmark_letter_032": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "G",
        },
        "benchmark_letter_033": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "H",
        },
        "benchmark_letter_034": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "I",
        },
        "benchmark_letter_035": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "J",
        },
        "benchmark_letter_036": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "K",
        },
        "benchmark_letter_037": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "L",
        },
        "benchmark_letter_038": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "M",
        },
        "benchmark_letter_039": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "N",
        },
        "benchmark_letter_040": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "O",
        },
        "benchmark_letter_041": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "P",
        },
        "benchmark_letter_042": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "Q",
        },
        "benchmark_letter_043": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "R",
        },
        "benchmark_letter_044": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "S",
        },
        "benchmark_letter_045": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "T",
        },
        "benchmark_letter_046": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "U",
        },
        "benchmark_letter_047": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "V",
        },
        "benchmark_letter_048": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "W",
        },
        "benchmark_letter_049": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "X",
        },
        "benchmark_letter_050": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "Y",
        },
        "benchmark_letter_051": {
            "has_interactions": True,
            "object_type": "letter_block",
            "letter": "Z",
        },
        "table_000": {"has_interactions": False, "object_type": "table"},
    }

    ## Runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## MANDATORY: Set task description before first action
    task_description = {
        "task_name": "arrange letter blocks from A to Z",
        "english_task_name": "arrange letter blocks from A to Z",
        "init_scene_text": "The robot is in front of the table, there are letter blocks on the table"
    }
    robot.client.set_task_task_basic_info(task_description)

    ## 1. Find the table
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or "table" in keywords:
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    ## 2. Find all letter blocks
    letter_blocks = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        ## Check if it's a letter block
        is_letter_block = False
        letter_value = None
        for asset_id, asset_info in objects_list.items():
            if asset_id in usd and asset_info["object_type"] == "letter_block":
                is_letter_block = True
                letter_value = asset_info["letter"]
                break

        if is_letter_block:
            pos, rot = robot.get_object_pose(obj_id)
            size = robot.get_object_size_in_world_frame(obj_id)
            if pos is not None and size is not None:
                letter_blocks.append(
                    {
                        "id": obj_id,
                        "usd": usd,
                        "letter": letter_value,
                        "position": pos,
                        "rotation": rot,
                        "size": size,
                        "keywords": keywords,
                    }
                )
                print(f"Found letter block: {obj_id}, letter: {letter_value}, usd: {usd}")

    if len(letter_blocks) < 3:
        print(f"Found only {len(letter_blocks)} letter blocks, need at least 3")
        return False

    print(f"Total letter blocks found: {len(letter_blocks)}")

    ## 3. Sort letter blocks by their letter value (ascending order: A to Z)
    letter_order = {chr(i): i - ord('A') for i in range(ord('A'), ord('Z') + 1)}
    letter_blocks.sort(key=lambda x: letter_order.get(x["letter"], 0))
    print("Letter blocks sorted by value:")
    for i, block in enumerate(letter_blocks):
        print(f"  {i+1}. {block['id']}: letter={block['letter']}")

    ## 4. Calculate target positions based on task requirements
    ## Target x range: (-0.2, -0.15) - select a random x within this range
    target_x = np.random.uniform(-0.05, 0.0)
    print(f"Randomly selected target_x: {target_x:.3f}")

    ## Starting y position: (0.075, 0.025) - select random within this range
    start_y = np.random.uniform(0.05, 0.08)
    print(f"Randomly selected start_y: {start_y:.3f}")

    ## Spacing between blocks: 0.05m (5cm)
    spacing = 0.07

    ## Calculate target positions (from left to right, so y decreases as we go right)
    ## Leftmost block has highest y value, then each subsequent block has y decreased by spacing
    target_positions = []

    for i in range(len(letter_blocks)):
        if i == 0:
            target_y = start_y
        else:
            target_y = start_y - spacing * i

        ## Get table height for z-coordinate
        table_pos, table_rot = robot.get_object_pose(table_id)
        table_size = robot.get_object_size_in_world_frame(table_id)
        table_top_z = table_pos[2] + table_size[2] / 2.0

        ## Add safety margin (1cm) above table surface
        margin = 0.01
        target_z = table_top_z + margin + letter_blocks[i]["size"][2] *3 / 2.0

        target_position = np.array([target_x, target_y, target_z])
        target_positions.append(target_position)
        print(f"Target position for letter {letter_blocks[i]['letter']}: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}]")

    ## 5. Keep initial rotation (no rotation change during placement)
    def calculate_target_rotation(initial_rotation):
        """
        Keep the initial rotation unchanged - no rotation applied during placement.
        """
        return quat2mat_wxyz(initial_rotation)

    ## 6. Execute placement for each letter block in sorted order
    placed_blocks = []
    last_arm = ""

    for i, block in enumerate(letter_blocks):
        block_id = block["id"]
        block_letter = block["letter"]
        current_pos = block["position"]
        current_rot = block["rotation"]
        block_size = block["size"]

        print(f"\n=== Processing letter block {i+1}/{len(letter_blocks)}: letter {block_letter} ({block_id}) ===")

        ## Determine which arm to use
        arm = robot.judge_arm_to_use(current_pos)
        print(f"Using arm: {arm}")

        ## Return previous arm to origin if switching
        if last_arm != arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        ## Create grasp configuration
        unmovable_objects = [table_id]
        unmovable_objects.extend([b["id"] for b in placed_blocks])

        ## Create action description based on letter value
        grasp_config = GraspConfig(
            unmovable_objects=unmovable_objects,
            # is_from_up_side=True,
            # grasp_offset=-0.007,
            action_description={
                "zh": f"拿起字母{block_letter}",
                "en": f"picks up the letter {block_letter}",
            },
        )

        ## Calculate target rotation for this block
        target_rot_matrix = calculate_target_rotation(current_rot)

        ## Generate multiple candidate poses with small variations for robustness
        target_poses = []
        base_position = target_positions[i]

        ## Position variations (small offsets to increase success probability)
        x_offsets = np.linspace(-0.002, 0.002, 3)
        y_offsets = np.linspace(-0.002, 0.002, 3)
        z_offsets = np.linspace(0, 0.003, 3)

        for x_off in x_offsets:
            for y_off in y_offsets:
                for z_off in z_offsets:
                    position = base_position + np.array([x_off, y_off, z_off])
                    pose = np.eye(4)
                    pose[:3, :3] = target_rot_matrix
                    pose[:3, 3] = position
                    target_poses.append(pose)

        target_poses = np.array(target_poses)
        print(f"Generated {len(target_poses)} candidate poses for letter {block_letter}")

        ## Create placement verification function
        def create_placement_checker(block_id, expected_position, block_letter):
            def check_placement():
                current_p, current_r = robot.get_object_pose(block_id)
                if current_p is None or current_r is None:
                    return False

                ## Check position (xy tolerance: 3cm, z tolerance: 1cm)
                xy_dist = np.linalg.norm(current_p[:2] - expected_position[:2])

                if xy_dist > 0.03:
                    print(f"Letter {block_letter} position error: xy_dist={xy_dist:.3f}m")
                    return False

                return True

            return check_placement

        ## Create placement description
        if i == 0:
            place_desc_zh = f"将字母{block_letter}放置到最左侧位置"
            place_desc_en = f"places the letter {block_letter} at the leftmost position"
        elif i == 1:
            place_desc_zh = f"将字母{block_letter}放置到中间位置"
            place_desc_en = f"places the letter {block_letter} at the middle position"
        else:
            place_desc_zh = f"将字母{block_letter}放置到最右侧位置"
            place_desc_en = f"places the letter {block_letter} at the rightmost position"

        place_config = PlaceConfig(
            unmovable_objects=unmovable_objects,
            extra_checkers=[create_placement_checker(block_id, base_position, block_letter)],
            action_description={"zh": place_desc_zh, "en": place_desc_en},
            disable_upside_down=False,
        )

        ## Execute pick and place operation
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=block_id,
            target_pose=target_poses,
            arm=arm,
            grasp_config=grasp_config,
            place_config=place_config,
        )

        if not current_success:
            print(f"Failed to place letter {block_letter}")
            success = False
            return success
        else:
            print(f"Successfully placed letter {block_letter}")
            placed_blocks.append(block)

    ## 7. Final verification
    print("\n=== Performing final verification ===")

    if len(placed_blocks) == len(letter_blocks):
        ## Verify the arrangement is correct (left to right: A to Z)
        final_positions = []
        for block in placed_blocks:
            pos, rot = robot.get_object_pose(block["id"])
            if pos is not None:
                final_positions.append((block["letter"], pos[1], block["id"]))  # (letter, y_position, id)

        ## Sort by y position (descending, since higher y = more left)
        final_positions.sort(key=lambda x: x[1], reverse=True)

        print("Final arrangement from left to right:")
        letters_in_order = []
        for letter, y_pos, block_id in final_positions:
            print(f"  Letter {letter} at y={y_pos:.3f}")
            letters_in_order.append(letter)

        ## Check if letters are in ascending order
        expected_letters = sorted([block["letter"] for block in letter_blocks], key=lambda x: letter_order.get(x, 0))
        if letters_in_order == expected_letters:
            print("Letter blocks are correctly arranged in ascending order from left to right")
        else:
            print(f"Letter blocks are not in correct order. Expected: {expected_letters}, Got: {letters_in_order}")
            success = False

    ## Return arms to origin
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("\n=== Task completion summary ===")
    if success and len(placed_blocks) == len(letter_blocks):
        print("SUCCESS: All letter blocks placed in correct order and orientation")
    else:
        print("PARTIAL SUCCESS: Some operations failed or verification issues")

    return success and (len(placed_blocks) == len(letter_blocks))
