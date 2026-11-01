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
        "benchmark_letter_026": {"has_interactions": True, "object_type": "letter_block", "letter": "A"},
        "benchmark_letter_027": {"has_interactions": True, "object_type": "letter_block", "letter": "B"},
        "benchmark_letter_028": {"has_interactions": True, "object_type": "letter_block", "letter": "C"},
        "benchmark_letter_029": {"has_interactions": True, "object_type": "letter_block", "letter": "D"},
        "benchmark_letter_030": {"has_interactions": True, "object_type": "letter_block", "letter": "E"},
        "benchmark_letter_031": {"has_interactions": True, "object_type": "letter_block", "letter": "F"},
        "benchmark_letter_032": {"has_interactions": True, "object_type": "letter_block", "letter": "G"},
        "benchmark_letter_033": {"has_interactions": True, "object_type": "letter_block", "letter": "H"},
        "benchmark_letter_034": {"has_interactions": True, "object_type": "letter_block", "letter": "I"},
        "benchmark_letter_035": {"has_interactions": True, "object_type": "letter_block", "letter": "J"},
        "benchmark_letter_036": {"has_interactions": True, "object_type": "letter_block", "letter": "K"},
        "benchmark_letter_037": {"has_interactions": True, "object_type": "letter_block", "letter": "L"},
        "benchmark_letter_038": {"has_interactions": True, "object_type": "letter_block", "letter": "M"},
        "benchmark_letter_039": {"has_interactions": True, "object_type": "letter_block", "letter": "N"},
        "benchmark_letter_040": {"has_interactions": True, "object_type": "letter_block", "letter": "O"},
        "benchmark_letter_041": {"has_interactions": True, "object_type": "letter_block", "letter": "P"},
        "benchmark_letter_042": {"has_interactions": True, "object_type": "letter_block", "letter": "Q"},
        "benchmark_letter_043": {"has_interactions": True, "object_type": "letter_block", "letter": "R"},
        "benchmark_letter_044": {"has_interactions": True, "object_type": "letter_block", "letter": "S"},
        "benchmark_letter_045": {"has_interactions": True, "object_type": "letter_block", "letter": "T"},
        "benchmark_letter_046": {"has_interactions": True, "object_type": "letter_block", "letter": "U"},
        "benchmark_letter_047": {"has_interactions": True, "object_type": "letter_block", "letter": "V"},
        "benchmark_letter_048": {"has_interactions": True, "object_type": "letter_block", "letter": "W"},
        "benchmark_letter_049": {"has_interactions": True, "object_type": "letter_block", "letter": "X"},
        "benchmark_letter_050": {"has_interactions": True, "object_type": "letter_block", "letter": "Y"},
        "benchmark_letter_051": {"has_interactions": True, "object_type": "letter_block", "letter": "Z"},
        "table_000": {"has_interactions": False, "object_type": "table"},
    }

    ## Runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## MANDATORY: Set task description
    task_description = {
        "task_name": "pick up each lowercase letter and place it next to its corresponding uppercase letter",
        "english_task_name": "pick up each lowercase letter and place it next to its corresponding uppercase letter",
        "init_scene_text": "The robot is in front of the table, there are 3 uppercase letters placed along y-axis and 3 corresponding lowercase letters randomly placed on the table"
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

    ## 2. Find all uppercase letters (keyword contains "uppercase" or usd range 026-051)
    uppercase_letters = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        is_uppercase = False
        letter_char = None

        ## Check keyword first
        for kw in keywords:
            if kw.startswith("uppercase_"):
                is_uppercase = True
                letter_char = kw.split("_")[1]
                break

        ## Fallback: check usd range
        if not is_uppercase:
            try:
                usd_num = int(usd.split("_")[-1])
                if 26 <= usd_num <= 51:
                    is_uppercase = True
                    letter_char = chr(ord('A') + (usd_num - 26))
            except (ValueError, IndexError):
                pass

        if is_uppercase:
            pos, rot = robot.get_object_pose(obj_id)
            size = robot.get_object_size_in_world_frame(obj_id)
            if pos is not None and size is not None:
                uppercase_letters.append({
                    "id": obj_id,
                    "usd": usd,
                    "letter": letter_char,
                    "position": pos,
                    "rotation": rot,
                    "size": size,
                })
                print(f"Found uppercase letter: {obj_id}, letter: {letter_char}")

    if len(uppercase_letters) < 3:
        print(f"Found only {len(uppercase_letters)} uppercase letters, need 3")
        return False

    ## Sort by y position (descending: left to right)
    uppercase_letters.sort(key=lambda x: x["position"][1], reverse=True)

    ## 3. Find all lowercase letters (keyword contains "lowercase" or usd range 052-077)
    lowercase_letters = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        is_lowercase = False
        letter_char = None

        ## Check keyword first
        for kw in keywords:
            if kw.startswith("lowercase_"):
                is_lowercase = True
                letter_char = kw.split("_")[1]
                break

        ## Fallback: check usd range
        if not is_lowercase:
            try:
                usd_num = int(usd.split("_")[-1])
                if 52 <= usd_num <= 77:
                    is_lowercase = True
                    letter_char = chr(ord('a') + (usd_num - 52))
            except (ValueError, IndexError):
                pass

        if is_lowercase:
            pos, rot = robot.get_object_pose(obj_id)
            size = robot.get_object_size_in_world_frame(obj_id)
            if pos is not None and size is not None:
                lowercase_letters.append({
                    "id": obj_id,
                    "usd": usd,
                    "letter": letter_char,
                    "position": pos,
                    "rotation": rot,
                    "size": size,
                })
                print(f"Found lowercase letter: {obj_id}, letter: {letter_char}")

    if len(lowercase_letters) < 3:
        print(f"Found only {len(lowercase_letters)} lowercase letters, need 3")
        return False

    ## 4. Build letter pairs by matching uppercase <-> lowercase characters
    letter_pairs = []

    for uppercase_letter in uppercase_letters:
        uppercase_char = uppercase_letter["letter"]
        corresponding_lowercase_char = uppercase_char.lower()

        for lowercase_letter in lowercase_letters:
            if lowercase_letter["letter"] == corresponding_lowercase_char:
                letter_pairs.append({
                    "uppercase": uppercase_letter,
                    "lowercase": lowercase_letter,
                })
                print(f"Found pair: {uppercase_char} <-> {corresponding_lowercase_char}")
                break

    if len(letter_pairs) < 3:
        print(f"Only found {len(letter_pairs)} letter pairs, need 3")
        return False

    print(f"\nFound {len(letter_pairs)} letter pairs")

    ## 5. Process each letter pair
    last_arm = ""

    for i, pair in enumerate(letter_pairs):
        uppercase = pair["uppercase"]
        lowercase = pair["lowercase"]
        uppercase_id = uppercase["id"]
        lowercase_id = lowercase["id"]
        uppercase_char = uppercase["letter"]
        lowercase_char = lowercase["letter"]

        print(f"\n=== Processing pair {i+1}/{len(letter_pairs)}: {uppercase_char} <-> {lowercase_char} ===")

        uppercase_pos = uppercase["position"]
        lowercase_pos = lowercase["position"]
        lowercase_size = lowercase["size"]

        ## Determine which arm to use
        arm = robot.judge_arm_to_use(lowercase_pos)
        print(f"Using arm: {arm}")

        if last_arm != arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        ## Calculate placement position: to the right of the uppercase letter
        placement_offset_y = -0.05  # 5cm to the right (-Y direction)

        table_pos, _ = robot.get_object_pose(table_id)
        table_size = robot.get_object_size_in_world_frame(table_id)
        table_top_z = table_pos[2] + table_size[2] / 2.0

        margin = 0.01  # 1cm safety margin
        placement_z = table_top_z + margin + lowercase_size[2] * 3 / 2.0

        placement_position = np.array([
            uppercase_pos[0],
            uppercase_pos[1] + placement_offset_y,
            placement_z
        ])

        print(f"Target position for lowercase {lowercase_char}: [{placement_position[0]:.3f}, {placement_position[1]:.3f}, {placement_position[2]:.3f}]")

        ## Generate candidate poses
        target_rot_matrix = quat2mat_wxyz(lowercase["rotation"])
        target_poses = []

        x_offsets = np.linspace(-0.002, 0.002, 3)
        y_offsets = np.linspace(-0.002, 0.002, 3)
        z_offsets = np.linspace(0, 0.003, 3)

        for x_off in x_offsets:
            for y_off in y_offsets:
                for z_off in z_offsets:
                    position = placement_position + np.array([x_off, y_off, z_off])
                    pose = np.eye(4)
                    pose[:3, :3] = target_rot_matrix
                    pose[:3, 3] = position
                    target_poses.append(pose)

        target_poses = np.array(target_poses)
        print(f"Generated {len(target_poses)} candidate poses")

        ## Create placement verification function
        def create_placement_checker(block_id, expected_position, block_letter):
            def check_placement():
                current_p, current_r = robot.get_object_pose(block_id)
                if current_p is None or current_r is None:
                    return False
                xy_dist = np.linalg.norm(current_p[:2] - expected_position[:2])
                if xy_dist > 0.03:
                    print(f"Letter {block_letter} position error: xy_dist={xy_dist:.3f}m")
                    return False
                return True
            return check_placement

        ## Grasp configuration
        grasp_config = GraspConfig(
            unmovable_objects=[table_id],
            action_description={
                "zh": f"拿起小写字母{lowercase_char}",
                "en": f"picks up the lowercase letter {lowercase_char}",
            },
        )

        ## Place configuration
        place_config = PlaceConfig(
            unmovable_objects=[table_id],
            extra_checkers=[create_placement_checker(lowercase_id, placement_position, lowercase_char)],
            action_description={
                "zh": f"将小写字母{lowercase_char}放置到大写字母{uppercase_char}旁边",
                "en": f"places the lowercase letter {lowercase_char} next to the uppercase letter {uppercase_char}",
            },
            disable_upside_down=False,
        )

        ## Execute pick and place
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=lowercase_id,
            target_pose=target_poses,
            arm=arm,
            grasp_config=grasp_config,
            place_config=place_config,
        )

        if not current_success:
            print(f"Failed to place lowercase letter {lowercase_char}")
            success = False
        else:
            print(f"Successfully placed lowercase letter {lowercase_char}")

    ## 6. Final verification
    print("\n=== Performing final verification ===")

    for pair in letter_pairs:
        uppercase_char = pair["uppercase"]["letter"]
        lowercase_char = pair["lowercase"]["letter"]
        uppercase_id = pair["uppercase"]["id"]
        lowercase_id = pair["lowercase"]["id"]

        lowercase_pos, _ = robot.get_object_pose(lowercase_id)
        uppercase_pos, _ = robot.get_object_pose(uppercase_id)

        if lowercase_pos is None or uppercase_pos is None:
            print(f"  Warning: Cannot get positions for pair {uppercase_char}/{lowercase_char}")
            continue

        distance = np.linalg.norm(lowercase_pos[:2] - uppercase_pos[:2])
        if distance < 0.15:
            print(f"  SUCCESS: Pair {uppercase_char}/{lowercase_char}: distance={distance:.3f}m")
        else:
            print(f"  FAIL: Pair {uppercase_char}/{lowercase_char}: distance={distance:.3f}m (too far)")
            success = False

    ## Return arm to origin
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("\n=== Task summary ===")
    if success:
        print("SUCCESS: All lowercase letters placed next to their corresponding uppercase letters")
    else:
        print("PARTIAL SUCCESS: Some operations had issues")

    return success
