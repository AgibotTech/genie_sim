# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random

def execute_task(robot: RobotAPI):
    # Initialize objects_list based on provided interaction data
    objects_list = {
        "benchmark_building_blocks_034": {"has_interactions": True, "object_type": "building_block", "digit": 0},
        "benchmark_building_blocks_035": {"has_interactions": True, "object_type": "building_block", "digit": 1},
        "benchmark_building_blocks_036": {"has_interactions": True, "object_type": "building_block", "digit": 2},
        "benchmark_building_blocks_037": {"has_interactions": True, "object_type": "building_block", "digit": 3},
        "benchmark_building_blocks_038": {"has_interactions": True, "object_type": "building_block", "digit": 4},
        "benchmark_building_blocks_039": {"has_interactions": True, "object_type": "building_block", "digit": 5},
        "benchmark_building_blocks_040": {"has_interactions": True, "object_type": "building_block", "digit": 6},
        "benchmark_building_blocks_041": {"has_interactions": True, "object_type": "building_block", "digit": 7},
        "benchmark_building_blocks_042": {"has_interactions": True, "object_type": "building_block", "digit": 8},
        "benchmark_building_blocks_043": {"has_interactions": True, "object_type": "building_block", "digit": 9},
        "table_000": {"has_interactions": False, "object_type": "table"}
    }

    # Runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    # Randomly choose whether to pick largest or smallest (according to task requirements)
    pick_largest = random.choice([True, False])
    task_target = "最大的数字积木" if pick_largest else "最小的数字积木"
    task_target_en = "the largest numbered block" if pick_largest else "the smallest numbered block"

    description_json = {
        "task_name": f"pick up {task_target_en}",
        "english_task_name": f"pick up {task_target_en}",
        "init_scene_text": "The robot is in front of the table, there are 3-5 numbered building blocks (digits 0-9) randomly placed on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Find table (table_000)
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or any("table" in kw.lower() for kw in keywords) or any("workspace" in kw.lower() for kw in keywords):
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. Find all numbered building blocks (digits 0-9)
    numbered_blocks = []
    digit_to_usd = {
        0: "benchmark_building_blocks_034",
        1: "benchmark_building_blocks_035",
        2: "benchmark_building_blocks_036",
        3: "benchmark_building_blocks_037",
        4: "benchmark_building_blocks_038",
        5: "benchmark_building_blocks_039",
        6: "benchmark_building_blocks_040",
        7: "benchmark_building_blocks_041",
        8: "benchmark_building_blocks_042",
        9: "benchmark_building_blocks_043"
    }

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check if this is a numbered block by USD
        block_digit = None
        for digit, block_usd in digit_to_usd.items():
            if block_usd in usd:
                block_digit = digit
                break

        # Alternatively, check by keywords
        if block_digit is None:
            for kw in keywords:
                if kw.startswith("digit_") and kw[6:].isdigit():
                    try:
                        block_digit = int(kw[6:])
                        break
                    except ValueError:
                        continue

        if block_digit is not None:
            pos, rot = robot.get_object_pose(obj_id)
            size = robot.get_object_size_in_world_frame(obj_id)

            if pos is not None and size is not None:
                numbered_blocks.append({
                    "id": obj_id,
                    "digit": block_digit,
                    "position": pos,
                    "rotation": rot,
                    "size": size,
                    "usd": usd,
                    "keywords": keywords,
                    "initial_z": pos[2]  # Store initial z position for verification
                })
                print(f"Found numbered block: {obj_id}, digit: {block_digit}, usd: {usd}")

    if len(numbered_blocks) < 3:
        print(f"Found only {len(numbered_blocks)} numbered blocks, need at least 3")
        return False

    print(f"\nTotal numbered blocks found: {len(numbered_blocks)}")
    for block in numbered_blocks:
        print(f"  Block {block['id']}: digit={block['digit']}, position={block['position']}")

    # 3. Determine which block to pick (largest or smallest digit)
    # Sort blocks by digit value
    numbered_blocks.sort(key=lambda x: x["digit"])

    if pick_largest:
        # Pick the largest digit (highest number)
        target_block = numbered_blocks[-1]
        target_desc_zh = f"数字{target_block['digit']}的积木"
        target_desc_en = f"digit {target_block['digit']} block"
        print(f"Randomly selected to pick LARGEST digit: {target_block['digit']} (block: {target_block['id']})")
    else:
        # Pick the smallest digit (lowest number)
        target_block = numbered_blocks[0]
        target_desc_zh = f"数字{target_block['digit']}的积木"
        target_desc_en = f"digit {target_block['digit']} block"
        print(f"Randomly selected to pick SMALLEST digit: {target_block['digit']} (block: {target_block['id']})")

    print(f"\n=== Processing target block ===")
    print(f"Block ID: {target_block['id']}")
    print(f"Digit: {target_block['digit']}")
    print(f"Initial Position: {target_block['position']}")
    print(f"Size: {target_block['size']}")

    # 4. Prepare for grasp operation
    target_id = target_block["id"]
    target_pos = target_block["position"]
    initial_z = target_block["initial_z"]

    # Determine which arm to use
    arm = robot.judge_arm_to_use(target_pos)
    print(f"Using arm: {arm}")

    # Create list of unmovable objects (all other blocks + table)
    unmovable_objects = [table_id]
    for block in numbered_blocks:
        if block["id"] != target_id:
            unmovable_objects.append(block["id"])

    # Determine action description based on whether picking largest or smallest
    if pick_largest:
        action_desc_zh = f"拿起数字{target_block['digit']}的积木"
        action_desc_en = f"pick up digit {target_block['digit']} block"
    else:
        action_desc_zh = f"拿起数字{target_block['digit']}的积木"
        action_desc_en = f"pick up digit {target_block['digit']} block"

    # Create grasp configuration
    grasp_config = GraspConfig(
        unmovable_objects=unmovable_objects,
        action_description={"zh": action_desc_zh, "en": action_desc_en}  # REQUIRED
    )

    # 5. Execute grasp operation (only pick up, don't place down)
    print(f"\nAttempting to grasp block {target_id} (digit: {target_block['digit']})")

    # Since all blocks have interactions (as specified), use pick_actor
    print(f"Using pick_actor with interaction primitives for block {target_id}")
    success = robot.pick_actor(
        actor=target_id,
        arm=arm,
        grasp_config=grasp_config
    )

    # 6. Final verification - check if final Z position is at least 3cm higher than initial
    if success:
        print(f"\nVerifying that block was lifted at least 3cm...")

        # Get current position after grasp
        final_pos, final_rot = robot.get_object_pose(target_id)

        if final_pos is not None:
            z_difference = final_pos[2] - initial_z
            print(f"Initial Z: {initial_z:.3f}m")
            print(f"Final Z: {final_pos[2]:.3f}m")
            print(f"Z difference: {z_difference:.3f}m")

            if z_difference >= 0.03:  # At least 3cm higher
                print("✅ SUCCESS: Block was lifted at least 3cm")
                success = True
            else:
                print(f"❌ FAILED: Block was only lifted {z_difference*100:.1f}cm, less than required 3cm")
                success = False
        else:
            print("❌ FAILED: Cannot get final position of block")
            success = False
    else:
        print(f"\n❌ FAILED: Failed to grasp block {target_id} (digit: {target_block['digit']})")


    # 7. Final summary
    print(f"\n=== Task summary ===")
    print(f"Task: Pick {'largest' if pick_largest else 'smallest'} numbered block")
    print(f"Target digit: {target_block['digit']}")
    print(f"Total blocks on table: {len(numbered_blocks)}")
    print(f"Success: {'YES' if success else 'NO'}")

    return success
