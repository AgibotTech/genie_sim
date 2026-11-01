# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Initialize objects_list from scene generation and interaction data.
    objects_list = {
        "benchmark_gearbox_000": {
            "has_interactions": True,
            "object_type": "gearbox",
            "interaction_type": "container_with_cylinders"
        },
        "benchmark_gear_000": {
            "has_interactions": True,
            "object_type": "gear",
            "size": "large",
            "interaction_type": "mechanical_part"
        },
        "benchmark_gear_001": {
            "has_interactions": True,
            "object_type": "gear",
            "size": "small",
            "interaction_type": "mechanical_part"
        },
        "benchmark_bearing_000": {
            "has_interactions": True,
            "object_type": "bearing",
            "size": "large",
            "interaction_type": "mechanical_part"
        },
        "benchmark_bearing_001": {
            "has_interactions": True,
            "object_type": "bearing",
            "size": "small",
            "interaction_type": "mechanical_part"
        }
    }

    # Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: set task description before the first action.
    description_json = {
        "task_name": "pick up the assembled part",
        "english_task_name": "pick up the assembled part",
        "init_scene_text": "The robot is in front of the table, there is a gearbox with a gear or bearing mounted on one of its cylinders"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Locate the table.
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or "main_table" in keywords or "table" in keywords:
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. Locate the gearbox.
    gearbox_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if "benchmark_gearbox_000" in usd or "gearbox_center" in keywords:
            gearbox_id = obj_id
            break

    if not gearbox_id:
        print("Gearbox not found")
        return False

    print(f"Found gearbox: {gearbox_id}")

    # 3. Locate the assembled gear or bearing.
    mounted_part_id = None
    mounted_part_info = None

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check whether this is a gear or bearing.
        is_part = False
        for part_usd in ["benchmark_gear_000", "benchmark_gear_001",
                        "benchmark_bearing_000", "benchmark_bearing_001"]:
            if part_usd in usd:
                is_part = True
                break

        if is_part and (any("mounted" in kw for kw in keywords) or any("on_cylinder" in kw for kw in keywords)):
            mounted_part_id = obj_id
            mounted_part_info = obj_info
            break

    if not mounted_part_id:
        print("Mounted part not found")
        return False

    part_usd = mounted_part_info.get("usd", "")
    print(f"Found mounted part: {mounted_part_id}, usd: {part_usd}")

    # 4. Determine the exact part type and size.
    part_type_zh = ""
    part_type_en = ""
    part_size_category = None  # "small" or "large"

    if "benchmark_gear_000" in part_usd:
        part_type_zh = "大齿轮"
        part_type_en = "large gear"
        part_size_category = "large"
    elif "benchmark_gear_001" in part_usd:
        part_type_zh = "小齿轮"
        part_type_en = "small gear"
        part_size_category = "small"
    elif "benchmark_bearing_001" in part_usd:
        part_type_zh = "大轴承"
        part_type_en = "large bearing"
        part_size_category = "large"
    elif "benchmark_bearing_000" in part_usd:
        part_type_zh = "小轴承"
        part_type_en = "small bearing"
        part_size_category = "small"
    else:
        # Try identifying it through keywords.
        keywords = mounted_part_info.get("keywords", [])
        if "gear" in part_usd.lower() or any("gear" in kw.lower() for kw in keywords):
            base_zh = "齿轮"
            base_en = "gear"
        elif "bearing" in part_usd.lower() or any("bearing" in kw.lower() for kw in keywords):
            base_zh = "轴承"
            base_en = "bearing"
        else:
            base_zh = "零件"
            base_en = "part"

        if any("large" in kw.lower() for kw in keywords):
            part_type_zh = "大" + base_zh
            part_type_en = "large " + base_en
            part_size_category = "large"
        elif any("small" in kw.lower() for kw in keywords):
            part_type_zh = "小" + base_zh
            part_type_en = "small " + base_en
            part_size_category = "small"
        else:
            part_type_zh = base_zh
            part_type_en = base_en
            # Default to small.
            part_size_category = "small"

    print(f"Part identified: {part_type_zh} / {part_type_en}, size: {part_size_category}")

    # 5. Set pick_up_distance.
    if part_size_category == "large":
        pick_up_distance = 0.20  # Large part: 0.2m.
    else:
        pick_up_distance = 0.15  # Small part: 0.15m.

    print(f"Pick up distance: {pick_up_distance}m")

    # 6. Get initial part position for the checker.
    initial_part_position, _ = robot.get_object_pose(mounted_part_id)
    if initial_part_position is None:
        print(f"Cannot get initial position for part {mounted_part_id}")
        return False

    print(f"Initial part position: {initial_part_position}")

    # 7. Define a checker requiring at least 10cm rise.
    def create_pick_height_checker(part_id, initial_position):
        def checker():
            current_position, _ = robot.get_object_pose(part_id)
            if current_position is None:
                return False

            # Calculate z-position difference.
            z_diff = current_position[2] - initial_position[2]

            # Threshold: 10cm = 0.1m.
            threshold = 0.10

            if z_diff >= threshold:
                print(f"Pick checker passed: Z increased by {z_diff:.3f}m >= {threshold}m")
                return True
            else:
                print(f"Pick checker failed: Z increased by {z_diff:.3f}m < {threshold}m")
                return False

        return checker

    pick_checker = create_pick_height_checker(mounted_part_id, initial_part_position)

    # 8. Get part position to choose the arm.
    part_position, _ = robot.get_object_pose(mounted_part_id)
    if part_position is None:
        print(f"Cannot get position for part {mounted_part_id}")
        return False

    arm = robot.judge_arm_to_use(part_position)
    print(f"Using arm: {arm}")

    # 9. Configure grasp parameters.
    grasp_config = GraspConfig(
        passive_primitive="default",
        pick_up_distance=pick_up_distance,  # Set lift distance by part size.
        set_grasp_along_axis="y",
        unmovable_objects=[table_id, gearbox_id],
        extra_checkers=[pick_checker],  # Add height check.
        action_description={
            "zh": f"把{part_type_zh}卸下来",
            "en": f"remove the {part_type_en}"
        }
    )

    # 10. Execute grasp.
    print(f"Attempting to pick {part_type_zh} ({mounted_part_id})")
    success = robot.pick_actor(
        actor=mounted_part_id,
        arm=arm,
        grasp_config=grasp_config
    )

    if not success:
        print(f"Failed to pick {part_type_zh}")
        return False

    print(f"Successfully picked {part_type_zh}")

    # 11. Final verification.
    final_position, _ = robot.get_object_pose(mounted_part_id)
    if final_position is not None and initial_part_position is not None:
        z_diff = final_position[2] - initial_part_position[2]
        print(f"Final Z increase: {z_diff:.3f}m")

    print(f"\n=== Task summary ===")
    print(f"✅ SUCCESS: {part_type_zh} removed and lifted")

    return success
