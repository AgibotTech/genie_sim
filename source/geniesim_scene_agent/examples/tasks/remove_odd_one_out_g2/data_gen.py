# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_apple_000": {"has_interactions": True, "object_type": "apple"},
        "benchmark_apple_001": {"has_interactions": True, "object_type": "apple"},
        "benchmark_apple_002": {"has_interactions": True, "object_type": "apple"},
        "benchmark_apple_003": {"has_interactions": True, "object_type": "apple"},
        "benchmark_green_apple_000": {"has_interactions": True, "object_type": "green_apple"},
        "benchmark_green_apple_001": {"has_interactions": True, "object_type": "green_apple"},
        "benchmark_orange_001": {"has_interactions": True, "object_type": "orange"},
        "benchmark_orange_002": {"has_interactions": True, "object_type": "orange"},
        "benchmark_orange_004": {"has_interactions": True, "object_type": "orange"},
        "benchmark_peach_000": {"has_interactions": True, "object_type": "peach"},
        "benchmark_peach_019": {"has_interactions": True, "object_type": "peach"},
        "benchmark_peach_020": {"has_interactions": True, "object_type": "peach"},
        "benchmark_peach_021": {"has_interactions": True, "object_type": "peach"},
        "benchmark_lemon_027": {"has_interactions": True, "object_type": "lemon"},
        "benchmark_lemon_028": {"has_interactions": True, "object_type": "lemon"},
        "benchmark_lemon_029": {"has_interactions": True, "object_type": "lemon"},
        "benchmark_lemon_030": {"has_interactions": True, "object_type": "lemon"},
        "benchmark_pomegranate_000": {"has_interactions": True, "object_type": "pomegranate"},
        "benchmark_pomegranate_001": {"has_interactions": True, "object_type": "pomegranate"},
        "benchmark_pomegranate_046": {"has_interactions": True, "object_type": "pomegranate"},
        "benchmark_pomegranate_047": {"has_interactions": True, "object_type": "pomegranate"},
        "benchmark_building_blocks_073": {"has_interactions": True, "object_type": "building_block"},
        "benchmark_building_blocks_085": {"has_interactions": True, "object_type": "building_block"},
        "benchmark_building_blocks_089": {"has_interactions": True, "object_type": "building_block"},
        "benchmark_building_blocks_011": {"has_interactions": True, "object_type": "building_block"},
        "benchmark_billiards_001": {"has_interactions": True, "object_type": "billiard"},
        "benchmark_billiards_002": {"has_interactions": True, "object_type": "billiard"},
        "benchmark_billiards_003": {"has_interactions": True, "object_type": "billiard"},
        "benchmark_billiards_004": {"has_interactions": True, "object_type": "billiard"},
        "benchmark_billiards_005": {"has_interactions": True, "object_type": "billiard"},
        "benchmark_billiards_009": {"has_interactions": True, "object_type": "billiard"},
        "benchmark_billiards_010": {"has_interactions": True, "object_type": "billiard"},
        "benchmark_billiards_011": {"has_interactions": True, "object_type": "billiard"},
        "benchmark_billiards_012": {"has_interactions": True, "object_type": "billiard"},
        "benchmark_billiards_013": {"has_interactions": True, "object_type": "billiard"},
        "benchmark_pen_000": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_001": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_002": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_003": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_005": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_006": {"has_interactions": True, "object_type": "pen"},
        "benchmark_toy_car_018": {"has_interactions": True, "object_type": "toy_car"},
        "benchmark_toy_plane_029": {"has_interactions": True, "object_type": "toy_plane"},
        "benchmark_snack_241": {"has_interactions": True, "object_type": "snack"},
        "benchmark_snack_242": {"has_interactions": True, "object_type": "snack"},
        "benchmark_cup_043": {"has_interactions": True, "object_type": "cup"},
        "benchmark_cup_105": {"has_interactions": True, "object_type": "cup"},
        "benchmark_beverage_bottle_001": {"has_interactions": True, "object_type": "beverage"},
        "benchmark_beverage_bottle_002": {"has_interactions": True, "object_type": "beverage"},
        "benchmark_beverage_bottle_003": {"has_interactions": True, "object_type": "beverage"},
        "table_000": {"has_interactions": False, "object_type": "table"}
    }

    scene_info = robot.scene_info
    success = True

    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        if obj_info.get("usd") == "table_000" or "table" in obj_info.get("keywords", []):
            table_id = obj_id
            break
    if not table_id:
        return False

    odd_id = None
    normal_ids = []

    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        usd = obj_info.get("usd", "")
        if obj_id == table_id:
            continue
        if "odd" in keywords:
            odd_id = obj_id
        elif "normal" in keywords:
            normal_ids.append(obj_id)

    if not odd_id:
        print("Odd item not found")
        return False

    odd_info = scene_info["layout"][odd_id]
    odd_usd = odd_info.get("usd", "")
    odd_category = None
    for kw in odd_info.get("keywords", []):
        if kw not in ["odd", "on_desk", "left", "right", "building_block", "fruit"]:
            odd_category = kw
            break

    category_zh = odd_category if odd_category else "异类物品"

    description_json = {
        "task_name": f"remove the odd one out from the group",
        "english_task_name": f"remove the odd one out from the group",
        "init_scene_text": "A group of similar objects and one odd item are on the table."
    }
    robot.client.set_task_task_basic_info(description_json)

    odd_pos, _ = robot.get_object_pose(odd_id)
    arm = robot.judge_arm_to_use(odd_pos)

    unmovable = [table_id] + normal_ids

    grasp_config = GraspConfig(
        unmovable_objects=unmovable,
        action_description={"zh": f"拿起不属于这组的物品", "en": f"picks up the odd one out"},
        pick_up_distance=0.15,
    )

    success = robot.pick_actor(
        actor=odd_id,
        arm=arm,
        grasp_config=grasp_config,
    )

    robot.back_to_origin(arm=arm)
    return success
