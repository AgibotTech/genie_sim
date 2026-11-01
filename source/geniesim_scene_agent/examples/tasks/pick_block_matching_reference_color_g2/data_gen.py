# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_building_blocks_011": {"has_interactions": True, "object_type": "building_block"},
        "benchmark_building_blocks_073": {"has_interactions": True, "object_type": "building_block"},
        "benchmark_building_blocks_085": {"has_interactions": True, "object_type": "building_block"},
        "benchmark_building_blocks_089": {"has_interactions": True, "object_type": "building_block"},
        "benchmark_coaster_000": {"has_interactions": False, "object_type": "coaster"},
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

    ref_id = None
    ref_color = None
    coaster_id = None
    candidates = []

    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        usd = obj_info.get("usd", "")
        if "coaster" in usd or "coaster" in keywords:
            coaster_id = obj_id
            continue
        if "building_blocks" not in usd:
            continue
        if "reference_block" in keywords or "on_coaster" in keywords:
            ref_id = obj_id
            for kw in keywords:
                if kw in ["red", "blue", "yellow", "purple"]:
                    ref_color = kw
                    break
        elif "candidate_block" in keywords:
            block_color = None
            for kw in keywords:
                if kw in ["red", "blue", "yellow", "purple"]:
                    block_color = kw
                    break
            candidates.append({"id": obj_id, "color": block_color, "keywords": keywords})

    if not ref_id or not ref_color:
        print("Reference block not found")
        return False

    matching_id = None
    for c in candidates:
        if c["color"] == ref_color or "matching" in c.get("keywords", []):
            matching_id = c["id"]
            break

    if not matching_id:
        print("No matching candidate found")
        return False

    color_zh = {"red": "红色", "blue": "蓝色", "yellow": "黄色", "purple": "紫色"}.get(ref_color, ref_color)
    description_json = {
        "task_name": f"pick up the {ref_color} block matching the reference on the coaster",
        "english_task_name": f"pick up the {ref_color} block matching the reference on the coaster",
        "init_scene_text": "A reference block on a coaster and several candidate blocks of different colors are on the table."
    }
    robot.client.set_task_task_basic_info(description_json)

    match_pos, _ = robot.get_object_pose(matching_id)
    arm = robot.judge_arm_to_use(match_pos)

    unmovable = [table_id, ref_id] + ([coaster_id] if coaster_id else []) + [c["id"] for c in candidates if c["id"] != matching_id]

    grasp_config = GraspConfig(
        unmovable_objects=unmovable,
        action_description={"zh": f"拿起{color_zh}积木", "en": f"picks up the {ref_color} block"},
        pick_up_distance=0.15,
    )

    success = robot.pick_actor(
        actor=matching_id,
        arm=arm,
        grasp_config=grasp_config,
    )

    robot.back_to_origin(arm=arm)
    return success
