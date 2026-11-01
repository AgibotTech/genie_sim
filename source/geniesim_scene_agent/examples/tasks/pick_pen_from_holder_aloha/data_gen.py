# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Initialize objects_list based on provided interaction data [1] and scene description
    objects_list = {
        "benchmark_pen_000": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_001": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_002": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_003": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_005": {"has_interactions": True, "object_type": "pen"},
        "benchmark_pen_006": {"has_interactions": True, "object_type": "pen"},
        "benchmark_stationery_021": {"has_interactions": False, "object_type": "pen_holder"},
        "table_000": {"has_interactions": False, "object_type": "table"}
    }

    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "pick up the pen from the pen holder",
        "english_task_name": "pick up the pen from the pen holder",
        "init_scene_text": "The robot is in front of the table, there is a pen holder on the table with a pen inside"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Find objects dynamically
    table_id = None
    pen_holder_id = None
    pen_id = None

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        if usd == "table_000" or any("table" in kw.lower() for kw in keywords):
            table_id = obj_id
        elif usd == "benchmark_stationery_021" or any("pen_holder" in kw.lower() for kw in keywords):
            pen_holder_id = obj_id
        elif "benchmark_pen_" in usd:
            pen_id = obj_id

    if not pen_id:
        print("Pen not found")
        return False
    if not pen_holder_id:
        print("Pen holder not found")
        return False
    if not table_id:
        print("Table not found")
        return False

    print(f"Found pen: {pen_id}, pen holder: {pen_holder_id}, table: {table_id}")

    # 2. Determine which arm to use based on pen position
    pen_pos, _ = robot.get_object_pose(pen_id)
    if pen_pos is None:
        print("Cannot get pen position")
        return False

    arm = robot.judge_arm_to_use(pen_pos)
    print(f"Using arm: {arm}")

    # 3. Prepare grasp configuration
    # Interaction data [1] confirms the pen assets have a 'default' grasp primitive
    # We protect the table and pen holder from moving during the grasp
    unmovable_objects = [table_id, pen_holder_id]

    grasp_config = GraspConfig(
        grasp_lower_percentile=70,
        passive_primitive="default",  # Use 'default' primitive from interaction data [1]
        pick_up_distance=0.15,         # Lift 15cm to ensure it clears the pen holder
        unmovable_objects=unmovable_objects,
        action_description={"zh": "拿起笔筒里的笔", "en": "picks up the pen from the pen holder"}
    )

    # 4. Execute the pick action
    success = robot.pick_actor(pen_id, arm=arm, grasp_config=grasp_config)

    if success:
        print(f"Successfully picked up pen {pen_id}")
    else:
        print(f"Failed to pick up pen {pen_id}")

    return success
