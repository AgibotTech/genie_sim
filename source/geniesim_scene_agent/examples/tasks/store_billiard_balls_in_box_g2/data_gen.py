# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np

def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_billiards_000": {"has_interactions": True, "object_type": "billiards"},
        "benchmark_billiards_001": {"has_interactions": True, "object_type": "billiards"},
        "benchmark_billiards_002": {"has_interactions": True, "object_type": "billiards"},
        "benchmark_billiards_003": {"has_interactions": True, "object_type": "billiards"},
        "benchmark_billiards_004": {"has_interactions": True, "object_type": "billiards"},
        "benchmark_billiards_005": {"has_interactions": True, "object_type": "billiards"},
        "benchmark_storage_box_000": {"has_interactions": True, "object_type": "storage_box"},
        "table_000": {"has_interactions": False, "object_type": "table"}
    }

    scene_info = robot.scene_info
    interactions = robot.interactions
    success = True

    description_json = {
        "task_name": "put billiard balls into the storage box",
        "english_task_name": "put billiard balls into the storage box",
        "init_scene_text": "Several billiard balls and a storage box are on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    table_id = None
    box_id = None
    ball_ids = []

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or "table" in keywords:
            table_id = obj_id
        elif "storage_box" in usd or "storage_box" in keywords or "container" in keywords:
            box_id = obj_id
        elif "billiard" in usd or "billiard" in keywords:
            ball_ids.append(obj_id)

    if not table_id or not box_id or not ball_ids:
        print("Required objects not found")
        return False

    box_pos, box_rot = robot.get_object_pose(box_id)
    box_size = robot.get_object_size_in_world_frame(box_id)

    for ball_id in ball_ids:
        ball_pos, _ = robot.get_object_pose(ball_id)
        arm = robot.judge_arm_to_use(ball_pos)

        grasp_config = GraspConfig(
            pick_up_distance=0.12,
            unmovable_objects=[table_id, box_id] + [b for b in ball_ids if b != ball_id],
            action_description={"zh": "拿起台球", "en": "picks up the billiard ball"}
        )

        def make_checker(bid):
            def check_in_box():
                cur_pos, cur_rot = robot.get_object_pose(bid)
                b_pos, b_rot = robot.get_object_pose(box_id)
                cur_size = robot.get_object_size_in_local_frame(bid)
                b_size = robot.get_object_size_in_local_frame(box_id)
                from common.base_utils.transform_utils import check_box_containment
                contained, _ = check_box_containment(
                    b_pos, b_rot, b_size, cur_pos, cur_rot, cur_size, overlap_threshold=0.5
                )
                return contained
            return check_in_box

        place_config = PlaceConfig(
            unmovable_objects=[table_id, box_id] + [b for b in ball_ids if b != ball_id],
            extra_checkers=[make_checker(ball_id)],
            action_description={"zh": "将台球放入收纳盒", "en": "places billiard ball into the box"}
        )

        current_success = robot.pick_and_place_actor_to_target_actor(
            actor=ball_id, target_actor=box_id,
            arm=arm, grasp_config=grasp_config, place_config=place_config
        )
        success = success and current_success
        robot.back_to_origin(arm=arm)

    return success
