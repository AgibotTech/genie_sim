# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import random

DIGIT_TO_ASSET = {
    0: "benchmark_building_blocks_034",
    1: "benchmark_building_blocks_035",
    2: "benchmark_building_blocks_036",
    3: "benchmark_building_blocks_037",
    4: "benchmark_building_blocks_038",
    5: "benchmark_building_blocks_039",
    6: "benchmark_building_blocks_040",
    7: "benchmark_building_blocks_041",
    8: "benchmark_building_blocks_042",
    9: "benchmark_building_blocks_043",
}
ASSET_TO_DIGIT = {v: k for k, v in DIGIT_TO_ASSET.items()}

def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_building_blocks_034": {"has_interactions": True, "object_type": "digit_block", "digit": 0},
        "benchmark_building_blocks_035": {"has_interactions": True, "object_type": "digit_block", "digit": 1},
        "benchmark_building_blocks_036": {"has_interactions": True, "object_type": "digit_block", "digit": 2},
        "benchmark_building_blocks_037": {"has_interactions": True, "object_type": "digit_block", "digit": 3},
        "benchmark_building_blocks_038": {"has_interactions": True, "object_type": "digit_block", "digit": 4},
        "benchmark_building_blocks_039": {"has_interactions": True, "object_type": "digit_block", "digit": 5},
        "benchmark_building_blocks_040": {"has_interactions": True, "object_type": "digit_block", "digit": 6},
        "benchmark_building_blocks_041": {"has_interactions": True, "object_type": "digit_block", "digit": 7},
        "benchmark_building_blocks_042": {"has_interactions": True, "object_type": "digit_block", "digit": 8},
        "benchmark_building_blocks_043": {"has_interactions": True, "object_type": "digit_block", "digit": 9},
        "table_000": {"has_interactions": False, "object_type": "table"},
    }

    scene_info = robot.scene_info
    success = True

    description_json = {
        "task_name": 'Arrange digit blocks to form the largest number',
        "english_task_name": 'Arrange digit blocks to form the largest number',
        "init_scene_text": "The robot is in front of the table, there are digit blocks on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        if usd == "table_000" or "table" in obj_info.get("keywords", []):
            table_id = obj_id
            break
    if not table_id:
        return False

    table_pos, _ = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0

    digit_blocks = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        for asset_id, digit in ASSET_TO_DIGIT.items():
            if asset_id in usd:
                pos, rot = robot.get_object_pose(obj_id)
                size = robot.get_object_size_in_world_frame(obj_id)
                if pos is not None and size is not None:
                    digit_blocks.append({"id": obj_id, "usd": usd, "digit": digit, "position": pos, "rotation": rot, "size": size})
                break

    if len(digit_blocks) < 3:
        return False


    digit_blocks.sort(key=lambda x: x["digit"], reverse=True)

    target_x = random.uniform(-0.05, 0.0)
    start_y = 0.08
    spacing = 0.05

    last_arm = ""
    placed = []

    for idx, block in enumerate(digit_blocks):
        b_id = block["id"]
        b_pos = block["position"]

        arm = robot.judge_arm_to_use(b_pos)
        if last_arm != arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        target_y = start_y - idx * spacing
        target_z = table_top_z + 0.01 + block["size"][2] / 2.0
        target_pos = np.array([target_x, target_y, target_z])

        target_rot_matrix = quat2mat_wxyz(block["rotation"])
        target_poses = []
        for dx in np.linspace(-0.002, 0.002, 3):
            for dy in np.linspace(-0.002, 0.002, 3):
                for dz in [0, 0.003]:
                    pose = np.eye(4)
                    pose[:3, :3] = target_rot_matrix
                    pose[:3, 3] = target_pos + np.array([dx, dy, dz])
                    target_poses.append(pose)
        target_poses = np.array(target_poses)

        unmovable = [table_id] + [b["id"] for b in placed]
        grasp_config = GraspConfig(
            unmovable_objects=unmovable,
            is_from_up_side=True, grasp_offset=-0.007,
            action_description={"zh": f"拿起数字{block['digit']}", "en": f"picks up digit {block['digit']}"}
        )
        place_config = PlaceConfig(
            unmovable_objects=unmovable,
            action_description={"zh": f"将数字{block['digit']}放到第{idx+1}位（从左到右）", "en": f"places digit {block['digit']} at position {idx+1} from left to right"}
        )

        cur_success = robot.pick_and_place_actor_to_target_pose(
            actor=b_id, target_pose=target_poses, arm=arm,
            grasp_config=grasp_config, place_config=place_config
        )
        if not cur_success:
            success = False
        else:
            placed.append(block)

    if last_arm != "":
        robot.back_to_origin(arm=last_arm)


    return success
