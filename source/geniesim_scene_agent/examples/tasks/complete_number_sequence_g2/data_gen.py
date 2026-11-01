# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import random

DIGIT_TO_ASSET = {
    0: "benchmark_building_blocks_034", 1: "benchmark_building_blocks_035",
    2: "benchmark_building_blocks_036", 3: "benchmark_building_blocks_037",
    4: "benchmark_building_blocks_038", 5: "benchmark_building_blocks_039",
    6: "benchmark_building_blocks_040", 7: "benchmark_building_blocks_041",
    8: "benchmark_building_blocks_042", 9: "benchmark_building_blocks_043",
}
ASSET_TO_DIGIT = {v: k for k, v in DIGIT_TO_ASSET.items()}

def execute_task(robot: RobotAPI):
    objects_list = {f"benchmark_building_blocks_{34+i:03d}": {"has_interactions": True, "object_type": "digit_block", "digit": i} for i in range(10)}
    objects_list["table_000"] = {"has_interactions": False, "object_type": "table"}

    scene_info = robot.scene_info
    success = True

    description_json = {
        "task_name": 'Complete a number sequence',
        "english_task_name": 'Complete a number sequence',
        "init_scene_text": "The robot is in front of the table with digit blocks forming an incomplete sequence"
    }
    robot.client.set_task_task_basic_info(description_json)

    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        if obj_info.get("usd", "") == "table_000" or "table" in obj_info.get("keywords", []):
            table_id = obj_id
            break
    if not table_id:
        return False

    table_pos, _ = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0

    in_sequence = []
    scattered = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        for asset_id, digit in ASSET_TO_DIGIT.items():
            if asset_id in usd:
                pos, rot = robot.get_object_pose(obj_id)
                size = robot.get_object_size_in_world_frame(obj_id)
                entry = {"id": obj_id, "digit": digit, "position": pos, "rotation": rot, "size": size, "keywords": keywords}
                if "missing" in keywords or "scattered" in keywords:
                    scattered.append(entry)
                else:
                    in_sequence.append(entry)
                break

    if not scattered:
        return False

    missing_block = scattered[0]
    in_sequence.sort(key=lambda x: x["position"][1], reverse=True)

    seq_digits = [b["digit"] for b in in_sequence]
    missing_digit = missing_block["digit"]

    all_digits = sorted(seq_digits + [missing_digit])
    missing_position_idx = all_digits.index(missing_digit)

    if in_sequence:
        row_x = in_sequence[0]["position"][0]
        start_y = max(b["position"][1] for b in in_sequence)
        if len(in_sequence) > 1:
            spacing = abs(in_sequence[0]["position"][1] - in_sequence[1]["position"][1])
        else:
            spacing = 0.05
    else:
        row_x = -0.08
        start_y = 0.1
        spacing = 0.05

    target_y = start_y - missing_position_idx * spacing
    target_z = table_top_z + 0.01 + missing_block["size"][2] / 2.0
    target_pos = np.array([row_x, target_y, target_z])

    arm = robot.judge_arm_to_use(missing_block["position"])

    target_rot_matrix = quat2mat_wxyz(missing_block["rotation"])
    target_poses = []
    for dx in np.linspace(-0.002, 0.002, 3):
        for dy in np.linspace(-0.002, 0.002, 3):
            for dz in [0, 0.003]:
                pose = np.eye(4)
                pose[:3, :3] = target_rot_matrix
                pose[:3, 3] = target_pos + np.array([dx, dy, dz])
                target_poses.append(pose)
    target_poses = np.array(target_poses)

    unmovable = [table_id] + [b["id"] for b in in_sequence]
    grasp_config = GraspConfig(
        unmovable_objects=unmovable,
        is_from_up_side=True, grasp_offset=-0.007,
        action_description={"zh": f"拿起缺失的数字{missing_digit}", "en": f"picks up the missing digit {missing_digit}"}
    )
    place_config = PlaceConfig(
        unmovable_objects=unmovable,
        action_description={"zh": f"将数字{missing_digit}放到序列中正确的位置", "en": f"places digit {missing_digit} at the correct position in the sequence"}
    )

    success = robot.pick_and_place_actor_to_target_pose(
        actor=missing_block["id"], target_pose=target_poses, arm=arm,
        grasp_config=grasp_config, place_config=place_config
    )

    robot.back_to_origin(arm=arm)
    return success
