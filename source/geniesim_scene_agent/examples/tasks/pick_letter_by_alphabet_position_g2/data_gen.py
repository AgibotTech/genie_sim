# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import random

ASSET_TO_DIGIT = {f"benchmark_building_blocks_{34+i:03d}": i for i in range(10)}
UPPERCASE_LETTERS = {chr(65+i): f"benchmark_letter_{26+i:03d}" for i in range(26)}
ASSET_TO_LETTER = {v: k for k, v in UPPERCASE_LETTERS.items()}

def execute_task(robot: RobotAPI):
    scene_info = robot.scene_info
    success = True

    description_json = {
        "task_name": 'Pick a letter by its position in the alphabet',
        "english_task_name": 'Pick a letter by its position in the alphabet',
        "init_scene_text": "The robot is in front of the table with a digit block and letter blocks"
    }
    robot.client.set_task_task_basic_info(description_json)

    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        if obj_info.get("usd", "") == "table_000" or "table" in obj_info.get("keywords", []):
            table_id = obj_id
            break
    if not table_id:
        return False

    digit_value = None
    digit_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        for asset_id, digit in ASSET_TO_DIGIT.items():
            if asset_id in usd:
                digit_value = digit
                digit_id = obj_id
                break
        if digit_id:
            break

    if digit_value is None or digit_value == 0:
        return False

    target_letter = chr(64 + digit_value)

    target_block = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        if obj_id in [table_id, digit_id]:
            continue
        for asset_id, letter in ASSET_TO_LETTER.items():
            if asset_id in usd and letter == target_letter:
                pos, rot = robot.get_object_pose(obj_id)
                size = robot.get_object_size_in_world_frame(obj_id)
                if pos is not None:
                    target_block = {"id": obj_id, "letter": letter, "position": pos, "rotation": rot, "size": size}
                break
        if target_block:
            break

    if not target_block:
        return False

    table_pos, _ = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0

    digit_pos, _ = robot.get_object_pose(digit_id)
    target_x = digit_pos[0] + 0.05
    target_y = digit_pos[1]
    target_z = table_top_z + 0.01 + target_block["size"][2] / 2.0
    target_pos = np.array([target_x, target_y, target_z])

    arm = robot.judge_arm_to_use(target_block["position"])

    rot_mat = quat2mat_wxyz(target_block["rotation"])
    target_poses = []
    for dx in np.linspace(-0.002, 0.002, 3):
        for dy in np.linspace(-0.002, 0.002, 3):
            for dz in [0, 0.003]:
                pose = np.eye(4)
                pose[:3, :3] = rot_mat
                pose[:3, 3] = target_pos + np.array([dx, dy, dz])
                target_poses.append(pose)
    target_poses = np.array(target_poses)

    unmovable = [table_id, digit_id]
    grasp_config = GraspConfig(unmovable_objects=unmovable, is_from_up_side=True, grasp_offset=-0.007,
        action_description={"zh": f"拿起字母{target_letter}", "en": f"picks up letter {target_letter}"})
    place_config = PlaceConfig(unmovable_objects=unmovable,
        action_description={"zh": f"将字母{target_letter}放到数字旁边", "en": f"places letter {target_letter} next to the digit"})

    success = robot.pick_and_place_actor_to_target_pose(actor=target_block["id"], target_pose=target_poses, arm=arm,
        grasp_config=grasp_config, place_config=place_config)

    robot.back_to_origin(arm=arm)
    return success
