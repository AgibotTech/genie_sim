# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np

def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_building_blocks_000": {"has_interactions": True, "object_type": "building_blocks"},
        "benchmark_building_blocks_001": {"has_interactions": True, "object_type": "building_blocks"},
        "benchmark_building_blocks_002": {"has_interactions": True, "object_type": "building_blocks"},
        "table_000": {"has_interactions": False, "object_type": "table"}
    }

    scene_info = robot.scene_info
    interactions = robot.interactions
    success = True

    description_json = {
        "task_name": "swap positions of two building blocks",
        "english_task_name": "swap positions of two building blocks",
        "init_scene_text": "Two colored building blocks are placed on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    table_id = None
    block_ids = []

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or "table" in keywords:
            table_id = obj_id
        elif "building_blocks" in usd or "building_block" in keywords or "block_A" in keywords or "block_B" in keywords:
            block_ids.append(obj_id)

    if not table_id or len(block_ids) < 2:
        print("Required objects not found")
        return False

    block_a_id = block_ids[0]
    block_b_id = block_ids[1]

    pos_a, rot_a = robot.get_object_pose(block_a_id)
    pos_b, rot_b = robot.get_object_pose(block_b_id)
    pos_a = np.array(pos_a)
    pos_b = np.array(pos_b)

    arm_a = robot.judge_arm_to_use(pos_a)

    temp_x = (pos_a[0] + pos_b[0]) / 2
    temp_y = 0.0
    temp_z = pos_a[2] + 0.01

    rot_a_mat = quat2mat_wxyz(rot_a)
    temp_poses = []
    for angle in np.linspace(0, 360, 12, endpoint=False):
        for x_off in np.linspace(-0.02, 0.02, 3):
            pose = np.eye(4)
            pose[:3, :3] = rot_a_mat
            pose[:3, 3] = [temp_x + x_off, temp_y, temp_z]
            pose = rotate_along_axis(pose, angle, "z", use_local=False)
            temp_poses.append(pose)
    temp_poses = np.array(temp_poses)

    grasp_config_a = GraspConfig(
        pick_up_distance=0.12,
        unmovable_objects=[table_id, block_b_id],
        action_description={"zh": "拿起积木A", "en": "picks up block A"}
    )
    place_config_temp = PlaceConfig(
        unmovable_objects=[table_id, block_b_id],
        action_description={"zh": "将积木A放到临时位置", "en": "places block A at temporary position"}
    )
    success = robot.pick_and_place_actor_to_target_pose(
        actor=block_a_id, target_pose=temp_poses,
        arm=arm_a, grasp_config=grasp_config_a, place_config=place_config_temp
    )
    if not success:
        return False
    robot.back_to_origin(arm=arm_a)

    arm_b = robot.judge_arm_to_use(pos_b)
    target_b_poses = []
    rot_b_mat = quat2mat_wxyz(rot_b)
    for angle in np.linspace(0, 360, 12, endpoint=False):
        for x_off in np.linspace(-0.02, 0.02, 3):
            pose = np.eye(4)
            pose[:3, :3] = rot_b_mat
            pose[:3, 3] = [pos_a[0] + x_off, pos_a[1], pos_a[2] + 0.01]
            pose = rotate_along_axis(pose, angle, "z", use_local=False)
            target_b_poses.append(pose)
    target_b_poses = np.array(target_b_poses)

    grasp_config_b = GraspConfig(
        pick_up_distance=0.12,
        unmovable_objects=[table_id, block_a_id],
        action_description={"zh": "拿起积木B", "en": "picks up block B"}
    )
    place_config_b = PlaceConfig(
        unmovable_objects=[table_id, block_a_id],
        action_description={"zh": "将积木B放到A的原位置", "en": "places block B at A's original position"}
    )
    success = robot.pick_and_place_actor_to_target_pose(
        actor=block_b_id, target_pose=target_b_poses,
        arm=arm_b, grasp_config=grasp_config_b, place_config=place_config_b
    )
    if not success:
        return False
    robot.back_to_origin(arm=arm_b)

    cur_a_pos, _ = robot.get_object_pose(block_a_id)
    arm_a2 = robot.judge_arm_to_use(cur_a_pos)
    target_a_poses = []
    for angle in np.linspace(0, 360, 12, endpoint=False):
        for x_off in np.linspace(-0.02, 0.02, 3):
            pose = np.eye(4)
            pose[:3, :3] = rot_a_mat
            pose[:3, 3] = [pos_b[0] + x_off, pos_b[1], pos_b[2] + 0.01]
            pose = rotate_along_axis(pose, angle, "z", use_local=False)
            target_a_poses.append(pose)
    target_a_poses = np.array(target_a_poses)

    grasp_config_a2 = GraspConfig(
        pick_up_distance=0.12,
        unmovable_objects=[table_id, block_b_id],
        action_description={"zh": "拿起积木A", "en": "picks up block A"}
    )
    place_config_a = PlaceConfig(
        unmovable_objects=[table_id, block_b_id],
        action_description={"zh": "将积木A放到B的原位置", "en": "places block A at B's original position"}
    )
    success = robot.pick_and_place_actor_to_target_pose(
        actor=block_a_id, target_pose=target_a_poses,
        arm=arm_a2, grasp_config=grasp_config_a2, place_config=place_config_a
    )
    robot.back_to_origin(arm=arm_a2)

    return success
