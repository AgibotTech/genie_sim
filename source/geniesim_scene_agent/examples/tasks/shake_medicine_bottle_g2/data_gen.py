# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import random

def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_medicine_bottle_000": {"has_interactions": True, "object_type": "medicine_bottle"},
        "table_000": {"has_interactions": False, "object_type": "table"}
    }

    scene_info = robot.scene_info
    interactions = robot.interactions
    success = True

    description_json = {
        "task_name": "pick up the medicine bottle and shake it",
        "english_task_name": "pick up the medicine bottle and shake it",
        "init_scene_text": "A medicine bottle is placed on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    table_id = None
    bottle_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or "table" in keywords:
            table_id = obj_id
        if "medicine_bottle" in usd or "medicine_bottle" in keywords:
            bottle_id = obj_id

    if not table_id or not bottle_id:
        print("Required objects not found")
        return False

    bottle_pos, bottle_rot = robot.get_object_pose(bottle_id)
    arm = robot.judge_arm_to_use(bottle_pos)

    grasp_config = GraspConfig(
        pick_up_distance=0.12,
        unmovable_objects=[table_id],
        action_description={"zh": "拿起药瓶", "en": "picks up the medicine bottle"}
    )
    success = robot.pick_actor(bottle_id, arm=arm, grasp_config=grasp_config)
    if not success:
        return False

    gripper_pose = robot.get_ee_pose(arm)
    shake_x = np.random.uniform(-0.1, 0.0)
    shake_y = 0.0
    shake_z = gripper_pose[2, 3]

    gripper_direction_quaternion = np.array([0, 0, 0.70711, -0.70711]) if arm == "left" else np.array([0, 0, 0.70711, 0.70711])
    base_rot_mat = quat2mat_wxyz(gripper_direction_quaternion)

    shake_poses = []
    for z_off in np.linspace(-0.05, 0.15, 5):
        for y_off in np.linspace(-0.02, 0.02, 5):
            for angle in np.linspace(-15, 15, 5):
                pose = np.eye(4)
                pose[:3, :3] = base_rot_mat
                pose[:3, 3] = [shake_x, shake_y + y_off, shake_z + z_off]
                pose = rotate_along_axis(pose, angle, "z", use_local=False)
                shake_poses.append(pose)
    shake_poses = np.array(shake_poses)

    def bottle_in_hand_checker():
        bottle_p, _ = robot.get_object_pose(bottle_id)
        gripper_p = robot.get_ee_pose(arm)[:3, 3]
        return np.linalg.norm(np.array(bottle_p) - gripper_p) < 0.08

    move_config = MoveConfig(
        extra_checkers=[bottle_in_hand_checker],
        action_description={"zh": "移动到胸前位置", "en": "moves to chest position"}
    )
    success = robot.move_actor_to_target_pose("gripper", shake_poses, arm=arm, move_config=move_config)
    if not success:
        return False

    angle_ranges = []
    for angle in np.linspace(15, 30, 5):
        rad = angle * np.pi / 180
        angle_ranges.append([-rad, rad])
    random.shuffle(angle_ranges)

    shake_config = ShakeConfig(
        action_description={"zh": "摇晃药瓶", "en": "shakes the medicine bottle"},
        extra_checkers=[bottle_in_hand_checker]
    )
    success = robot.shake(actor=bottle_id, arm=arm, horizontal=False, angle_ranges=angle_ranges, times=4, shake_config=shake_config)

    return success
