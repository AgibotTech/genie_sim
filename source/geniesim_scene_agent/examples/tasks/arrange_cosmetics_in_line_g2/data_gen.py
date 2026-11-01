# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import random

def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_cosmetics_000": {"has_interactions": True, "object_type": "cosmetics"},
        "benchmark_cosmetics_001": {"has_interactions": True, "object_type": "cosmetics"},
        "benchmark_cosmetics_002": {"has_interactions": True, "object_type": "cosmetics"},
        "benchmark_cosmetics_003": {"has_interactions": True, "object_type": "cosmetics"},
        "benchmark_cosmetics_004": {"has_interactions": True, "object_type": "cosmetics"},
        "benchmark_cosmetics_005": {"has_interactions": True, "object_type": "cosmetics"},
        "benchmark_cosmetics_006": {"has_interactions": True, "object_type": "cosmetics"},
        "benchmark_cosmetics_007": {"has_interactions": True, "object_type": "cosmetics"},
        "benchmark_cosmetics_008": {"has_interactions": True, "object_type": "cosmetics"},
        "benchmark_cosmetics_009": {"has_interactions": True, "object_type": "cosmetics"},
        "table_000": {"has_interactions": False, "object_type": "table"},
    }

    scene_info = robot.scene_info
    success = True

    description_json = {
        "task_name": 'Arrange cosmetics in a line',
        "english_task_name": 'Arrange cosmetics in a line',
        "init_scene_text": "The robot is in front of a table with 3-5 cosmetics randomly placed on it"
    }
    robot.client.set_task_task_basic_info(description_json)

    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        if obj_info.get("usd", "") == "table_000":
            table_id = obj_id
            break

    if not table_id:
        return False

    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0

    cosmetics = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        if usd.startswith("benchmark_cosmetics_"):
            pos, rot = robot.get_object_pose(obj_id)
            size = robot.get_object_size_in_world_frame(obj_id)
            cosmetics.append({
                "id": obj_id,
                "usd": usd,
                "position": pos,
                "rotation": rot,
                "size": size,
            })

    if len(cosmetics) < 2:
        return False

    cosmetics.sort(key=lambda c: c["position"][1], reverse=True)

    target_x = random.uniform(-0.05, 0.0)
    start_y = 0.15
    spacing = 0.06
    target_positions = []
    for i in range(len(cosmetics)):
        ty = start_y - i * spacing
        target_positions.append(np.array([target_x, ty, table_top_z + 0.01]))

    last_arm = ""
    for idx, cosmetic in enumerate(cosmetics):
        c_id = cosmetic["id"]
        c_pos = cosmetic["position"]
        c_rot = cosmetic["rotation"]
        c_size = cosmetic["size"]

        arm = robot.judge_arm_to_use(c_pos)
        if last_arm != arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        initial_rot_matrix = quat2mat_wxyz(c_rot)
        target_xy = target_positions[idx]
        target_z = target_xy[2] + c_size[2] / 2.0

        rotation_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        target_poses = []
        for dx in [-0.002, 0, 0.002]:
            for dy in [-0.002, 0, 0.002]:
                for dz in [0, 0.003]:
                    position = np.array([target_xy[0] + dx, target_xy[1] + dy, target_z + dz])
                    for angle in rotation_angles:
                        rot_z = euler2mat([0, 0, angle])
                        final_rot = np.dot(rot_z, initial_rot_matrix)
                        pose = np.eye(4)
                        pose[:3, :3] = final_rot
                        pose[:3, 3] = position
                        target_poses.append(pose)

        target_poses = np.array(target_poses)

        unmovable = [table_id] + [c["id"] for c in cosmetics[:idx]]

        def make_checker(obj_id, expected_pos):
            def checker():
                cur_pos, _ = robot.get_object_pose(obj_id)
                if cur_pos is None:
                    return False
                dist = np.linalg.norm(cur_pos[:2] - expected_pos[:2])
                return dist < 0.04
            return checker

        grasp_config = GraspConfig(
            unmovable_objects=unmovable,
            action_description={"zh": "拿起化妆品", "en": "picks up the cosmetic"}
        )

        place_config = PlaceConfig(
            unmovable_objects=unmovable,
            extra_checkers=[make_checker(c_id, target_xy)],
            action_description={"zh": f"将化妆品放到第{idx+1}个位置", "en": f"places the cosmetic at position {idx+1}"}
        )

        cur_success = robot.pick_and_place_actor_to_target_pose(
            actor=c_id,
            target_pose=target_poses,
            arm=arm,
            grasp_config=grasp_config,
            place_config=place_config
        )

        if not cur_success:
            success = False

    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    return success
