# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import numpy as np
from collections import deque

from geniesim.utils.data_utils import pose_difference


def simple_check_completion(
    goal,
    objects,
    last_statement=None,
    pos_threshold=0.06,
    angle_threshold=70,
    is_grasped=False,
):
    (
        active_obj_id,
        passive_obj_id,
        target_pose_canonical,
        gripper_action,
        transform_world,
        motion_type,
    ) = goal
    if target_pose_canonical is None:
        return True
    if gripper_action == "open":
        return True

    current_pose_world = objects[active_obj_id].obj_pose
    if len(target_pose_canonical.shape) == 3:
        target_pose_canonical = target_pose_canonical[-1]
        transform_world = transform_world[-1]
    target_pose_world = objects[passive_obj_id].obj_pose @ target_pose_canonical
    if not is_grasped:
        target_pose_world = np.dot(transform_world, target_pose_world)

    pos_diff, angle_diff = pose_difference(current_pose_world, target_pose_world)
    success = (pos_diff < pos_threshold) and (angle_diff < angle_threshold)
    return success


def solve_target_gripper_pose(stage, objects):
    (
        active_obj_ID,
        passive_obj_ID,
        target_pose_canonical,
        gripper_action,
        transform_world,
        motion_type,
    ) = stage

    anchor_pose = objects[passive_obj_ID].obj_pose

    if motion_type == "Trajectory":
        assert len(target_pose_canonical.shape) == 3, "The target_pose should be a list of poses"
        target_pose = anchor_pose[np.newaxis, ...] @ target_pose_canonical
        target_pose = transform_world @ target_pose
    else:
        target_pose = anchor_pose @ target_pose_canonical
        target_pose = transform_world @ target_pose
    assert "gripper" in objects, "The gripper should be the first one in the object list"
    current_gripper_pose = objects["gripper"].obj_pose

    if active_obj_ID == "gripper":
        target_gripper_pose = target_pose
    else:
        current_obj_pose = objects[active_obj_ID].obj_pose
        gripper2obj = np.linalg.inv(current_obj_pose) @ current_gripper_pose
        if len(target_pose.shape) == 3:
            gripper2obj = gripper2obj[np.newaxis, ...]

        target_obj_pose = target_pose
        target_gripper_pose = target_obj_pose @ gripper2obj

    return target_gripper_pose


class StageTemplate:
    def __init__(self, active_obj_id, passive_obj_id, active_element, passive_element):
        self.active_obj_id = active_obj_id
        self.passive_obj_id = passive_obj_id
        self.active_element = active_element
        self.passive_element = passive_element

        self.last_statement = None
        self.sub_stages = deque()
        self.step_id = 0
        self.extra_params = {}

    def generate_substage(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.sub_stages) - self.step_id

    def get_action(self, objects):
        if self.__len__() == 0:
            return None
        gripper_pose_canonical, gripper_action, transform_world, motion_type = self.sub_stages[self.step_id]

        if motion_type == "local_gripper":
            delta_pose = gripper_pose_canonical
            gripper_pose = objects["gripper"].obj_pose
            target_gripper_pose = gripper_pose @ delta_pose
            motion_type = "Straight"
        else:

            if gripper_pose_canonical is None:
                target_gripper_pose = None
            else:
                goal_datapack = [
                    self.active_obj_id,
                    self.passive_obj_id,
                ] + self.sub_stages[self.step_id]
                target_gripper_pose = solve_target_gripper_pose(goal_datapack, objects)

            last_statement = {
                "objects": copy.deepcopy(objects),
                "target_gripper_pose": target_gripper_pose,
            }
            self.last_statement = last_statement
        return (
            target_gripper_pose,
            motion_type,
            gripper_action,
            self.extra_params.get("arm", "right"),
        )

    def check_completion(self, objects):
        if self.__len__() == 0:
            return True
        goal_datapack = [self.active_obj_id, self.passive_obj_id] + self.sub_stages[self.step_id]
        succ = simple_check_completion(goal_datapack, objects)
        if succ:
            self.step_id += 1
        return succ
