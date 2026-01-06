# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .base import (
    StageTemplate,
    simple_check_completion,
    pose_difference,
)

import numpy as np


class PickStage(StageTemplate):
    def __init__(
        self, active_obj_id, passive_obj_id, active_element, passive_element, grasp_pose, extra_params=None, **kwargs
    ):
        super().__init__(active_obj_id, passive_obj_id, active_element, passive_element)
        self.grasp_pose_canonical = grasp_pose
        self.use_pre_grasp = False

        self.extra_params = {} if extra_params is None else extra_params
        self.pick_up_step = 999
        self.generate_substage(grasp_pose)

    def generate_substage(self, grasp_pose):
        pick_up_distance = self.extra_params.get("pick_up_distance", 0.15)
        pick_up_type = self.extra_params.get("pick_up_type", "Simple")

        if self.use_pre_grasp:
            pre_grasp_distance = self.extra_params.get("pre_grasp_distance", 0.08)
            # sub-stage-0   moveTo pregrasp pose
            pre_pose = np.array(
                [
                    [1.0, 0, 0, 0],
                    [0, 1.0, 0, 0],
                    [0, 0, 1.0, -pre_grasp_distance],
                    [0, 0, 0, 1],
                ]
            )
            pre_grasp_pose = grasp_pose @ pre_pose
            self.sub_stages.append([pre_grasp_pose, None, np.eye(4), "AvoidObs"])
            # sub-stage-1   moveTo grasp pose
            self.sub_stages.append([grasp_pose, "close", np.eye(4), "Simple"])
            self.pick_up_step = 2
        else:
            # sub-stage-0   moveTo grasp pose
            self.sub_stages.append([grasp_pose, "close", np.eye(4), "AvoidObs"])
            self.pick_up_step = 1

        # pick-up
        gripper_action = None
        motion_type = pick_up_type
        transform_up = np.eye(4)
        transform_up[2, 3] = pick_up_distance
        self.sub_stages.append([grasp_pose, gripper_action, transform_up, motion_type])

    def check_completion(self, objects):
        if self.__len__() == 0:
            return True

        goal_datapack = [self.active_obj_id, self.passive_obj_id] + self.sub_stages[self.step_id]

        succ = True
        if self.step_id < self.pick_up_step:
            succ = simple_check_completion(goal_datapack, objects, is_grasped=True)

        if succ:
            self.step_id += 1
        return succ


class GraspStage(PickStage):
    def generate_substage(self, grasp_pose):
        motion_type = self.extra_params.get("move_type", "AvoidObs")

        if self.extra_params.get("use_pre_grasp", False):
            pre_pose = np.array(
                [
                    [1.0, 0, 0, 0],
                    [0, 1.0, 0, 0],
                    [0, 0, 1.0, -0.08],
                    [0, 0, 0, 1],
                ]
            )
            pre_grasp_pose = grasp_pose @ pre_pose
            gripper_action = None

            self.sub_stages.append([pre_grasp_pose, None, np.eye(4), motion_type])
        # sub-stage-1   moveTo grasp pose
        self.sub_stages.append([grasp_pose, "close", np.eye(4), motion_type])

    def check_completion(self, objects):
        if self.__len__() == 0:
            return True

        goal_datapack = [self.active_obj_id, self.passive_obj_id] + self.sub_stages[self.step_id]

        pos_threshold = self.extra_params.get("position_threshold", 0.06)
        succ = simple_check_completion(goal_datapack, objects, pos_threshold=pos_threshold, is_grasped=True)

        if self.step_id >= 2:
            gripper_pose = objects["gripper"].obj_pose
            grasp_pose = objects[self.passive_obj_id].obj_pose @ self.grasp_pose_canonical
            pos_diff, _ = pose_difference(gripper_pose, grasp_pose)
            grasp_succ = pos_diff < pos_threshold
            succ = succ and grasp_succ

        if succ:
            self.step_id += 1
        return succ


class HookStage(PickStage):
    def generate_substage(self, grasp_pose, *args, **kwargs):
        self.sub_stages.append([grasp_pose, None, np.eye(4), "AvoidObs"])
