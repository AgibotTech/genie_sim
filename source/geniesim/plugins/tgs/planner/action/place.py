# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .base import (
    StageTemplate,
    solve_target_gripper_pose,
    simple_check_completion,
    pose_difference,
)

import numpy as np


class PlaceStage(StageTemplate):
    def __init__(
        self,
        active_obj_id,
        passive_obj_id,
        active_element=None,
        passive_element=None,
        target_pose=None,
        extra_params={},
        **kwargs
    ):
        super().__init__(active_obj_id, passive_obj_id, active_element, passive_element)
        self.pre_transform_up = 0.12
        self.place_transform_up = np.array([0, 0, 0.01])
        self.extra_params = {} if extra_params is None else extra_params
        self.use_pre_place = extra_params.get("use_pre_place", False)
        self.generate_substage(target_pose)

    def generate_substage(self, target_pose):
        target_pose_canonical = target_pose
        gripper_cmd = self.extra_params.get("gripper_state", "open")
        pre_place_direction = self.extra_params.get("pre_place_direction", "z")
        num_against = self.extra_params.get("against", 0)
        if self.use_pre_place:
            # moveTo pre-place position
            transform_up = np.eye(4)
            if pre_place_direction == "x":
                transform_up[0, 3] = self.pre_transform_up
                transform_up[2, 3] = 0.02
            elif pre_place_direction == "y":
                transform_up[1, 3] = self.pre_transform_up
            else:
                transform_up[2, 3] = self.pre_transform_up
            self.sub_stages.append([target_pose_canonical, None, transform_up, "AvoidObs"])

            # place
            palce_transform_up = np.eye(4)
            palce_transform_up[:3, 3] = self.place_transform_up
            self.sub_stages.append([target_pose_canonical, gripper_cmd, palce_transform_up, "Simple"])
        else:
            palce_transform_up = np.eye(4)
            palce_transform_up[:3, 3] = self.place_transform_up
            self.sub_stages.append([target_pose_canonical, None, palce_transform_up, "AvoidObs"])

            self.sub_stages.append([None, gripper_cmd, np.eye(4), "Simple"])

    def check_completion(self, objects):
        if self.__len__() == 0:
            return True
        goal_datapack = [self.active_obj_id, self.passive_obj_id] + self.sub_stages[self.step_id]
        succ = True
        if self.step_id == 0:
            succ = simple_check_completion(goal_datapack, objects)
        if succ:
            self.step_id += 1
        return succ
