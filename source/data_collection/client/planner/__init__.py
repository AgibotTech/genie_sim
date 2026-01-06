# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from abc import abstractmethod


class Planner:
    @abstractmethod
    def deduce_target_pose(self, active_obj, passive_obj, N=1):
        pass

    @abstractmethod
    def plan_trajectory(
        self,
        active_obj,
        target_obj_pose,
        gripper_pose,
        task,
        gripper_id=None,
        ik_checker=None,
    ):
        pass
