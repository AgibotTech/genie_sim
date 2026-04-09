# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np

from .pi_env import PiEnv

from geniesim.plugins.logger import Logger

logger = Logger()

from geniesim.utils.name_utils import *
from geniesim.utils.infer_pre_process import *
from geniesim.utils.infer_post_process import *


class AbsPoseEnv(PiEnv):

    def __init__(self, api_core, task_file: str, init_task_config, need_setup=True):
        super().__init__(api_core, task_file, init_task_config, need_setup)

    def step(self, action):
        self.current_step += 1
        need_update = False
        if self.current_step != 1 and self.current_step % 30 == 0:
            self.task.step(self)
            self.action_update()
            need_update = True

        raw_action = action

        action = process_action(self.ikfk_solver, self.cur_arm, raw_action, type="abs_ee")
        gripper_action = relabel_gripper_action(action[14:16], self.LIMIT_VAL)

        # fmt: off
        self.api_core.set_joint_positions([float(v) for v in action[:14]], joint_indices=[self.robot_joint_indices[v] for v in self.cfg["arm_joints"]], is_trajectory=True)
        self.api_core.set_joint_positions([float(v) for v in gripper_action], joint_indices=[self.robot_joint_indices[v] for v in self.cfg["gripper_joints"]], is_trajectory=True)

        if "G2" in self.robot_cfg and len(raw_action) >= 19:
            self.api_core.set_joint_positions([float(v) for v in raw_action[18:19]], joint_indices=[self.robot_joint_indices[v] for v in self.cfg["waist_joints"][0:1]], is_trajectory=True)
        # fmt: on

        next_obs = self.get_observation()
        if self.data_courier.enable_ros:
            self.data_courier.sim_ros_node.publish_image()
        return next_obs, self.has_done, need_update, self.task.task_progress
