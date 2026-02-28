# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import numpy as np

from .pi_env import PiEnv

from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance

from geniesim.utils.name_utils import *
from geniesim.utils.infer_pre_process import *
from geniesim.utils.infer_post_process import *
from geniesim.utils.ikfk_utils import IKFKSolver
import geniesim.utils.system_utils as system_utils


class AbsPoseEnv(PiEnv):
    """
    AbsPoseEnv extends PiEnv with IK/FK solver support for end-effector control.
    The main difference is in the step() method which uses abs_pose action type.
    """

    def __init__(self, api_core, task_file: str, init_task_config, need_setup=True):
        # Initialize parent class first
        super().__init__(api_core, task_file, init_task_config, need_setup)

        # Initialize IK/FK solver for end-effector control
        self.ikfk_solver = IKFKSolver(
            self.init_arm,
            self.init_head,
            self.init_waist,
            robot_cfg=self.robot_cfg,
        )

    def step(self, action):
        """
        Override step to use IK/FK solver for end-effector control.
        """
        self.current_step += 1
        need_update = False
        # Update task progress every 30 steps
        if self.current_step != 1 and self.current_step % 30 == 0:
            self.task.step(self)
            self.action_update()
            need_update = True

        # fmt: off
        action = process_action(self.ikfk_solver, self.cur_arm, action, type="abs_ee")
        gripper_action = relabel_gripper_action(action[14:16], self.LIMIT_VAL)
        if self.robot_cfg == "G1_omnipicker":
            self.api_core.set_joint_positions([float(v) for v in action[:14]],joint_indices=[self.robot_joint_indices[v] for v in G1_DUAL_ARM_JOINT_NAMES],is_trajectory=True)
            self.api_core.set_joint_positions([float(v) for v in gripper_action], joint_indices=[self.robot_joint_indices[v] for v in OMNIPICKER_AJ_NAMES], is_trajectory=True)
        elif self.robot_cfg == "G2_omnipicker":
            self.api_core.set_joint_positions([float(v) for v in action[:14]],joint_indices=[self.robot_joint_indices[v] for v in G2_DUAL_ARM_JOINT_NAMES],is_trajectory=True)
            self.api_core.set_joint_positions([float(v) for v in gripper_action], joint_indices=[self.robot_joint_indices[v] for v in OMNIPICKER_AJ_NAMES], is_trajectory=True)
            if len(action) > 14: # Including waist control
                self.api_core.set_joint_positions([float(v) for v in action[18:19]],joint_indices=[self.robot_joint_indices[v] for v in G2_WAIST_JOINT_NAMES[0:1]],is_trajectory=True)

        next_obs = self.get_observation()
        return next_obs, self.has_done, need_update, self.task.task_progress
