# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import time
import glob
import pickle
import numpy as np
import os
from scipy.spatial.transform import Rotation

from .dummy_env import DummyEnv

from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance

from geniesim.benchmark.tasks.llm_task import LLMTask
from geniesim.utils.name_utils import *
from geniesim.utils.infer_pre_process import *
from geniesim.utils.infer_post_process import *


class PiEnv(DummyEnv):
    def __init__(
        self,
        api_core,
        task_file: str,
        init_task_config,
        need_setup=True,
    ):
        super().__init__(
            api_core,
            task_file,
            init_task_config,
            need_setup,
        )
        self.LIMIT_VAL = 0.8
        self.load_task_setup()

    def load_task_setup(self):
        self.task = LLMTask(self)

    def get_observation(self):
        for i in range(10):
            images = self.data_courier.get_observation_image()
            if images == {}:
                time.sleep(0.1)
            else:
                break
        full_joint_states = self.data_courier.get_joint_state_dict()
        states = []
        if "G1" in self.robot_cfg:
            for name in G1_DUAL_ARM_JOINT_NAMES:
                states.append(full_joint_states[name])
        elif "G2" in self.robot_cfg:
            for name in G2_DUAL_ARM_JOINT_NAMES:
                states.append(full_joint_states[name])
        else:
            raise ValueError(f"Invalid robot cfg: {self.robot_cfg}")

        self.cur_arm = deepcopy(states)

        for name in OMNIPICKER_AJ_NAMES:
            states.append(full_joint_states[name])
        obs = {"images": images, "states": states}
        relabel_gripper_state(obs, self.LIMIT_VAL)
        return obs

    def reset(self):
        self.last_update_time = time.time()
        self.has_done = False
        self.task.reset(self)
        self.robot_joint_indices = self.api_core.get_robot_joint_indices()
        eps = 1e-2
        while True:
            print("Robot reset...")
            init_gripper = [1 - v for v in self.init_gripper]

            # fmt: off
            if self.robot_cfg == "G1_omnipicker":
                self.api_core.set_joint_positions(self.init_arm,joint_indices=[self.robot_joint_indices[v] for v in G1_DUAL_ARM_JOINT_NAMES],is_trajectory=False)
                self.api_core.set_joint_positions(self.init_waist,joint_indices=[self.robot_joint_indices[v] for v in G1_WAIST_JOINT_NAMES],is_trajectory=False)
                self.api_core.set_joint_positions(self.init_head,joint_indices=[self.robot_joint_indices[v] for v in G1_HEAD_JOINT_NAMES],is_trajectory=False)
                self.api_core.set_joint_positions(init_gripper, joint_indices=[self.robot_joint_indices[v] for v in OMNIPICKER_AJ_NAMES], is_trajectory=False)
            elif self.robot_cfg == "G2_omnipicker":
                self.api_core.set_joint_positions(self.init_arm,joint_indices=[self.robot_joint_indices[v] for v in G2_DUAL_ARM_JOINT_NAMES], is_trajectory=False)
                self.api_core.set_joint_positions(self.init_waist, joint_indices=[self.robot_joint_indices[v] for v in G2_WAIST_JOINT_NAMES], is_trajectory=False)
                self.api_core.set_joint_positions(self.init_head,joint_indices=[self.robot_joint_indices[v] for v in G2_HEAD_JOINT_NAMES], is_trajectory=False)
                self.api_core.set_joint_positions(init_gripper, joint_indices=[self.robot_joint_indices[v] for v in OMNIPICKER_AJ_NAMES], is_trajectory=False)
            # fmt: on

            time.sleep(0.1)

            for i in range(10):
                full_joint_states = self.data_courier.get_joint_state_dict()
                if full_joint_states != {}:
                    break
                time.sleep(0.5)

            arm_position, waist_position = [], []
            if self.robot_cfg == "G1_omnipicker":
                for name in G1_DUAL_ARM_JOINT_NAMES:
                    arm_position.append(full_joint_states[name])
                for name in G1_WAIST_JOINT_NAMES:
                    waist_position.append(full_joint_states[name])
            elif self.robot_cfg == "G2_omnipicker":
                for name in G2_DUAL_ARM_JOINT_NAMES:
                    arm_position.append(full_joint_states[name])
                for name in G2_WAIST_JOINT_NAMES:
                    waist_position.append(full_joint_states[name])

            c1 = np.max(np.abs(np.array(arm_position) - np.array(self.init_arm))) < eps
            c2 = np.max(np.abs(np.array(waist_position) - np.array(self.init_waist))) < eps

            if c1 and c2:
                break

        logger.info("Finish reset robot...")
        time.sleep(1)
        self.api_core.reset_env()
        obs = self.get_observation()
        logger.info("Finish reset env...")
        return obs

    def step(self, action):
        self.current_step += 1
        need_update = False
        if self.current_step != 1 and self.current_step % 30 == 0:
            self.task.step(self)
            self.action_update()
            need_update = True

        # fmt: off
        action = process_action(None, self.cur_arm, action, type="abs_joint", smooth_alpha=0.5)
        gripper_action = relabel_gripper_action(action[14:16], self.LIMIT_VAL)
        if self.robot_cfg == "G1_omnipicker":
            self.api_core.set_joint_positions([float(v) for v in action[:14]],joint_indices=[self.robot_joint_indices[v] for v in G1_DUAL_ARM_JOINT_NAMES],is_trajectory=True)
            self.api_core.set_joint_positions([float(v) for v in gripper_action], joint_indices=[self.robot_joint_indices[v] for v in OMNIPICKER_AJ_NAMES], is_trajectory=True)
        elif self.robot_cfg == "G2_omnipicker":
            self.api_core.set_joint_positions([float(v) for v in action[:14]],joint_indices=[self.robot_joint_indices[v] for v in G2_DUAL_ARM_JOINT_NAMES],is_trajectory=True)
            self.api_core.set_joint_positions([float(v) for v in gripper_action], joint_indices=[self.robot_joint_indices[v] for v in OMNIPICKER_AJ_NAMES], is_trajectory=True)
        # fmt: on
        next_obs = self.get_observation()
        if self.data_courier.enable_ros:
            self.data_courier.sim_ros_node.publish_image()
        return next_obs, self.has_done, need_update, self.task.task_progress
