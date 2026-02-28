# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import os, json
from geniesim.utils import system_utils
from geniesim.utils.data_courier import DataCourier
from collections import deque


class BasePolicy:
    def __init__(self, task_name="", sub_task_name="") -> None:
        self.task_name = task_name
        self.sub_task_name = sub_task_name
        self.data_courier: DataCourier = None
        self.action_buffer = deque()

    def init_ros_node(self):
        pass

    def need_infer(self):
        return len(self.action_buffer) == 0

    def shutdown(self):
        pass
        # if rclpy.ok():
        #     self.sim_ros_node.destroy_node()
        #     rclpy.shutdown()

        # if self.spin_thread.is_alive():
        #     self.spin_thread.join(timeout=5)

    def reset(self):
        """Called at the beginning of an episode."""
        pass

    def set_robot(self, robot, cam_dict):
        pass

    def act(self, observations, **kwargs) -> np.ndarray:
        """Act based on the observations."""
        pass

    def set_data_courier(self, data_courier):
        self.data_courier = data_courier

    def load_robot_task_config(self):
        # fmt: off
        task_path = os.path.join(system_utils.benchmark_conf_path(), "eval_tasks", f"{self.task_name}.json")
        with open(task_path, "r") as f:
            self.task_content = json.load(f)

        self.robot_cfg_file = self.task_content["robot"]["robot_cfg"]
        robot_cfg_path = os.path.join(system_utils.app_root_path(), "robot_cfg", self.robot_cfg_file)
        # fmt: on
        with open(robot_cfg_path, "r") as f:
            self.robot_cfg = json.load(f)

        self.gripper_control_joint = self.robot_cfg["gripper"]["gripper_control_joint"]
        self.robot_name = self.robot_cfg["robot"]["robot_name"]
        self.camera_list = self.task_content["recording_setting"]["camera_list"]


class RandomPolicy(BasePolicy):
    def __init__(self, action_space=1):
        self.action_space = action_space

    def act(self, observations, **kwargs):
        action = np.random.uniform(low=-1, high=1, size=(self.action_dim,))
        return action

    @classmethod
    def get_obs_mode(cls, env_id: str) -> str:
        return "rgbd"

    @classmethod
    def get_control_mode(cls, env_id: str) -> str:
        return None
