# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import os, json
import geniesim.utils.system_utils as system_utils

from .base_task import BaseTask


class LLMTask(BaseTask):
    """
    Point Nav Fixed Task
    The goal is to navigate to a fixed goal position
    """

    def __init__(self, env):
        super(LLMTask, self).__init__(env)
        self.current_episode_id = 0

        instance_id = env.init_task_config["scene"]["scene_instance_id"]
        sub_task_name = env.init_task_config["sub_task_name"]
        path = os.path.join(
            system_utils.benchmark_conf_path(),
            "llm_task",
            sub_task_name,
            str(instance_id),
        )
        try:
            with open(
                os.path.join(path, "instructions.json"),
                "r",
            ) as f:
                instr_info = json.load(f)
                self.instructions = instr_info["instructions"]
        except:
            print(f"No instructions found for {path}")
            self.instructions = [""]

    def set_task(self, episode_id):
        self.current_episode_id = episode_id

    def get_instruction(self):
        if self.instructions[0] != "":
            idx = self.current_episode_id % len(self.instructions)
            return [
                self.instructions[idx]["instruction"],
                self.instructions[idx].get("target", {}).get("id1", ""),
                self.instructions[idx].get("gripper", ""),
            ]
        else:
            return ["default", "", ""]

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset scene objects or floor plane

        :param env: environment instance
        """
        return

    def reset_agent(self, env):
        """
        Task-specific agent reset: land the robot to initial pose, compute initial potential

        :param env: environment instance
        """
        return

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        return np.zeros(0, dtype=np.float32)
