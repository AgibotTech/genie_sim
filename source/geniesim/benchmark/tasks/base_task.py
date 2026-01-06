# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from abc import abstractmethod
from geniesim.plugins.ader import AderTask
from geniesim.utils.system_utils import benchmark_task_definitions_path

# PATHS
EVAL_CONFIGS_PATH = benchmark_task_definitions_path()


class BaseTask(AderTask):

    def __init__(self, env):
        super().__init__(env, EVAL_CONFIGS_PATH)
        self.config = env.task_info
        self.instructions = [""]
        self.env = env

    def set_task(self, episode_id):
        self.current_episode_id = episode_id

    def get_instruction(self):
        return self.instructions

    def set_instruction(self, instruction):
        self.instructions[0] = instruction

    @abstractmethod
    def reset_scene(self, env):
        """
        Task-specific scene reset

        :param env: environment instance
        """
        raise NotImplementedError()

    @abstractmethod
    def reset_agent(self, env):
        """
        Task-specific agent reset

        :param env: environment instance
        """
        raise NotImplementedError()

    def reset_variables(self, env):
        """
        Task-specific variable reset

        :param env: environment instance
        """
        return

    def reset(self, env):
        super().reset(env)
        self.reset_variables(env)

    @abstractmethod
    def get_task_obs(self, env):
        """
        Get task-specific observation

        :param env: environment instance
        :return: task-specific observation (numpy array)
        """
        raise NotImplementedError()

    def step(self, env):
        """
        Perform task-specific step for every timestep

        :param env: environment instance
        """
        return
