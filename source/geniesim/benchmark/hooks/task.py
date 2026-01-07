# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np

from .hook_base import HookBase
import time


class TaskMetric(HookBase):
    def __init__(self):
        self.satisfied_predicates = []
        self.timesteps = 0

    def start_callback(self, env, _):
        pass

    def step_callback(self, env, _):
        self.timesteps += 1
        self.satisfied_predicates.append(env.task.current_goal_status)

    def end_callback(self, env, _):
        candidate_q_score = []
        for option in env.task.ground_goal_state_options:
            predicate_truth_values = []
            for predicate in option:
                predicate_truth_values.append(predicate.evaluate())
            candidate_q_score.append(np.mean(predicate_truth_values))
        self.final_q_score = np.max(candidate_q_score)

    def gather_results(self):
        return {
            "satisfied_predicates": {
                "timestep": self.satisfied_predicates,
            },
            "q_score": {"final": self.final_q_score},
            "time": {
                "simulator_steps": self.timesteps,
                "simulator_time": self.timesteps,
            },
        }


class TaskHook(HookBase):
    def __init__(self, policy):
        self.policy = policy

    def start_callback(self, env, _):
        init_joint_state = self.policy.reset()
        print("SET INIT POS", init_joint_state)
        if init_joint_state != None:
            env.robot.client.set_joint_positions(init_joint_state, False)

        task_name = env.init_task_config["task"]

    def step_callback(self, env, action):
        pass

    def end_callback(self, env, _):
        pass

    def gather_results(self):
        pass
