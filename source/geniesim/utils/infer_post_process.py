# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from copy import deepcopy
from hmac import new
import numpy as np
import time


def relabel_gripper_state(obs, limit):
    state_dict = obs["states"]
    state_dict[14] = min(max(1 - state_dict[14] / limit, 0), 1)
    state_dict[15] = min(max(1 - state_dict[15] / limit, 0), 1)


def relabel_gripper_action(action, limit):
    new_action = np.zeros(2)
    new_action[0] = (1 - action[0]) * limit
    new_action[1] = (1 - action[1]) * limit

    return new_action


def abs_ee_to_abs_joint(ikfk_solver, arm_joint_state, action: np.ndarray):
    abs_eef_action = [action]
    joint_actions = ikfk_solver.eef_actions_to_joint(abs_eef_action, arm_joint_state, [0, 0])
    return joint_actions[0]


def process_action(ikfk_solver, arm_joint_state, action: np.ndarray, type, smooth_alpha=1.0):
    if type == "delta_ee":
        raise ValueError("Delta EE to Abs Joint is not supported")
    elif type == "abs_ee":
        return abs_ee_to_abs_joint(ikfk_solver, arm_joint_state, action)
    elif type == "abs_joint":
        return filter_abs_joint(arm_joint_state, action, smooth_alpha)
    else:
        raise ValueError(f"Failed to process unknown action type: {type}")


def filter_abs_joint(arm_joint_state, action, alpha):
    # return action
    return list((1 - alpha) * np.array(arm_joint_state) + alpha * np.array(action[0:14])) + list(action[14:])
    return list(arm_joint_state) + list(action[14:])
    ret = []
    mft = MovingAVGFilter(list(arm_joint_state), action[:14])
    mft.move(lambda target_joints: ret.append(target_joints))
    ret[0].extend(action[14:])
    return ret[0]


class MovingAVGFilter:
    def __init__(self, qpos, trajectory, alpha=0.03, repeat=1, freq=30, sleep=None):
        self.trajectory = np.array(trajectory)
        self.i = 0
        self.curr_traj = np.array(qpos)
        self.alpha = alpha
        self.repeat = repeat
        self.sleep = 1 / freq / repeat if sleep is None else sleep

    def move(self, fn):
        for r in range(self.repeat):
            self.curr_traj = (1 - self.alpha) * self.curr_traj + self.alpha * self.trajectory[self.i]
            fn(self.curr_traj.tolist())
            time.sleep(self.sleep)
        self.i += 1
