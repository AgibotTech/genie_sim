# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from copy import deepcopy
import numpy as np
import time

DEFAULT_ARM_DIM = 14

# Keyed-states flattening order — matches the legacy flat list layout
# (arm + gripper + waist + extra) so the pi server payload is unchanged.
STATE_KEYS = ["left_arm", "right_arm", "left_gripper", "right_gripper", "waist", "head"]


def flatten_states(states):
    """Keyed states dict (PiEnv) or legacy flat list -> flat list."""
    if isinstance(states, dict):
        out = []
        for key in STATE_KEYS:
            out.extend(states.get(key) or [])
        return out
    return list(states)


def get_arm_states(states, arm_dim=DEFAULT_ARM_DIM):
    """Dual-arm joint values from keyed states dict or legacy flat list."""
    if isinstance(states, dict):
        return list(states["left_arm"]) + list(states["right_arm"])
    return list(states[:arm_dim])


def gripper_state_start(cfg):
    """States layout is arm + gripper [+ waist + extra]; gripper starts at len(arm_joints)."""
    return len(cfg.get("arm_joints") or []) or DEFAULT_ARM_DIM


def relabel_gripper_state(obs, limit, start=DEFAULT_ARM_DIM):
    def relabel(v):
        return min(max(limit - v / limit, 0), 1) * 120

    states = obs["states"]
    if isinstance(states, dict):
        states["left_gripper"] = [relabel(v) for v in states["left_gripper"]]
        states["right_gripper"] = [relabel(v) for v in states["right_gripper"]]
        return
    states[start] = relabel(states[start])
    states[start + 1] = relabel(states[start + 1])


def relabel_gripper_action(action, limit):
    new_action = np.zeros(2)
    new_action[0] = (1 - action[0]) * limit
    new_action[1] = (1 - action[1]) * limit

    return new_action


def label_state_omnipicker(obs, cfg):
    relabel_gripper_state(obs, cfg["limit_val"], gripper_state_start(cfg))


def label_state_crsb(obs, cfg):
    scale = cfg["gripper_scale"]
    states = obs["states"]
    if isinstance(states, dict):
        states["left_gripper"] = [v * scale for v in states["left_gripper"]]
        states["right_gripper"] = [v * scale for v in states["right_gripper"]]
        return
    start = gripper_state_start(cfg)
    states[start] *= scale
    states[start + 1] *= scale


def label_state_passthrough(obs, cfg):
    return


def process_gripper_action_relabel(action_slice, cfg):
    g = relabel_gripper_action(action_slice, cfg["limit_val"])
    return [float(v) + cfg["gripper_offset"] for v in g]


def process_gripper_action_crsb(action_slice, cfg):
    scale = cfg["gripper_scale"]
    limit = cfg["limit_val"]
    # Position-driven fingers settle where they touch, so a target equal to the
    # commanded width leaves the drive with ~no error and ~no grip force. Shift
    # every target closed by gripper_shift so the fingers keep pushing inward.
    shift = cfg.get("gripper_shift", 0.0)

    def _map(v):
        return max(0.0, min(float(v) / scale - shift, limit))

    return [_map(v) for v in action_slice]


def process_gripper_action_passthrough(action_slice, cfg):
    return [float(v) + cfg["gripper_offset"] for v in action_slice]


def abs_ee_to_abs_joint(ikfk_solver, arm_joint_state, action: np.ndarray):
    abs_eef_action = [action]
    joint_actions = ikfk_solver.eef_actions_to_joint(abs_eef_action, arm_joint_state, [0, 0])
    return joint_actions[0]


def process_action(ikfk_solver, arm_joint_state, action: np.ndarray, type, smooth_alpha=1.0):
    if type == "delta_ee":
        raise ValueError("Delta EE to Abs Joint is not supported")
    elif type == "abs_pose":
        return abs_ee_to_abs_joint(ikfk_solver, arm_joint_state, action)
    elif type == "abs_joint":
        return filter_abs_joint(arm_joint_state, action, smooth_alpha)
    else:
        raise ValueError(f"Failed to process unknown action type: {type}")


def filter_abs_joint(arm_joint_state, action, alpha):
    return action
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
