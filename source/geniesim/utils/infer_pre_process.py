# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import cv2


class TaskInfo:
    def __init__(self, task_config, robot_cfg):
        self._task_config = task_config
        self._arm_init_position = self._task_config["init_arm"]
        self._hand_init_position = None
        self._gripper_init_position = self._task_config["init_hand"]

        if "G1_omnipicker" == robot_cfg:
            self._head_init_position = self._task_config["body_state"][:2]
            self._waist_init_position = self._task_config["body_state"][2:4]
        elif "G2_omnipicker" == robot_cfg:
            self._head_init_position = self._task_config["head_state"]
            self._waist_init_position = self._task_config["body_state"]
        else:
            raise ValueError(f"Invalid robot cfg {robot_cfg}")

    def init_pose(self):
        return (
            self._arm_init_position,
            self._head_init_position,
            self._waist_init_position,
            self._hand_init_position,
            self._gripper_init_position,
        )


def add_gaussian_noise(img, mean=0, std=0.1):
    noisy_img = img.astype(np.float32)
    noise = np.random.normal(mean, std, img.shape)
    noisy_img += noise
    noisy_img = np.clip(noisy_img, 0, 255 if img.max() > 1 else 1)  # Automatically determine range
    return noisy_img.astype(img.dtype)


def add_obs_noise(obs, cams):
    for key in cams:
        obs["images"][key] = add_gaussian_noise(obs["images"][key])
    return obs


def crop_obs(obs, cams, shape):
    for key in cams:
        obs[key] = cv2.resize(obs[key], shape)
    return obs
