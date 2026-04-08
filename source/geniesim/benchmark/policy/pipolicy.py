# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, time, signal
import json
from typing import Dict
from .base import BasePolicy

from scipy.spatial.transform import Rotation as R


from geniesim.plugins.logger import Logger
import geniesim.utils.system_utils as system_utils
from geniesim.utils.comm.websocket_client import WebsocketClientPolicy
from collections import deque
import numpy as np
import geniesim.utils.system_utils as system_utils
from geniesim.utils.generalization_utils import apply_camera_image_augmentation
import cv2

logger = Logger()  # Create singleton instance

ROOT_DIR = os.environ.get("SIM_REPO_ROOT")

TIMEOUT_SEC = 30


class PiPolicy(BasePolicy):
    def __init__(self, task_name, host_ip, port, sub_task_name="", debug=False, preview=False):
        super().__init__(task_name=task_name, sub_task_name=sub_task_name)
        self.ts_str = time.strftime("%Y%m%d_%H%M", time.localtime(time.time()))
        self.initialized = False
        self.preview = preview
        self.debug = debug
        if not self.preview:
            self.policy = WebsocketClientPolicy(host=host_ip, port=port)
        self.infer_cnt = 0
        self._camera_dirt_cache: Dict[tuple, np.ndarray] = {}
        self._current_episode_idx = 0

    def set_episode_idx(self, idx):
        self._current_episode_idx = idx

    def get_payload(self, obs, task_instruction, gen_config):
        s = np.asarray(obs["states"]).flatten()
        states = np.concatenate([s, np.zeros(32)])[:32]

        eef = obs["eef"]
        left_eef = np.asarray(eef["left"], dtype=np.float64)
        right_eef = np.asarray(eef["right"], dtype=np.float64)

        def _encode_depth(depth_map: np.ndarray, scale: int) -> np.ndarray:
            depth = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
            depth = np.clip(depth * scale, 0, np.iinfo(np.uint16).max)
            return depth.astype(np.uint16)

        if gen_config is not None:
            obs["images"] = apply_camera_image_augmentation(self._camera_dirt_cache, obs["images"], gen_config)

        def _encode_depth(depth_map: np.ndarray, scale: int) -> np.ndarray:
            depth = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
            depth = np.clip(depth * scale, 0, np.iinfo(np.uint16).max)
            return depth.astype(np.uint16)

        payload = {
            "state": states,
            "eef": {
                "left": left_eef.tolist(),
                "right": right_eef.tolist(),
            },
            "images": {
                "top_head": np.transpose(obs["images"]["head"], (2, 0, 1)),
                "hand_left": np.transpose(obs["images"]["left_hand"], (2, 0, 1)),
                "hand_right": np.transpose(obs["images"]["right_hand"], (2, 0, 1)),
            },
            "depth": {
                "top_head": _encode_depth(obs["depth"]["head"], 1000),
                "hand_left": _encode_depth(obs["depth"]["left_hand"], 10000),
                "hand_right": _encode_depth(obs["depth"]["right_hand"], 10000),
            },
            "prompt": task_instruction,
            "task_name": self.sub_task_name,
            "episode_idx": self._current_episode_idx,
        }
        if self.debug:
            logger.debug(f"task_name: {payload['task_name']}")
            logger.debug(f"state: {payload['state']}")
            logger.debug(f"eef: {payload['eef']}")
            logger.debug(f"prompt: {payload['prompt']}")
            cv2.imwrite("head.png", obs["images"]["head"])
            cv2.imwrite("left_hand.png", obs["images"]["left_hand"])
            cv2.imwrite("right_hand.png", obs["images"]["right_hand"])

        if self.preview:
            ts = int(time.time() * 1000)
            debug_dir = os.path.join(ROOT_DIR, "debug_preview")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(debug_dir, f"preview_{self.infer_cnt:04d}_{ts}_head.png"),
                cv2.cvtColor(obs["images"]["head"], cv2.COLOR_BGR2RGB),
            )
            cv2.imwrite(
                os.path.join(debug_dir, f"preview_{self.infer_cnt:04d}_{ts}_left_hand.png"),
                cv2.cvtColor(obs["images"]["left_hand"], cv2.COLOR_BGR2RGB),
            )
            cv2.imwrite(
                os.path.join(debug_dir, f"preview_{self.infer_cnt:04d}_{ts}_right_hand.png"),
                cv2.cvtColor(obs["images"]["right_hand"], cv2.COLOR_BGR2RGB),
            )
            logger.info(f"[Preview] Saved images to {debug_dir}/preview_{self.infer_cnt:04d}_{ts}_*.png")
            self.infer_cnt += 1
            return None

        return payload

    def reset(self):
        self.action_buffer.clear()

    def infer(self, payload):
        try:
            result = self.policy.infer(payload)
            actions = result["actions"]
            n = max(len(actions), 1)
            self.action_buffer = deque(actions, maxlen=n)
            return True

        except Exception as e:
            logger.warning(f"Model inference failed: {str(e)}")
            time.sleep(1)
            return False

    def act(self, observation, **kwargs):
        # return observation["states"]
        if len(self.action_buffer) == 0:
            logger.info("policy.at call model infer")
            task_instruction = kwargs.get("task_instruction", "")
            gen_config = kwargs.get("gen_config")
            logger.info(f"\nInstruction: {task_instruction}\n")
            payload = self.get_payload(observation, task_instruction, gen_config)

            if payload is None:
                return None

            infer_success = self.infer(payload)
            infer_start = time.time()

            while not infer_success:
                infer_success = self.infer(payload)
                if time.time() - infer_start > TIMEOUT_SEC:
                    logger.error("Model inference timeout, please check if the policy server is running normally")
                    os.kill(os.getpid(), signal.SIGINT)
                    return None
            self.infer_cnt += 1

        return self.action_buffer.popleft()
