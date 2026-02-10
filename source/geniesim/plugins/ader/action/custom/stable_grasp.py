# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
import numpy as np
from geniesim.plugins.logger import Logger

logger = Logger()


class StableGrasp(EvaluateAction):
    def __init__(self, env, obj_name, gripper_id):
        super().__init__(env)
        # Parse placeholder for obj_name and gripper_id
        self._holder_name, self._obj_name = self.placeholder_sparser(obj_name)
        self._holder_id, self._gripper_id = self.placeholder_sparser(gripper_id)

        # Threshold parameters
        self.distance_threshold = 0.1  # Distance threshold (meters)
        self.required_frames = 2       # Required consecutive stable frames

        # Status record
        self._done_flag = False
        self._pass_frame = 0  # Consecutive stable frame count
        self.success_detected = False

    @property
    def obj_name(self) -> str:
        if self._holder_name:
            return getattr(self, self._obj_name)
        return self._obj_name

    @property
    def gripper_id(self) -> str:
        if self._holder_id:
            return getattr(self, self._gripper_id)
        return self._gripper_id

    def check_stable_grasp(self, gripper_pos, obj_pos):
        distance = np.linalg.norm(gripper_pos - obj_pos)
        return distance < self.distance_threshold

    def update(self, delta_time: float) -> float:
        # If success has already been detected, return directly
        if self._done_flag:
            return super().update(delta_time)

        # Get robot type dynamically
        robot_cfg = getattr(self.env, 'robot_cfg', None)
        if robot_cfg is None:
            # Fallback: try to get from init_task_config
            robot_cfg = getattr(self.env, 'init_task_config', {}).get('robot_cfg', 'G2_omnipicker')

        # Determine robot base prim path (G1 uses /G1, G2 uses /genie)
        # See: G1_omnipicker.json -> base_prim_path: "/G1"
        # See: G2_omnipicker.json -> base_prim_path: "/genie"
        if "G1" in robot_cfg:
            robot_base = "/G1"
        else:
            robot_base = "/genie"

        # Get gripper pose
        link_prim_path = f"{robot_base}/gripper_r_center_link" if "right" in self.gripper_id else f"{robot_base}/gripper_l_center_link"

        try:
            gripper_pose = self.get_world_pose_matrix(link_prim_path)
        except Exception as e:
            logger.error(f"[StableGrasp] Failed to get gripper pose from {link_prim_path}: {e}")
            return super().update(delta_time)

        obj_pose = self.get_obj_pose(self.obj_name)
        gripper_pos = gripper_pose[:3, 3]
        obj_pos = obj_pose[:3, 3]
        # Check stable grasp condition
        if self.check_stable_grasp(gripper_pos, obj_pos):
            self._pass_frame += 1
        else:
            self._pass_frame = 0

        # Check if consecutive frames reached
        if self._pass_frame >= self.required_frames:
            self._done_flag = True
            self.success_detected = True
            logger.info(f"[StableGrasp] Stable grasp detected: {self.obj_name}.")

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag and self.success_detected:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [StableGrasp] {self.gripper_id}, obj: {self.obj_name}, evt: {event.value}")

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            if self.success_detected:
                self.progress_info["SCORE"] = 1
            else:
                self.progress_info["SCORE"] = 0
