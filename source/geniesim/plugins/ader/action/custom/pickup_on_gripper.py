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


class PickUpOnGripper(EvaluateAction):
    def __init__(self, env, obj_name, gripper_id):
        super().__init__(env)
        self._holder_name, self._obj_name = self.placeholder_sparser(obj_name)
        self._done_flag = False
        self._holder_id, self._gripper_id = self.placeholder_sparser(gripper_id)

        # Threshold Parameters
        self.z_threshold = 0.02
        self.distance_threshold = 0.1

        # Status record
        self.initial_z = None
        self.success_detected = False

    @property
    def obj_name(self) -> bool:
        if self._holder_name:
            return getattr(self, self._obj_name)
        return self._obj_name

    @property
    def gripper_id(self) -> bool:
        if self._holder_id:
            return getattr(self, self._gripper_id)
        return self._gripper_id

    def check_pickup_success(self, gripper_pose, obj_pose):
        # Record initial Z-coordinate
        current_z = obj_pose[2, 3]
        if self.initial_z is None:
            self.initial_z = current_z
            return False

        # Check 1: Is the object lifted?
        z_diff = current_z - self.initial_z
        if z_diff <= self.z_threshold:
            return False

        # Check 2: Distance between object and gripper
        gripper_pos = gripper_pose[:3, 3]
        obj_pos = obj_pose[:3, 3]
        distance = np.linalg.norm(gripper_pos - obj_pos)

        # If the object is lifted and close to the gripper, consider it a successful grasp
        if distance < self.distance_threshold:
            return True

        return False

    def update(self, delta_time: float) -> float:
        # If success has already been detected, return directly
        if self._done_flag:
            return super().update(delta_time)

        link_prim_path = "/G1/gripper_r_center_link" if "right" in self.gripper_id else "/G1/gripper_l_center_link"
        current_gripper_pose = self.get_world_pose_matrix(link_prim_path)
        current_obj_pose = self.get_obj_pose(self.obj_name)

        # Single frame detection: success if any single frame meets the condition
        if self.check_pickup_success(current_gripper_pose, current_obj_pose):
            self._done_flag = True
            self.success_detected = True
            logger.info(f"[PickUpOnGripper] Grasp detected successfully: {self.obj_name}")

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag and self.success_detected:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [PickUpOnGripper] {self.gripper_id}, obj: {self.obj_name}, evt: {event.value}")

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            # Simplified scoring: 1 point for success, 0 for failure
            if self.success_detected:
                self.progress_info["SCORE"] = 1
            else:
                self.progress_info["SCORE"] = 0
