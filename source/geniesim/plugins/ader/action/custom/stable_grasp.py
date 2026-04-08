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


def _rotation_matrix_to_angle_rad(R: np.ndarray) -> float:
    """Extract rotation angle (radians) from 3x3 rotation matrix."""
    trace = np.trace(R)
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    return float(np.arccos(cos_angle))


class StableGrasp(EvaluateAction):
    def __init__(
        self,
        env,
        obj_name,
        gripper_id,
        distance_threshold: float = 0.1,
        pose_diff_pos_threshold: float = 0.02,
        pose_diff_rot_threshold_rad: float = 0.1,
        min_movement_threshold: float = 0.05,
    ):
        super().__init__(env)
        # Parse placeholder for obj_name and gripper_id
        self._holder_name, self._obj_name = self.placeholder_sparser(obj_name)
        self._holder_id, self._gripper_id = self.placeholder_sparser(gripper_id)

        # Threshold parameters
        self.distance_threshold = distance_threshold  # Distance (m) between gripper and object
        self.pose_diff_pos_threshold = pose_diff_pos_threshold  # Position delta diff (m)
        self.pose_diff_rot_threshold_rad = pose_diff_rot_threshold_rad  # Rotation delta diff (rad)
        self.min_movement_threshold = min_movement_threshold  # Min movement to rule out stationary (m)
        self.required_frames = 2  # Required consecutive stable frames

        # Status record
        self._done_flag = False
        self._pass_frame = 0  # Consecutive stable frame count
        self.success_detected = False
        self._prev_gripper_pose: np.ndarray | None = None
        self._prev_obj_pose: np.ndarray | None = None

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

    def check_stable_grasp(
        self,
        gripper_pose: np.ndarray,
        obj_pose: np.ndarray,
        prev_gripper_pose: np.ndarray,
        prev_obj_pose: np.ndarray,
    ) -> bool:
        gripper_pos = gripper_pose[:3, 3]
        obj_pos = obj_pose[:3, 3]
        gripper_R = gripper_pose[:3, :3]
        obj_R = obj_pose[:3, :3]

        prev_gripper_pos = prev_gripper_pose[:3, 3]
        prev_obj_pos = prev_obj_pose[:3, 3]
        prev_gripper_R = prev_gripper_pose[:3, :3]
        prev_obj_R = prev_obj_pose[:3, :3]

        # 1. Distance constraint: gripper center to object center
        distance = np.linalg.norm(gripper_pos - obj_pos)
        if distance >= self.distance_threshold:
            return False

        # 2. Delta pose: change from prev frame to current frame
        delta_obj_pos = obj_pos - prev_obj_pos
        delta_gripper_pos = gripper_pos - prev_gripper_pos
        delta_obj_R = obj_R @ prev_obj_R.T
        delta_gripper_R = gripper_R @ prev_gripper_R.T

        # 3. Movement check: both gripper and object must be moving (not stationary)
        gripper_movement = np.linalg.norm(delta_gripper_pos)
        obj_movement = np.linalg.norm(delta_obj_pos)
        if gripper_movement < self.min_movement_threshold or obj_movement < self.min_movement_threshold:
            return False

        # 4. Diff between object delta_pose and gripper delta_pose
        pos_diff = np.linalg.norm(delta_obj_pos - delta_gripper_pos)
        R_diff = delta_obj_R @ delta_gripper_R.T
        rot_diff_rad = _rotation_matrix_to_angle_rad(R_diff)

        return pos_diff < self.pose_diff_pos_threshold and rot_diff_rad < self.pose_diff_rot_threshold_rad

    def update(self, delta_time: float) -> float:
        # If success has already been detected, return directly
        if self._done_flag:
            return super().update(delta_time)

        # Get robot type dynamically
        robot_cfg = getattr(self.env, "robot_cfg", None)
        if robot_cfg is None:
            # Fallback: try to get from init_task_config
            robot_cfg = getattr(self.env, "init_task_config", {}).get("robot_cfg", "G2_omnipicker")

        # Determine robot base prim path (G1 uses /G1, G2 uses /genie)
        # See: G1_omnipicker.json -> base_prim_path: "/G1"
        # See: G2_omnipicker.json -> base_prim_path: "/genie"
        if "G1" in robot_cfg:
            robot_base = "/G1"
        else:
            robot_base = "/genie"

        # Get gripper pose
        link_prim_path = (
            f"{robot_base}/gripper_r_center_link"
            if "right" in self.gripper_id
            else f"{robot_base}/gripper_l_center_link"
        )

        try:
            gripper_pose = self.get_world_pose_matrix(link_prim_path)
        except Exception as e:
            logger.error(f"[StableGrasp] Failed to get gripper pose from {link_prim_path}: {e}")
            return super().update(delta_time)

        obj_pose = self.get_obj_pose(self.obj_name)
        obj_pose = np.array(obj_pose)
        gripper_pose = np.array(gripper_pose)

        # Check stable grasp: need prev frame for delta_pose
        if self._prev_gripper_pose is not None and self._prev_obj_pose is not None:
            if self.check_stable_grasp(gripper_pose, obj_pose, self._prev_gripper_pose, self._prev_obj_pose):
                self._pass_frame += 1
            else:
                self._pass_frame = 0
        # else: first frame or no prev, skip check (do not increment)

        self._prev_gripper_pose = gripper_pose.copy()
        self._prev_obj_pose = obj_pose.copy()

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
            self._prev_gripper_pose = None
            self._prev_obj_pose = None
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            if self.success_detected:
                self.progress_info["SCORE"] = 1
            else:
                self.progress_info["SCORE"] = 0
