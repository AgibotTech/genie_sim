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


class ChassisAtTarget(EvaluateAction):

    def __init__(self, env, target_str, thresh_str):
        super().__init__(env)

        # Parse target "[x,y,yaw]" -> x, y, yaw
        target_str = target_str.strip("[]")
        target_parts = target_str.split(",")
        self.target_x = float(target_parts[0])
        self.target_y = float(target_parts[1])
        self.target_yaw = float(target_parts[2])

        # Parse threshold "[x_thresh,y_thresh,yaw_thresh]"
        thresh_str = thresh_str.strip("[]")
        thresh_parts = thresh_str.split(",")
        self.x_thresh = float(thresh_parts[0])
        self.y_thresh = float(thresh_parts[1])
        self.yaw_thresh = float(thresh_parts[2])

        # Status record
        self._done_flag = False
        self.success_detected = False

    def quaternion_to_yaw(self, quat):
        """
        Extract yaw angle (rotation around Z-axis) from quaternion [w, x, y, z].
        Returns angle in degrees.
        """
        w, x, y, z = quat
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw_rad = np.arctan2(siny_cosp, cosy_cosp)
        return np.degrees(yaw_rad)

    def normalize_angle(self, angle):
        """
        Normalize angle to [-180, 180] range.
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def check_position(self, current_x, current_y):
        """
        Check if current position is within 2D box around target.
        Returns (is_ok, x_diff, y_diff).
        """
        x_diff = abs(current_x - self.target_x)
        y_diff = abs(current_y - self.target_y)
        is_ok = (x_diff < self.x_thresh) and (y_diff < self.y_thresh)
        return is_ok, x_diff, y_diff

    def check_orientation(self, current_yaw):
        """
        Check if current orientation is within threshold of target.
        Returns (is_ok, yaw_diff).
        """
        yaw_diff = abs(self.normalize_angle(current_yaw - self.target_yaw))
        return yaw_diff < self.yaw_thresh, yaw_diff

    def update(self, delta_time: float) -> float:
        if self._done_flag:
            return super().update(delta_time)

        # Get robot type dynamically
        robot_cfg = getattr(self.env, "robot_cfg", None)
        if robot_cfg is None:
            robot_cfg = getattr(self.env, "init_task_config", {}).get("robot_cfg", "G2_omnipicker")

        # Determine chassis prim path (G1 uses /G1, G2 uses /genie)
        if "G1" in robot_cfg:
            chassis_path = "/G1"
        else:
            chassis_path = "/genie"

        # Get chassis pose
        try:
            pos, quat = self.api_core.get_obj_world_pose(chassis_path)
        except Exception as e:
            logger.error(f"[ChassisAtTarget] Failed to get chassis pose from {chassis_path}: {e}")
            return super().update(delta_time)

        current_x, current_y = pos[0], pos[1]
        current_yaw = self.quaternion_to_yaw(quat)

        # Check position (2D box)
        position_ok, x_diff, y_diff = self.check_position(current_x, current_y)

        # Check orientation
        orientation_ok, yaw_diff = self.check_orientation(current_yaw)

        # Check if target reached (once detected, pass immediately)
        if position_ok and orientation_ok:
            self._done_flag = True
            self.success_detected = True
            logger.info(
                f"[ChassisAtTarget] Target reached: pos=({current_x:.3f}, {current_y:.3f}), " f"yaw={current_yaw:.1f}°"
            )

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag and self.success_detected:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        target_str = f"[{self.target_x:.2f}, {self.target_y:.2f}, {self.target_yaw:.1f}°]"
        thresh_str = f"[{self.x_thresh:.2f}, {self.y_thresh:.2f}, {self.yaw_thresh:.1f}°]"
        logger.info(f"Action [ChassisAtTarget] target={target_str}, thresh={thresh_str}, evt: {event.value}")

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
