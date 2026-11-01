# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from geniesim_benchmark.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
from geniesim_benchmark.plugins.logger import Logger

logger = Logger()


class RobotNearObject(EvaluateAction):
    _REQUIRED_CONSECUTIVE = 5

    def __init__(self, env, obj_id, distance_thresh):
        super().__init__(env)
        self._holder, self._obj = self.placeholder_sparser(obj_id)
        self.distance_thresh = float(distance_thresh)
        self._consecutive_count = 0
        self._done_flag = False
        self._update_count = 0

    @property
    def obj_name(self) -> str:
        if self._holder:
            return getattr(self, self._obj)
        return self._obj

    def _resolve_obj_path(self, obj_name):
        if obj_name.startswith("/World") or obj_name.startswith("/Workspace"):
            return obj_name
        return self._analyze_obj_name(obj_name)

    def _get_base_link_path(self):
        robot_cfg = getattr(self.env, "robot_cfg", None)
        if robot_cfg is None:
            robot_cfg = getattr(self.env, "init_task_config", {}).get("robot_cfg", "G2_omnipicker")
        if "G1" in robot_cfg:
            return "/G1/base_link"
        return "/genie/base_link"

    def _get_robot_rotation_matrix(self):
        link_path = self._get_base_link_path()
        try:
            _, quat = self.get_world_pose(link_path)
            qw, qx, qy, qz = np.array(quat)
            return np.array(
                [
                    [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                    [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
                    [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
                ]
            )
        except Exception as e:
            logger.warning(f"[RobotNearObject] Failed to get robot rotation: {e}")
            return np.eye(3)

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        if self._done_flag:
            return super().update(delta_time)

        self._update_count += 1

        try:
            link_path = self._get_base_link_path()
            base_pos, _ = self.get_world_pose(link_path)

            obj_path = self._resolve_obj_path(self.obj_name)
            aabb = self.api_core.get_obj_aabb(obj_path)
            obj_pos = np.array([(aabb[0] + aabb[3]) / 2, (aabb[1] + aabb[4]) / 2, (aabb[2] + aabb[5]) / 2])

            dx = obj_pos[0] - base_pos[0]
            dy = obj_pos[1] - base_pos[1]
            dist_xy = (dx * dx + dy * dy) ** 0.5

            rotation = self._get_robot_rotation_matrix()
            forward_axis = rotation @ np.array([1, 0, 0])
            in_front = forward_axis[0] * dx + forward_axis[1] * dy > 0

            near = dist_xy < self.distance_thresh
            if near and in_front:
                self._consecutive_count += 1
                logger.info(
                    f"[RobotNearObject] Conditions satisfied "
                    f"(count={self._consecutive_count}/{self._REQUIRED_CONSECUTIVE}, "
                    f"dist_xy={dist_xy:.4f}, in_front={in_front})"
                )
                if self._consecutive_count >= self._REQUIRED_CONSECUTIVE:
                    self._done_flag = True
                    logger.info(
                        f"[RobotNearObject] Robot is near object '{self.obj_name}' "
                        f"(dist_xy={dist_xy:.4f} < {self.distance_thresh})"
                    )
            else:
                self._consecutive_count = 0

        except Exception as e:
            logger.warning(f"[RobotNearObject] Error during update #{self._update_count}: {e}")

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"
            self.progress_info["OBJECT"] = self.obj_name

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [RobotNearObject] {self.obj_name} evt: {event.value}")
        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
