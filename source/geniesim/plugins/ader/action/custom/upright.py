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
from isaacsim.core.utils.stage import get_current_stage

logger = Logger()


class Upright(EvaluateAction):
    def __init__(self, env, obj_name, tilt_threshold=15.0):
        super().__init__(env)
        self._holder_name, self._obj_name = self.placeholder_sparser(obj_name)
        self._done_flag = False

        self.tilt_threshold = tilt_threshold

    @property
    def obj_name(self) -> str:
        if self._holder_name:
            return getattr(self, self._obj_name)
        return self._obj_name

    def check_upright(self, obj_pose):
        rotation_matrix = obj_pose[:3, :3]
        local_y_axis = np.array([0, 1, 0])

        world_y_axis_obj = rotation_matrix.dot(local_y_axis)

        threshold_rad = np.radians(self.tilt_threshold)
        min_dot_product = np.cos(threshold_rad)

        is_upright = world_y_axis_obj[2] >= min_dot_product

        if is_upright:
            angle_deg = np.degrees(np.arccos(world_y_axis_obj[2]))
            logger.info(
                f"[Upright] Object is upright: {self.obj_name}, "
                f"dot product: {world_y_axis_obj[2]:.3f} (min: {min_dot_product:.3f}), "
                f"angle with world z-axis: {angle_deg:.2f}° (threshold: {self.tilt_threshold:.2f}°)"
            )

        return is_upright

    def update(self, delta_time: float) -> float:
        if self._done_flag:
            return super().update(delta_time)

        current_obj_pose = self.get_obj_pose(self.obj_name)

        if self.check_upright(current_obj_pose):
            self._done_flag = True

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [Upright] obj: {self.obj_name}, evt: {event.value}")

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            if self._done_flag:
                self.progress_info["SCORE"] = 1
                logger.info(f"[Upright] Task completed successfully for {self.obj_name}")
            else:
                self.progress_info["SCORE"] = 0
                logger.warning(f"[Upright] Task failed for {self.obj_name}")
