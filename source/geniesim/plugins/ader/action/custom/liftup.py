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


class LiftUp(EvaluateAction):
    def __init__(self, env, obj_name, lift_threshold=0.05):
        super().__init__(env)
        self._holder_name, self._obj_name = self.placeholder_sparser(obj_name)
        self._done_flag = False

        # Lift threshold (default 5cm)
        self.lift_threshold = lift_threshold

        # Status record
        self.initial_z = None

    @property
    def obj_name(self) -> str:
        if self._holder_name:
            return getattr(self, self._obj_name)
        return self._obj_name

    def check_liftup_success(self, obj_pose):
        # Record initial Z-coordinate
        current_z = obj_pose[2, 3]
        if self.initial_z is None:
            self.initial_z = current_z
            logger.info(f"[LiftUp] Initial Z position recorded: {self.initial_z:.4f}m for {self.obj_name}")
            return False

        # Check if the object is lifted above the threshold
        z_diff = current_z - self.initial_z
        if z_diff >= self.lift_threshold:
            logger.info(
                f"[LiftUp] Object lifted successfully: {self.obj_name}, "
                f"height change: {z_diff:.4f}m (threshold: {self.lift_threshold:.4f}m)"
            )
            return True

        return False

    def update(self, delta_time: float) -> float:
        # If success has already been detected, return directly
        if self._done_flag:
            return super().update(delta_time)

        current_obj_pose = self.get_obj_pose(self.obj_name)

        # Single frame detection: success if any single frame meets the condition
        if self.check_liftup_success(current_obj_pose):
            self._done_flag = True

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [LiftUp] obj: {self.obj_name}, evt: {event.value}")

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            if self._done_flag:
                self.progress_info["SCORE"] = 1
                logger.info(f"[LiftUp] Task completed successfully for {self.obj_name}")
            else:
                self.progress_info["SCORE"] = 0
