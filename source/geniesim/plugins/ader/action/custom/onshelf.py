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
import ast

logger = Logger()


class OnShelf(EvaluateAction):
    def __init__(self, env, obj_name, target_name, bbox, height):
        super().__init__(env)
        self.obj_name = obj_name
        self.target_name = target_name
        self.height = float(height)
        self._done_flag = False
        self.success_frame = 0
        self.success_time = 5
        self.bbox = ast.literal_eval(bbox)

    def update(self, delta_time: float) -> float:
        [x, y, z] = self.get_obj_pose(self.obj_name)[0:3, 3]
        [x_t, y_t, z_t] = self.get_obj_pose(self.target_name)[0:3, 3]

        x_min, x_max = self.bbox[0:2]
        y_min, y_max = self.bbox[2:4]
        z_min, z_max = self.bbox[4:6]
        if z_min < z - self.height < z_max and x_min < x - x_t < x_max and y_min < y - y_t < y_max:
            self.success_frame += 1
        if self.success_frame > self.success_time:
            self.success_frame = 0
            self._done_flag = True
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [OnShelf] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
