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


class Stack(EvaluateAction):
    """Check if center points of multiple objects are all within threshold of each other.

    Input format: "[a,b,c]|[x,y]"
    - a, b, c: object IDs (at least 2, can be more)
    - x, y: XY threshold (meters) for center deviation

    Passes when all objects' XY centers are within (x_thresh, y_thresh) of the first
    object's center. Used for stack-alignment tasks.
    """

    def __init__(self, env, obj_list_str, thresh_str):
        super().__init__(env)
        obj_part = obj_list_str.strip().strip("[]")
        self._obj_ids = [o.strip() for o in obj_part.split(",") if o.strip()]
        thresh_part = thresh_str.strip().strip("[]")
        thresh_vals = [float(t.strip()) for t in thresh_part.split(",") if t.strip()]
        self._x_thresh = float(thresh_vals[0]) if len(thresh_vals) >= 1 else 0.05
        self._y_thresh = float(thresh_vals[1]) if len(thresh_vals) >= 2 else 0.05
        self._done_flag = False
        self._pass_frame = 0

    def update(self, delta_time: float) -> float:
        if len(self._obj_ids) < 2:
            return super().update(delta_time)

        ref_pose = self.get_obj_pose(self._obj_ids[0])
        ref_center = ref_pose[:3, 3]

        all_within = True
        for obj_id in self._obj_ids[1:]:
            pose = self.get_obj_pose(obj_id)
            center = pose[:3, 3]
            dx = abs(center[0] - ref_center[0])
            dy = abs(center[1] - ref_center[1])
            if dx > self._x_thresh or dy > self._y_thresh:
                all_within = False
                break

        if all_within:
            self._pass_frame += 1
        else:
            self._pass_frame = 0

        if self._pass_frame > 1:
            self._done_flag = True

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [Stack] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
