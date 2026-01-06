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


class Inside(EvaluateAction):
    def __init__(self, env, active_obj, passive_obj, scale):
        super().__init__(env)
        self.active_obj = active_obj
        self.passive_obj = passive_obj
        self._done_flag = False
        self._pass_frame = 0
        self.scale = float(scale)

    def update(self, delta_time: float) -> float:
        pose_A = self.get_obj_pose(self.active_obj)
        pos_A = pose_A[:3, 3].reshape(
            -1,
        )
        self.aa_B, self.bb_B = self.get_obj_aabb_new(self.passive_obj)

        mid = (self.aa_B + self.bb_B) / 2
        rescaled_size = (self.bb_B - self.aa_B) * self.scale
        aa_new = mid - rescaled_size / 2
        bb_new = mid + rescaled_size / 2
        if self.aabb_contains_point(pos_A, (aa_new, bb_new)):
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
        logger.info("Action [Inside] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
