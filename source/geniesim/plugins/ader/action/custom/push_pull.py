# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
import numpy as np
from collections import deque
from geniesim.plugins.logger import Logger

logger = Logger()


class PushPull(EvaluateAction):
    def __init__(self, env, obj_name, thresh_min, thresh_max, joint_index=0):
        super().__init__(env)
        self.obj_name = obj_name
        self._done_flag = False
        self.thresh_min = float(thresh_min)
        self.thresh_max = float(thresh_max)
        self.env = env
        self.joint_index = joint_index
        self._pass_frame = 0

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        prismatic_joint = self.get_prismatic_joint(self.obj_name)

        condition_met = self.thresh_min <= prismatic_joint[self.joint_index] <= self.thresh_max

        if condition_met:
            self._pass_frame += 1

        if self._pass_frame > 2:
            self._done_flag = True
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [PushPull] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
