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


class Pass2People(EvaluateAction):
    def __init__(self, env, obj_name):
        super().__init__(env)
        self.obj_name = obj_name
        self.people_pose = self.get_obj_pose("/World/people")
        self.threshold = 0.2
        self._done_flag = False

    def update(self, delta_time: float) -> float:
        pose_obj_y = self.get_obj_pose(self.obj_name)[1, 3]
        pose_people_y = self.people_pose[1, 3]
        if abs(pose_obj_y - pose_people_y) < pose_people_y:
            self._done_flag = True
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [Pass2People] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            pass
