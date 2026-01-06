# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
from geniesim.plugins.ader.action.common_actions import (
    EvalExitAction,
    ActionBase,
    ActionEvent,
)
import numpy as np
from collections import deque
from geniesim.plugins.logger import Logger

logger = Logger()


class Onfloor(EvalExitAction):
    def __init__(self, env, obj_name, height):
        super().__init__(env)
        self._holder_name, self._obj_name = self.placeholder_sparser(obj_name)
        self._done_flag = False
        self.z_ref = float(height)
        self.z_threshold = 0.3
        self.env = env

    @property
    def obj_name(self) -> bool:
        if self._holder_name:
            return getattr(self, self._obj_name)
        return self._obj_name

    def _analyze_obj_name(self, obj_name):
        if obj_name.startswith("/World"):
            return obj_name

        if not self.env.init_task_config.get("sub_task_name"):
            return "/World/Objects/" + obj_name
        else:
            return "/Workspace/Objects/" + obj_name

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        pos, quat = self.env.api_core.get_obj_world_pose(self._analyze_obj_name(self.obj_name))
        current_z = pos[2]
        if abs(current_z - self.z_ref) < self.z_threshold:
            self._done_flag = True
            self.progress_info["SCORE"] = 0
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        pass

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [Onfloor] evt: %d" % (event.value))

        if event == ActionEvent.FINISHED:
            self.env.cancel_eval()
