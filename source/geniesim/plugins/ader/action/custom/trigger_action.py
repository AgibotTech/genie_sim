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


class TriggerAction(EvaluateAction):
    def __init__(self, env, prim_path, target_rsp):
        super().__init__(env)
        self.prim_path = prim_path
        self.target_rsp = target_rsp
        self._done_flag = False
        self.progress_info["target"] = prim_path

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        output = self.api_core.get_trigger_action(self.prim_path)
        if output == self.target_rsp:
            self._done_flag = True
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [TriggerAction] {self.prim_path} evt: {event.value}")
        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
