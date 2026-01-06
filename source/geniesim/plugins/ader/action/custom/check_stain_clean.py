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


class CheckStainClean(EvaluateAction):
    def __init__(self, env, stain_prim_path, threshold):
        super().__init__(env)

        self.stain_prim_path = stain_prim_path
        self.threshold = threshold
        self._done_flag = False

    def update(self, delta_time: float) -> float:
        num = self.api_core.count_visible_meshes(self.stain_prim_path)
        self._done_flag = num < self.threshold
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [CheckStainClean] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            pass
