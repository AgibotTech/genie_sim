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


class CheckParticleInBBox(EvaluateAction):
    def __init__(self, env, threshold, bbox):
        super().__init__(env)

        self.parti_prim_path = "/World/Objects/part/sampledParticles"
        self.threshold = threshold
        self._done_flag = False
        self.bbox = bbox

    def update(self, delta_time: float) -> float:
        rsp = self.rpc_robot.client.GetPartiPointNumInbbox(self.parti_prim_path, self.bbox)
        self._done_flag = rsp.num < self.threshold
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [CheckParticleInBBox] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            pass
