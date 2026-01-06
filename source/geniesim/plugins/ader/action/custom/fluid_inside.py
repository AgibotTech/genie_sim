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


class FluidInside(EvaluateAction):
    def __init__(self, env, passive_obj, object_info_dir):
        super().__init__(env)
        self.fluid_prim = "/World/Objects/part/sampledParticles"
        self.obj_info_dir = object_info_dir
        self.passive_obj = passive_obj
        self.threshold = 50
        self._done_flag = False

    def update(self, delta_time: float) -> float:
        object_size = self.get_object_size(self.obj_info_dir)
        a, b = self.get_obj_aabb(self.passive_obj, object_size)
        bbox = [a[0], a[1], a[2], b[0], b[1], b[2]]
        rsp = self.rpc_robot.client.GetPartiPointNumInbbox(self.fluid_prim, bbox)
        self._done_flag = rsp.num > self.threshold
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            pass

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [FluidInside] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            self.progress_info["SCORE"] = 0
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
