# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)

from geniesim.plugins.logger import Logger

logger = Logger()


class Approach(EvaluateAction):
    def __init__(self, env, t_x, t_y, t_z):
        super().__init__(env)
        self._done_flag = False
        self.t_x = t_x
        self.t_y = t_y
        self.t_z = t_z

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0

        g2_c_link = "/G2/gripper_r_center_link"
        g_pos, _ = self.get_world_pose(g2_c_link)
        print(g_pos)

        if abs(self.t_x - g_pos[0]) < 0.01 and abs(self.t_y - g_pos[1]) < 0.01 and abs(self.t_z - g_pos[2]) < 0.01:
            self._done_flag = True
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [Approach] evt: {event.value}")
        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
