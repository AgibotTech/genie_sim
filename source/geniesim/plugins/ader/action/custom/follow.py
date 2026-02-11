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
import ast


logger = Logger()


class Follow(EvaluateAction):
    def __init__(self, env, obj_name, bbox, gripper_id):
        super().__init__(env)
        self._holder_name, self._obj_name = self.placeholder_sparser(obj_name)
        self._done_flag = False
        self.bbox = ast.literal_eval(bbox)
        self.env = env
        self._holder_id, self._gripper_id = self.placeholder_sparser(gripper_id)

    @property
    def obj_name(self) -> bool:
        if self._holder_name:
            return getattr(self, self._obj_name)
        return self._obj_name

    @property
    def gripper_id(self) -> bool:
        if self._holder_id:
            return getattr(self, self._gripper_id)
        return self._gripper_id

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        aa, bb = self.get_obj_aabb(self.obj_name, self.bbox)

        # Get robot type
        robot_cfg = getattr(self.env, "robot_cfg", None)
        if robot_cfg is None:
            robot_cfg = getattr(self.env, "init_task_config", {}).get("robot_cfg", "G2_omnipicker")
        # Determine robot base prim path (G1 uses /G1, G2 uses /genie)
        if "G1" in robot_cfg:
            robot_base = "/G1"
        else:
            robot_base = "/genie"
        # Get gripper pose
        link_prim_path = (
            f"{robot_base}/gripper_r_center_link"
            if "right" in self.gripper_id
            else f"{robot_base}/gripper_l_center_link"
        )

        g_pos, _ = self.get_world_pose(link_prim_path)
        if self.aabb_contains_point(g_pos, (aa, bb)):
            self._done_flag = True
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [Follow] {self.obj_name} evt: {event.value}")
        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
