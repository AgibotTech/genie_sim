# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
from collections import deque
import numpy as np
from geniesim.plugins.logger import Logger
from enum import Enum, auto
import time

logger = Logger()


class HandoverState(Enum):
    IDLE = auto()
    LEFT_HOLD = auto()
    RIGHT_HOLD = auto()
    BOTH_HOLD = auto()


class HandoverEvent(Enum):
    NONE = auto()
    TRANSFERRED = auto()


class GripperPassing(EvaluateAction):
    def __init__(self, env, obj_prim, reverse=False):
        super().__init__(env)

        self.reverse = reverse  # True: right to true; False: left to right
        self.obj_prim = obj_prim
        self.handover_state = HandoverState.IDLE
        self.cur_joint_positions = None
        self.handover_timeout = 5.0  # Time window for handover event
        self.pose_history = deque(maxlen=3)  # Pose History Queue
        self.pos_threshold = 0.06
        self.rot_threshold = 10

    def line_aabb_intersect(self, p0, p1, min_pt, max_pt) -> bool:
        dir = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        tmin, tmax = 0.0, 1.0
        for i in range(3):
            if abs(dir[i]) < 1e-6:
                if p0[i] < min_pt[i] or p0[i] > max_pt[i]:
                    return False
            else:
                ood = 1.0 / dir[i]
                t1 = (min_pt[i] - p0[i]) * ood
                t2 = (max_pt[i] - p0[i]) * ood
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return False
        return True

    def eepose_across_obj(self) -> bool:
        """Detects if the gripper end-effector passes through the object"""
        left_ee_pose = self.get_world_pose_matrix("/G1/gripper_l_center_link")
        right_ee_pose = self.get_world_pose_matrix("/G1/gripper_r_center_link")
        obj_min, obj_max = self.get_obj_aabb_new(self.obj_prim)
        return self.line_aabb_intersect(left_ee_pose[:3, 3], right_ee_pose[:3, 3], obj_min, obj_max)

    def _is_grasped(self, is_left):
        if is_left:
            return self.cur_joint_positions["idx41_gripper_l_outer_joint1"] < 0.3
        else:
            return self.cur_joint_positions["idx81_gripper_r_outer_joint1"] < 0.3

    def change_handover_state(self, state):
        print(f"Change Handover State: {self.handover_state} -> {state}")
        self.handover_state = state

    def sm_update(self, left_hold: bool, right_hold: bool) -> HandoverEvent:
        """
        left_hold / right_hold : bool
            Indicates whether the current left/right gripper is grasping the object
        """
        now = time.time()

        if self.handover_state == HandoverState.IDLE:
            if not self.reverse and left_hold and not right_hold:
                self.change_handover_state(HandoverState.LEFT_HOLD)
            elif self.reverse and right_hold and not left_hold:
                self.change_handover_state(HandoverState.RIGHT_HOLD)
            elif left_hold and right_hold:
                self.change_handover_state(HandoverState.BOTH_HOLD)
            return HandoverEvent.NONE

        if self.handover_state == HandoverState.LEFT_HOLD:
            if left_hold and right_hold:
                self.change_handover_state(HandoverState.BOTH_HOLD)
            elif not left_hold and right_hold:
                self.change_handover_state(HandoverState.RIGHT_HOLD)
                if not self.reverse:  # Left -> Right
                    return HandoverEvent.TRANSFERRED
            elif not left_hold and not right_hold:
                self.change_handover_state(HandoverState.IDLE)
            return HandoverEvent.NONE

        if self.handover_state == HandoverState.RIGHT_HOLD:
            if left_hold and right_hold:
                self.change_handover_state(HandoverState.BOTH_HOLD)
            elif left_hold and not right_hold:
                self.change_handover_state(HandoverState.LEFT_HOLD)
                if self.reverse:  # Right -> Left
                    return HandoverEvent.TRANSFERRED
            elif not left_hold and not right_hold:
                self.change_handover_state(HandoverState.IDLE)
            return HandoverEvent.NONE

        if self.handover_state == HandoverState.BOTH_HOLD:
            if right_hold and not left_hold:
                self.change_handover_state(HandoverState.RIGHT_HOLD)
                if not self.reverse:
                    return HandoverEvent.TRANSFERRED
            if left_hold and not right_hold:
                self.change_handover_state(HandoverState.LEFT_HOLD)
                if self.reverse:
                    return HandoverEvent.TRANSFERRED
            if not left_hold and not right_hold:
                self.change_handover_state(HandoverState.IDLE)
            return HandoverEvent.NONE

        return HandoverEvent.NONE

    def update(self, delta_time: float) -> float:
        self.cur_joint_positions = self.api_core.get_joint_state_dict()
        left_hold = self._is_grasped(True)
        right_hold = self._is_grasped(False)
        evt = self.sm_update(left_hold, right_hold)
        self._done_flag = evt == HandoverEvent.TRANSFERRED and self.eepose_across_obj()

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [GripperPassing] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            pass
