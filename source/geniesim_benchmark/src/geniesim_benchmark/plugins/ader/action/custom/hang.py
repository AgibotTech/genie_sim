# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from collections import deque

import numpy as np

from geniesim_benchmark.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
from geniesim_benchmark.plugins.logger import Logger

logger = Logger()


class Hang(EvaluateAction):
    """Evaluate whether an object ends up hanging: its height stays above
    `min_height`, settles (varies by no more than `tolerance` over
    `stable_checks` consecutive checks), and no gripper still holds it.

    Unlike Ontop this makes no assumption about the support geometry, which
    is what a hook-shaped rack breaks: a hung cup has no face resting on the
    rack's top surface.

    Args:
        env: The simulation environment.
        obj_name: Object name (placeholder format supported).
        min_height: World-frame z the object must stay above (meters).
        stable_checks: Consecutive checks the height must hold. Checks run on
            the eval tick (every 30 env steps), not every physics step.
        tolerance: Allowed spread over those checks (meters).
        open_ratio: Fraction of the gripper's open position below which a
            gripper counts as still clamped.
        near_dist: Only grippers within this distance of the object have to be
            open; the idle arm's gripper is irrelevant (meters).
    """

    def __init__(
        self,
        env,
        obj_name,
        min_height,
        stable_checks=5,
        tolerance=0.01,
        open_ratio=0.5,
        near_dist=0.2,
    ):
        super().__init__(env)
        self._holder_name, self._obj_name = self.placeholder_sparser(obj_name)
        self.min_height = float(min_height)
        self.stable_checks = int(stable_checks)
        self.tolerance = float(tolerance)
        self.open_ratio = float(open_ratio)
        self.near_dist = float(near_dist)
        self._heights = deque(maxlen=self.stable_checks)
        self._done_flag = False

    @property
    def obj_name(self) -> str:
        if self._holder_name:
            return getattr(self, self._obj_name)
        return self._obj_name

    def _released(self, obj_pos):
        """True unless a gripper next to the object is still clamped shut.

        The open reference comes from the robot config's open pose, which for
        the parallel-gripper robots is the joint value itself; robots whose
        action space differs from joint space need their own reference here.
        """
        cfg = getattr(self.env, "cfg", None) or {}
        joints = cfg.get("gripper_joints") or []
        open_pose = cfg.get("init_gripper_open") or []
        if not joints or not open_pose:
            return True

        states = self.api_core.get_joint_state_dict() or {}
        half = len(joints) // 2
        sides = (("left_gripper", joints[:half], open_pose[0]), ("right_gripper", joints[half:], open_pose[-1]))
        for gripper_id, side_joints, side_open in sides:
            g_pos, _ = self.get_world_pose(self.gripper_link_path(gripper_id))
            if np.linalg.norm(np.asarray(g_pos) - obj_pos) > self.near_dist:
                continue
            if any(states.get(j, side_open) < self.open_ratio * side_open for j in side_joints):
                return False
        return True

    def update(self, delta_time: float) -> float:
        if self._done_flag:
            return super().update(delta_time)

        obj_pose = self.get_obj_pose(self.obj_name)
        current_z = obj_pose[2, 3]
        if current_z < self.min_height or not self._released(obj_pose[:3, 3]):
            # Dropped, still on the table, or being carried -- the run has to
            # restart from scratch.
            self._heights.clear()
            return super().update(delta_time)

        self._heights.append(current_z)
        spread = max(self._heights) - min(self._heights)
        if len(self._heights) == self.stable_checks and spread <= self.tolerance:
            self._done_flag = True
            logger.info(
                f"[Hang] {self.obj_name} hanging at z={current_z:.4f}m "
                f"(min {self.min_height:.4f}m, spread {spread:.4f}m over {self.stable_checks} checks)"
            )

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [Hang] obj: {self.obj_name}, evt: {event.value}")

        if event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1 if self._done_flag else 0
