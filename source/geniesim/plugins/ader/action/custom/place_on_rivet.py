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


def _quat_diff_angle(q1, q2):
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


class PlaceOnRivet(EvaluateAction):
    """Check whether *active_obj* has been placed at the correct relative
    position and orientation w.r.t. *passive_obj* (the workspace) and is
    stationary for a configurable number of consecutive steps.

    Parameters (pipe-separated string)
    -----------------------------------
    active_obj  : body name of the workpiece
    passive_obj : body name of the workspace
    target_rel_pos : "x,y,z" relative position (workpiece − workspace)
    target_quat    : "qw,qx,qy,qz" target orientation of the workpiece
    xy_tol      : float, metres  (default 0.02)
    z_tol       : float, metres  (default 0.01)
    orient_tol  : float, radians (default 0.15)
    still_thresh: float, m/s     (default 0.02)
    still_steps : int            (default 15)
    """

    def __init__(
        self,
        env,
        active_obj: str,
        passive_obj: str,
        target_rel_pos: str,
        target_quat: str,
        xy_tol: float = 0.02,
        z_tol: float = 0.01,
        orient_tol: float = 0.15,
        still_thresh: float = 0.02,
        still_steps: int = 15,
    ):
        super().__init__(env)
        self.active_obj = active_obj
        self.passive_obj = passive_obj

        rp = [float(v) for v in target_rel_pos.split(",")]
        self.target_rel_pos = np.array(rp, dtype=np.float64)

        tq = [float(v) for v in target_quat.split(",")]
        self.target_quat = np.array(tq, dtype=np.float64)
        self.target_quat /= np.linalg.norm(self.target_quat) + 1e-12

        self.xy_tol = xy_tol
        self.z_tol = z_tol
        self.orient_tol = orient_tol
        self.still_thresh = still_thresh
        self.still_steps = int(still_steps)

        self._done_flag = False
        self._still_counter = 0
        self._prev_wp_pos = None

    def update(self, delta_time: float) -> float:
        if self._done_flag:
            return super().update(delta_time)

        wp_pos, wp_quat = self.get_world_pose(
            self._analyze_obj_name(self.active_obj)
        )
        ws_pos, _ = self.get_world_pose(
            self._analyze_obj_name(self.passive_obj)
        )

        rel_pos = wp_pos - ws_pos
        diff_xy = np.linalg.norm(rel_pos[:2] - self.target_rel_pos[:2])
        diff_z = abs(rel_pos[2] - self.target_rel_pos[2])
        orient_diff = _quat_diff_angle(wp_quat, self.target_quat)

        xy_ok = diff_xy < self.xy_tol
        z_ok = diff_z < self.z_tol
        orient_ok = orient_diff < self.orient_tol

        if self._prev_wp_pos is not None:
            speed = np.linalg.norm(wp_pos - self._prev_wp_pos) * 30.0
        else:
            speed = 0.0
        self._prev_wp_pos = wp_pos.copy()

        still = speed < self.still_thresh

        if xy_ok and z_ok and orient_ok and still:
            self._still_counter += 1
        else:
            self._still_counter = 0

        if self._still_counter >= self.still_steps:
            self._done_flag = True
            logger.info(
                f"[PlaceOnRivet] SUCCESS — rel_pos={rel_pos}, "
                f"diff_xy={diff_xy:.4f}, diff_z={diff_z:.4f}, "
                f"orient_diff={orient_diff:.4f} rad, "
                f"still_counter={self._still_counter}"
            )

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(
            f"Action [PlaceOnRivet] {self.active_obj} -> {self.passive_obj}, evt: {event.value}"
        )

        if event == ActionEvent.FINISHED:
            if self._done_flag:
                self.progress_info["SCORE"] = 1
            else:
                self.progress_info["SCORE"] = 0
