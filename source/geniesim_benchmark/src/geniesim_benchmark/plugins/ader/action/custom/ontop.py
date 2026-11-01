# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim_benchmark.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
import numpy as np

from geniesim_benchmark.plugins.logger import Logger

logger = Logger()


class Ontop(EvaluateAction):
    def __init__(self, env, active_obj, passive_obj, area_denom="active", threshold=0.5):
        super().__init__(env)
        self.active_obj = active_obj
        self.passive_obj = passive_obj
        self._done_flag = False
        # "active": area ratio denominator is active's own largest face (legacy
        # default). "min": denominator is min(active_face, passive_face) -- a
        # better fit when active is much larger than passive (e.g. a long beam
        # resting on a small support), where the legacy denominator makes the
        # ratio structurally unreachable regardless of placement quality.
        self.area_denom = area_denom
        self.threshold = threshold

    def update(self, delta_time: float) -> float:
        # Get the AABB bounding box of two objects
        aa_A, bb_A = self.get_obj_aabb_new(self.active_obj)
        aa_B, bb_B = self.get_obj_aabb_new(self.passive_obj)
        aabb_A_str = f"min=[{aa_A[0]:.4f},{aa_A[1]:.4f},{aa_A[2]:.4f}] max=[{bb_A[0]:.4f},{bb_A[1]:.4f},{bb_A[2]:.4f}]"
        aabb_B_str = f"min=[{aa_B[0]:.4f},{aa_B[1]:.4f},{aa_B[2]:.4f}] max=[{bb_B[0]:.4f},{bb_B[1]:.4f},{bb_B[2]:.4f}]"

        # Calculate the bottom of active and the top of passive
        active_bottom = aa_A[2]
        passive_top = bb_B[2]
        z_gap = active_bottom - passive_top
        z_ok = abs(z_gap) <= 0.02

        if not z_ok:
            logger.info(
                f"[Ontop Check] active={self.active_obj} passive={self.passive_obj}\n"
                f"  active AABB: {aabb_A_str}\n"
                f"  passive AABB: {aabb_B_str}\n"
                f"  active_bottom={active_bottom:.4f}  passive_top={passive_top:.4f}  "
                f"z_gap={z_gap:+.4f}  (|gap|<=0.02 required)  z_ok={z_ok}\n"
                f"  gate_failed=z_gap (area ratio not evaluated)"
            )
            return super().update(delta_time)

        # Calculate projection intersection of X-Y planes
        x_min = max(aa_A[0], aa_B[0])
        x_max = min(bb_A[0], bb_B[0])
        y_min = max(aa_A[1], aa_B[1])
        y_max = min(bb_A[1], bb_B[1])

        # Calculate the effective intersection area
        inter_area = 0.0
        if x_max > x_min and y_max > y_min:
            inter_area = (x_max - x_min) * (y_max - y_min)

        # Calculate the length of three dimensions of active objects
        dx = bb_A[0] - aa_A[0]
        dy = bb_A[1] - aa_A[1]
        dz = bb_A[2] - aa_A[2]

        # Largest of the three face areas, used as the base area
        area_A = max(dx * dy, dx * dz, dy * dz)

        if self.area_denom == "min":
            dxB = bb_B[0] - aa_B[0]
            dyB = bb_B[1] - aa_B[1]
            dzB = bb_B[2] - aa_B[2]
            area_B = max(dxB * dyB, dxB * dzB, dyB * dzB)
            area_denom_val = min(area_A, area_B)
        else:
            area_B = None
            area_denom_val = area_A

        ratio = inter_area / area_denom_val if area_denom_val > 0 else 0.0
        ratio_ok = area_denom_val > 0 and ratio >= self.threshold

        logger.info(
            f"[Ontop Check] active={self.active_obj} passive={self.passive_obj}\n"
            f"  active AABB: {aabb_A_str}\n"
            f"  passive AABB: {aabb_B_str}\n"
            f"  z_gap={z_gap:+.4f}  z_ok=True\n"
            f"  xy_intersection=[{x_min:.4f},{x_max:.4f}]x[{y_min:.4f},{y_max:.4f}]  inter_area={inter_area:.6f}\n"
            f"  area_A(active_face)={area_A:.6f}"
            + (f"  area_B(passive_face)={area_B:.6f}" if area_B is not None else "")
            + f"  area_denom[{self.area_denom}]={area_denom_val:.6f}\n"
            f"  ratio={ratio:.4f}  threshold={self.threshold:.4f}  ratio_ok={ratio_ok}\n"
            f"  gate_failed={'none (frame passed)' if ratio_ok else 'area_ratio'}"
        )

        # If the area ratio exceeds the threshold, the mark is completed
        if ratio_ok:
            self._done_flag = True

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [Ontop] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
