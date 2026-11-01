# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Explicit failure exit when an object returns to a horizontal surface."""

import math

from geniesim_benchmark.plugins.ader.action.common_actions import EvalExitAction, EvaluateAction


class OnSurface(EvalExitAction, EvaluateAction):
    def __init__(
        self,
        env,
        active_obj: str,
        surface_obj: str,
        z_tolerance: float = 0.02,
        confirm_checks: int = 2,
    ):
        if any(not isinstance(name, str) or not name for name in (active_obj, surface_obj)):
            raise ValueError("OnSurface needs active_obj and surface_obj")
        if type(z_tolerance) not in (int, float) or not math.isfinite(z_tolerance) or z_tolerance < 0:
            raise ValueError("OnSurface.z_tolerance must be finite and nonnegative")
        if type(confirm_checks) is not int or confirm_checks < 1:
            raise ValueError("OnSurface.confirm_checks must be a positive integer")
        super().__init__(env)
        self.active_obj = active_obj
        self.surface_obj = surface_obj
        self.z_tolerance = z_tolerance
        self.confirm_checks = confirm_checks
        self._surface_checks = 0
        self._done_flag = False

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        low, high = self.get_obj_aabb_new(self.active_obj)
        surface_low, surface_high = self.get_obj_aabb_new(self.surface_obj)
        gap = float(low[2] - surface_high[2])
        overlap_x = min(high[0], surface_high[0]) - max(low[0], surface_low[0])
        overlap_y = min(high[1], surface_high[1]) - max(low[1], surface_low[1])
        on_surface = abs(gap) <= self.z_tolerance and overlap_x > 0 and overlap_y > 0
        self._surface_checks = self._surface_checks + 1 if on_surface else 0
        self.progress_info["SURFACE"] = {
            "active_obj": self.active_obj,
            "surface_obj": self.surface_obj,
            "z_gap": gap,
            "overlap_x": float(overlap_x),
            "overlap_y": float(overlap_y),
            "consecutive_checks": self._surface_checks,
        }
        if self._surface_checks >= self.confirm_checks:
            self._done_flag = True
            self.progress_info["SCORE"] = 0
            self.progress_info["STATUS"] = "FAILED"
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag
