# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Opt-in generic partial insertion; geometry and tolerances belong to task data."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from ..common_actions import EvaluateAction, ActionEvent
from .shaft_insertion_geometry import (
    insertion,
    pose_from_wxyz,
    segment,
    validated_sections,
)


class ShaftInserted(EvaluateAction):
    def __init__(
        self,
        env,
        object: str,
        reference: str,
        hole: list,
        shaft: list,
        hole_radius: float,
        shaft_radius: float,
        min_overlap: float,
        max_angle: float,
        radial_tolerance: float = 0.0,
        shaft_sections: list | None = None,
        diagnostics_path: str | None = None,
    ):
        super().__init__(env)
        if (
            not isinstance(object, str)
            or not object
            or not isinstance(reference, str)
            or not reference
        ):
            raise ValueError("object and reference must be nonempty names")
        self.active_obj, self.passive_obj = object, reference
        self.hole, self.shaft = segment(hole), segment(shaft)
        values = [hole_radius, shaft_radius, min_overlap, max_angle, radial_tolerance]
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
            for v in values
        ):
            raise ValueError("Expected finite numeric geometry tolerances")
        if not (
            0 < shaft_radius < hole_radius
            and 0 <= radial_tolerance < shaft_radius
            and 0 < max_angle < np.pi / 2
            and 0
            < min_overlap
            <= min(
                np.linalg.norm(self.hole[1] - self.hole[0]),
                np.linalg.norm(self.shaft[1] - self.shaft[0]),
            )
        ):
            raise ValueError("Invalid radii, angle or minimum overlap")
        self.hole_radius, self.shaft_radius = hole_radius, shaft_radius
        self.min_overlap, self.max_angle = min_overlap, max_angle
        self.radial_tolerance = radial_tolerance
        self.shaft_sections = validated_sections(
            shaft_sections, self.shaft, shaft_radius
        )
        if diagnostics_path is not None and (
            not isinstance(diagnostics_path, str) or not diagnostics_path
        ):
            raise ValueError("diagnostics_path must be a nonempty string")
        self.diagnostics_path = Path(diagnostics_path) if diagnostics_path else None
        if self.diagnostics_path is not None:
            self.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.diagnostics_path.open("a", encoding="utf-8"):
                pass
        self._done_flag = False
        self.last_measurement = {}

    def measured_pose(self, name: str) -> np.ndarray:
        return pose_from_wxyz(*self.get_world_pose(self._analyze_obj_name(name)))

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        active = reference = relative = None
        try:
            reference = self.measured_pose(self.passive_obj)
            active = self.measured_pose(self.active_obj)
            relative = np.linalg.solve(reference, active)
            self.last_measurement = insertion(
                relative,
                self.hole,
                self.shaft,
                self.hole_radius,
                self.shaft_radius,
                self.min_overlap,
                self.max_angle,
                self.radial_tolerance,
                self.shaft_sections,
            )
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError):
            self.last_measurement = {"pass": False, "invalid_pose": True}
        self._done_flag = self._done_flag or self.last_measurement["pass"]
        if self.diagnostics_path is not None:
            record = {
                "wall_time": time.time(),
                "sim_time": getattr(
                    getattr(
                        getattr(self.api_core, "ui_builder", None), "my_world", None
                    ),
                    "current_time",
                    None,
                ),
                "step": getattr(self.env, "current_step", None),
                "object": self.active_obj,
                "reference": self.passive_obj,
                "active_pose": active.tolist() if active is not None else None,
                "reference_pose": reference.tolist() if reference is not None else None,
                "relative_pose": relative.tolist() if relative is not None else None,
                "measurement": self.last_measurement,
                "latched": self._done_flag,
                "configured_sections": self.shaft_sections,
                "thresholds": {
                    "min_overlap_m": self.min_overlap,
                    "max_angle_rad": self.max_angle,
                    "hole_radius_m": self.hole_radius,
                    "shaft_radius_m": self.shaft_radius,
                    "radial_tolerance_m": self.radial_tolerance,
                },
            }
            with self.diagnostics_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, allow_nan=False) + "\n")
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        self.progress_info["MEASUREMENT"] = self.last_measurement
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action, event):
        if event == ActionEvent.FINISHED and self._done_flag:
            self.progress_info["SCORE"] = 1
