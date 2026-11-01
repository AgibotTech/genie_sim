# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Opt-in continuous carry credit, sealed on the first return to a surface."""

from __future__ import annotations

import json
import math

import numpy as np

from ..common_actions import EvaluateAction
from geniesim_benchmark.plugins.logger import Logger

logger = Logger()


class CarryDistanceScore(EvaluateAction):
    """Score a MixedRules goal by directed displacement after its stage starts.

    Sample only when the ordinary MixedRules evaluator runs. No additional
    callbacks or changes to the native checker cadence are installed.
    """

    def __init__(
        self,
        owner,
        active_obj: str,
        surface_obj: str,
        direction: str,
        reference_frame: str,
        distance_fraction: float = 1 / 3,
        z_tolerance: float = 0.02,
    ):
        super().__init__(owner.env)
        if any(not isinstance(s, str) or not s for s in (active_obj, surface_obj, reference_frame)):
            raise ValueError("Carry distance requires object, surface and reference frame")
        if direction not in ("leftof", "rightof"):
            raise ValueError("Carry direction must be leftof or rightof")
        for value, name in ((distance_fraction, "distance_fraction"), (z_tolerance, "z_tolerance")):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Carry {name} must be positive and finite")
        if distance_fraction > 1:
            raise ValueError("Carry distance_fraction must not exceed one")
        if owner._mode != "and" or not owner._sub_checkers:
            raise ValueError("Carry distance requires a nonempty AND MixedRules goal")
        self.owner = owner
        self.active_obj, self.surface_obj = active_obj, surface_obj
        self.reference_frame, self.direction = reference_frame, direction
        self.distance_fraction, self.z_tolerance = float(distance_fraction), float(z_tolerance)
        self._start = None
        self._axis = None
        self._surface_bounds = None
        self._full_distance = None
        self._distance = 0.0
        self._score = 0.0
        self.sealed = False
        self.complete = False
        self.last_measurement = {}

    def _initialize(self):
        frame = np.asarray(self.api_core.get_obj_world_pose_matrix(self.reference_frame), dtype=float)
        pose = self.get_obj_pose(self.active_obj)
        low, high = self.get_obj_aabb_new(self.surface_obj)
        if (
            frame.shape != (4, 4)
            or pose.shape != (4, 4)
            or not all(np.isfinite(a).all() for a in (frame, pose, low, high))
            or np.any(high <= low)
        ):
            raise ValueError("Invalid carry reference, object pose or surface bounds")
        axis = frame[:3, 1].copy()
        axis[2] = 0
        length = np.linalg.norm(axis)
        if length < 1e-8:
            raise ValueError("Carry reference must define a horizontal left/right axis")
        axis /= length
        self._full_distance = float(np.dot(np.abs(axis), high - low)) * self.distance_fraction
        if self._full_distance <= 1e-8:
            raise ValueError("Carry surface has no left/right span")
        self._axis = axis if self.direction == "leftof" else -axis
        self._start = pose[:3, 3].copy()
        self._surface_bounds = (low, high)

    def _sample(self):
        pose = self.get_obj_pose(self.active_obj)
        low, high = self.get_obj_aabb_new(self.active_obj)
        if (
            pose.shape != (4, 4)
            or not all(np.isfinite(a).all() for a in (pose, low, high))
            or np.any(high <= low)
        ):
            raise ValueError("Invalid measured carry geometry")
        surface_low, surface_high = self._surface_bounds
        gap = float(low[2] - surface_high[2])
        overlap = np.minimum(high[:2], surface_high[:2]) - np.maximum(low[:2], surface_low[:2])
        on_surface = abs(gap) <= self.z_tolerance and bool(np.all(overlap > 0))
        # Include the first detected touchdown position, then seal below.
        # Native sampling cannot separate carry from surface motion within
        # this final interval; later samples never contribute once sealed.
        self._distance = max(0.0, float(np.dot(pose[:3, 3] - self._start, self._axis)))
        self._score = min(self._distance / self._full_distance, 1.0)
        if self._distance + 1e-9 >= self._full_distance:
            self._score = 1.0
        self.owner._replay_placeholders()
        self.owner._sub_results = [
            self.owner._evaluate_sub_checker(entry, force_update=True)
            for entry in self.owner._sub_checkers
        ]
        self.last_measurement = {
            "sampling": "native",
            "checker_step": getattr(self.env, "current_step", None),
            "start_position": self._start.tolist(),
            "sample_position": pose[:3, 3].tolist(),
            "direction_axis": self._axis.tolist(),
            "surface_left_right_span": self._full_distance / self.distance_fraction,
            "full_distance": self._full_distance,
            "carried_distance": self._distance,
            "fraction": self._score,
            "z_gap": gap,
            "on_surface": on_surface,
            "sealed": on_surface,
            "policy_dispatch_count": getattr(self.env, "policy_dispatch_count", None),
            "object_aabb": [low.tolist(), high.tolist()],
            "surface_aabb": [surface_low.tolist(), surface_high.tolist()],
            "goal_results": list(self.owner._sub_results),
        }
        if on_surface:
            self.sealed = True
            self.complete = self._score == 1.0 and all(self.owner._sub_results)
            logger.info("[CarryDistanceSealed] " + json.dumps(self.last_measurement))

    def update(self, delta_time: float) -> float:
        if not self.owner.is_running():
            return 0.0
        if self._start is None:
            self._initialize()
        if not self.sealed:
            self._sample()
        self.owner._best_score = self._score
        self.owner._done_flag = self.complete
        if self.sealed and not self.complete:
            self.env.cancel_eval()
        return EvaluateAction.update(self.owner, delta_time)
