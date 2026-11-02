# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Generic local-frame checks. Geometry, object names and tolerances are task data."""

from __future__ import annotations

import numpy as np

from ..common_actions import EvaluateAction, ActionEvent
from .precise_geometry import points, rigid_pose, transformed, containment, coverage


class _Precise(EvaluateAction):
    def __init__(self, env, object: str, reference: str):
        super().__init__(env)
        if not isinstance(object, str) or not object or not isinstance(reference, str) or not reference:
            raise ValueError("object and reference must be nonempty names")
        self.active_obj, self.passive_obj = object, reference
        self._done_flag, self._pass_frame = False, 0
        self.last_measurement = {}

    def relative_pose(self, name: str, frame: np.ndarray) -> np.ndarray:
        pose = np.asarray(self.get_obj_pose(name), dtype=float)
        if not rigid_pose(pose):
            raise ValueError("Invalid measured rigid pose")
        return np.linalg.solve(frame, pose)

    def update(self, delta_time: float) -> float:
        try:
            frame = np.asarray(self.get_obj_pose(self.passive_obj), dtype=float)
            if not rigid_pose(frame):
                raise ValueError("Invalid reference rigid pose")
            self.last_measurement = self.measure(frame)
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError):
            self.last_measurement = {"pass": False, "invalid_pose": True}
        self._pass_frame = self._pass_frame + 1 if self.last_measurement["pass"] else 0
        if self._pass_frame >= 2:
            self._done_flag = True
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action, event):
        if event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1


class InsidePrecise(_Precise):
    """Contain supplied local collision samples in a reference-local region."""

    def __init__(
        self,
        env,
        object: str,
        reference: str,
        points: list,
        lower: list,
        upper: list,
        quantifier: str = "all",
    ):
        super().__init__(env, object, reference)
        from .precise_geometry import points as validate_points

        self.points = validate_points(points)
        if quantifier not in ("all", "any") or len(lower) != 3 or len(upper) != 3:
            raise ValueError("Expected all/any and three boundaries per side")
        if all(v is None for v in list(lower) + list(upper)):
            raise ValueError("At least one boundary must be enabled")
        for lo, hi in zip(lower, upper):
            if (
                any(v is not None and not np.isfinite(v) for v in (lo, hi))
                or lo is not None
                and hi is not None
                and lo > hi
            ):
                raise ValueError("Invalid containment interval")
        self.lower, self.upper, self.quantifier = list(lower), list(upper), quantifier

    def measure(self, frame: np.ndarray) -> dict:
        return containment(
            transformed(self.points, self.relative_pose(self.active_obj, frame)),
            self.lower,
            self.upper,
            self.quantifier,
        )


class CoverPrecise(_Precise):
    """Cover a target projection, optionally bounding surface height and facing."""

    def __init__(
        self,
        env,
        object: str,
        reference: str,
        source_points: list,
        target_points: list,
        up_axis: int,
        threshold: float,
        target_object: str | None = None,
        max_gap: float | None = None,
        normal_axis: int | None = None,
        min_normal_dot: float | None = None,
    ):
        super().__init__(env, object, reference)
        self.source_points, self.target_points = points(source_points), points(target_points)
        if (
            type(up_axis) is not int
            or up_axis not in (0, 1, 2)
            or isinstance(threshold, bool)
            or not np.isfinite(threshold)
            or not 0 < threshold <= 1
        ):
            raise ValueError("Expected a coordinate axis and coverage threshold in (0, 1]")
        if max_gap is not None and not np.isfinite(max_gap):
            raise ValueError("max_gap must be finite metres")
        if (
            (normal_axis is None) != (min_normal_dot is None)
            or normal_axis is not None
            and (
                type(normal_axis) is not int
                or normal_axis not in (0, 1, 2)
                or not np.isfinite(min_normal_dot)
                or not -1 <= min_normal_dot < 1
            )
        ):
            raise ValueError("Facing requires a local normal axis and minimum dot product")
        if target_object is not None and (not isinstance(target_object, str) or not target_object):
            raise ValueError("target_object must be a nonempty name")
        self.up_axis, self.threshold = up_axis, threshold
        self.target_object = target_object or reference
        self.max_gap, self.normal_axis, self.min_normal_dot = max_gap, normal_axis, min_normal_dot

    def measure(self, frame: np.ndarray) -> dict:
        relative = self.relative_pose(self.active_obj, frame)
        target_pose = (
            np.eye(4)
            if self.target_object == self.passive_obj
            else self.relative_pose(self.target_object, frame)
        )
        source = transformed(self.source_points, relative)
        target = transformed(self.target_points, target_pose)
        normal = float(relative[self.up_axis, self.normal_axis]) if self.normal_axis is not None else None
        return coverage(
            source, target, self.up_axis, self.threshold, self.max_gap, normal, self.min_normal_dot
        )
