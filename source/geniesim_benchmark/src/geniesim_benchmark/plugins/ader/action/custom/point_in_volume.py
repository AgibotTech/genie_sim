# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License, v. 2.0
"""Continuous containment of an authored point in a moving, piecewise convex loft."""

from __future__ import annotations

import json
import threading

import numpy as np

from geniesim_benchmark.plugins.logger import Logger

from ..common_actions import ActionEvent, EvaluateAction
from .opening_geometry import affine_pose, convex_polygon, opening_slabs
from .precise import _Precise
from .precise_geometry import points, transformed

logger = Logger()


def point_in_volume(point: np.ndarray, axis: int, slabs: list, tolerance: float = 0.0) -> dict:
    """Contain a point, with an optional reference-local numerical allowance.

    The default retains the historical strictly-positive clearance rule.
    A task can explicitly allow surface contact and floating-point residuals;
    this does not change its hold duration or other insertion predicates.
    """
    coordinate = float(point[axis])
    axes = [i for i in range(3) if i != axis]
    axial = min(coordinate - slabs[0]["lower"], slabs[-1]["upper"] - coordinate)
    # Slight axial overshoot still needs a lateral test against the nearest
    # cap. Otherwise a resting tip just below the floor has no slab at all.
    sample_coordinate = (
        float(np.clip(coordinate, slabs[0]["lower"], slabs[-1]["upper"])) if tolerance > 0 else coordinate
    )
    lateral = None
    for slab in slabs:
        if slab["lower"] <= sample_coordinate <= slab["upper"]:
            fraction = (sample_coordinate - slab["lower"]) / (slab["upper"] - slab["lower"])
            support = slab["lower_support"] + fraction * (slab["upper_support"] - slab["lower_support"])
            clearance = float((support - point[axes] @ slab["normals"].T).min())
            lateral = clearance if lateral is None else min(lateral, clearance)
    passed = (
        axial > 0 and lateral is not None and lateral > 0
        if tolerance == 0
        else axial >= -tolerance and lateral is not None and lateral >= -tolerance
    )
    return {
        "pass": bool(passed),
        "point_reference_local_m": point.tolist(),
        "axial_clearance_m": axial,
        "wall_clearance_m": lateral,
        "depth_below_upper_m": slabs[-1]["upper"] - coordinate,
        "tolerance_m": tolerance,
    }


class PointInVolume(_Precise):
    """Require the current point to remain inside for a continuous physics interval.

    Sampling never latches an earlier pass. An exit, invalid pose, skipped frame,
    or unavailable physics clock clears the interval immediately. A native check
    also re-reads the current geometry before consuming the measured interval.
    """

    def __init__(
        self,
        env,
        object: str,
        reference: str,
        point: list,
        axis: int,
        slices: list,
        hold_seconds: float,
        tolerance: float = 0.0,
    ):
        super().__init__(env, object, reference)
        self.point = points([point])
        if type(axis) is not int or axis not in (0, 1, 2):
            raise ValueError("Volume axis must be 0, 1 or 2")
        if isinstance(hold_seconds, bool) or not np.isfinite(hold_seconds) or hold_seconds <= 0:
            raise ValueError("hold_seconds must be finite positive simulation seconds")
        if isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite nonnegative reference-local metres")
        if not isinstance(slices, list) or len(slices) < 2:
            raise ValueError("A volume requires at least two sections")
        self.slices = []
        for item in slices:
            if (
                set(item) != {"coordinate", "polygon"}
                or isinstance(item["coordinate"], bool)
                or not np.isfinite(item["coordinate"])
            ):
                raise ValueError("Each section requires a finite coordinate and convex polygon")
            self.slices.append(
                {"coordinate": float(item["coordinate"]), "polygon": convex_polygon(item["polygon"])}
            )
        if any(b["coordinate"] <= a["coordinate"] for a, b in zip(self.slices, self.slices[1:])):
            raise ValueError("Volume sections must increase strictly")
        self.axis, self.hold_seconds = axis, float(hold_seconds)
        self.tolerance = float(tolerance)
        self.slabs = opening_slabs(self.slices)
        self._sample_lock = threading.RLock()
        self._sampling_world = None
        self._callback_name = f"point-in-volume-{id(self):x}"
        self._last_physics_index = None
        self._inside_seconds = 0.0
        self._physics_ready = False
        self._composite_managed = False

    def set_composite_managed(self, managed: bool) -> None:
        """Keep observing a ready child until its composite finishes."""
        if not isinstance(managed, bool):
            raise ValueError("Composite ownership must be a boolean")
        self._composite_managed = managed

    def get_obj_pose(self, obj_name: str) -> np.ndarray:
        read = getattr(self.api_core, "get_obj_world_transform_matrix", None)
        if not callable(read):
            raise ValueError("Composed geometry transforms are unavailable")
        pose = np.asarray(read(self._analyze_obj_name(obj_name)), dtype=float)
        if not affine_pose(pose):
            raise ValueError("Invalid composed geometry transform")
        return pose

    def measure_current(self) -> dict:
        try:
            frame = self.get_obj_pose(self.passive_obj)
            relative = np.linalg.solve(frame, self.get_obj_pose(self.active_obj))
            return point_in_volume(
                transformed(self.point, relative)[0], self.axis, self.slabs, self.tolerance
            )
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError):
            return {"pass": False, "invalid_pose": True}

    def _clear_interval(self) -> None:
        self._inside_seconds = 0.0
        self._pass_frame = 0
        self._physics_ready = self._done_flag = False

    def _sample_physics(self, step_size: float) -> None:
        with self._sample_lock:
            if self._sampling_world is None:
                return
            index = self._sampling_world.current_time_step_index
            if index == self._last_physics_index:
                return
            was_ready = self._physics_ready
            if self._last_physics_index is not None and index != self._last_physics_index + 1:
                self._clear_interval()
            self._last_physics_index = index
            self.last_measurement = self.measure_current()
            if not self.last_measurement["pass"] or not np.isfinite(step_size) or step_size <= 0:
                self._clear_interval()
            else:
                if self._pass_frame:
                    self._inside_seconds += float(step_size)
                self._pass_frame += 1
                self._physics_ready = self._inside_seconds + 1e-10 >= self.hold_seconds
                self._done_flag = self._physics_ready
            self.last_measurement.update(
                physics_frame=int(index),
                consecutive_physics_frames=self._pass_frame,
                inside_seconds=self._inside_seconds,
                ready=self._physics_ready,
            )
            if was_ready != self._physics_ready:
                logger.info("[PointInVolumeState] " + json.dumps(self.last_measurement))

    def update(self, delta_time: float) -> float:
        with self._sample_lock:
            if self._sampling_world is None:
                world = getattr(getattr(self.api_core, "ui_builder", None), "my_world", None)
                if world is None or not callable(getattr(world, "add_physics_callback", None)):
                    self._clear_interval()
                    self.last_measurement = {"pass": False, "physics_sampling_unavailable": True}
                    return EvaluateAction.update(self, delta_time)
                self._sampling_world = world
                try:
                    world.add_physics_callback(self._callback_name, self._sample_physics)
                except Exception:
                    self.close()
                    raise
            current = self.measure_current()
            frame = self._sampling_world.current_time_step_index
            fresh = self._last_physics_index is not None and 0 <= frame - self._last_physics_index <= 1
            if not current["pass"] or not fresh:
                self._clear_interval()
            self._done_flag = self._physics_ready and current["pass"] and fresh
            self.last_measurement = dict(
                current,
                physics_frame=int(frame),
                inside_seconds=self._inside_seconds,
                consecutive_physics_frames=self._pass_frame,
                ready=bool(self._done_flag),
            )
            if self._done_flag:
                logger.info(
                    "[PointInVolumePass] "
                    + json.dumps(
                        {
                            "object": self.active_obj,
                            "reference": self.passive_obj,
                            "measurement": self.last_measurement,
                            "object_world_transform": self.get_obj_pose(self.active_obj).tolist(),
                            "reference_world_transform": self.get_obj_pose(self.passive_obj).tolist(),
                        }
                    )
                )
        return EvaluateAction.update(self, delta_time)

    def close(self) -> None:
        with self._sample_lock:
            world, self._sampling_world = self._sampling_world, None
        if world is not None:
            world.remove_physics_callback(self._callback_name)

    def handle_action_event(self, action, event) -> None:
        if event == ActionEvent.CANCELED or event == ActionEvent.FINISHED and not self._composite_managed:
            self.close()
        super().handle_action_event(action, event)
