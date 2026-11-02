# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License, v. 2.0
"""Native lifecycle for generic convex material passing through an opening."""

from __future__ import annotations

import json
import threading

import numpy as np

from geniesim_benchmark.plugins.logger import Logger

from ..common_actions import ActionEvent, EvaluateAction
from .opening_geometry import affine_pose, convex_polygon, opening_slabs, through_opening
from .precise import _Precise
from .precise_geometry import points, transformed

logger = Logger()


class ThroughOpening(_Precise):
    def __init__(
        self,
        env,
        object: str,
        reference: str,
        source_points: list,
        axis: int,
        slices: list,
        tolerance: float = 1e-5,
        sampling: str = "native",
        required_samples: int = 2,
    ):
        super().__init__(env, object, reference)
        self.source_points = points(source_points)
        if (
            len(self.source_points) < 4
            or np.linalg.matrix_rank(self.source_points - self.source_points.mean(axis=0)) < 3
        ):
            raise ValueError("Source samples must span a three-dimensional convex hull")
        if type(axis) is not int or axis not in (0, 1, 2):
            raise ValueError("Opening axis must be 0, 1 or 2")
        if not isinstance(slices, list) or len(slices) < 2:
            raise ValueError("An opening requires at least two aperture sections")
        if isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Opening tolerance must be finite nonnegative metres")
        self.slices = []
        for item in slices:
            if (
                set(item) != {"coordinate", "polygon"}
                or isinstance(item["coordinate"], bool)
                or not np.isfinite(item["coordinate"])
            ):
                raise ValueError("Each section requires a finite coordinate and a convex polygon")
            self.slices.append(
                {"coordinate": float(item["coordinate"]), "polygon": convex_polygon(item["polygon"])}
            )
        if any(b["coordinate"] <= a["coordinate"] for a, b in zip(self.slices, self.slices[1:])):
            raise ValueError("Aperture sections must have strictly increasing coordinates")
        self.axis, self.tolerance = axis, float(tolerance)
        self.slabs = opening_slabs(self.slices)
        if sampling not in ("native", "physics"):
            raise ValueError("Sampling must be native or physics")
        self.sampling = sampling
        if type(required_samples) is not int or required_samples < 1:
            raise ValueError("required_samples must be a positive integer")
        self.required_samples = required_samples
        self._sample_lock = threading.RLock()
        self._sampling_world = None
        self._callback_name = f"through-opening-{id(self):x}"
        self._last_physics_index = None
        self._physics_done = False

    def get_obj_pose(self, obj_name: str) -> np.ndarray:
        # The ordinary pose API intentionally drops scale. Opening sections
        # need the composed affine transform, including nonuniform ancestors.
        read = getattr(self.api_core, "get_obj_world_transform_matrix", None)
        if not callable(read):
            raise ValueError("Composed geometry transforms are unavailable")
        return np.asarray(read(self._analyze_obj_name(obj_name)), dtype=float)

    def relative_pose(self, name: str, frame: np.ndarray) -> np.ndarray:
        pose = self.get_obj_pose(name)
        if not affine_pose(pose):
            raise ValueError("Invalid measured object transform")
        return np.linalg.solve(frame, pose)

    def _sample(self) -> None:
        try:
            frame = self.get_obj_pose(self.passive_obj)
            if not affine_pose(frame):
                raise ValueError("Invalid reference transform")
            self.last_measurement = self.measure(frame)
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError):
            self.last_measurement = {"pass": False, "invalid_pose": True}
        self._pass_frame = self._pass_frame + 1 if self.last_measurement["pass"] else 0
        if self._pass_frame >= self.required_samples:
            self._done_flag = True

    def _sample_physics(self, step_size: float = 0.0) -> None:
        with self._sample_lock:
            if self._physics_done or self._sampling_world is None:
                return
            index = self._sampling_world.current_time_step_index
            if index == self._last_physics_index:
                return
            if self._last_physics_index is not None and index != self._last_physics_index + 1:
                self._pass_frame = 0
            self._last_physics_index = index
            self._sample()
            self._physics_done = self._done_flag
            self.last_measurement["physics_frame"] = int(index)
            self.last_measurement["consecutive_physics_frames"] = self._pass_frame
            if self._physics_done:
                logger.info(
                    "[ThroughOpeningPass] "
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

    def update(self, delta_time: float) -> float:
        with self._sample_lock:
            if self.sampling == "physics":
                if not self._physics_done and self._sampling_world is None:
                    world = getattr(getattr(self.api_core, "ui_builder", None), "my_world", None)
                    if world is None or not callable(getattr(world, "add_physics_callback", None)):
                        self.last_measurement = {"pass": False, "physics_sampling_unavailable": True}
                        return EvaluateAction.update(self, delta_time)
                    self._sampling_world = world
                    try:
                        world.add_physics_callback(self._callback_name, self._sample_physics)
                    except Exception:
                        self.close()
                        raise
                self._done_flag = self._physics_done
            else:
                self._sample()
        return EvaluateAction.update(self, delta_time)

    def close(self) -> None:
        with self._sample_lock:
            world, self._sampling_world = self._sampling_world, None
        if world is not None:
            world.remove_physics_callback(self._callback_name)

    def handle_action_event(self, action, event) -> None:
        if event in (ActionEvent.FINISHED, ActionEvent.CANCELED):
            self.close()
        super().handle_action_event(action, event)

    def measure(self, frame: np.ndarray) -> dict:
        return through_opening(
            transformed(self.source_points, self.relative_pose(self.active_obj, frame)),
            self.axis,
            self.slices,
            self.tolerance,
            self.slabs,
        )
