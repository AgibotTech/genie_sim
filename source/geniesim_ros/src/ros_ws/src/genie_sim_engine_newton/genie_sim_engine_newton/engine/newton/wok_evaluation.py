# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Read-only, single-world chef evaluation and its NumPy state machine.

Sampling uses the solved state, not the asynchronous ROS/viewer mirrors.  Food
is grouped by the plugin's original soft-grid emissions: a noodle strand is
one object, as is a ham cube or vegetable strip.  No rendering, ROS, Newton or
Warp import is needed by the evaluator or its CPU tests.

Lift requires solved hand contact and no solved contact with static support.
Two points are sticky milestones, not an instruction to end the episode.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import json
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WokEvaluationConfig:
    """Thresholds supplied by the launcher's ``wok_evaluation_json`` key."""

    lift_height_m: float = 0.05
    stable_hold_s: float = 0.5
    toss_height_m: float = 0.10
    toss_object_fraction: float = 0.20
    min_food_upward_speed_m_s: float = 0.05
    max_tilt_deg: float = 45.0
    rim_xy_margin_m: float = 0.03
    in_pan_margin_m: float = 0.01

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} must be a finite number")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field.name} must be a finite nonnegative number")
        if not 0.0 < self.toss_object_fraction <= 1.0:
            raise ValueError("toss_object_fraction must be in (0, 1]")
        if not 0.0 < self.max_tilt_deg < 90.0:
            raise ValueError("max_tilt_deg must be in (0, 90)")
        if self.lift_height_m <= 0.0 or self.toss_height_m <= 0.0:
            raise ValueError("lift_height_m and toss_height_m must be positive")
        if self.min_food_upward_speed_m_s <= 0.0:
            raise ValueError("min_food_upward_speed_m_s must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> WokEvaluationConfig:
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError("wok evaluation thresholds must be an object")
        # Previously generated layout bundles contain these retired gates.
        # Accept their validated numeric values so those bundles remain usable;
        # the lift decision now depends only on contact, height and hold time.
        retired = {"max_linear_speed_m_s", "max_angular_speed_rad_s", "hand_handle_distance_m"}
        unknown = set(values) - {field.name for field in fields(cls)} - retired
        if unknown:
            raise ValueError(f"unknown wok evaluation thresholds: {sorted(unknown)}")
        current = dict(values)
        for name in retired & current.keys():
            value = current.pop(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        return cls(**current)

    @classmethod
    def from_json(cls, raw: str | None) -> WokEvaluationConfig:
        return cls.from_mapping(json.loads(raw) if raw else None)


def _array(value: Any, *, width: int | None = None) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.array(value, dtype=np.float64, copy=True)
    if width is not None and (result.ndim != 2 or result.shape[1] != width):
        raise ValueError(f"expected N x {width} state array, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("non-finite wok evaluation state")
    return result


def _pose(value: Any) -> np.ndarray:
    result = _array(value)
    if result.shape != (7,):
        raise ValueError(f"expected xyz+xyzw pose, got {result.shape}")
    length = float(np.linalg.norm(result[3:]))
    if length < 1e-8 or abs(length - 1.0) > 1e-3:
        raise ValueError("wok evaluation pose has an invalid quaternion")
    result[3:] /= length
    return result


def _rotation(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _transform(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ _rotation(pose[3:]).T + pose[:3]


@dataclass(frozen=True)
class WokEvaluationGeometry:
    table_top_z: float
    rest_rim_z: float
    rim_radius: float
    bowl_depth: float
    bowl_vertices: np.ndarray  # Actual thin-shell mesh, in wok body coordinates.
    handle_start: np.ndarray  # Graspable capsule center-line, wok coordinates.
    handle_end: np.ndarray
    handle_radius: float

    def __post_init__(self) -> None:
        for name in ("table_top_z", "rest_rim_z", "rim_radius", "bowl_depth", "handle_radius"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"invalid geometry {name}")
        if self.rim_radius <= 0.0 or self.bowl_depth <= 0.0 or self.handle_radius < 0.0:
            raise ValueError("wok geometry dimensions must be positive")
        vertices = _array(self.bowl_vertices, width=3)
        if len(vertices) < 3:
            raise ValueError("wok evaluation requires the actual bowl mesh")
        object.__setattr__(self, "bowl_vertices", vertices)
        for name in ("handle_start", "handle_end"):
            point = _array(getattr(self, name))
            if point.shape != (3,):
                raise ValueError(f"invalid geometry {name}")
            object.__setattr__(self, name, point)
        if np.linalg.norm(self.handle_end - self.handle_start) < 1e-8:
            raise ValueError("wok handle has zero length")


@dataclass(frozen=True)
class FoodObjectSample:
    name: str
    points: np.ndarray


@dataclass(frozen=True)
class WokEvaluationSample:
    sim_time: float
    reset_epoch: int
    wok_pose: np.ndarray
    hand_points: np.ndarray
    food_objects: tuple[FoodObjectSample, ...]
    hand_contact: bool
    support_contact: bool


@dataclass
class _FoodFlight:
    in_pan: bool = False
    active: bool = False
    valid: bool = False
    anchor_food_z: float = 0.0


class WokEvaluator:
    """Pure state machine; ``dt=0`` refreshes can detect failure only.

    Positive ``dt`` must correspond to a completed physics tick. Hold duration
    is bounded by both that dt and the actual simulation-time advancement.
    Refresh calls never arm a launch, earn a point, or advance the reference
    used to measure velocities. Epoch changes clear all score and history.
    """

    def __init__(self, geometry: WokEvaluationGeometry, config: WokEvaluationConfig | None = None) -> None:
        self.geometry = geometry
        self.config = config or WokEvaluationConfig()
        self.reset()

    def reset(self) -> None:
        self._epoch: int | None = None
        self._previous: WokEvaluationSample | None = None
        self._hold = 0.0
        self._was_stable = False
        self._initial_rim_z: float | None = None
        self._lifted = False
        self._tossed = False
        self._failed = False
        self._failure_reason: str | None = None
        self._flights: dict[str, _FoodFlight] = {}
        self._metrics: dict[str, Any] = {}

    def _validated(self, sample: WokEvaluationSample) -> WokEvaluationSample:
        if not math.isfinite(sample.sim_time) or sample.sim_time < 0.0:
            raise ValueError("invalid evaluation simulation time")
        if isinstance(sample.reset_epoch, bool) or not isinstance(sample.reset_epoch, (int, np.integer)):
            raise ValueError("invalid evaluation reset epoch")
        foods: list[FoodObjectSample] = []
        names: set[str] = set()
        for food in sample.food_objects:
            points = _array(food.points, width=3)
            if not food.name or food.name in names or len(points) == 0:
                raise ValueError("food objects must have unique names and nonempty particles")
            names.add(food.name)
            foods.append(FoodObjectSample(food.name, points))
        if not foods:
            raise ValueError("wok evaluation requires real food objects")
        hand = _array(sample.hand_points, width=3)
        if len(hand) == 0:
            raise ValueError("wok evaluation requires right-hand body positions")
        for name in ("hand_contact", "support_contact"):
            if not isinstance(getattr(sample, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be a boolean")
        return WokEvaluationSample(
            float(sample.sim_time),
            int(sample.reset_epoch),
            _pose(sample.wok_pose),
            hand,
            tuple(foods),
            bool(sample.hand_contact),
            bool(sample.support_contact),
        )

    def update(self, sample: WokEvaluationSample, *, dt: float) -> dict[str, Any]:
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("evaluation dt must be finite and nonnegative")
        sample = self._validated(sample)
        if sample.reset_epoch != self._epoch:
            self.reset()
            self._epoch = sample.reset_epoch
        prev = self._previous
        if prev is not None:
            if sample.sim_time < prev.sim_time:
                raise ValueError("simulation time went backwards without a reset epoch")
            old = {food.name: food.points.shape for food in prev.food_objects}
            if old != {food.name: food.points.shape for food in sample.food_objects}:
                raise ValueError("food object topology changed within an episode")

        g, c = self.geometry, self.config
        pose = sample.wok_pose
        rotation = _rotation(pose[3:])
        tilt = math.degrees(math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0))))
        rim_z = float(pose[2])
        if self._initial_rim_z is None:
            self._initial_rim_z = rim_z
        highest_rim = float(np.max(_transform(pose, g.bowl_vertices)[:, 2]))
        handle = _transform(pose, np.stack([g.handle_start, g.handle_end]))
        delta = handle[1] - handle[0]
        along = np.clip((sample.hand_points - handle[0]) @ delta / np.dot(delta, delta), 0.0, 1.0)
        hand_distance = max(
            0.0,
            float(np.min(np.linalg.norm(sample.hand_points - handle[0] - along[:, None] * delta, axis=1)))
            - g.handle_radius,
        )
        required = int(math.ceil(c.toss_object_fraction * len(sample.food_objects)))
        elapsed = min(float(dt), sample.sim_time - prev.sim_time) if prev is not None else 0.0
        linear: float | None = None
        angular: float | None = None
        if elapsed > 0.0 and prev is not None:
            actual_elapsed = sample.sim_time - prev.sim_time
            linear = float(np.linalg.norm(pose[:3] - prev.wok_pose[:3]) / actual_elapsed)
            dot = abs(float(np.dot(pose[3:], prev.wok_pose[3:])))
            angular = 2.0 * math.acos(min(1.0, dot)) / actual_elapsed
        self._metrics = {
            "rim_center_z": rim_z,
            "highest_rim_z": highest_rim,
            "table_top_z": float(g.table_top_z),
            "initial_rim_z": self._initial_rim_z,
            "lift_height_m": rim_z - self._initial_rim_z,
            "linear_speed_m_s": linear,
            "angular_speed_rad_s": angular,
            "hand_handle_distance_m": hand_distance,
            "hand_contact": sample.hand_contact,
            "support_contact": sample.support_contact,
            "tilt_deg": tilt,
            "stable_hold_s": self._hold,
            "food_object_count": len(sample.food_objects),
            "food_particle_count": sum(len(food.points) for food in sample.food_objects),
            "max_food_clearance_m": max(float(np.min(food.points[:, 2])) - highest_rim for food in sample.food_objects),
            "required_toss_object_count": required,
            "eligible_object_count": sum(f.in_pan or f.active and f.valid for f in self._flights.values()),
            "tossed_object_count": 0,
        }
        # Failure precedes all awards, including a would-be award on this tick.
        if rim_z < g.table_top_z:
            self._failed = True
            self._failure_reason = "wok_below_table"
        if self._failed:
            return self.progress()
        if prev is None:
            self._previous = sample
            return self.progress()
        if elapsed <= 0.0:
            return self.progress()

        had_lift = self._lifted
        stable = rim_z >= self._initial_rim_z + c.lift_height_m and sample.hand_contact and not sample.support_contact
        # The first qualifying sample starts the interval. The preceding tick
        # may have been below the lift threshold, so it is not stable time.
        self._hold = self._hold + elapsed if stable and self._was_stable else 0.0
        self._was_stable = stable
        if stable and self._hold + 1e-10 >= c.stable_hold_s:
            self._lifted = True

        previous_food = {food.name: food.points for food in prev.food_objects}
        tossed_names: list[str] = []
        eligible_count = 0
        # This is the actual spherical-cap cavity used by the chef plugin.
        sphere_radius = (g.rim_radius**2 + g.bowl_depth**2) / (2.0 * g.bowl_depth)
        center_z = sphere_radius - g.bowl_depth
        for food in sample.food_objects:
            points = food.points
            local = (points - pose[:3]) @ rotation
            radii = np.linalg.norm(local[:, :2], axis=1)
            surface = center_z - np.sqrt(np.maximum(0.0, sphere_radius**2 - radii**2))
            inside = bool(
                np.all(radii <= g.rim_radius)
                and np.all(local[:, 2] <= c.in_pan_margin_m)
                and np.all(local[:, 2] >= surface - c.in_pan_margin_m)
            )
            # A horizontal projection of the tilted rim is an ellipse. Invert
            # its 2x2 map; a 3D inverse would reject vertically rising food as
            # a side spill merely because the pan is pitched.
            near_rim = False
            if tilt <= c.max_tilt_deg:
                projected = np.linalg.solve(rotation[:2, :2], (points[:, :2] - pose[:2]).T).T
                near_rim = bool(np.all(np.linalg.norm(projected, axis=1) <= g.rim_radius + c.rim_xy_margin_m))
            flight = self._flights.setdefault(food.name, _FoodFlight())
            min_z = float(np.min(points[:, 2]))
            old_points = previous_food[food.name]
            actual_elapsed = sample.sim_time - prev.sim_time
            bottom_up = (min_z - float(np.min(old_points[:, 2]))) / actual_elapsed
            center_up = float(np.mean(points[:, 2] - old_points[:, 2])) / actual_elapsed
            if inside and tilt <= c.max_tilt_deg:
                flight.in_pan = True
                flight.active = False
                flight.valid = True
                flight.anchor_food_z = min_z
            else:
                if flight.in_pan:
                    flight.active = True
                    flight.valid = had_lift and near_rim
                flight.in_pan = False
                if flight.active:
                    if not near_rim or tilt > c.max_tilt_deg or bottom_up <= 0.0 or center_up <= 0.0:
                        flight.valid = False
            if flight.in_pan or flight.active and flight.valid:
                eligible_count += 1
            if (
                had_lift
                and flight.active
                and flight.valid
                and near_rim
                and min_z >= highest_rim + c.toss_height_m
                # A real world-space rise is required independently of
                # clearance. Merely lowering the pan cannot earn a toss;
                # pan recoil after launching food remains legitimate.
                and min_z - flight.anchor_food_z >= c.toss_height_m
                and bottom_up >= c.min_food_upward_speed_m_s
                and center_up >= c.min_food_upward_speed_m_s
            ):
                tossed_names.append(food.name)
        if had_lift and len(tossed_names) >= required:
            self._tossed = True
        self._metrics.update(
            stable_hold_s=self._hold,
            stable_lift_candidate=stable,
            eligible_object_count=eligible_count,
            tossed_object_count=len(tossed_names),
            tossed_object_names=tossed_names,
        )
        self._previous = sample
        return self.progress()

    def progress(self) -> dict[str, Any]:
        return {
            "score": int(self._lifted) + int(self._tossed),
            "max_score": 2,
            "milestones": {"stable_lift": self._lifted, "food_toss": self._tossed},
            "failed": self._failed,
            "failure_reason": self._failure_reason,
            "metrics": dict(self._metrics),
            "thresholds": asdict(self.config),
        }


class WokEvaluationSampler:
    """Extract chef geometry once and copy the solved state on demand.

    Supports the dynamic, round-handle, single-world MJWarp Wuji chef scene.
    Unsupported/missing geometry is an error rather than a guessed fallback.
    The adapter never edits the plugin, model, particles or lifecycle state.
    """

    def __init__(self, sim: Any, *, contacts: Any = None) -> None:
        self._sim = sim
        model = sim._model
        plugin = sim._scene_plugin
        if plugin is None or int(getattr(model, "world_count", 1)) != 1:
            raise ValueError("wok evaluation requires a single-world chef plugin")
        cfg = plugin.cfg
        if not cfg.wok.wok_dynamic or bool(getattr(plugin, "auto_flip", False)):
            raise ValueError("wok evaluation requires a dynamic, externally controlled wok")
        labels = [str(label).rsplit("/", 1)[-1] for label in model.body_label]
        self._pan_id = int(plugin.pan_id)
        if not 0 <= self._pan_id < len(labels) or labels[self._pan_id] != "chef_wok":
            raise ValueError("chef wok body index/label mismatch")
        self._hand_ids = np.array([i for i, name in enumerate(labels) if name.startswith("hand_r_")], dtype=np.int64)
        if not len(self._hand_ids):
            raise ValueError("no right Wuji hand bodies found")
        shape_labels = [str(label).rsplit("/", 1)[-1] for label in model.shape_label]
        shape_body = _array(model.shape_body).astype(np.int64)
        shape_poses = _array(model.shape_transform, width=7)
        shape_scales = _array(model.shape_scale, width=3)

        def shape_named(name: str) -> int:
            hits = [i for i, value in enumerate(shape_labels) if value == name]
            if len(hits) != 1:
                raise ValueError(f"expected one chef shape {name}, found {len(hits)}")
            return hits[0]

        bowl_id = shape_named("chef_wok_visual")
        grip_id = shape_named("chef_wok_grip_round")
        table_id = shape_named("chef_stove_base")
        if int(plugin.visual_shape) != bowl_id or int(plugin.grip_shape) != grip_id:
            raise ValueError("chef plugin shape bookkeeping mismatch")
        if shape_body[bowl_id] != self._pan_id or shape_body[grip_id] != self._pan_id:
            raise ValueError("wok shapes are not attached to the wok body")
        if shape_body[table_id] != -1:
            raise ValueError("chef table is not static")
        table_pose = _pose(shape_poses[table_id])
        if not np.allclose(_rotation(table_pose[3:]), np.eye(3), atol=1e-6):
            raise ValueError("chef table must be axis aligned")
        if np.any(shape_scales[table_id] <= 0.0):
            raise ValueError("invalid chef table box dimensions")
        source = model.shape_source[bowl_id]
        if source is None or not hasattr(source, "vertices"):
            raise ValueError("chef thin-shell bowl mesh is unavailable")
        vertices = _array(source.vertices, width=3) * shape_scales[bowl_id]
        vertices = _transform(_pose(shape_poses[bowl_id]), vertices)
        grip_pose = _pose(shape_poses[grip_id])
        radius, half_height = shape_scales[grip_id, :2]
        if radius <= 0.0 or half_height <= 0.0:
            raise ValueError("invalid chef handle capsule dimensions")
        handle = _transform(grip_pose, np.array([[0.0, 0.0, -half_height], [0.0, 0.0, half_height]]))
        self.geometry = WokEvaluationGeometry(
            table_top_z=float(table_pose[2] + shape_scales[table_id, 2]),
            rest_rim_z=float(cfg.wok.home_z),
            rim_radius=float(cfg.wok.rim_r),
            bowl_depth=float(cfg.wok.depth),
            bowl_vertices=vertices,
            handle_start=handle[0],
            handle_end=handle[1],
            handle_radius=float(radius),
        )
        self._body_count = len(labels)
        self._particle_count = int(model.particle_count)
        self._food_ranges = self._object_ranges(plugin, self._particle_count)
        if contacts is None:
            from .wok_contacts import WokContactSampler

            contacts = WokContactSampler(sim, pan_id=self._pan_id, hand_ids=self._hand_ids)
        self._contacts = contacts

    @staticmethod
    def _object_ranges(plugin: Any, particle_count: int) -> tuple[tuple[str, int, int], ...]:
        cfg = plugin.cfg
        starts = np.asarray(plugin.starts_np, dtype=np.int64).reshape(-1)
        n_noodles = int(cfg.noodle.n_noodles)
        per_noodle = 4 * (int(plugin.n_len_noodle) + 1)
        if n_noodles <= 0 or len(starts) != n_noodles or per_noodle <= 4:
            raise ValueError("chef noodle emission bookkeeping is unavailable")
        if np.any(starts < 0) or (len(starts) > 1 and not np.all(np.diff(starts) == per_noodle)):
            raise ValueError("chef noodle particle ranges are not contiguous")
        ranges = [(f"noodle_{i}", int(start), int(start) + per_noodle) for i, start in enumerate(starts)]
        cursor = ranges[-1][2]
        ham_size = (int(cfg.ham.dim) + 1) ** 3
        for i in range(int(cfg.ham.n_cubes)):
            ranges.append((f"ham_{i}", cursor, cursor + ham_size))
            cursor += ham_size
        for name in ("pepper", "carrot"):
            params = getattr(cfg, name)
            count = int(params.n)
            if count and int(getattr(plugin, f"{name}_pstart")) != cursor:
                raise ValueError(f"chef {name} particle start mismatch")
            size = 4 * (int(params.n_len) + 1)
            if count < 0 or size <= 4:
                raise ValueError(f"invalid chef {name} object dimensions")
            for i in range(count):
                ranges.append((f"{name}_{i}", cursor, cursor + size))
                cursor += size
        if cursor > particle_count:
            raise ValueError("chef food particle ranges exceed the actual model")
        return tuple(ranges)

    def sample(self) -> WokEvaluationSample:
        state = self._sim._state_0
        poses = _array(state.body_q, width=7)
        particles = _array(state.particle_q, width=3)
        if len(poses) != self._body_count or len(particles) != self._particle_count:
            raise ValueError("chef evaluation state topology changed")
        hand_contact, support_contact = self._contacts.sample()
        return WokEvaluationSample(
            sim_time=float(self._sim._sim_time_acc),
            reset_epoch=int(self._sim._reset_epoch),
            wok_pose=poses[self._pan_id].copy(),
            hand_points=poses[self._hand_ids, :3].copy(),
            food_objects=tuple(
                FoodObjectSample(name, particles[start:end].copy()) for name, start, end in self._food_ranges
            ),
            hand_contact=hand_contact,
            support_contact=support_contact,
        )
