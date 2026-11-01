# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Small self-contained GenieSim-style DSL for arranging external USD assets."""

from __future__ import annotations

import math
import random
import types
from dataclasses import dataclass

import numpy as np
from pxr import Usd, UsdGeom

from .asset_index import resolve_asset


@dataclass
class DslObject:
    asset_id: str
    matrix: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    keywords: list[str]


Shape = list[DslObject]


def translation_matrix(offset) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(offset, dtype=float)
    return matrix


def rotation_matrix(angle: float, direction=(0, 0, 1), point=(0, 0, 0)) -> np.ndarray:
    axis = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("Rotation axis must be nonzero")
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([
        [c + x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s, c + y*y*(1-c), y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c + z*z*(1-c)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    pivot = np.asarray(point, dtype=float)
    matrix[:3, 3] = pivot - rotation @ pivot
    return matrix


def scale_matrix(scale, origin=(0, 0, 0)) -> np.ndarray:
    values = np.asarray(scale, dtype=float)
    if values.ndim == 0:
        values = np.repeat(values, 3)
    if values.shape != (3,) or np.any(values <= 0):
        raise ValueError("Scale must be a positive scalar or three positive values")
    matrix = np.diag([*values, 1.0])
    pivot = np.asarray(origin, dtype=float)
    return translation_matrix(pivot) @ matrix @ translation_matrix(-pivot)


def identity_matrix() -> np.ndarray:
    return np.eye(4)


def quaternion_to_angle_direction(xyzw) -> tuple[float, np.ndarray]:
    values = np.asarray(xyzw, dtype=float)
    if values.shape != (4,):
        raise ValueError("Quaternion must have four xyzw values")
    norm = np.linalg.norm(values)
    if norm < 1e-12:
        return 0.0, np.array([1.0, 0.0, 0.0])
    x, y, z, w = values / norm
    w = float(np.clip(w, -1.0, 1.0))
    angle = 2 * math.acos(w)
    sine = math.sqrt(max(0.0, 1.0 - w*w))
    axis = np.array([1.0, 0.0, 0.0]) if sine < 1e-8 else np.array([x, y, z]) / sine
    return angle, axis


def transform_shape(shape: Shape, pose) -> Shape:
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError("A shape transform must be a 4x4 matrix")
    return [DslObject(obj.asset_id, pose @ obj.matrix, obj.bbox_min, obj.bbox_max,
                      obj.keywords.copy()) for obj in shape]


def concat_shapes(*shapes: Shape) -> Shape:
    result: Shape = []
    for shape in shapes:
        if isinstance(shape, DslObject):
            result.append(shape)
        else:
            result.extend(shape)
    return result


def _corners(obj: DslObject) -> np.ndarray:
    points = np.array([[x, y, z, 1.0]
                       for x in (obj.bbox_min[0], obj.bbox_max[0])
                       for y in (obj.bbox_min[1], obj.bbox_max[1])
                       for z in (obj.bbox_min[2], obj.bbox_max[2])])
    return (obj.matrix @ points.T).T[:, :3]


def _bounds(shape: Shape) -> tuple[np.ndarray, np.ndarray]:
    if not shape:
        raise ValueError("A DSL shape is empty")
    points = np.concatenate([_corners(obj) for obj in shape])
    return points.min(axis=0), points.max(axis=0)


def compute_shape_min(shape: Shape) -> np.ndarray:
    return _bounds(shape)[0]


def compute_shape_max(shape: Shape) -> np.ndarray:
    return _bounds(shape)[1]


def compute_shape_center(shape: Shape) -> np.ndarray:
    minimum, maximum = _bounds(shape)
    return (minimum + maximum) / 2


def compute_shape_sizes(shape: Shape) -> np.ndarray:
    minimum, maximum = _bounds(shape)
    return maximum - minimum


def get_object_info(shape: Shape) -> dict:
    minimum, maximum = _bounds(shape)
    if len(shape) == 1:
        center = shape[0].matrix[:3, 3].copy()
    else:
        center = (minimum + maximum) / 2
    return {"size": maximum - minimum, "min": minimum, "max": maximum,
            "center": center, "z_offset": float(center[2] - minimum[2]),
            "num_object": len(shape), "objects": [obj.asset_id for obj in shape]}


def _asset_bounds(record: dict) -> tuple[np.ndarray, np.ndarray]:
    for part in record["shapes"]:
        if part.get("name") == "bbox" and len(part.get("size", [])) == 3:
            center = np.asarray(part.get("position", (0, 0, 0)), dtype=float)
            half = np.asarray(part["size"], dtype=float) / 2
            return center - half, center + half
    stage = Usd.Stage.Open(str(record["path"]), load=Usd.Stage.LoadAll)
    if stage is None or not stage.GetDefaultPrim():
        raise ValueError(f"Cannot inspect asset bounds: {record['path']}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    if box.IsEmpty():
        raise ValueError(f"Empty asset bounds: {record['path']}")
    return np.asarray(box.GetMin(), dtype=float), np.asarray(box.GetMax(), dtype=float)


def _quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    """Convert a right-handed 3x3 rotation to a normalized xyzw quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [(rotation[2, 1] - rotation[1, 2]) / s,
             (rotation[0, 2] - rotation[2, 0]) / s,
             (rotation[1, 0] - rotation[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[i] = s / 4
        q[j] = (rotation[i, j] + rotation[j, i]) / s
        q[k] = (rotation[i, k] + rotation[k, i]) / s
        q[3] = (rotation[k, j] - rotation[j, k]) / s
    vector = np.asarray(q)
    return (vector / np.linalg.norm(vector)).tolist()


def add_dsl_objects(builder, shape: Shape, index: dict[str, dict]) -> None:
    if not shape:
        raise ValueError("root_scene() returned an empty scene")
    for obj in shape:
        record = resolve_asset(index, obj.asset_id)
        affine = obj.matrix[:3, :3]
        scale = np.linalg.norm(affine, axis=0)
        rotation = affine / scale
        if np.linalg.det(rotation) < 0 or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError(f"Asset {obj.asset_id} has shear or reflection; use a USD patch")
        semantic = record["description"].get("semantic_name", [])
        role = semantic[0] if isinstance(semantic, list) and semantic else obj.asset_id.rsplit("_", 1)[0]
        builder.add_asset(role, record["path"], xyz=obj.matrix[:3, 3],
                          xyzw=_quaternion_xyzw(rotation), scale=scale,
                          keywords=obj.keywords, description=record["description"])


def create_helper_module(index: dict[str, dict]) -> types.ModuleType:
    """Supply `from helper import *` without importing another source tree."""
    module = types.ModuleType("helper")
    registry = {}

    def register(docstring=None):
        def decorate(func):
            registry[func.__name__] = func
            return func
        return decorate

    def library_call(func_name: str, **kwargs) -> Shape:
        if func_name == "usd":
            asset_id = kwargs["oid"]
            record = resolve_asset(index, asset_id)
            minimum, maximum = _asset_bounds(record)
            pose = translation_matrix((0, 0, -minimum[2]))
            return [DslObject(asset_id, pose, minimum, maximum,
                              list(kwargs.get("keywords", [])))]
        if func_name not in registry:
            raise KeyError(f"Unregistered DSL function: {func_name}")
        return registry[func_name](**kwargs)

    def get_subpart_info(object_id: str, subpart_id: str) -> dict:
        record = resolve_asset(index, object_id)
        parts = record["shapes"]
        part = next((part for part in parts if subpart_id.lower() in part.get("name", "").lower()), None)
        if part is None:
            part = next((part for part in parts if part.get("name") == "bbox"), None)
        if part is None:
            raise ValueError(f"No {subpart_id} or bbox metadata for {object_id}")
        center = np.asarray(part["position"], dtype=float)
        size = np.asarray(part["size"], dtype=float)
        return {"center": center, "quat": part.get("quaternion", [0, 0, 0, 1]),
                "size": size, "xyz_min": center - size / 2, "xyz_max": center + size / 2}

    def loop(n, fn):
        return concat_shapes(*(fn(i) for i in range(n)))

    def if_else(condition, true_fn, false_fn):
        return true_fn() if condition else false_fn()

    exports = {"np": np, "math": math, "random": random, "Shape": Shape,
               "ASSETS_INDEX": index, "register": register, "library_call": library_call,
               "get_subpart_info": get_subpart_info, "get_object_info": get_object_info,
               "translation_matrix": translation_matrix, "rotation_matrix": rotation_matrix,
               "quaternion_to_angle_direction": quaternion_to_angle_direction,
               "scale_matrix": scale_matrix, "identity_matrix": identity_matrix,
               "transform_shape": transform_shape, "concat_shapes": concat_shapes,
               "compute_shape_min": compute_shape_min, "compute_shape_max": compute_shape_max,
               "compute_shape_center": compute_shape_center, "compute_shape_sizes": compute_shape_sizes,
               "loop": loop, "if_else": if_else}
    module.__dict__.update(exports)
    return module
