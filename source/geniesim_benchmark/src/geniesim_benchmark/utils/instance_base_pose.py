# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License 2.0

"""Resolve explicitly authored base poses without resampling other tasks."""

from collections.abc import Mapping
from math import isfinite, sqrt
from typing import Any


def instance_base_pose(task_config: Mapping[str, Any]) -> dict[str, list[float]] | None:
    """Return a fresh pose for the selected foreground, or retain legacy sampling.

    A configured map must cover the requested instance. Falling back to one
    shared pre-generated pose would silently reintroduce duplicate starts.
    """
    init_base = task_config.get("generalization", {}).get("init_base", {})
    poses = init_base.get("poses_by_instance")
    if not init_base.get("enable", False) or poses is None:
        return None
    instance = task_config.get("scene", {}).get("scene_instance_id")
    if isinstance(instance, bool) or not isinstance(instance, int) or instance < 0:
        raise ValueError("poses_by_instance requires a nonnegative scene_instance_id")
    if not isinstance(poses, Mapping) or str(instance) not in poses:
        raise ValueError(f"poses_by_instance has no pose for instance {instance}")
    value = poses[str(instance)]
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid base pose for instance {instance}")
    pose = {}
    for key, size in (("position", 3), ("quaternion", 4)):
        numbers = value.get(key)
        if not isinstance(numbers, (list, tuple)) or len(numbers) != size:
            raise ValueError(f"instance {instance}: {key} needs {size} numbers")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x) for x in numbers):
            raise ValueError(f"instance {instance}: {key} must be finite numbers")
        pose[key] = [float(x) for x in numbers]
    if abs(sqrt(sum(x * x for x in pose["quaternion"])) - 1.0) > 1e-5:
        raise ValueError(f"instance {instance}: quaternion must have unit length")
    return pose
