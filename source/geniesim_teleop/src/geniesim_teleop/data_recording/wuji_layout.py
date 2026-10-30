# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Named Wuji recording layout; finger positions stay in radians."""

import numpy as np

from geniesim_teleop.config.robot_interface import (
    BODY_JOINT_NAMES,
    HEAD_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)
from geniesim_teleop.wuji.contracts import HAND_JOINT_NAMES

JOINT_GROUPS = {
    "joint": tuple(LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES),
    "head": tuple(HEAD_JOINT_NAMES),
    "waist": tuple(BODY_JOINT_NAMES),
    "left_effector": HAND_JOINT_NAMES["left"],
    "right_effector": HAND_JOINT_NAMES["right"],
}


def joint_indices(names):
    names = [name.decode() if isinstance(name, bytes) else str(name) for name in names]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate Wuji recording joint names")
    lookup = {name: index for index, name in enumerate(names)}
    missing = [name for group in JOINT_GROUPS.values() for name in group if name not in lookup]
    if missing:
        raise ValueError(f"Incomplete Wuji recording; missing joints: {missing}")
    return {group: [lookup[name] for name in joints] for group, joints in JOINT_GROUPS.items()}


def reorder_state(msg):
    """Retain all measured joints, with a stable canonical prefix for conversion."""
    joint_indices(msg.name)
    required = [name for group in JOINT_GROUPS.values() for name in group]
    names = required + sorted(set(msg.name) - set(required))
    lookup = {name: index for index, name in enumerate(msg.name)}
    indices = [lookup[name] for name in names]
    arrays = []
    for field in ("position", "velocity", "effort"):
        values = np.asarray(getattr(msg, field), dtype=float)
        if field != "position" and not len(values):
            values = np.zeros(len(msg.name))
        if values.shape != (len(msg.name),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid Wuji recording joint {field}")
        arrays.append(values[indices].tolist())
    return names, *arrays
