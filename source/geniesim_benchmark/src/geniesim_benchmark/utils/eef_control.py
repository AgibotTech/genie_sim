# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Shared CoRobot EEF conversion; effector and optional channels stay untouched."""

from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation as R


def eef_poses_to_joints(solver, left, right, joints, frame, arm_base_transform=None):
    """Convert xyz+xyzw pose chunks in the declared frame to arm joint chunks."""
    if frame not in ("base_link", "arm_base_link"):
        raise ValueError(f"Unsupported EEF frame: {frame}")
    arrays = []
    for values in (left, right):
        values = np.asarray(values)
        if values.dtype.kind not in "fiu" or values.ndim != 2 or values.shape[1] != 7 or len(values) == 0:
            raise ValueError("EEF values require a numeric chunk [H,7]")
        values = values.astype(float)
        if not np.isfinite(values).all() or np.any(np.linalg.norm(values[:, 3:], axis=1) < 1e-12):
            raise ValueError("EEF pose must be finite with a nonzero quaternion")
        positions, rotations = values[:, :3], R.from_quat(values[:, 3:])
        if frame == "base_link" and solver.frame == "arm_base_link":
            if arm_base_transform is None:
                raise RuntimeError("EEF_ABS requires arm_base_transform to convert to the IK frame")
            transform = np.asarray(arm_base_transform, dtype=float)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError("Invalid arm_base_transform")
            positions = (positions - transform[:3, 3]) @ transform[:3, :3]
            rotations = R.from_matrix(transform[:3, :3].T) * rotations
        arrays.append(np.c_[positions, rotations.as_euler("xyz")])
    if len(arrays[0]) != len(arrays[1]):
        raise ValueError("Left/right EEF horizons must match")
    targets = np.c_[arrays[0], arrays[1], np.zeros((len(arrays[0]), 2))]
    actions = solver.eef_actions_to_joint(targets, joints, [])
    return np.asarray(actions)[:, : 2 * solver.arm_dof]


def convert_eef_result(result, joints, *, robot_cfg="G2_crsB_wuji", arm_base_transform=None, solver=None):
    """Return a joint reply for EEF arms, preserving the original reply otherwise."""
    arms = [result.get(side + "_arm") for side in ("left", "right")]
    kinds = [arm.get("kind", "JOINT_ABS") if isinstance(arm, Mapping) else None for arm in arms]
    if "EEF_ABS" not in kinds:
        return result
    if kinds != ["EEF_ABS", "EEF_ABS"]:
        raise ValueError("Left/right arm kind must match")
    frames = [arm.get("base_link") or result.get("base_link") or "base_link" for arm in arms]
    if frames[0] != frames[1]:
        raise ValueError("Left/right EEF frames must match")
    from geniesim_benchmark.utils.ikfk_utils import get_shared_ikfk_solver

    if solver is None:
        solver = get_shared_ikfk_solver([], [], [], robot_cfg, "hard")
    converted = eef_poses_to_joints(solver, arms[0]["values"], arms[1]["values"], joints, frames[0], arm_base_transform)
    output = dict(result)
    for side, start in (("left", 0), ("right", solver.arm_dof)):
        output[side + "_arm"] = {"kind": "JOINT_ABS", "values": converted[:, start : start + solver.arm_dof].tolist()}
    return output
