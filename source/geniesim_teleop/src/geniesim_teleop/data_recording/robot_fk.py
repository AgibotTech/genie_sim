# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# SPDX-License-Identifier: MPL-2.0
"""Arm FK from the robot URDFs already used by data collection."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from geniesim_teleop.data_recording.derive_fields import chain_transform, load_link_chain

_ROBOT_CONFIGS = {
    "G1": ("G1/G1_omnipicker.urdf", "arm_{side}_link7"),
    "G2": ("G2/G2_omnipicker_from_usd.urdf", "arm_{side}_end_link"),
    "G2_crsB_wuji": ("G2/G2_crsB_wuji.urdf", "arm_{side}_end_link"),
    "G2_crsB": ("G2/G2_crsB_90D_from_usd.urdf", "arm_{side}_end_link"),
}
_ROBOT_DIR = Path(__file__).resolve().parent / "config" / "robot_cfg"


@lru_cache(maxsize=None)
def _arm_chains(robot: str):
    if robot not in _ROBOT_CONFIGS:
        raise ValueError(f"Unsupported FK robot: {robot}")
    relative_urdf, target_pattern = _ROBOT_CONFIGS[robot]
    urdf = str(_ROBOT_DIR / relative_urdf)
    chains = []
    for side in ("l", "r"):
        chain = load_link_chain(urdf, "arm_base_link", target_pattern.format(side=side))
        if not chain:
            raise ValueError(f"Missing arm {side} FK chain in {urdf}")
        active_names = [joint["name"] for joint in chain if joint["type"] in ("revolute", "continuous")]
        if len(active_names) != 7:
            raise ValueError(f"Expected 7 arm {side} joints in {urdf}, got {active_names}")
        chains.append((chain, active_names))
    return chains


def calculate_ee_pose_batch(joint_positions: np.ndarray, robot: str):
    """Return arm-end positions and xyzw quaternions in arm_base_link coordinates."""
    joints = np.asarray(joint_positions, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 14:
        raise ValueError(f"joint_positions must be (N, 14), got {joints.shape}")

    positions = np.empty((len(joints), 2, 3), dtype=np.float32)
    orientations = np.empty((len(joints), 2, 4), dtype=np.float32)
    for side_idx, (chain, active_names) in enumerate(_arm_chains(robot)):
        values = joints[:, side_idx * 7 : (side_idx + 1) * 7]
        for frame_idx, frame_joints in enumerate(values):
            matrix = chain_transform(chain, dict(zip(active_names, frame_joints)))
            positions[frame_idx, side_idx] = matrix[:3, 3]
            orientations[frame_idx, side_idx] = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return positions, orientations
