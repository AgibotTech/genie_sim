# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from scipy.spatial.transform import Rotation as R


def get_quaternion_xyzw_from_rotation_matrix(rotation_matrix):
    """Convert rotation matrices to xyzw quaternions."""
    rot = R.from_matrix(rotation_matrix)
    quat = rot.as_quat()

    if quat.shape[0] == 4:
        quaternions_wxyz = quat[[0, 1, 2, 3]]
    else:
        quaternions_wxyz = quat[:, [0, 1, 2, 3]]
    return quaternions_wxyz


def get_rotation_matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = quat
    rot = np.array(
        [
            [2 * (w**2 + x**2) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 2 * (w**2 + y**2) - 1, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w**2 + z**2) - 1],
        ]
    )
    return rot


def get_pose(xyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = get_rotation_matrix_from_quaternion(quat_wxyz)
    pose[:3, 3] = xyz
    return pose
