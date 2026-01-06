# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from scipy.spatial.transform import Rotation


def batch_matrices_to_quaternions_scipy(pose_matrices):
    """
    Convert batch 4x4 pose matrices to quaternions using Scipy's Rotation library

    Args:
        pose_matrices: numpy array of shape (n, 4, 4)

    Returns:
        numpy array of shape (n, 4), each row is a quaternion [x, y, z, w] (Scipy default order)
    """
    # Extract rotation matrix part (n, 3, 3)
    rotation_matrices = pose_matrices[:, :3, :3]

    # Create Rotation object and convert to quaternion
    rot = Rotation.from_matrix(rotation_matrices)
    quaternions = rot.as_quat()  # Returns (n, 4) array, order is [x, y, z, w]

    return quaternions


def batch_matrices_to_quaternions_scipy_w_first(pose_matrices):
    """
    Use Scipy's Rotation library, return quaternions in [w, x, y, z] order

    Args:
        pose_matrices: numpy array of shape (n, 4, 4)

    Returns:
        numpy array of shape (n, 4), each row is a quaternion [w, x, y, z]
    """
    quaternions = batch_matrices_to_quaternions_scipy(pose_matrices)
    # Adjust order to [w, x, y, z]
    quaternions_w_first = np.zeros_like(quaternions)
    quaternions_w_first[:, 0] = quaternions[:, 3]  # w
    quaternions_w_first[:, 1:] = quaternions[:, :3]  # x, y, z
    return quaternions_w_first
