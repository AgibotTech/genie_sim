# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_rotate(quaternion, axis, angle):
    """
    Rotate a quaternion around a specified axis by a given angle.

    Parameters:
    quaternion (numpy array): The input quaternion [w, x, y, z].
    axis (str): The axis to rotate around ('x', 'y', or 'z').
    angle (float): The rotation angle in degrees.

    Returns:
    numpy array: The rotated quaternion.
    """
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle)

    # Calculate the rotation quaternion based on the specified axis
    cos_half_angle = np.cos(angle_rad / 2)
    sin_half_angle = np.sin(angle_rad / 2)

    if axis == "x":
        q_axis = np.array([cos_half_angle, sin_half_angle, 0, 0])
    elif axis == "y":
        q_axis = np.array([cos_half_angle, 0, sin_half_angle, 0])
    elif axis == "z":
        q_axis = np.array([cos_half_angle, 0, 0, sin_half_angle])
    else:
        raise ValueError("Unsupported axis. Use 'x', 'y', or 'z'.")

    # Extract components of the input quaternion
    w1, x1, y1, z1 = quaternion
    w2, x2, y2, z2 = q_axis

    # Quaternion multiplication
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def axis_to_quaternion(axis, target_axis="y", up_side_down=False):
    """
    Calculate the quaternion that rotates a given axis to the target axis.

    Parameters:
    axis (str): The axis in the object's local coordinate system ('x', 'y', or 'z').
    target_axis (str): The target axis in the world coordinate system ('x', 'y', or 'z').

    Returns:
    numpy array: The quaternion representing the rotation.
    """
    # Define unit vectors for each axis
    unit_vectors = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }

    if axis not in unit_vectors or target_axis not in unit_vectors:
        raise ValueError("Unsupported axis. Use 'x', 'y', or 'z'.")

    if axis == "z" and up_side_down:
        # Special case: 180 degree rotation around x or y axis
        return np.array([0, 1, 0, 0])  # 180 degree rotation around x-axis

    v1 = unit_vectors[axis] * (-1 if up_side_down else 1)  # Flip the axis if up_side_down is True
    v2 = unit_vectors[target_axis]

    # Calculate the cross product and dot product
    cross_prod = np.cross(v1, v2)
    dot_prod = np.dot(v1, v2)

    # Calculate the quaternion
    w = np.sqrt((np.linalg.norm(v1) ** 2) * (np.linalg.norm(v2) ** 2)) + dot_prod
    x, y, z = cross_prod

    # Normalize the quaternion
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)

    return q


def is_y_axis_up(pose_matrix):
    """
    Determine if the object's y-axis is pointing upward in the world coordinate system.

    Args:
        pose_matrix (numpy.ndarray): 4x4 homogeneous transformation matrix.

    Returns:
        bool: True if y-axis is upward, False if y-axis is downward.
    """
    # Extract the second column of the rotation matrix
    y_axis_vector = pose_matrix[:3, 1]

    # World coordinate system y-axis
    world_y_axis = np.array([0, 1, 0])

    # Calculate dot product
    dot_product = np.dot(y_axis_vector, world_y_axis)

    # Return True if upward, otherwise False
    return dot_product > 0


def is_local_axis_facing_world_axis(pose_matrix, local_axis="y", world_axis="z"):
    # Define local coordinate system axis indices
    local_axis_index = {"x": 0, "y": 1, "z": 2}

    # Define world coordinate system axis vectors
    world_axes = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }

    # Extract specified axis of local coordinate system
    local_axis_vector = pose_matrix[:3, local_axis_index[local_axis]]

    # Get specified axis vector of world coordinate system
    world_axis_vector = world_axes[world_axis]

    # Calculate dot product
    dot_product = np.dot(local_axis_vector, world_axis_vector)

    # Return True if facing specified world axis, otherwise False
    return dot_product > 0


def rotate_180_along_axis(target_affine, rot_axis="z"):
    """
    Gripper is a symmetric structure, rotating 180 degrees around Z-axis is equivalent.
    Choose the target pose closer to current pose to avoid unnecessary rotation.
    """
    if rot_axis == "z":
        R_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif rot_axis == "y":
        R_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    elif rot_axis == "x":
        R_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        assert False, "Invalid rotation axis. Please choose from 'x', 'y', 'z'."

    # Extract rotation part (3x3 matrix)
    target_rotation = target_affine[:3, :3]
    # Rotate target_rotation 180 degrees around its own Z-axis to get target_rotation_2
    target_rotation_2 = np.dot(target_rotation, R_180)

    # Recombine rotation matrix target_rotation_2 with original translation part
    target_affine_2 = np.eye(4)
    target_affine_2[:3, :3] = target_rotation_2
    target_affine_2[:3, 3] = target_affine[:3, 3]  # Preserve original translation part
    return target_affine_2


def rotate_along_axis(target_affine, angle_degrees, rot_axis="z", use_local=True):
    """
    Rotate target_affine according to specified angle and rotation axis.

    Args:
        target_affine: 4x4 affine transformation matrix
        angle_degrees: Rotation angle (in degrees)
        rot_axis: Rotation axis, 'x', 'y', or 'z'
    """
    # Convert angle to radians
    angle_radians = np.deg2rad(angle_degrees)

    # Create rotation object
    if rot_axis == "z":
        rotation_vector = np.array([0, 0, angle_radians])
    elif rot_axis == "y":
        rotation_vector = np.array([0, angle_radians, 0])
    elif rot_axis == "x":
        rotation_vector = np.array([angle_radians, 0, 0])
    else:
        raise ValueError("Invalid rotation axis. Please choose from 'x', 'y', 'z'.")

    # Generate rotation matrix
    R_angle = R.from_rotvec(rotation_vector).as_matrix()

    # Extract rotation part (3x3 matrix)
    target_rotation = target_affine[:3, :3]

    # Rotate target_rotation around specified axis by specified angle to get target_rotation_2
    if use_local:
        target_rotation_2 = np.dot(target_rotation, R_angle)
    else:
        target_rotation_2 = np.dot(R_angle, target_rotation)

    # Recombine rotation matrix target_rotation_2 with original translation part
    target_affine_2 = np.eye(4)
    target_affine_2[:3, :3] = target_rotation_2
    target_affine_2[:3, 3] = target_affine[:3, 3]  # Preserve original translation part

    return target_affine_2


# @nb.jit(nopython=True)
def skew(vector: np.ndarray) -> np.ndarray:
    """Convert vector to skew symmetric matrix.

    This function returns a skew-symmetric matrix to perform cross-product
    as a matrix multiplication operation, i.e.:

        np.cross(a, b) = np.dot(skew(a), b)


    Args:
        vector (np.ndarray): A 3x1 vector.

    Returns:
        np.ndarray: The resluting skew-symmetric matrix.
    """
    mat = np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )
    return mat
