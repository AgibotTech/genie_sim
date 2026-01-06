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


def axis_to_quaternion(axis, target_axis="y"):
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

    v1 = unit_vectors[axis]
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
    is obj y-axis pointing upwards in global cordinates

    arg:
    pose_matrix (numpy.ndarray): 4x4 Homogeneous Transformation Matrix (HTM)

    return:
    bool: True if y-axis pointing up False y-axis pointing down
    """
    # 2nd row
    y_axis_vector = pose_matrix[:3, 1]

    # world y-axis
    world_y_axis = np.array([0, 1, 0])

    dot_product = np.dot(y_axis_vector, world_y_axis)

    return dot_product > 0


def is_local_axis_facing_world_axis(pose_matrix, local_axis="y", world_axis="z"):
    local_axis_index = {"x": 0, "y": 1, "z": 2}

    world_axes = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }

    # local axis
    local_axis_vector = pose_matrix[:3, local_axis_index[local_axis]]

    # global axis
    world_axis_vector = world_axes[world_axis]

    dot_product = np.dot(local_axis_vector, world_axis_vector)

    return dot_product > 0


def rotate_180_along_axis(target_affine, rot_axis="z"):
    """
    The gripper is a symmetrical structure, and it is equivalent to rotate 180 degrees around the Z axis. \
    Select a target pose closer to the current pose to avoid unnecessary rotation
    """
    if rot_axis == "z":
        R_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif rot_axis == "y":
        R_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    elif rot_axis == "x":
        R_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        assert False, "Invalid rotation axis. Please choose from 'x', 'y', 'z'."

    single_mode = pose.ndim == 2
    if single_mode:
        pose = pose[np.newaxis, :, :]
    R_180 = np.tile(R_180[np.newaxis, :, :], (pose.shape[0], 1, 1))
    pose[:, :3, :3] = pose[:, :3, :3] @ R_180

    if single_mode:
        pose = pose[0]

    return pose


def rotate_along_axis(target_affine, angle_degrees, rot_axis="z", use_local=True):
    """
    Rotate target_affine according to the specified angle and rotation axis.
    parameter:
    -target_affine: 4x4 affine transformation matrix
    -angle_degrees: rotation angle (in degrees)
    -rot_axis: rotation axis, 'x', 'y' or 'z'
    """
    # Convert angle to radians
    angle_radians = np.deg2rad(angle_degrees)

    # Create a rotating object
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
    # Extract the rotation part (3x3 matrix)
    target_rotation = target_affine[:3, :3]

    # Rotate the target_rotation about the specified axis to obtain target_rotation_2
    if use_local:
        target_rotation_2 = np.dot(target_rotation, R_angle)
    else:
        target_rotation_2 = np.dot(R_angle, target_rotation)

    # Recombind the rotation matrix target_rotation_2 and the original translation part
    target_affine_2 = np.eye(4)
    target_affine_2[:3, :3] = target_rotation_2
    target_affine_2[:3, 3] = target_affine[:3, 3]  # Keep the original translation part

    return target_affine_2


import numpy as np
from scipy.spatial.transform import Rotation as R


def get_quaternion_wxyz_from_rotation_matrix(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to a quaternion in the wxyz format.

    Parameters:
    R (numpy array): A 3x3 rotation matrix.

    Returns:
    numpy array: A 4x1 quaternion in the wxyz format.
    """
    # Convert the rotation matrix to a quaternion
    rot = R.from_matrix(rotation_matrix)
    quat = rot.as_quat()

    # Reorder the quaternion to the wxyz format
    if quat.shape[0] == 4:
        quaternions_wxyz = quat[[3, 0, 1, 2]]
    else:
        quaternions_wxyz = quat[:, [3, 0, 1, 2]]
    return quaternions_wxyz


def get_quaternion_from_rotation_matrix(R):
    assert R.shape == (3, 3)

    # Calculate quaternion components
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])


# @nb.jit(nopython=True)
def get_rotation_matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a rotation matrix.

    Args:
        quat (np.ndarray): A 4x1 vector in order (w, x, y, z)

    Returns:
        np.ndarray: The resulting 3x3 rotation matrix.
    """
    w, x, y, z = quat
    rot = np.array(
        [
            [2 * (w**2 + x**2) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 2 * (w**2 + y**2) - 1, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w**2 + z**2) - 1],
        ]
    )
    return rot


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert input quaternion to rotation matrix.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < 1e-10:
        return np.identity(3)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
            (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
            (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
        ),
        dtype=np.float64,
    )


# @nb.jit(nopython=True)
def get_xyz_euler_from_quaternion(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion to XYZ euler angles.

    Args:
        quat (np.ndarray): A 4x1 vector in order (w, x, y, z).

    Returns:
        np.ndarray: A 3x1 vector containing (roll, pitch, yaw).
    """
    w, x, y, z = quat
    y_sqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y_sqr)
    eulerx = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    eulery = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y_sqr + z * z)
    eulerz = np.arctan2(t3, t4)

    result = np.zeros(3)
    result[0] = eulerx
    result[1] = eulery
    result[2] = eulerz

    return result


# @nb.jit(nopython=True)
def get_quaternion_from_euler(euler: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """Convert an euler angle to a quaternion based on specified euler angle order.

    Supported Euler angle orders: {'XYZ', 'YXZ', 'ZXY', 'ZYX', 'YZX', 'XZY'}.

    Args:
        euler (np.ndarray): A 3x1 vector with angles in radians.
        order (str, optional): The specified order of input euler angles. Defaults to "XYZ".

    Raises:
        ValueError: If input order is not valid.

    Reference:
        [1] https://github.com/mrdoob/three.js/blob/master/src/math/Quaternion.js
    """
    # extract input angles
    r, p, y = euler
    # compute constants
    y = y / 2.0
    p = p / 2.0
    r = r / 2.0
    c3 = np.cos(y)
    s3 = np.sin(y)
    c2 = np.cos(p)
    s2 = np.sin(p)
    c1 = np.cos(r)
    s1 = np.sin(r)
    # convert to quaternion based on order
    if order == "XYZ":
        result = np.array(
            [
                c1 * c2 * c3 - s1 * s2 * s3,
                c1 * s2 * s3 + c2 * c3 * s1,
                c1 * c3 * s2 - s1 * c2 * s3,
                c1 * c2 * s3 + s1 * c3 * s2,
            ]
        )
        if result[0] < 0:
            result = -result
        return result
    elif order == "YXZ":
        result = np.array(
            [
                c1 * c2 * c3 + s1 * s2 * s3,
                s1 * c2 * c3 + c1 * s2 * s3,
                c1 * s2 * c3 - s1 * c2 * s3,
                c1 * c2 * s3 - s1 * s2 * c3,
            ]
        )
        return result
    elif order == "ZXY":
        result = np.array(
            [
                c1 * c2 * c3 - s1 * s2 * s3,
                s1 * c2 * c3 - c1 * s2 * s3,
                c1 * s2 * c3 + s1 * c2 * s3,
                c1 * c2 * s3 + s1 * s2 * c3,
            ]
        )
        return result
    elif order == "ZYX":
        result = np.array(
            [
                c1 * c2 * c3 + s1 * s2 * s3,
                s1 * c2 * c3 - c1 * s2 * s3,
                c1 * s2 * c3 + s1 * c2 * s3,
                c1 * c2 * s3 - s1 * s2 * c3,
            ]
        )
        return result
    elif order == "YZX":
        result = np.array(
            [
                c1 * c2 * c3 - s1 * s2 * s3,
                s1 * c2 * c3 + c1 * s2 * s3,
                c1 * s2 * c3 + s1 * c2 * s3,
                c1 * c2 * s3 - s1 * s2 * c3,
            ]
        )
        return result
    elif order == "XZY":
        result = np.array(
            [
                c1 * c2 * c3 + s1 * s2 * s3,
                s1 * c2 * c3 - c1 * s2 * s3,
                c1 * s2 * c3 - s1 * c2 * s3,
                c1 * c2 * s3 + s1 * s2 * c3,
            ]
        )
        return result
    else:
        raise ValueError("Input euler angle order is meaningless.")


# @nb.jit(nopython=True)
def get_rotation_matrix_from_euler(euler: np.ndarray, order: str = "XYZ") -> np.ndarray:
    quat = get_quaternion_from_euler(euler, order)
    return get_rotation_matrix_from_quaternion(quat)


# @nb.jit(nopython=True)
def quat_multiplication(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compute the product of two quaternions.

    Args:
        q (np.ndarray): First quaternion in order (w, x, y, z).
        p (np.ndarray): Second quaternion in order (w, x, y, z).

    Returns:
        np.ndarray: A 4x1 vector representing a quaternion in order (w, x, y, z).
    """
    quat = np.array(
        [
            p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
            p[0] * q[1] + p[1] * q[0] - p[2] * q[3] + p[3] * q[2],
            p[0] * q[2] + p[1] * q[3] + p[2] * q[0] - p[3] * q[1],
            p[0] * q[3] - p[1] * q[2] + p[2] * q[1] + p[3] * q[0],
        ]
    )
    return quat


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


import math

_POLE_LIMIT = 1.0 - 1e-6


def matrix_to_euler_angles(mat: np.ndarray, degrees: bool = False, extrinsic: bool = True) -> np.ndarray:
    """Convert rotation matrix to Euler XYZ extrinsic or intrinsic angles.

    Args:
        mat (np.ndarray): A 3x3 rotation matrix.
        degrees (bool, optional): Whether returned angles should be in degrees.
        extrinsic (bool, optional): True if the rotation matrix follows the extrinsic matrix
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic matrix conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        np.ndarray: Euler XYZ angles (intrinsic form) if extrinsic is False and Euler XYZ angles (extrinsic form) if extrinsic is True.
    """
    if extrinsic:
        if mat[2, 0] > _POLE_LIMIT:
            roll = np.arctan2(mat[0, 1], mat[0, 2])
            pitch = -np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        if mat[2, 0] < -_POLE_LIMIT:
            roll = np.arctan2(mat[0, 1], mat[0, 2])
            pitch = np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        roll = np.arctan2(mat[2, 1], mat[2, 2])
        pitch = -np.arcsin(mat[2, 0])
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        if degrees:
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
        return np.array([roll, pitch, yaw])
    else:
        if mat[0, 2] > _POLE_LIMIT:
            roll = np.arctan2(mat[1, 0], mat[1, 1])
            pitch = np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        if mat[0, 2] < -_POLE_LIMIT:
            roll = np.arctan2(mat[1, 0], mat[1, 1])
            pitch = -np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])
        roll = -math.atan2(mat[1, 2], mat[2, 2])
        pitch = math.asin(mat[0, 2])
        yaw = -math.atan2(mat[0, 1], mat[0, 0])

        if degrees:
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
        return np.array([roll, pitch, yaw])


def rotation_matrix_to_quaternion(R):
    assert R.shape == (3, 3)

    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])
