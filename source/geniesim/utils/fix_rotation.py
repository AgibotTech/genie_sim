# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.transform import Rotation as R, Slerp


def interpolate_rotation_matrices(rot_matrix1, rot_matrix2, num_interpolations):
    # Convert the rotation matrices to rotation objects
    rot1 = R.from_matrix(rot_matrix1)
    rot2 = R.from_matrix(rot_matrix2)

    # Convert the rotation objects to quaternions
    quat1 = rot1.as_quat()
    quat2 = rot2.as_quat()

    # Define the times of the known rotations
    times = [0, 1]

    # Create the Slerp object
    slerp = Slerp(times, R.from_quat([quat1, quat2]))

    # Define the times of the interpolations
    interp_times = np.linspace(0, 1, num_interpolations)

    # Perform the interpolation
    interp_rots = slerp(interp_times)

    # Convert the interpolated rotations to matrices
    interp_matrices = interp_rots.as_matrix()

    return interp_matrices


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


def fix_gripper_rotation(source_affine, target_affine, rot_axis="z"):
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

    # source_rotation, target_rotation (3x3)
    source_rotation = source_affine[:3, :3]
    target_rotation = target_affine[:3, :3]
    # target_rotation rotate 180deg around z-axis, target_rotation_2
    target_rotation_2 = np.dot(target_rotation, R_180)

    def rotation_matrix_distance(R1, R2):
        # Use singular value decomposition to calculate the distance between two rotation matrices
        U, _, Vh = np.linalg.svd(np.dot(R1.T, R2))
        # Make sure the determinant of the rotation matrix is 1, i.e. a rotation rather than a reflection
        det_check = np.linalg.det(U) * np.linalg.det(Vh)
        if det_check < 0:
            Vh = -Vh
        return np.arccos(np.trace(Vh) / 2)

    # distance between source_rotation & target_rotation
    distance_target_rotation = rotation_matrix_distance(source_rotation, target_rotation)
    # distance between source_rotation & target_rotation_2
    distance_target_rotation_2 = rotation_matrix_distance(source_rotation, target_rotation_2)
    # which one is nearer to source_rotation
    if distance_target_rotation < distance_target_rotation_2:
        return target_affine
    else:
        # Recombining the rotation matrix target_rotation_2 and the original translation part
        target_affine_2 = np.eye(4)
        target_affine_2[:3, :3] = target_rotation_2
        target_affine_2[:3, 3] = target_affine[:3, 3]  # use orignal trans
        return target_affine_2


def rotate_180_along_axis(pose, rot_axis="z"):
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


def translate_along_axis(pose, shift, axis="z", use_local=True):
    """
    Rotate target_affine according to the specified angle and rotation axis.
    parameter:
    -target_affine: 4x4 affine transformation matrix
    -angle_degrees: rotation angle (in degrees)
    -rot_axis: rotation axis, 'x', 'y' or 'z'
    """
    pose = pose.copy()
    translation = np.zeros(3)
    if axis == "z":
        translation[2] = shift
    elif axis == "y":
        translation[1] = shift
    elif axis == "x":
        translation[0] = shift
    if len(pose.shape) == 3:
        for i in range(pose.shape[0]):
            if use_local:
                pose[i][:3, 3] += pose[i][:3, :3] @ translation
            else:
                pose[i][:3, 3] += translation
    else:
        if use_local:
            pose[:3, 3] += pose[:3, :3] @ translation
        else:
            pose[:3, 3] += translation

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


def quat_wxyz_to_rotation_matrix(quat):
    qw, qx, qy, qz = quat
    return np.array(
        [
            [
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx**2 - 2 * qy**2,
            ],
        ]
    )


def estimate_affine_3d_fixed_scale(src_points, dst_points):
    ransac = RANSACRegressor()
    ransac.fit(src_points, dst_points)
    inlier_mask = ransac.inlier_mask_

    src = src_points[inlier_mask]
    dst = dst_points[inlier_mask]

    # Normalize the input points to ensure uniform scaling
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Estimate rotation using singular value decomposition (SVD)
    U, _, Vt = np.linalg.svd(np.dot(dst_centered.T, src_centered))
    R_est = np.dot(U, Vt)

    # Ensure a right-handed coordinate system
    if np.linalg.det(R_est) < 0:
        Vt[2, :] *= -1
        R_est = np.dot(U, Vt)

    # Calculate the uniform scale
    scale = np.sum(dst_centered * (R_est @ src_centered.T).T) / np.sum(src_centered**2)

    # Construct the affine transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = scale * R_est
    transform[:3, 3] = dst_mean - scale * R_est @ src_mean

    return transform
