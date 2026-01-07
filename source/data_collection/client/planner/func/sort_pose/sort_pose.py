# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np

from client.planner.func.sort_pose.grasp_point_select_config import (
    LEFT_HAND_Z_AZIMUTH_TABLE_PICK_SCORE_CONFIG,
    LEFT_HAND_Z_ELEVATION_TABLE_PICK_FROM_UP_SIDE_SCORE_CONFIG,
    LEFT_HAND_Z_ELEVATION_TABLE_PICK_SCORE_CONFIG,
    RIGHT_HAND_Z_AZIMUTH_TABLE_PICK_SCORE_CONFIG,
    RIGHT_HAND_Z_ELEVATION_TABLE_PICK_FROM_UP_SIDE_SCORE_CONFIG,
    RIGHT_HAND_Z_ELEVATION_TABLE_PICK_SCORE_CONFIG,
)
from client.planner.func.sort_pose.scorer import PiecewiseScorer


def quaternion_to_z_axis(quaternions):
    """
    Extract z-axis direction vector from quaternions (supports batch computation).

    Args:
        quaternions: np.ndarray, shape (N, 4) or (4,), each row is (w, x, y, z)

    Returns:
        z_axis: np.ndarray, shape (N, 3) or (3,), each row is (zx, zy, zz)

    For quaternion q = (w, x, y, z), the third column of the corresponding rotation matrix
    is the z-axis direction after rotation.
    """
    quaternions = np.asarray(quaternions)
    # Handle single quaternion case
    if quaternions.ndim == 1:
        quaternions = quaternions[np.newaxis, :]
        squeeze_output = True
    else:
        squeeze_output = False

    # Ensure quaternion normalization
    norm = np.sqrt(np.sum(quaternions**2, axis=1, keepdims=True))
    norm = np.where(norm > 0, norm, 1.0)  # Avoid division by zero
    quaternions = quaternions / norm

    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    # Calculate z-axis direction vector after rotation
    # Third column of rotation matrix: [2(xz + wy), 2(yz - wx), 1 - 2(x² + y²)]
    zx = 2 * (x * z + w * y)
    zy = 2 * (y * z - w * x)
    zz = 1 - 2 * (x * x + y * y)

    # Normalize (theoretically already a unit vector, but for numerical stability)
    z_axis = np.stack([zx, zy, zz], axis=1)
    length = np.sqrt(np.sum(z_axis**2, axis=1, keepdims=True))
    length = np.where(length > 0, length, 1.0)  # Avoid division by zero
    z_axis = z_axis / length

    # Handle nearly vertical case (default z-axis upward)
    mask = length.squeeze() < 1e-10
    if np.any(mask):
        z_axis[mask] = [0, 0, 1]

    if squeeze_output:
        return z_axis[0]
    return z_axis


def quaternion_to_y_axis(quaternions):
    """
    Extract y-axis direction vector from quaternions (supports batch computation).

    Args:
        quaternions: np.ndarray, shape (N, 4) or (4,), each row is (w, x, y, z)

    Returns:
        y_axis: np.ndarray, shape (N, 3) or (3,), each row is (yx, yy, yz)

    For quaternion q = (w, x, y, z), the second column of the corresponding rotation matrix
    is the y-axis direction after rotation.
    """
    quaternions = np.asarray(quaternions)
    # Handle single quaternion case
    if quaternions.ndim == 1:
        quaternions = quaternions[np.newaxis, :]
        squeeze_output = True
    else:
        squeeze_output = False

    # Ensure quaternion normalization
    norm = np.sqrt(np.sum(quaternions**2, axis=1, keepdims=True))
    norm = np.where(norm > 0, norm, 1.0)  # Avoid division by zero
    quaternions = quaternions / norm

    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    # Calculate y-axis direction vector after rotation
    # Second column of rotation matrix: [2(xy - wz), 1 - 2(x² + z²), 2(yz + wx)]
    yx = 2 * (x * y - w * z)
    yy = 1 - 2 * (x * x + z * z)
    yz = 2 * (y * z + w * x)

    # Normalize (theoretically already a unit vector, but for numerical stability)
    y_axis = np.stack([yx, yy, yz], axis=1)
    length = np.sqrt(np.sum(y_axis**2, axis=1, keepdims=True))
    length = np.where(length > 0, length, 1.0)  # Avoid division by zero
    y_axis = y_axis / length

    # Handle nearly vertical case (default y-axis forward)
    mask = length.squeeze() < 1e-10
    if np.any(mask):
        y_axis[mask] = [0, 1, 0]

    if squeeze_output:
        return y_axis[0]
    return y_axis


def vector_to_angles(vectors):
    """
    Convert direction vectors to azimuth and elevation angles (supports batch computation).

    Args:
        vectors: np.ndarray, shape (N, 3) or (3,), each row is (vx, vy, vz)

    Returns:
        angles: np.ndarray, shape (N, 2) or (2,), each row is (azimuth, elevation)
            azimuth: Azimuth angle (horizontal angle, 0° points to positive x-axis, 90° points to positive y-axis)
            elevation: Elevation angle (vertical angle, 90° is vertically upward, -90° is vertically downward)
    """
    vectors = np.asarray(vectors)
    # Handle single vector case
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis, :]
        squeeze_output = True
    else:
        squeeze_output = False

    vx = vectors[:, 0]
    vy = vectors[:, 1]
    vz = vectors[:, 2]

    # Calculate elevation angle (angle with horizontal plane)
    # Range: -90° (downward) to 90° (upward)
    elevation = np.degrees(np.arcsin(vz))

    # Calculate azimuth angle (angle in horizontal plane)
    # Range: 0° to 360°
    horizontal_length = np.sqrt(vx**2 + vy**2)

    # Calculate angle and convert to 0-360 degree range
    azimuth = np.degrees(np.arctan2(vy, vx))
    azimuth = np.where(azimuth < 0, azimuth + 360, azimuth)

    # Set azimuth to 0 when vector is nearly vertical
    azimuth = np.where(horizontal_length < 1e-10, 0.0, azimuth)

    angles = np.stack([azimuth, elevation], axis=1)

    if squeeze_output:
        return angles[0]
    return angles


def get_hand_z_score(hand_quat, is_right, is_from_up_side=False):
    hand_z_axis = quaternion_to_z_axis(hand_quat)
    hand_azimuth_elevation = vector_to_angles(hand_z_axis)
    hand_azimuth = hand_azimuth_elevation[:, 0]
    hand_elevation = hand_azimuth_elevation[:, 1]
    if is_right:
        scorer_azimuth = PiecewiseScorer(RIGHT_HAND_Z_AZIMUTH_TABLE_PICK_SCORE_CONFIG)
        if is_from_up_side:
            scorer_elevation = PiecewiseScorer(RIGHT_HAND_Z_ELEVATION_TABLE_PICK_FROM_UP_SIDE_SCORE_CONFIG)
        else:
            scorer_elevation = PiecewiseScorer(RIGHT_HAND_Z_ELEVATION_TABLE_PICK_SCORE_CONFIG)
    else:
        scorer_azimuth = PiecewiseScorer(LEFT_HAND_Z_AZIMUTH_TABLE_PICK_SCORE_CONFIG)
        if is_from_up_side:
            scorer_elevation = PiecewiseScorer(LEFT_HAND_Z_ELEVATION_TABLE_PICK_FROM_UP_SIDE_SCORE_CONFIG)
        else:
            scorer_elevation = PiecewiseScorer(LEFT_HAND_Z_ELEVATION_TABLE_PICK_SCORE_CONFIG)
    azimuth_score = scorer_azimuth.batch_score(hand_azimuth)
    elevation_score = scorer_elevation.batch_score(hand_elevation)
    # scorer_azimuth.plot_segments(x_min=0, x_max=360)
    return azimuth_score, elevation_score


def sorted_by_position_humanlike(
    joint_positions,
    joint_names,
    link_poses,
    is_right,
    elbow_name,
    hand_name,
    is_from_up_side=False,
):
    """
    Args:
        link_poses: list of N * dict (str: list[[x,y,z],[w,x,y,z]]), name : [pos(x,y,z), quat(w,x,y,z)]
        is_right: bool, whether the arm is right
        elbow_name: str, the name of the elbow link
        hand_name: str, the name of the hand link
    Returns:
        idx_sorted: np.ndarray, shape (N,), the indices of the link poses sorted by the position humanlike

    Human-likeness criteria:
    - Elbow further outward (left arm: larger y, right arm: smaller y) is more human-like
    - Elbow further back (smaller x) is more human-like
    - Hand camera pointing outward is more human-like
    Each criterion has a weight, and the final cost is the weighted sum of all criteria.
    """
    # Batch compute scores for all poses for better efficiency

    joint_position_name = []
    for index in range(len(joint_names)):
        joint_names_element = joint_names[index]
        joint_position_element = joint_positions[index]
        joint_name_position_dict = {
            name: position for name, position in zip(joint_names_element, joint_position_element)
        }
        joint_position_name.append(joint_name_position_dict)
    len(link_poses)
    # Weights can be adjusted as needed
    weight_elbow_out = 1.0
    weight_elbow_back = 0.5

    elbow_pos = np.array([pose[elbow_name][0] for pose in link_poses])  # N x 3
    hand_quat = np.array([pose[hand_name][1] for pose in link_poses])  # N x 4

    elbow_y = elbow_pos[:, 1]  # (N,)
    elbow_x = elbow_pos[:, 0]

    hand_z_azimuth_score, hand_z_elevation_score = get_hand_z_score(
        hand_quat, is_right, is_from_up_side=is_from_up_side
    )

    if is_right:
        costs = (
            0
            + weight_elbow_out * elbow_y
            + weight_elbow_back * elbow_x
            + 0.1 * (hand_z_azimuth_score + hand_z_elevation_score)
        )
    else:
        costs = (
            0
            - weight_elbow_out * elbow_y
            + weight_elbow_back * elbow_x
            + 0.1 * (hand_z_azimuth_score + hand_z_elevation_score)
        )
    idx_sorted = np.argsort(costs)
    return idx_sorted


def main(ik_info_file):
    import json
    import pickle

    with open(ik_info_file, "rb") as f:
        ik_infos = pickle.load(f)
    ik_success = ik_infos["ik_success"]
    ik_info = ik_infos["ik_info"]
    is_right = ik_infos["is_right"]
    elbow_name = ik_infos["elbow_name"]
    hand_name = ik_infos["hand_name"]
    ik_joint_positions = ik_info["joint_positions"][ik_success]
    ik_joint_names = ik_info["joint_names"][ik_success]
    ik_info["jacobian_score"][ik_success]
    ik_link_poses = ik_info["link_poses"][ik_success]
    idx_sorted, costs_sorted = sorted_by_position_humanlike(
        ik_joint_positions,
        ik_joint_names,
        ik_link_poses,
        is_right=is_right,
        elbow_name=elbow_name,
        hand_name=hand_name,
    )
    joint_name_position_sorted = []
    for idx in idx_sorted:
        joint_positions = ik_joint_positions[idx]
        joint_names = ik_joint_names[idx]
        joint_name_position_sorted.append(
            {joint_name: joint_positions[list(joint_names).index(joint_name)] for joint_name in joint_names}
        )
    import os

    current_directory = os.path.dirname(os.path.abspath(__file__))
    with open(current_directory + "/joint_name_position_sorted_new.json", "w") as f:
        json.dump(joint_name_position_sorted, f)
    origin_joint_name_position = []
    for idx in range(len(ik_joint_names)):
        joint_positions = ik_joint_positions[idx]
        joint_names = ik_joint_names[idx]
        origin_joint_name_position.append(
            {joint_name: joint_positions[list(joint_names).index(joint_name)] for joint_name in joint_names}
        )
    with open(current_directory + "origin_joint_name_position.json", "w") as f:
        json.dump(origin_joint_name_position, f)


if __name__ == "__main__":
    ik_info_file = "ik_info_left.pkl"
    main(ik_info_file)
