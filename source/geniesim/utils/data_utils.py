# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance


def interpolate_pose_trajectory_rel(pose1, pose2, steps):
    """Return a list of incremental poses between two poses.

    Args:
        pose1 (np.ndarray): current pose, position + quat
        pose2 (np.ndarray): desired_pose, position + quat

    Returns:
        increments (np.ndarray): (N, 7) with each element position + quat increment.
    """

    # Calculate the absolute path
    rot1 = R.from_matrix(pose1[:3, :3])
    rot2 = R.from_matrix(pose2[:3, :3])
    quat1 = rot1.as_quat()
    quat2 = rot2.as_quat()
    translation_path = np.linspace(pose1[:3], pose2[:3], steps)
    key_times = [0, 1]
    key_rots = R.from_quat(np.stack([quat1, quat2]))
    slerp = Slerp(key_times, key_rots)
    times = np.linspace(0, 1, steps)
    interp_rots = slerp(times)
    eulers = interp_rots.as_euler("xyz", degrees=False)

    # Calculate increments
    translation_increments = np.diff(translation_path, axis=0) * 1000.0
    euler_increments = np.diff(eulers, axis=0)

    # Combine increments
    increments = np.concatenate([translation_increments, euler_increments], axis=1)

    # Add initial zero increment for the first step
    initial_increment = np.zeros((1, 6))
    increments = np.vstack([initial_increment, increments])

    return increments.tolist()


def interpolate_pose_trajectory(pose1, pose2, num_interpolations):
    # interpolate rotation
    rot1 = R.from_matrix(pose1[:3, :3])
    rot2 = R.from_matrix(pose2[:3, :3])
    quat1 = rot1.as_quat()
    quat2 = rot2.as_quat()
    times = [0, 1]
    slerp = Slerp(times, R.from_quat([quat1, quat2]))
    interp_times = np.linspace(0, 1, num_interpolations)
    interp_rots = slerp(interp_times)
    interp_matrices = interp_rots.as_matrix()  # (num_interpolations, 3, 3)
    # interpolate translation
    interp_xyz = np.linspace(pose1[:3, 3], pose2[:3, 3], num_interpolations)  # (num_interpolations, 3)

    interp_poses = np.tile(np.eye(4), (num_interpolations, 1, 1))
    interp_poses[:, :3, :3] = interp_matrices
    interp_poses[:, :3, 3] = interp_xyz
    return interp_poses


class CameraInfo:
    """Camera intrisics for point cloud creation."""

    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def move_camera_to_object_center(object_pose_camera, camera_pose_world):
    # get tran & rot
    R_cw = camera_pose_world[:3, :3]
    t_cw = camera_pose_world[:3, 3]

    object_pos_camera = object_pose_camera[:3, 3]

    object_pos_world = R_cw @ object_pos_camera + t_cw

    z_distance = object_pos_camera[2]

    new_camera_pos_world = object_pos_world - R_cw @ np.array([0, 0, z_distance])

    new_camera_pose_world = np.eye(4)
    new_camera_pose_world[:3, :3] = R_cw
    new_camera_pose_world[:3, 3] = -new_camera_pos_world

    return new_camera_pose_world


def create_point_cloud_from_depth_image(depth, cam_info, organized=True):
    """Generate point cloud using depth image only.

    Input:
        depth: [numpy.ndarray, (H,W), numpy.float32]
            depth image
        camera: [CameraInfo]
            camera intrinsics
        organized: bool
            whether to keep the cloud in image shape (H,W,3)

    Output:
        cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
            generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    if not isinstance(cam_info, CameraInfo):
        camera = CameraInfo(
            width=cam_info["W"],
            height=cam_info["H"],
            fx=cam_info["K"][0][0],
            fy=cam_info["K"][1][1],
            cx=cam_info["K"][0][2],
            cy=cam_info["K"][1][2],
            scale=cam_info["scale"],
        )
    else:
        camera = cam_info
    assert depth.shape[0] == camera.height and depth.shape[1] == camera.width
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def transform_point_cloud_numpy(cloud, transform, format="4x4"):
    """Transform points to new coordinates with transformation matrix.

    Input:
        cloud: [np.ndarray, (N,3), np.float32]
            points in original coordinates
        transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
            transformation matrix, could be rotation only or rotation+translation
        format: [string, '3x3'/'3x4'/'4x4']
            the shape of transformation matrix
            '3x3' --> rotation matrix
            '3x4'/'4x4' --> rotation matrix + translation matrix

    Output:
        cloud_transformed: [np.ndarray, (N,3), np.float32]
            points in new coordinates
    """
    if not (format == "3x3" or format == "4x4" or format == "3x4"):
        raise ValueError("Unknown transformation format, only support '3x3' or '4x4' or '3x4'.")
    if format == "3x3":
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == "4x4" or format == "3x4":
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed


def transform_point_cloud_tensor(cloud, transform, format="4x4"):
    """Transform points to new coordinates with transformation matrix.

    Input:
        cloud: [torch.FloatTensor, (N,3)]
            points in original coordinates
        transform: [torch.FloatTensor, (3,3)/(3,4)/(4,4)]
            transformation matrix, could be rotation only or rotation+translation
        format: [string, '3x3'/'3x4'/'4x4']
            the shape of transformation matrix
            '3x3' --> rotation matrix
            '3x4'/'4x4' --> rotation matrix + translation matrix

    Output:
        cloud_transformed: [torch.FloatTensor, (N,3)]
            points in new coordinates
    """
    if not (format == "3x3" or format == "4x4" or format == "3x4"):
        raise ValueError("Unknown transformation format, only support '3x3' or '4x4' or '3x4'.")
    if format == "3x3":
        cloud_transformed = torch.matmul(transform, cloud.T).T
    elif format == "4x4" or format == "3x4":
        ones = cloud.new_ones(cloud.size(0), device=cloud.device).unsqueeze(-1)
        cloud_ = torch.cat([cloud, ones], dim=1)
        cloud_transformed = torch.matmul(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed


def transform_point_cloud(cloud, transform, format="4x4"):
    """Transform points to new coordinates with transformation matrix.

    Input:
        cloud: [np.ndarray or torch.tensor, (N,3), np.float32]
            points in original coordinates
        transform: [np.ndarray or torch.tensor, (3,3)/(3,4)/(4,4), np.float32]
            transformation matrix, could be rotation only or rotation+translation
        format: [string, '3x3'/'3x4'/'4x4']
            the shape of transformation matrix
            '3x3' --> rotation matrix
            '3x4'/'4x4' --> rotation matrix + translation matrix

    Output:
        cloud_transformed: [np.ndarray or torch.tensor, (N,3), np.float32]
            points in new coordinates
    """
    if type(cloud) != type(transform):
        raise ValueError("cloud and transform must be the same type.")
    if type(cloud) == np.ndarray:
        return transform_point_cloud_numpy(cloud, transform, format)
    elif type(cloud) == torch.Tensor:
        return transform_point_cloud_tensor(cloud, transform, format)


def compute_point_dists(A, B):
    """Compute pair-wise point distances in two matrices.

    Input:
        A: [np.ndarray, (N,3), np.float32]
            point cloud A
        B: [np.ndarray, (M,3), np.float32]
            point cloud B

    Output:
        dists: [np.ndarray, (N,M), np.float32]
            distance matrix
    """
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A - B, axis=-1)
    return dists


def remove_invisible_grasp_points(cloud, grasp_points, pose, th=0.01):
    """Remove invisible part of object model according to scene point cloud.

    Input:
        cloud: [np.ndarray, (N,3), np.float32]
            scene point cloud
        grasp_points: [np.ndarray, (M,3), np.float32]
            grasp point label in object coordinates
        pose: [np.ndarray, (4,4), np.float32]
            transformation matrix from object coordinates to world coordinates
        th: [float]
            if the minimum distance between a grasp point and the scene points is greater than outlier, the point will be removed

    Output:
        visible_mask: [np.ndarray, (M,), np.bool]
            mask to show the visible part of grasp points
    """
    grasp_points_trans = transform_point_cloud(grasp_points, pose)
    dists = compute_point_dists(grasp_points_trans, cloud)
    min_dists = dists.min(axis=1)
    visible_mask = min_dists < th
    return visible_mask


def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """Keep points in workspace as input.

    Input:
        cloud: [np.ndarray, (H,W,3), np.float32]
            scene point cloud
        seg: [np.ndarray, (H,W,), np.uint8]
            segmentation label of scene points
        trans: [np.ndarray, (4,4), np.float32]
            transformation matrix for scene points, default: None.
        organized: [bool]
            whether to keep the cloud in image shape (H,W,3)
        outlier: [float]
            if the distance between a point and workspace is greater than outlier, the point will be removed

    Output:
        workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
            mask to indicate whether scene points are in workspace
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h * w, 3])
        seg = seg.reshape(h * w)
    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
    foreground = cloud[seg > 0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = (cloud[:, 0] > xmin - outlier) & (cloud[:, 0] < xmax + outlier)
    mask_y = (cloud[:, 1] > ymin - outlier) & (cloud[:, 1] < ymax + outlier)
    mask_z = (cloud[:, 2] > zmin - outlier) & (cloud[:, 2] < zmax + outlier)
    workspace_mask = mask_x & mask_y & mask_z
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask


def get_rot_dim(rot_mode):
    assert rot_mode in [
        "quat_wxyz",
        "quat_xyzw",
        "euler_xyz",
        "rot_6d",
    ], f"the rotation mode {rot_mode} is not supported!"

    if rot_mode == "quat_wxyz" or rot_mode == "quat_xyzw":
        rot_dim = 4
    elif rot_mode == "euler_xyz":
        rot_dim = 3
    elif rot_mode == "rot_6d":
        rot_dim = 6
    else:
        raise NotImplementedError
    return rot_dim


def get_grasp_pose_dim(rot_mode, grasp_type="point_level"):
    if grasp_type == "point_level":
        return 3 + get_rot_dim(rot_mode)  # score, width, depth, rotation
    elif grasp_type == "scene_level":
        return 6 + get_rot_dim(rot_mode)  # translation, score, width, depth, rotation


def normalize_rotation(rotation, rotation_mode):
    rot = SO3Group()
    rot.from_rotation(rotation, rotation_mode)
    rot.get_rotation(rotation_mode)
    return rotation


def get_rot_dim(rotation_mode):
    if rotation_mode == "quat_wxyz" or rotation_mode == "quat_xyzw":
        rot_dim = 4
    elif rotation_mode == "euler_xyz":
        rot_dim = 3
    elif rotation_mode == "rot_matrix":
        rot_dim = 9
    elif rotation_mode == "rot_6d":
        rot_dim = 6
    else:
        logger.exception(f"the rotation mode {rotation_mode} is not supported!")
        raise NotImplementedError
    return rot_dim


def map_ambiguous_rot(rot):
    # Conversion to PyTorch: Define the rotation along the x-axis by 180 degrees.
    rotate_along_x_axis_180 = torch.tensor([[[1, 0, 0], [0, -1, 0], [0, 0, -1]]], dtype=rot.dtype, device=rot.device)

    # Conversion to PyTorch: Perform matrix multiplication.
    rot_new = torch.matmul(rot, rotate_along_x_axis_180)

    # Conversion to PyTorch: Compute the rotation error before and after applying the new rotation.
    # The trace function in PyTorch does not support 'axis1' and 'axis2', so we need to handle it manually.
    rot_error = torch.acos(torch.clamp((torch.einsum("bii->b", rot) - 1) / 2, -1.0, 1.0)) * 180 / torch.pi
    rot_new_error = torch.acos(torch.clamp((torch.einsum("bii->b", rot_new) - 1) / 2, -1.0, 1.0)) * 180 / torch.pi

    # Conversion to PyTorch: Choose between the original and new rotation based on the error.
    chosen_rot = torch.where(rot_error[:, None, None] <= rot_new_error[:, None, None], rot, rot_new)

    return chosen_rot


def mask2bbox(mask):
    nonzero_indices = np.nonzero(mask)
    min_y, min_x = np.min(nonzero_indices, axis=1)
    max_y, max_x = np.max(nonzero_indices, axis=1)
    bbox = [min_x, min_y, max_x, max_y]
    return bbox


def get_3d_center_of_rgbd_mask(masks, depth, cam_info, c2w):
    H, W = depth.shape[:2]
    pts = create_point_cloud_from_depth_image(depth, cam_info)
    pts_world = transform_point_cloud_numpy(pts.reshape(-1, 3), c2w).reshape(H, W, 3)
    depth_mask = (depth > 0.1) & (depth < 2)
    centers = []
    for id in masks:
        mask = masks[id] & depth_mask
        if mask.sum() == 0:
            center_xyz = np.array([np.inf, np.inf, np.inf])
        else:
            xyz = pts_world[mask]
            center_xyz = (np.percentile(xyz, 98, axis=0) + np.percentile(xyz, 2, axis=0)) / 2
        centers.append(center_xyz)
    centers = np.stack(centers)
    return centers


def match_obj_with_new_view(target_xyz, masks, depth, cam_info, c2w, **kwargs):
    obj_centers = get_3d_center_of_rgbd_mask(masks, depth, cam_info, c2w)
    diff = np.linalg.norm(obj_centers - target_xyz, axis=1)
    min_index = np.argmin(diff)
    return min_index


def pose_difference_batch(pose1, pose2):
    # Ensure pose2 is a 4x4 matrix
    assert pose2.shape == (4, 4), "pose2 must be a 4x4 matrix"

    # Extract positions
    positions1 = pose1[:, :3, 3]
    position2 = pose2[:3, 3]

    # Calculate Euclidean distances for positions
    position_distances = np.linalg.norm(positions1 - position2, axis=1)

    # Extract rotation matrices
    rotations1 = pose1[:, :3, :3]
    rotation2 = pose2[:3, :3]

    # Convert to rotation objects
    r1 = R.from_matrix(rotations1)
    r2 = R.from_matrix(rotation2)

    # Calculate rotation differences
    relative_rotations = r1.inv() * r2
    angle_differences = relative_rotations.magnitude()

    return position_distances, np.degrees(angle_differences)


def pose_difference(pose1, pose2):
    # position
    position1 = pose1[:3, 3]
    position2 = pose2[:3, 3]

    # distance
    position_distance = np.linalg.norm(position1 - position2)

    # roation matrix
    rotation1 = pose1[:3, :3]
    rotation2 = pose2[:3, :3]

    # roation matrix angle diff
    r1 = R.from_matrix(rotation1)
    r2 = R.from_matrix(rotation2)

    # angle diff
    relative_rotation = r1.inv() * r2
    angle_difference = relative_rotation.magnitude()

    return position_distance, np.degrees(angle_difference)


def vector_difference(pose1, pose2, vector=np.array([0, 0, 1])):
    rotation1 = pose1[:3, :3]
    rotation2 = pose2[:3, :3]

    transformed_vector1 = rotation1 @ vector
    transformed_vector2 = rotation2 @ vector

    dot_product = np.dot(transformed_vector1, transformed_vector2)
    norms_product = np.linalg.norm(transformed_vector1) * np.linalg.norm(transformed_vector2)
    cos_angle = dot_product / norms_product

    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_difference = np.arccos(cos_angle)

    return np.degrees(angle_difference)


def vector_difference_batch(pose1, pose2, vector=np.array([0, 0, 1])):
    assert pose2.shape == (4, 4), "pose2 must be a 4x4 matrix"

    rotations1 = pose1[:, :3, :3]
    rotation2 = pose2[:3, :3]

    transformed_vectors1 = np.einsum("bij,j->bi", rotations1, vector)
    transformed_vector2 = rotation2 @ vector

    dot_products = np.einsum("bi,i->b", transformed_vectors1, transformed_vector2)
    norms_product1 = np.linalg.norm(transformed_vectors1, axis=1)
    norm_product2 = np.linalg.norm(transformed_vector2)
    norms_products = norms_product1 * norm_product2

    cos_angles = dot_products / norms_products

    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    angle_differences = np.arccos(cos_angles)

    return np.degrees(angle_differences)


import os
import numpy as np
from numba import njit
import datetime
import scipy.interpolate as interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline
import geniesim.utils.transform_utils as T


# ===============================================
# = optimization utils
# ===============================================
def normalize_vars(vars, og_bounds):
    """
    Given 1D variables and bounds, normalize the variables to [-1, 1] range.
    """
    normalized_vars = np.empty_like(vars)
    for i, (b_min, b_max) in enumerate(og_bounds):
        normalized_vars[i] = (vars[i] - b_min) / (b_max - b_min) * 2 - 1
    return normalized_vars


def unnormalize_vars(normalized_vars, og_bounds):
    """
    Given 1D variables in [-1, 1] and original bounds, denormalize the variables to the original range.
    """
    vars = np.empty_like(normalized_vars)
    for i, (b_min, b_max) in enumerate(og_bounds):
        vars[i] = (normalized_vars[i] + 1) / 2 * (b_max - b_min) + b_min
    return vars


def calculate_collision_cost(poses, sdf_func, collision_points, threshold):
    assert poses.shape[1:] == (4, 4)
    transformed_pcs = batch_transform_points(collision_points, poses)
    transformed_pcs_flatten = transformed_pcs.reshape(-1, 3)  # [num_poses * num_points, 3]
    signed_distance = sdf_func(transformed_pcs_flatten) + threshold  # [num_poses * num_points]
    signed_distance = signed_distance.reshape(-1, collision_points.shape[0])  # [num_poses, num_points]
    non_zero_mask = signed_distance > 0
    collision_cost = np.sum(signed_distance[non_zero_mask])
    return collision_cost


@njit(cache=True, fastmath=True)
def consistency(poses_a, poses_b, rot_weight=0.5):
    assert poses_a.shape[1:] == (4, 4) and poses_b.shape[1:] == (
        4,
        4,
    ), "poses must be of shape (N, 4, 4)"
    min_distances = np.zeros(len(poses_a), dtype=np.float64)
    for i in range(len(poses_a)):
        min_distance = 9999999
        a = poses_a[i]
        for j in range(len(poses_b)):
            b = poses_b[j]
            pos_distance = np.linalg.norm(a[:3, 3] - b[:3, 3])
            rot_distance = angle_between_rotmat(a[:3, :3], b[:3, :3])
            distance = pos_distance + rot_distance * rot_weight
            min_distance = min(min_distance, distance)
        min_distances[i] = min_distance
    return np.mean(min_distances)


def transform_keypoints(transform, keypoints, movable_mask):
    assert transform.shape == (4, 4)
    transformed_keypoints = keypoints.copy()
    if movable_mask.sum() > 0:
        transformed_keypoints[movable_mask] = np.dot(keypoints[movable_mask], transform[:3, :3].T) + transform[:3, 3]
    return transformed_keypoints


@njit(cache=True, fastmath=True)
def batch_transform_points(points, transforms):
    """
    Apply multiple of transformation to point cloud, return results of individual transformations.
    Args:
        points: point cloud (N, 3).
        transforms: M 4x4 transformations (M, 4, 4).
    Returns:
        np.array: point clouds (M, N, 3).
    """
    assert transforms.shape[1:] == (4, 4), "transforms must be of shape (M, 4, 4)"
    transformed_points = np.zeros((transforms.shape[0], points.shape[0], 3))
    for i in range(transforms.shape[0]):
        pos, R = transforms[i, :3, 3], transforms[i, :3, :3]
        transformed_points[i] = np.dot(points, R.T) + pos
    return transformed_points


@njit(cache=True, fastmath=True)
def get_samples_jitted(
    control_points_homo,
    control_points_quat,
    opt_interpolate_pos_step_size,
    opt_interpolate_rot_step_size,
):
    assert control_points_homo.shape[1:] == (4, 4)
    # calculate number of samples per segment
    num_samples_per_segment = np.empty(len(control_points_homo) - 1, dtype=np.int64)
    for i in range(len(control_points_homo) - 1):
        start_pos = control_points_homo[i, :3, 3]
        start_rotmat = control_points_homo[i, :3, :3]
        end_pos = control_points_homo[i + 1, :3, 3]
        end_rotmat = control_points_homo[i + 1, :3, :3]
        pos_diff = np.linalg.norm(start_pos - end_pos)
        rot_diff = angle_between_rotmat(start_rotmat, end_rotmat)
        pos_num_steps = np.ceil(pos_diff / opt_interpolate_pos_step_size)
        rot_num_steps = np.ceil(rot_diff / opt_interpolate_rot_step_size)
        num_path_poses = int(max(pos_num_steps, rot_num_steps))
        num_path_poses = max(num_path_poses, 2)  # at least 2 poses, start and end
        num_samples_per_segment[i] = num_path_poses
    # fill in samples
    num_samples = num_samples_per_segment.sum()
    samples_7 = np.empty((num_samples, 7))
    sample_idx = 0
    for i in range(len(control_points_quat) - 1):
        start_pos, start_xyzw = control_points_quat[i, :3], control_points_quat[i, 3:]
        end_pos, end_xyzw = (
            control_points_quat[i + 1, :3],
            control_points_quat[i + 1, 3:],
        )
        # using proper quaternion slerp interpolation
        poses_7 = np.empty((num_samples_per_segment[i], 7))
        for j in range(num_samples_per_segment[i]):
            alpha = j / (num_samples_per_segment[i] - 1)
            pos = start_pos * (1 - alpha) + end_pos * alpha
            blended_xyzw = T.quat_slerp_jitted(start_xyzw, end_xyzw, alpha)
            pose_7 = np.empty(7)
            pose_7[:3] = pos
            pose_7[3:] = blended_xyzw
            poses_7[j] = pose_7
        samples_7[sample_idx : sample_idx + num_samples_per_segment[i]] = poses_7
        sample_idx += num_samples_per_segment[i]
    assert num_samples >= 2, f"num_samples: {num_samples}"
    return samples_7, num_samples


@njit(cache=True, fastmath=True)
def path_length(samples_homo):
    assert samples_homo.shape[1:] == (4, 4), "samples_homo must be of shape (N, 4, 4)"
    pos_length = 0
    rot_length = 0
    for i in range(len(samples_homo) - 1):
        pos_length += np.linalg.norm(samples_homo[i, :3, 3] - samples_homo[i + 1, :3, 3])
        rot_length += angle_between_rotmat(samples_homo[i, :3, :3], samples_homo[i + 1, :3, :3])
    return pos_length, rot_length


# ===============================================
# = others
# ===============================================
def get_callable_grasping_cost_fn(env):
    def get_grasping_cost(keypoint_idx):
        keypoint_object = env.get_object_by_keypoint(keypoint_idx)
        return (
            -env.is_grasping(candidate_obj=keypoint_object) + 1
        )  # return 0 if grasping an object, 1 if not grasping any object

    return get_grasping_cost


def get_config(config_path=None):
    if config_path is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_file_dir, "configs/config.yaml")
    assert config_path and os.path.exists(config_path), f"config file does not exist ({config_path})"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_clock_time(milliseconds=False):
    curr_time = datetime.datetime.now()
    if milliseconds:
        return f"{curr_time.hour}:{curr_time.minute}:{curr_time.second}.{curr_time.microsecond // 1000}"
    else:
        return f"{curr_time.hour}:{curr_time.minute}:{curr_time.second}"


def angle_between_quats(q1, q2):
    """Angle between two quaternions"""
    return 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), -1, 1))


def filter_points_by_bounds(points, bounds_min, bounds_max, strict=True):
    """
    Filter points by taking only points within workspace bounds.
    """
    assert points.shape[1] == 3, "points must be (N, 3)"
    bounds_min = bounds_min.copy()
    bounds_max = bounds_max.copy()
    if not strict:
        bounds_min[:2] = bounds_min[:2] - 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_max[:2] = bounds_max[:2] + 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_min[2] = bounds_min[2] - 0.1 * (bounds_max[2] - bounds_min[2])
    within_bounds_mask = (
        (points[:, 0] >= bounds_min[0])
        & (points[:, 0] <= bounds_max[0])
        & (points[:, 1] >= bounds_min[1])
        & (points[:, 1] <= bounds_max[1])
        & (points[:, 2] >= bounds_min[2])
        & (points[:, 2] <= bounds_max[2])
    )
    return within_bounds_mask


def merge_dicts(dicts):
    return {k: v for d in dicts for k, v in d.items()}


def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ["import", "__"]
    for phrase in banned_phrases:
        assert phrase not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([gvars, {"exec": empty_fn, "eval": empty_fn}])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        logger.exception(f"Error executing code:\n{code_str}")
        raise e


def load_functions_from_txt(txt_path, get_grasping_cost_fn):
    if txt_path is None:
        return []
    # load txt file
    with open(txt_path, "r") as f:
        functions_text = f.read()
    # execute functions
    gvars_dict = {
        "np": np,
        "get_grasping_cost_by_keypoint_idx": get_grasping_cost_fn,
    }  # external library APIs
    lvars_dict = dict()
    exec_safe(functions_text, gvars=gvars_dict, lvars=lvars_dict)
    return list(lvars_dict.values())


@njit(cache=True, fastmath=True)
def angle_between_rotmat(P, Q):
    R = np.dot(P, Q.T)
    cos_theta = (np.trace(R) - 1) / 2
    if cos_theta > 1:
        cos_theta = 1
    elif cos_theta < -1:
        cos_theta = -1
    return np.arccos(cos_theta)


def fit_b_spline(control_points):
    # determine appropriate k
    k = min(3, control_points.shape[0] - 1)
    spline = interpolate.splprep(control_points.T, s=0, k=k)
    return spline


def sample_from_spline(spline, num_samples):
    sample_points = np.linspace(0, 1, num_samples)
    if isinstance(spline, RotationSpline):
        samples = spline(sample_points).as_matrix()  # [num_samples, 3, 3]
    else:
        assert isinstance(spline, tuple) and len(spline) == 2, "spline must be a tuple of (tck, u)"
        tck, u = spline
        samples = interpolate.splev(np.linspace(0, 1, num_samples), tck)  # [spline_dim, num_samples]
        samples = np.array(samples).T  # [num_samples, spline_dim]
    return samples


def linear_interpolate_poses(start_pose, end_pose, num_poses):
    """
    Interpolate between start and end pose.
    """
    assert num_poses >= 2, "num_poses must be at least 2"
    if start_pose.shape == (6,) and end_pose.shape == (6,):
        start_pos, start_euler = start_pose[:3], start_pose[3:]
        end_pos, end_euler = end_pose[:3], end_pose[3:]
        start_rotmat = T.euler2mat(start_euler)
        end_rotmat = T.euler2mat(end_euler)
    elif start_pose.shape == (4, 4) and end_pose.shape == (4, 4):
        start_pos = start_pose[:3, 3]
        start_rotmat = start_pose[:3, :3]
        end_pos = end_pose[:3, 3]
        end_rotmat = end_pose[:3, :3]
    elif start_pose.shape == (7,) and end_pose.shape == (7,):
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        start_rotmat = T.quat2mat(start_quat)
        end_pos, end_quat = end_pose[:3], end_pose[3:]
        end_rotmat = T.quat2mat(end_quat)
    else:
        raise ValueError("start_pose and end_pose not recognized")
    slerp = Slerp([0, 1], R.from_matrix([start_rotmat, end_rotmat]))
    poses = []
    for i in range(num_poses):
        alpha = i / (num_poses - 1)
        pos = start_pos * (1 - alpha) + end_pos * alpha
        rotmat = slerp(alpha).as_matrix()
        if start_pose.shape == (6,):
            euler = T.mat2euler(rotmat)
            poses.append(np.concatenate([pos, euler]))
        elif start_pose.shape == (4, 4):
            pose = np.eye(4)
            pose[:3, :3] = rotmat
            pose[:3, 3] = pos
            poses.append(pose)
        elif start_pose.shape == (7,):
            quat = T.mat2quat(rotmat)
            pose = np.concatenate([pos, quat])
            poses.append(pose)
    return np.array(poses)


def spline_interpolate_poses(control_points, num_steps):
    """
    Interpolate between through the control points using spline interpolation.
    1. Fit a b-spline through the positional terms of the control points.
    2. Fit a RotationSpline through the rotational terms of the control points.
    3. Sample the b-spline and RotationSpline at num_steps.

    Args:
        control_points: [N, 6] position + euler or [N, 4, 4] pose or [N, 7] position + quat
        num_steps: number of poses to interpolate
    Returns:
        poses: [num_steps, 6] position + euler or [num_steps, 4, 4] pose or [num_steps, 7] position + quat
    """
    assert num_steps >= 2, "num_steps must be at least 2"
    if isinstance(control_points, list):
        control_points = np.array(control_points)
    if control_points.shape[1] == 6:
        control_points_pos = control_points[:, :3]  # [N, 3]
        control_points_euler = control_points[:, 3:]  # [N, 3]
        control_points_rotmat = []
        for control_point_euler in control_points_euler:
            control_points_rotmat.append(T.euler2mat(control_point_euler))
        control_points_rotmat = np.array(control_points_rotmat)  # [N, 3, 3]
    elif control_points.shape[1] == 4 and control_points.shape[2] == 4:
        control_points_pos = control_points[:, :3, 3]  # [N, 3]
        control_points_rotmat = control_points[:, :3, :3]  # [N, 3, 3]
    elif control_points.shape[1] == 7:
        control_points_pos = control_points[:, :3]
        control_points_rotmat = []
        for control_point_quat in control_points[:, 3:]:
            control_points_rotmat.append(T.quat2mat(control_point_quat))
        control_points_rotmat = np.array(control_points_rotmat)
    else:
        raise ValueError("control_points not recognized")
    # remove the duplicate points (threshold 1e-3)
    diff = np.linalg.norm(np.diff(control_points_pos, axis=0), axis=1)
    mask = diff > 1e-3
    # always keep the first and last points
    mask = np.concatenate([[True], mask[:-1], [True]])
    control_points_pos = control_points_pos[mask]
    control_points_rotmat = control_points_rotmat[mask]
    # fit b-spline through positional terms control points
    pos_spline = fit_b_spline(control_points_pos)
    # fit RotationSpline through rotational terms control points
    times = pos_spline[1]
    rotations = R.from_matrix(control_points_rotmat)
    rot_spline = RotationSpline(times, rotations)
    # sample from the splines
    pos_samples = sample_from_spline(pos_spline, num_steps)  # [num_steps, 3]
    rot_samples = sample_from_spline(rot_spline, num_steps)  # [num_steps, 3, 3]
    if control_points.shape[1] == 6:
        poses = []
        for i in range(num_steps):
            pose = np.concatenate([pos_samples[i], T.mat2euler(rot_samples[i])])
            poses.append(pose)
        poses = np.array(poses)
    elif control_points.shape[1] == 4 and control_points.shape[2] == 4:
        poses = np.empty((num_steps, 4, 4))
        poses[:, :3, :3] = rot_samples
        poses[:, :3, 3] = pos_samples
        poses[:, 3, 3] = 1
    elif control_points.shape[1] == 7:
        poses = np.empty((num_steps, 7))
        for i in range(num_steps):
            quat = T.mat2quat(rot_samples[i])
            pose = np.concatenate([pos_samples[i], quat])
            poses[i] = pose
    return poses


def get_linear_interpolation_steps(start_pose, end_pose, pos_step_size, rot_step_size):
    """
    Given start and end pose, calculate the number of steps to interpolate between them.
    Args:
        start_pose: [6] position + euler or [4, 4] pose or [7] position + quat
        end_pose: [6] position + euler or [4, 4] pose or [7] position + quat
        pos_step_size: position step size
        rot_step_size: rotation step size
    Returns:
        num_path_poses: number of poses to interpolate
    """
    if start_pose.shape == (6,) and end_pose.shape == (6,):
        start_pos, start_euler = start_pose[:3], start_pose[3:]
        end_pos, end_euler = end_pose[:3], end_pose[3:]
        start_rotmat = T.euler2mat(start_euler)
        end_rotmat = T.euler2mat(end_euler)
    elif start_pose.shape == (4, 4) and end_pose.shape == (4, 4):
        start_pos = start_pose[:3, 3]
        start_rotmat = start_pose[:3, :3]
        end_pos = end_pose[:3, 3]
        end_rotmat = end_pose[:3, :3]
    elif start_pose.shape == (7,) and end_pose.shape == (7,):
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        start_rotmat = T.quat2mat(start_quat)
        end_pos, end_quat = end_pose[:3], end_pose[3:]
        end_rotmat = T.quat2mat(end_quat)
    else:
        raise ValueError("start_pose and end_pose not recognized")
    pos_diff = np.linalg.norm(start_pos - end_pos)
    rot_diff = angle_between_rotmat(start_rotmat, end_rotmat)
    pos_num_steps = np.ceil(pos_diff / pos_step_size)
    rot_num_steps = np.ceil(rot_diff / rot_step_size)
    num_path_poses = int(max(pos_num_steps, rot_num_steps))
    num_path_poses = max(num_path_poses, 2)  # at least start and end poses
    return num_path_poses


def farthest_point_sampling(pc, num_points):
    """
    Given a point cloud, sample num_points points that are the farthest apart.
    Use o3d farthest point sampling.
    """
    assert pc.ndim == 2 and pc.shape[1] == 3, "pc must be a (N, 3) numpy array"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    downpcd_farthest = pcd.farthest_point_down_sample(num_points)
    return np.asarray(downpcd_farthest.points)
