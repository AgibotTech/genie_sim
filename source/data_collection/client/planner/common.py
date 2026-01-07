# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
from typing import Tuple

import numpy as np

from common.base_utils.logger import logger
from common.base_utils.transform_utils import (
    calculate_rotation_from_two_axes,
    calculate_rotation_matrix,
    euler2mat,
    rotate_around_axis,
    transform_coordinates_3d,
)

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]


def overweite_grasp_data(N=100):
    x = np.random.uniform(-0.01, 0.01, N)
    y = np.random.uniform(-0.01, 0.01, N)
    z = np.random.uniform(-0.01, 0.01, N)

    transforms = np.zeros((N, 4, 4))
    grasp_widths = np.full(N, 0.02)

    for i in range(N):
        position = np.array([x[i], y[i], z[i]])
        # Direction vector from grasp point to origin (i.e., -z direction)
        direction_to_origin = -position
        # Handle case when position is at origin
        if np.linalg.norm(direction_to_origin) < 1e-6:
            # If at origin, use default orientation (z-axis forward)
            rotation = np.eye(3)
        else:
            direction_to_origin = direction_to_origin / np.linalg.norm(direction_to_origin)
            # Align gripper's y-axis with world coordinate system y-axis as much as possible
            y_axis = np.array([0, 1, 0])
            # Calculate x-axis perpendicular to main direction
            if np.abs(np.dot(direction_to_origin, y_axis)) > 0.99:
                # Avoid parallel case, use backup reference axis
                y_axis = np.array([0, 0, 1])
                z_axis = direction_to_origin
                x_axis = np.cross(y_axis, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
            else:
                z_axis = direction_to_origin
                x_axis = np.cross(y_axis, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)

            # Build rotation matrix
            rotation = np.vstack((x_axis, y_axis, z_axis)).T

        # 3. Build 4x4 homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position

        # 4. Store result
        transforms[i] = T

    return transforms, grasp_widths


def generate_random_pose(
    pose: np.ndarray,
    position_std: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    rotation_std: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    num: int = 10,
):
    """
    Generate a random pose by adding Gaussian noise to the position and rotation.
    Args:
        pose (np.ndarray): The original pose in the form of a 4x4 transformation matrix.
        position_std (Tuple[float, float, float]): Standard deviation for the position noise.
        rotation_std (Tuple[float, float, float]): Standard deviation for the rotation noise in radians.
        num (int): Number of random poses to generate.
    Returns:
        np.ndarray: A set of new poses with added noise.
    """
    assert pose.shape == (4, 4), "Pose must be a 4x4 transformation matrix."
    # Generate random noise for position
    res = []
    for _ in range(num):
        position_noise = np.random.normal(loc=0.0, scale=position_std, size=(3,))
        # Generate random noise for rotation
        rotation_noise = np.random.normal(loc=0.0, scale=rotation_std, size=(3,))
        # Create a rotation matrix from the noise
        rotation_matrix = euler2mat(rotation_noise, order="xyz")
        # Create a new pose by applying the noise
        new_pose = np.eye(4)
        new_pose[:3, :3] = pose[:3, :3] @ rotation_matrix
        new_pose[:3, 3] = pose[:3, 3] + position_noise
        res.append(new_pose)
    return np.array(res)


def format_object(obj, distance, type="active"):
    if obj is None:
        return None
    xyz, direction = obj.xyz, obj.direction

    direction = direction / np.linalg.norm(direction) * distance
    type = type.lower()
    if type == "active":
        xyz_start = xyz
        xyz_end = xyz_start + direction
    elif type == "passive" or type == "plane":
        xyz_end = xyz
        xyz_start = xyz_end - direction

    part2obj = np.eye(4)
    part2obj[:3, 3] = xyz_start
    obj.obj2part = np.linalg.inv(part2obj)

    # prepare constraint axis (local) if available on object
    constraint_local = None
    if hasattr(obj, "constraint_axis"):
        ca = np.array(obj.constraint_axis, dtype=float)
        if np.linalg.norm(ca) > 0:
            constraint_local = ca / np.linalg.norm(ca)

    obj_info = {
        "pose": obj.obj_pose,
        "length": obj.obj_length,
        "xyz_start": xyz_start,
        "xyz_end": xyz_end,
        "obj2part": obj.obj2part,
        "constraint_local": constraint_local,
    }
    return obj_info


def obj2world(obj_info):
    obj_pose = obj_info["pose"]
    obj_length = obj_info["length"]
    obj2part = obj_info["obj2part"]
    xyz_start = obj_info["xyz_start"]
    xyz_end = obj_info["xyz_end"]

    arrow_in_obj = np.array([xyz_start, xyz_end]).transpose(1, 0)
    arrow_in_world = transform_coordinates_3d(arrow_in_obj, obj_pose).transpose(1, 0)

    xyz_start_world, xyz_end_world = arrow_in_world
    direction_world = xyz_end_world - xyz_start_world
    direction_world = direction_world / np.linalg.norm(direction_world)

    obj_info_world = {
        "pose": obj_pose,
        "length": obj_length,
        "obj2part": obj2part,
        "xyz_start": xyz_start_world,
        "xyz_end": xyz_end_world,
        "direction": direction_world,
    }
    # handle optional constraint axis local->world
    constraint_local = obj_info.get("constraint_local", None)
    if constraint_local is not None:
        constraint_world = obj_pose[:3, :3] @ constraint_local
        # ensure orthogonalization not necessary here; keep as direction vector
        obj_info_world["constraint"] = constraint_world / np.linalg.norm(constraint_world)
    return obj_info_world
    return obj_info_world


def get_aligned_fix_pose(active_obj, passive_obj, distance=0.01, N=1):
    try:
        active_object = format_object(active_obj, type="active", distance=distance)
    except Exception as e:
        logger.error(f"error format active_object{e}")
        active_object = None
    try:
        passive_object = format_object(passive_obj, type="passive", distance=distance)
    except Exception as e:
        logger.error(f"error format passive_object{e}")
        passive_object = None
    active_obj_world = obj2world(active_object)
    current_obj_pose = active_obj_world["pose"]
    if passive_object is None:
        return current_obj_pose[np.newaxis, ...]

    passive_obj_world = obj2world(passive_object)
    # use passive object's own axis if available
    try:
        passive_obj_world["direction"] = passive_obj.direction
    except Exception:
        pass

    R = calculate_rotation_matrix(active_obj_world["direction"], passive_obj_world["direction"])
    T = passive_obj_world["xyz_end"] - R @ active_obj_world["xyz_start"]
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T
    target_obj_pose = transform_matrix @ current_obj_pose
    target_obj_pose[:3, 3] = passive_obj.obj_pose[:3, 3]

    poses = []
    for angle in [i * 360 / N for i in range(N)]:
        pose_rotated = rotate_around_axis(
            target_obj_pose,
            passive_obj_world["xyz_start"],
            passive_obj_world["direction"],
            angle,
        )
        poses.append(pose_rotated)
    return np.stack(poses)


def get_aligned_pose(active_obj, passive_obj, distance=0.01, N=1):
    try:
        active_object = format_object(active_obj, type="active", distance=distance)
    except Exception as e:
        logger.error(f"error format active_object{e}")
        active_object = None
    try:
        passive_object = format_object(passive_obj, type="passive", distance=distance)
    except Exception as e:
        logger.error(f"error format passive_object{e}")
        passive_object = None

    active_obj_world = obj2world(active_object)
    current_obj_pose = active_obj_world["pose"]
    if passive_object is None:
        return current_obj_pose[np.newaxis, ...]

    passive_obj_world = obj2world(passive_object)

    # calculate rotation and translation so active's xyz_start maps to passive's xyz_end
    try:
        a_cons = active_obj_world.get("constraint", None)
        p_cons = passive_obj_world.get("constraint", None)
        if a_cons is not None and p_cons is not None:
            R = calculate_rotation_from_two_axes(
                active_obj_world["direction"],
                a_cons,
                passive_obj_world["direction"],
                p_cons,
            )
        else:
            R = calculate_rotation_matrix(active_obj_world["direction"], passive_obj_world["direction"])
    except Exception:
        R = calculate_rotation_matrix(active_obj_world["direction"], passive_obj_world["direction"])
    T = passive_obj_world["xyz_end"] - R @ active_obj_world["xyz_start"]
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T
    target_obj_pose = transform_matrix @ current_obj_pose

    # start angle: no per-object scalar reference used (rotation determined by constraint axes)
    start_angle = 0.0

    if N <= 0:
        raise ValueError("N must be >= 1")

    poses = []
    step = 360.0 / N
    for i in range(N):
        angle = start_angle + i * step
        pose_rotated = rotate_around_axis(
            target_obj_pose,
            passive_obj_world["xyz_start"],
            passive_obj_world["direction"],
            angle,
        )
        poses.append(pose_rotated)
    return np.stack(poses)


def parse_stage(stage, objects):
    action = stage["action"]
    if action in ["reset"]:
        return (
            action,
            "gripper",
            "gripper",
            None,
            None,
            None,
            None,
            stage.get("action_description", {"action_text": "", "english_action_text": ""}),
        )

    active_obj_id = stage["active"]["object_id"]
    if "part_id" in stage["active"]:
        active_obj_id += "/%s" % stage["active"]["part_id"]

    passive_obj_id = stage["passive"]["object_id"]
    if "part_id" in stage["passive"]:
        passive_obj_id += "/%s" % stage["passive"]["part_id"]

    active_obj = objects[active_obj_id]
    passive_obj = objects[passive_obj_id]

    single_obj = action in ["pull", "rotate", "slide", "shave", "brush", "wipe"]

    gripper_only = action in ["clamp", "move"]

    def _load_element(obj, type):
        if action in ["pick", "hook", "move"]:
            action_mapped = "grasp"
        else:
            action_mapped = action
        if action_mapped == "grasp" and type == "active":
            return None, None
        elif obj.name == "gripper":
            element = obj.elements[type][action_mapped]
            return element, "default"
        primitive = stage[type]["primitive"] if stage[type]["primitive"] is not None else "default"
        if primitive != "default" or (action_mapped == "grasp" and type == "passive"):
            if action_mapped not in obj.elements[type]:
                logger.info("No %s element for %s" % (action_mapped, obj.name))
                return None, None
            element = obj.elements[type][action_mapped][primitive]
        else:
            element = []
            for primitive in obj.elements[type][action_mapped]:
                _element = obj.elements[type][action_mapped][primitive]
                if isinstance(_element, list):
                    element += _element
                else:
                    element.append(_element)

        return element, primitive

    if gripper_only:
        active_element, active_primitive = _load_element(active_obj, "active")
        passive_element, passive_primitive = active_element, active_primitive
    else:
        passive_element, passive_primitive = _load_element(passive_obj, "passive")

        if not single_obj:
            active_element, active_primitive = _load_element(active_obj, "active")
        else:
            active_element, active_primitive = passive_element, passive_primitive
    return (
        action,
        active_obj_id,
        passive_obj_id,
        active_element,
        passive_element,
        active_primitive,
        passive_primitive,
        stage.get("action_description", {"action_text": "", "english_action_text": ""}),
    )
