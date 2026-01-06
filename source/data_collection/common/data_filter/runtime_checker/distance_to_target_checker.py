# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from scipy.spatial.transform import Rotation as R

from common.data_filter.runtime_checker.checker_base import ValueChecker
from common.data_filter.runtime_checker.checker_factory import register_checker


def transform_in_local_frame(world_pos, world_quat, offset, local_rotation_quat=None):
    """
    Calculate new world coordinates and rotation after object offset and rotation along its own coordinate system

    Parameters:
        world_pos: Object world coordinates [x, y, z]
        world_quat: Object world rotation quaternion [w, x, y, z]
        offset: Offset in own coordinate system [dx, dy, dz]
        local_rotation_quat: Rotation quaternion in own coordinate system [w, x, y, z] (optional)

    Returns:
        new_world_pos: New world coordinates [x, y, z]
        new_world_quat: New world rotation quaternion [w, x, y, z]
    """

    # Create current world rotation object
    current_rotation = R.from_quat([world_quat[1], world_quat[2], world_quat[3], world_quat[0]])

    # Transform local offset to world coordinate system
    offset_world = current_rotation.apply(offset)

    # Calculate new world coordinates
    new_world_pos = np.array(world_pos) + offset_world

    # Calculate new rotation
    if local_rotation_quat is not None:
        # Create local rotation object
        local_rotation = R.from_quat(
            [
                local_rotation_quat[1],
                local_rotation_quat[2],
                local_rotation_quat[3],
                local_rotation_quat[0],
            ]
        )

        # Combine rotations: current rotation * local rotation
        new_rotation = current_rotation * local_rotation
    else:
        new_rotation = current_rotation

    # Get new quaternion and convert back to [w, x, y, z] format
    new_quat_scipy = new_rotation.as_quat()  # [x, y, z, w]
    new_world_quat = [
        new_quat_scipy[3],
        new_quat_scipy[0],
        new_quat_scipy[1],
        new_quat_scipy[2],
    ]

    return new_world_pos, np.array(new_world_quat)


def transform_in_world_frame(world_pos, world_quat, offset, world_rotation_quat=None):
    """
    Calculate new world coordinates and rotation after object offset and rotation in world coordinate system

    Parameters:
        world_pos: Object world coordinates [x, y, z]
        world_quat: Object world rotation quaternion [w, x, y, z]
        offset: Offset in world coordinate system [dx, dy, dz]
        world_rotation_quat: Rotation quaternion in world coordinate system [w, x, y, z] (optional)

    Returns:
        new_world_pos: New world coordinates [x, y, z]
        new_world_quat: New world rotation quaternion [w, x, y, z]
    """

    # Directly add offset in world coordinate system
    new_world_pos = np.array(world_pos) + np.array(offset)

    # Calculate new rotation
    if world_rotation_quat is not None:
        # Create current rotation object
        current_rotation = R.from_quat([world_quat[1], world_quat[2], world_quat[3], world_quat[0]])

        # Create rotation object in world coordinate system
        world_rotation = R.from_quat(
            [
                world_rotation_quat[1],
                world_rotation_quat[2],
                world_rotation_quat[3],
                world_rotation_quat[0],
            ]
        )

        # Combine rotations: world rotation * current rotation
        # Note: Rotations in world coordinate system are left-multiplied
        new_rotation = world_rotation * current_rotation
    else:
        # If no new rotation, keep original rotation
        new_rotation = R.from_quat([world_quat[1], world_quat[2], world_quat[3], world_quat[0]])

    # Get new quaternion and convert back to [w, x, y, z] format
    new_quat_scipy = new_rotation.as_quat()  # [x, y, z, w]
    new_world_quat = [
        new_quat_scipy[3],
        new_quat_scipy[0],
        new_quat_scipy[1],
        new_quat_scipy[2],
    ]

    return new_world_pos, np.array(new_world_quat)


def calculate_offset(world_pose, world_ori, offset_frame, offset_pos, offset_ori):
    if offset_pos is None:
        offset_pos = [0, 0, 0]
    if offset_frame == "world":
        return transform_in_world_frame(world_pose, world_ori, offset_pos, offset_ori)
    elif offset_frame == "local":
        return transform_in_local_frame(world_pose, world_ori, offset_pos, offset_ori)
    else:
        raise ValueError("Invalid offset frame. Must be 'world' or 'local'.")


@register_checker("distance_to_target")
class DistanceToTargetChecker(ValueChecker):
    def __init__(
        self,
        object_id,
        target_id,
        object_offset=None,
        target_offset=None,
        ignore_axises=[],
        is_local=False,
        **kwargs,
    ):
        super().__init__(name="distance_to_target", **kwargs)
        self.object_id = object_id
        self.target_id = target_id
        self.object_offset = object_offset
        self.target_offset = target_offset
        self.ignore_axises = ignore_axises
        self.is_local = is_local

    def get_value(self):
        # Get target and object positions
        target_pos, target_ori = self.get_world_pose(self.target_id, self.target_offset)
        object_pos, object_ori = self.get_world_pose(self.object_id, self.object_offset)

        # If using local, first transform object_pos to target coordinate system
        if self.is_local:
            # Get target pose
            # Target rotation quaternion is [w, x, y, z]
            # First get object_pos - target_pose to get relative vector in world coordinates
            rel_vec = object_pos - target_pos
            # Construct rotation object
            from scipy.spatial.transform import Rotation as R

            r = R.from_quat([target_ori[1], target_ori[2], target_ori[3], target_ori[0]])
            # Calculate coordinates in target coordinate system
            rel_vec_local = r.inv().apply(rel_vec)
            # Local coordinates with target as origin
            object_pos = rel_vec_local
            target_pos = np.zeros_like(object_pos)  # target is origin in its own coordinate system

        # Handle ignore_axises
        if self.ignore_axises:
            axis_map = {"x": 0, "y": 1, "z": 2}
            indexes = [axis_map[a] for a in self.ignore_axises if a in axis_map]
            mask = np.ones(3, dtype=bool)
            mask[indexes] = False
            target_pos = target_pos[mask]
            object_pos = object_pos[mask]

        return np.linalg.norm(target_pos - object_pos)

    def get_world_pose(self, obj_id, obj_offset):
        if obj_id == "right" or obj_id == "left":
            is_right = obj_id == "right"
            world_pose, world_ori = self.command_controller._get_ee_pose(is_right)
        else:
            world_pose, world_ori = self.command_controller._get_object_pose(obj_id)
        if obj_offset is not None:
            world_pose, world_ori = calculate_offset(
                world_pose,
                world_ori,
                offset_frame=obj_offset.get("frame", "world"),
                offset_pos=obj_offset.get("position", None),
                offset_ori=obj_offset.get("orientation", None),
            )
        return world_pose, world_ori
