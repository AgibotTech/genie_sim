# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from scipy.spatial.transform import Rotation as R

from common.data_filter.runtime_checker.checker_base import ValueChecker
from common.data_filter.runtime_checker.checker_factory import register_checker
from common.data_filter.runtime_checker.distance_to_target_checker import calculate_offset


@register_checker("local_axis_angle")
class LocalAxisAngleChecker(ValueChecker):
    """
    Check if angle difference (radians) between object's local specified axis and target vector meets requirements

    Parameters:
        object_id: Object ID (string)
        axis: Local axis to check, "x", "y", or "z"
        target_vector: Target vector [x, y, z]
        value: Threshold (radians)
        rule: Comparison rule, such as "lessThan", "greaterThan", "equalTo"
        object_offset: Object offset (optional)
    """

    def __init__(
        self,
        object_id,
        axis,
        target_vector,
        value,
        rule="lessThan",
        object_offset=None,
        **kwargs,
    ):
        super().__init__(name="local_axis_angle_checker", value=value, rule=rule, **kwargs)
        self.object_id = object_id
        self.axis = axis
        self.target_vector = target_vector
        self.object_offset = object_offset

    def get_value(self):
        """Calculate angle difference (radians)"""
        # Get object's world pose
        _, world_ori = self.get_world_pose(self.object_id, self.object_offset)

        # Get local axis direction in world coordinate system
        local_axis_world = self._get_local_axis_world_direction(world_ori, self.axis)

        # Calculate angle difference
        angle_diff = self._calculate_angle_difference(local_axis_world, self.target_vector)

        return angle_diff

    def _get_local_axis_world_direction(self, world_ori, axis):
        """
        Get object's local axis direction vector in world coordinate system

        Parameters:
            world_ori: Object's world rotation quaternion [w, x, y, z]
            axis: Local axis name "x", "y", or "z"

        Returns:
            Direction vector in world coordinate system [x, y, z]
        """
        # Define local axis direction vectors
        axis_map = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }

        if axis not in axis_map:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")

        local_axis = axis_map[axis]

        # Create rotation object (scipy uses [x, y, z, w] format)
        rotation = R.from_quat([world_ori[1], world_ori[2], world_ori[3], world_ori[0]])

        # Transform local axis to world coordinate system
        world_axis = rotation.apply(local_axis)

        # Normalize
        world_axis = world_axis / np.linalg.norm(world_axis)

        return world_axis

    def _calculate_angle_difference(self, vec1, vec2):
        """
        Calculate angle difference (radians) between two vectors

        Parameters:
            vec1: First vector [x, y, z]
            vec2: Second vector [x, y, z]

        Returns:
            Angle difference (radians), range [0, π]
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        # Normalize
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        # Calculate dot product
        dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)

        # Calculate angle (radians)
        angle = np.abs(np.arccos(dot_product))

        return angle

    def get_world_pose(self, obj_id, obj_offset):
        """Get object's world pose"""
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
