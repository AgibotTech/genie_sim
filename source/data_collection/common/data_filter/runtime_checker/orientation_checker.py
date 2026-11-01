from common.data_filter.runtime_checker.checker_base import ValueChecker
from common.data_filter.runtime_checker.checker_factory import register_checker
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_axis_in_world_frame(world_quat, axis_name):
    """
    Compute the world-frame direction of an object-local axis.

    Args:
        world_quat: Object world rotation quaternion [w, x, y, z].
        axis_name: One of x, y, z, -x, -y, or -z.

    Returns:
        World-frame axis direction [x, y, z].
    """
    # Define the local coordinate axes.
    axis_map = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
        "-x": np.array([-1, 0, 0]),
        "-y": np.array([0, -1, 0]),
        "-z": np.array([0, 0, -1]),
    }

    axis_name_lower = axis_name.lower()
    if axis_name_lower not in axis_map:
        raise ValueError(f"Invalid axis name: {axis_name}. Must be one of {list(axis_map.keys())}")

    local_axis = axis_map[axis_name_lower]

    # Create the rotation object; scipy uses the [x, y, z, w] quaternion order.
    rotation = R.from_quat([world_quat[1], world_quat[2], world_quat[3], world_quat[0]])

    # Transform the local axis into the world frame.
    world_axis = rotation.apply(local_axis)

    return world_axis


def calculate_angle_between_vectors(vec1, vec2):
    """
    Compute the angle between two vectors in degrees.

    Args:
        vec1: First vector [x, y, z].
        vec2: Second vector [x, y, z].

    Returns:
        Angle in degrees, in the range [0, 180].
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # Normalize the vectors.
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)

    # Compute the dot product.
    dot_product = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)

    # Compute the angle and convert radians to degrees.
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


@register_checker("orientation")
class OrientationChecker(ValueChecker):
    """
    Check the angular differences between object axes and target directions.

    Args:
        object_id: Object ID, or right/left for an end effector.
        target_orientation: Axis-to-direction mapping, for example
            {"y": [0, 0, 1], "x": [0, 1, 0]}. Keys support x, y, z,
            -x, -y, and -z; values are target direction vectors [x, y, z].
        aggregate: How to combine angular differences for multiple axes:
            max returns the largest difference (default), mean the average,
            sum the total, and min the smallest difference. A single axis
            returns its angular difference directly.

    Example:
        OrientationChecker(
            object_id="bottle_001",
            target_orientation={"y": [0, 0, 1], "x": [0, 1, 0]},
            aggregate="max",
            value=10,  # Angle threshold in degrees.
            rule="lessThan"  # Pass when the angular difference is below 10 degrees.
        )
    """

    def __init__(self, object_id, target_orientation, aggregate="max", **kwargs):
        super().__init__(name="orientation", **kwargs)
        self.object_id = object_id
        self.target_orientation = target_orientation
        self.aggregate = aggregate

        # Validate the target_orientation format.
        if not isinstance(target_orientation, dict) or len(target_orientation) == 0:
            raise ValueError("target_orientation must be a non-empty dict, e.g. {'y': [0, 0, 1]}")

        for axis, direction in target_orientation.items():
            if axis.lower() not in ["x", "y", "z", "-x", "-y", "-z"]:
                raise ValueError(
                    f"Invalid axis: {axis}. Must be one of 'x', 'y', 'z', '-x', '-y', '-z'"
                )
            if len(direction) != 3:
                raise ValueError(f"Direction must be a 3D vector, got {direction}")

    def get_world_orientation(self):
        """
        Get the object world rotation quaternion.

        Returns:
            world_quat: Quaternion [w, x, y, z].
        """
        if self.object_id == "right" or self.object_id == "left":
            is_right = self.object_id == "right"
            _, world_ori = self.command_controller._get_ee_pose(is_right)
        else:
            _, world_ori = self.command_controller._get_object_pose(self.object_id)
        return world_ori

    def get_axis_angles(self):
        """
        Compute angular differences between object axes and target directions.

        Returns:
            dict: Mapping from axis names to angular differences.
        """
        world_ori = self.get_world_orientation()

        angles = {}
        for axis_name, target_direction in self.target_orientation.items():
            # Get the object axis direction in the world frame.
            world_axis = get_axis_in_world_frame(world_ori, axis_name)

            # Compute its angular difference from the target direction.
            angle = calculate_angle_between_vectors(world_axis, target_direction)
            angles[axis_name] = angle

        return angles

    def get_value(self):
        """
        Return the angular difference using the configured aggregation method.

        Returns:
            float: Aggregated angular difference.
        """
        angles = self.get_axis_angles()
        angle_values = list(angles.values())

        if len(angle_values) == 1:
            return angle_values[0]

        if self.aggregate == "max":
            return max(angle_values)
        elif self.aggregate == "mean":
            return np.mean(angle_values)
        elif self.aggregate == "sum":
            return sum(angle_values)
        elif self.aggregate == "min":
            return min(angle_values)
        else:
            raise ValueError(
                f"Invalid aggregate method: {self.aggregate}. Must be 'max', 'mean', 'sum', or 'min'"
            )

    def get_debug_info(self):
        """
        Get diagnostic information, including angular differences for each axis.

        Returns:
            dict: Complete orientation diagnostics.
        """
        world_ori = self.get_world_orientation()
        angles = self.get_axis_angles()

        debug_info = {
            "object_id": self.object_id,
            "world_orientation_wxyz": (
                world_ori.tolist() if hasattr(world_ori, "tolist") else list(world_ori)
            ),
            "target_orientation": self.target_orientation,
            "axis_angles": angles,
            "aggregated_value": self.get_value(),
            "aggregate_method": self.aggregate,
        }

        # Add the world-frame direction of each axis.
        world_axes = {}
        for axis_name in self.target_orientation.keys():
            world_axis = get_axis_in_world_frame(world_ori, axis_name)
            world_axes[axis_name] = world_axis.tolist()
        debug_info["world_axes"] = world_axes

        return debug_info
