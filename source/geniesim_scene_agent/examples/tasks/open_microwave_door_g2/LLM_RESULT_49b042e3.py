# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np

"""
scene_name: table_with_microwave
description: A microwave placed on a table with specific rotation and position constraints
"""


def get_desktop_top_position(desk_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the desktop's top surface center and its size.

    Args:
        desk_shape (Shape): The shape of the desk object.

    Returns:
        Tuple[P, P]: A tuple containing the desktop's center position and its size.
    """
    desk_info = get_object_info(desk_shape)
    # Get subpart info for desktop from assets
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # The desktop center in world coordinates
    desktop_center = desk_info["center"] + desktop_subpart_info["center"]

    # The top surface z-coordinate of the desktop in world coordinates
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    # The final desktop position (center_x, center_y, top_z)
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])

    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


@register()
def microwave_on_table(microwave_oid: str) -> Shape:
    """
    Creates a microwave with random rotation around z-axis (175-185 degrees)
    and places it at specified position constraints.

    Args:
        microwave_oid (str): Object ID for the microwave.

    Returns:
        Shape: The transformed microwave shape.
    """
    # Load microwave from assets
    microwave_shape = library_call(
        "usd",
        oid=microwave_oid,
        keywords=["microwave", "kitchen_appliance", "gray", "rectangular"],
    )

    # Get microwave size for positioning
    microwave_size = compute_shape_sizes(microwave_shape)

    # Random rotation angle between 175-185 degrees (convert to radians)
    rotation_angle_deg = np.random.uniform(175, 185)
    rotation_angle_rad = np.radians(rotation_angle_deg)

    # Random position within specified ranges
    # x range: (-0.07, -0.12459)
    x_pos = np.random.uniform(-0.12459, -0.07)
    # y range: (-0.05, 0.03)
    y_pos = np.random.uniform(-0.05, 0.03)
    # z is fixed at 0.87465 (absolute height)
    z_pos = 0.87465 - 0.14135

    # First translate to target position
    translated_shape = transform_shape(
        microwave_shape,
        translation_matrix((x_pos, y_pos, z_pos)),
    )

    # Compute center after translation for rotation
    microwave_center = compute_shape_center(translated_shape)

    # Then rotate around z-axis
    rotated_shape = transform_shape(
        translated_shape,
        rotation_matrix(
            rotation_angle_rad,
            direction=(0, 0, 1),
            point=microwave_center,
        ),
    )

    return rotated_shape


@register()
def table_with_microwave() -> Shape:
    """
    Main scene composition: table with microwave on top.

    Returns:
        Shape: Combined scene shape.
    """
    # Load table from assets
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )

    # Create microwave with specified constraints
    microwave_shape = library_call("microwave_on_table", microwave_oid="benchmark_microwave_oven_003")

    # Combine table and microwave
    return concat_shapes(table_shape, microwave_shape)


@register()
def root_scene() -> Shape:
    """
    Root scene entry point.

    Returns:
        Shape: The complete scene.
    """
    return table_with_microwave()
