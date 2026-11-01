# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np

"""
scene_name: table_with_pot
description: A pot placed on a table with specific position offsets and rotation
"""


def get_table_top_surface(table_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the table's top surface center and its size.

    Args:
        table_shape (Shape): The shape of the table object.

    Returns:
        Tuple[P, P]: A tuple containing the table top's center position and its size.
    """
    table_info = get_object_info(table_shape)
    # Get subpart info for desktop (assuming "desktop" is the subpart name for table top)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # The desktop center in world coordinates
    desktop_center = table_info["center"] + desktop_subpart_info["center"]

    # The top surface z-coordinate of the desktop in world coordinates
    desktop_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    # The final desktop position (center_x, center_y, top_z)
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])

    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


@register()
def pot_on_table(table_shape: Shape) -> Shape:
    """
    Places a pot on the table with specified position offsets and rotation.

    Args:
        table_shape (Shape): The shape of the table object.

    Returns:
        Shape: Combined shape of table and pot.
    """
    # Get table top surface information
    table_top_pos, table_top_size = get_table_top_surface(table_shape)

    # Random x offset within (-0.2, -0.1) range
    x_offset = np.random.uniform(-0.2, -0.05)

    # Random y offset within (-0.02, 0.02) range
    y_offset = np.random.uniform(-0.1, 0.1)

    # Random rotation around z-axis between 85-95 degrees (convert to radians)
    rotation_angle_deg = np.random.uniform(80, 100)
    rotation_angle_rad = np.radians(rotation_angle_deg)

    # Load the pot
    pot_shape = library_call(
        "usd",
        oid="benchmark_pot_001",
        keywords=["pot", "kitchenware", "cooking_pot", "on_table"],
    )

    # Get pot size to calculate proper placement height
    pot_info = get_object_info(pot_shape)
    pot_height = compute_shape_sizes(pot_shape)[2]

    # Calculate pot bottom position (z = table top z)
    pot_bottom_z = table_top_pos[2]

    # Final pot center position
    pot_center_pos = np.array(
        [
            table_top_pos[0] + x_offset,
            table_top_pos[1] + y_offset,
            pot_bottom_z + pot_height / 2.0,  # Center is half height above bottom
        ]
    )

    # Translate pot to position
    pot_translated = transform_shape(
        pot_shape,
        translation_matrix(pot_center_pos),
    )

    # Rotate pot around z-axis
    pot_rotated = transform_shape(
        pot_translated,
        rotation_matrix(rotation_angle_rad, direction=(0, 0, 1), point=pot_center_pos),
    )

    # Combine table and pot
    return concat_shapes(table_shape, pot_rotated)


@register()
def table_with_pot() -> Shape:
    """
    Main scene function that creates a table with a pot on it.

    Returns:
        Shape: Combined shape of table and pot.
    """
    # Load table on ground (z=0.0 by default)
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "furniture", "ground_level"],
    )

    # Place pot on table
    return pot_on_table(table_shape)


@register()
def root_scene() -> Shape:
    """
    Root scene entry point.

    Returns:
        Shape: The complete scene.
    """
    return table_with_pot()
