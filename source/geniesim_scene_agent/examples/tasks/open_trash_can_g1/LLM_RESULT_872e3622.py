# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
"""
scene_name: open_trash_can_g2
description: A trash can placed at the center of a table with random rotation and position, plus a piece of trash on the table
"""


@register()
def trash_can() -> Shape:
    """
    Creates a trash can object loaded from assets library.
    """
    return library_call(
        "usd",
        oid="benchmark_trash_can_015",
        keywords=["trash_can", "black", "cylinder", "waste_container"],
    )


@register()
def place_trash_can_on_table(table_shape: Shape) -> Shape:
    """
    Places the trash can at a random position on the table's top surface with random rotation.
    Args:
        table_shape (Shape): The shape of the table object.
    Returns:
        Shape: Combined shape of table with trash can positioned on top.
    """
    # Get table information in world coordinates
    table_info = get_object_info(table_shape)

    # Get trash can subpart info to understand its dimensions
    trash_can_subpart_info = get_subpart_info(
        object_id="benchmark_trash_can_015", subpart_id="body"
    )

    # Calculate the top surface z-coordinate of the table
    table_top_z = table_info["center"][2] + trash_can_subpart_info["xyz_max"][2] + 0.4
    print(f"table_top_z: {table_top_z}")
    # Generate random position for trash can
    # x range: (-0.06, -0.04), y range: (-0.02, 0.02)
    trash_can_x = random.uniform(-0.05, 0.1)
    trash_can_y = random.uniform(-0.1, 0.1)
    trash_can_pos = np.array([trash_can_x, trash_can_y, table_top_z])

    # Generate random rotation around z-axis (175~185 degrees)
    # Convert to radians: 175° = 175 * π/180, 185° = 185 * π/180
    angle_degrees = random.uniform(170, 190)
    angle_radians = angle_degrees * math.pi / 180

    # Create the trash can shape
    trash_can_shape = library_call("trash_can")

    # Transform trash can to the calculated position
    trash_can_transformed = transform_shape(
        trash_can_shape, translation_matrix(trash_can_pos)
    )

    # Apply rotation around z-axis
    trash_can_center = compute_shape_center(trash_can_transformed)
    trash_can_transformed = transform_shape(
        trash_can_transformed,
        rotation_matrix(angle_radians, direction=(0, 0, 1), point=trash_can_center)
    )

    # Combine table and trash can shapes
    return concat_shapes(table_shape, trash_can_transformed)


@register()
def table_with_trash_can() -> Shape:
    """
    Main scene composition: table with trash can at center and a piece of trash on the table.
    """
    # Load table at origin (0, 0, 0) - default ground plane is z=0
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )

    # Place trash can on the table (with random position and rotation)
    table_with_trash_can = place_trash_can_on_table(table_shape)

    return table_with_trash_can


@register()
def root_scene() -> Shape:
    """
    Root scene entry point.
    """
    return table_with_trash_can()
