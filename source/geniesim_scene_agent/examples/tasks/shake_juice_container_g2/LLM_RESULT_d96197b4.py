# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np

"""
scene_name: table_with_juice_bottle
description: A juice bottle randomly selected from available beverage bottles, placed on a table with specific bounds and random rotation
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
    desktop_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2] - 0.1

    # The final desktop position (center_x, center_y, top_z)
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])

    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


@register()
def juice_bottle() -> Shape:
    """
    Creates a juice bottle with random selection from available beverage bottles
    and random rotation around z-axis.
    """
    # Randomly select between the two available beverage bottles
    bottle_options = ["benchmark_beverage_bottle_008", "genie_beverage_bottle_009"]
    selected_bottle = np.random.choice(bottle_options)

    # Load the selected bottle from assets library
    bottle_shape = library_call(
        "usd",
        oid=selected_bottle,
        keywords=["juice_bottle", "beverage", "bottle", "glass", "transparent"],
    )

    # Get bottle center for rotation transformation
    bottle_center = compute_shape_center(bottle_shape)

    # Apply random rotation around z-axis (vertical axis)
    random_angle = np.random.uniform(-np.pi, np.pi)
    rotated_bottle = transform_shape(
        bottle_shape,
        rotation_matrix(random_angle, direction=(0, 0, 1), point=bottle_center),
    )

    return rotated_bottle


@register()
def place_bottle_on_table(table_shape: Shape) -> Shape:
    """
    Places a juice bottle on the table within specified bounds.

    Args:
        table_shape (Shape): The shape of the table object.

    Returns:
        Shape: Combined shape of table and bottle.
    """
    # Get table top surface information
    table_top_pos, table_top_size = get_table_top_surface(table_shape)

    # Define placement bounds as specified by user
    x_min, x_max = -0.15, 0.15
    y_min, y_max = -0.2, 0.2

    # Get bottle size to ensure it fits within bounds
    bottle_shape = library_call("juice_bottle")
    bottle_size = compute_shape_sizes(bottle_shape)

    # Calculate safe placement range (accounting for bottle half-size)
    safe_x_range = (
        table_top_pos[0] + x_min + bottle_size[0] / 2.0,
        table_top_pos[0] + x_max - bottle_size[0] / 2.0,
    )
    safe_y_range = (
        table_top_pos[1] + y_min + bottle_size[1] / 2.0,
        table_top_pos[1] + y_max - bottle_size[1] / 2.0,
    )

    # Random position within bounds
    random_x = np.random.uniform(safe_x_range[0], safe_x_range[1])
    random_y = np.random.uniform(safe_y_range[0], safe_y_range[1])

    # Bottle sits on table top surface
    bottle_z = table_top_pos[2]

    # Position the bottle
    positioned_bottle = transform_shape(
        bottle_shape,
        translation_matrix((random_x, random_y, bottle_z)),
    )

    return concat_shapes(table_shape, positioned_bottle)


@register()
def table_with_juice_bottle() -> Shape:
    """
    Main scene function: creates a table with a randomly selected juice bottle on it.
    """
    # Load table from assets library
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "wood", "desktop_with_legs"],
    )

    # Place bottle on table
    return place_bottle_on_table(table_shape)


@register()
def root_scene() -> Shape:
    """
    Root scene entry point.
    """
    return table_with_juice_bottle()
