# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np
import random

"""
scene_name: table_with_random_cup
description: A table with a randomly selected cup placed on top with random rotation
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
    try:
        desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    except Exception:
        # Fallback: use the table's max z coordinate as the top surface
        desktop_subpart_info = {
            "center": np.array([0.0, 0.0, 0.0]),
            "size": np.array([0.8, 0.6, 0.05]),  # Approximate desktop size
            "xyz_max": np.array([0.4, 0.3, 0.025]),
            "xyz_min": np.array([-0.4, -0.3, -0.025]),
        }

    # The desktop center in world coordinates
    desktop_center = table_info["center"] + desktop_subpart_info["center"]

    # The top surface z-coordinate of the desktop in world coordinates
    desktop_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2] - 0.1

    # The final desktop position (center_x, center_y, top_z)
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])

    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


@register()
def random_cup() -> Shape:
    """
    Randomly selects and returns one cup from the available cup options.

    Returns:
        Shape: The selected cup shape.
    """
    # List of available cup object IDs
    cup_options = [
        "benchmark_cup_043",
        "benchmark_cup_105",
        "benchmark_beer_mug_007",
        "benchmark_beer_mug_001",
        "benchmark_beer_mug_002",
    ]

    # Randomly select one cup
    selected_cup_id = random.choice(cup_options)

    # Create keywords for the selected cup
    keywords = ["cup", "random_selection", f"{selected_cup_id}"]

    # Load the cup using usd
    cup_shape = library_call("usd", oid=selected_cup_id, keywords=keywords)

    return cup_shape


@register()
def place_cup_on_table(table_shape: Shape) -> Shape:
    """
    Places a randomly selected cup on the table with random rotation at specified position range.

    Args:
        table_shape (Shape): The shape of the table object.

    Returns:
        Shape: Combined table and cup shapes.
    """
    # Get table top surface information
    table_top_pos, table_top_size = get_table_top_surface(table_shape)

    # Get cup shape
    cup_shape = library_call("random_cup")

    # Get cup size to ensure it fits on the table
    cup_size = compute_shape_sizes(cup_shape)

    # Define position ranges (relative to table top center)
    x_range = (-0.15, 0.15)
    y_range = (-0.2, 0.2)

    # Generate random position within the specified range
    random_x = np.random.uniform(x_range[0], x_range[1])
    random_y = np.random.uniform(y_range[0], y_range[1])

    # Calculate final cup position (on table top surface)
    # Place cup bottom on table surface (z = table_top_z)
    cup_position = np.array(
        [table_top_pos[0] + random_x, table_top_pos[1] + random_y, table_top_pos[2]]  # Cup sits on the table surface
    )

    # Translate cup to position
    cup_transformed = transform_shape(cup_shape, translation_matrix(cup_position))

    # Apply random rotation around z-axis (vertical axis)
    random_rotation_angle = np.random.uniform(-np.pi, np.pi)
    cup_center = compute_shape_center(cup_transformed)

    cup_final = transform_shape(
        cup_transformed, rotation_matrix(angle=random_rotation_angle, direction=(0, 0, 1), point=cup_center)
    )

    # Combine table and cup
    return concat_shapes(table_shape, cup_final)


@register()
def table_with_random_cup() -> Shape:
    """
    Main scene function that creates a table with a randomly selected cup on top.

    Returns:
        Shape: The complete scene with table and cup.
    """
    # Load table using usd
    table_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )

    # Place cup on table
    return place_cup_on_table(table_shape)


@register()
def root_scene() -> Shape:
    """
    Root scene entry point.

    Returns:
        Shape: The complete scene.
    """
    return table_with_random_cup()
