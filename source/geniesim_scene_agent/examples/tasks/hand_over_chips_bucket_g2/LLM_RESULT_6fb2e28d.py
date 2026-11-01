# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np
import random

"""
scene_name: table_with_chips_bucket
description: A bucket of chips placed randomly on left or right side of a table
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
def random_chips_bucket(position_tag: str) -> Shape:
    """
    Randomly selects and returns a bucket of chips from available options.

    Args:
        position_tag (str): "left" or "right" to indicate placement position.

    Returns:
        Shape: The selected chips bucket shape with position keyword.
    """
    # Randomly choose between two chip bucket options
    snack_options = ["benchmark_snack_241", "benchmark_snack_242"]
    selected_snack = random.choice(snack_options)

    chips_shape = library_call(
        "usd",
        oid=selected_snack,
        keywords=["chips_bucket", "snack", "cylinder", "food", position_tag],
    )
    return chips_shape


@register()
def place_chips_on_table(table_shape: Shape) -> Shape:
    """
    Places a random chips bucket on either the left or right side of the table.

    Args:
        table_shape (Shape): The table shape to place chips on.

    Returns:
        Shape: Combined table and chips shape.
    """
    # Get table top surface info
    table_top_center, table_top_size = get_table_top_surface(table_shape)

    # Randomly select left or right side
    is_left = random.choice([True, False])

    if is_left:
        # Left side: x (-0.15, 0.1), y (0.15, 0.25)
        pos_x = np.random.uniform(-0.15, 0.1)
        pos_y = np.random.uniform(0.15, 0.25)
        position_tag = "left"
    else:
        # Right side: x (-0.15, 0.1), y (-0.25, -0.15)
        pos_x = np.random.uniform(-0.15, 0.1)
        pos_y = np.random.uniform(-0.25, -0.15)
        position_tag = "right"
    print("position_y: ", pos_y, "location: ", "right" if not is_left else "left")
    # Get chips bucket with position tag in keywords
    chips_shape = library_call("random_chips_bucket", position_tag=position_tag)

    # Get chips size for proper placement
    chips_size = compute_shape_sizes(chips_shape)

    # Calculate chips center height (place on table top)
    chips_bottom_z = table_top_center[2]
    chips_center_z = chips_bottom_z + chips_size[2] / 2.0 - 0.2

    # Create translation matrix for chips position
    chips_position = np.array([pos_x, pos_y, chips_center_z])

    # Apply random rotation around z-axis
    random_rotation_angle = np.random.uniform(0, 2 * np.pi)

    # Transform chips: first translate, then rotate
    chips_transformed = transform_shape(
        chips_shape,
        translation_matrix(chips_position),
    )
    chips_center = compute_shape_center(chips_transformed)
    # Rotate around z-axis at the chips center
    chips_final = transform_shape(
        chips_transformed,
        rotation_matrix(random_rotation_angle, direction=(0, 0, 1), point=chips_center),
    )

    # Combine table and chips
    return concat_shapes(
        table_shape,
        chips_final,
    )


@register()
def table_with_chips_bucket() -> Shape:
    """
    Main scene function: creates a table with a chips bucket placed on it.

    Returns:
        Shape: The complete scene with table and chips.
    """
    # Load table from assets
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )

    # Place chips on table
    return place_chips_on_table(table_shape)


@register()
def root_scene() -> Shape:
    """
    Root scene entry point.

    Returns:
        Shape: The complete scene.
    """
    return table_with_chips_bucket()
