# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np

"""
scene_name: desk_with_pen_holder
description: A pen holder with a pen placed on a table
"""


def get_pen_holder_position() -> Tuple[P, float]:
    """
    Generates random position and rotation for the pen holder on the table.

    Returns:
        Tuple[P, float]: Position (x, y, 0) and random z-rotation angle
    """
    x_pos = np.random.uniform(-0.15, 0.15)
    y_pos = np.random.uniform(-0.2, 0.2)
    z_rotation = np.random.uniform(0, 2 * np.pi)
    return (x_pos, y_pos, 0), z_rotation


def get_random_pen() -> str:
    """
    Randomly selects a pen from the available pen assets (excluding benchmark_pen_004).

    Returns:
        str: The object_id of the selected pen
    """
    # Sample a pen from the supported asset pool.
    pen_ids = [
        "benchmark_pen_000",
        "benchmark_pen_001",
        "benchmark_pen_002",
        "benchmark_pen_003",
        "benchmark_pen_005",
        "benchmark_pen_006",
    ]
    return np.random.choice(pen_ids)


@register()
def pen_in_holder(pen_holder_shape: Shape) -> Shape:
    """
    Places a randomly selected pen inside the pen holder.
    The pen is positioned at the center, slightly above, and rotated 90 degrees around x-axis.

    Args:
        pen_holder_shape (Shape): The shape of the pen holder

    Returns:
        Shape: The pen shape transformed to its final position
    """
    # Get pen holder info to find its center
    holder_info = get_object_info(pen_holder_shape)
    holder_center = holder_info["center"]

    # Select random pen
    pen_oid = get_random_pen()

    # Load the pen
    pen_shape = library_call(
        "usd",
        oid=pen_oid,
        keywords=["pen", "writing_tool", "cylinder", "long"],
    )

    # Get pen size to calculate offset
    pen_size = compute_shape_sizes(pen_shape)

    # Calculate pen position: at holder center, slightly above (add half pen height + small offset)
    # Since pen will be rotated 90 deg around x, we need to account for its length along z
    pen_offset_z = holder_center[2] + pen_size[2] / 2.0 + 0.05

    # Translate pen to position
    pen_shape = transform_shape(
        pen_shape,
        translation_matrix([holder_center[0], holder_center[1] + 0.05, holder_center[2]]),
    )

    # Rotate pen 90 degrees around world x-axis
    pen_shape = transform_shape(
        pen_shape,
        rotation_matrix(np.pi / 2, direction=(1, 0, 0), point=holder_center),
    )

    return pen_shape


@register()
def place_pen_holder_on_table(table_shape: Shape) -> Shape:
    """
    Places the pen holder on the table with random position and rotation.

    Args:
        table_shape (Shape): The shape of the table

    Returns:
        Shape: Combined table and pen holder shapes
    """
    # Get pen holder position and rotation
    pos, z_rotation = get_pen_holder_position()

    # Load pen holder
    pen_holder_shape = library_call(
        "usd",
        oid="benchmark_stationery_021",
        keywords=["pen_holder", "stationery", "container", "cylinder"],
    )

    # Get pen holder size to place it on table surface
    holder_info = get_object_info(pen_holder_shape)
    holder_min = compute_shape_min(pen_holder_shape)
    holder_max = compute_shape_max(pen_holder_shape)
    holder_height = holder_max[2] - holder_min[2]

    # Get table top surface info
    table_info = get_object_info(table_shape)
    table_top_z = table_info["center"][2] + get_subpart_info("table_000", "desktop")["xyz_max"][2] - 0.1

    # Place pen holder on table surface
    pen_holder_shape = transform_shape(
        pen_holder_shape,
        translation_matrix((pos[0], pos[1], table_top_z)),
    )

    # Apply random z-rotation
    pen_holder_shape = transform_shape(
        pen_holder_shape,
        rotation_matrix(z_rotation, direction=(0, 0, 1), point=pos),
    )

    # Add pen inside holder
    pen_shape = pen_in_holder(pen_holder_shape)

    return concat_shapes(table_shape, pen_holder_shape, pen_shape)


@register()
def desk_with_pen_holder() -> Shape:
    """
    Main scene composition: table with pen holder containing a pen.

    Returns:
        Shape: Complete scene shape
    """
    # Load table on ground (z=0 by default)
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )

    return place_pen_holder_on_table(table_shape)


@register()
def root_scene() -> Shape:
    """
    Root scene entry point.

    Returns:
        Shape: The complete scene
    """
    return desk_with_pen_holder()
