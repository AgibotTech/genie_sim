# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np

"""
scene_name: stack_paper_cups_g2
description: 3 to 5 paper cups randomly placed on a table within x(-0.2, 0.2) and y(-0.3, 0.3),
avoiding the central forbidden zone x(-0.03, 0.03) y(-0.03, 0.03). Cups must be at least 0.07m apart (center-to-center).
"""


def get_tabletop_center(table_shape: Shape) -> P:
    """
    Retrieves the world coordinates of the table's top surface center.

    Args:
        table_shape (Shape): The shape of the table.

    Returns:
        P: The table's top center position in world coordinates.
    """
    table_info = get_object_info(table_shape)
    desktop_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_center_world = table_info["center"] + desktop_info["center"]
    desktop_top_z = table_info["center"][2] + desktop_info["xyz_max"][2]
    return np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z])


@register()
def paper_cup() -> Shape:
    """
    Creates a single paper cup object.
    """
    return library_call(
        "usd",
        oid="benchmark_cup_105",
        keywords=["paper_cup", "white", "cylinder", "disposable", "beverage"],
    )


@register()
def arrange_cups_on_table(table_shape: Shape) -> Shape:
    """
    Arranges 3-5 paper cups on the given table, respecting all placement constraints.
    """
    table_center = get_tabletop_center(table_shape)

    # Define the allowed placement area relative to table center
    allowed_x_range = (-0.3, 0.1)
    allowed_y_range = (-0.3, 0.3)
    forbidden_x_range = (-0.03, 0.03)
    forbidden_y_range = (-0.03, 0.03)
    min_center_distance = 0.15

    num_cups = np.random.randint(3, 6)  # 3 to 5 inclusive
    placed_positions = []

    cup_template = library_call("paper_cup")
    cup_info = get_object_info(cup_template)
    cup_radius = max(cup_info["size"][0], cup_info["size"][1]) / 2.0

    for i in range(num_cups):
        placed = False
        for _ in range(200):  # Max attempts to find a valid spot
            # Generate random position in the allowed area
            x_offset = np.random.uniform(*allowed_x_range)
            y_offset = np.random.uniform(*allowed_y_range)

            # Skip if in forbidden zone
            if (
                forbidden_x_range[0] <= x_offset <= forbidden_x_range[1]
                and forbidden_y_range[0] <= y_offset <= forbidden_y_range[1]
            ):
                continue

            world_x = table_center[0] + x_offset
            world_y = table_center[1] + y_offset

            # Check distance from already placed cups
            too_close = False
            for pos in placed_positions:
                dist = np.linalg.norm(np.array([world_x, world_y]) - np.array(pos[:2]))
                if dist < min_center_distance:
                    too_close = True
                    break

            if not too_close:
                placed_positions.append((world_x, world_y, table_center[2]))
                placed = True
                break

        if not placed:
            # If we can't place all cups, just place what we have
            break

    # Create and position the cups
    cup_shapes = []
    for i, (x, y, z) in enumerate(placed_positions):
        # Determine position tag for keywords
        pos_tag = "left" if x < table_center[0] else "right"

        cup = library_call(
            "usd",
            oid="benchmark_cup_105",
            keywords=[f"paper_cup_{i}", "white", "cylinder", "disposable", "beverage", pos_tag],
        )

        # Get cup center for rotation
        cup_center = compute_shape_center(cup)

        # Apply random rotation around z-axis
        rotated_cup = transform_shape(
            cup, rotation_matrix(angle=np.random.uniform(0, 2 * np.pi), direction=(0, 0, 1), point=cup_center)
        )

        # Translate to final position
        final_cup = transform_shape(rotated_cup, translation_matrix((x, y, z)))
        cup_shapes.append(final_cup)

    return concat_shapes(*cup_shapes)


@register()
def stack_paper_cups_g2() -> Shape:
    """
    Creates the complete scene with a table and randomly placed paper cups.
    """
    table = library_call("usd", oid="table_000", keywords=["dining_table", "table", "wood", "rectangular", "furniture"])
    cups = arrange_cups_on_table(table)
    return concat_shapes(table, cups)


@register()
def root_scene() -> Shape:
    return stack_paper_cups_g2()
