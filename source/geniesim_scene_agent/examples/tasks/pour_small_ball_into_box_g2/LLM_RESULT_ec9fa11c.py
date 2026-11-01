# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np
import math

"""
scene_name: table_with_cup_and_box
description: A table with a paper cup containing a small ball and a storage box placed on top.
             The cup is placed in specific ranges on the table (left-back or right-back),
             and the box is near the center. Constraints on distance and rotation are applied.
"""


@register()
def table() -> Shape:
    """
    Create the base table.
    """
    return library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "furniture"],
    )


@register()
def storage_box() -> Shape:
    """
    Create the storage box.
    """
    return library_call(
        "usd",
        oid="benchmark_storage_box_007",
        keywords=["storage_box", "rectangular", "plastic", "container"],
    )


@register()
def paper_cup() -> Shape:
    """
    Create the paper cup.
    """
    return library_call(
        "usd",
        oid="benchmark_cup_112",
        keywords=["paper_cup", "cylinder", "white", "container"],
    )


@register()
def small_ball() -> Shape:
    """
    Create the small ball to be placed inside the cup.
    """
    return library_call(
        "usd",
        oid="benchmark_building_blocks_111",
        keywords=["ball", "sphere", "small", "toy"],
    )


@register()
def root_scene() -> Shape:
    """
    Main function to assemble the scene.
    """
    # 1. Setup Table
    table_shape = table()
    table_info = get_object_info(table_shape)

    # Get desktop height relative to table center
    desktop_info = get_subpart_info("table_000", "desktop")
    table_center = table_info["center"]
    surface_z = table_center[2] + desktop_info["xyz_max"][2] - 0.125

    # 2. Setup Storage Box
    box_shape = storage_box()

    # Box position: x in (-0.05, 0.01), y in (-0.01, 0.01)
    box_pos_x = np.random.uniform(-0.05, 0.01)
    box_pos_y = np.random.uniform(-0.01, 0.01)

    # Box rotation: small angle around Z (< 7 degrees)
    box_angle_rad = np.random.uniform(np.radians(-7), np.radians(7))

    # Apply transformations
    box_shape = transform_shape(
        box_shape,
        rotation_matrix(box_angle_rad, direction=(0, 0, 1), point=(0, 0, 0)),
    )
    box_shape = transform_shape(
        box_shape,
        translation_matrix((box_pos_x, box_pos_y, surface_z)),
    )

    # Store box center for distance calculation
    box_center_xy = np.array([box_pos_x, box_pos_y])

    # 3. Setup Paper Cup with constraints
    cup_shape = paper_cup()

    valid_cup_pos = False
    cup_pos_x = 0.0
    cup_pos_y = 0.0
    cup_side_tag = ""

    # Find a valid position for the cup satisfying distance constraint
    while not valid_cup_pos:
        region_choice = np.random.choice(["A", "B"])

        if region_choice == "A":
            # Region A: x(-0.2, 0.1), y(-0.2, -0.1) -> Right side
            cup_pos_x = np.random.uniform(-0.1, 0.2)
            cup_pos_y = np.random.uniform(-0.2, -0.1)
            cup_side_tag = "right"
        else:
            # Region B: x(-0.2, 0.1), y(0.1, 0.2) -> Left side
            cup_pos_x = np.random.uniform(-0.1, 0.2)
            cup_pos_y = np.random.uniform(0.1, 0.2)
            cup_side_tag = "left"

        # Check distance constraint: distance between centers >= 0.15
        dist = np.linalg.norm(np.array([cup_pos_x, cup_pos_y]) - box_center_xy)
        if dist >= 0.2:
            valid_cup_pos = True

    # Cup rotation: random around Z axis
    cup_angle_rad = np.random.uniform(0, 2 * np.pi)

    # Apply transformations
    cup_shape = transform_shape(
        cup_shape,
        rotation_matrix(cup_angle_rad, direction=(0, 0, 1), point=(0, 0, 0)),
    )
    cup_shape = transform_shape(
        cup_shape,
        translation_matrix((cup_pos_x, cup_pos_y, surface_z)),
    )

    # 4. Setup Small Ball inside Cup
    ball_shape = small_ball()

    # Calculate position inside the cup
    cup_info = get_object_info(cup_shape)
    cup_min = cup_info["min"]
    cup_max = cup_info["max"]

    # Place ball at the center of the cup's xy base and half height up
    ball_x = (cup_min[0] + cup_max[0]) / 2.0
    ball_y = (cup_min[1] + cup_max[1]) / 2.0
    ball_z = cup_min[2] + (cup_max[2] - cup_min[2]) * 0.5

    ball_shape = transform_shape(
        ball_shape,
        translation_matrix((ball_x, ball_y, ball_z)),
    )

    # 5. Combine all shapes
    return concat_shapes(table_shape, box_shape, cup_shape, ball_shape)
