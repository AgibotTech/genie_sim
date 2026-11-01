# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np
import math

"""
scene_name: table_with_scattered_and_stacked_blocks
description: A table workspace (table_000) with exactly three scattered building blocks (red, yellow, purple)
placed in two specific regions with a minimum center distance of 0.1m.
A stack of three blocks is placed in a separate demonstration area.
"""

def get_table_top_z(table_shape: Shape) -> float:
    """
    Helper to get the Z height of the table top surface.
    """
    table_info = get_object_info(table_shape)
    try:
        subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
        top_z = table_info["center"][2] + subpart_info["xyz_max"][2]
    except:
        top_z = table_info["max"][2]
    return top_z

@register()
def scattered_blocks(table_shape: Shape) -> Shape:
    """
    Places exactly 3 blocks (Red, Yellow, Purple) in two specific regions on the table.
    Regions:
    1. x(-0.05, 0.1), y(-0.2, -0.1)
    2. x(-0.05, 0.1), y(0.1, 0.2)
    Constraint: Center distance >= 0.1m.
    """
    top_z = get_table_top_z(table_shape)

    blocks = [
        {"oid": "benchmark_building_blocks_073", "color": "red", "name": "red_block"},
        {"oid": "benchmark_building_blocks_085", "color": "yellow", "name": "yellow_block"},
        {"oid": "benchmark_building_blocks_089", "color": "purple", "name": "purple_block"}
    ]

    positions = []
    shapes = []

    # Place 3 blocks
    for block in blocks:
        placed = False
        attempts = 0
        while not placed and attempts < 500:
            # Randomly choose a region to ensure distribution
            region = np.random.choice([0, 1])

            if region == 0:
                x = np.random.uniform(-0.05, 0.1)
                y = np.random.uniform(-0.2, -0.1)
            else:
                x = np.random.uniform(-0.05, 0.1)
                y = np.random.uniform(0.1, 0.2)

            # Check distance constraint (>= 0.1m)
            valid = True
            for px, py in positions:
                if np.hypot(x - px, y - py) < 0.1:
                    valid = False
                    break

            if valid:
                positions.append((x, y))

                # Create block
                block_shape = library_call(
                    "usd",
                    oid=block["oid"],
                    keywords=[block["name"], "building_block", block["color"], "cube", "scattered"]
                )

                # Translate to table surface
                block_shape = transform_shape(block_shape, translation_matrix([x, y, top_z]))

                # Random rotation +/- 10 degrees
                angle = np.random.uniform(-math.pi/18, math.pi/18)
                center = compute_shape_center(block_shape)
                block_shape = transform_shape(
                    block_shape,
                    rotation_matrix(angle, direction=(0, 0, 1), point=center)
                )

                shapes.append(block_shape)
                placed = True
            attempts += 1

    return concat_shapes(*shapes)

@register()
def stacked_blocks(table_shape: Shape) -> Shape:
    """
    Stacks 3 blocks in a specific area: x(0.23, 0.27), y(-0.3, 0.3).
    Random order, consistent rotation.
    """
    top_z = get_table_top_z(table_shape)

    stack_x = np.random.uniform(0.23, 0.27)
    stack_y = np.random.uniform(-0.3, 0.3)

    # Consistent rotation for the stack
    stack_angle = np.random.uniform(-math.pi/18, math.pi/18)

    blocks = [
        {"oid": "benchmark_building_blocks_073", "color": "red"},
        {"oid": "benchmark_building_blocks_085", "color": "yellow"},
        {"oid": "benchmark_building_blocks_089", "color": "purple"}
    ]

    order = np.random.permutation(3)
    z_offsets = [0.0, 0.04, 0.08]
    pos_tags = ["bottom", "middle", "top"]

    shapes = []
    for i, idx in enumerate(order):
        block = blocks[idx]
        z = top_z + z_offsets[i]

        block_shape = library_call(
            "usd",
            oid=block["oid"],
            keywords=[f"stacked_{pos_tags[i]}", "building_block", block["color"], "cube", "stacked_demo", pos_tags[i]]
        )

        block_shape = transform_shape(block_shape, translation_matrix([stack_x, stack_y, z]))

        center = compute_shape_center(block_shape)
        block_shape = transform_shape(
            block_shape,
            rotation_matrix(stack_angle, direction=(0, 0, 1), point=center)
        )

        shapes.append(block_shape)

    return concat_shapes(*shapes)

@register()
def root_scene() -> Shape:
    """
    Main scene composition.
    """
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )

    scattered = library_call("scattered_blocks", table_shape=table_shape)
    stacked = library_call("stacked_blocks", table_shape=table_shape)

    return concat_shapes(table_shape, scattered, stacked)
