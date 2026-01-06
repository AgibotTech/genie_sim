# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np
import random

"""
scene_name: random_number_blocks_on_desk
description: 4-7 number-shaped building blocks randomly placed on a desk surface at a specific workspace center.
"""

# The workspace center provided by the user
WORKSPACE_CENTER = np.array([0.0, 0.0, 0.80])
# Define a safe placement area on the desk (width and depth)
# Original desk area
DESK_HALF_WIDTH = 0.22  # x direction
DESK_HALF_DEPTH = 0.4  # y direction
# Edge margin to keep blocks away from table edges (at least 15cm from edges)
EDGE_MARGIN = 0.15
# Safe placement area (reduced by edge margin)
SAFE_HALF_WIDTH = DESK_HALF_WIDTH - EDGE_MARGIN
SAFE_HALF_DEPTH = DESK_HALF_DEPTH - EDGE_MARGIN
# Minimum distance between block centers (10cm = 0.1m)
MIN_DISTANCE = 0.1

BLOCK_ASSETS = [
    {"oid": "benchmark_building_blocks_034", "number": 0},
    {"oid": "benchmark_building_blocks_035", "number": 1},
    {"oid": "benchmark_building_blocks_036", "number": 2},
    {"oid": "benchmark_building_blocks_037", "number": 3},
    {"oid": "benchmark_building_blocks_038", "number": 4},
    {"oid": "benchmark_building_blocks_039", "number": 5},
    {"oid": "benchmark_building_blocks_040", "number": 6},
    {"oid": "benchmark_building_blocks_041", "number": 7},
    {"oid": "benchmark_building_blocks_042", "number": 8},
]


@register()
def single_number_block(oid: str, number: int, pos: P, rotation_z: float, tag: str) -> Shape:
    """
    Creates and transforms a single number block.
    """
    # Generate keywords for the object
    keywords = [f"block_number_{number}", "building_block", "wood", "toy", "on_table", tag]

    # Load the block
    block_shape = library_call("usd", oid=oid, keywords=keywords)

    # Apply rotation around its own center (which is at the bottom after usd call)
    # Since it's at origin (0,0,0) before translation, we rotate around (0,0,0)
    block_shape = transform_shape(block_shape, rotation_matrix(rotation_z, direction=(0, 0, 1), point=(0, 0, 0)))

    # Translate to the target position on the desk
    block_shape = transform_shape(block_shape, translation_matrix(pos))

    return block_shape


@register()
def random_number_blocks() -> Shape:
    """
    Randomly selects 4-7 blocks and places them on the desk without collisions.
    Ensures minimum distance of 10cm between blocks and keeps them away from table edges.
    """
    num_to_place = np.random.randint(4, 8)
    selected_assets = random.sample(BLOCK_ASSETS, num_to_place)

    placed_positions = []
    all_blocks_shape = []

    for asset in selected_assets:
        # Try to find a non-colliding position
        max_retries = 200  # Increased retries due to stricter distance requirement
        found_pos = False

        for _ in range(max_retries):
            # Random x, y within safe area (away from edges)
            rand_x = WORKSPACE_CENTER[0] + np.random.uniform(-SAFE_HALF_WIDTH, SAFE_HALF_WIDTH)
            rand_y = WORKSPACE_CENTER[1] + np.random.uniform(-SAFE_HALF_DEPTH, SAFE_HALF_DEPTH)
            new_pos = np.array([rand_x, rand_y, WORKSPACE_CENTER[2]])

            # Check collision with already placed blocks
            collision = False
            for p_pos in placed_positions:
                if np.linalg.norm(new_pos[:2] - p_pos[:2]) < MIN_DISTANCE:
                    collision = True
                    break

            if not collision:
                # Determine position tag based on y coordinate relative to workspace center
                # +y is left, -y is right
                pos_tag = "left" if rand_y > WORKSPACE_CENTER[1] else "right"

                # Random rotation for natural look
                rand_rot = np.random.uniform(0, 2 * math.pi)

                # Create the block
                block = library_call(
                    "single_number_block",
                    oid=asset["oid"],
                    number=asset["number"],
                    pos=new_pos,
                    rotation_z=rand_rot,
                    tag=pos_tag,
                )

                all_blocks_shape.append(block)
                placed_positions.append(new_pos)
                found_pos = True
                break

        if not found_pos:
            # If we can't find a spot after retries, we just skip this block
            print(
                f"Warning: Could not find position with {MIN_DISTANCE}m spacing for block number {asset['number']} after {max_retries} attempts"
            )
            continue

    return concat_shapes(*all_blocks_shape)


@register()
def root_scene() -> Shape:
    """
    Generates the scene with a table and number blocks placed on it.
    """
    # Load the table at the origin
    table_shape = library_call("usd", oid="table_000", keywords=["table", "wooden", "furniture", "base_surface"])

    # Place number blocks on the table
    blocks_shape = library_call("random_number_blocks")

    return concat_shapes(table_shape, blocks_shape)
