# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np
from typing import Tuple, List

"""
scene_name: table_with_scattered_blocks
description: Three building blocks scattered randomly on a table within specified regions,
             with random rotation and minimum distance constraints.
"""

def find_desktop_top_surface(desk_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the desktop's top surface center and its size.
    """
    desk_info = get_object_info(desk_shape)
    # Assuming standard subpart naming from assets library
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    desktop_center = desk_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]
    return desktop_pos, desktop_size


def generate_valid_positions(num_blocks: int, regions: List[Tuple[float, float, float, float]], min_dist: float) -> List[P]:
    """
    Generates random 2D positions within given regions satisfying a minimum distance constraint.
    Uses rejection sampling.
    """
    positions = []
    max_attempts = 1000
    attempts = 0

    while len(positions) < num_blocks and attempts < max_attempts:
        attempts += 1
        # Randomly select one of the valid regions
        region = regions[np.random.randint(0, len(regions))]
        x_min, x_max, y_min, y_max = region

        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        pos = np.array([x, y])

        # Check distance against already placed blocks
        valid = True
        for p in positions:
            if np.linalg.norm(pos - p) < min_dist:
                valid = False
                break
        if valid:
            positions.append(pos)

    return positions


@register()
def scattered_blocks(table_center: P, table_top_z: float) -> Shape:
    """
    Creates and places three building blocks on the table according to constraints.
    """
    # Define the two allowed placement regions relative to table center: (x_min, x_max, y_min, y_max)
    regions = [
        (-0.2, 0.05, -0.25, -0.1),
        (-0.2, 0.05, 0.1, 0.25)
    ]

    # Generate 3 valid positions with at least 0.1m distance between them
    positions = generate_valid_positions(3, regions, min_dist=0.1)

    blocks_shape = []
    for i, pos in enumerate(positions):
        # Load block asset
        block = library_call(
            "usd",
            oid="benchmark_building_blocks_074",
            keywords=[f"scattered_block_{i}", "cube", "building_block", "toy", f"region_{i}"]
        )

        # Scale to target side length 0.05m (Note: prompt says 0.05cm, assumed 0.05m for standard scene scale)
        current_size = compute_shape_sizes(block)
        scale_factor = 0.05 / current_size[0]
        block_center = compute_shape_center(block)
        block = transform_shape(block, scale_matrix(scale_factor, block_center))

        # Random rotation around Z-axis within +/- 10 degrees
        angle = np.random.uniform(-np.pi / 18, np.pi / 18)
        block_center = compute_shape_center(block) # Recompute center after scaling
        block = transform_shape(block, rotation_matrix(angle, direction=(0, 0, 1), point=block_center))

        # Calculate Z translation to sit exactly on table surface
        block_min_z = compute_shape_min(block)[2]
        z_offset = table_top_z - block_min_z

        # Translate to final world position (region coords are relative to table center)
        world_x = table_center[0] + pos[0]
        world_y = table_center[1] + pos[1]
        block = transform_shape(block, translation_matrix((world_x, world_y, z_offset)))

        blocks_shape.append(block)

    return concat_shapes(*blocks_shape)


@register()
def table_with_scattered_blocks() -> Shape:
    """
    Loads the table and places the scattered blocks on top.
    """
    desk_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "furniture", "desktop_with_legs"]
    )

    desk_pos, desk_size = find_desktop_top_surface(desk_shape)

    # Generate blocks relative to the computed table surface
    blocks_shape = library_call(
        "scattered_blocks",
        table_center=desk_pos,
        table_top_z=desk_pos[2]
    )

    return concat_shapes(desk_shape, blocks_shape)


@register()
def root_scene() -> Shape:
    """
    Root scene entry point.
    """
    return table_with_scattered_blocks()
