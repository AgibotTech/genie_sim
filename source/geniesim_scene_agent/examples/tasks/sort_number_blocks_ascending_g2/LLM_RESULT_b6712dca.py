# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
import numpy as np

"""
scene_name: table_with_digit_blocks
description: A white rectangular table with three different digit-shaped building blocks placed randomly on its surface within x ∈ (-0.1, 0.15), y ∈ (-0.3, 0.3), ensuring no collisions and at least 5 cm spacing between blocks. Each block is randomly rotated around the vertical (z) axis.
"""

# Mapping from digit to asset ID
DIGIT_TO_ASSET = {
    0: "benchmark_building_blocks_034",
    1: "benchmark_building_blocks_035",
    2: "benchmark_building_blocks_036",
    3: "benchmark_building_blocks_037",
    4: "benchmark_building_blocks_038",
    5: "benchmark_building_blocks_039",
    6: "benchmark_building_blocks_040",
    7: "benchmark_building_blocks_041",
    8: "benchmark_building_blocks_042",
    9: "benchmark_building_blocks_043",
}


@register()
def digit_block(digit: int) -> Shape:
    """Creates a single digit-shaped building block."""
    asset_id = DIGIT_TO_ASSET[digit]
    return library_call(
        "usd", oid=asset_id, keywords=[f"digit_{digit}", "plastic", "small", "building_block", "educational_toy"]
    )


@register()
def place_three_digit_blocks_on_table(table_shape: Shape) -> Shape:
    """
    Places three non-colliding digit blocks on the table within the specified region.
    Ensures at least 5 cm (0.05 m) spacing between any two blocks.
    Each block is randomly rotated around the z-axis after placement.
    """
    # Get table top surface info
    table_info = get_object_info(table_shape)
    desktop_subpart = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Compute world position of desktop top surface center
    desktop_center_world = table_info["center"] + desktop_subpart["center"]
    desktop_top_z = table_info["center"][2] + desktop_subpart["xyz_max"][2]
    desktop_pos = np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z])
    desktop_size = desktop_subpart["size"]

    # Define placement region relative to desktop center
    x_min, x_max = -0.2, -0.1
    y_min, y_max = -0.2, 0.2

    # Select three distinct digits
    selected_digits = random.sample(list(DIGIT_TO_ASSET.keys()), 3)

    placed_positions = []
    block_shapes = []

    for i, digit in enumerate(selected_digits):
        block_shape = library_call("digit_block", digit=digit)
        block_info = get_object_info(block_shape)
        block_size = block_info["size"]
        block_radius = max(block_size[0], block_size[1]) / 2.0  # conservative estimate for collision

        max_attempts = 100
        for attempt in range(max_attempts):
            # Random position within region
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            pos = np.array([x, y, desktop_top_z])

            # Check distance to already placed blocks
            valid = True
            for placed_pos in placed_positions:
                dist = np.linalg.norm(pos[:2] - placed_pos[:2])
                if dist < 0.05 + block_radius + 0.015:  # 5cm + both radii (approx 1.5cm each)
                    valid = False
                    break

            if valid:
                placed_positions.append(pos)

                # First translate to position
                block_translated = transform_shape(block_shape, translation_matrix(pos))
                # Then rotate around its own center (which is now at pos)
                rotation_angle = random.uniform(160.0 / 180 * np.pi, 200.0 / 180 * np.pi)  # Full random rotation
                block_rotated = transform_shape(
                    block_translated, rotation_matrix(angle=rotation_angle, direction=(0, 0, 1), point=pos)
                )
                block_shapes.append(block_rotated)
                break
        else:
            # Fallback: place at edge if failed (should not happen often)
            fallback_x = x_min + i * 0.1
            fallback_y = 0.0
            fallback_pos = np.array([fallback_x, fallback_y, desktop_top_z])
            placed_positions.append(fallback_pos)

            block_translated = transform_shape(block_shape, translation_matrix(fallback_pos))
            rotation_angle = random.uniform(0, 2 * np.pi)
            block_rotated = transform_shape(
                block_translated, rotation_matrix(angle=rotation_angle, direction=(0, 0, 1), point=fallback_pos)
            )
            block_shapes.append(block_rotated)

    return concat_shapes(table_shape, *block_shapes)


@register()
def table_with_digit_blocks() -> Shape:
    table_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "rectangular", "study_desk"]
    )
    return place_three_digit_blocks_on_table(table_shape)


@register()
def root_scene() -> Shape:
    return table_with_digit_blocks()
