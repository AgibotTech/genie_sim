# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
import numpy as np

"""
scene_name: table_with_digit_blocks
description: A table with 4-6 distinct digit blocks randomly placed on the surface.
"""

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
    asset_id = DIGIT_TO_ASSET[digit]
    return library_call("usd", oid=asset_id, keywords=[f"digit_{digit}", "plastic", "small", "building_block", "educational_toy"])


@register()
def place_digit_blocks_on_table(table_shape: Shape) -> Shape:
    table_info = get_object_info(table_shape)
    desktop_sub = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_top_z = table_info["center"][2] + desktop_sub["xyz_max"][2]

    x_min, x_max = -0.2, -0.1
    y_min, y_max = -0.2, 0.2

    num_blocks = random.randint(4, 6)
    safe_pool = [0, 1, 2, 3, 4, 5, 7, 8]
    selected_digits = random.sample(safe_pool, num_blocks)

    placed_positions = []
    block_shapes = []

    for i, digit in enumerate(selected_digits):
        block_shape = library_call("digit_block", digit=digit)
        block_info = get_object_info(block_shape)
        block_radius = max(block_info["size"][0], block_info["size"][1]) / 2.0

        for attempt in range(100):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            pos = np.array([x, y, desktop_top_z])

            valid = True
            for pp in placed_positions:
                if np.linalg.norm(pos[:2] - pp[:2]) < 0.05 + block_radius + 0.015:
                    valid = False
                    break

            if valid:
                placed_positions.append(pos)
                block_translated = transform_shape(block_shape, translation_matrix(pos))
                rot_angle = random.uniform(160.0/180*np.pi, 200.0/180*np.pi)
                block_rotated = transform_shape(block_translated, rotation_matrix(angle=rot_angle, direction=(0,0,1), point=pos))
                block_shapes.append(block_rotated)
                break
        else:
            fallback_pos = np.array([x_min + i*0.08, 0.0, desktop_top_z])
            placed_positions.append(fallback_pos)
            block_translated = transform_shape(block_shape, translation_matrix(fallback_pos))
            block_shapes.append(block_translated)

    return concat_shapes(table_shape, *block_shapes)


@register()
def table_with_digit_blocks() -> Shape:
    table_shape = library_call("usd", oid="table_000", keywords=["workspace_table", "table", "white", "rectangular", "study_desk"])
    return place_digit_blocks_on_table(table_shape)


@register()
def root_scene() -> Shape:
    return table_with_digit_blocks()
