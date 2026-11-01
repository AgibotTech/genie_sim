# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: table_with_two_building_blocks
description: Two colored building blocks placed on a table at random positions for swap task
"""


@register()
def table_with_two_building_blocks() -> Shape:
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "wood", "desktop_with_legs"],
    )
    table_info = get_object_info(table_shape)
    table_top_z = table_info["max"][2]
    table_center = table_info["center"]

    x_min = table_center[0] - 0.11
    x_max = table_center[0] + 0.146
    y_min = table_center[1] - 0.22
    y_max = table_center[1] + 0.22

    block_candidates = [
        "benchmark_building_blocks_000", "benchmark_building_blocks_001",
        "benchmark_building_blocks_002", "benchmark_building_blocks_003",
        "benchmark_building_blocks_004", "benchmark_building_blocks_005",
        "benchmark_building_blocks_006", "benchmark_building_blocks_007",
        "benchmark_building_blocks_008", "benchmark_building_blocks_009",
        "benchmark_building_blocks_010", "benchmark_building_blocks_011",
        "benchmark_building_blocks_012", "benchmark_building_blocks_013",
        "benchmark_building_blocks_014", "benchmark_building_blocks_015",
        "benchmark_building_blocks_016", "benchmark_building_blocks_017",
        "benchmark_building_blocks_018", "benchmark_building_blocks_019",
        "benchmark_building_blocks_020", "benchmark_building_blocks_021",
        "benchmark_building_blocks_022", "benchmark_building_blocks_023",
        "benchmark_building_blocks_024", "benchmark_building_blocks_025",
        "benchmark_building_blocks_026", "benchmark_building_blocks_027",
        "benchmark_building_blocks_028", "benchmark_building_blocks_029",
        "benchmark_building_blocks_030", "benchmark_building_blocks_031",
        "benchmark_building_blocks_032", "benchmark_building_blocks_033",
    ]
    selected_blocks = random.sample(block_candidates, 2)

    block_a_shape = library_call(
        "usd",
        oid=selected_blocks[0],
        keywords=["block_A", "building_block", "colored", "on_desk", "left"],
    )
    block_a_info = get_object_info(block_a_shape)
    half_a = max(block_a_info["size"][0], block_a_info["size"][1]) / 2

    block_b_shape = library_call(
        "usd",
        oid=selected_blocks[1],
        keywords=["block_B", "building_block", "colored", "on_desk", "right"],
    )
    block_b_info = get_object_info(block_b_shape)
    half_b = max(block_b_info["size"][0], block_b_info["size"][1]) / 2

    min_dist = half_a + half_b + 0.08

    ax = random.uniform(x_min + half_a, x_max - half_a)
    ay = random.uniform(table_center[1] + 0.03, y_max - half_a)

    for _ in range(50):
        bx = random.uniform(x_min + half_b, x_max - half_b)
        by = random.uniform(y_min + half_b, table_center[1] - 0.03)
        dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        if dist >= min_dist:
            break

    angle_a = random.uniform(0, 2 * math.pi)
    block_a_rotated = transform_shape(
        block_a_shape,
        rotation_matrix(angle_a, direction=(0, 0, 1), point=(0, 0, 0)),
    )
    block_a_final = transform_shape(block_a_rotated, translation_matrix([ax, ay, table_top_z]))

    angle_b = random.uniform(0, 2 * math.pi)
    block_b_rotated = transform_shape(
        block_b_shape,
        rotation_matrix(angle_b, direction=(0, 0, 1), point=(0, 0, 0)),
    )
    block_b_final = transform_shape(block_b_rotated, translation_matrix([bx, by, table_top_z]))

    return concat_shapes(table_shape, block_a_final, block_b_final)


@register()
def root_scene() -> Shape:
    return library_call("table_with_two_building_blocks")
