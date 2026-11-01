# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np
import random

"""
scene_name: find_hidden_block_in_three_drawers_set_3_g2
description: Three storage boxes placed side-by-side on a table, all rotated ~180 deg
around the world Z axis with small random variation. One randomly chosen box hides a
single building block whose center sits 0.04844 m above that box center; the chosen
box carries a "target" tag in keywords.
"""

BUILDING_BLOCK_CANDIDATES = [
    "benchmark_building_blocks_073",
    "benchmark_building_blocks_085",
    "benchmark_building_blocks_089",
]

BLOCK_HEIGHT_OFFSET = 0.04844


def _make_storage_box(position_tag: str, role: str) -> Shape:
    keywords = [
        f"{role}_drawer_{position_tag}",
        "storage_box",
        "container",
        "drawer",
        position_tag,
    ]
    if role == "target":
        keywords.append("target")
        keywords.append("contains_block")
    box = library_call(
        "usd",
        oid="benchmark_storage_box_019",
        keywords=keywords,
    )
    yaw = random.uniform(np.deg2rad(175.0), np.deg2rad(185.0))
    box_center_local = compute_shape_center(box)
    return transform_shape(
        box, rotation_matrix(yaw, (0, 0, 1), tuple(box_center_local))
    )


def _put_box_at(box_shape: Shape, x: float, y: float, table_top_z: float) -> Shape:
    info = get_object_info(box_shape)
    box_half_h = info["size"][2] / 2.0
    cur_center = info["center"]
    delta = (
        x - cur_center[0],
        y - cur_center[1],
        (table_top_z + box_half_h) - cur_center[2],
    )
    return transform_shape(box_shape, translation_matrix(delta))


def _make_block_inside_box(box_shape: Shape) -> Shape:
    block_oid = random.choice(BUILDING_BLOCK_CANDIDATES)
    block = library_call(
        "usd",
        oid=block_oid,
        keywords=[
            "hidden_block",
            "building_block",
            "cube",
            "target_object",
            "inside_drawer",
        ],
    )
    block_center = compute_shape_center(block)
    yaw = np.random.uniform(-np.pi, np.pi)
    block = transform_shape(
        block, rotation_matrix(yaw, (0, 0, 1), tuple(block_center))
    )

    box_info = get_object_info(box_shape)
    box_center = box_info["center"]
    target_xyz = (
        box_center[0],
        box_center[1],
        box_center[2] + BLOCK_HEIGHT_OFFSET,
    )

    cur_center = compute_shape_center(block)
    delta = (
        target_xyz[0] - cur_center[0],
        target_xyz[1] - cur_center[1],
        target_xyz[2] - cur_center[2],
    )
    return transform_shape(block, translation_matrix(delta))


@register()
def three_storage_boxes_with_hidden_block() -> Shape:
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "wood", "rectangular", "furniture"],
    )
    table_info = get_object_info(table_shape)
    table_top_z = table_info["max"][2]
    table_center = table_info["center"]

    # Probe a non-rotated storage box to learn its world-space Y footprint;
    # under a ~180 deg yaw the projected Y extent stays effectively unchanged.
    probe = library_call(
        "usd",
        oid="benchmark_storage_box_019",
        keywords=["probe_box", "storage_box", "auxiliary"],
    )
    box_y_extent = compute_shape_sizes(probe)[1]

    # Middle drawer offsets are given relative to the table center.
    mid_x = table_center[0] + random.uniform(0.03, 0.1)
    mid_y = table_center[1] + random.uniform(-0.02, 0.02)

    # Edge-to-edge gap > 0.02 m -> center spacing = box_y_extent + gap.
    gap_left = random.uniform(0.022, 0.04)
    gap_right = random.uniform(0.022, 0.04)
    left_x = mid_x + random.uniform(-0.01, 0.01)
    right_x = mid_x + random.uniform(-0.01, 0.01)
    left_y = mid_y + (box_y_extent + gap_left)
    right_y = mid_y - (box_y_extent + gap_right)

    # Weighted target choice: right : middle : left = 3 : 2 : 1.
    # layout indices: 0=left, 1=middle, 2=right.
    target_idx = random.choices([0, 1, 2], weights=[1, 2, 3], k=1)[0]
    layout = [
        ("left", left_x, left_y),
        ("middle", mid_x, mid_y),
        ("right", right_x, right_y),
    ]

    shapes = [table_shape]
    for idx, (tag, x, y) in enumerate(layout):
        role = "target" if idx == target_idx else "regular"
        box = _make_storage_box(position_tag=tag, role=role)
        positioned = _put_box_at(box, x, y, table_top_z)
        shapes.append(positioned)
        if idx == target_idx:
            block = _make_block_inside_box(positioned)
            shapes.append(block)

    return concat_shapes(*shapes)


@register()
def root_scene() -> Shape:
    return library_call("three_storage_boxes_with_hidden_block")
