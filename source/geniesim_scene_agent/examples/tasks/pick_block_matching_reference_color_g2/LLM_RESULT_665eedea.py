# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: pick_block_matching_reference_color_g2
description: A reference block is placed ON a coaster (visually distinct as the reference).
Several candidate blocks of different colors are scattered on the table nearby.
The robot must pick the block that matches the reference block's color (not the one on the coaster).
"""

COLOR_MAP = {
    "red": "benchmark_building_blocks_073",
    "blue": "benchmark_building_blocks_011",
    "yellow": "benchmark_building_blocks_085",
    "purple": "benchmark_building_blocks_089",
}

COASTER_POOL = [f"benchmark_coaster_{i:03d}" for i in range(26)]


@register()
def pick_block_matching_reference_color_g2() -> Shape:
    table_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )
    table_info = get_object_info(table_shape)
    table_top_z = table_info["max"][2]
    table_center = table_info["center"]

    x_min, x_max = -0.11, 0.146
    y_min, y_max = -0.22, 0.22

    ref_color = random.choice(list(COLOR_MAP.keys()))
    ref_oid = COLOR_MAP[ref_color]

    other_colors = [c for c in COLOR_MAP.keys() if c != ref_color]
    candidate_colors = random.sample(other_colors, 2)
    candidate_colors.append(ref_color)
    random.shuffle(candidate_colors)

    coaster_oid = random.choice(COASTER_POOL)

    coaster_x = random.uniform(x_min + 0.03, x_max - 0.03)
    coaster_y = random.uniform(y_min + 0.05, y_max - 0.05)
    coaster_pos_tag = "left" if coaster_y > table_center[1] else "right"

    coaster_shape = library_call(
        "usd",
        oid=coaster_oid,
        keywords=["coaster", "round", "reference_platform", coaster_pos_tag],
    )
    coaster_info = get_object_info(coaster_shape)
    coaster_height = coaster_info["size"][2]

    coaster_angle = random.uniform(0, 2 * math.pi)
    coaster_shape = transform_shape(
        coaster_shape,
        rotation_matrix(coaster_angle, direction=(0, 0, 1), point=(0, 0, 0)),
    )
    coaster_shape = transform_shape(coaster_shape, translation_matrix([coaster_x, coaster_y, table_top_z]))

    ref_z = table_top_z + coaster_height
    ref_shape = library_call(
        "usd",
        oid=ref_oid,
        keywords=["reference_block", ref_color, "building_block", "on_coaster", "fixed"],
    )
    ref_shape = transform_shape(ref_shape, translation_matrix([coaster_x, coaster_y, ref_z]))

    shapes = [table_shape, coaster_shape, ref_shape]
    positions = [(coaster_x, coaster_y)]

    for i, color in enumerate(candidate_colors):
        oid = COLOR_MAP[color]
        is_match = (color == ref_color)

        placed = False
        for _ in range(100):
            x = random.uniform(x_min + 0.02, x_max - 0.02)
            y = random.uniform(y_min + 0.03, y_max - 0.03)
            ok = True
            for px, py in positions:
                if ((x - px) ** 2 + (y - py) ** 2) ** 0.5 < 0.09:
                    ok = False
                    break
            if ok:
                positions.append((x, y))
                placed = True
                break

        if not placed:
            x = x_min + 0.05 + i * 0.08
            y = (y_max - 0.05) if i % 2 == 0 else (y_min + 0.05)
            positions.append((x, y))

        x, y = positions[-1]
        pos_tag = "left" if y > table_center[1] else "right"

        kws = ["candidate_block", color, "building_block", "scattered", "on_desk", pos_tag]
        if is_match:
            kws.append("matching")

        block_shape = library_call("usd", oid=oid, keywords=kws)
        block_shape = transform_shape(block_shape, translation_matrix([x, y, table_top_z]))
        angle = random.uniform(-math.pi / 6, math.pi / 6)
        center = compute_shape_center(block_shape)
        block_shape = transform_shape(block_shape, rotation_matrix(angle, direction=(0, 0, 1), point=center))
        shapes.append(block_shape)

    return concat_shapes(*shapes)


@register()
def root_scene() -> Shape:
    return pick_block_matching_reference_color_g2()
