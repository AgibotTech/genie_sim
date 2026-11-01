# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: table_with_cosmetics
description: A table with 3-5 cosmetics randomly placed on the surface, each rotated randomly around Z axis.
"""

COSMETICS_CANDIDATES = [
    "benchmark_cosmetics_000", "benchmark_cosmetics_001", "benchmark_cosmetics_002",
    "benchmark_cosmetics_003", "benchmark_cosmetics_004", "benchmark_cosmetics_005",
    "benchmark_cosmetics_006", "benchmark_cosmetics_007", "benchmark_cosmetics_008",
    "benchmark_cosmetics_009",
]


def get_desktop_info(table_shape: Shape) -> Tuple[P, P]:
    table_info = get_object_info(table_shape)
    desktop_sub = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_center = table_info["center"] + desktop_sub["center"]
    desktop_top_z = table_info["center"][2] + desktop_sub["xyz_max"][2]
    return np.array([desktop_center[0], desktop_center[1], desktop_top_z]), desktop_sub["size"]


@register()
def place_cosmetic(oid: str, idx: int, table_top_center: P, existing_positions: List[P]) -> Shape:
    side_tag = "left" if idx % 2 == 0 else "right"
    shape = library_call("usd", oid=oid, keywords=[f"cosmetic_{idx}", "cosmetics", "on_desk", side_tag])
    obj_info = get_object_info(shape)
    half_x = obj_info["size"][0] / 2
    half_y = obj_info["size"][1] / 2
    height_offset = -compute_shape_min(shape)[2]

    rot_angle = random.uniform(0, 2 * np.pi)
    rotated = transform_shape(shape, rotation_matrix(rot_angle, (0, 0, 1), (0, 0, 0)))

    x_range = (-0.1, 0.14)
    y_range = (-0.2, 0.2)
    margin = 0.03

    for _ in range(100):
        cx = random.uniform(x_range[0], x_range[1])
        cy = random.uniform(y_range[0], y_range[1])
        collision = False
        for px, py, phx, phy in existing_positions:
            if abs(cx - px) < (half_x + phx + margin) and abs(cy - py) < (half_y + phy + margin):
                collision = True
                break
        if not collision:
            final = transform_shape(rotated, translation_matrix([
                table_top_center[0] + cx,
                table_top_center[1] + cy,
                table_top_center[2] + height_offset
            ]))
            return final

    final = transform_shape(rotated, translation_matrix([
        table_top_center[0] + cx,
        table_top_center[1] + cy,
        table_top_center[2] + height_offset
    ]))
    return final


@register()
def table_with_cosmetics() -> Shape:
    table_shape = library_call("usd", oid="table_000", keywords=["table", "furniture", "workspace"])
    table_top_center, _ = get_desktop_info(table_shape)

    num_cosmetics = random.randint(3, 5)
    selected = random.sample(COSMETICS_CANDIDATES, num_cosmetics)

    placed_positions = []
    cosmetic_shapes = []
    for idx, oid in enumerate(selected):
        c_shape = library_call("place_cosmetic", oid=oid, idx=idx, table_top_center=table_top_center, existing_positions=placed_positions)
        c_info = get_object_info(c_shape)
        half_x = c_info["size"][0] / 2
        half_y = c_info["size"][1] / 2
        placed_positions.append((c_info["center"][0] - table_top_center[0], c_info["center"][1] - table_top_center[1], half_x, half_y))
        cosmetic_shapes.append(c_shape)

    return concat_shapes(table_shape, *cosmetic_shapes)


@register()
def root_scene() -> Shape:
    return library_call("table_with_cosmetics")
