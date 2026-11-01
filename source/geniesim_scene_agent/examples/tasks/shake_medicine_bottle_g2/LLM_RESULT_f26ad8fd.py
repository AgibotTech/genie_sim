# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: table_with_medicine_bottle
description: A medicine bottle randomly placed on the table with random rotation
"""


@register()
def table_with_medicine_bottle() -> Shape:
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

    bottle_shape = library_call(
        "usd",
        oid="benchmark_medicine_bottle_000",
        keywords=["medicine_bottle", "bottle", "medical", "cylindrical", "on_desk"],
    )
    bottle_info = get_object_info(bottle_shape)
    half_x = bottle_info["size"][0] / 2
    half_y = bottle_info["size"][1] / 2

    pos_x = random.uniform(x_min + half_x, x_max - half_x)
    pos_y = random.uniform(y_min + half_y, y_max - half_y)

    position_tag = "left" if pos_y > table_center[1] else "right"
    bottle_shape = library_call(
        "usd",
        oid="benchmark_medicine_bottle_000",
        keywords=["medicine_bottle", "bottle", "medical", "cylindrical", "on_desk", position_tag],
    )

    random_angle = random.uniform(0, 2 * math.pi)
    rotated = transform_shape(
        bottle_shape,
        rotation_matrix(random_angle, direction=(0, 0, 1), point=(0, 0, 0)),
    )
    final = transform_shape(rotated, translation_matrix([pos_x, pos_y, table_top_z]))

    return concat_shapes(table_shape, final)


@register()
def root_scene() -> Shape:
    return library_call("table_with_medicine_bottle")
