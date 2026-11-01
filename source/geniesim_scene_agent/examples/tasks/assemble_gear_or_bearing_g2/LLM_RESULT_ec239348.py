# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np

"""
scene_name: table_with_gearbox_and_part
description: A table with a gearbox in the center and a gear or bearing placed nearby, avoiding collision.
"""


@register()
def table_scene() -> Shape:
    """
    Generates a scene with a table, a centered gearbox, and a randomly placed gear/bearing.
    """
    # 1. Load the table
    table_shape = library_call("usd", oid="table_000", keywords=["main_table", "furniture", "table_000"])
    table_info = get_object_info(table_shape)
    desktop_z = table_info["max"][2] + 0.1  # Top surface height

    # 2. Place the Gearbox (Center)
    # Random position in range x(-0.1, 0.05), y(-0.05, 0.05)
    g_x = np.random.uniform(-0.1, 0.05)
    g_y = np.random.uniform(-0.05, 0.05)

    # Random rotation within 5 degrees around Z
    g_rot_deg = np.random.uniform(-5.0 + 180.0, 5.0 + 180.0)

    gearbox_shape = library_call(
        "usd",
        oid="benchmark_gearbox_000",
        keywords=["gearbox_center", "mechanical_part", "placed_middle"],
    )
    # Transform Gearbox
    # First translate to position (z=0 for now, will lift later)
    gearbox_shape = transform_shape(gearbox_shape, translation_matrix((g_x, g_y, 0)))
    # Then rotate around its center
    g_center = compute_shape_center(gearbox_shape)
    gearbox_shape = transform_shape(
        gearbox_shape,
        rotation_matrix(np.radians(g_rot_deg), (0, 0, 1), g_center),
    )

    # 3. Place Side Part (Gear or Bearing)
    # Candidates list
    parts_list = [
        "benchmark_bearing_001",  # Small bearing
        "benchmark_bearing_000",  # Large bearing
        "benchmark_gear_001",  # Small gear
        "benchmark_gear_000",  # Large gear
    ]
    side_oid = np.random.choice(parts_list)

    # Find valid position satisfying distance constraint
    # Range x(-0.2, 0.15), y (-0.2, 0.2)
    # Distance >= 0.2 from gearbox center

    valid_pos_found = False
    s_x, s_y = 0.0, 0.0
    max_attempts = 100

    for _ in range(max_attempts):
        s_x = np.random.uniform(-0.2, 0.15)
        s_y = np.random.uniform(-0.2, 0.2)

        dist = np.sqrt((s_x - g_x) ** 2 + (s_y - g_y) ** 2)
        if dist >= 0.2:
            valid_pos_found = True
            break

    # If no position found after attempts, keep the last sampled one (though logic suggests it should find one quickly)
    # Random rotation arbitrary angle around Z
    s_rot_deg = np.random.uniform(0.0, 360.0)

    side_shape = library_call(
        "usd",
        oid=side_oid,
        keywords=[f"part_{side_oid}", "mechanical", "random_placement", "side_item"],
    )
    # Transform Side Part
    side_shape = transform_shape(side_shape, translation_matrix((s_x, s_y, 0)))
    s_center = compute_shape_center(side_shape)
    side_shape = transform_shape(
        side_shape,
        rotation_matrix(np.radians(s_rot_deg), (0, 0, 1), s_center),
    )

    # 4. Combine objects and lift to table height
    objects_on_table = concat_shapes(gearbox_shape, side_shape)
    # Lift objects so their bottom sits on desktop_z
    objects_on_table = transform_shape(objects_on_table, translation_matrix((0, 0, desktop_z)))

    # Final scene
    return concat_shapes(table_shape, objects_on_table)


@register()
def root_scene() -> Shape:
    return library_call("table_scene")
