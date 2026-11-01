# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np

"""
scene_name: table_with_gearbox_and_mounted_part
description: A table with a gearbox and a gear/bearing already mounted on a specific cylinder.
             The part is placed according to specific rules: gear_000 and bearing_001 on right/back cylinders,
             others on the small cylinder. Coordinates are relative to the gearbox.
"""


@register()
def table_with_gearbox_and_mounted_part() -> Shape:
    # 1. Load the table
    table_shape = library_call("usd", oid="table_000", keywords=["main_table", "furniture", "table_000"])
    table_info = get_object_info(table_shape)

    # Calculate desktop center and height
    # The table center (x, y) is table_info["center"][:2]
    # The top surface height is table_info["max"][2]
    desktop_center_x = table_info["center"][0]
    desktop_center_y = table_info["center"][1]
    desktop_z = table_info["max"][2] + 0.1

    # 2. Load and Place Gearbox
    # Gearbox parameters: Random rotation within 5 degrees around Z
    g_rot_deg = np.random.uniform(0, 360)
    g_rot_rad = np.radians(g_rot_deg)

    # Gearbox position: Centered on the desktop
    g_x = np.random.uniform(-0.15, 0.1)
    g_y = np.random.uniform(-0.15, 0.15)
    g_pos = np.array([desktop_center_x + g_x, desktop_center_y + g_y, desktop_z])

    # Create gearbox transformation matrices
    # Rotation around Z axis at origin
    g_rot_mat = rotation_matrix(g_rot_rad, (0, 0, 1), (0, 0, 0))
    # Translation to desktop position
    g_trans_mat = translation_matrix(g_pos)
    # Combine: Rotation then Translation
    g_pose = g_trans_mat @ g_rot_mat

    # Load gearbox shape
    gearbox_shape = library_call(
        "usd",
        oid="benchmark_gearbox_000",
        keywords=["gearbox_center", "mechanical_part", "fixed"],
    )
    # Apply transformation
    gearbox_shape = transform_shape(gearbox_shape, g_pose)

    # 3. Select Part and Determine Target Cylinder
    # Define local coordinates for cylinders (relative to gearbox center)
    # Based on user provided data
    cyl_local_coords = {
        "small_cylinder": np.array([0.049, -0.029, 0.03872 - 0.02]),
        "right_cylinder": np.array([0.019, 0.06, 0.11479 - 0.06]),
        "back_cylinder": np.array([-0.061, -0.019, 0.11479 - 0.06]),
    }

    # List of available parts
    parts_pool = [
        "benchmark_gear_000",  # Large Gear
        "benchmark_bearing_001",  # Small Bearing
        "benchmark_gear_001",  # Small Gear
        "benchmark_bearing_000",  # Large Bearing
    ]

    # Randomly select one part to place in the scene
    part_oid = np.random.choice(parts_pool)

    # Determine target cylinder based on user rules:
    # "gear_000 and bearing_001 on right or back, others on small"
    target_cylinders = []
    if part_oid in ["benchmark_gear_000", "benchmark_bearing_001"]:
        target_cylinders = ["right_cylinder", "back_cylinder"]
    else:
        target_cylinders = ["small_cylinder"]

    target_cyl_name = np.random.choice(target_cylinders)
    target_local_pos = cyl_local_coords[target_cyl_name]

    # 4. Calculate Part World Position and Rotation
    # Calculate world position: P_world = T_gearbox * P_local
    # We need to treat P_local as a homogeneous point [x, y, z, 1]
    p_local_h = np.append(target_local_pos, 1.0)
    p_world_h = g_pose @ p_local_h
    part_pos = p_world_h[:3]

    # Calculate Part Rotation
    # The part is mounted on the cylinder.
    # The cylinder axis is rotated by g_rot_rad relative to world Z.
    # To align the part's hole with the cylinder, the part must rotate by g_rot_rad.
    # Additionally, the user allows "arbitrary random rotation around Z".
    # This means the part can spin around the cylinder axis.
    # Total Rotation = Gearbox_Rotation + Random_Spin
    p_spin_rad = np.random.uniform(0, 2 * np.pi)
    p_total_rot_rad = g_rot_rad + p_spin_rad

    # 5. Load and Place Part
    part_shape = library_call(
        "usd",
        oid=part_oid,
        keywords=[f"mounted_{part_oid}", "mechanical", "on_cylinder", target_cyl_name],
    )

    # Create part transformation
    # Rotate around Z, then translate to calculated world position
    part_rot_mat = rotation_matrix(p_total_rot_rad, (0, 0, 1), (0, 0, 0))
    part_trans_mat = translation_matrix(part_pos)
    part_pose = part_trans_mat @ part_rot_mat

    part_shape = transform_shape(part_shape, part_pose)

    # 6. Combine shapes
    return concat_shapes(table_shape, gearbox_shape, part_shape)


@register()
def root_scene() -> Shape:
    return library_call("table_with_gearbox_and_mounted_part")
