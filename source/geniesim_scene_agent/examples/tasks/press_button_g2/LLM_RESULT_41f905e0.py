# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np

"""
scene_name: random_button_on_table
description: A table (table_000) with a randomly selected button placed on its top surface
             within the specified bounds x(-0.2, 0.2), y(-0.2, 0.2).
"""

# List of available button object IDs derived from the asset library
AVAILABLE_BUTTONS = [
    "benchmark_button_009",
    "benchmark_button_011",
    "benchmark_button_013",
    "benchmark_button_017",
    "benchmark_button_018",
    "benchmark_button_022",
    "benchmark_button_023",
    "benchmark_button_026",
    "benchmark_button_027",
    "benchmark_button_028"
]

@register()
def random_button() -> Shape:
    """
    Randomly selects one button from the available list and returns its shape.
    """
    # Select a random ID
    selected_oid = np.random.choice(AVAILABLE_BUTTONS)

    return library_call(
        "usd",
        oid=selected_oid,
        keywords=[selected_oid, "button", "control", "randomly_selected"]
    )

@register()
def scene_setup() -> Shape:
    """
    Sets up the table and places the random button on top.
    """
    # 1. Load the table
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["table", "white", "furniture", "desktop"]
    )

    # 2. Get table info to calculate placement
    table_info = get_object_info(table_shape)
    table_center = table_info["center"]

    # Get desktop subpart info to determine the top surface height
    # 'desktop' is the standard subpart name for table_000
    desktop_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # 3. Calculate the target Z (top surface of the desktop)
    # table center Z + desktop half-height
    target_z = table_center[2] + desktop_info["xyz_max"][2]

    # 4. Calculate random X and Y within the specified range relative to table center
    # User range: x(-0.2, 0.2), y(-0.2, 0.2)
    random_offset_x = np.random.uniform(-0.2, 0.2)
    random_offset_y = np.random.uniform(-0.2, 0.2)

    target_x = table_center[0] + random_offset_x
    target_y = table_center[1] + random_offset_y

    # 5. Create and place the button
    btn_shape = library_call("random_button")

    # Transform button to the calculated position
    placed_btn = transform_shape(
        btn_shape,
        translation_matrix((target_x, target_y, target_z))
    )

    # 6. Combine shapes
    return concat_shapes(table_shape, placed_btn)

@register()
def root_scene() -> Shape:
    return scene_setup()
