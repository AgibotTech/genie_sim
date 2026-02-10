# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: tilted_beverage_bottle_scene
description: Randomly place one of three beverage bottles at specific positions with 30-degree tilt
"""


@register()
def place_tilted_beverage_bottle() -> Shape:
    """
    Randomly select one of three positions and place the corresponding beverage bottle with 30-degree tilt.
    Position 1: (-0.3258698616779385, -0.961097970731979, 1.1171925071287725) -> genie_beverage_bottle_007
    Position 2: (-0.3258698616779385, -0.6978373934951332, 1.1171925071287725) -> genie_beverage_bottle_008
    Position 3: (-0.3258698616779385, -0.46964291938826225, 1.1171925071287725) -> genie_beverage_bottle_009
    """
    # Define three position options with corresponding object IDs
    position_options = [
        ((-0.3258698616779385, -0.961097970731979, 1.1171925071287725), "genie_beverage_bottle_007"),
        ((-0.3258698616779385, -0.6978373934951332, 1.1171925071287725), "genie_beverage_bottle_008"),
        ((-0.3258698616779385, -0.46964291938826225, 1.1171925071287725), "genie_beverage_bottle_009"),
    ]

    # Randomly select one position and object
    selected_index = np.random.choice(len(position_options))
    selected_position, selected_oid = position_options[selected_index]

    # Create beverage bottle object
    bottle_shape = library_call(
        "usd",
        oid=selected_oid,
        keywords=["beverage_bottle", "bottle", "drink", f"bottle_{selected_index + 7}"],
    )

    # Get object center for rotation
    bottle_center = compute_shape_center(bottle_shape)

    # Apply 30-degree tilt around X-axis (forward tilt)
    tilt_angle = math.pi / 6  # 30 degrees in radians
    bottle_shape = transform_shape(
        bottle_shape, rotation_matrix(angle=tilt_angle, direction=(1, 0, 0), point=bottle_center)
    )

    # Get object info after rotation for proper placement
    object_info = get_object_info(bottle_shape)
    object_center = object_info["center"]

    # Translate to selected position
    final_shape = transform_shape(bottle_shape, translation_matrix(np.array(selected_position) - object_center))

    return final_shape


@register()
def root_scene() -> Shape:
    return place_tilted_beverage_bottle()
