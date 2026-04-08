# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np
import math


@register()
def place_trash_can() -> Shape:
    # Define the valid placement
    placement_pos = np.array([4.08154, -2.48619, 0.60445])
    block_shape = library_call(
        "usd",
        oid="benchmark_trash_can_005",
        keywords=["trash_can"],
    )
    translated_block = transform_shape(block_shape, translation_matrix(placement_pos))

    block_center = compute_shape_center(translated_block)
    final_block = transform_shape(
        translated_block,
        rotation_matrix(angle=math.pi, direction=(0, 0, 1), point=block_center),
    )

    return final_block


@register()
def home_task_open_the_door() -> Shape:

    placed_blocks = []

    placed_blocks.append(place_trash_can())

    return concat_shapes(*placed_blocks)


@register()
def root_scene() -> Shape:
    """
    The root function that generates the entire scene.
    """
    return home_task_open_the_door()
