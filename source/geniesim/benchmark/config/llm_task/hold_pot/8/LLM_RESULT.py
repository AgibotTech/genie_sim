# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np
import math


def place_random() -> Shape:
    # Find the usable surface area on top of the desk.
    workspace_center = np.array([0.74, -1.0, 0.95])

    # Define the blocks to be placed, with their asset IDs and descriptive keywords.
    items_to_place = [
        {
            "oid": "benchmark_pot_000",
            "keywords": ["dual"],
        },
        {
            "oid": "benchmark_pot_001",
            "keywords": ["dual"],
        },
        # {
        #     "oid": "benchmark_pot_002",
        #     "keywords": ["dual"],
        # },
        {
            "oid": "benchmark_pot_007",
            "keywords": ["dual"],
        },
        {
            "oid": "benchmark_pot_008",
            "keywords": ["dual"],
        },
        {
            "oid": "benchmark_pot_009",
            "keywords": ["dual"],
        },
        {
            "oid": "benchmark_pot_010",
            "keywords": ["dual"],
        },
    ]

    # Define the valid placement range on the table, leaving a small margin to avoid collision or falling off.
    margin = 0.00
    range_x_min = workspace_center[0] - 0.01 + margin
    range_x_max = workspace_center[0] + 0.01 - margin
    range_y_min = workspace_center[1] - 0.01 + margin
    range_y_max = workspace_center[1] + 0.01 - margin

    # Generate a random position on the table surface.
    rand_x = np.random.uniform(range_x_min, range_x_max)
    rand_y = np.random.uniform(range_y_min, range_y_max)

    # The z position is the top of the table.
    placement_pos = (rand_x, rand_y, workspace_center[2] + 0.01)

    # choose one item
    item_chosen = random.choice(items_to_place)

    # Load the block shape.
    block_shape = library_call("usd", oid=item_chosen["oid"], keywords=item_chosen["keywords"])

    # First, translate the block to the random position on the table.
    translated_block = transform_shape(block_shape, translation_matrix(placement_pos))

    # Then, apply a random rotation around its new center's Z-axis.
    block_center = compute_shape_center(translated_block)
    random_angle = np.random.uniform(0, math.pi / 36)
    final_block = transform_shape(
        translated_block,
        rotation_matrix(angle=math.pi / 2 + random_angle, direction=(0, 0, 1), point=block_center),
    )

    return concat_shapes(final_block)


@register()
def kitchen_task_put_pot_onto_desk() -> Shape:

    placed_blocks = []

    placed_blocks.append(place_random())

    return concat_shapes(*placed_blocks)


@register()
def root_scene() -> Shape:
    """
    The root function that generates the entire scene.
    """
    return kitchen_task_put_pot_onto_desk()
