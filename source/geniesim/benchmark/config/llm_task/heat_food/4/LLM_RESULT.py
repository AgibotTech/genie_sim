# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np
import math


def get_table_surface_z():
    # Kitchen table height
    return 0.85


def get_table_surface(desk_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the desktop's top surface center and its size.

    Args:
        desk_shape (Shape): The shape of the desk object.

    Returns:
        Tuple[P, P]: A tuple containing the desktop's top surface center position and its size (width, depth).
    """
    # Get information about the whole desk object in world coordinates
    desk_info = get_object_info(desk_shape)

    # Get the local information of the 'desktop' subpart
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # The desktop's center in world coordinates is the desk's center plus the subpart's local center offset
    desktop_center_world = desk_info["center"] + desktop_subpart_info["center"]

    # The top surface z-coordinate is the desk's center z plus the subpart's max z offset
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    # The final position for placing objects is the center of the top surface
    desktop_top_surface_pos = np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z])

    # The size of the desktop surface (width, depth)
    desktop_size = desktop_subpart_info["size"]

    return desktop_top_surface_pos, desktop_size


@register()
def place_microwave_oven() -> Shape:
    # Define the valid placement
    placement_pos = np.array([0.94476, 0.53471, get_table_surface_z()])
    block_shape = library_call(
        "usd",
        oid="benchmark_microwave_oven_003",
        keywords=["microwave_oven"],
    )
    translated_block = transform_shape(block_shape, translation_matrix(placement_pos))
    # microwave_oven_block_r = transform_shape(
    #     microwave_oven_block_t,
    #     rotation_matrix(
    #         angle=math.pi, direction=(0, 0, 1), point=compute_shape_center(microwave_oven_block_t)
    #     ),
    # )

    block_center = compute_shape_center(translated_block)
    final_block = transform_shape(
        translated_block,
        rotation_matrix(angle=math.pi, direction=(0, 0, 1), point=block_center),
    )

    return final_block


@register()
def place_random() -> Shape:
    # Find the usable surface area on top of the desk.
    workspace_center = np.array([0.91915, 0.06544, get_table_surface_z()])

    # Define the blocks to be placed, with their asset IDs and descriptive keywords.
    foods_to_place = [
        {
            "oid": "benchmark_food_008",
            "keywords": ["food", "right"],
        },
        {
            "oid": "benchmark_food_013",
            "keywords": ["food", "right"],
        },
        {
            "oid": "benchmark_food_019",
            "keywords": ["food", "right"],
        },
        {
            "oid": "benchmark_food_020",
            "keywords": ["food", "right"],
        },
        {
            "oid": "benchmark_food_022",
            "keywords": ["food", "right"],
        },
    ]

    # Define the valid placement range on the table, leaving a small margin to avoid collision or falling off.
    margin = 0.05
    range_x_min = workspace_center[0] - 0.1 + margin
    range_x_max = workspace_center[0] + 0.1 - margin
    range_y_min = workspace_center[1] - 0.14 + margin
    range_y_max = workspace_center[1] + 0.14 - margin

    # Generate a random position on the table surface.
    rand_x = np.random.uniform(range_x_min, range_x_max)
    rand_y = np.random.uniform(range_y_min, range_y_max)

    # The z position is the top of the table.
    placement_pos0 = (rand_x, rand_y, workspace_center[2])
    placement_pos = (rand_x, rand_y, workspace_center[2] + 0.03)

    # choose one food
    food_chosen = random.choice(foods_to_place)

    # Load the block shape.
    block_shape0 = library_call("usd", oid="benchmark_cushion_000", keywords=[""])
    translated_block0 = transform_shape(block_shape0, translation_matrix(placement_pos0))

    block_shape = library_call("usd", oid=food_chosen["oid"], keywords=food_chosen["keywords"])

    # First, translate the block to the random position on the table.
    translated_block = transform_shape(block_shape, translation_matrix(placement_pos))

    # Then, apply a random rotation around its new center's Z-axis.
    block_center = compute_shape_center(translated_block)
    random_angle = np.random.uniform(0, 2 * math.pi)
    final_block = transform_shape(
        translated_block,
        rotation_matrix(angle=random_angle, direction=(0, 0, 1), point=block_center),
    )

    return concat_shapes(translated_block0, final_block)


@register()
def kitchen_task_put_food_into_microwave_oven() -> Shape:

    placed_blocks = []

    placed_blocks.append(place_microwave_oven())

    placed_blocks.append(place_random())

    return concat_shapes(*placed_blocks)


@register()
def root_scene() -> Shape:
    """
    The root function that generates the entire scene.
    """
    return kitchen_task_put_food_into_microwave_oven()
