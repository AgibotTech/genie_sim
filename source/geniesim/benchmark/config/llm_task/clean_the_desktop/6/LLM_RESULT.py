# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np
import math


def get_desktop_z():
    return 0.75


@register()
def place_trash_can() -> Shape:
    # Find the usable surface area on top of the desk.
    workspace_center = np.array([0.60457, -0.19779, get_desktop_z()])

    # Define the blocks to be placed, with their asset IDs and descriptive keywords.
    items_to_place = [
        {
            "oid": "benchmark_trash_can_005",
            "keywords": ["right"],
        },
    ]

    # Define the valid placement range on the table, leaving a small margin to avoid collision or falling off.
    margin = 0.00
    range_x_min = workspace_center[0] - 0.02 + margin
    range_x_max = workspace_center[0] + 0.02 - margin
    range_y_min = workspace_center[1] - 0.02 + margin
    range_y_max = workspace_center[1] + 0.02 - margin

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
    random_angle = np.random.uniform(0, math.pi / 60)
    final_block = transform_shape(
        translated_block,
        rotation_matrix(angle=random_angle, direction=(0, 0, 1), point=block_center),
    )

    return concat_shapes(final_block)


@register()
def place_trash() -> Shape:
    # Find the usable surface area on top of the desk.
    workspace_center = np.array([0.39264, 0.36097, get_desktop_z()])

    # Define the blocks to be placed, with their asset IDs and descriptive keywords.
    items_to_place = [
        {
            "oid": "iros_crumpled_paper_000",
            "keywords": ["right"],
        },
    ]

    # Define the valid placement range on the table, leaving a small margin to avoid collision or falling off.
    margin = 0.00
    range_x_min = workspace_center[0] - 0.02 + margin
    range_x_max = workspace_center[0] + 0.02 - margin
    range_y_min = workspace_center[1] - 0.02 + margin
    range_y_max = workspace_center[1] + 0.02 - margin

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
    random_angle = np.random.uniform(0, math.pi / 60)
    final_block = transform_shape(
        translated_block,
        rotation_matrix(angle=random_angle, direction=(0, 0, 1), point=block_center),
    )

    return concat_shapes(final_block)


@register()
def place_mouse() -> Shape:
    # Find the usable surface area on top of the desk.
    workspace_center = np.array([0.60778, 0.3914, get_desktop_z()])

    # Define the blocks to be placed, with their asset IDs and descriptive keywords.
    items_to_place = [
        {
            "oid": "benchmark_mouse_000",
            "keywords": ["right"],
        },
    ]

    # Define the valid placement range on the table, leaving a small margin to avoid collision or falling off.
    margin = 0.00
    range_x_min = workspace_center[0] - 0.02 + margin
    range_x_max = workspace_center[0] + 0.02 - margin
    range_y_min = workspace_center[1] - 0.02 + margin
    range_y_max = workspace_center[1] + 0.02 - margin

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
    random_angle = np.random.uniform(0, math.pi / 60)
    final_block = transform_shape(
        translated_block,
        rotation_matrix(angle=random_angle, direction=(0, 0, 1), point=block_center),
    )

    return concat_shapes(final_block)


@register()
def place_pencils() -> Shape:
    # Find the usable surface area on top of the desk.
    workspace_center = np.array([0.45416, -0.36965, get_desktop_z()])

    # Define the blocks to be placed, with their asset IDs and descriptive keywords.
    items_to_place = [
        {
            "oid": "sft_pen_001",
            "keywords": ["right"],
        },
    ]

    # Define the valid placement range on the table, leaving a small margin to avoid collision or falling off.
    margin = 0.00
    range_x_min = workspace_center[0] - 0.02 + margin
    range_x_max = workspace_center[0] + 0.02 - margin
    range_y_min = workspace_center[1] - 0.02 + margin
    range_y_max = workspace_center[1] + 0.02 - margin

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
    random_angle = np.random.uniform(0, math.pi / 60)
    final_block = transform_shape(
        translated_block,
        rotation_matrix(angle=random_angle, direction=(0, 0, 1), point=block_center),
    )

    return concat_shapes(final_block)


@register()
def place_laptops() -> Shape:
    # Find the usable surface area on top of the desk.
    workspace_center = np.array([0.52189, 0.11988, get_desktop_z()])

    # Define the blocks to be placed, with their asset IDs and descriptive keywords.
    items_to_place = [
        {
            "oid": "benchmark_laptop_000",
            "keywords": ["dual"],
        },
    ]

    # Define the valid placement range on the table, leaving a small margin to avoid collision or falling off.
    margin = 0.00
    range_x_min = workspace_center[0] - 0.02 + margin
    range_x_max = workspace_center[0] + 0.02 - margin
    range_y_min = workspace_center[1] - 0.02 + margin
    range_y_max = workspace_center[1] + 0.02 - margin

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
    random_angle = np.random.uniform(-math.pi / 180, math.pi / 180)
    final_block = transform_shape(
        translated_block,
        rotation_matrix(angle=math.pi + random_angle, direction=(0, 0, 1), point=block_center),
    )

    return concat_shapes(final_block)


@register()
def place_marker() -> Shape:
    # Find the usable surface area on top of the desk.
    workspace_center = np.array([0.43006, 0.5, get_desktop_z()])

    # Define the blocks to be placed, with their asset IDs and descriptive keywords.
    items_to_place = [
        {
            "oid": "sft_pen_000",
            "keywords": ["right"],
        },
    ]

    # Define the valid placement range on the table, leaving a small margin to avoid collision or falling off.
    margin = 0.00
    range_x_min = workspace_center[0] - 0.02 + margin
    range_x_max = workspace_center[0] + 0.02 - margin
    range_y_min = workspace_center[1] - 0.02 + margin
    range_y_max = workspace_center[1] + 0.02 - margin

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
    block_center0 = compute_shape_center(translated_block)
    random_angle = np.random.uniform(0, math.pi / 60)
    final_block0 = transform_shape(
        translated_block,
        rotation_matrix(angle=math.pi / 2, direction=(1, 0, 0), point=block_center0),
    )
    block_center0 = compute_shape_center(final_block0)
    final_block = transform_shape(
        final_block0,
        rotation_matrix(angle=random_angle, direction=(0, 0, 1), point=block_center0),
    )

    return concat_shapes(final_block)


@register()
def place_pen_set() -> Shape:
    # Find the usable surface area on top of the desk.
    workspace_center = np.array([0.55519, 0.52985, 0.75])

    # Define the blocks to be placed, with their asset IDs and descriptive keywords.
    items_to_place = [
        {
            "oid": "sft_pen_000",
            "keywords": ["left"],
        },
    ]

    set_shape = library_call("usd", oid="iros_pen_cup_000", keywords=[""])

    set_shapes = []
    k = random.randint(1, min(1, len(items_to_place)))
    selected_items = random.sample(items_to_place, k)
    for i, selected_item in enumerate(selected_items):
        shape = library_call(
            "usd",
            oid=selected_item["oid"],
            keywords=selected_item["keywords"],
        )
        _x = np.random.uniform(-0.01, 0.01)
        _y = np.random.uniform(-0.01, 0.01)
        _z = 0.03 + i * 0.08
        pos = np.array([_x, _y, _z])
        translated_shape = transform_shape(shape, translation_matrix(pos))
        set_shapes.append(translated_shape)

    set_shapes.append(set_shape)

    set_shapes = concat_shapes(*set_shapes)

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

    # First, translate the block to the random position on the table.
    translated_block = transform_shape(set_shapes, translation_matrix(placement_pos))

    # Then, apply a random rotation around its new center's Z-axis.
    block_center = compute_shape_center(translated_block)
    random_angle = np.random.uniform(0, math.pi / 36)
    final_block = transform_shape(
        translated_block,
        rotation_matrix(angle=random_angle, direction=(0, 0, 1), point=block_center),
    )

    return concat_shapes(final_block)


@register()
def study_room_task_desktop_5_combo() -> Shape:

    placed_blocks = []
    placed_blocks.append(place_trash_can())
    placed_blocks.append(place_trash())
    placed_blocks.append(place_mouse())
    placed_blocks.append(place_pencils())
    placed_blocks.append(place_laptops())
    placed_blocks.append(place_marker())
    placed_blocks.append(place_pen_set())

    return concat_shapes(*placed_blocks)


@register()
def root_scene() -> Shape:
    """
    The root function that generates the entire scene.
    """
    return study_room_task_desktop_5_combo()
