# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: table_with_building_blocks
description: A table with 6 building blocks of different sizes (0.05, 0.04, 0.03) randomly placed on its surface.
"""


def get_table_surface_info(table_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the table's top surface and its size.
    """
    table_info = get_object_info(table_shape)
    # table_000 has a subpart named 'desktop'
    desktop_subpart = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Calculate the top surface center and size
    # The usd function returns shape with origin at bottom, so table_info['min'][2] is 0.
    surface_center = table_info["center"] + desktop_subpart["center"]
    surface_top_z = table_info["center"][2] + desktop_subpart["xyz_max"][2]

    surface_pos = np.array([surface_center[0], surface_center[1], surface_top_z])
    surface_size = desktop_subpart["size"]

    return surface_pos, surface_size


@register()
def place_blocks_on_table(table_shape: Shape) -> Shape:
    """
    Places the 6 building blocks randomly on the table surface.
    Ensures minimum distance of 10cm between blocks and keeps them away from table edges.
    """
    surface_pos, surface_size = get_table_surface_info(table_shape)

    # Load the blocks
    blocks_shape = library_call("building_blocks")

    # Since blocks_shape is a concatenated list of 6 objects, we need to handle them individually.
    # However, the DSL concat_shapes/transform_shape works on the whole list.
    # To place them randomly, we'll reconstruct the scene by placing each block.

    final_scene = table_shape

    # Define block IDs again to iterate and place
    # Large blocks (0.05 size) - randomly select one from multiple options
    blocks_05_options = [
        "benchmark_building_blocks_086",
        "benchmark_building_blocks_074",
        "benchmark_building_blocks_082",
        "benchmark_building_blocks_078",
        "benchmark_building_blocks_090",
    ]
    blocks_05 = np.random.choice(blocks_05_options, size=4, replace=False).tolist()
    # blocks_04_options = [
    #     "benchmark_building_blocks_085",
    #     "benchmark_building_blocks_073",
    #     "benchmark_building_blocks_081",
    #     "benchmark_building_blocks_077",
    #     "benchmark_building_blocks_089",
    # ]
    # blocks_04 = np.random.choice(
    #     blocks_04_options, size=3, replace=False
    # ).tolist()  # Randomly select 3 blocks

    blocks_03_options = [
        "benchmark_building_blocks_084",
        "benchmark_building_blocks_072",
        "benchmark_building_blocks_080",
        "benchmark_building_blocks_076",
        "benchmark_building_blocks_088",
    ]
    blocks_03 = np.random.choice(blocks_03_options, size=1, replace=False).tolist()
    all_block_ids = blocks_05 + blocks_03

    # Define margins to keep blocks away from table edges
    # Use larger margin for x direction to keep objects further from x-axis edges
    edge_margin_x = 0.15  # 15cm margin for x direction (left/right edges)
    edge_margin_y = 0.15  # 15cm margin for y direction (front/back edges)
    margin_x = surface_size[0] / 2.0 - edge_margin_x
    margin_y = surface_size[1] / 2.0 - edge_margin_y

    # Minimum distance between blocks (10cm = 0.1m)
    min_distance = 0.1

    placed_positions = []  # Store (x, y) positions of placed blocks

    for i, oid in enumerate(all_block_ids):
        # Determine size tag based on position in the list
        # blocks_05: index 0
        # blocks_04: indices 1-3
        # blocks_03: index 4
        if i == 4:
            size_tag = "0.03 size"
        else:
            size_tag = "0.05 size"

        # Try to find a valid position with minimum distance constraint
        # Use multiple attempts to ensure we place all required blocks
        max_attempts = 500
        valid_position_found = False
        rand_x = None
        rand_y = None
        fallback_x = None
        fallback_y = None
        fallback_min_dist = min_distance * 0.8  # Fallback: 80% of minimum distance (8cm)

        for attempt in range(max_attempts):
            # Calculate random position within the safe area (away from edges)
            rand_x = np.random.uniform(-margin_x, margin_x)
            rand_y = np.random.uniform(-margin_y, margin_y)

            # Check distance to all previously placed blocks
            min_actual_distance = float("inf")
            valid_position = True
            for prev_x, prev_y in placed_positions:
                distance = np.sqrt((rand_x - prev_x) ** 2 + (rand_y - prev_y) ** 2)
                min_actual_distance = min(min_actual_distance, distance)
                if distance < min_distance:
                    valid_position = False
                    break

            if valid_position:
                valid_position_found = True
                break
            elif min_actual_distance >= fallback_min_dist:
                # Store as fallback if it meets the relaxed distance requirement
                fallback_x = rand_x
                fallback_y = rand_y

        # Ensure we always place the block (required: 1x0.05, 3x0.04, 1x0.03)
        if not valid_position_found:
            if fallback_x is not None and fallback_y is not None:
                # Use fallback position with relaxed distance
                rand_x = fallback_x
                rand_y = fallback_y
                print(f"Warning: Using fallback position with relaxed distance for block {i} (0.08m instead of 0.1m)")
            else:
                # Use the last generated position as last resort
                print(f"Warning: Using last attempt position for block {i} (may not meet distance requirement)")

        # Store the position
        placed_positions.append((rand_x, rand_y))

        # Determine POSITION TAG
        pos_tag = "left" if rand_y > 0 else "right"

        # Load individual block
        block = library_call(
            "usd",
            oid=oid,
            keywords=["building_block", "square", size_tag, pos_tag, f"random_placed_{i}"],
        )

        # Transform to random position on surface
        # surface_pos is (center_x, center_y, top_z)
        world_x = surface_pos[0] + rand_x
        world_y = surface_pos[1] + rand_y
        world_z = surface_pos[2]

        block = transform_shape(block, translation_matrix([world_x, world_y, world_z]))

        # Add random rotation around Z axis
        block_center = compute_shape_center(block)
        block = transform_shape(block, rotation_matrix(np.random.uniform(0, 2 * math.pi), [0, 0, 1], block_center))

        final_scene = concat_shapes(final_scene, block)

    return final_scene


@register()
def root_scene() -> Shape:
    """
    Root function to generate the table with building blocks.
    """
    # Load the table at the origin
    table_shape = library_call("usd", oid="table_000", keywords=["table", "wooden", "furniture", "base_surface"])

    # Place blocks on the table
    return place_blocks_on_table(table_shape)
