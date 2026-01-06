# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: random_blocks_on_table
description: A table with 5 building blocks, each randomly selected from a different shape category (cube, cuboid, rectangular prism, triangular prism, and cylindrical), placed in the middle area of the desktop.
"""


def find_desktop_surface(table_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the table's top surface and its size.
    """
    table_info = get_object_info(table_shape)
    # Accessing subpart info for table_000
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # The desktop center in world coordinates
    desktop_center = table_info["center"] + desktop_subpart_info["center"]

    # The top surface z-coordinate of the desktop in world coordinates
    desktop_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    # The final desktop position (center_x, center_y, top_z)
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


@register()
def place_random_blocks(table_shape: Shape) -> Shape:
    """
    Randomly selects one block from each of the 5 shape categories and places them on the table.
    Ensures minimum distance of 10cm between blocks and keeps them away from table edges.
    """
    surface_pos, surface_size = find_desktop_surface(table_shape)

    # Define shape categories and their corresponding asset IDs
    categories = {
        "cube": [
            "benchmark_building_blocks_019",
            "benchmark_building_blocks_078",
            "benchmark_building_blocks_086",
            "benchmark_building_blocks_079",
            "benchmark_building_blocks_090",
        ],
        "rectangular_prism": [
            "benchmark_building_blocks_046",
            "benchmark_building_blocks_020",
            "benchmark_building_blocks_022",
        ],
        "triangular_prism": ["benchmark_building_blocks_018"],
        "cylindrical": [
            "benchmark_building_blocks_023",
            "benchmark_building_blocks_024",
            "benchmark_building_blocks_004",
            "benchmark_building_blocks_048",
        ],
        "hexagonal": ["benchmark_building_blocks_047"],
        "l_shape": ["benchmark_building_blocks_013"],
    }

    # Define margins to keep blocks away from table edges (at least 15cm from edges)
    edge_margin = 0.15
    margin_x = surface_size[0] / 2.0 - edge_margin
    margin_y = surface_size[1] / 2.0 - edge_margin

    # Minimum distance between blocks (10cm = 0.1m)
    min_distance = 0.1

    placed_blocks = []
    placed_positions = []  # Store (x, y) positions of placed blocks

    # Iterate through each category to ensure 5 different shapes
    for i, (shape_name, ids) in enumerate(categories.items()):
        # Randomly pick one ID from the current category
        chosen_id = np.random.choice(ids)

        # Try to find a valid position with minimum distance constraint
        max_attempts = 200
        valid_position_found = False
        rand_x = None
        rand_y = None

        for attempt in range(max_attempts):
            # Calculate random position within the safe area (away from edges)
            rand_x = surface_pos[0] + np.random.uniform(-margin_x, margin_x)
            rand_y = surface_pos[1] + np.random.uniform(-margin_y, margin_y)

            # Check distance to all previously placed blocks
            valid_position = True
            for prev_x, prev_y in placed_positions:
                distance = np.sqrt((rand_x - prev_x) ** 2 + (rand_y - prev_y) ** 2)
                if distance < min_distance:
                    valid_position = False
                    break

            if valid_position:
                valid_position_found = True
                break

        # If no valid position found after max attempts, use the last generated position
        if not valid_position_found:
            print(
                f"Warning: Could not find position with {min_distance}m spacing for {shape_name} block after {max_attempts} attempts"
            )

        # Store the position
        placed_positions.append((rand_x, rand_y))

        # Determine position tag (left or right) based on y-coordinate relative to table center
        pos_tag = "left" if rand_y > surface_pos[1] else "right"

        # Load the block
        block_shape = library_call(
            "usd",
            oid=chosen_id,
            keywords=[f"block_{shape_name}_{i}", shape_name, "building block", pos_tag, "on table"],
        )

        # Move to the calculated position on the table surface
        block_shape = transform_shape(block_shape, translation_matrix((rand_x, rand_y, surface_pos[2])))

        # Apply a random rotation around the vertical axis (Z)
        block_center = compute_shape_center(block_shape)
        block_shape = transform_shape(
            block_shape,
            rotation_matrix(np.random.uniform(0, 2 * math.pi), direction=(0, 0, 1), point=block_center),
        )

        placed_blocks.append(block_shape)

    return concat_shapes(table_shape, *placed_blocks)


@register()
def root_scene() -> Shape:
    """
    Generates the complete scene with a table and 5 randomly selected blocks.
    """
    # Load the table at the origin (0, 0, 0)
    table_shape = library_call("usd", oid="table_000", keywords=["main_table", "table", "wooden", "center"])

    # Place the randomly selected blocks on the table
    return place_random_blocks(table_shape)
