# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: table_with_random_blocks
description: A white table (table_000) with five different colored cubic building blocks (size >= 0.04m) placed randomly on its surface.
"""


def get_desktop_surface_info(table_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the table's top surface and its size.
    This is a helper function and is not registered.
    """
    table_info = get_object_info(table_shape)
    # table_000 has a subpart named 'desktop'
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
    Places 5 different colored blocks at random positions and rotations on the table surface.
    Ensures minimum distance of 10cm between any two blocks.
    """
    surface_pos, surface_size = get_desktop_surface_info(table_shape)

    # List of selected assets: Red, Yellow, Blue, Green, Dark Purple
    # All are cubes with size >= 0.04m
    block_assets = [
        ("benchmark_building_blocks_073", "red"),
        ("benchmark_building_blocks_085", "yellow"),
        ("benchmark_building_blocks_077", "blue"),
        ("benchmark_building_blocks_081", "green"),
        ("benchmark_building_blocks_089", "purple"),
    ]

    # Define margins to keep blocks away from the edge (block size is ~0.05m, add extra margin)
    # Keep blocks at least 0.15m away from the edges
    margin_x = surface_size[0] / 2.0 - 0.15
    margin_y = surface_size[1] / 2.0 - 0.15

    # Limit placement in the upper half (positive x direction) to avoid objects too far in the upper half
    # Upper half range is reduced to 50% of the full margin to keep objects closer to center
    margin_x_upper = margin_x * 0.5

    # Minimum distance between blocks (10cm = 0.1m)
    min_distance = 0.1

    blocks_combined = []
    placed_positions = []  # Store (x, y) positions of placed blocks

    for oid, color in block_assets:
        # Try to find a valid position with minimum distance constraint
        max_attempts = 500
        valid_position_found = False

        for attempt in range(max_attempts):
            # Generate random coordinates within the desktop surface bounds
            # For x direction: lower half uses full range, upper half uses reduced range
            # This prevents objects from being placed too far in the upper half
            if np.random.random() < 0.5:
                # Lower half (negative x): full range
                rand_x = np.random.uniform(-margin_x, 0)
            else:
                # Upper half (positive x): reduced range to keep closer to center
                rand_x = np.random.uniform(0, margin_x_upper)
            rand_y = np.random.uniform(-margin_y, margin_y)

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
                f"Warning: Could not find position with {min_distance}m spacing for {color} block after {max_attempts} attempts"
            )

        # Store the position
        placed_positions.append((rand_x, rand_y))

        # Calculate world position
        world_x = surface_pos[0] + rand_x
        world_y = surface_pos[1] + rand_y
        world_z = surface_pos[2]

        # Determine POSITION TAG based on y coordinate (+y is left, -y is right)
        pos_tag = "left" if rand_y > 0 else "right"

        # Load the block
        block_shape = library_call(
            "usd",
            oid=oid,
            keywords=[f"{color}_block", "cube", "toy", color, pos_tag, "random_placement"],
        )

        # Apply random rotation around Z-axis for a natural "scattered" look
        rand_angle = np.random.uniform(0, 2 * math.pi)

        # First translate to the random position on the table
        # (usd objects are already bottom-centered)
        block_shape = transform_shape(block_shape, translation_matrix([world_x, world_y, world_z]))

        # Then rotate around its own center at that position
        block_center = compute_shape_center(block_shape)
        block_shape = transform_shape(block_shape, rotation_matrix(rand_angle, direction=(0, 0, 1), point=block_center))

        blocks_combined.append(block_shape)

    return concat_shapes(table_shape, *blocks_combined)


@register()
def table_scene() -> Shape:
    """
    Initializes the table and triggers the random block placement.
    """
    # Load the table at the origin on the ground
    table_shape = library_call("usd", oid="table_000", keywords=["table_000", "white_table", "furniture", "center"])

    return library_call("place_random_blocks", table_shape=table_shape)


@register()
def root_scene() -> Shape:
    """
    Main entry point for the scene generation.
    """
    return table_scene()
