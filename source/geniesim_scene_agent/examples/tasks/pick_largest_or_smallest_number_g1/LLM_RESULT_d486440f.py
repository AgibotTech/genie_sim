# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: pick_largest_or_smallest_number_g2
description: A table with 3-5 randomly placed non-repeating numbered building blocks (digits 0-9) in a specified region, with no collisions and minimum 5cm spacing.
"""


def get_tabletop_workspace(table_shape: Shape) -> Tuple[P, P]:
    """
    Get the tabletop surface position and size for placing objects.

    Args:
        table_shape (Shape): The table shape

    Returns:
        Tuple[P, P]: (workspace_center, workspace_size) in world coordinates
    """
    table_info = get_object_info(table_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Desktop center in world coordinates
    desktop_center = table_info["center"] + desktop_subpart_info["center"]

    # Top surface z-coordinate
    desktop_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    workspace_center = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    workspace_size = desktop_subpart_info["size"]

    return workspace_center, workspace_size


def generate_non_colliding_positions(num_blocks: int, min_spacing: float = 0.05) -> List[Tuple[float, float]]:
    """
    Generate non-colliding positions within the specified region.

    Args:
        num_blocks (int): Number of blocks to place
        min_spacing (float): Minimum spacing between blocks in meters

    Returns:
        List[Tuple[float, float]]: List of (x, y) positions
    """
    positions = []
    max_attempts = 1000

    # Define the placement region relative to table center
    x_min, x_max = -0.2, 0.15
    y_min, y_max = -0.3, 0.3

    for i in range(num_blocks):
        for attempt in range(max_attempts):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)

            # Check if this position is far enough from all existing positions
            valid = True
            for existing_x, existing_y in positions:
                distance = ((x - existing_x) ** 2 + (y - existing_y) ** 2) ** 0.5
                if distance < min_spacing:
                    valid = False
                    break

            if valid:
                positions.append((x, y))
                break
        else:
            # If we can't find a valid position after max_attempts, just use the last generated one
            positions.append((x, y))

    return positions


@register()
def numbered_building_block(digit: int) -> Shape:
    """
    Create a numbered building block for a specific digit.

    Args:
        digit (int): The digit (0-9) to create

    Returns:
        Shape: The building block shape
    """
    # Map digits to object IDs
    digit_to_oid = {
        0: "benchmark_building_blocks_034",
        1: "benchmark_building_blocks_035",
        2: "benchmark_building_blocks_036",
        3: "benchmark_building_blocks_037",
        4: "benchmark_building_blocks_038",
        5: "benchmark_building_blocks_039",
        6: "benchmark_building_blocks_040",
        7: "benchmark_building_blocks_041",
        8: "benchmark_building_blocks_042",
        9: "benchmark_building_blocks_043"
    }

    oid = digit_to_oid[digit]
    block_shape = library_call(
        "usd",
        oid=oid,
        keywords=[f"digit_{digit}", "building_block", "colored", "numbered", f"digit_{digit}_block"]
    )

    # Random rotation around z-axis
    block_center = compute_shape_center(block_shape)
    rotation_angle = random.uniform(0, 2 * math.pi)

    rotated_block = transform_shape(
        block_shape,
        rotation_matrix(rotation_angle, direction=(0, 0, 1), point=block_center)
    )

    return rotated_block


@register()
def place_numbered_blocks_on_table(table_shape: Shape) -> Shape:
    """
    Place 3-5 randomly selected non-repeating numbered blocks on the table.

    Args:
        table_shape (Shape): The table shape

    Returns:
        Shape: Table with placed blocks
    """
    # Get table workspace
    workspace_center, workspace_size = get_tabletop_workspace(table_shape)

    # Select random number of blocks (3-5)
    num_blocks = random.randint(3, 5)

    # Select random non-repeating digits
    available_digits = list(range(10))
    selected_digits = random.sample(available_digits, num_blocks)

    # Generate non-colliding positions
    positions = generate_non_colliding_positions(num_blocks, min_spacing=0.05)

    # Place each block
    all_shapes = [table_shape]

    for i, digit in enumerate(selected_digits):
        block_shape = library_call("numbered_building_block", digit=digit)

        # Get block size to ensure proper placement
        block_info = get_object_info(block_shape)
        block_height = block_info["size"][2]

        # Position relative to table center (workspace_center is at table top surface)
        x_pos, y_pos = positions[i]
        world_x = workspace_center[0] + x_pos
        world_y = workspace_center[1] + y_pos
        world_z = workspace_center[2]  # Already at top surface

        positioned_block = transform_shape(
            block_shape,
            translation_matrix([world_x, world_y, world_z])
        )

        all_shapes.append(positioned_block)

    return concat_shapes(*all_shapes)


@register()
def pick_largest_or_smallest_number_g2() -> Shape:
    """
    Create the scene with numbered building blocks on a table for picking largest or smallest number task.
    """
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs", "for_number_blocks"]
    )

    return place_numbered_blocks_on_table(table_shape)


@register()
def root_scene() -> Shape:
    return pick_largest_or_smallest_number_g2()
