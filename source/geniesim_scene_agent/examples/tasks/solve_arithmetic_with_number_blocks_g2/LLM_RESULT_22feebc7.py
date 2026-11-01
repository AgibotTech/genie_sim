# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: table_with_number_blocks
description: A table with 4-7 randomly placed number blocks (digits 0-9), with at least 3 different digits,
             placed in region x(-0.1, 0.15), y(-0.3, 0.3), with minimum 5cm spacing and rotation between 160-200 degrees.
"""


def get_table_top_info(table_shape: Shape) -> Tuple[P, P]:
    """
    Get the table top surface center position and size in world coordinates.
    """
    table_info = get_object_info(table_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Table top center in world coordinates
    table_top_center = table_info["center"] + desktop_subpart_info["center"]

    # Table top surface z-coordinate (top of desktop)
    table_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    table_top_pos = np.array([table_top_center[0], table_top_center[1], table_top_z])
    table_top_size = desktop_subpart_info["size"]

    return table_top_pos, table_top_size


def generate_random_digits(num_blocks: int) -> List[int]:
    """
    Generate random digits ensuring at least 3 different digits.
    """
    if num_blocks < 3:
        # This shouldn't happen given our constraints (4-7 blocks)
        digits = random.choices(range(10), k=num_blocks)
    else:
        # Ensure at least 3 different digits
        unique_digits = random.sample(range(10), 3)
        remaining_count = num_blocks - 3
        remaining_digits = random.choices(range(10), k=remaining_count)
        digits = unique_digits + remaining_digits
        random.shuffle(digits)

    return digits


def get_block_object_id(digit: int) -> str:
    """
    Map digit to corresponding object ID.
    """
    base_id = 34  # benchmark_building_blocks_034 is digit 0
    return f"benchmark_building_blocks_{base_id + digit:03d}"


def blocks_overlap(pos1: P, pos2: P, min_distance: float = 0.05) -> bool:
    """
    Check if two blocks overlap (distance less than minimum required).
    """
    distance = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
    return distance < min_distance


@register()
def number_block(digit: int, block_index: int) -> Shape:
    """
    Create a single number block with specified digit.
    """
    object_id = get_block_object_id(digit)
    block_shape = library_call(
        "usd",
        oid=object_id,
        keywords=[f"number_block_{digit}", "building_block", "colored", f"digit_{digit}", f"block_{block_index}"],
    )

    # Get block size for proper placement
    block_info = get_object_info(block_shape)
    block_size = block_info["size"]

    return block_shape


@register()
def place_number_blocks_on_table(table_shape: Shape) -> Shape:
    """
    Place 4-7 number blocks on the table with specified constraints.
    """
    # Get table top information
    table_top_pos, table_top_size = get_table_top_info(table_shape)

    # Determine number of blocks (4-7)
    num_blocks = random.randint(4, 7)

    # Generate digits ensuring at least 3 different ones
    digits = generate_random_digits(num_blocks)

    # Placement region relative to table center
    x_min, x_max = -0.1, 0.15
    y_min, y_max = -0.3, 0.3

    # Store placed block positions to avoid collisions
    placed_positions = []
    all_blocks = []

    for i in range(num_blocks):
        digit = digits[i]
        block_shape = library_call("number_block", digit=digit, block_index=i)
        block_info = get_object_info(block_shape)
        block_size = block_info["size"]

        # Try to find a valid position (avoid collisions)
        max_attempts = 100
        position_found = False

        for attempt in range(max_attempts):
            # Generate random position in the specified region
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            z = table_top_pos[2]  # Place on table top

            candidate_pos = [x, y, z]

            # Check for collisions with already placed blocks
            collision = False
            for placed_pos in placed_positions:
                if blocks_overlap(candidate_pos, placed_pos, min_distance=0.05):
                    collision = True
                    break

            if not collision:
                position_found = True
                placed_positions.append(candidate_pos)
                break

        # If no valid position found after max attempts, use the last candidate
        if not position_found:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            candidate_pos = [x, y, table_top_pos[2]]
            placed_positions.append(candidate_pos)

        # Apply translation to place block on table
        block_translated = transform_shape(
            block_shape,
            translation_matrix(candidate_pos)
        )

        # Apply random rotation (160-200 degrees around z-axis)
        rotation_angle = random.uniform(160, 200) * (math.pi / 180.0)  # Convert to radians
        block_center = compute_shape_center(block_translated)
        block_rotated = transform_shape(
            block_translated,
            rotation_matrix(rotation_angle, direction=(0, 0, 1), point=block_center)
        )

        all_blocks.append(block_rotated)

    # Combine table and all blocks
    return concat_shapes(table_shape, *all_blocks)


@register()
def table_with_number_blocks() -> Shape:
    """
    Create the complete scene with table and number blocks.
    """
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )
    return place_number_blocks_on_table(table_shape)


@register()
def root_scene() -> Shape:
    return table_with_number_blocks()
