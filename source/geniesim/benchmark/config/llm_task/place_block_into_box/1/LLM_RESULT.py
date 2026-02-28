# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: building_blocks_on_desk
description: Place specific building blocks on the desk:
             - benchmark_building_blocks_000 (x > 0, y < desk_length/2)
             - benchmark_building_blocks_007, 009, 010 (required)
             - 2 additional blocks randomly selected from 001-013 (excluding 000, 002, 007, 010)
             Minimum distance: 15cm from blocks_000, 10cm between other objects.
             The desk is already present at workspace_center = [0.0, 0.0, 0.80] with size [0.5, 0.8].
"""


@register()
def place_random_objects_on_desk() -> Shape:
    """
    Places specific building blocks on the desk with distance constraints.
    - benchmark_building_blocks_000: x > 0, y < desk_length/2 (0.4m)
    - Required: benchmark_building_blocks_002, 007, 010
    - Additional: 2 random blocks from 001-013 (excluding 000, 002, 007, 010)
    - Distance from blocks_000: >= 15cm
    - Distance between other objects: >= 10cm
    """
    workspace_center = np.array([0.0, 0.0, 0.80])
    # Desk dimensions: x_size=0.5, y_size=0.8
    desk_half_width = 0.25  # x direction
    desk_half_depth = 0.4  # y direction (desk_length/2 = 0.4)

    # Edge margin to keep objects away from table edges (at least 15cm from edges)
    edge_margin = 0.15
    # Safe placement area (reduced by edge margin)
    safe_half_width = desk_half_width - edge_margin
    safe_half_depth = desk_half_depth - edge_margin

    # Distance constraints
    min_distance_from_000 = 0.15  # 15cm from blocks_000
    min_distance_between_others = 0.1  # 10cm between other objects

    # Required objects
    required_blocks = [
        "benchmark_building_blocks_000",
        "benchmark_building_blocks_002",
        "benchmark_building_blocks_007",
        "benchmark_building_blocks_010",
    ]

    # Select 2 additional blocks from 001-013, excluding required ones
    all_block_ids = [f"benchmark_building_blocks_{i:03d}" for i in range(14)]  # 000-013
    available_for_random = [bid for bid in all_block_ids if bid not in required_blocks]
    additional_blocks = np.random.choice(available_for_random, size=2, replace=False).tolist()

    # All blocks to place
    all_blocks = required_blocks + additional_blocks

    all_placed_shapes = []
    blocks_000_position = None  # Store blocks_000 position separately
    other_blocks_positions = []  # Store positions of other blocks (excluding blocks_000)

    max_attempts = 500

    # First, place benchmark_building_blocks_000 with constraints: x > 0, y < desk_length/2
    box_oid = "benchmark_building_blocks_000"
    valid_position_found = False
    target_x = None
    target_y = None

    for attempt in range(max_attempts):
        # x > 0, so rand_x should be positive
        rand_x = np.random.uniform(0.0, safe_half_width)  # x > 0
        # y < desk_length/2 (0.4m), so rand_y should be less than 0.4
        # But considering workspace_center[1] = 0.0, and safe range is [-0.25, 0.25]
        # y < 0.4 means y should be in [-0.25, 0.25] (safe range)
        rand_y = np.random.uniform(-safe_half_depth, safe_half_depth)

        # No previous objects to check against
        target_x = workspace_center[0] + rand_x
        target_y = workspace_center[1] + rand_y
        valid_position_found = True
        break

    if not valid_position_found:
        print(f"Warning: Could not find position for {box_oid} after {max_attempts} attempts")
        target_x = workspace_center[0] + np.random.uniform(0.0, safe_half_width)
        target_y = workspace_center[1] + np.random.uniform(-safe_half_depth, safe_half_depth)

    blocks_000_position = (target_x - workspace_center[0], target_y - workspace_center[1])

    # Place blocks_000
    pos_tag = "blocks_000"
    keywords = [box_oid, "building blocks", "on desk", pos_tag]
    box_shape = library_call("usd", oid=box_oid, keywords=keywords)
    target_z = workspace_center[2]
    box_shape = transform_shape(box_shape, translation_matrix((target_x, target_y, target_z)))
    obj_center = compute_shape_center(box_shape)
    random_angle = np.random.uniform(0, 2 * math.pi)
    box_shape = transform_shape(box_shape, rotation_matrix(random_angle, (0, 0, 1), obj_center))
    all_placed_shapes.append(box_shape)

    # Then, place other blocks
    other_blocks = [bid for bid in all_blocks if bid != box_oid]

    for oid in other_blocks:
        valid_position_found = False
        target_x = None
        target_y = None

        for attempt in range(max_attempts):
            # Random position within safe area
            rand_x = np.random.uniform(-safe_half_width, safe_half_width)
            rand_y = np.random.uniform(-safe_half_depth, safe_half_depth)

            # Check distance to blocks_000 (must be >= 15cm)
            distance_to_000 = np.sqrt((rand_x - blocks_000_position[0]) ** 2 + (rand_y - blocks_000_position[1]) ** 2)
            if distance_to_000 < min_distance_from_000:
                continue

            # Check distance to all other previously placed objects (must be >= 10cm)
            valid_position = True
            for prev_x, prev_y in other_blocks_positions:
                distance = np.sqrt((rand_x - prev_x) ** 2 + (rand_y - prev_y) ** 2)
                if distance < min_distance_between_others:
                    valid_position = False
                    break

            if valid_position:
                target_x = workspace_center[0] + rand_x
                target_y = workspace_center[1] + rand_y
                valid_position_found = True
                break

        if not valid_position_found:
            print(f"Warning: Could not find position for {oid} after {max_attempts} attempts")
            # Fallback: try to find any valid position
            for fallback_attempt in range(100):
                rand_x = np.random.uniform(-safe_half_width, safe_half_width)
                rand_y = np.random.uniform(-safe_half_depth, safe_half_depth)
                distance_to_000 = np.sqrt(
                    (rand_x - blocks_000_position[0]) ** 2 + (rand_y - blocks_000_position[1]) ** 2
                )
                if distance_to_000 >= min_distance_from_000:
                    valid_position = True
                    for prev_x, prev_y in other_blocks_positions:
                        distance = np.sqrt((rand_x - prev_x) ** 2 + (rand_y - prev_y) ** 2)
                        if distance < min_distance_between_others:
                            valid_position = False
                            break
                    if valid_position:
                        target_x = workspace_center[0] + rand_x
                        target_y = workspace_center[1] + rand_y
                        valid_position_found = True
                        break
            if not valid_position_found:
                # Last resort: place at a random position
                target_x = workspace_center[0] + np.random.uniform(-safe_half_width, safe_half_width)
                target_y = workspace_center[1] + np.random.uniform(-safe_half_depth, safe_half_depth)

        other_blocks_positions.append((target_x - workspace_center[0], target_y - workspace_center[1]))
        pos_tag = "other_block"
        keywords = [oid, "building blocks", "on desk", pos_tag]
        obj_shape = library_call("usd", oid=oid, keywords=keywords)
        target_z = workspace_center[2]
        obj_shape = transform_shape(obj_shape, translation_matrix((target_x, target_y, target_z)))
        obj_center = compute_shape_center(obj_shape)
        random_angle = np.random.uniform(0, 2 * math.pi)
        obj_shape = transform_shape(obj_shape, rotation_matrix(random_angle, (0, 0, 1), obj_center))
        all_placed_shapes.append(obj_shape)

    return concat_shapes(*all_placed_shapes)


@register()
def create_desk() -> Shape:
    """
    Creates the desk at the workspace center.
    The table_000 asset should have its origin at the bottom, and when placed at z=0,
    its top surface should be at approximately 0.80m height.
    """
    workspace_center = np.array([0.0, 0.0, 0.80])

    # Load table asset
    table_shape = library_call("usd", oid="table_000", keywords=["table_000", "desk", "table"])

    # Position the table on the ground (z=0), so its top surface is at the correct height
    # The workspace_center[2] = 0.80 represents the table top height
    table_shape = transform_shape(
        table_shape,
        translation_matrix((workspace_center[0], workspace_center[1], 0.0)),
    )

    return table_shape


@register()
def root_scene() -> Shape:
    """
    Main entry point for the scene generation.
    Creates the desk and places building blocks on it.
    """
    desk = library_call("create_desk")
    objects = library_call("place_random_objects_on_desk")
    return concat_shapes(desk, objects)
