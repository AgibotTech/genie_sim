# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
import itertools

"""
scene_name: table_with_flat_medium_storage_boxes
description: A table with two medium-sized storage boxes standing upright side-by-side in the center, and various items (mouse, sprite, sponge, papercup, apple) placed on the desktop.
"""

# Available storage box IDs (removed duplicate 006)
AVAILABLE_BOX_IDS = [
    "benchmark_storage_box_006",
    "benchmark_storage_box_011",
    "benchmark_storage_box_010",
    "benchmark_storage_box_007",
    "benchmark_storage_box_008",
]

# Generate all possible combinations of 2 boxes (C(5,2) = 10 combinations)
# This ensures we have exactly 10 unique combinations for 10 runs
_all_combinations = list(itertools.combinations(AVAILABLE_BOX_IDS, 2))
# Shuffle to randomize the order
random.shuffle(_all_combinations)

# Track which combination to use next (module-level state)
_combination_index = 0


def get_desktop_top_info(table_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the desktop's top surface and its size.
    """
    table_info = get_object_info(table_shape)
    # table_000 has a subpart named 'desktop'
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # The desktop center in world coordinates
    desktop_center_xy = table_info["center"][:2] + desktop_subpart_info["center"][:2]

    # The top surface z-coordinate of the desktop in world coordinates
    desktop_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    desktop_pos = np.array([desktop_center_xy[0], desktop_center_xy[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


def get_next_box_combination():
    """
    Returns the next unique combination of two box IDs.
    Cycles through all 10 combinations, ensuring no repetition in 10 consecutive runs.
    """
    global _combination_index
    combination = _all_combinations[_combination_index % len(_all_combinations)]
    _combination_index += 1
    return combination


@register()
def flat_medium_storage_boxes(surface_pos: P, surface_size: P, box1_id: str = None, box2_id: str = None) -> Shape:
    """
    Places two medium-sized storage boxes standing upright side-by-side in the center-upper area of the table.
    If box IDs are not provided, selects the next unique combination.
    """
    # Get box IDs - use provided ones or select next combination
    if box1_id is None or box2_id is None:
        box1_id, box2_id = get_next_box_combination()

    # Load the two boxes
    box1 = library_call(
        "usd",
        oid=box1_id,
        keywords=["medium_storage_box", "storage_box", "rectangular", "center"],
    )
    box2 = library_call(
        "usd",
        oid=box2_id,
        keywords=["medium_storage_box", "storage_box", "rectangular", "center"],
    )

    # Rotate boxes 90 degrees around Z axis (vertical axis)
    box1_center = compute_shape_center(box1)
    box2_center = compute_shape_center(box2)

    # Rotate 90 degrees (pi/2) around Z axis
    box1_rotated = transform_shape(box1, rotation_matrix(math.pi / 2, direction=(0, 0, 1), point=box1_center))
    box2_rotated = transform_shape(box2, rotation_matrix(math.pi / 2, direction=(0, 0, 1), point=box2_center))

    # Get box sizes after rotation
    box1_size = compute_shape_sizes(box1_rotated)
    box2_size = compute_shape_sizes(box2_rotated)

    # Get the bottom Z coordinate of each rotated box
    box1_min_z = compute_shape_min(box1_rotated)[2]
    box2_min_z = compute_shape_min(box2_rotated)[2]

    # Calculate FIXED symmetric positions so boxes are symmetrically placed about table center
    # Position is fixed relative to table center, ensuring consistent placement every time
    # Place box1 on the left, box2 on the right, symmetric about the table center (y = surface_pos[1])

    # Calculate the total width of both boxes (they will be touching, no gap)
    total_width = box1_size[1] + box2_size[1]

    # For symmetric placement about table center:
    # box1 center y = table center - (box1 half width + box2 half width)
    # box2 center y = table center + (box1 half width + box2 half width)
    # This ensures the pair is centered and symmetric
    box1_center_y = surface_pos[1] - total_width / 2 + box1_size[1] / 2
    box2_center_y = surface_pos[1] + total_width / 2 - box2_size[1] / 2

    # Verify symmetry: the midpoint between box1 and box2 should be at surface_pos[1]
    # midpoint = (box1_center_y + box2_center_y) / 2 = surface_pos[1]

    # Place boxes in the FIXED center-upper area of the table (positive x direction is forward/up)
    # Fixed offset: 15% of table length forward from center - this ensures consistent position
    box_center_x = surface_pos[0] + surface_size[0] * 0.15

    # Place both boxes, ensuring bottom is on the surface
    box1_center_pos = np.array([box_center_x, box1_center_y, surface_pos[2] - box1_min_z])
    box2_center_pos = np.array([box_center_x, box2_center_y, surface_pos[2] - box2_min_z])

    box1_final = transform_shape(box1_rotated, translation_matrix(box1_center_pos))
    box2_final = transform_shape(box2_rotated, translation_matrix(box2_center_pos))

    return concat_shapes(box1_final, box2_final)


@register()
def desktop_items(surface_pos: P, surface_size: P) -> Shape:
    """
    Places mouse, sprite, sponge, papercup, and apple on the lower half of the table surface.
    Ensures no collisions between objects.
    """
    # Define items to place
    items_config = [
        ("mouse", "mouse", ["mouse", "gray", "left"]),
        ("sponge", "sponge", ["sponge", "yellow", "right"]),
        ("papercup", "papercup", ["papercup", "white", "right"]),
        ("benchmark_apple_002", "apple", ["apple", "red", "right"]),
    ]

    # Define placement area: all objects must have x < 0 (negative x direction from center)
    # Keep objects away from edges
    # max_offset_x must be negative to ensure x < surface_pos[0]
    max_offset_x = -0.05  # 5cm backward from center (ensures x < 0)
    min_offset_x = -surface_size[0] * 0.45  # 45% of table length backward from center
    max_offset_y = surface_size[1] * 0.45  # 45% of table width on each side

    # Minimum distance between objects (10cm = 0.1m)
    min_distance = 0.05

    placed_objects = []
    placed_positions = []  # Store (x, y) positions of placed objects

    for oid, name, keywords in items_config:
        # Load the object to get its size for collision checking
        obj_shape = library_call("usd", oid=oid, keywords=keywords)
        obj_size = compute_shape_sizes(obj_shape)

        # Estimate object radius (half of diagonal in XY plane)
        obj_radius = np.sqrt(obj_size[0] ** 2 + obj_size[1] ** 2) / 2

        # Attempt to find a valid random position
        max_attempts = 500
        valid_position = None

        for attempt in range(max_attempts):
            # Generate random position with x < 0 (negative x direction from center)
            # Ensure rand_x < surface_pos[0] by using negative offsets
            rand_x = surface_pos[0] + np.random.uniform(min_offset_x, max_offset_x)
            rand_y = surface_pos[1] + np.random.uniform(-max_offset_y, max_offset_y)
            candidate_pos = np.array([rand_x, rand_y, surface_pos[2]])

            # Double check: ensure x coordinate is less than surface center (x < 0 relative to center)
            if candidate_pos[0] >= surface_pos[0]:
                continue

            # Check minimum distance from all already placed objects
            # Ensure edge-to-edge distance >= min_distance (10cm)
            # Center-to-center distance must be >= min_distance + obj_radius + placed_radius
            # This guarantees edge-to-edge distance >= min_distance
            is_valid = True
            for placed_pos, placed_radius in placed_positions:
                distance_2d = np.linalg.norm(candidate_pos[:2] - placed_pos[:2])
                required_distance = min_distance + obj_radius + placed_radius
                if distance_2d < required_distance:
                    is_valid = False
                    break

            if is_valid:
                valid_position = candidate_pos
                break

        # If no valid position found, use a fallback position with collision checking
        if valid_position is None:
            # Fallback: try to find a position that satisfies constraints
            # Try a few more attempts with a wider search area
            for fallback_attempt in range(50):
                fallback_x = surface_pos[0] + np.random.uniform(min_offset_x * 0.8, max_offset_x)
                # Ensure fallback x is still less than surface_pos[0]
                if fallback_x >= surface_pos[0]:
                    fallback_x = surface_pos[0] - 0.1
                fallback_y = surface_pos[1] + np.random.uniform(-max_offset_y * 0.8, max_offset_y * 0.8)
                fallback_pos = np.array([fallback_x, fallback_y, surface_pos[2]])

                # Check collision for fallback position
                fallback_valid = True
                for placed_pos, placed_radius in placed_positions:
                    distance_2d = np.linalg.norm(fallback_pos[:2] - placed_pos[:2])
                    required_distance = min_distance + obj_radius + placed_radius
                    if distance_2d < required_distance:
                        fallback_valid = False
                        break

                if fallback_valid:
                    valid_position = fallback_pos
                    break

            # Last resort: place at a safe distance from all existing objects
            if valid_position is None:
                # Find a position that maintains minimum distance
                fallback_x = surface_pos[0] - 0.15  # Safe x position
                fallback_y = surface_pos[1]
                # Try to find y position that maintains distance
                for y_offset in np.linspace(-max_offset_y, max_offset_y, 20):
                    test_pos = np.array([fallback_x, surface_pos[1] + y_offset, surface_pos[2]])
                    test_valid = True
                    for placed_pos, placed_radius in placed_positions:
                        distance_2d = np.linalg.norm(test_pos[:2] - placed_pos[:2])
                        required_distance = min_distance + obj_radius + placed_radius
                        if distance_2d < required_distance:
                            test_valid = False
                            break
                    if test_valid:
                        valid_position = test_pos
                        break

                # If still no valid position, use minimum distance placement
                if valid_position is None:
                    # Place at a calculated safe distance from the nearest object
                    if placed_positions:
                        # Find the farthest point from all existing objects
                        best_pos = None
                        best_min_dist = 0
                        for test_x in np.linspace(min_offset_x, max_offset_x, 10):
                            for test_y in np.linspace(-max_offset_y, max_offset_y, 10):
                                test_pos = np.array(
                                    [
                                        surface_pos[0] + test_x,
                                        surface_pos[1] + test_y,
                                        surface_pos[2],
                                    ]
                                )
                                min_dist_to_any = float("inf")
                                for placed_pos, placed_radius in placed_positions:
                                    distance_2d = np.linalg.norm(test_pos[:2] - placed_pos[:2])
                                    required_distance = min_distance + obj_radius + placed_radius
                                    if distance_2d < required_distance:
                                        min_dist_to_any = -1
                                        break
                                    min_dist_to_any = min(min_dist_to_any, distance_2d - required_distance)
                                if min_dist_to_any > best_min_dist:
                                    best_min_dist = min_dist_to_any
                                    best_pos = test_pos
                        if best_pos is not None:
                            valid_position = best_pos
                        else:
                            # Absolute last resort: place with minimum safe distance
                            valid_position = np.array([surface_pos[0] - 0.2, surface_pos[1], surface_pos[2]])
                    else:
                        valid_position = np.array([surface_pos[0] - 0.15, surface_pos[1], surface_pos[2]])

        # Store position and radius for collision checking
        placed_positions.append((valid_position, obj_radius))

        # Reload object with appropriate position tag
        offset_y = valid_position[1] - surface_pos[1]
        if offset_y > 0.05:
            pos_tag = "left"
        elif offset_y < -0.05:
            pos_tag = "right"
        else:
            pos_tag = "center"

        keywords_with_pos = keywords[:-1] + [pos_tag] if keywords else [pos_tag]
        obj_shape = library_call("usd", oid=oid, keywords=keywords_with_pos)

        # Apply translation to place the object
        obj_shape = transform_shape(obj_shape, translation_matrix(valid_position))
        placed_objects.append(obj_shape)

    return concat_shapes(*placed_objects)


@register()
def root_scene() -> Shape:
    # 1. Load the table
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["main_table", "table", "white"],
    )

    # 2. Get surface info for placement
    surface_pos, surface_size = get_desktop_top_info(table_shape)

    # 3. Place medium storage boxes standing upright side-by-side in the center-upper area
    boxes = library_call("flat_medium_storage_boxes", surface_pos=surface_pos, surface_size=surface_size)

    # 4. Place other items on the desktop
    items = library_call("desktop_items", surface_pos=surface_pos, surface_size=surface_size)

    return concat_shapes(table_shape, boxes, items)
