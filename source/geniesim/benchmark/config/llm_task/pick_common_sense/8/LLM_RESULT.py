# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: random_objects_on_desk
description: Randomly place one object from each category (beverage, fruit, stationery, toy) on a desk surface.
             The desk is already present at workspace_center = [2.91, 0.76, 0.80] with size [0.5, 0.8].
"""


@register()
def place_random_objects_on_desk() -> Shape:
    """
    Selects one random object from each category and places them on the desk.
    Ensures minimum distance of 10cm between objects and keeps them away from table edges.
    """
    workspace_center = np.array([0.0, 0.0, 0.80])
    # Desk dimensions: x_size=0.5, y_size=0.8
    desk_half_width = 0.25  # x direction
    desk_half_depth = 0.4  # y direction

    # Edge margin to keep objects away from table edges (at least 15cm from edges)
    edge_margin = 0.15
    # Safe placement area (reduced by edge margin)
    safe_half_width = desk_half_width - edge_margin
    safe_half_depth = desk_half_depth - edge_margin

    # Minimum distance between objects (10cm = 0.1m)
    min_distance = 0.1

    # Asset lists provided by user
    assets_by_type = {
        "beverage": [
            "benchmark_beverage_bottle_001",
            "benchmark_beverage_bottle_003",
            "benchmark_beverage_bottle_004",
            "genie_beverage_bottle_001",
            "genie_beverage_bottle_009",
        ],
        "fruit": [
            "benchmark_apple_002",
            "benchmark_green_apple_001",
            "benchmark_lemon_030",
            "benchmark_orange_004",
            "benchmark_peach_021",
        ],
        "stationery": [
            "benchmark_pen_000",
            "benchmark_pen_001",
            # "benchmark_pen_002",
            # "benchmark_pen_003",
            # "benchmark_pen_005",
            # "benchmark_pen_006",
            # "benchmark_stationery_008",
            # "benchmark_stationery_009",
            # "benchmark_stationery_010",
            # "benchmark_stationery_011",
            # "benchmark_stationery_012",
            # "benchmark_stationery_013",
            # "benchmark_stationery_014",
            # "benchmark_stationery_016",
            # "benchmark_stationery_017",
            # "benchmark_stationery_018",
            # "benchmark_stationery_019",
            # "benchmark_stationery_020",
        ],
        "toy": [
            "benchmark_garage_kit_064",
            "benchmark_rubik_cube_001",
            "benchmark_toy_car_018",
            "benchmark_toy_plane_029",
        ],
    }

    categories = ["beverage", "fruit", "stationery", "toy"]
    all_placed_shapes = []
    placed_positions = []  # Store (x, y) positions of placed objects

    for i, cat in enumerate(categories):
        # Randomly select an object ID from the category
        oid = np.random.choice(assets_by_type[cat])

        # Try to find a valid position with minimum distance constraint
        max_attempts = 200
        valid_position_found = False
        target_x = None
        target_y = None

        for attempt in range(max_attempts):
            # Calculate random position within the safe area (away from edges)
            rand_x = np.random.uniform(-safe_half_width, safe_half_width)
            rand_y = np.random.uniform(-safe_half_depth, safe_half_depth)

            # Check distance to all previously placed objects
            valid_position = True
            for prev_x, prev_y in placed_positions:
                distance = np.sqrt((rand_x - prev_x) ** 2 + (rand_y - prev_y) ** 2)
                if distance < min_distance:
                    valid_position = False
                    break

            if valid_position:
                target_x = workspace_center[0] + rand_x
                target_y = workspace_center[1] + rand_y
                valid_position_found = True
                break

        # If no valid position found after max attempts, use the last generated position
        if not valid_position_found:
            print(
                f"Warning: Could not find position with {min_distance}m spacing for {cat} object after {max_attempts} attempts"
            )
            # Use a fallback position
            target_x = workspace_center[0] + np.random.uniform(-safe_half_width, safe_half_width)
            target_y = workspace_center[1] + np.random.uniform(-safe_half_depth, safe_half_depth)

        # Store the position
        placed_positions.append((target_x - workspace_center[0], target_y - workspace_center[1]))

        # Determine position tag based on y coordinate relative to workspace center
        pos_tag = "left" if target_y > workspace_center[1] else "right"
        keywords = [oid, cat, "on desk", pos_tag]

        # Load the object (usd returns shape with origin at bottom)
        obj_shape = library_call("usd", oid=oid, keywords=keywords)

        target_z = workspace_center[2]  # Placed on the surface

        # Apply translation
        obj_shape = transform_shape(obj_shape, translation_matrix((target_x, target_y, target_z)))

        # Apply a random rotation around the vertical Z axis for natural look
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
    Creates the desk and places random objects on it.
    """
    desk = library_call("create_desk")
    objects = library_call("place_random_objects_on_desk")
    return concat_shapes(desk, objects)
