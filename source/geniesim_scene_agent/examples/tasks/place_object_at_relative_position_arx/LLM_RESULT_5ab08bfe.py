# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: table_with_random_objects
description: A white desk (table_000) with 3 to 6 unique randomly selected objects placed within a bounded region (x: -0.2 to 0.2, y: -0.25 to 0.25), ensuring no duplicates and minimum distance of 0.15m between objects.
"""


def get_desktop_placement_area(desk_shape: Shape) -> Tuple[P, P]:
    """
    Get the top surface center and size of the desktop for object placement.

    Args:
        desk_shape (Shape): The shape of the desk.

    Returns:
        Tuple[P, P]: (center_position, size) of the desktop's top surface in world coordinates.
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Compute desktop top surface center in world frame
    desktop_center = desk_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


def sample_position_within_bounds(center: P, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> P:
    """
    Sample a position within specified x and y bounds relative to a center point.
    """
    x = np.random.uniform(x_range[0], x_range[1])
    y = np.random.uniform(y_range[0], y_range[1])
    return np.array([center[0] + x, center[1] + y, center[2]])


def check_minimum_distance(new_obj: Shape, existing_objs: List[Shape], min_dist: float = 0.15) -> bool:
    """
    Check if the new object is at least min_dist away from all existing objects (center-to-center distance).
    """
    new_center = compute_shape_center(new_obj)
    for obj in existing_objs:
        obj_center = compute_shape_center(obj)
        distance = np.linalg.norm(new_center[:2] - obj_center[:2])  # Only consider x,y distance
        if distance < min_dist:
            return False
    return True


# Object ID to semantic name mapping
OBJECT_SEMANTIC_NAMES = {
    "benchmark_building_blocks_072": "building blocks",
    "benchmark_apple_002": "apple",
    "benchmark_orange_004": "orange",
    "benchmark_rubik_cube_001": "rubik cube",
    "benchmark_medicine_bottle_000": "medicine bottle",
}


@register()
def place_random_objects_on_desk(desk_shape: Shape) -> Shape:
    # Available object IDs (validated to exist in assets library)
    available_objects = [
        "benchmark_building_blocks_072",
        "benchmark_apple_002",
        "benchmark_orange_004",
        "benchmark_rubik_cube_001",
        "benchmark_medicine_bottle_000",
    ]

    # Determine number of objects to place (3 to 6)
    num_objects = 2

    # Ensure we don't try to place more objects than available
    num_objects = min(num_objects, len(available_objects))

    # Randomly select unique objects
    selected_objects = random.sample(available_objects, num_objects)

    # Get desktop placement area
    desktop_center, desktop_size = get_desktop_placement_area(desk_shape)

    # Ensure we don't try to place more objects than available
    num_objects = min(num_objects, len(available_objects))

    # Randomly select unique objects
    selected_objects = random.sample(available_objects, num_objects)

    # Get desktop placement area
    desktop_center, desktop_size = get_desktop_placement_area(desk_shape)

    # Define placement bounds relative to desktop center
    x_bounds = (-0.2, 0.15)
    y_bounds = (-0.25, 0.25)

    placed_objects = []
    attempts_per_object = 0
    max_attempts_per_object = 20

    for obj_id in selected_objects:
        placed = False
        attempts = 0

        while not placed and attempts < max_attempts_per_object:
            # Load the object with semantic name in keywords
            semantic_name = OBJECT_SEMANTIC_NAMES.get(obj_id, obj_id)

            # Determine position tag based on x coordinate
            base_x = np.random.uniform(x_bounds[0], x_bounds[1])
            base_y = np.random.uniform(y_bounds[0], y_bounds[1])
            position_tag = "left" if base_x < 0 else "right"

            obj_shape = library_call("usd", oid=obj_id, keywords=[semantic_name, obj_id, "on_desk", position_tag])

            # Sample a position within bounds
            base_pos = np.array([desktop_center[0] + base_x, desktop_center[1] + base_y, desktop_center[2]])

            # Get object size to adjust placement height
            obj_min = compute_shape_min(obj_shape)
            obj_height = compute_shape_sizes(obj_shape)[2]
            placement_z = desktop_center[2] + obj_height / 2.0 - obj_min[2]  # Place on surface

            # Apply translation
            obj_transformed = transform_shape(obj_shape, translation_matrix([base_pos[0], base_pos[1], placement_z]))

            # Add small random rotation around z-axis for natural look
            obj_center = compute_shape_center(obj_transformed)
            obj_transformed = transform_shape(
                obj_transformed, rotation_matrix(np.random.uniform(-0.3, 0.3), direction=(0, 0, 1), point=obj_center)
            )

            # Check minimum distance constraint
            if check_minimum_distance(obj_transformed, placed_objects, min_dist=0.15):
                placed_objects.append(obj_transformed)
                placed = True

            attempts += 1

        # If we couldn't place this object after max attempts, skip it
        if not placed:
            continue

    # Combine all shapes
    all_shapes = [desk_shape]
    all_shapes.extend(placed_objects)

    return concat_shapes(*all_shapes)


@register()
def table_with_random_objects() -> Shape:
    desk_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )
    return place_random_objects_on_desk(desk_shape)


@register()
def root_scene() -> Shape:
    return table_with_random_objects()
