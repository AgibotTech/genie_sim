# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: table_with_letters
description: A table with 3 uppercase letters placed along y-axis at fixed positions and 3 corresponding lowercase letters randomly placed with maximum separation.
"""


def get_desktop_top_surface(desk_shape: Shape) -> Tuple[P, P]:
    """
    Get the top surface position and size of the desktop.
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    desktop_center = desk_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


@register()
def place_uppercase_letters(desk_shape: Shape) -> Shape:
    """
    Place 3 uppercase letters along y-axis at fixed y positions: -0.15, 0, 0.15
    All uppercase letters have the same x value within (0.07, 0.17)
    """
    desktop_pos, desktop_size = get_desktop_top_surface(desk_shape)

    # Select 3 unique uppercase letters
    uppercase_indices = random.sample(range(26), 3)
    uppercase_letters = []

    # Fixed y positions
    y_positions = [-0.15, 0.0, 0.15]

    # Same x value for all uppercase letters
    x_pos = random.uniform(0.07, 0.17)

    for i, idx in enumerate(uppercase_indices):
        letter_id = f"benchmark_letter_{26 + idx:03d}"
        letter_char = chr(ord('A') + idx)

        y_pos = y_positions[i]

        letter_shape = library_call(
            "usd",
            oid=letter_id,
            keywords=[f"uppercase_{letter_char}", "letter", "uppercase", letter_char, "placed_along_y_axis"]
        )

        letter_min = compute_shape_min(letter_shape)
        letter_height = -letter_min[2]

        letter_transformed = transform_shape(
            letter_shape,
            translation_matrix([x_pos, y_pos, desktop_pos[2] + letter_height])
        )
        uppercase_letters.append(letter_transformed)

    return concat_shapes(*uppercase_letters)


@register()
def place_lowercase_letters(desk_shape: Shape) -> Shape:
    """
    Place 3 lowercase letters that correspond exactly to the uppercase letters.
    Letters are placed to maximize distance from each other in the x range (-0.15, 0).
    """
    desktop_pos, desktop_size = get_desktop_top_surface(desk_shape)

    # Get the same indices used for uppercase letters
    random_state = random.getstate()
    lowercase_indices = random.sample(range(26), 3)

    lowercase_letters = []
    placed_positions = []

    # Define the placement area for lowercase letters
    x_min, x_max = -0.15, 0.0
    y_min, y_max = -0.2, 0.2

    for idx in lowercase_indices:
        letter_id = f"benchmark_letter_{52 + idx:03d}"
        letter_char = chr(ord('a') + idx)

        # Find position that maximizes minimum distance to existing letters
        best_x, best_y = None, None
        max_min_distance = -1

        # Sample many candidate positions and pick the best one
        num_candidates = 200
        for _ in range(num_candidates):
            candidate_x = random.uniform(x_min, x_max)
            candidate_y = random.uniform(y_min, y_max)

            # Calculate minimum distance to all already placed letters
            min_dist = float('inf')
            for placed_x, placed_y in placed_positions:
                dist = ((candidate_x - placed_x) ** 2 + (candidate_y - placed_y) ** 2) ** 0.5
                min_dist = min(min_dist, dist)

            # If no letters placed yet, use a large distance
            if len(placed_positions) == 0:
                min_dist = 1.0

            if min_dist > max_min_distance:
                max_min_distance = min_dist
                best_x, best_y = candidate_x, candidate_y

        # Use the best position found
        if best_x is None or best_y is None:
            best_x = random.uniform(x_min, x_max)
            best_y = random.uniform(y_min, y_max)

        placed_positions.append((best_x, best_y))

        letter_shape = library_call(
            "usd",
            oid=letter_id,
            keywords=[f"lowercase_{letter_char}", "letter", "lowercase", letter_char, "corresponding_to_uppercase"]
        )

        letter_min = compute_shape_min(letter_shape)
        letter_height = -letter_min[2]

        letter_transformed = transform_shape(
            letter_shape,
            translation_matrix([best_x, best_y, desktop_pos[2] + letter_height])
        )
        lowercase_letters.append(letter_transformed)

    return concat_shapes(*lowercase_letters)


@register()
def table_with_letters() -> Shape:
    desk_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )

    # Save random state to ensure consistency between uppercase and lowercase
    original_state = random.getstate()

    uppercase_shape = library_call("place_uppercase_letters", desk_shape=desk_shape)

    # Restore state to get the same indices for lowercase
    random.setstate(original_state)
    lowercase_shape = library_call("place_lowercase_letters", desk_shape=desk_shape)

    return concat_shapes(desk_shape, uppercase_shape, lowercase_shape)


@register()
def root_scene() -> Shape:
    return table_with_letters()
