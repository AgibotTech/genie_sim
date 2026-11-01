# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
"""
scene_name: table_with_random_letters
description: A table with three randomly selected letters placed within a specific region, avoiding a central sub-region, and each pair of letters is at least 0.05m apart.
"""
def get_tabletop_placement_area(table_shape: Shape) -> Tuple[P, P]:
    """
    Get the top surface center and size of the table's desktop in world coordinates.
    """
    table_info = get_object_info(table_shape)
    desktop_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_center_world = table_info["center"] + desktop_info["center"]
    desktop_top_z = table_info["center"][2] + desktop_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z])
    desktop_size = desktop_info["size"]
    return desktop_pos, desktop_size

def positions_too_close(pos1: P, pos2: P, min_dist: float = 0.05) -> bool:
    """
    Check if two positions are closer than min_dist in XY plane.
    """
    dist = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
    return dist < min_dist

@register()
def place_single_letter(letter_oid: str, letter_char: str, table_top_center: P, existing_positions: List[P]) -> Shape:
    """
    Place a single letter on the table within the allowed region without additional rotation,
    ensuring it is at least 0.05m away from all existing letters.
    """
    letter_shape = library_call(
        "usd",
        oid=letter_oid,
        keywords=[f"letter_{letter_char}", "alphabet", "plastic", "educational"],
    )
    # Compute letter dimensions for collision-aware placement
    letter_min = compute_shape_min(letter_shape)
    letter_max = compute_shape_max(letter_shape)
    letter_size = letter_max - letter_min
    # Define placement bounds relative to table top center
    x_min_allowed, x_max_allowed = -0.2, 0.1
    y_min_allowed, y_max_allowed = -0.2, 0.2
    # Forbidden zone: x in (0, 0.07), y in (-0.07, 0.07)
    forbidden_x_min, forbidden_x_max = 0.0, 0.07
    forbidden_y_min, forbidden_y_max = -0.07, 0.07
    min_letter_spacing = 0.05
    max_attempts = 1000  # Increased attempts due to stricter constraints
    for _ in range(max_attempts):
        x_offset = random.uniform(x_min_allowed, x_max_allowed)
        y_offset = random.uniform(y_min_allowed, y_max_allowed)
        candidate_pos = [table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]]
        # Check if the sampled point is in the forbidden zone
        if (forbidden_x_min < x_offset < forbidden_x_max) and (forbidden_y_min < y_offset < forbidden_y_max):
            continue
        # Ensure the entire letter fits within the broader table area
        if (abs(x_offset) + letter_size[0] / 2 > 0.25) or (abs(y_offset) + letter_size[1] / 2 > 0.25):
            continue
        # Check distance to all existing positions
        too_close = False
        for existing_pos in existing_positions:
            if positions_too_close(candidate_pos, existing_pos, min_dist=min_letter_spacing):
                too_close = True
                break
        if too_close:
            continue
        # Valid position found
        break
    else:
        # Fallback: use a safe default position if sampling fails
        # Try to place in corners with sufficient spacing
        fallback_positions = [
            [-0.18, 0.18],
            [-0.18, -0.18],
            [0.08, 0.18]
        ]
        candidate_pos = None
        for fx, fy in fallback_positions:
            fallback_pos = [table_top_center[0] + fx, table_top_center[1] + fy, table_top_center[2]]
            too_close = False
            for existing_pos in existing_positions:
                if positions_too_close(fallback_pos, existing_pos, min_dist=min_letter_spacing):
                    too_close = True
                    break
            if not too_close:
                candidate_pos = fallback_pos
                x_offset, y_offset = fx, fy
                break
        if candidate_pos is None:
            # Last resort: use first fallback even if slightly close
            x_offset, y_offset = -0.15, 0.15
            candidate_pos = [table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]]
    # Translate to the final position on the table
    placed_letter = transform_shape(
        letter_shape,
        translation_matrix([table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]])
    )
    return placed_letter

@register()
def table_with_random_letters() -> Shape:
    # Load the table
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )
    # Get table top info for placement
    table_top_center, _ = get_tabletop_placement_area(table_shape)
    # Define available letters
    letter_map = {
        'A': 'benchmark_letter_026',
        'B': 'benchmark_letter_027',
        'C': 'benchmark_letter_028',
        'D': 'benchmark_letter_029',
        'E': 'benchmark_letter_030',
        'F': 'benchmark_letter_031',
        'G': 'benchmark_letter_032',
        'H': 'benchmark_letter_033',
        'I': 'benchmark_letter_034',
        'J': 'benchmark_letter_035',
        'K': 'benchmark_letter_036',
        'L': 'benchmark_letter_037',
        'M': 'benchmark_letter_038',
        'N': 'benchmark_letter_039',
        'O': 'benchmark_letter_040',
        'P': 'benchmark_letter_041',
        'Q': 'benchmark_letter_042',
        'R': 'benchmark_letter_043',
        'S': 'benchmark_letter_044',
        'T': 'benchmark_letter_045',
        'U': 'benchmark_letter_046',
        'V': 'benchmark_letter_047',
        'W': 'benchmark_letter_048',
        'X': 'benchmark_letter_049',
        'Y': 'benchmark_letter_050',
        'Z': 'benchmark_letter_051',
    }
    # Randomly select 3 distinct letters
    selected_chars = random.sample(list(letter_map.keys()), 3)
    # Place each selected letter with spacing constraint
    letter_shapes = []
    placed_positions = []  # Track positions of already placed letters
    for char in selected_chars:
        oid = letter_map[char]
        letter_shape = library_call(
            "place_single_letter",
            letter_oid=oid,
            letter_char=char,
            table_top_center=table_top_center,
            existing_positions=placed_positions,
        )
        # Record the position of this letter for future spacing checks
        letter_info = get_object_info(letter_shape)
        placed_positions.append(letter_info["center"])
        letter_shapes.append(letter_shape)
    # Combine table and letters
    return concat_shapes(table_shape, *letter_shapes)

@register()
def root_scene() -> Shape:
    return table_with_random_letters()
