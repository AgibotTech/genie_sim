# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: desk_with_storage_box_and_toys
description: A desk with a randomly selected storage box containing 0-3 toys, and 1-3 additional toys placed on the rest of the desktop. Total toys <= 4, with at least one outside the box. Toys are non-repeating, spaced at least 0.08m apart, and avoid a 5cm margin around the box.
"""


def find_desktop_top_surface(desk_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the desktop's top surface center and its size.

    Args:
        desk_shape (Shape): The shape of the desk object.

    Returns:
        Tuple[P, P]: A tuple containing the desktop's top surface center position and its size.
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # The desktop center in world coordinates
    desktop_center = desk_info["center"] + desktop_subpart_info["center"]

    # The top surface z-coordinate of the desktop in world coordinates
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    # The final desktop position (center_x, center_y, top_z)
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


def is_position_valid(candidate_pos: P, existing_positions: List[P], min_dist: float = 0.08) -> bool:
    """
    Check if a candidate position is valid given existing positions and a minimum distance.
    """
    for pos in existing_positions:
        if np.linalg.norm(np.array(candidate_pos[:2]) - np.array(pos[:2])) < min_dist:
            return False
    return True


def sample_position_in_region(x_range: Tuple[float, float], y_range: Tuple[float, float]) -> P:
    """
    Sample a random (x, y) position within the specified ranges, z=0 for now (will be set later).
    """
    x = np.random.uniform(x_range[0], x_range[1])
    y = np.random.uniform(y_range[0], y_range[1])
    return np.array([x, y, 0.0])


@register()
def create_toy(toy_oid: str, keywords: List[str]) -> Shape:
    """
    Creates a single toy with a small random rotation around Z-axis.
    """
    toy_shape = library_call("usd", oid=toy_oid, keywords=keywords)
    toy_center = compute_shape_center(toy_shape)
    angle = np.random.uniform(-0.2, 0.2)  # approx +/- 11.5 degrees, more than enough for "random"
    return transform_shape(toy_shape, rotation_matrix(angle=angle, direction=(0, 0, 1), point=toy_center))


@register()
def place_toys_on_desktop(desk_shape: Shape, storage_box_shape: Shape, num_toys_outside: int) -> Shape:
    """
    Places a specified number of toys on the desktop outside the storage box, respecting constraints.
    """
    # Get desktop info
    desktop_pos, desktop_size = find_desktop_top_surface(desk_shape)
    desktop_min = desktop_pos - desktop_size / 2.0
    desktop_max = desktop_pos + desktop_size / 2.0

    # Get storage box info to define exclusion zone
    box_info = get_object_info(storage_box_shape)
    box_center = box_info["center"]
    box_size = box_info["size"]
    margin = 0.05  # 5cm exclusion zone
    box_exclusion_min = box_center - box_size / 2.0 - margin
    box_exclusion_max = box_center + box_size / 2.0 + margin

    # Define the allowed region for outside toys: x(-0.25, 0.0), y(-0.25, 0.25) relative to desk center
    # Since `desktop_pos` is the absolute center of the desktop top, we add the offsets to it.
    allowed_x_min = desktop_pos[0] - 0.25
    allowed_x_max = desktop_pos[0] + 0.0
    allowed_y_min = desktop_pos[1] - 0.25
    allowed_y_max = desktop_pos[1] + 0.25

    # Toy asset list
    toy_candidates = [
        "benchmark_building_blocks_057",
        "benchmark_toy_car_018",
        "benchmark_toy_plane_029",
        "benchmark_building_blocks_029",
        "benchmark_building_blocks_028",
        "benchmark_rubik_cube_001",
        "benchmark_garage_kit_064",
        "benchmark_chess_001",
    ]
    selected_toys = random.sample(toy_candidates, k=len(toy_candidates))  # Shuffle the list

    placed_shapes = []
    placed_positions = []  # Keep track of (x, y, z) for distance checks
    toy_index = 0

    for _ in range(num_toys_outside):
        placed = False
        attempts = 0
        max_attempts = 100

        while not placed and attempts < max_attempts:
            attempts += 1
            # Sample a candidate position in the allowed region
            candidate_xy = sample_position_in_region((allowed_x_min, allowed_x_max), (allowed_y_min, allowed_y_max))
            candidate_pos = np.array([candidate_xy[0], candidate_xy[1], desktop_pos[2]])

            # Check if it's inside the box exclusion zone
            if (
                box_exclusion_min[0] <= candidate_pos[0] <= box_exclusion_max[0]
                and box_exclusion_min[1] <= candidate_pos[1] <= box_exclusion_max[1]
            ):
                continue

            # Check distance from already placed toys
            if not is_position_valid(candidate_pos, placed_positions):
                continue

            # If we get here, the position is valid
            toy_oid = selected_toys[toy_index]
            toy_index += 1
            toy_shape = library_call(
                "create_toy", toy_oid=toy_oid, keywords=[f"toy_{toy_index}", "small", "plaything", "randomly_placed"]
            )

            # Get toy size for correct vertical placement
            toy_min = compute_shape_min(toy_shape)
            toy_height = compute_shape_sizes(toy_shape)[2]
            # Place the toy so its bottom is on the desktop
            final_translation = translation_matrix(
                [
                    candidate_pos[0],
                    candidate_pos[1],
                    desktop_pos[2] - toy_min[2],  # toy_min[2] is negative, so this adds height
                ]
            )

            placed_shape = transform_shape(toy_shape, final_translation)
            placed_shapes.append(placed_shape)
            placed_positions.append(candidate_pos)
            placed = True

        if not placed:
            # If we can't find a spot, just break to avoid infinite loop
            break

    return concat_shapes(*placed_shapes)


@register()
def place_toys_in_storage_box(storage_box_shape: Shape, num_toys_inside: int) -> Shape:
    """
    Places a specified number of toys inside the storage box.
    """
    box_info = get_object_info(storage_box_shape)
    box_center = box_info["center"]
    box_size = box_info["size"]

    # Define the inner volume for placing toys, leave a small margin
    inner_margin = 0.01
    inner_x_range = (box_center[0] - box_size[0] / 2.0 + inner_margin, box_center[0] + box_size[0] / 2.0 - inner_margin)
    inner_y_range = (box_center[1] - box_size[1] / 2.0 + inner_margin, box_center[1] + box_size[1] / 2.0 - inner_margin)
    inner_z_base = box_center[2] - box_size[2] / 2.0 + inner_margin  # Start placing from the bottom

    toy_candidates = [
        "benchmark_building_blocks_057",
        "benchmark_toy_car_018",
        "benchmark_toy_plane_029",
        "benchmark_building_blocks_029",
        "benchmark_building_blocks_028",
        "benchmark_rubik_cube_001",
        "benchmark_garage_kit_064",
        "benchmark_chess_001",
    ]
    selected_toys = random.sample(toy_candidates, k=len(toy_candidates))

    placed_shapes = []
    current_z = inner_z_base
    toy_index = 0

    for i in range(num_toys_inside):
        toy_oid = selected_toys[toy_index]
        toy_index += 1
        toy_shape = library_call(
            "create_toy", toy_oid=toy_oid, keywords=[f"toy_in_box_{i+1}", "small", "plaything", "inside_storage"]
        )

        toy_size = compute_shape_sizes(toy_shape)
        # Center the toy in X and Y within the box
        tx = box_center[0]
        ty = box_center[1]
        tz = current_z + toy_size[2] / 2.0  # Place on top of previous or base
        current_z = tz + toy_size[2] / 2.0  # Update z for next toy

        placed_shape = transform_shape(toy_shape, translation_matrix([tx, ty, tz]))
        placed_shapes.append(placed_shape)

    return concat_shapes(*placed_shapes)


@register()
def table_with_storage_box_and_toys() -> Shape:
    # Load the desk
    desk_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )
    desktop_pos, _ = find_desktop_top_surface(desk_shape)

    # Select a random storage box
    storage_box_options = [
        "benchmark_storage_box_001",
        "benchmark_storage_box_004",
        "benchmark_storage_box_005",
        "benchmark_storage_box_003",
    ]
    chosen_box = random.choice(storage_box_options)
    storage_box_shape = library_call("usd", oid=chosen_box, keywords=["storage_container", "plastic", "box", "on_desk"])

    # Determine toy distribution
    # Total toys must be <= 4, and at least 1 outside.
    total_toys = random.randint(2, 4)  # Min 2 (1 in, 1 out), max 4
    num_toys_outside = random.randint(1, min(3, total_toys - 0))  # At least 1 outside, max 3 outside
    num_toys_inside = total_toys - num_toys_outside
    # Ensure inside count is between 0 and 3
    num_toys_inside = max(0, min(3, num_toys_inside))
    num_toys_outside = total_toys - num_toys_inside

    # Position the storage box on the desk within its specified region
    # Box region: x(-0.1, 0.05), y(-0.03, 0.03) relative to desk center
    box_tx = np.random.uniform(-0.1, 0.05)
    box_ty = np.random.uniform(-0.03, 0.03)
    box_tz = desktop_pos[2]  # Place directly on the desktop

    # Apply a small random rotation around Z-axis (up to 5 degrees = ~0.087 rad)
    box_angle = np.random.uniform(-0.087, 0.087)
    box_center_initial = compute_shape_center(storage_box_shape)
    storage_box_transformed = transform_shape(
        storage_box_shape, translation_matrix([desktop_pos[0] + box_tx, desktop_pos[1] + box_ty, box_tz])
    )
    box_center_final = compute_shape_center(storage_box_transformed)
    storage_box_transformed = transform_shape(
        storage_box_transformed, rotation_matrix(angle=box_angle, direction=(0, 0, 1), point=box_center_final)
    )

    # Create toys inside and outside
    toys_inside_shape = library_call(
        "place_toys_in_storage_box", storage_box_shape=storage_box_transformed, num_toys_inside=num_toys_inside
    )
    toys_outside_shape = library_call(
        "place_toys_on_desktop",
        desk_shape=desk_shape,
        storage_box_shape=storage_box_transformed,
        num_toys_outside=num_toys_outside,
    )

    # Combine all shapes
    return concat_shapes(desk_shape, storage_box_transformed, toys_inside_shape, toys_outside_shape)


@register()
def root_scene() -> Shape:
    return table_with_storage_box_and_toys()
