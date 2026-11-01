# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: sort_beverages_by_type_g2
description: A table with 3-5 canned beverages of exactly two types, randomly placed and rotated within specified bounds
"""


def find_desktop_top_surface(desk_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the desktop's top surface and its size.
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # The desktop center in world coordinates
    desktop_center = desk_info["center"] + desktop_subpart_info["center"]

    # The top surface z-coordinate of the desktop in world coordinates
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2] - 0.1

    # The final desktop position (center_x, center_y, top_z)
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])

    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


def generate_random_positions(
    num_items: int, x_range: Tuple[float, float], y_range: Tuple[float, float], min_distance: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Generate random positions within given ranges ensuring minimum distance between any two points.
    """
    positions = []
    max_attempts = 1000

    for i in range(num_items):
        attempt = 0
        while attempt < max_attempts:
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])

            # Check distance to all existing positions
            valid = True
            for existing_x, existing_y in positions:
                distance = ((x - existing_x) ** 2 + (y - existing_y) ** 2) ** 0.5
                if distance < min_distance:
                    valid = False
                    break

            if valid:
                positions.append((x, y))
                break
            attempt += 1

        if attempt >= max_attempts:
            # If we can't find a valid position, just use the last attempted one
            positions.append((x, y))

    return positions


@register()
def canned_beverage(oid: str, keywords: List[str]) -> Shape:
    """
    Create a single canned beverage with random rotation.
    """
    beverage_shape = library_call("usd", oid=oid, keywords=keywords)

    # Get the current center for rotation
    center = compute_shape_center(beverage_shape)

    # Apply random rotation around z-axis
    rotation_angle = random.uniform(0, 2 * math.pi)
    rotated_shape = transform_shape(beverage_shape, rotation_matrix(rotation_angle, direction=(0, 0, 1), point=center))

    return rotated_shape


@register()
def place_beverages_on_desktop(desk_shape: Shape) -> Shape:
    """
    Place 3-5 canned beverages of exactly two types on the desktop within specified bounds.
    """
    workspace, _ = find_desktop_top_surface(desk_shape)

    # Define the placement bounds relative to workspace center
    x_min, x_max = -0.05, 0.15
    y_min, y_max = -0.25, 0.25

    # Available canned beverage types (only benchmark assets with interaction)
    beverage_types = [
        ("benchmark_beverage_bottle_001", ["coca_cola_can", "red", "aluminum_can", "carbonated_drink"]),
        ("benchmark_beverage_bottle_002", ["sprite_can", "green", "aluminum_can", "carbonated_drink"]),
        ("benchmark_beverage_bottle_003", ["fanta_can", "orange", "aluminum_can", "carbonated_drink"]),
        ("benchmark_beverage_bottle_029", ["wang_lao_ji_can", "red", "aluminum_can", "herbal_tea"]),
    ]

    # Select exactly two types randomly
    selected_types = random.sample(beverage_types, 2)

    # Determine number of beverages (3-5 total)
    num_beverages = random.randint(3, 5)

    # Distribute beverages between the two types
    num_type1 = random.randint(1, num_beverages - 1)
    num_type2 = num_beverages - num_type1

    # Generate random positions ensuring minimum distance
    positions = generate_random_positions(num_beverages, (x_min, x_max), (y_min, y_max), min_distance=0.1)

    beverage_shapes = []

    # Create beverages of first type
    for i in range(num_type1):
        oid, keywords = selected_types[0]
        beverage = library_call("canned_beverage", oid=oid, keywords=keywords)
        pos_x, pos_y = positions[i]
        beverage_placed = transform_shape(
            beverage, translation_matrix([workspace[0] + pos_x, workspace[1] + pos_y, workspace[2]])
        )
        beverage_shapes.append(beverage_placed)

    # Create beverages of second type
    for i in range(num_type2):
        oid, keywords = selected_types[1]
        beverage = library_call("canned_beverage", oid=oid, keywords=keywords)
        pos_x, pos_y = positions[num_type1 + i]
        beverage_placed = transform_shape(
            beverage, translation_matrix([workspace[0] + pos_x, workspace[1] + pos_y, workspace[2]])
        )
        beverage_shapes.append(beverage_placed)

    return concat_shapes(desk_shape, *beverage_shapes)


@register()
def sort_beverages_by_type_g2() -> Shape:
    """
    Main scene function: table with randomly placed and rotated canned beverages.
    """
    desk_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "rectangular_desk"]
    )
    return place_beverages_on_desktop(desk_shape)


@register()
def root_scene() -> Shape:
    return sort_beverages_by_type_g2()
