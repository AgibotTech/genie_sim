# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: table_with_beverages
description: Four distinct beverage cans placed in a 2x2 grid on table_000, with randomized types, uniform spacing (with minor noise), and random rotations.
"""


def get_desktop_placement_area(desk_shape: Shape) -> Tuple[P, P]:
    """
    Retrieves the top surface center and size of the desktop for object placement.

    Args:
        desk_shape (Shape): The shape of the desk.

    Returns:
        Tuple[P, P]: (desktop_center_at_top_z, desktop_size)
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    desktop_center = desk_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2] - 0.16
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


@register()
def single_beverage_can(beverage_type: str, keywords: List[str]) -> Shape:
    """
    Creates a single beverage can based on the given type.

    Args:
        beverage_type (str): The asset ID for the beverage can.
        keywords (List[str]): Descriptive keywords for the object.

    Returns:
        Shape: The beverage can shape.
    """
    return library_call("usd", oid=beverage_type, keywords=keywords)


@register()
def four_beverages_on_table() -> Shape:
    # Load the desk
    desk_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )
    desktop_center, desktop_size = get_desktop_placement_area(desk_shape)

    # Define the beverage mapping
    beverage_map = {
        "coca_cola": "benchmark_beverage_bottle_001",
        "sprite": "benchmark_beverage_bottle_002",
        "fanta": "benchmark_beverage_bottle_003",
        "wang_lao_ji": "benchmark_beverage_bottle_029",
    }
    beverage_names = list(beverage_map.keys())
    random.shuffle(beverage_names)  # Randomize the order

    # Determine the uniform base spacing within the specified range

    # grid_positions = [
    #     (-half_spacing_x, -half_spacing_y),
    #     (-half_spacing_x, half_spacing_y),
    #     (half_spacing_x, -half_spacing_y),
    #     (half_spacing_x, half_spacing_y),
    # ]
    base_position = [0.0, 0.0]
    grid_positions = []
    for i in range(4):
        base_spacing_x = np.random.uniform(0.15, 0.18)
        base_spacing_y = np.random.uniform(0.15, 0.18)
        # Apply small random noise to each instance's spacing (as per requirement, but keep grid structure)
        # We'll apply the noise when calculating each can's position relative to the grid origin.

        # The 2x2 grid layout: positions relative to the grid center
        # We define them in a way that the center of the 2x2 block is at (0,0)
        # So the four positions are: (-s/2, -s/2), (-s/2, s/2), (s/2, -s/2), (s/2, s/2)
        # where s is the distance between centers in one row/column.
        half_spacing_x = base_spacing_x / 2.0
        half_spacing_y = base_spacing_y / 2.0
        if i == 0:
            this_position = (base_position[0] - half_spacing_x, base_position[1] - half_spacing_y)
        elif i == 1:
            this_position = (base_position[0] - half_spacing_x, base_position[1] + half_spacing_y)
        elif i == 2:
            this_position = (base_position[0] + half_spacing_x, base_position[1] - half_spacing_y)
        elif i == 3:
            this_position = (base_position[0] + half_spacing_x, base_position[1] + half_spacing_y)
        grid_positions.append(this_position)

    all_cans = []
    for idx, name in enumerate(beverage_names):
        asset_id = beverage_map[name]
        # Get the can to compute its size for correct placement
        temp_can = library_call("usd", oid=asset_id, keywords=[name, "beverage", "can", "drink"])
        can_size = compute_shape_sizes(temp_can)
        can_height = can_size[2]

        # Add small random noise to the position as allowed
        noise_x = np.random.uniform(-0.005, 0.005)
        noise_y = np.random.uniform(-0.005, 0.005)
        grid_x, grid_y = grid_positions[idx]
        final_x = desktop_center[0] + grid_x + noise_x
        final_y = desktop_center[1] + grid_y + noise_y
        final_z = desktop_center[2] + can_height / 2.0  # Place on top of the desk

        # Create the final can with its specific keywords
        can_shape = library_call("usd", oid=asset_id, keywords=[name, "beverage", "can", "drink"])

        # Translate to the final position
        can_shape = transform_shape(can_shape, translation_matrix([final_x, final_y, final_z]))

        # Apply a random rotation around the Z-axis
        can_center = compute_shape_center(can_shape)
        random_angle = np.random.uniform(0, 2 * math.pi)
        can_shape = transform_shape(
            can_shape,
            rotation_matrix(random_angle, direction=(0, 0, 1), point=can_center),
        )

        all_cans.append(can_shape)

    return concat_shapes(desk_shape, *all_cans)


@register()
def root_scene() -> Shape:
    return four_beverages_on_table()
