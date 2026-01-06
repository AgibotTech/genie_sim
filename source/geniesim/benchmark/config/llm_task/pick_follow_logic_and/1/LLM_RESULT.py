# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np


def find_desktop_surface(table_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the desktop's top surface center and its size.

    Args:
        table_shape (Shape): The shape of the table object.

    Returns:
        Tuple[P, P]: The center position of the top surface and its dimensions.
    """
    table_info = get_object_info(table_shape)
    # table_000 has a subpart named 'desktop'
    desktop_subpart = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # The desktop center in world coordinates (XY plane)
    desktop_center_xy = table_info["center"][:2] + desktop_subpart["center"][:2]

    # The top surface z-coordinate of the desktop in world coordinates
    desktop_top_z = table_info["center"][2] + desktop_subpart["xyz_max"][2]

    # The final desktop surface position (center_x, center_y, top_z)
    surface_pos = np.array([desktop_center_xy[0], desktop_center_xy[1], desktop_top_z])
    surface_size = desktop_subpart["size"]

    return surface_pos, surface_size


@register()
def place_random_objects_on_table(table_shape: Shape) -> Shape:
    """
    Selects 5 objects from the 8 candidates and places them randomly on the table.
    Ensures a minimum distance of 20cm between objects and avoids the edges.
    """
    surface_pos, surface_size = find_desktop_surface(table_shape)

    # The 8 candidate assets identified from the library
    asset_pool = [
        ("sprite", "sprite", ["beverage", "green", "can"]),
        ("mouse", "mouse", ["peripheral", "gray", "electronics"]),
        ("papercup", "paper_cup", ["drinkware", "white", "disposable"]),
        ("benchmark_apple_003", "red_apple", ["fruit", "red", "round"]),
        ("sponge", "cleaning_sponge", ["tool", "yellow_green", "rectangular"]),
        ("benchmark_building_blocks_074", "red_block", ["toy", "red", "cube", "0.05m"]),
        ("benchmark_building_blocks_078", "blue_block", ["toy", "blue", "cube", "0.05m"]),
        ("benchmark_building_blocks_086", "yellow_block", ["toy", "yellow", "cube", "0.05m"]),
        ("benchmark_pen_005", "green_pen", ["writing_tool", "green", "cylinder"]),
    ]

    # Randomly select 5 unique objects from the pool
    selected_indices = np.random.choice(len(asset_pool), 5, replace=False)

    # Define placement constraints
    min_distance = 0.15  # 15cm minimum distance between objects
    edge_margin = 0.15  # 15cm margin from the table edge

    # Calculate the valid placement range on the desktop
    range_x = (surface_size[0] / 2.0) - edge_margin
    range_y = (surface_size[1] / 2.0) - edge_margin

    placed_positions = []
    scene_shapes = [table_shape]

    for i in range(5):
        asset_id, name, tags = asset_pool[selected_indices[i]]

        # Attempt to find a valid random position
        valid_pos = False
        attempts = 0
        while not valid_pos and attempts < 200:
            # Generate random local offset within the safe zone
            local_x = np.random.uniform(-range_x, range_x)
            local_y = np.random.uniform(-range_y, range_y)
            candidate_pos = np.array([local_x, local_y, 0.0])

            # Check distance from all previously placed objects
            if all(np.linalg.norm(candidate_pos - p) >= min_distance for p in placed_positions):
                valid_pos = True
                placed_positions.append(candidate_pos)
            attempts += 1

        if not valid_pos:
            # If we can't find a spot after 100 tries, skip this object (unlikely for 5 objects)
            continue

        # Determine Position Tag (left/right) based on y-coordinate
        # In our coordinate system, +y is left, -y is right.
        if candidate_pos[1] > 0.0:
            pos_tag = "left"
        else:
            pos_tag = "right"

        keywords = [name] + tags + [pos_tag, "on_table", "random_placement"]

        # Load the object (usd function returns shape with origin at its bottom)
        obj_shape = library_call("usd", oid=asset_id, keywords=keywords)

        # Calculate world position by adding local offset to the surface center
        world_pos = surface_pos + candidate_pos

        # Apply translation to place the object on the table
        obj_shape = transform_shape(obj_shape, translation_matrix(world_pos))

        # Add a small random rotation around Z for realism
        obj_center = compute_shape_center(obj_shape)
        obj_shape = transform_shape(
            obj_shape,
            rotation_matrix(np.random.uniform(0, 2 * math.pi), direction=(0, 0, 1), point=obj_center),
        )

        scene_shapes.append(obj_shape)

    return concat_shapes(*scene_shapes)


@register()
def table_scene() -> Shape:
    """
    Initializes the table and triggers object placement.
    """
    # Load the table at the origin (0, 0, 0)
    table_shape = library_call("usd", oid="table_000", keywords=["minimalist_table", "white", "furniture", "center"])

    return library_call("place_random_objects_on_table", table_shape=table_shape)


@register()
def root_scene() -> Shape:
    """
    Root function to generate the entire scene.
    """
    return library_call("table_scene")
