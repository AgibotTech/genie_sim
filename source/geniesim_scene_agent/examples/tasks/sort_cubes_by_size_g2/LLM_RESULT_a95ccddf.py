# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: sort_cubes_by_size_g2
description: A table with three cubic building blocks of different sizes (small, medium, large) placed randomly on the specified region x(-0.1, 0), y(-0.2, 0.2), with random rotation angles around the world z-axis and at least 4cm spacing between them.
"""


def get_table_top_surface(table_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the table's top surface center and its size.
    """
    table_info = get_object_info(table_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Table top surface center in world coordinates (x, y from table center, z from top of desktop)
    table_top_center = np.array(
        [table_info["center"][0], table_info["center"][1], table_info["center"][2] + desktop_subpart_info["xyz_max"][2]]
    )
    table_top_size = desktop_subpart_info["size"]

    return table_top_center, table_top_size


@register()
def small_cube() -> Shape:
    """Small cubic building block (0.02m) with interaction."""
    return library_call(
        "usd",
        oid="benchmark_building_blocks_079",
        keywords=["small_cube", "green", "cube", "building_block", "interaction", "up_axis_y"],
    )


@register()
def medium_cube(color: str = "red") -> Shape:
    """Medium cubic building block (~0.03m) with interaction. Color options: red, blue, yellow, reddish_brown."""
    oid_map = {
        "blue": "benchmark_building_blocks_076",
        "reddish_brown": "benchmark_building_blocks_019",
    }
    color = np.random.choice(["blue", "reddish_brown"])
    oid = oid_map.get(color, "benchmark_building_blocks_076")  # default to blue
    return library_call(
        "usd", oid=oid, keywords=[f"medium_cube_{color}", color, "cube", "building_block", "interaction", "up_axis_y"]
    )


@register()
def large_cube(color: str = "red") -> Shape:
    """Large cubic building block (0.04m or 0.05m) with interaction. Color options: red, blue."""
    oid_map = {"red": "benchmark_building_blocks_073", "blue": "benchmark_building_blocks_011"}
    color = np.random.choice(["red", "blue"])
    oid = oid_map.get(color, "benchmark_building_blocks_073")  # default to red
    return library_call(
        "usd", oid=oid, keywords=[f"large_cube_{color}", color, "cube", "building_block", "interaction", "up_axis_y"]
    )


@register()
def place_cubes_on_table(table_shape: Shape) -> Shape:
    """
    Places small, medium, and large cubes on the table within the specified region x(-0.1, 0), y(-0.2, 0.2),
    with random positions and random rotation angles around the world z-axis, ensuring at least 4cm spacing.
    """
    # Get table top info
    table_top_center, table_top_size = get_table_top_surface(table_shape)

    # Define placement region boundaries (absolute world coordinates as per user request)
    x_min, x_max = -0.1, 0.0
    y_min, y_max = -0.2, 0.2

    # Create cubes and get their sizes
    small = library_call("small_cube")
    medium = library_call("medium_cube", color="blue")
    large = library_call("large_cube", color="red")

    small_size = compute_shape_sizes(small)
    medium_size = compute_shape_sizes(medium)
    large_size = compute_shape_sizes(large)

    # Minimum spacing required (4cm = 0.04m)
    min_spacing = 0.04

    # Helper function to check if a position is valid (within bounds and spaced from others)
    def is_valid_position(pos: P, existing_positions: List[Tuple[P, P]], obj_size: P) -> bool:
        # Check bounds: ensure object doesn't exceed placement region
        if not (x_min + obj_size[0] / 2 <= pos[0] <= x_max - obj_size[0] / 2):
            return False
        if not (y_min + obj_size[1] / 2 <= pos[1] <= y_max - obj_size[1] / 2):
            return False
        # Check spacing with existing objects
        for existing_pos, existing_size in existing_positions:
            dx = abs(pos[0] - existing_pos[0])
            dy = abs(pos[1] - existing_pos[1])
            # Minimum distance required is half of each object's size plus spacing
            min_dist_x = (obj_size[0] + existing_size[0]) / 2 + min_spacing
            min_dist_y = (obj_size[1] + existing_size[1]) / 2 + min_spacing
            if dx < min_dist_x or dy < min_dist_y:
                return False
        return True

    # Place cubes with retry logic and increased max attempts for tighter space
    placed_positions = []
    cubes = [("large", large, large_size), ("medium", medium, medium_size), ("small", small, small_size)]
    placed_shapes = []
    max_attempts = 500  # Increased attempts for tighter space

    for name, cube, size in cubes:
        placed = False
        for _ in range(max_attempts):
            # Generate random position within bounds
            x = np.random.uniform(x_min + size[0] / 2, x_max - size[0] / 2)
            y = np.random.uniform(y_min + size[1] / 2, y_max - size[1] / 2)
            pos_2d = [x, y, 0]

            if is_valid_position(pos_2d, placed_positions, size):
                # Place at table top height
                world_pos = [x, y, table_top_center[2]]
                # Apply random rotation around world z-axis (0,0,1)
                angle = np.random.uniform(0, 2 * np.pi)
                cube_transformed = transform_shape(cube, translation_matrix(world_pos))
                cube_center = compute_shape_center(cube_transformed)
                cube_rotated = transform_shape(
                    cube_transformed, rotation_matrix(angle, direction=(0, 0, 1), point=cube_center)
                )

                placed_shapes.append(cube_rotated)
                placed_positions.append((pos_2d, size))
                placed = True
                break

        if not placed:
            # Fallback: place in a line if random placement fails
            if len(placed_positions) == 0:
                # First cube: place at center
                x = (x_min + x_max) / 2
                y = (y_min + y_max) / 2
            elif len(placed_positions) == 1:
                # Second cube: offset along x
                x = placed_positions[0][0][0] + (placed_positions[0][1][0] + size[0]) / 2 + min_spacing + 0.01
                y = (y_min + y_max) / 2
                if x > x_max - size[0] / 2:
                    x = placed_positions[0][0][0] - (placed_positions[0][1][0] + size[0]) / 2 - min_spacing - 0.01
            else:
                # Third cube: offset along y
                x = (x_min + x_max) / 2
                y = placed_positions[0][0][1] + (placed_positions[0][1][1] + size[1]) / 2 + min_spacing + 0.01
                if y > y_max - size[1] / 2:
                    y = placed_positions[0][0][1] - (placed_positions[0][1][1] + size[1]) / 2 - min_spacing - 0.01

            world_pos = [x, y, table_top_center[2]]
            angle = np.random.uniform(0, 2 * np.pi)
            cube_transformed = transform_shape(cube, translation_matrix(world_pos))
            cube_center = compute_shape_center(cube_transformed)
            cube_rotated = transform_shape(
                cube_transformed, rotation_matrix(angle, direction=(0, 0, 1), point=cube_center)
            )
            placed_shapes.append(cube_rotated)
            placed_positions.append(([x, y, 0], size))

    return concat_shapes(table_shape, *placed_shapes)


@register()
def sort_cubes_by_size_g2() -> Shape:
    # Load the table
    table_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )
    return place_cubes_on_table(table_shape)


@register()
def root_scene() -> Shape:
    return sort_cubes_by_size_g2()
