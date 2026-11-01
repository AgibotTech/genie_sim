# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
"""
scene_name: open_trash_can_g2
description: A trash can placed at the center of a table with random rotation and position, plus a piece of trash on the table
"""


@register()
def trash_can() -> Shape:
    """
    Creates a trash can object loaded from assets library.
    """
    return library_call(
        "usd",
        oid="benchmark_trash_can_015",
        keywords=["trash_can", "black", "cylinder", "waste_container"],
    )


@register()
def trash_item() -> Shape:
    """
    Creates a random trash item loaded from assets library.
    Randomly selects from benchmark_trash_021, benchmark_trash_028, benchmark_trash_024
    """
    # Randomly select one trash item
    trash_options = ["benchmark_trash_021", "benchmark_trash_028", "benchmark_trash_024"]
    selected_trash = random.choice(trash_options)

    return library_call(
        "usd",
        oid=selected_trash,
        keywords=["trash", "waste", "litter", "on_table"],
    )


@register()
def place_trash_can_on_table(table_shape: Shape) -> Shape:
    """
    Places the trash can at a random position on the table's top surface with random rotation.
    Args:
        table_shape (Shape): The shape of the table object.
    Returns:
        Shape: Combined shape of table with trash can positioned on top.
    """
    # Get table information in world coordinates
    table_info = get_object_info(table_shape)

    # Get trash can subpart info to understand its dimensions
    trash_can_subpart_info = get_subpart_info(
        object_id="benchmark_trash_can_015", subpart_id="body"
    )

    # Calculate the top surface z-coordinate of the table
    table_top_z = table_info["center"][2] + trash_can_subpart_info["xyz_max"][2] + 0.4
    print(f"table_top_z: {table_top_z}")
    # Generate random position for trash can
    # x range: (-0.06, -0.04), y range: (-0.02, 0.02)
    trash_can_x = random.uniform(-0.06, 0.02)
    trash_can_y = random.uniform(-0.02, 0.02)
    trash_can_pos = np.array([trash_can_x, trash_can_y, table_top_z])

    # Generate random rotation around z-axis (175~185 degrees)
    # Convert to radians: 175° = 175 * π/180, 185° = 185 * π/180
    angle_degrees = random.uniform(175, 185)
    angle_radians = angle_degrees * math.pi / 180

    # Create the trash can shape
    trash_can_shape = library_call("trash_can")

    # Transform trash can to the calculated position
    trash_can_transformed = transform_shape(
        trash_can_shape, translation_matrix(trash_can_pos)
    )

    # Apply rotation around z-axis
    trash_can_center = compute_shape_center(trash_can_transformed)
    trash_can_transformed = transform_shape(
        trash_can_transformed,
        rotation_matrix(angle_radians, direction=(0, 0, 1), point=trash_can_center)
    )

    # Combine table and trash can shapes
    return concat_shapes(table_shape, trash_can_transformed)


@register()
def place_trash_on_table(table_shape: Shape, trash_can_shape: Shape) -> Shape:
    """
    Places a trash item on the table at a random position away from the trash can.
    Args:
        table_shape (Shape): The shape of the table object.
        trash_can_shape (Shape): The shape of the trash can object.
    Returns:
        Shape: Combined shape of table, trash can, and trash item.
    """
    # Get table information in world coordinates
    table_info = get_object_info(table_shape)

    # Get trash item subpart info to understand its dimensions
    trash_item_subpart_info = get_subpart_info(
        object_id=random.choice(["benchmark_trash_021", "benchmark_trash_028", "benchmark_trash_024"]),
        subpart_id="body"
    )

    # Calculate the top surface z-coordinate of the table
    table_top_z = table_info["center"][2] + trash_item_subpart_info["xyz_max"][2] + 0.4

    # Generate random position for trash item
    # Two possible regions:
    # Region 1: x(-0.2, 0.1), y(-0.2, -0.15)
    # Region 2: x(-0.2, 0.1), y(0.2, 0.15)
    region_choice = random.choice([1, 2])

    if region_choice == 1:
        trash_x = random.uniform(-0.2, 0.1)
        trash_y = random.uniform(-0.2, -0.15)
    else:
        trash_x = random.uniform(-0.2, 0.1)
        trash_y = random.uniform(0.15, 0.2)  # Note: swapped order for correct range

    trash_pos = np.array([trash_x, trash_y, table_top_z])

    # Create the trash item shape
    trash_shape = library_call("trash_item")

    # Transform trash to the calculated position
    trash_transformed = transform_shape(
        trash_shape, translation_matrix(trash_pos)
    )

    # Check for collision with trash can (simple bounding box check)
    trash_min = compute_shape_min(trash_transformed)
    trash_max = compute_shape_max(trash_transformed)
    trash_can_min = compute_shape_min(trash_can_shape)
    trash_can_max = compute_shape_max(trash_can_shape)

    # If there's overlap, adjust position slightly
    if not (trash_max[0] < trash_can_min[0] or trash_min[0] > trash_can_max[0] or
            trash_max[1] < trash_can_min[1] or trash_min[1] > trash_can_max[1]):
        # Shift trash position to avoid collision
        shift_direction = random.choice([-1, 1])
        trash_transformed = transform_shape(
            trash_transformed,
            translation_matrix((shift_direction * 0.05, 0, 0))
        )

    # Combine all shapes
    return concat_shapes(table_shape, trash_can_shape, trash_transformed)


@register()
def table_with_trash_can_and_trash() -> Shape:
    """
    Main scene composition: table with trash can at center and a piece of trash on the table.
    """
    # Load table at origin (0, 0, 0) - default ground plane is z=0
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )

    # Place trash can on the table (with random position and rotation)
    table_with_trash_can = place_trash_can_on_table(table_shape)

    # Extract trash can shape from combined shape for collision checking
    # We need to get the trash can part separately
    trash_can_shape = library_call("trash_can")
    trash_can_subpart_info = get_subpart_info(
        object_id="benchmark_trash_can_015", subpart_id="body"
    )
    table_info = get_object_info(table_shape)
    table_top_z = table_info["center"][2] + trash_can_subpart_info["xyz_max"][2] + 0.4
    trash_can_x = random.uniform(-0.06, 0.02)
    trash_can_y = random.uniform(-0.02, 0.02)
    trash_can_pos = np.array([trash_can_x, trash_can_y, table_top_z])
    trash_can_transformed = transform_shape(
        trash_can_shape, translation_matrix(trash_can_pos)
    )
    angle_degrees = random.uniform(175, 185)
    angle_radians = angle_degrees * math.pi / 180
    trash_can_center = compute_shape_center(trash_can_transformed)
    trash_can_transformed = transform_shape(
        trash_can_transformed,
        rotation_matrix(angle_radians, direction=(0, 0, 1), point=trash_can_center)
    )

    # Place trash item on the table
    final_scene = place_trash_on_table(table_shape, trash_can_transformed)

    return final_scene


@register()
def root_scene() -> Shape:
    """
    Root scene entry point.
    """
    return table_with_trash_can_and_trash()
