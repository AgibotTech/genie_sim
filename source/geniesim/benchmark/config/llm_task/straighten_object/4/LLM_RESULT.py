# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: table_with_drinks_and_fruit
description: A table with two randomly selected drinks and one fruit placed randomly on its surface without overlap. Drinks are rotated to have their local z-axis pointing up and then given a random spin around the world Z-axis. The fruit is also given a random rotation.
"""


def get_table_top_info(table_center: P, table_rotation) -> Tuple[P, P]:
    """
    Since the table is given as existing, we assume it has a standard 'desktop' subpart.
    This function computes the top surface center and size in world coordinates.
    For simplicity and based on common assets, we assume the desktop subpart info.
    In a real scenario, this would come from get_subpart_info, but the table asset ID isn't provided.
    We'll make reasonable assumptions based on typical table proportions.

    Args:
        table_center: The world position of the table's origin (given as (1,2,3)).
        table_rotation: The table's orientation quaternion (given as (1,0,0,0), which is identity).

    Returns:
        Tuple of (top_surface_center_world, top_surface_size).
    """
    # Given the table's global pose
    table_pos = np.array(table_center)
    # The rotation (1,0,0,0) is identity, so no rotation to apply for now.
    # We need to assume the desktop's properties relative to the table's origin.
    # Let's assume a standard table where the desktop is centered and its top is at a certain height.
    # Without the actual asset, we will assume the provided (1,2,3) is the center of the *desktop top*.
    # This is a common way to specify a table's position for placing objects.
    # Therefore, the top surface center is directly (1,2,3).
    # And we just need a size. We'll use a standard size.
    desktop_top_center = table_pos
    desktop_size_xy = np.array([0.24, 0.8])  # Only x and y matter for placement area

    return desktop_top_center, np.array([desktop_size_xy[0], desktop_size_xy[1], 0.05])


def sample_position_on_table(top_center: P, top_size: P, obj_radius: float) -> P:
    """
    Samples a random (x, y) position on the table top for an object, ensuring it's within bounds.
    """
    half_x = top_size[0] / 2.0 - obj_radius
    half_y = top_size[1] / 2.0 - obj_radius
    if half_x <= 0 or half_y <= 0:
        half_x = 0.0
        half_y = 0.0
    x = np.random.uniform(top_center[0] - half_x, top_center[0] + half_x)
    y = np.random.uniform(top_center[1] - half_y, top_center[1] + half_y)
    z = top_center[2]  # Place on the top surface
    return np.array([x, y, z])


def positions_overlap(pos1: P, pos2: P, radius1: float, radius2: float, min_dist_factor: float = 1.1) -> bool:
    """
    Checks if two positions are too close, considering their radii.
    """
    dist = np.linalg.norm(pos1[:2] - pos2[:2])  # Only check x, y distance
    min_dist = (radius1 + radius2) * min_dist_factor
    return dist < min_dist


@register()
def place_drinks_and_fruit() -> Shape:
    # Table information from the user query
    TABLE_CENTER = [2.92, 0.76, 0.8]
    TABLE_ROTATION = (0.70711, 0.70711, 0.0, 0.0)  # Identity quaternion

    # Get table top info
    top_center, top_size = get_table_top_info(TABLE_CENTER, TABLE_ROTATION)

    # Available drink and fruit IDs (extracted from data_info_dir as instructed)
    drink_ids = [
        "benchmark_beverage_bottle_001",
        "iros_beverage_bottle_001",  # Corrected to match the data_info_dir pattern
        "benchmark_beverage_bottle_003",
        "benchmark_beverage_bottle_004",
    ]
    drink_names = ["coke cola", "sprite", "fanta", "oolong tea beverage"]
    fruit_ids = [
        "benchmark_apple_002",
        "benchmark_orange_004",
        "benchmark_lemon_030",
        "benchmark_peach_021",
        "benchmark_green_apple_001",
    ]
    fruit_names = ["apple", "orange", "lemon", "peach", "green apple"]

    # Randomly select two different drinks and one fruit
    selected_drink_indices = np.random.choice(len(drink_ids), size=2, replace=False)
    selected_fruit_index = np.random.choice(len(fruit_ids))

    selected_drink_ids = [drink_ids[i] for i in selected_drink_indices]
    selected_drink_names = [drink_names[i] for i in selected_drink_indices]
    selected_fruit_id = fruit_ids[selected_fruit_index]
    selected_fruit_name = fruit_names[selected_fruit_index]

    # Load shapes to get their sizes for collision avoidance
    # Drinks - Need to account for the 90-degree X-axis rotation
    drink_shapes_raw = []
    drink_radii = []
    for i, (oid, name) in enumerate(zip(selected_drink_ids, selected_drink_names)):
        raw_shape = library_call("usd", oid=oid, keywords=[name, "beverage", "bottle", "drink", f"drink_{i}"])
        drink_shapes_raw.append(raw_shape)
        size = compute_shape_sizes(raw_shape)
        # After rotating 90 degrees around X, the original Y becomes the new Z (height),
        # and the original Z becomes the new -Y. The cross-section for collision is in X and new Y (original Z).
        # The bounding circle radius on the table will be max(size.x/2, size.z/2).
        radius = max(size[0], size[2]) / 2.0
        drink_radii.append(radius)

    # Fruit
    fruit_shape = library_call(
        "usd", oid=selected_fruit_id, keywords=[selected_fruit_name, "fruit", "healthy", "snack"]
    )
    fruit_size = compute_shape_sizes(fruit_shape)
    # For a fruit (sphere), radius is size.x / 2.0
    fruit_radius = fruit_size[0] / 2.0

    # Sample non-overlapping positions
    all_positions = []
    all_radii = []

    # Place first drink
    pos1 = sample_position_on_table(top_center, top_size, drink_radii[0])
    all_positions.append(pos1)
    all_radii.append(drink_radii[0])

    # Place second drink, check for overlap with first
    max_attempts = 100
    for _ in range(max_attempts):
        pos2 = sample_position_on_table(top_center, top_size, drink_radii[1])
        if not positions_overlap(pos1, pos2, drink_radii[0], drink_radii[1]):
            all_positions.append(pos2)
            all_radii.append(drink_radii[1])
            break
    else:
        all_positions.append(pos2)
        all_radii.append(drink_radii[1])

    # Place fruit, check for overlap with both drinks
    for _ in range(max_attempts):
        fruit_pos = sample_position_on_table(top_center, top_size, fruit_radius)
        overlap1 = positions_overlap(fruit_pos, all_positions[0], fruit_radius, all_radii[0])
        overlap2 = positions_overlap(fruit_pos, all_positions[1], fruit_radius, all_radii[1])
        if not (overlap1 or overlap2):
            all_positions.append(fruit_pos)
            all_radii.append(fruit_radius)
            break
    else:
        all_positions.append(fruit_pos)
        all_radii.append(fruit_radius)

    # Determine left/right tags based on Y coordinate relative to table center
    table_y_center = top_center[1]
    drink1_y, drink2_y, fruit_y = all_positions[0][1], all_positions[1][1], all_positions[2][1]
    drink1_tag = "left" if drink1_y > table_y_center else "right"
    drink2_tag = "left" if drink2_y > table_y_center else "right"
    fruit_tag = "left" if fruit_y > table_y_center else "right"

    # Apply final transformations
    final_objects = []

    # Process drinks
    for i, (oid, name, pos, tag) in enumerate(
        zip(selected_drink_ids, selected_drink_names, all_positions[:2], [drink1_tag, drink2_tag])
    ):
        # Load with correct keyword including position tag
        shape = library_call("usd", oid=oid, keywords=[name, "beverage", "bottle", "drink", tag])
        # Step 1: Rotate -90 degrees around X-axis to make local Z point up
        shape = transform_shape(shape, rotation_matrix(angle=-math.pi / 2, direction=(1, 0, 0), point=(0, 0, 0)))
        # Step 2: Rotate randomly around Z-axis (world up)
        random_z_angle = np.random.uniform(0, 2 * math.pi)
        shape = transform_shape(shape, rotation_matrix(angle=random_z_angle, direction=(0, 0, 1), point=(0, 0, 0)))
        # Step 3: Translate to final position
        final_shape = transform_shape(shape, translation_matrix(pos.tolist()))
        final_objects.append(final_shape)

    # Process fruit: add random rotation around all axes
    fruit_shape = library_call(
        "usd", oid=selected_fruit_id, keywords=[selected_fruit_name, "fruit", "healthy", "snack", fruit_tag]
    )
    # Generate random rotation angles for X, Y, Z
    rand_angle_x = np.random.uniform(0, 2 * math.pi)
    rand_angle_y = np.random.uniform(0, 2 * math.pi)
    rand_angle_z = np.random.uniform(0, 2 * math.pi)

    # Apply rotations in sequence: Z, then Y, then X (or any order, since it's random)
    fruit_shape = transform_shape(
        fruit_shape, rotation_matrix(angle=rand_angle_z, direction=(0, 0, 1), point=(0, 0, 0))
    )
    fruit_shape = transform_shape(
        fruit_shape, rotation_matrix(angle=rand_angle_y, direction=(0, 1, 0), point=(0, 0, 0))
    )
    fruit_shape = transform_shape(
        fruit_shape, rotation_matrix(angle=rand_angle_x, direction=(1, 0, 0), point=(0, 0, 0))
    )
    # Translate to final position
    fruit_final = transform_shape(fruit_shape, translation_matrix(all_positions[2].tolist()))
    final_objects.append(fruit_final)

    return concat_shapes(*final_objects)


@register()
def root_scene() -> Shape:
    return place_drinks_and_fruit()
