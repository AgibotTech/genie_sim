# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: desk_storage_and_fruits
description: A table with two storage boxes placed on it at specific offsets, containing fruits.
             Additional fruits are scattered on the desk surface.
"""

# Asset Lists
STORAGE_BOXES = [
    {
        "oid": "benchmark_storage_box_006",
        "keywords": ["storage_box_006", "storage box", "plastic", "rectangular"],
    },
    {
        "oid": "benchmark_storage_box_007",
        "keywords": ["storage_box_007", "storage box", "plastic", "rectangular"],
    },
    {
        "oid": "benchmark_storage_box_008",
        "keywords": ["storage_box_008", "storage box", "plastic", "rectangular"],
    },
    {
        "oid": "benchmark_storage_box_009",
        "keywords": ["storage_box_009", "storage box", "plastic", "rectangular"],
    },
    {
        "oid": "benchmark_storage_box_010",
        "keywords": ["storage_box_010", "storage box", "plastic", "rectangular"],
    },
]

FRUITS = [
    {"oid": "benchmark_apple_000", "keywords": ["apple_000", "apple", "red", "fruit"]},
    {"oid": "benchmark_orange_001", "keywords": ["orange_001", "orange", "orange", "fruit"]},
    {"oid": "benchmark_peach_000", "keywords": ["peach_000", "peach", "pink", "fruit"]},
    {"oid": "benchmark_lemon_027", "keywords": ["lemon_027", "lemon", "yellow", "fruit"]},
]


def get_desktop_surface(desk_shape: Shape) -> tuple:
    """
    Calculates the world coordinates of the desktop's top surface center and its size.

    Args:
        desk_shape (Shape): The shape of the desk object, already placed in the world.

    Returns:
        tuple: A tuple containing the desktop's top surface center position and its size.
    """
    # Information about the whole desk object in world coordinates
    desk_info = get_object_info(desk_shape)

    # Information about the 'desktop' subpart in the desk's local coordinates
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # The desktop's center in world coordinates is the desk's center plus the subpart's local offset
    desktop_center_world = desk_info["center"] + desktop_subpart_info["center"]

    # The z-coordinate of the desktop's top surface in world coordinates
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    # The final position for placing objects is the center of the desktop surface
    desktop_surface_pos = np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z])

    desktop_size = desktop_subpart_info["size"]

    return desktop_surface_pos, desktop_size


@register()
def storage_box_with_fruits(box_oid: str, fruit_oid: str, num_fruits: int, target_pos: P, pos_tag: str) -> Shape:
    """
    Creates a storage box, rotates it 90 degrees around Z, and places fruits inside.
    Fruits are placed with staggered Z and random XY to avoid initial collision.
    """
    # Load the box
    box_shape = library_call("usd", oid=box_oid, keywords=[f"box_{pos_tag}", "storage", "container", pos_tag])

    # Rotate 90 degrees around Z axis (local origin)
    box_shape = transform_shape(box_shape, rotation_matrix(math.pi / 2, direction=(0, 0, 1), point=(0, 0, 0)))

    # Translate to target world position
    box_shape = transform_shape(box_shape, translation_matrix(target_pos))

    # Get box info for fruit placement
    box_info = get_object_info(box_shape)
    box_center = box_info["center"]
    box_size = box_info["size"]
    box_bottom = box_info["min"][2]

    def place_fruit_inside(i: int) -> Shape:
        fruit_shape = library_call("usd", oid=fruit_oid, keywords=[f"fruit_in_box_{pos_tag}_{i}", "contained", pos_tag])
        # Random XY within box bounds (safe margin)
        # Staggered Z to allow natural falling/stacking
        f_x = box_center[0] + np.random.uniform(-box_size[0] * 0.2, box_size[0] * 0.2)
        f_y = box_center[1] + np.random.uniform(-box_size[1] * 0.2, box_size[1] * 0.2)
        f_z = box_bottom + 0.05 + i * 0.07

        return transform_shape(fruit_shape, translation_matrix([f_x, f_y, f_z]))

    fruits_shape = loop(num_fruits, place_fruit_inside)
    return concat_shapes(box_shape, fruits_shape)


@register()
def scattered_fruits_on_desk(fruit_oids: list, desk_center: P, box_positions: list) -> Shape:
    """
    Places a list of fruits randomly on the desk surface, ensuring no collision
    with boxes or other fruits.
    """
    # We define a safe area for scattered fruits away from the boxes.
    # Boxes are at x_offset ~ 0.11. We place scattered fruits at x_offset ~ -0.2.
    # We use a simple grid/offset approach to ensure no overlap between fruits.

    def place_scattered(i: int) -> Shape:
        fruit_shape = library_call("usd", oid=fruit_oids[i], keywords=[f"scattered_fruit_{i}", "on_desk", "random"])

        # Base position for scattered fruits (negative X relative to desk center)
        base_x = desk_center[0] - 0.2
        base_y = desk_center[1]

        # Offset each fruit along the Y axis to avoid collision with each other
        # Fruit size is roughly 0.08m, so 0.15m spacing is safe.
        y_offset = (i - (len(fruit_oids) - 1) / 2.0) * 0.15

        # Add small random jitter
        jitter_x = np.random.uniform(-0.05, 0.05)
        jitter_y = np.random.uniform(-0.02, 0.02)

        f_x = base_x + jitter_x
        f_y = base_y + y_offset + jitter_y
        f_z = desk_center[2]  # Surface height

        return transform_shape(fruit_shape, translation_matrix([f_x, f_y, f_z]))

    return loop(len(fruit_oids), place_scattered)


@register()
def root_scene() -> Shape:
    # 1. Create the table
    import random

    desk_shape = library_call("usd", oid="table_000", keywords=["table", "white", "minimalist", "center"])

    # Get the desktop surface position and size
    desk_center, desk_size = get_desktop_surface(desk_shape)

    # 2. Select Assets
    selected_box_indices = random.sample(range(len(STORAGE_BOXES)), 2)
    selected_fruit_type_indices = random.sample(range(len(FRUITS)), 2)

    # 3. Calculate Box Positions
    # User provided offsets: [0.11, 0.09, 0.9] and [0.11, -0.1, 0.9]
    # Place boxes on the desk surface
    delta_x = np.random.uniform(-0.03, 0.03)

    box1_pos = np.array([desk_center[0] + 0.11 + delta_x, desk_center[1] + 0.09, desk_center[2]])
    box2_pos = np.array([desk_center[0] + 0.11 + delta_x, desk_center[1] - 0.1, desk_center[2]])

    # 4. Create Box 1 with Fruits
    num_f1 = np.random.randint(1, 3)
    box1_shape = library_call(
        "storage_box_with_fruits",
        box_oid=STORAGE_BOXES[selected_box_indices[0]]["oid"],
        fruit_oid=FRUITS[selected_fruit_type_indices[0]]["oid"],
        num_fruits=num_f1,
        target_pos=box1_pos,
        pos_tag="left",
    )

    # 5. Create Box 2 with Fruits
    num_f2 = np.random.randint(1, 3)
    box2_shape = library_call(
        "storage_box_with_fruits",
        box_oid=STORAGE_BOXES[selected_box_indices[1]]["oid"],
        fruit_oid=FRUITS[selected_fruit_type_indices[1]]["oid"],
        num_fruits=num_f2,
        target_pos=box2_pos,
        pos_tag="right",
    )

    # 6. Create Scattered Fruits
    # num_scattered = np.random.randint(2, 5)
    num_scattered = 1
    # One must match a box fruit type
    match_type_idx = random.choice(selected_fruit_type_indices)
    scattered_oids = [FRUITS[match_type_idx]["oid"]]
    for _ in range(num_scattered - 1):
        scattered_oids.append(random.choice(FRUITS)["oid"])

    scattered_shape = library_call(
        "scattered_fruits_on_desk",
        fruit_oids=scattered_oids,
        desk_center=desk_center,
        box_positions=[box1_pos, box2_pos],
    )

    # Return all objects (including the desk)
    return concat_shapes(desk_shape, box1_shape, box2_shape, scattered_shape)
