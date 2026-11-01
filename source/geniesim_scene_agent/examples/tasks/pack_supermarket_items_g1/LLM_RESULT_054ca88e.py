# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np

"""
scene_name: desk_with_product_and_bag
description: A product and a handbag placed on a desk with specific positioning constraints
- Bag in left half area: x (-0.15, 0.1), y (0.07, 0.15)
- Product in right half area: x (-0.2, 0.15), y (-0.1, -0.2)
- Minimum distance between bag and product center >= 0.25
"""


def find_desktop_top_surface(desk_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the desktop's top surface and its size.

    Args:
        desk_shape (Shape): The shape of the desk object.

    Returns:
        Tuple[P, P]: A tuple containing the desktop's center position and its size.
    """
    desk_info = get_object_info(desk_shape)
    # Get subpart info for desktop - assuming "desktop" is the subpart name
    try:
        desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    except:
        # If subpart not found, use object info directly
        desktop_subpart_info = {
            "center": [0, 0, 0],
            "size": desk_info["size"],
            "xyz_max": [desk_info["size"][0] / 2, desk_info["size"][1] / 2, desk_info["size"][2] / 2],
        }

    # The desktop center in world coordinates
    desktop_center = desk_info["center"] + desktop_subpart_info["center"]

    # The top surface z-coordinate of the desktop in world coordinates
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2] - 0.05

    # The final desktop position (center_x, center_y, top_z)
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])

    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


@register()
def select_product() -> Shape:
    """
    Selects and returns a product from the available options.

    Args:
        product_choice (int): Index to select product (0=orange, 1=apple, 2=green_apple)

    Returns:
        Shape: The selected product shape
    """
    product_options = [
        ("benchmark_orange_004", "orange", "orange"),
        ("benchmark_apple_001", "red", "apple"),
        ("benchmark_green_apple_000", "green", "green_apple"),
        ("benchmark_fruit_021", "yellow", "banana"),
        ("benchmark_beverage_bottle_073", "red", "red_bull"),
    ]

    oid, color, category = random.choice(product_options)

    product_shape = library_call(
        "usd",
        oid=oid,
        keywords=["product", color, category, "on_desk"],
    )
    return product_shape


@register()
def place_bag_on_desk(desk_shape: Shape) -> Shape:
    """
    Places a bag on the desk within specified bounds (left half area).
    Bag range: x (-0.15, 0.1), y (0.07, 0.15)

    Args:
        desk_shape (Shape): The desk shape

    Returns:
        Shape: Bag shape positioned on desk
    """
    workspace, margin = find_desktop_top_surface(desk_shape)

    # Bag placement range: x (-0.15, 0.1), y (0.07, 0.15) - LEFT HALF AREA
    bag_x = np.random.uniform(0.02, 0.15)
    bag_y = np.random.uniform(0.1, 0.15)

    bag_shape = library_call(
        "usd",
        oid="benchmark_handbag_004",
        keywords=["handbag", "bag", "left_area", "on_desk"],
    )

    # Apply random rotation around z-axis
    random_angle = np.random.uniform(-np.pi, np.pi)
    bag_shape = transform_shape(
        bag_shape,
        translation_matrix((workspace[0] + bag_x, workspace[1] + bag_y, workspace[2])),
    )
    bag_shape = transform_shape(
        bag_shape,
        rotation_matrix(random_angle, direction=(0, 0, 1), point=compute_shape_center(bag_shape)),
    )

    return bag_shape


@register()
def place_product_on_desk(desk_shape: Shape, bag_shape: Shape) -> Shape:
    """
    Places a product on the desk within specified bounds (right half area),
    ensuring minimum distance from bag.
    Product range: x (-0.2, 0.15), y (-0.1, -0.2)

    Args:
        desk_shape (Shape): The desk shape
        bag_shape (Shape): The bag shape already placed
        product_choice (int): Which product to place

    Returns:
        Shape: Product shape positioned on desk
    """
    workspace, margin = find_desktop_top_surface(desk_shape)

    # Get bag center for distance calculation
    bag_info = get_object_info(bag_shape)
    bag_center = bag_info["center"]

    # Product placement range: x (-0.2, 0.15), y (-0.1, -0.2) - RIGHT HALF AREA
    min_distance = 0.25

    # Try to find valid product position that maintains minimum distance from bag
    max_attempts = 100
    product_shape = None

    for attempt in range(max_attempts):
        product_x = np.random.uniform(-0.2, 0.15)
        product_y = np.random.uniform(-0.1, -0.25)

        # Calculate potential product center
        potential_product_center = np.array([workspace[0] + product_x, workspace[1] + product_y, workspace[2]])

        # Check distance from bag
        distance = np.linalg.norm(potential_product_center[:2] - bag_center[:2])

        if distance >= min_distance:
            # Valid position found
            product_shape = select_product()

            # Apply random rotation around z-axis
            random_angle = np.random.uniform(-np.pi, np.pi)
            product_shape = transform_shape(
                product_shape,
                translation_matrix(
                    (potential_product_center[0], potential_product_center[1], potential_product_center[2])
                ),
            )
            product_shape = transform_shape(
                product_shape,
                rotation_matrix(random_angle, direction=(0, 0, 1), point=compute_shape_center(product_shape)),
            )
            break

    if product_shape is None:
        # Fallback: place at closest valid position within bounds
        product_shape = select_product()
        fallback_x = -0.2
        fallback_y = -0.15
        product_shape = transform_shape(
            product_shape,
            translation_matrix((workspace[0] + fallback_x, workspace[1] + fallback_y, workspace[2])),
        )
        random_angle = np.random.uniform(-np.pi, np.pi)
        product_shape = transform_shape(
            product_shape,
            rotation_matrix(random_angle, direction=(0, 0, 1), point=compute_shape_center(product_shape)),
        )

    return product_shape


@register()
def desk_with_product_and_bag() -> Shape:
    """
    Main scene composition: places a product and bag on a desk.
    - Bag in left half area: x (-0.15, 0.1), y (0.07, 0.15)
    - Product in right half area: x (-0.2, 0.15), y (-0.1, -0.2)
    - Minimum distance between bag and product center >= 0.25

    Args:
        product_choice (int): Which product to place (0=orange, 1=apple, 2=green_apple)

    Returns:
        Shape: Complete scene with desk, product, and bag
    """
    # Load desk
    desk_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )

    # Place bag on desk first (left half area)
    bag_shape = place_bag_on_desk(desk_shape)

    # Place product on desk (right half area, maintaining distance from bag)
    product_shape = place_product_on_desk(desk_shape, bag_shape)

    # Combine all shapes
    return concat_shapes(desk_shape, bag_shape, product_shape)


@register()
def root_scene() -> Shape:
    """
    Root scene function that generates the complete scene.
    Default to orange fruit (choice 0).
    """
    return desk_with_product_and_bag()
