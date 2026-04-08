# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
import numpy as np

"""
scene_name: random_object_placement
description: Randomly select one object from a given list and place it within a defined area with yaw variance
"""


def get_random_object_from_list(object_list):
    """
    Randomly selects one object ID from the provided list.

    Args:
        object_list (list): List of object IDs to choose from

    Returns:
        str: Selected object ID
    """
    return random.choice(object_list)


def calculate_random_position(center, x_range, y_range):
    """
    Calculates a random position within the specified area.

    Args:
        center (P): Center point (x, y, z) of the area
        x_range (tuple): (min_x, max_x) relative to center
        y_range (tuple): (min_y, max_y) relative to center

    Returns:
        P: Random position within the area
    """
    random_x = center[0] + random.uniform(x_range[0], x_range[1])
    random_y = center[1] + random.uniform(
        y_range[1], y_range[0]
    )  # Note: y_range is swapped to match left/right convention
    return np.array([random_x, random_y, center[2]])


def calculate_yaw_variance(yaw_range):
    """
    Calculates a random yaw rotation within the specified range.

    Args:
        yaw_range (tuple): (min_yaw, max_yaw) in radians

    Returns:
        float: Random yaw angle in radians
    """
    return random.uniform(yaw_range[0], yaw_range[1])


@register()
def random_object_from_list():
    """
    Creates a single randomly selected object from the list.
    The object is placed with its origin at the ground (z=0).
    """
    object_list = [
        "apple",
        "blocks",
        "cola",
        "facecleaner",
        "benchmark_building_blocks_010",
        "benchmark_building_blocks_008",
        "sprite",
        "benchmark_building_blocks_013",
    ]

    selected_object = get_random_object_from_list(object_list)

    # Determine POSITION TAG based on object ID for better scene description
    # Since the placement area is centered at y=0, we'll use "center" tag
    position_tag = "center"

    # Create keywords with global unique name, tags, and position information
    keywords = [f"{selected_object}_random", selected_object, position_tag, "randomly_placed", "single_object"]

    # Load the object using usd function
    object_shape = library_call("usd", oid=selected_object, keywords=keywords)

    return object_shape


@register()
def place_random_object():
    """
    Places a randomly selected object within the specified area with yaw variance.
    Area: x(-0.07, 0.07), y(-0.15, 0.15) centered at (2.3, 0.0, 1.2)
    """
    # Define placement parameters
    center_position = np.array([2.3, 0.0, 1.2])
    x_range = (-0.07, 0.07)
    y_range = (-0.15, 0.15)
    yaw_range = (-0.5, 0.5)  # Approximately ±28.6 degrees in radians

    # Get the random object shape
    object_shape = library_call("random_object_from_list")

    # Calculate object information to ensure proper placement
    object_info = get_object_info(object_shape)
    object_size = object_info["size"]
    object_center = object_info["center"]

    # Calculate random position within the area
    random_position = calculate_random_position(center_position, x_range, y_range)

    # Calculate random yaw rotation
    random_yaw = calculate_yaw_variance(yaw_range)

    # Apply transformations: first translation to position, then rotation around z-axis
    # Since objects are loaded with origin at ground (z=0), we need to place them at z=1.15
    # The random_position already includes the z coordinate

    # First, center the object properly (since object_center might not be at origin)
    offset_to_center = object_center

    # Apply translation to random position
    translated_shape = transform_shape(object_shape, translation_matrix(random_position - offset_to_center))

    # Get new center after translation for rotation
    new_center = compute_shape_center(translated_shape)

    # Apply yaw rotation around z-axis
    final_shape = transform_shape(
        translated_shape,
        rotation_matrix(random_yaw, direction=(0, 0, 1), point=new_center),  # Rotate around z-axis for yaw
    )

    return final_shape


@register()
def root_scene() -> Shape:
    """
    Main scene function that places a random object in the specified area.
    """
    return library_call("place_random_object")
