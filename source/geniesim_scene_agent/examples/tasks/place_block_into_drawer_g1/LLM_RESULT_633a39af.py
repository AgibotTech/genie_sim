# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import numpy as np

"""
scene_name: drawer_with_single_building_block
description: A drawer cabinet with a single building block randomly placed on top, rotating only around Z-axis
"""


def get_cabinet_top_surface(cabinet_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the cabinet's top surface and its size.

    Args:
        cabinet_shape (Shape): The shape of the cabinet object.

    Returns:
        Tuple[P, P]: A tuple containing the cabinet top's center position and its size.
    """
    cabinet_info = get_object_info(cabinet_shape)
    # Get subpart info for the top surface of the cabinet
    cabinet_subpart_info = get_subpart_info(object_id="iros_drawer_cabinet_000", subpart_id="top")

    # The top surface center in world coordinates
    top_center = cabinet_info["center"] + cabinet_subpart_info["center"]

    # The top surface z-coordinate in world coordinates
    top_z = cabinet_info["center"][2] + cabinet_subpart_info["xyz_max"][2]

    # The final top surface position (center_x, center_y, top_z)
    top_pos = np.array([top_center[0], top_center[1], top_z])

    top_size = cabinet_subpart_info["size"]

    return top_pos, top_size


@register()
def single_building_block() -> Shape:
    """
    Generates a single building block with random rotation ONLY around Z-axis.

    Returns:
        Shape: The building block shape with random orientation around Z-axis.
    """
    block_shape = library_call(
        "usd",
        oid="benchmark_building_blocks_086",
        keywords=["building_block", "colorful", "cube", "small", "single"],
    )

    # Random rotation ONLY around Z-axis (360 degrees)
    rot_angle_z = np.random.uniform(-np.pi, np.pi)  # Full 360 degrees

    # Create rotation transformation around block center
    block_center = compute_shape_center(block_shape)

    # Apply rotation around Z-axis only
    rotated_shape = transform_shape(block_shape, rotation_matrix(rot_angle_z, direction=(0, 0, 1), point=block_center))

    return rotated_shape


@register()
def place_block_on_cabinet(cabinet_shape: Shape) -> Shape:
    """
    Places a single building block randomly on top of the cabinet while keeping it within bounds.
    Block rotates only around Z-axis.

    Args:
        cabinet_shape (Shape): The cabinet shape.

    Returns:
        Shape: Cabinet with block placed on top.
    """
    top_pos, top_size = get_cabinet_top_surface(cabinet_shape)

    # Get block size to ensure it fits within cabinet boundaries
    block_shape = library_call(
        "usd",
        oid="benchmark_building_blocks_086",
        keywords=["building_block", "colorful", "cube", "small", "reference"],
    )
    block_size = compute_shape_sizes(block_shape)

    # Calculate safe placement range (keep block within cabinet boundaries)
    # Subtract half block size from each side to prevent overhang
    margin_x = top_size[0] / 2.0 - block_size[0] / 2.0
    margin_y = top_size[1] / 2.0 - block_size[1] / 2.0

    # Ensure margins are positive (if block is too large, clamp to edge)
    margin_x = max(0.01, margin_x)
    margin_y = max(0.01, margin_y)

    # Random position within safe bounds relative to top center
    x_offset = np.random.uniform(-margin_x, margin_x)
    y_offset = np.random.uniform(-margin_y, margin_y)

    # Z position is top surface height plus half block height (to sit on surface)
    z_offset = top_pos[2] + block_size[2] / 2.0

    # Generate the single block with random Z-axis rotation
    block_shape_rotated = library_call("single_building_block")

    # Translate block to random position on top surface
    block_positioned = transform_shape(block_shape_rotated, translation_matrix((x_offset, y_offset, z_offset)))

    return concat_shapes(cabinet_shape, block_positioned)


@register()
def drawer_with_single_building_block() -> Shape:
    """
    Main scene composition: drawer cabinet with a single building block on top.
    Block rotates only around Z-axis.
    """
    # Load drawer cabinet on ground (z=0.0 by default)
    cabinet_shape = library_call(
        "usd",
        oid="iros_drawer_cabinet_000",
        keywords=["drawer_cabinet", "storage", "wood", "furniture", "ground_level"],
    )
    cabinet_shape_center = compute_shape_center(cabinet_shape)

    # Apply rotation around Z-axis only
    rotated_shape = transform_shape(
        cabinet_shape, rotation_matrix(np.pi, direction=(0, 0, 1), point=cabinet_shape_center)
    )

    # Place a single building block randomly on top
    return place_block_on_cabinet(rotated_shape)


@register()
def root_scene() -> Shape:
    """
    Root scene entry point.
    """
    return drawer_with_single_building_block()
