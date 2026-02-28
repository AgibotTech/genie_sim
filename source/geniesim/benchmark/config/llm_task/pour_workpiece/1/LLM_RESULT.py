# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: pour_workpiece
description: A laboratory scene with plastic boxes and various accessories arranged based on laboratory_00.usda
"""


@register()
def plastic_box_cluster() -> Shape:
    """
    Create a cluster of plastic boxes based on laboratory_00.usda scene.
    Uses sft_plastic_box assets with actual positions from the scene.
    """
    # Use sft_plastic_box assets from the scene
    # Quaternion format in USDA: (x, y, z, w)
    # First two boxes are rotated ±90° around Z axis to face each other
    box_configs = [
        ("sft_plastic_box_001", (-1.837, 2.006, 0.886), (0, 0, 0.707, 0.707)),  # Counter-clockwise 90° around Z
        ("sft_plastic_box_002", (-1.828, 2.464, 0.886), (0, 0, -0.707, 0.707)),  # Clockwise 90° around Z
    ]

    box_shapes = []

    for i, (box_id, position, quaternion) in enumerate(box_configs):
        box_shape = library_call(
            "usd",
            oid=box_id,
            keywords=[f"plastic_box_{i}", "storage", "plastic", "container"]
        )

        # Convert quaternion to angle+axis for rotation_matrix
        angle, axis = quaternion_to_angle_direction(quaternion)

        # Apply rotation around origin
        box_transformed = transform_shape(
            box_shape,
            rotation_matrix(angle, axis, (0, 0, 0))
        )

        # Apply translation
        box_transformed = transform_shape(
            box_transformed,
            translation_matrix([position[0], position[1], position[2]])
        )

        box_shapes.append(box_transformed)

    return concat_shapes(*box_shapes)


@register()
def accessories_scatter() -> Shape:
    """
    Create scattered accessories based on laboratory_00.usda scene.
    Uses all sft_accessories assets (000, 001, 002, 003, 004) with actual positions.
    Total: 69 accessories for a rich, detailed scene.
    """
    # All accessories from the scene (69 total)
    # Quaternion format in USDA: (x, y, z, w)
    accessory_configs = [
        # sft_accessories_000 - 10 items (kept: inside the two boxes at x ? -1.8)
        ("sft_accessories_000", (-1.818, 2.449, 0.907), (0.726, 0.0, 0.0, 0.688)),
        ("sft_accessories_000", (-1.850, 2.452, 0.851), (0.664, -0.293, -0.278, 0.629)),
        ("sft_accessories_000", (-1.794, 2.447, 0.829), (0.726, 0.0, 0.0, 0.688)),
        ("sft_accessories_000", (-1.870, 2.384, 0.987), (0.736, 0.082, 0.539, -0.402)),
        ("sft_accessories_000", (-1.867, 2.089, 0.871), (0.708, 0.0, 0.0, -0.706)),
        ("sft_accessories_000", (-1.799, 2.090, 0.870), (0.648, -0.286, 0.285, -0.646)),
        ("sft_accessories_000", (-1.857, 1.934, 0.843), (0.727, 0.0, 0.0, 0.687)),
        ("sft_accessories_000", (-1.858, 2.026, 0.879), (0.380, -0.622, 0.583, 0.359)),
        ("sft_accessories_000", (-1.803, 1.934, 0.866), (0.735, 0.083, 0.538, -0.403)),
        ("sft_accessories_000", (-1.884, 2.089, 0.850), (0.708, 0.0, 0.0, -0.706)),
    ]

    accessory_shapes = []

    for i, (accessory_id, position, quaternion) in enumerate(accessory_configs):
        accessory_shape = library_call(
            "usd",
            oid=accessory_id,
            keywords=[f"accessory_{i}", "small", "detail", "laboratory"]
        )

        # Convert quaternion to angle+axis for rotation_matrix
        angle, axis = quaternion_to_angle_direction(quaternion)

        # Apply rotation around origin
        accessory_transformed = transform_shape(
            accessory_shape,
            rotation_matrix(angle, axis, (0, 0, 0))
        )

        # Apply translation with elevated height to make accessories float above boxes
        # Boxes are at z ? 0.886, add offset to make them float visibly above
        z_position = max(position[2], 0.886) + 0.15  # Add 15cm offset for floating effect
        accessory_transformed = transform_shape(
            accessory_transformed,
            translation_matrix([position[0], position[1], z_position])
        )

        accessory_shapes.append(accessory_transformed)

    return concat_shapes(*accessory_shapes)


@register()
def arrange_scene_elements() -> Shape:
    """
    Arrange all scene elements - plastic boxes and accessories.
    Objects are placed at absolute positions from the laboratory_00.usda scene.
    No table - objects placed directly in world coordinates.
    """
    # Create plastic box cluster with actual scene positions
    box_cluster = library_call("plastic_box_cluster")

    # Create accessories scatter with actual scene positions
    accessories = library_call("accessories_scatter")

    # IMPORTANT: Generate boxes FIRST so they can serve as platforms
    # Then accessories on top - this ensures proper physics and prevents falling through
    return concat_shapes(box_cluster, accessories)


@register()
def pour_workpiece() -> Shape:
    """
    Main function to generate the laboratory scene with boxes and accessories.
    """
    return arrange_scene_elements()


@register()
def root_scene() -> Shape:
    return pour_workpiece()
