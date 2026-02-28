# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: popcorn_bucket_scene
description: A popcorn bucket positioned at the L-shaped area below-left of the popcorn machine based on popcorn_01.usda
"""


@register()
def popcorn_bucket_placement() -> Shape:
    """
    Create a popcorn bucket positioned at the L-shaped area below-left of the popcorn machine.
    Based on popcorn_01.usda where the machine is at (0.838, -0.238, 1.178).
    XY position varies within ±0.1 range.
    """
    base_x, base_y, base_z = 0.7, 0.2, 1.0
    random_x = np.random.uniform(base_x - 0.1, base_x + 0.1)
    random_y = np.random.uniform(base_y - 0.1, base_y + 0.1)
    bucket_position = (random_x, random_y, base_z)

    # Use benchmark_popcorn_bucket_003 asset
    bucket_shape = library_call(
        "usd", oid="benchmark_popcorn_bucket_003", keywords=["popcorn_bucket", "container", "bucket", "popcorn"]
    )

    # Quaternion for rotation - slight rotation to face toward the machine
    # (x, y, z, w) format: Rotate slightly around Z to face the machine
    bucket_quaternion = (0, 0, 0.383, 0.924)  # ~45 degrees around Z

    # Convert quaternion to angle+axis for rotation_matrix
    angle, axis = quaternion_to_angle_direction(bucket_quaternion)

    # Apply rotation around origin
    bucket_transformed = transform_shape(bucket_shape, rotation_matrix(angle, axis, (0, 0, 0)))

    # Apply translation to position bucket in L-shape below-left of machine
    bucket_transformed = transform_shape(
        bucket_transformed, translation_matrix([bucket_position[0], bucket_position[1], bucket_position[2]])
    )

    return bucket_transformed


@register()
def scoop_popcorn() -> Shape:
    """
    Main function to generate the popcorn bucket scene.
    """
    return popcorn_bucket_placement()


@register()
def root_scene() -> Shape:
    return scoop_popcorn()
