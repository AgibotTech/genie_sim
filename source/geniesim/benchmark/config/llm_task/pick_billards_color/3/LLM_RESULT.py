# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: table_with_solid_billiard_balls
description: A table with 6 different solid-colored billiard balls placed randomly on its surface.
"""


def get_desktop_surface_info(table_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the table's desktop top surface and its size.
    """
    table_info = get_object_info(table_shape)
    # table_000 has a subpart named 'desktop'
    desktop_subpart = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Calculate the top surface center and size
    desktop_center_world = table_info["center"] + desktop_subpart["center"]
    desktop_top_z = table_info["center"][2] + desktop_subpart["xyz_max"][2]

    surface_pos = np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z])
    surface_size = desktop_subpart["size"]

    return surface_pos, surface_size


@register()
def single_billiard_ball(oid: str, color_name: str, position_tag: str) -> Shape:
    """
    Creates a single billiard ball with specific color and position tag.
    """
    ball_shape = library_call(
        "usd",
        oid=oid,
        keywords=[f"{color_name}_billiard_ball", "solid", "sphere", "glossy", position_tag],
    )
    return ball_shape


@register()
def random_billiard_balls(surface_pos: P, surface_size: P) -> Shape:
    """
    Creates 6 different solid-colored billiard balls and places them randomly on the surface.
    """
    # List of 6 different solid-colored billiard ball OIDs
    ball_configs = [
        ("benchmark_billiards_008", "red"),
        ("benchmark_billiards_010", "yellow"),
        ("benchmark_billiards_013", "blue"),
        ("benchmark_billiards_011", "purple"),
        ("benchmark_billiards_009", "green"),
        ("benchmark_billiards_007", "black"),
    ]

    ball_radius = 0.0285  # Standard billiard ball radius is ~2.85cm
    margin = 0.05  # Keep away from edges

    all_balls = []

    # Define boundaries for random placement
    x_min = surface_pos[0] - surface_size[0] / 2.0 + margin
    x_max = surface_pos[0] + surface_size[0] / 2.0 - margin
    y_min = surface_pos[1] - surface_size[1] / 2.0 + margin
    y_max = surface_pos[1] + surface_size[1] / 2.0 - margin

    # To avoid simple collisions, we can use a simple grid or just random with check
    # For 6 balls on a table, simple random is usually fine if the table is large enough
    for oid, color in ball_configs:
        rand_x = np.random.uniform(x_min, x_max)
        rand_y = np.random.uniform(y_min, y_max)

        # Determine position tag
        pos_tag = "left" if rand_y > surface_pos[1] else "right"

        ball = library_call("single_billiard_ball", oid=oid, color_name=color, position_tag=pos_tag)

        # Transform to random position on the surface
        # Note: usd() returns shape with origin at bottom, so we place it at surface_pos[2]
        ball_transformed = transform_shape(ball, translation_matrix([rand_x, rand_y, surface_pos[2]]))
        all_balls.append(ball_transformed)

    return concat_shapes(*all_balls)


@register()
def table_with_billiard_balls() -> Shape:
    """
    Main function to assemble the table and the balls.
    """
    # Load the table
    table_shape = library_call("usd", oid="table_000", keywords=["table", "white", "minimalist", "center_piece"])

    # Get surface info for placement
    surface_pos, surface_size = get_desktop_surface_info(table_shape)

    # Generate and place balls
    balls_shape = library_call("random_billiard_balls", surface_pos=surface_pos, surface_size=surface_size)

    return concat_shapes(table_shape, balls_shape)


@register()
def root_scene() -> Shape:
    """
    Root scene function.
    """
    return table_with_billiard_balls()
