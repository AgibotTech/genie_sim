# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: table_with_three_bowls
description: Three bowls randomly placed on a table, avoiding an expanded central forbidden zone (x: -0.13 to 0.13, y: -0.13 to 0.13), with minimum distance of 0.25m between any two bowls, and random rotation.
"""


def get_desktop_workspace_info(desk_shape: Shape) -> Tuple[P, P]:
    """
    Retrieves the top surface center and size of the desktop.

    Args:
        desk_shape (Shape): The shape of the desk.

    Returns:
        Tuple[P, P]: (desktop_center_on_top, desktop_size)
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    desktop_center = desk_info["center"] + desktop_subpart_info["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    desktop_size = desktop_subpart_info["size"]

    return desktop_pos, desktop_size


def point_in_forbidden_zone(
    x: float, y: float, forbidden_x_min: float, forbidden_x_max: float, forbidden_y_min: float, forbidden_y_max: float
) -> bool:
    """Check if a point (x, y) is inside the central forbidden rectangle."""
    return (forbidden_x_min <= x <= forbidden_x_max) and (forbidden_y_min <= y <= forbidden_y_max)


@register()
def single_bowl(keywords: List[str]) -> Shape:
    """Creates a single bowl with given keywords."""
    return library_call("usd", oid="benchmark_bowl_025", keywords=keywords)


@register()
def three_well_spaced_rotated_bowls_on_table(desk_shape: Shape) -> Shape:
    desktop_center, _ = get_desktop_workspace_info(desk_shape)

    # Define the placement zones relative to the desktop center
    outer_x_min, outer_x_max = -0.2, 0.2
    outer_y_min, outer_y_max = -0.3, 0.3
    # Further expanded forbidden central zone
    forbidden_x_min, forbidden_x_max = -0.15, 0.15
    forbidden_y_min, forbidden_y_max = -0.15, 0.15

    min_separation = 0.25  # Increased minimum distance between any two bowls

    placed_positions = []
    bowl_shapes = []

    for i in range(3):
        max_attempts = 1000  # Significantly increased attempts due to very tight constraints
        for attempt in range(max_attempts):
            # Sample a candidate position in the world frame
            x_offset = random.uniform(outer_x_min, outer_x_max)
            y_offset = random.uniform(outer_y_min, outer_y_max)

            # Skip if in the forbidden central zone
            if point_in_forbidden_zone(
                x_offset, y_offset, forbidden_x_min, forbidden_x_max, forbidden_y_min, forbidden_y_max
            ):
                continue

            candidate_pos = np.array([desktop_center[0] + x_offset, desktop_center[1] + y_offset, desktop_center[2]])

            # Check for collision with already placed bowls
            collision = False
            for placed_pos in placed_positions:
                dist = np.linalg.norm(candidate_pos[:2] - placed_pos[:2])
                if dist < min_separation:
                    collision = True
                    break

            if not collision:
                placed_positions.append(candidate_pos)
                break
        else:
            print(f"Failed to place bowl {i+1} after {max_attempts} attempts")
            # Fallback: place at safe default locations that are guaranteed to be outside the forbidden zone and far apart
            # These positions are carefully chosen to be near the corners/edges of the allowed area
            fallback_offsets = [(-0.145, -0.245), (0.095, -0.245), (-0.025, 0.245)]
            fallback_pos = np.array(
                [
                    desktop_center[0] + fallback_offsets[i][0],
                    desktop_center[1] + fallback_offsets[i][1],
                    desktop_center[2],
                ]
            )
            placed_positions.append(fallback_pos)

        # Create the bowl
        bowl = library_call(
            "single_bowl",
            keywords=[f"bowl_{i+1}", "white", "ceramic", "kitchenware", "on_table"],
        )

        # Apply random rotation around the Z-axis (vertical)
        random_angle = random.uniform(0, 2 * math.pi)
        bowl = transform_shape(bowl, rotation_matrix(angle=random_angle, direction=(0, 0, 1), point=(0, 0, 0)))

        # Then translate to the final position
        bowl = transform_shape(bowl, translation_matrix(placed_positions[-1]))
        bowl_shapes.append(bowl)

    return concat_shapes(desk_shape, *bowl_shapes)


@register()
def table_with_three_bowls() -> Shape:
    desk = library_call("usd", oid="table_000", keywords=["dining_table", "table", "wood", "furniture"])
    return three_well_spaced_rotated_bowls_on_table(desk)


@register()
def root_scene() -> Shape:
    return table_with_three_bowls()
