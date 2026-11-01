# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: stack_trash_cans_g2
description: A desk with two trash cans. One is placed in the central area of the desk, and the other is placed on the outer area of the desk. Both are randomly positioned and rotated, with a minimum center-to-center distance of 0.2m between them.
"""


def get_desk_workspace_info(desk_shape: Shape) -> dict:
    """
    Extracts the desk's center and size in world coordinates.
    """
    desk_info = get_object_info(desk_shape)
    return {
        "center": desk_info["center"],
        "size": desk_info["size"],
        "min": desk_info["min"],
        "max": desk_info["max"]
    }


def sample_position_in_region(center: P, region_x: Tuple[float, float], region_y: Tuple[float, float]) -> P:
    """
    Samples a random (x, y) position within a specified region relative to a center point.
    The z-coordinate is set to 0 and should be adjusted later based on surface height.
    """
    x = np.random.uniform(region_x[0], region_x[1])
    y = np.random.uniform(region_y[0], region_y[1])
    return np.array([center[0] + x, center[1] + y, 0.0])


def place_trash_can_on_desk(desk_shape: Shape) -> Shape:
    """
    Places two trash cans on the desk according to the specified regions and constraints.
    Ensures minimum center-to-center distance of 0.2m between the two trash cans.
    """
    desk_info = get_desk_workspace_info(desk_shape)
    desk_center = desk_info["center"]

    # Get desktop subpart info for height calculation
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_top_z = desk_center[2] + desktop_subpart_info["xyz_max"][2]

    # Define regions
    inner_region_x = (-0.15, 0.05)
    inner_region_y = (-0.1, 0.1)
    outer_region_x = (-0.25, 0.1)
    outer_region_y = (-0.25, 0.25)

    max_attempts = 1000  # Increased attempts for better success rate
    for _ in range(max_attempts):
        # Sample position for inner trash can
        inner_pos = sample_position_in_region(desk_center, inner_region_x, inner_region_y)
        # Sample position for outer trash can
        outer_pos = sample_position_in_region(desk_center, outer_region_x, outer_region_y)

        # Calculate center-to-center distance in XY plane
        distance = np.linalg.norm(inner_pos[:2] - outer_pos[:2])

        # Check if distance constraint is satisfied (minimum 0.2m center-to-center)
        if distance >= 0.2:
            # Determine POSITION TAGs based on Y coordinate relative to desk center
            inner_tag = "left" if inner_pos[1] > desk_center[1] else "right"
            outer_tag = "left" if outer_pos[1] > desk_center[1] else "right"

            # Place inner trash can with random rotation
            inner_trash_can = library_call(
                "usd",
                oid="benchmark_trash_can_013",
                keywords=[f"inner_trash_can_{inner_tag}", "gray", "cylinder", "small", f"position_{inner_tag}"]
            )
            inner_trash_can = transform_shape(
                inner_trash_can,
                translation_matrix([inner_pos[0], inner_pos[1], desktop_top_z])
            )
            inner_trash_can = transform_shape(
                inner_trash_can,
                rotation_matrix(
                    angle=np.random.uniform(0, 2 * math.pi),
                    direction=(0, 0, 1),
                    point=compute_shape_center(inner_trash_can)
                )
            )

            # Place outer trash can with random rotation
            outer_trash_can = library_call(
                "usd",
                oid="benchmark_trash_can_013",
                keywords=[f"outer_trash_can_{outer_tag}", "gray", "cylinder", "small", f"position_{outer_tag}"]
            )
            outer_trash_can = transform_shape(
                outer_trash_can,
                translation_matrix([outer_pos[0], outer_pos[1], desktop_top_z])
            )
            outer_trash_can = transform_shape(
                outer_trash_can,
                rotation_matrix(
                    angle=np.random.uniform(0, 2 * math.pi),
                    direction=(0, 0, 1),
                    point=compute_shape_center(outer_trash_can)
                )
            )

            return concat_shapes(desk_shape, inner_trash_can, outer_trash_can)

    # Fallback: If no valid placement found, use fixed positions that guarantee distance
    # Place inner trash can at center of inner region
    inner_pos = np.array([
        desk_center[0] + (inner_region_x[0] + inner_region_x[1]) / 2.0,
        desk_center[1] + (inner_region_y[0] + inner_region_y[1]) / 2.0,
        desktop_top_z
    ])

    # Place outer trash can at a position that ensures 0.2m+ distance
    # Use the farthest point in outer region from inner_pos
    candidate_positions = [
        [desk_center[0] + outer_region_x[0], desk_center[1] + outer_region_y[0]],
        [desk_center[0] + outer_region_x[0], desk_center[1] + outer_region_y[1]],
        [desk_center[0] + outer_region_x[1], desk_center[1] + outer_region_y[0]],
        [desk_center[0] + outer_region_x[1], desk_center[1] + outer_region_y[1]]
    ]

    best_outer_pos = None
    max_distance = -1
    for candidate in candidate_positions:
        dist = np.linalg.norm(np.array([inner_pos[0], inner_pos[1]]) - np.array(candidate))
        if dist > max_distance:
            max_distance = dist
            best_outer_pos = candidate

    outer_pos = np.array([best_outer_pos[0], best_outer_pos[1], desktop_top_z])

    inner_tag = "left" if inner_pos[1] > desk_center[1] else "right"
    outer_tag = "left" if outer_pos[1] > desk_center[1] else "right"

    # Place inner trash can with random rotation (fallback)
    inner_trash_can = library_call(
        "usd",
        oid="benchmark_trash_can_013",
        keywords=[f"inner_trash_can_{inner_tag}_fallback", "gray", "cylinder", "small", f"position_{inner_tag}"]
    )
    inner_trash_can = transform_shape(
        inner_trash_can,
        translation_matrix([inner_pos[0], inner_pos[1], desktop_top_z])
    )
    inner_trash_can = transform_shape(
        inner_trash_can,
        rotation_matrix(
            angle=np.random.uniform(0, 2 * math.pi),
            direction=(0, 0, 1),
            point=compute_shape_center(inner_trash_can)
        )
    )

    # Place outer trash can with random rotation (fallback)
    outer_trash_can = library_call(
        "usd",
        oid="benchmark_trash_can_013",
        keywords=[f"outer_trash_can_{outer_tag}_fallback", "gray", "cylinder", "small", f"position_{outer_tag}"]
    )
    outer_trash_can = transform_shape(
        outer_trash_can,
        translation_matrix([outer_pos[0], outer_pos[1], desktop_top_z])
    )
    outer_trash_can = transform_shape(
        outer_trash_can,
        rotation_matrix(
            angle=np.random.uniform(0, 2 * math.pi),
            direction=(0, 0, 1),
            point=compute_shape_center(outer_trash_can)
        )
    )

    return concat_shapes(desk_shape, inner_trash_can, outer_trash_can)


@register()
def stack_trash_cans_g2() -> Shape:
    desk_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )
    return place_trash_can_on_desk(desk_shape)


@register()
def root_scene() -> Shape:
    return stack_trash_cans_g2()
