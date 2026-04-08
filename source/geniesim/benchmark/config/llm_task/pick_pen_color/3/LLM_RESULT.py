# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

"""
scene_name: table_with_colored_pens
description: A table (table_000) with 5 randomly placed colored pens of different colors on its desktop.
"""


def get_desktop_placement_area(desk_shape: Shape) -> Tuple[P, P]:
    """
    Computes the top surface center and size of the desktop for object placement.
    """
    desk_info = get_object_info(desk_shape)
    desktop_subpart = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Desktop top center in world coordinates
    desktop_center = desk_info["center"] + desktop_subpart["center"]
    desktop_top_z = desk_info["center"][2] + desktop_subpart["xyz_max"][2]
    placement_center = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    placement_size = desktop_subpart["size"]

    return placement_center, placement_size


@register()
def colored_pen(pen_id: str, color_name: str, position_tag: str) -> Shape:
    """
    Creates a single colored pen with appropriate keywords.
    """
    return library_call(
        "usd", oid=pen_id, keywords=[f"{color_name}_pen", "pen", color_name, "cylinder", "writing_tool", position_tag]
    )


@register()
def randomly_placed_pens_on_desktop(desk_shape: Shape) -> Shape:
    """
    Places all available benchmark_pen_00* pens randomly on the desktop.
    """
    placement_center, placement_size = get_desktop_placement_area(desk_shape)

    # Available colored pens from benchmark_pen_00* series
    pen_assets_candidates = [
        ("benchmark_pen_000", "pink"),
        ("benchmark_pen_001", "blue"),
        ("benchmark_pen_002", "purple"),
        ("benchmark_pen_003", "yellow"),
        ("benchmark_pen_005", "green"),
        ("benchmark_pen_006", "white"),
    ]
    pen_assets = random.sample(pen_assets_candidates, 5)

    placed_pens = []
    for i, (pen_id, color) in enumerate(pen_assets):
        pen_shape = library_call("colored_pen", pen_id=pen_id, color_name=color, position_tag="")
        pen_info = get_object_info(pen_shape)
        pen_half_size = pen_info["size"] / 2.0

        # Ensure pen stays within desktop bounds
        max_offset_x = (placement_size[0] / 2.0) - pen_half_size[0] - 0.02
        max_offset_y = (placement_size[1] / 2.0) - pen_half_size[1] - 0.02

        offset_x = np.random.uniform(-max_offset_x, max_offset_x)
        offset_y = np.random.uniform(-max_offset_y, max_offset_y)

        # Determine position tag based on world Y coordinate (left/right)
        world_y = placement_center[1] + offset_y
        position_tag = "left" if world_y > placement_center[1] else "right"

        # Re-create pen with correct position tag
        pen_shape = library_call("colored_pen", pen_id=pen_id, color_name=color, position_tag=position_tag)

        # Apply random small rotation around Z-axis for natural look
        pen_center = compute_shape_center(pen_shape)
        pen_shape = transform_shape(
            pen_shape, rotation_matrix(angle=np.random.uniform(-0.3, 0.3), direction=(0, 0, 1), point=pen_center)
        )

        # Translate to final position on desktop
        final_position = [
            placement_center[0] + offset_x,
            placement_center[1] + offset_y,
            placement_center[2],  # sits directly on desktop top
        ]
        pen_shape = transform_shape(pen_shape, translation_matrix(final_position))
        placed_pens.append(pen_shape)

    return concat_shapes(*placed_pens)


@register()
def table_with_colored_pens() -> Shape:
    """
    Main scene function: a white table with 5 distinct colored pens randomly placed on its desktop.
    Note: Only 4 different colored benchmark_pen_00* assets are available in the library.
    """
    desk = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "rectangular", "furniture"]
    )
    pens = library_call("randomly_placed_pens_on_desktop", desk_shape=desk)
    return concat_shapes(desk, pens)


@register()
def root_scene() -> Shape:
    return table_with_colored_pens()
