# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
import numpy as np

"""
scene_name: desk_with_stationery_and_toys
description: A desk with a pen holder, a randomly selected pen, and 0-2 additional toys or items placed in specified proportional regions without any collisions. All objects are randomly rotated around the world Z-axis.
"""

# Desk parameters (provided)
DESK_CENTER = [0.0, 0.0, 0.0]
DESK_SIZE = [0.4, 0.6, 0.2]  # x, y, z dimensions in meters


def find_desktop_surface(table_shape: Shape) -> Tuple[P, P]:
    """
    Calculates the world coordinates of the desktop's top surface center and its size.
    table_000 has a subpart named 'desktop'.
    """
    table_info = get_object_info(table_shape)
    desktop_subpart = get_subpart_info(object_id="table_000", subpart_id="desktop")

    desktop_center_xy = table_info["center"][:2] + desktop_subpart["center"][:2]
    desktop_top_z = table_info["center"][2] + desktop_subpart["xyz_max"][2]
    surface_pos = np.array([desktop_center_xy[0], desktop_center_xy[1], desktop_top_z])
    surface_size = desktop_subpart["size"]

    return surface_pos, surface_size


def sample_position_in_region(
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    size_xy: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Sample a position in desk-local coordinates based on proportional ranges."""
    sx, sy = (DESK_SIZE[0], DESK_SIZE[1]) if size_xy is None else size_xy
    u = np.random.uniform(u_range[0], u_range[1])
    v = np.random.uniform(v_range[0], v_range[1])
    local_x = (u - 0.5) * sx
    local_y = (v - 0.5) * sy
    return local_x, local_y


def world_from_local(
    local_x: float, local_y: float, center: tuple[float, float, float] | None = None
) -> tuple[float, float, float]:
    """Convert desk-local xy to world coordinates."""
    c = center if center is not None else DESK_CENTER
    return c[0] + local_x, c[1] + local_y, c[2]


def check_collision(bbox1: tuple[P, P], bbox2: tuple[P, P]) -> bool:
    """Check if two axis-aligned bounding boxes collide."""
    min1, max1 = bbox1
    min2, max2 = bbox2
    return not (
        max1[0] < min2[0]
        or min1[0] > max2[0]
        or max1[1] < min2[1]
        or min1[1] > max2[1]
        or max1[2] < min2[2]
        or min1[2] > max2[2]
    )


def get_bounding_box(shape: Shape) -> tuple[P, P]:
    """Get the world-space bounding box of a shape."""
    info = get_object_info(shape)
    return info["min"], info["max"]


def place_and_rotate_object(obj_shape: Shape, world_pos: tuple[float, float, float]) -> Shape:
    """
    Places an object at world_pos and applies a random rotation around the world Z-axis.
    Rotation is applied after translation, around the object's new center.
    """
    translated = transform_shape(obj_shape, translation_matrix(world_pos))
    center = compute_shape_center(translated)
    angle = np.random.uniform(0, 2 * math.pi)
    rotated = transform_shape(translated, rotation_matrix(angle=angle, direction=(0, 0, 1), point=center))
    return rotated


@register()
def place_all_objects_on_desk() -> Shape:
    """
    Places all objects (pen holder, one pen, and 0-2 extra items) with full collision checking.
    All objects are randomly rotated around the world Z-axis after placement.
    Uses proportional regions and ensures no collisions via iterative sampling.
    """
    # Load table_000 at DESK_CENTER and get desktop surface for placement
    table_shape = library_call("usd", oid="table_000", keywords=["minimalist_table", "white", "furniture", "center"])
    table_shape = transform_shape(table_shape, translation_matrix(DESK_CENTER))
    surface_pos, surface_size = find_desktop_surface(table_shape)
    desk_center = (float(surface_pos[0]), float(surface_pos[1]), float(surface_pos[2]))
    desk_size_xy = (float(surface_size[0]), float(surface_size[1]))

    all_shapes = [table_shape]
    all_bboxes = []

    # 1. Place pen holder in region x∈[0.2,0.6], y∈[0.3,0.7]
    pen_holder = library_call(
        "usd",
        oid="benchmark_stationery_014",
        keywords=["pen_holder", "stationery", "cylindrical", "desk_organizer", "right"],
    )

    placed = False
    for _ in range(30):
        local_x, local_y = sample_position_in_region((0.2, 0.6), (0.3, 0.7), desk_size_xy)
        world_pos = world_from_local(local_x, local_y, desk_center)
        candidate = place_and_rotate_object(pen_holder, world_pos)
        cand_bbox = get_bounding_box(candidate)

        # First object, no collision possible
        all_shapes.append(candidate)
        all_bboxes.append(cand_bbox)
        placed = True
        break

    if not placed:
        # Fallback
        local_x = (0.4 - 0.5) * desk_size_xy[0]
        local_y = (0.5 - 0.5) * desk_size_xy[1]
        world_pos = world_from_local(local_x, local_y, desk_center)
        candidate = place_and_rotate_object(pen_holder, world_pos)
        all_shapes.append(candidate)
        all_bboxes.append(get_bounding_box(candidate))

    # 2. Place one pen in region x∈[0.05,0.5], y∈[0.0,1.0]
    pen_ids = [
        "benchmark_pen_006",
        "benchmark_pen_005",
        "benchmark_pen_003",
        "benchmark_pen_002",
        "benchmark_pen_001",
        "benchmark_pen_000",
    ]
    color_map = {
        "benchmark_pen_000": "red",
        "benchmark_pen_001": "blue",
        "benchmark_pen_002": "purple",
        "benchmark_pen_003": "yellow",
        "benchmark_pen_005": "green",
        "benchmark_pen_006": "black",
    }

    chosen_pen_id = random.choice(pen_ids)
    pen_color = color_map.get(chosen_pen_id, "marker")

    pen = library_call(
        "usd", oid=chosen_pen_id, keywords=[f"{pen_color}_pen", "pen", "cylinder", "writing_tool", "left"]
    )

    placed = False
    for _ in range(30):
        local_x, local_y = sample_position_in_region((0.05, 0.5), (0.0, 1.0), desk_size_xy)
        world_pos = world_from_local(local_x, local_y, desk_center)
        candidate = place_and_rotate_object(pen, world_pos)
        cand_bbox = get_bounding_box(candidate)

        collision = any(check_collision(cand_bbox, bbox) for bbox in all_bboxes)
        if not collision:
            all_shapes.append(candidate)
            all_bboxes.append(cand_bbox)
            placed = True
            break

    if not placed:
        local_x = (0.3 - 0.5) * desk_size_xy[0]
        local_y = (0.5 - 0.5) * desk_size_xy[1]
        world_pos = world_from_local(local_x, local_y, desk_center)
        candidate = place_and_rotate_object(pen, world_pos)
        all_shapes.append(candidate)
        all_bboxes.append(get_bounding_box(candidate))

    # 3. Place 0-2 extra items in region x∈[0.6,1.0], y∈[0.0,1.0]
    num_extra = random.randint(0, 2)
    if num_extra > 0:
        item_ids = [
            "benchmark_toy_car_018",
            "benchmark_toy_plane_029",
            "benchmark_rubik_cube_001",
            "benchmark_garage_kit_064",
        ]

        for i in range(num_extra):
            item_id = random.choice(item_ids)
            item = library_call("usd", oid=item_id, keywords=[f"extra_item_{i}", "toy", "small_object", "right"])

            placed = False
            for _ in range(30):
                local_x, local_y = sample_position_in_region((0.6, 1.0), (0.0, 1.0), desk_size_xy)
                world_pos = world_from_local(local_x, local_y, desk_center)
                candidate = place_and_rotate_object(item, world_pos)
                cand_bbox = get_bounding_box(candidate)

                collision = any(check_collision(cand_bbox, bbox) for bbox in all_bboxes)
                if not collision:
                    all_shapes.append(candidate)
                    all_bboxes.append(cand_bbox)
                    placed = True
                    break

            if not placed:
                v_fallback = 0.2 + i * 0.35
                if v_fallback > 0.8:
                    v_fallback = 0.8
                local_x = (0.85 - 0.5) * desk_size_xy[0]
                local_y = (v_fallback - 0.5) * desk_size_xy[1]
                world_pos = world_from_local(local_x, local_y, desk_center)
                candidate = place_and_rotate_object(item, world_pos)
                all_shapes.append(candidate)
                all_bboxes.append(get_bounding_box(candidate))

    return concat_shapes(*all_shapes)


@register()
def desk_with_stationery_and_toys() -> Shape:
    return place_all_objects_on_desk()


@register()
def root_scene() -> Shape:
    return desk_with_stationery_and_toys()
