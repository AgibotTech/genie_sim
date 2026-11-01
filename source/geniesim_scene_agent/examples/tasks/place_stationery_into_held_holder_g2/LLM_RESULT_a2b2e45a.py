# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: place_stationery_into_held_holder_g2
description: A table with one pen holder near the centre-back and one upright stationery item on the opposite front side. One hand lifts and steadies the pen holder while the other places the stationery item into it, then the holder is returned to the table.
"""

# One stationery item is sampled per scene. Every candidate is oriented from its runtime AABB so
# its longest axis points up, independent of the asset's authored orientation.
# Keep this pool in sync with STATIONERY_SPECS in data_gen.py and object_urls.json.
# Real AABBs are from each asset's item.py shapes[bbox].size, in metres.
STATIONERY_CANDIDATES = {
    "benchmark_stationery_004": {"label": "glue_bottle"},  # 0.026 x 0.051 x 0.115
    "benchmark_stationery_006": {"label": "glue_stick"},   # 0.028 x 0.028 x 0.100
    "benchmark_stationery_007": {"label": "glue_stick"},   # 0.022 x 0.022 x 0.114
    "benchmark_stationery_011": {"label": "eraser"},        # 0.029 x 0.070 x 0.015
    "benchmark_stationery_012": {"label": "eraser"},        # 0.031 x 0.074 x 0.016
    # The rabbit-ear toy is the non-stationery member of the pool. Install its
    # interaction metadata and grasp poses before running this task.
    "benchmark_toy_024": {"label": "figurine"},             # 0.026 x 0.028 x 0.056
}
# Use pool assets with grasp annotations and geometry compatible with the
# held holder's aperture and available gripper approach.

# The straight-walled pen holder has a measured inner mouth radius of about 0.0298 m and
# a depth of 0.080 m. Every sampled item is upright and fits through that opening.
CUP_CANDIDATES = ["benchmark_stationery_021"]

# Scale the holder within the gripper opening limit. Keep scaled geometry,
# grasp offsets and insertion clearances consistent.
CUP_SCALE_XY = 1.45

# Horizontal clearance between the pen holder and the item. The separation preserves clean
# grasp approaches for both arms while keeping both objects in a comfortable reach band.
CUP_ITEM_MARGIN = 0.08


def _horiz_radius(shape) -> float:
    """Conservative horizontal footprint radius from the transformed world AABB."""
    size = np.asarray(get_object_info(shape)["size"], dtype=float)
    return 0.5 * max(float(size[0]), float(size[1]))


def _orient_longest_axis_up(shape: Shape) -> Shape:
    """Rotate an asset so its longest current AABB axis is aligned with world Z."""
    longest_axis = int(np.argmax(np.asarray(get_object_info(shape)["size"], dtype=float)))
    if longest_axis == 0:
        return transform_shape(
            shape, rotation_matrix(-np.pi / 2, (0, 1, 0), (0, 0, 0))
        )
    if longest_axis == 1:
        return transform_shape(
            shape, rotation_matrix(np.pi / 2, (1, 0, 0), (0, 0, 0))
        )
    return shape


@register()
def place_stationery_into_held_holder_g2() -> Shape:
    table_shape = library_call(
        "usd", oid="table_000", keywords=["table", "furniture", "workspace"]
    )
    table_info = get_object_info(table_shape)
    desktop_sub = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_top_z = table_info["center"][2] + desktop_sub["xyz_max"][2]
    tc = table_info["center"] + desktop_sub["center"]

    # Robot stands at -X facing +X. Randomize which arm holds the pen holder; keep the
    # stationery item on the opposite side so neither arm needs to cross the body.
    hold_left = random.random() < 0.5
    hold_sign = 1.0 if hold_left else -1.0
    place_sign = -hold_sign
    cup_position_tag = "POSITION_TAG_left" if hold_left else "POSITION_TAG_right"
    item_position_tag = "POSITION_TAG_right" if hold_left else "POSITION_TAG_left"
    common_x = random.uniform(-0.08, 0.02)

    cup_oid = random.choice(CUP_CANDIDATES)
    cup_shape = library_call(
        "usd",
        oid=cup_oid,
        keywords=[
            "pen_holder",
            "stationery",
            "container",
            "target_container",
            cup_position_tag,
        ],
    )
    cup_shape = transform_shape(
        cup_shape,
        rotation_matrix(random.uniform(0, 2 * np.pi), (0, 0, 1), (0, 0, 0)),
    )
    # Scale only the two horizontal axes about the asset's own origin (the holder is still at the origin, with yaw only).
    # data_gen.py derives grasp compensation by multiplying local coordinates componentwise by s; another scale center invalidates it.
    # Yaw about Z commutes with [s,s,1] scaling, so either operation order is valid.
    cup_shape = transform_shape(
        cup_shape,
        scale_matrix([CUP_SCALE_XY, CUP_SCALE_XY, 1.0], (0, 0, 0)),
    )
    # compute_shape_min / _horiz_radius use the transformed shape, so both values below
    # already include scaling; bottom alignment and scatter spacing grow accordingly without manual multiplication.
    cup_bottom_offset = -compute_shape_min(cup_shape)[2]
    cup_r = _horiz_radius(cup_shape)
    cup_x = common_x + random.uniform(-0.01, 0.01)
    cup_y = hold_sign * random.uniform(0.08, 0.13)
    cup_placed = transform_shape(
        cup_shape,
        translation_matrix(
            [tc[0] + cup_x, tc[1] + cup_y, desktop_top_z + cup_bottom_offset]
        ),
    )

    item_oid = random.choice(list(STATIONERY_CANDIDATES))
    item_spec = STATIONERY_CANDIDATES[item_oid]
    item_shape = library_call(
        "usd",
        oid=item_oid,
        keywords=[
            "stationery_item_to_place",
            "stationery",
            item_spec["label"],
            "on_desk",
            "to_place",
            item_position_tag,
        ],
    )
    item_shape = _orient_longest_axis_up(item_shape)
    item_shape = transform_shape(
        item_shape,
        rotation_matrix(random.uniform(0, 2 * np.pi), (0, 0, 1), (0, 0, 0)),
    )
    item_bottom_offset = -compute_shape_min(item_shape)[2]
    item_r = _horiz_radius(item_shape)

    def _clear_of_cup(x, y):
        return np.hypot((tc[0] + cup_x) - x, (tc[1] + cup_y) - y) >= (
            cup_r + item_r + CUP_ITEM_MARGIN
        )

    item_x = item_y = None
    for _ in range(400):
        x = tc[0] + common_x + random.uniform(-0.02, 0.02)
        y = tc[1] + place_sign * random.uniform(0.08, 0.13)
        if _clear_of_cup(x, y):
            item_x, item_y = x, y
            break
    if item_x is None:
        item_x = tc[0] + common_x
        item_y = tc[1] + place_sign * 0.13

    item_placed = transform_shape(
        item_shape,
        translation_matrix([item_x, item_y, desktop_top_z + item_bottom_offset]),
    )

    return concat_shapes(table_shape, cup_placed, item_placed)


@register()
def root_scene() -> Shape:
    return library_call("place_stationery_into_held_holder_g2")
