# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: place_block_into_held_cup_g2
description: A table with one cup set near the centre-back (to be lifted and steadied by one hand) and a single small building block on the opposite front side (to be picked by the other hand and dropped into the held cup). One hand holds the cup up while the other does a single pick-and-place into it.
"""

# Sample one block from the supported asset pool for the held cup. Use each
# block's collision footprint and height to check aperture and depth clearance.
# Keep BLOCK_CANDIDATES, BLOCK_USDS, ITEM_R_CIRC and object_urls.json aligned;
# install interaction metadata and grasp poses for every selected asset.
BLOCK_CANDIDATES = [
    "benchmark_building_blocks_072",
    "benchmark_building_blocks_076",
    "benchmark_building_blocks_001",
    "benchmark_building_blocks_005",
    "benchmark_building_blocks_007",
    "benchmark_building_blocks_009",
    "benchmark_building_blocks_010",
    "benchmark_building_blocks_015",
    "benchmark_building_blocks_048",
    "benchmark_building_blocks_049",
]

# Use benchmark_cup_107, an opening-up paper cup with grasp annotations.
# Its approximately 76 x 76 x 94 mm geometry is uniformly scaled from
# cup_105; keep aperture, scale and grasp offsets consistent.
CUP_CANDIDATES = [
    "benchmark_cup_107",
]

# Horizontal clearance (m) kept between the cup footprint and the block footprint,
# enforced by rejection sampling against the real world AABBs. Generous for the same reason
# as the sibling held-container tasks: an item packed close to the cup prunes the clean
# head-on grasp approaches (the data-collection grasp planner filters grasp poses with a
# collision-aware AvoidObs IK gate). With the cup on the hold side and the item on the
# opposite side, this keeps a wide surface gap at the worst sampled pose.
CUP_ITEM_MARGIN = 0.08


def _horiz_radius(shape) -> float:
    """Outer horizontal radius (m) of a placed shape from its world AABB: half of
    the larger of the x/y extents (a conservative circular footprint)."""
    info = get_object_info(shape)
    size = np.asarray(info["size"], dtype=float)
    return 0.5 * max(float(size[0]), float(size[1]))


@register()
def place_block_into_held_cup_g2() -> Shape:
    # --- table ---
    table_shape = library_call("usd", oid="table_000", keywords=["table", "furniture", "workspace"])
    table_info = get_object_info(table_shape)
    desktop_sub = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_top_z = table_info["center"][2] + desktop_sub["xyz_max"][2]
    # Anchor XY to the real DESKTOP-surface centre (table object centre + desktop subpart
    # centre), EXACTLY like the sibling held-container / bowl / coaster tasks. Anchoring to
    # the bare table_info["center"] (the object pivot) shifts everything toward the robot.
    tc = table_info["center"] + desktop_sub["center"]

    # robot stands at -x facing +x; +y is the robot's LEFT, -y is the robot's RIGHT.
    # One hand LIFTS the cup and holds it up while the OTHER hand drops an item into it.
    # To keep both motions on short, comfortable reaches with no awkward cross-body work,
    # the cup and the item are laid out on a COMMON x line near the MIDDLE of the desktop
    # (slightly toward the robot), separated only in y: the cup just to the HOLD side of
    # centre, the single item on the OPPOSITE side. The holding arm grasps the cup on its
    # OWN side (its far edge) so the opposite half of the cup mouth is left open for the
    # other hand to drop the item in. The HOLD side is randomised per variant so the
    # dataset is balanced across which arm holds. The side is resolved GEOMETRICALLY at
    # collection time (by world y), not by model.
    hold_left = random.random() < 0.5          # True => left hand holds the cup
    hold_sign = 1.0 if hold_left else -1.0     # cup sits on this side of centre
    place_sign = -hold_sign                    # item (and placing arm) on the other

    # A common x band near the middle of the desktop, biased slightly toward the robot
    # (negative x = toward the robot) so both objects are on an easy reach.
    common_x = random.uniform(-0.08, 0.02)

    # --- cup on a common x line, just to the HOLD side of centre in y ---
    cup_oid = random.choice(CUP_CANDIDATES)
    cup_shape = library_call("usd", oid=cup_oid, keywords=["cup_107", "cup", "container", "target_container"])
    cup_bottom_offset = -compute_shape_min(cup_shape)[2]
    # free random yaw about world Z -- varies which way the handle faces
    cup_shape = transform_shape(cup_shape, rotation_matrix(random.uniform(0, 2 * np.pi), (0, 0, 1), (0, 0, 0)))
    cup_r = _horiz_radius(cup_shape)

    cup_x = common_x + random.uniform(-0.01, 0.01)
    cup_y = hold_sign * random.uniform(0.08, 0.13)            # just to the hold side of centre
    cup_placed = transform_shape(
        cup_shape,
        translation_matrix([tc[0] + cup_x, tc[1] + cup_y, desktop_top_z + cup_bottom_offset]),
    )

    # --- single block on the OPPOSITE (place) side, on the SAME x line as the cup ---
    # Each block loads resting on its own base (Aligned.usda bakes its upAxis -> world Z),
    # so only a random yaw about world Z is applied -- it varies the appearance and the
    # grasp approach without tipping the block onto a face it would not naturally rest on.
    item_oid = random.choice(BLOCK_CANDIDATES)
    item_shape = library_call("usd", oid=item_oid, keywords=["building_blocks", "block", "toy", "on_desk", "to_place"])
    item_bottom_offset = -compute_shape_min(item_shape)[2]
    item_shape = transform_shape(item_shape, rotation_matrix(random.uniform(0, 2 * np.pi), (0, 0, 1), (0, 0, 0)))
    item_r = _horiz_radius(item_shape)

    def _clear_of_cup(x, y):
        return np.hypot((tc[0] + cup_x) - x, (tc[1] + cup_y) - y) >= (cup_r + item_r + CUP_ITEM_MARGIN)

    item_x = item_y = None
    for _ in range(400):
        x = tc[0] + common_x + random.uniform(-0.02, 0.02)     # same x line as the cup
        y = tc[1] + place_sign * random.uniform(0.08, 0.13)    # opposite (place) side, mirroring the cup
        if _clear_of_cup(x, y):
            item_x, item_y = x, y
            break
    if item_x is None:
        # Fallback: mirror the cup's offset on the place side, on the same x line.
        item_x = tc[0] + common_x
        item_y = tc[1] + place_sign * 0.13

    item_placed = transform_shape(
        item_shape,
        translation_matrix([item_x, item_y, desktop_top_z + item_bottom_offset]),
    )

    return concat_shapes(
        table_shape,
        cup_placed,
        item_placed,
    )


@register()
def root_scene() -> Shape:
    return library_call("place_block_into_held_cup_g2")
