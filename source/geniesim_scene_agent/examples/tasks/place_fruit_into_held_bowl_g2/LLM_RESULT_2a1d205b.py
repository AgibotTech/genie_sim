# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: place_fruit_into_held_bowl_g2
description: A table with one bowl set near the centre-back (to be steadied by one hand) and a single fruit on the opposite front side (to be picked by the other hand and dropped into the held bowl). One hand holds the bowl steady while the other does a single pick-and-place into it.
"""

# Sample one fruit with passive.grasp.default annotations and real grasp poses
# for placement into the held bowl.
FRUIT_CANDIDATES = [
    "benchmark_peach_000",
    "benchmark_peach_022",
    "benchmark_lemon_028",
    "benchmark_lemon_027",
    "benchmark_apple_000",
    "benchmark_apple_003",
    "benchmark_green_apple_000",
    "benchmark_green_apple_001",
    "benchmark_orange_002",
    "benchmark_orange_004",
    "benchmark_pomegranate_001",
    "benchmark_pomegranate_047",
]
# Mass comes from physics:mass in each asset's Aligned.usda; dimensions come from item.py bbox size.
#   id                     maximum dimension (m)   mass (kg)
#   benchmark_peach_000       0.072   0.17555231
#   benchmark_peach_022       0.058   0.10000000
#   benchmark_lemon_028       0.073   0.11424155
#   benchmark_lemon_027       0.078   0.12800069
#   benchmark_apple_000       0.094   0.42000000
#   benchmark_apple_003       0.101   0.39000000
#   benchmark_green_apple_000 0.090   0.32000000
#   benchmark_green_apple_001 0.071   0.19000000
#   benchmark_orange_002      0.091   0.39000000
#   benchmark_orange_004      0.075   0.22000000
#   benchmark_pomegranate_001 0.093   0.29000000
#   benchmark_pomegranate_047 0.090   0.24000000

# The container. benchmark_bowl_025 is the bowl every existing bowl task
# (place_fruit_into_bowl / stack_bowls_g2 / sort_apples_and_oranges_into_bowls_g2) uses and is
# the asset the spec names for this task; it loads opening-up (Aligned.usda bakes
# upAxis -> Z) so a fruit can be dropped straight in. Exactly one bowl is placed.
BOWL_CANDIDATES = [
    "benchmark_bowl_025",
]

# benchmark_bowl_025: {mass: 1.06; size: [0.156, 0.157, 0.083]}

# Horizontal clearance (m) kept between the bowl footprint and the fruit footprint,
# enforced by rejection sampling against the real world AABBs. Generous for the same
# reason as the sibling tasks: a fruit packed close to the bowl prunes the clean
# head-on grasp approaches (the data-collection grasp planner filters grasp poses
# with a collision-aware AvoidObs IK gate). With the bowl centre-back and the fruit
# on the opposite front side, this keeps a wide surface gap at the worst sampled
# pose.
BOWL_FRUIT_MARGIN = 0.08


def _horiz_radius(shape) -> float:
    """Outer horizontal radius (m) of a placed shape from its world AABB: half of
    the larger of the x/y extents (a conservative circular footprint)."""
    info = get_object_info(shape)
    size = np.asarray(info["size"], dtype=float)
    return 0.5 * max(float(size[0]), float(size[1]))


@register()
def place_fruit_into_held_bowl_g2() -> Shape:
    # --- table ---
    table_shape = library_call("usd", oid="table_000", keywords=["table", "furniture", "workspace"])
    table_info = get_object_info(table_shape)
    desktop_sub = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_top_z = table_info["center"][2] + desktop_sub["xyz_max"][2]
    # Anchor XY to the real DESKTOP-surface centre (table object centre + desktop
    # subpart centre), EXACTLY like the sibling bowl / coaster tasks. Anchoring to
    # the bare table_info["center"] (the object pivot) shifts everything toward the
    # robot.
    tc = table_info["center"] + desktop_sub["center"]

    # robot stands at -x facing +x; +y is the robot's LEFT, -y is the robot's RIGHT.
    # One hand LIFTS the bowl and holds it up while the OTHER hand drops a fruit into
    # it. To keep both motions on short, comfortable reaches with no awkward cross-
    # body work, the bowl and the fruit are laid out on a COMMON x line near the
    # MIDDLE of the desktop (slightly toward the robot), separated only in y: the bowl
    # just to the HOLD side of centre, the single fruit on the OPPOSITE side. The
    # holding arm grasps the bowl on its OWN side (its far edge) so the opposite half
    # of the bowl is left open for the other hand to drop the fruit in. The HOLD side
    # is randomised per variant so the dataset is balanced across which arm holds. The
    # side is resolved GEOMETRICALLY at collection time (by world y), not by model.
    hold_left = random.random() < 0.5          # True => left hand holds the bowl
    hold_sign = 1.0 if hold_left else -1.0     # bowl sits on this side of centre
    place_sign = -hold_sign                    # fruit (and placing arm) on the other

    # A common x band near the middle of the desktop, biased slightly toward the
    # robot (negative x = toward the robot) so both objects are on an easy reach.
    common_x = random.uniform(-0.08, 0.02)

    # --- bowl on a common x line, just to the HOLD side of centre in y ---
    bowl_oid = random.choice(BOWL_CANDIDATES)
    bowl_shape = library_call("usd", oid=bowl_oid, keywords=["bowl_025", "bowl", "container", "target_container"])
    bowl_bottom_offset = -compute_shape_min(bowl_shape)[2]
    # free random yaw about world Z -- harmless on a round bowl, varies appearance
    bowl_shape = transform_shape(bowl_shape, rotation_matrix(random.uniform(0, 2 * np.pi), (0, 0, 1), (0, 0, 0)))
    bowl_r = _horiz_radius(bowl_shape)

    bowl_x = common_x + random.uniform(-0.01, 0.01)
    bowl_y = hold_sign * random.uniform(0.08, 0.13)           # just to the hold side of centre
    bowl_placed = transform_shape(
        bowl_shape,
        translation_matrix([tc[0] + bowl_x, tc[1] + bowl_y, desktop_top_z + bowl_bottom_offset]),
    )

    # --- single fruit on the OPPOSITE (place) side, on the SAME x line as the bowl ---
    fruit_oid = random.choice(FRUIT_CANDIDATES)
    fruit_shape = library_call("usd", oid=fruit_oid, keywords=["fruit", "on_desk", "to_place"])
    fruit_bottom_offset = -compute_shape_min(fruit_shape)[2]
    fruit_shape = transform_shape(fruit_shape, rotation_matrix(random.uniform(0, 2 * np.pi), (0, 0, 1), (0, 0, 0)))
    fruit_r = _horiz_radius(fruit_shape)

    def _clear_of_bowl(x, y):
        return np.hypot((tc[0] + bowl_x) - x, (tc[1] + bowl_y) - y) >= (bowl_r + fruit_r + BOWL_FRUIT_MARGIN)

    fruit_x = fruit_y = None
    for _ in range(400):
        x = tc[0] + common_x + random.uniform(-0.02, 0.02)     # same x line as the bowl
        y = tc[1] + place_sign * random.uniform(0.08, 0.13)    # opposite (place) side, mirroring the bowl
        if _clear_of_bowl(x, y):
            fruit_x, fruit_y = x, y
            break
    if fruit_x is None:
        # Fallback: mirror the bowl's offset on the place side, on the same x line.
        fruit_x = tc[0] + common_x
        fruit_y = tc[1] + place_sign * 0.13

    fruit_placed = transform_shape(
        fruit_shape,
        translation_matrix([fruit_x, fruit_y, desktop_top_z + fruit_bottom_offset]),
    )

    return concat_shapes(
        table_shape,
        bowl_placed,
        fruit_placed,
    )


@register()
def root_scene() -> Shape:
    return library_call("place_fruit_into_held_bowl_g2")
