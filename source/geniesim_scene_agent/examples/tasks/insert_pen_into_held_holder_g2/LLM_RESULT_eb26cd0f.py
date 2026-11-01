# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: insert_pen_into_held_holder_g2
description: A table with one pen holder set near the centre, to be steadied/lifted by one hand, and a single pen lying on the opposite side, to be picked by the other hand and inserted into the held holder. One hand holds the holder steady in the air while the other does a single pick-and-insert into it.
"""

# The single pen to insert into the held cup. Every id is verified to have a
# passive/grasp annotation under $SIM_ASSETS/interaction/<id>/interaction.json with
# real grasp poses (the same 6 marker pens the proven put_the_pen_into_the_pen_holder
# task uses), so the data-collection grasp planner finds its passive grasp elements.
# The task is "insert A pen into the cup", so the specific pen model is irrelevant --
# exactly one is sampled per variant.
PEN_CANDIDATES = [
    "benchmark_pen_000",
    "benchmark_pen_001",
    "benchmark_pen_002",
    "benchmark_pen_003",
    "benchmark_pen_005",
    "benchmark_pen_006",
]

# Use benchmark_stationery_021: its 62.0 mm outer diameter fits the G2 gripper
# and its opening provides clearance for the pen. Install its interaction.json
# and grasp poses with the asset. Keep CUP_CANDIDATES aligned with CUP_GEOM.
CUP_CANDIDATES = ["benchmark_stationery_021"]

# Horizontal clearance (m) kept between the cup footprint and the pen footprint,
# enforced by rejection sampling against the real world AABBs. Generous so a pen
# packed close to the cup does not prune the clean head-on grasp approaches (the
# data-collection grasp planner filters grasp poses with a collision-aware AvoidObs
# IK gate). With the cup on the hold side and the pen on the opposite side this keeps
# a wide surface gap at the worst sampled pose.
CUP_PEN_MARGIN = 0.08


def _horiz_radius(shape) -> float:
    """Outer horizontal radius (m) of a placed shape from its world AABB: half of
    the larger of the x/y extents (a conservative circular footprint)."""
    info = get_object_info(shape)
    size = np.asarray(info["size"], dtype=float)
    return 0.5 * max(float(size[0]), float(size[1]))


@register()
def insert_pen_into_held_holder_g2() -> Shape:
    # --- table ---
    table_shape = library_call("usd", oid="table_000", keywords=["table", "furniture", "workspace"])
    table_info = get_object_info(table_shape)
    desktop_sub = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_top_z = table_info["center"][2] + desktop_sub["xyz_max"][2]
    # Anchor XY to the real DESKTOP-surface centre (table object centre + desktop
    # subpart centre), EXACTLY like the sibling held-container / bowl / coaster tasks.
    tc = table_info["center"] + desktop_sub["center"]

    # robot stands at -x facing +x; +y is the robot's LEFT, -y is the robot's RIGHT.
    # One hand LIFTS the cup and holds it up while the OTHER hand inserts a pen into
    # it. To keep both motions on short, comfortable reaches with no awkward cross-
    # body work, the cup and the pen are laid out on a COMMON x line near the MIDDLE
    # of the desktop (slightly toward the robot), separated only in y: the cup just to
    # the HOLD side of centre, the single pen on the OPPOSITE side. The holding arm
    # grasps the cup on its OWN side (its far edge) so the opposite half of the cup
    # mouth is left open for the other hand to insert the pen. The HOLD side is
    # randomised per variant so the dataset is balanced across which arm holds. The
    # side is resolved GEOMETRICALLY at collection time (by world y), not by model.
    hold_left = random.random() < 0.5          # True => left hand holds the cup
    hold_sign = 1.0 if hold_left else -1.0     # cup sits on this side of centre
    place_sign = -hold_sign                    # pen (and inserting arm) on the other

    # A common x band near the middle of the desktop, biased slightly toward the
    # robot (negative x = toward the robot) so both objects are on an easy reach.
    common_x = random.uniform(-0.08, 0.02)

    # --- cup on a common x line, just to the HOLD side of centre in y ---
    cup_oid = random.choice(CUP_CANDIDATES)
    cup_shape = library_call("usd", oid=cup_oid, keywords=[cup_oid.replace("benchmark_", ""), "pen_holder", "holder", "container", "target_container"])
    cup_bottom_offset = -compute_shape_min(cup_shape)[2]
    # free random yaw about world Z -- varies which way the handle faces
    cup_shape = transform_shape(cup_shape, rotation_matrix(random.uniform(0, 2 * np.pi), (0, 0, 1), (0, 0, 0)))
    cup_r = _horiz_radius(cup_shape)

    cup_x = common_x + random.uniform(-0.01, 0.01)
    # Start y offsets from the actual cup radius rather than fixed 0.08~0.13: cup wall 4.5~9.5cm from tabletop centerline.
    # The pool now contains only 021 (outer diameter 62mm), so a fixed offset would suffice; retain the radius-based calculation
    # to tolerate asset-pool size changes, the dimension repeatedly adjusted for this task.
    cup_y = hold_sign * (cup_r + random.uniform(0.045, 0.095))
    cup_placed = transform_shape(
        cup_shape,
        translation_matrix([tc[0] + cup_x, tc[1] + cup_y, desktop_top_z + cup_bottom_offset]),
    )

    # --- single pen on the OPPOSITE (place) side, on the SAME x line as the cup ---
    # The pen loads FLAT on the desktop (Aligned.usda bakes the long axis to the
    # horizontal); we only apply a random yaw about world Z so it lies at a random
    # in-plane heading -- EXACTLY as the proven put_the_pen_into_the_pen_holder task
    # does (which is collected successfully).
    pen_oid = random.choice(PEN_CANDIDATES)
    pen_shape = library_call("usd", oid=pen_oid, keywords=["pen", "stationery", "writing_tool", "on_desk", "to_place"])
    pen_bottom_offset = -compute_shape_min(pen_shape)[2]
    pen_r = _horiz_radius(pen_shape)

    def _clear_of_cup(x, y):
        return np.hypot((tc[0] + cup_x) - x, (tc[1] + cup_y) - y) >= (cup_r + pen_r + CUP_PEN_MARGIN)

    pen_x = pen_y = None
    for _ in range(400):
        x = tc[0] + common_x + random.uniform(-0.02, 0.02)     # same x line as the cup
        y = tc[1] + place_sign * random.uniform(0.08, 0.13)    # opposite (place) side, mirroring the cup
        if _clear_of_cup(x, y):
            pen_x, pen_y = x, y
            break
    if pen_x is None:
        # Fallback: mirror the cup's offset on the place side, on the same x line.
        # The fallback itself must satisfy center distance cup_r+pen_r+margin, or its layout interpenetrates.
        # Only 021 remains (outer radius 0.031), requiring 0.031+0.010+0.08 = 0.121 m; 0.15 leaves clearance.
        pen_x = tc[0] + common_x
        pen_y = tc[1] + place_sign * 0.15

    pen_placed = transform_shape(
        pen_shape,
        translation_matrix([pen_x, pen_y, desktop_top_z + pen_bottom_offset]),
    )
    pen_yaw = random.uniform(0, 2 * np.pi)
    pen_placed = transform_shape(
        pen_placed,
        rotation_matrix(pen_yaw, direction=(0, 0, 1), point=(pen_x, pen_y, desktop_top_z + pen_bottom_offset)),
    )

    return concat_shapes(
        table_shape,
        cup_placed,
        pen_placed,
    )


@register()
def root_scene() -> Shape:
    return library_call("insert_pen_into_held_holder_g2")
