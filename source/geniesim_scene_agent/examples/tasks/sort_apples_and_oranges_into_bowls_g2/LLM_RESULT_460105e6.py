# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: sort_apples_and_oranges_into_bowls_g2
description: A table with two bowls (one on the left side, one on the right side) and an apple and an orange placed near the center-front, for sorting the apple into the left bowl and the orange into the right bowl with separate arms (no crossing).
"""

APPLE_CANDIDATES = [
    "benchmark_apple_000",
    "benchmark_apple_001",
    "benchmark_apple_002",
    "benchmark_apple_003",
]

ORANGE_CANDIDATES = [
    "benchmark_orange_001",
    "benchmark_orange_002",
    "benchmark_orange_004",
]

BOWL_CANDIDATES = [
    "benchmark_bowl_000",
    "benchmark_bowl_001",
    "benchmark_bowl_002",
    "benchmark_bowl_003",
    "benchmark_bowl_004",
    "benchmark_bowl_025",
]

# Keep horizontal clearance between each fruit and the bowl footprints
# so grasp approaches remain reachable at the sampled corners.
FRUIT_BOWL_MARGIN = 0.06
FRUIT_FRUIT_MARGIN = 0.04


def _horiz_radius(shape) -> float:
    """Outer horizontal radius (m) of a placed shape from its world AABB:
    half of the larger of the x/y extents (bowls/fruits are ~round)."""
    info = get_object_info(shape)
    size = np.asarray(info["size"], dtype=float)
    return 0.5 * max(float(size[0]), float(size[1]))


def _center_xy(shape):
    info = get_object_info(shape)
    c = np.asarray(info["center"], dtype=float)
    return float(c[0]), float(c[1])


@register()
def sort_apples_and_oranges_into_bowls_g2() -> Shape:
    # --- table ---
    table_shape = library_call("usd", oid="table_000", keywords=["table", "furniture", "workspace"])
    table_info = get_object_info(table_shape)
    desktop_sub = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_top_z = table_info["center"][2] + desktop_sub["xyz_max"][2]
    tc = table_info["center"]

    # The robot stands at -X facing +X; +Y is left. Place bowls in the
    # configured outer bands and fruits near the front-center pickup band.

    # --- two bowls: pushed to BACK (+x) and SIDES (large |y|) ---
    left_bowl_oid, right_bowl_oid = random.sample(BOWL_CANDIDATES, 2)

    left_bowl_shape = library_call("usd", oid=left_bowl_oid, keywords=["bowl_left", "bowl", "container", "left"])
    left_bowl_bottom = -compute_shape_min(left_bowl_shape)[2]
    left_bowl_x = random.uniform(0.09, 0.16)
    left_bowl_y = random.uniform(0.25, 0.31)
    left_bowl_placed = transform_shape(
        left_bowl_shape,
        translation_matrix([tc[0] + left_bowl_x, tc[1] + left_bowl_y, desktop_top_z + left_bowl_bottom]),
    )

    right_bowl_shape = library_call("usd", oid=right_bowl_oid, keywords=["bowl_right", "bowl", "container", "right"])
    right_bowl_bottom = -compute_shape_min(right_bowl_shape)[2]
    right_bowl_x = random.uniform(0.09, 0.16)
    right_bowl_y = random.uniform(-0.31, -0.25)
    right_bowl_placed = transform_shape(
        right_bowl_shape,
        translation_matrix([tc[0] + right_bowl_x, tc[1] + right_bowl_y, desktop_top_z + right_bowl_bottom]),
    )

    # Real world AABB footprints of the two placed bowls (collision check basis).
    lbx, lby = _center_xy(left_bowl_placed)
    rbx, rby = _center_xy(right_bowl_placed)
    left_bowl_r = _horiz_radius(left_bowl_placed)
    right_bowl_r = _horiz_radius(right_bowl_placed)

    # --- apple near center-left, orange near center-right (in FRONT of the bowls,
    #     toward the robot). Each is rotated by a random yaw, then its world XY is
    #     rejection-sampled so its footprint clears BOTH bowls (and the other
    #     fruit) by at least the margin. ---
    apple_oid = random.choice(APPLE_CANDIDATES)
    orange_oid = random.choice(ORANGE_CANDIDATES)

    apple_shape = library_call("usd", oid=apple_oid, keywords=["apple", "fruit", "on_desk", "left"])
    apple_bottom = -compute_shape_min(apple_shape)[2]
    apple_yaw = random.uniform(0, 2 * np.pi)
    apple_rot = transform_shape(apple_shape, rotation_matrix(apple_yaw, (0, 0, 1), (0, 0, 0)))
    apple_r = _horiz_radius(apple_rot)

    orange_shape = library_call("usd", oid=orange_oid, keywords=["orange", "fruit", "on_desk", "right"])
    orange_bottom = -compute_shape_min(orange_shape)[2]
    orange_yaw = random.uniform(0, 2 * np.pi)
    orange_rot = transform_shape(orange_shape, rotation_matrix(orange_yaw, (0, 0, 1), (0, 0, 0)))
    orange_r = _horiz_radius(orange_rot)

    def _clear_of_bowls(x, y, fruit_r):
        for (bx, by, br) in ((lbx, lby, left_bowl_r), (rbx, rby, right_bowl_r)):
            if np.hypot(x - bx, y - by) < (br + fruit_r + FRUIT_BOWL_MARGIN):
                return False
        return True

    def _sample_fruit_xy(x_lo, x_hi, y_lo, y_hi, fruit_r, avoid=None):
        """Sample a desktop XY (world frame) for a fruit, rejecting any pose that
        overlaps a bowl footprint or the already-placed other fruit. Falls back to
        the safe front-most position on that side if no sample passes."""
        for _ in range(400):
            x = tc[0] + random.uniform(x_lo, x_hi)
            y = tc[1] + random.uniform(y_lo, y_hi)
            if not _clear_of_bowls(x, y, fruit_r):
                continue
            if avoid is not None:
                ax, ay, ar = avoid
                if np.hypot(x - ax, y - ay) < (ar + fruit_r + FRUIT_FRUIT_MARGIN):
                    continue
            return x, y
        # Fallback: front-most, side-most corner of the band (always bowl-clear).
        return tc[0] + x_lo, tc[1] + (y_lo if y_lo > 0 else y_hi)

    # Place fruits in the configured front-center band, with enough clearance
    # from both bowl footprints for independent grasps.
    apple_x, apple_y = _sample_fruit_xy(-0.17, -0.08, 0.07, 0.14, apple_r)
    orange_x, orange_y = _sample_fruit_xy(
        -0.17, -0.08, -0.14, -0.07, orange_r, avoid=(apple_x, apple_y, apple_r)
    )

    apple_placed = transform_shape(
        apple_rot,
        translation_matrix([apple_x, apple_y, desktop_top_z + apple_bottom]),
    )
    orange_placed = transform_shape(
        orange_rot,
        translation_matrix([orange_x, orange_y, desktop_top_z + orange_bottom]),
    )

    return concat_shapes(
        table_shape,
        left_bowl_placed,
        right_bowl_placed,
        apple_placed,
        orange_placed,
    )


@register()
def root_scene() -> Shape:
    return library_call("sort_apples_and_oranges_into_bowls_g2")
