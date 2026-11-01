# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

import json
import os
import random
import numpy as np

"""
scene_name: move_shelf_fruit_onto_plate_g2
description:
    A multi-layer supermarket shelf (benchmark_supermarket_shelf_006) standing on
    the floor in front of the robot. On the shelf's TOP layer (layers.json
    index=4) we place 1 apple (the target the robot will move) and 1-2 oranges
    (distractors), with surface-to-surface spacing. The target apple is
    FRONT-BIASED: it is placed first and constrained to the front ~40% of the
    plank depth (the robot-facing edge), because the top layer is at the EDGE of
    the G2's reach - an apple pushed to the back would be both unreachable and
    occluded from the head camera. Oranges stay unconstrained across the full
    depth. On the MIDDLE layer (index=3) we place ONE empty round plate
    (benchmark_plate_*) as the TARGET SURFACE, nudged toward the front edge so the
    robot can set the apple down on it from the front.
    The downstream task picks the top-layer apple and places it DOWN ONTO the
    plate on the middle layer.

    This is a plate variant of shelf_fruit_cross_layer (which used an open storage
    box as the target container). Picking from the OPEN top layer (no plank
    above -> the grasp does not graze the shelf) and placing DOWN onto the plate on
    the reachable middle layer avoids the over-reach topple that an on-the-TOP-layer
    target caused. A plate has NO walls, so the descent onto it is even cleaner than
    dropping into a box: the gripper never has to dip below a box rim under the top
    plank.

Coordinate handling:
    The shelf asset's Aligned.usda applies an internal +90 deg rotation about
    local X (local +Y -> world +Z). usd() pins the loaded shelf's bottom to z=0;
    SHELF_YAW_RAD then yaws it 180 deg about world +Z so its FRONT faces the
    robot, and finally we translate it to SHELF_WORLD_XY. layers.json local
    coords map into world via the shelf's queried world bbox.

Resting (FLOAT FIX):
    usd() builds each object from a proxy bbox (the asset's real mesh bounding
    box) and applies its z_offset lift, so on return the object's proxy-AABB
    BOTTOM sits exactly at z=0 (the "origin" marker is at the bbox CENTRE, not the
    bottom). Every object is rested via _rest_object_on(), which reads the
    object's ACTUAL world AABB (get_object_info -> min/max) and translates it so
    its AABB-bottom sits FLUSH on the plank top (REST_MARGIN = 0.0) and its
    AABB-centre lands at the target xy. The proxy bbox equals the real mesh bbox,
    so flush proxy resting == flush mesh resting: no float.
"""

# ---------------------------------------------------------------------------
# Asset candidate pools
# ---------------------------------------------------------------------------
APPLE_CANDIDATES = [
    "benchmark_apple_000",
    "benchmark_apple_001",
    "benchmark_apple_002",
    "benchmark_apple_003",
    "benchmark_green_apple_000",
    "benchmark_green_apple_001",
]

ORANGE_CANDIDATES = [
    "benchmark_orange_001",
    "benchmark_orange_002",
    "benchmark_orange_004",
]

# Empty round plates (the PLACE target SURFACE). All are flat round dishes whose
# thin axis is baked to world-Z by Aligned.usda (they load lying flat, opening
# up). We use the reach-friendly mid-size plates (diameter ~0.169-0.202 m): the
# plate sits on the MIDDLE plank whose front-to-back DEPTH is only ~0.254 m, and
# the apple must land near the FRONT of it to stay in reach, so a plate much
# wider than ~0.20 m would be forced deep into the shelf (out of reach) or
# overhang the plank. Larger plates (>=0.224 m: 001/005/006/007) and tiny
# saucers (009 ~0.08 m, 010 ~0.05 m) are intentionally excluded. Colour varies
# (white/beige), which gives per-variant variety without changing reachability.
PLATE_CANDIDATES = [
    "benchmark_plate_000",  # white,  ~0.169 m
    "benchmark_plate_002",  # beige,  ~0.187 m
    "benchmark_plate_003",  # white,  ~0.200 m
    "benchmark_plate_004",  # white,  ~0.200 m
    "benchmark_plate_008",  # ~0.202 m
]

SHELF_OID = "benchmark_supermarket_shelf_006"

# layers.json lives under assets/interaction/<asset_id>/. Probe both
# the configured external asset root.
_SHELF_LAYERS_PATH_CANDIDATES = [
    os.path.join(
        os.environ.get("GENIESIM_ASSETS") or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "assets")
        ),
        "interaction", SHELF_OID, "layers.json",
    ),
]

# 5 planks, layers.json index 0=lowest .. 4=highest. The apple+oranges sit on the
# OPEN TOP plank (4) and the empty round plate sits on the MIDDLE plank (3); the
# robot moves the apple DOWN from the top layer onto the plate on the middle layer.
# Picking from the open top + placing down onto the reachable middle plate avoids
# both the under-plank grasp graze and the over-reach topple that an
# on-the-top-layer target produced.
FRUIT_LAYER_INDEX = 4
PLATE_LAYER_INDEX = 3

# benchmark_supermarket_shelf_006 body local-Y half extent (asset size_y/2 ~
# 1.887/2). The shelf's local +Y maps to world +Z with unit scale, so a layer's
# local-Y top maps to world z = shelf_center_z + y_local.
SHELF_LOCAL_Y_HALF_EXTENT = 0.9436

# Shelf XY in world (no table - shelf on the floor in front of the robot, which
# inits at (-0.5, 0, 0) facing +X).
SHELF_WORLD_XY = (0.4, 0.0)

# layers.json reports back_panel at local -X; with 0 yaw the back would face the
# robot. yaw=pi flips the shelf so its FRONT faces world -X (toward the robot).
SHELF_YAW_RAD = np.pi

# Surface-to-surface spacing between any two fruits on the top plank (meters).
FRUIT_SURFACE_GAP = 0.04

# Cluster top-layer fruits toward the centre of the plank's long (world-Y) axis
# so the apple stays reachable and away from the frame posts.
LONG_AXIS_HALF_SPAN = 0.18

# Fraction of the plank DEPTH (front-to-back, world -X to +X), measured from the
# FRONT (robot-facing) edge, within which the TARGET apple is allowed to land.
# The apple sits on the TOP plank, which is at the EDGE of the G2's reach, so it
# MUST be front-biased: an apple pushed to the back of the top layer is both out
# of reach and occluded from the head camera. Distractor oranges are never
# picked, so they stay unconstrained (full depth) for variety.
# The plank depth is only 0.254 m; at 0.4 the apple could land ~11 cm deep (near
# the arm's reach limit / head-cam occlusion). Tightened to 0.25 so the target
# apple sits CONSISTENTLY in the front ~4 cm band (its front rim already ends up
# ~0.5 cm inside the plank front edge), i.e. as close to the robot as it can rest
# without overhanging -> easier, more repeatable grasp.
APPLE_FRONT_DEPTH_FRAC = 0.25

# Inset (m) from the plank's front edge for the plate, so the plate sits near the
# front of the MIDDLE plank. Keeping the plate at the front edge means the robot
# sets the apple down from the front (gripper in the head camera's view, in front
# of the plank above) without reaching deep under that plank.
# Reduced 0.02 -> 0.005 to push the plate ~1.5 cm closer to the robot: the middle
# plank is only 0.254 m deep and the release must stay under the top plank, so the
# nearer the plate the shallower (and more reachable) the place. 0.005 leaves the
# plate front rim 0.5 cm inside the layers.json front edge (which is itself a
# pad_depth=0.03 conservative inset of the real plank) -> no overhang.
PLATE_FRONT_INSET = 0.005

# Rest objects FLUSH on the plank: place each object's world-AABB bottom exactly
# on the plank top, with NO clearance. A positive margin here is precisely what
# makes objects "float" by that amount. 0.0 matches the validated
# `world_z = top - min[2]` flush-rest recipe.
REST_MARGIN = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_layers():
    for cand in _SHELF_LAYERS_PATH_CANDIDATES:
        if os.path.exists(cand):
            with open(cand) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"layers.json missing for {SHELF_OID}. Run generate-shelf-layers first."
    )


def _layer_world_geometry(layer: dict, shelf_world_center: np.ndarray, shelf_world_size: np.ndarray):
    """Map a layers.json entry from the asset LOCAL frame into the WORLD frame.

    Asset local axes after Aligned.usda's +90 deg rotation about local X:
        local +X -> world +X        (length axis preserved)
        local +Y -> world +Z        (the shelf "up", unit scale)
        local +Z -> world -Y        (the shelf "depth")
    Plus SHELF_YAW_RAD about world +Z.

    The vertical map is exact unit scale: world_z = shelf_center_z + y_local
    (the shelf origin sits at local Y=0, i.e. the body's vertical centre). We do
    NOT rescale by the world bbox here - that fractional rescale is the fragile
    part that can mis-place planks; unit scale matches the asset's geometry.
    """
    cos_y = float(np.cos(SHELF_YAW_RAD))
    sin_y = float(np.sin(SHELF_YAW_RAD))

    def _xy_after_rot(cx_l, cz_l):
        # Aligned.usda orient: local +X -> world +X, local +Z -> world -Y.
        pre_x = cx_l
        pre_y = -cz_l
        # yaw about world +Z.
        rx = cos_y * pre_x - sin_y * pre_y
        ry = sin_y * pre_x + cos_y * pre_y
        return rx, ry

    def _z_world(y_local):
        return float(shelf_world_center[2]) + float(y_local)

    corners_world = []
    for cx_l, cy_l, cz_l in layer["corners_local"]:
        rx, ry = _xy_after_rot(float(cx_l), float(cz_l))
        wx = float(shelf_world_center[0]) + rx
        wy = float(shelf_world_center[1]) + ry
        corners_world.append((wx, wy, _z_world(cy_l)))

    xs = [c[0] for c in corners_world]
    ys = [c[1] for c in corners_world]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cl = layer["center_local"]
    cx_w, cy_w = _xy_after_rot(float(cl[0]), float(cl[2]))
    center_world = np.array(
        [
            float(shelf_world_center[0]) + cx_w,
            float(shelf_world_center[1]) + cy_w,
            _z_world(cl[1]),
        ]
    )

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "top_z_world": _z_world(cl[1]),
        "center_world": center_world,
        "kind": layer.get("kind", "solid_plank"),
    }


def _rest_object_on(shape, x_target, y_target, surface_top_z, margin=REST_MARGIN):
    """Translate `shape` so its world AABB bottom rests flush on a surface at
    `surface_top_z` (+margin) and its AABB centre lands at (x_target, y_target).

    Reads the object's ACTUAL world AABB via get_object_info, so it works no
    matter whether usd() left the origin at the object's bottom or its centre -
    this is what prevents the "object floats above the plank" bug.
    """
    info = get_object_info(shape)
    mn = np.asarray(info["min"], dtype=float)
    mx = np.asarray(info["max"], dtype=float)
    cur_cx = 0.5 * (mn[0] + mx[0])
    cur_cy = 0.5 * (mn[1] + mx[1])
    cur_bottom = float(mn[2])
    dx = float(x_target) - cur_cx
    dy = float(y_target) - cur_cy
    dz = (float(surface_top_z) + float(margin)) - cur_bottom
    return transform_shape(shape, translation_matrix([dx, dy, dz]))


def _draw_fruit_counts():
    """1 apple (the moved target) + 1-2 oranges (distractors) on the top layer."""
    return 1, random.randint(1, 2)


def _try_place_one(fruit_radius, placed, x_range, y_range, surface_gap=0.10, max_tries=300):
    for _ in range(max_tries):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        ok = True
        for px, py, pr in placed:
            min_center = pr + fruit_radius + surface_gap
            if (x - px) ** 2 + (y - py) ** 2 < min_center ** 2:
                ok = False
                break
        if ok:
            return x, y
    return None


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@register()
def move_shelf_fruit_onto_plate_g2() -> Shape:
    """Shelf + 1 apple + 1-2 oranges on the TOP layer + 1 empty round plate on
    the MIDDLE layer (target surface). Everything rests flush (no floating)."""
    layers_data = _load_layers()
    plate_layer = layers_data["layers"][PLATE_LAYER_INDEX]
    fruit_layer = layers_data["layers"][FRUIT_LAYER_INDEX]

    # 1. Shelf on the floor: usd() pins bottom to z=0, yaw 180 deg about world
    #    +Z so the front faces the robot, then translate to SHELF_WORLD_XY.
    shelf_shape = library_call(
        "usd",
        oid=SHELF_OID,
        keywords=["target_shelf", "shelf", "storage_furniture", "supermarket_display", "wood", "left"],
    )
    if abs(SHELF_YAW_RAD) > 1e-6:
        shelf_shape = transform_shape(
            shelf_shape,
            rotation_matrix(SHELF_YAW_RAD, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        )
    shelf_shape = transform_shape(
        shelf_shape,
        translation_matrix([SHELF_WORLD_XY[0], SHELF_WORLD_XY[1], 0.0]),
    )

    # 2. Read the shelf's actual world bbox to ground the layer mapping.
    shelf_info = get_object_info(shelf_shape)
    shelf_center = np.asarray(shelf_info["center"], dtype=float)
    shelf_size = np.asarray(shelf_info["size"], dtype=float)

    plate_world = _layer_world_geometry(plate_layer, shelf_center, shelf_size)
    fruit_world = _layer_world_geometry(fruit_layer, shelf_center, shelf_size)

    # 3. Empty round plate on the MIDDLE layer (the TARGET surface). Nudge it
    #    toward the front edge (world -X, the side facing the robot) and centre it
    #    in y, so the apple is set down from the front (gripper visible, clear of
    #    the plank above) rather than deep under the plank.
    plate_oid = random.choice(PLATE_CANDIDATES)
    plate = library_call(
        "usd",
        oid=plate_oid,
        keywords=[
            "target_plate",
            "plate",
            "tableware",
            "kitchenware",
            "dish",
            "on_middle_layer",
            "target",
            "left",
        ],
    )
    plate_info = get_object_info(plate)
    plate_size = np.asarray(plate_info["size"], dtype=float)
    plate_half_x = float(plate_size[0]) / 2.0
    plate_half_y = float(plate_size[1]) / 2.0

    plate_x = plate_world["x_min"] + plate_half_x + PLATE_FRONT_INSET
    plate_x = min(plate_x, plate_world["x_max"] - plate_half_x - 0.005)
    plate_y = float(plate_world["center_world"][1])
    plate_y = max(
        plate_world["y_min"] + plate_half_y + 0.005,
        min(plate_y, plate_world["y_max"] - plate_half_y - 0.005),
    )

    plate_placed = _rest_object_on(plate, plate_x, plate_y, plate_world["top_z_world"])

    # 4. Top-layer fruits: 1 apple (target) + 1-2 oranges (distractors).
    n_apple, n_orange = _draw_fruit_counts()
    apples_chosen = random.sample(APPLE_CANDIDATES, n_apple)
    oranges_chosen = (
        random.sample(ORANGE_CANDIDATES, n_orange)
        if n_orange <= len(ORANGE_CANDIDATES)
        else [random.choice(ORANGE_CANDIDATES) for _ in range(n_orange)]
    )

    fruits_plan = (
        [("apple", oid, i) for i, oid in enumerate(apples_chosen)]
        + [("orange", oid, i) for i, oid in enumerate(oranges_chosen)]
    )
    # Place the TARGET APPLE(S) FIRST so they claim a reachable front-band spot
    # before the distractor oranges fill the rest of the plank. (No shuffle: the
    # apples already lead `fruits_plan`.)

    placed = []
    fruit_shapes = []

    for fruit_type, oid, idx in fruits_plan:
        unique_name = f"{fruit_type}_{idx}"
        fruit = library_call(
            "usd",
            oid=oid,
            keywords=[
                unique_name,
                fruit_type,
                "fruit",
                "edible",
                "on_top_layer",
                "target" if fruit_type == "apple" else "distractor",
            ],
        )
        info = get_object_info(fruit)
        size = np.asarray(info["size"], dtype=float)
        half_x = float(size[0]) / 2.0
        half_y = float(size[1]) / 2.0
        radius = max(half_x, half_y)

        x_lo = fruit_world["x_min"] + half_x + 0.005
        x_hi = fruit_world["x_max"] - half_x - 0.005

        # FRONT-BIAS the target apple into the reachable front band of the plank
        # (front = robot-facing, world -X = x_min side). Distractor oranges keep
        # the full depth.
        if fruit_type == "apple":
            x_hi = x_lo + APPLE_FRONT_DEPTH_FRAC * (x_hi - x_lo)

        y_lo = fruit_world["y_min"] + half_y + 0.005
        y_hi = fruit_world["y_max"] - half_y - 0.005

        # Cluster along the plank's long (world-Y) axis.
        center_y = float(fruit_world["center_world"][1])
        cy_lo = center_y - LONG_AXIS_HALF_SPAN + half_y
        cy_hi = center_y + LONG_AXIS_HALF_SPAN - half_y
        y_lo = max(y_lo, cy_lo)
        y_hi = min(y_hi, cy_hi)

        if x_lo >= x_hi or y_lo >= y_hi:
            continue

        spot = _try_place_one(radius, placed, (x_lo, x_hi), (y_lo, y_hi), surface_gap=FRUIT_SURFACE_GAP)
        if spot is None and fruit_type == "apple":
            # The apple is the task target and MUST exist. If the front band is
            # too crowded to satisfy spacing, drop it at the front-band centre
            # (ignoring spacing) rather than skipping the target entirely.
            spot = (0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi))
        if spot is None:
            continue
        x, y = spot
        placed.append((x, y, radius))

        # FLOAT FIX: rest the fruit flush on the top plank via its real AABB.
        fruit = _rest_object_on(fruit, x, y, fruit_world["top_z_world"])
        fruit_shapes.append(fruit)

    return concat_shapes(shelf_shape, plate_placed, *fruit_shapes)


@register()
def root_scene() -> Shape:
    return library_call("move_shelf_fruit_onto_plate_g2")
