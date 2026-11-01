# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *

import json
import os
import random
import numpy as np

"""
scene_name: move_beverage_from_top_to_middle_shelf_g2
description:
    A multi-layer supermarket shelf (benchmark_supermarket_shelf_006) standing on
    the floor in front of the robot. On the shelf's OPEN TOP layer (layers.json
    index=4) we place ONE beverage container (a can or small bottle) -- the
    SINGLE target the robot will move. The MIDDLE layer (index=3) is left EMPTY:
    it is the DESTINATION. No other objects are placed, so "move the beverage to
    the middle layer" is never ambiguous; variety across the 50 variants comes
    from which beverage is chosen and where on the top plank it lands.

    The target beverage is FRONT-BIASED: it is constrained to the front ~25% of
    the top plank's depth (the robot-facing edge). The top layer is OPEN (no
    plank above it), so the grasp can approach from the front-side without
    ceiling constraints. Keeping the beverage near the front edge keeps the
    gripper in the head camera's view and within the arm's reachable workspace.

    The downstream task picks the top-layer beverage, pulls it horizontally
    clear of the shelf, and sets it DOWN onto the MIDDLE layer in one place
    motion. This is the same direction as the proven move_shelf_fruit_onto_plate_g2
    reference task (top -> middle), so the place is a reachable descent rather
    than a difficult upward push.

Direction (why TOP -> MIDDLE):
    Moving from the OPEN top layer downward onto the middle layer matches the
    proven direction of move_shelf_fruit_onto_plate_g2.
    The grasp accesses the open top (no ceiling graze), the horizontal pull-out
    brings the beverage to WAYPOINT_X in open space, and place_actor_to_target_pose
    descends into the middle compartment via cuRobo's obstacle-aware planner --
    the same motion that successfully moves apples onto the middle plate.

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
    bottom). The beverage is rested via _rest_object_on(), which reads its ACTUAL
    world AABB (get_object_info -> min/max) and translates it so its AABB-bottom
    sits FLUSH on the top plank (REST_MARGIN = 0.0) and its AABB-centre lands
    at the target xy. The shelf asset itself uses an SDF collision approximation
    (updated upstream), which removed the residual float of the
    convex-decomposition hull.
"""

# ---------------------------------------------------------------------------
# Asset candidate pool
# ---------------------------------------------------------------------------
# The TARGET is a SINGLE beverage container (can or small bottle). The robot
# moves it from the open top layer DOWN onto the middle layer.
#
# IMPORTANT (grasp-ability): every beverage here is drawn ONLY from assets that
# already appear in the existing beverage tasks (sort_beverages_by_type_g2,
# pick_object_with_specific_type_beverage_g2, straighten_and_place_beverage_g2,
# ...). Each one has an interaction.json with a `passive` grasp annotation under
# assets/interaction/<id>/, which the data-collection grasp planner needs -- so
# they are all known to be graspable by the G2 gripper. They are short enough
# (tallest ~0.167 m) to stand on the top plank and slim enough (~0.05-0.067 m
# diameter) for the gripper to close on.
BEVERAGE_CANDIDATES = [
    "benchmark_beverage_bottle_001",  # red Coca-Cola can,   ~0.115 m tall, 0.065 dia
    "benchmark_beverage_bottle_002",  # green Sprite can,     ~0.114 m tall, 0.066 dia
    "benchmark_beverage_bottle_003",  # orange can,           ~0.116 m tall, 0.066 dia
    "benchmark_beverage_bottle_004",  # red beverage bottle,  ~0.150 m tall, 0.062 dia
    "benchmark_beverage_bottle_029",  # red Wang Lao Ji can,  ~0.115 m tall, 0.067 dia
    "iros_beverage_bottle_001",       # green Sprite can,     ~0.141 m tall, 0.056 dia
    "genie_beverage_bottle_009",      # black beverage bottle,~0.167 m tall, 0.060 dia
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

# 5 planks, layers.json index 0=lowest .. 4=highest. The beverage starts on the
# OPEN TOP plank (4) and is moved DOWN onto the MIDDLE plank (3).
BEVERAGE_LAYER_INDEX = 4   # start: open top layer
MIDDLE_LAYER_INDEX = 3     # destination, left EMPTY in the scene

# Shelf XY in world (no table - shelf on the floor in front of the robot, which
# inits at (-0.5, 0, 0) facing +X).
SHELF_WORLD_XY = (0.4, 0.0)

# layers.json reports back_panel at local -X; with 0 yaw the back would face the
# robot. yaw=pi flips the shelf so its FRONT faces world -X (toward the robot).
SHELF_YAW_RAD = np.pi

# Cluster the beverage toward the centre of the plank's long (world-Y) axis so
# it stays reachable and away from the frame posts.
LONG_AXIS_HALF_SPAN = 0.10

# Fraction of the plank DEPTH (front-to-back, world -X to +X), measured from the
# FRONT (robot-facing) edge, within which the TARGET beverage is allowed to land.
# The top layer is at the EDGE of the G2's reach; front-biasing keeps it
# reachable and visible in the head camera.
BEVERAGE_FRONT_DEPTH_FRAC = 0.25

# Rest the beverage FLUSH on the plank: place its world-AABB bottom exactly on
# the plank top, with NO clearance.
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
    """
    cos_y = float(np.cos(SHELF_YAW_RAD))
    sin_y = float(np.sin(SHELF_YAW_RAD))

    def _xy_after_rot(cx_l, cz_l):
        pre_x = cx_l
        pre_y = -cz_l
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


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@register()
def move_beverage_from_top_to_middle_shelf_g2() -> Shape:
    """Shelf + 1 beverage (target) on the OPEN TOP layer; the MIDDLE layer is
    left EMPTY as the destination. The beverage rests flush (no floating)."""
    layers_data = _load_layers()
    beverage_layer = layers_data["layers"][BEVERAGE_LAYER_INDEX]

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

    beverage_world = _layer_world_geometry(beverage_layer, shelf_center, shelf_size)

    # 3. ONE beverage on the OPEN TOP layer (the target). Front-biased into the
    #    reachable front band of the plank (front = robot-facing, world -X =
    #    x_min side) and clustered along the plank's long (world-Y) axis, with
    #    per-variant xy jitter for variety.
    beverage_oid = random.choice(BEVERAGE_CANDIDATES)
    beverage = library_call(
        "usd",
        oid=beverage_oid,
        keywords=[
            "target_beverage",
            "beverage",
            "bottle",
            "drink",
            "container",
            "on_top_layer",
            "target",
            "left",
        ],
    )
    info = get_object_info(beverage)
    size = np.asarray(info["size"], dtype=float)
    half_x = float(size[0]) / 2.0
    half_y = float(size[1]) / 2.0

    # Front-band along the plank DEPTH (world -X .. +X), restricted to the front
    # BEVERAGE_FRONT_DEPTH_FRAC of the depth.
    x_lo = beverage_world["x_min"] + half_x + 0.005
    x_hi = beverage_world["x_max"] - half_x - 0.005
    x_hi = x_lo + BEVERAGE_FRONT_DEPTH_FRAC * (x_hi - x_lo)

    # Cluster along the plank's long (world-Y) axis around its centre.
    y_lo = beverage_world["y_min"] + half_y + 0.005
    y_hi = beverage_world["y_max"] - half_y - 0.005
    center_y = float(beverage_world["center_world"][1])
    y_lo = max(y_lo, center_y - LONG_AXIS_HALF_SPAN + half_y)
    y_hi = min(y_hi, center_y + LONG_AXIS_HALF_SPAN - half_y)

    if x_lo >= x_hi:
        x_target = 0.5 * (beverage_world["x_min"] + beverage_world["x_max"])
    else:
        x_target = random.uniform(x_lo, x_hi)
    if y_lo >= y_hi:
        y_target = center_y
    else:
        y_target = random.uniform(y_lo, y_hi)

    # FLOAT FIX: rest the beverage flush on the top plank via its real AABB.
    beverage_placed = _rest_object_on(
        beverage, x_target, y_target, beverage_world["top_z_world"]
    )

    return concat_shapes(shelf_shape, beverage_placed)


@register()
def root_scene() -> Shape:
    return library_call("move_beverage_from_top_to_middle_shelf_g2")
