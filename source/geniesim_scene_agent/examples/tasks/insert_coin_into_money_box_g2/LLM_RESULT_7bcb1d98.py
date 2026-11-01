# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
import numpy as np

"""scene_name: insert_coin_into_money_box_g2
description: A table with a coin box and a money box (piggy bank) side by side.
A single arm picks the coin from the tray and inserts it into the money box.

Use the configured collision-trimmed tray and scaled coin asset. Keep
slot coordinates, collision selection and data_gen.py geometry consistent.
"""

COIN_OID = "benchmark_coin_w000"      # Scale variant ×1.03; see the COIN_SCALE section.
COIN_BOX_OID = "benchmark_coin_box_000"
MONEY_BOX_OID = "benchmark_money_box_000"

# Use the collision-mesh convex hull for coin clearance. The scaled coin variant
# keeps its diameter within the 40 mm tray cell and its thickness within the slot.
# Bake uniform scale into the asset variant and keep COIN_RADIUS in data_gen.py
# consistent. Install the referenced asset and its metadata in every runtime.
COIN_SCALE = 1.03

COIN_RADIUS = 0.0187794 * COIN_SCALE
COIN_HALF_T = 0.0010016 * COIN_SCALE

# World-frame footprint half-extents (X/Y) of the coin_box, measured off the
# shape `library_call` actually hands back (printed with compute_shape_min/max):
# x ∈ [−0.110, +0.110], y ∈ [−0.1255, +0.1315]. The Y extent is NOT symmetric
# about the origin — the asset's bbox sits +2.88 mm along Y. Rather than carry
# that asymmetry (and a sign that flips with cb_yaw) through the edge checks,
# take the larger half-width and be symmetric. Costs ~3 mm of X reach; buys not
# having a sign bug in a check whose whole job is to stop the table tipping over.
CB_WORLD_HX = 0.110   # coin_box world half-X  (short axis)
CB_WORLD_HY = 0.1315  # coin_box world half-Y  (long axis, conservative)
MB_WORLD_HX = 0.0675  # money_box world half-X
MB_WORLD_HY = 0.0675  # money_box world half-Y

# ⚠ WHAT `library_call` HANDS BACK, AND WHY IT IS NOT WHAT YOU'D GUESS.
# The shape comes back RESTING ON z = 0 — its bbox min z is 0, not its entity
# origin. So `translation_matrix([x, y, z])` positions the shape's BOTTOM at z,
# while `scene_info.json` / `scene.usda` record the ENTITY ORIGIN, which sits
# higher by the constants below (derived from each asset's bbox: Aligned.usda's
# lowpoly cube for the box, item.py's for the coin).
#   coin_box : bbox centre +3.0971 mm above the origin, half-height 14.4443 mm
#              → origin is 11.3472 mm above the bottom
#   coin     : bbox centred on the origin, half-thickness 1.0 mm
# Getting this wrong put the coin 11 mm too low in every scene — it reads as a
# clean "coin_cz = cb_world_z + seat", but cb_world_z is the box's UNDERSIDE.
CB_SHAPE_ORIGIN_Z = 0.0113472
COIN_SHAPE_ORIGIN_Z = 0.001 * COIN_SCALE   # Scaling is about z=0, so this offset scales too.

# Coin-box interior dimensions use the payload frame: Y is height and Z follows
# the cell. Measure clearances against the collision meshes.
CB_ORIGIN_ABOVE_BOTTOM = 0.011346  # box origin above the desktop once placed
CB_RIM_ABOVE_ORIGIN = 0.017540     # box origin → rim. data_gen's rim_z uses this
CB_FLOOR_ABOVE_ORIGIN = -0.005103  # box origin → cell floor (below the origin)

# Keep the coin-box collider set within the planner mesh-cache budget.
# The intended scene uses 16 box colliders and 28 meshes total against a
# 30-mesh limit. Keep the collision selection aligned with COIN_SLOT_LOCAL_Y
# when re-exporting the asset or changing the selected slot.

# ── where the coin goes ──────────────────────────────────────────────────────
# The chosen slot is the MIDDLE of the middle cell — slot 3 of that cell's 6,
# bounded by ribs collider_022 (face at local_y = −2.672) and collider_023
# (+2.903). 5.575 mm wide. The cell spans local_y ∈ [−19.702, +25.489] (partitions
# c009/c008), so its centre is +2.894 and this slot's centre is +0.116 — the
# closest of the six, and local_y = 0 is a whole millimetre almost exactly on it.
# The full list, with distance from the cell centre:
#     slot 1  −19.702~−12.167  7.535 mm  −18.83 mm   walls 22.6 / 8.2 mm
#     slot 2  −10.207~ −4.632  5.575 mm  −10.31 mm   walls  8.2 / 8.2 mm
#     slot 3   −2.672~ +2.903  5.575 mm   −2.78 mm   walls  8.2 / 8.2 mm  ← used
#     slot 4   +4.864~+10.439  5.575 mm   +4.76 mm   walls  8.2 / 8.2 mm
#     slot 5  +12.399~+17.974  5.575 mm  +12.29 mm   walls  8.2 / 8.2 mm
#     slot 6  +19.934~+25.489  5.555 mm  +19.82 mm   walls  8.2 / 22.6 mm
#
# ⚠ WHAT MOVING OFF SLOT 6 COSTS. Less than it looks, and the reason is worth
# writing down because the obvious version of this argument is WRONG.
#
# Wall height bounds how far the coin can tip before it wedges (2t·cosθ + h·sinθ = w).
# Slot 6's selling point was that ONE of its walls is a full-height partition. But a
# tip has a DIRECTION, and each slot has two walls:
#                      toward −y                toward +y
#       slot 6    rib c025  (8.244)  27.24°   partition c008 (22.643)   9.09°
#       slot 3    rib c022  (8.244)  27.42°   rib c023       (8.244)   27.42°
# So slot 6 never capped the tip at 9.09° — it capped it in one direction only, and
# in the other direction it was 27.24°, i.e. the same as slot 3. Both slots sit in a
# 45.19 mm cell against a 37.56 mm coin, so in BOTH of them a coin that gets past
# its wedge angle can end up flat. The real cost of the move is one direction going
# from 9.09° to 27.42°; the other direction is unchanged.
# Exposure above the rim: 14.92 mm upright, 10.70 mm at 27.4°, 14.44 mm at 9.1°.
#
# And the cap was not load-bearing anyway: the coin does not tip. Measured settled
# tilt in slot 6 was 0.3~4.4° (median 0.5°) from a 5.4° spawn — it rotates BACK, and
# the restoring contact cannot be the tall partition, because leaning on that parks
# it at 9°, not 0.5°. What is left is the V-groove ramps under the coin (c042/c043),
# which run the full length of the column and so are IDENTICAL in every slot.
# That is an inference, not a measurement — see the watch-list in data_gen.py.
# Clearance is unaffected: worst-case is +0.307 mm (ramp c042) at every local_y.
#
# All three offsets below are WHOLE MILLIMETRES so that the generator's
# round(xyz, 3) reproduces them exactly; see the module docstring. Do not
# "improve" any of them to the true measured value:
#   · 75 not 75.236 — the groove's flat floor is 12.36 mm wide and the coin only
#     has 0.31 mm of roll before it touches a ramp, so 0.236 mm off-centre is
#     nearly the whole budget. It is still the right trade: exact-but-off-centre
#     beats centred-but-±1 mm.
#   · 0 not 0.116 — slot 3's centre, to within 0.116 mm.
#   · 14 is the first whole millimetre at or above the seated height, leaving a
#     0.31 mm drop.
COIN_GROOVE_X = 0.075        # payload x: across the groove (flat floor centre)
COIN_SLOT_LOCAL_Y = 0.000    # local_y: disc centre inside the slot (slot 3 centre)
COIN_SEAT_LOCAL_Z = 0.015    # payload y: disc centre above the box origin
# 15 not 14: scaled seating height is 14.312mm; 14 would interpenetrate. This remains the first
# whole millimeter at or above seating height, with a 0.688mm drop (0.324mm at k=1). Recompute when changing COIN_SCALE.

# Spawn the coin vertically at the center of the slot for collision clearance.
# Read the settled pose at runtime when generating grasps.
COIN_LEAN_TILT = 0.0

# Keep the kinematic money box above the dynamic table to avoid depenetration
# pushing the table. The 1 mm gap accommodates millimetre placement rounding.
MB_TABLE_GAP = 0.001

# Derive placement bounds from the 600 x 1000 mm desktop and rotated object
# footprints. Place the boxes side by side along Y, aligning the money-box X
# position with the coin for a short carry path.
DESK_HX = 0.280      # Keep the base half-width for X: there is no overhang in X, so this value was already correct.
DESK_HY = 0.400      # Widen Y (the basis for the original 0.2933 has been disproved).
EDGE_MARGIN = 0.030  # Footprint-to-boundary margin.
# Separate the two footprints along Y. Keep the coin box in its reachable band;
# the money box may sit near the centerline to accommodate the configured gap.
PAIR_CLEARANCE = 0.080

# Place the pair along Y, offset toward the selected arm. The configured gap
# puts the money box slightly across the centerline; validate reachability and
# table stability for sampled layouts.
PAIR_CENTRE_Y = (0.113, 0.117)

# Apply both lower and upper horizontal reach bounds. These are measured
# working envelopes with a 25 mm margin, rather than physical arm limits;
# recheck them when changing the robot pose or target height.
ROBOT_BASE_XY = (-0.700, 0.000)   # Must match robot_init_pose.position in task_info.json.
REACH_PICK = (0.560, 0.785)       # Coin pickup @ z≈0.758; clean grasps measured at 765/780mm.
REACH_INSERT = (0.580, 0.750)     # Coin insertion @ z≈0.96.
# X bands: upper bounds are the tabletop boundary (DESK_HX − EDGE_MARGIN − each half-width); lower bounds leave room.
# REACH_PICK / REACH_INSERT above enforce reachability; X bands only keep objects on the table.
#   cb half-width 131.5 ⇒ upper bound 0.280 − 0.030 − 0.1315 = 0.1185
#   mb half-width ≤95.5 ⇒ upper bound 0.280 − 0.030 − 0.0955 = 0.1545
CB_REACH_X = (-0.100, 0.118)
MB_REACH_X = (-0.150, 0.154)
CELL_DRIFT_X = 0.050              # money_box X drift relative to the coin x coordinate.
# The slot's yaw. Its BASE is π/2, not 0, and that is load-bearing.
#
# In the money_box payload frame the slot's 54.5 mm long axis runs along X and its
# 9.6 mm narrow axis along Z (read off the collider ring: collider_003/004 bound
# the opening at x = ±27.4 mm, collider_001/002 at z = ±4.9 mm). The asset bakes a
# +90°X orient, which leaves X alone and sends Z → −Y, so at mb_yaw = 0 the slot
# lies LONG-ALONG-WORLD-X.
#
# The coin leaves its cell with its face normal along world ±X, so its 37.5 mm
# diameter spans world Y. To drop through, that diameter has to lie along the
# slot's LONG axis — so the long axis must be world Y too, i.e. mb_yaw = π/2.
# Leaving the base at 0 leaves coin and slot crossed at 90°, and data_gen has to
# swing the wrist a quarter turn between grasp and insert to undo it.
MB_YAW_BASE = np.pi / 2
# Jitter about that base is the task's generalisation axis: data_gen.py solves the
# slot axis from the runtime pose and turns the wrist by whatever it finds, so this
# is not a tolerance anything has to fit inside. Kept modest so the wrist turn
# between "coin gripped" and "coin over the slot" stays small.
MB_YAW_JITTER = np.deg2rad(20.0)


def _rotated_half_extents(hx: float, hy: float, yaw: float):
    """Axis-aligned half-extents of an (hx, hy) box after a yaw rotation."""
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    return hx * c + hy * s, hx * s + hy * c


def _solve_layout(tcx: float, tcy: float):
    """Pose the coin tray and money box side by side along Y, both on the same side.

    Return (cb_x, cb_y, cb_side, mb_x, mb_y, mb_yaw, pair_side).
      cb_side ±1 sets the tray's ±90° yaw. The coin has payload x = COIN_GROOVE_X = 75mm,
              world y = cb_y + 75·cb_side ⇒ the coin is offset ±75mm from the tray center.
              Choose the sign toward the centerline to reduce pickup reach and keep the coin on the table.
      pair_side ±1: +1 places the pair to the left (+Y). data_gen selects the arm on the target side.

    A ±90° tray yaw swaps the world footprint X/Y:
      cb_hx = CB_WORLD_HY(131.5), cb_hy = CB_WORLD_HX(110).
    """
    cb_hx, cb_hy = CB_WORLD_HY, CB_WORLD_HX
    bx, by = ROBOT_BASE_XY

    for _ in range(600):
        pair_side = random.choice((1.0, -1.0))
        mb_yaw = MB_YAW_BASE + random.uniform(-MB_YAW_JITTER, MB_YAW_JITTER)
        mb_hx, mb_hy = _rotated_half_extents(MB_WORLD_HX, MB_WORLD_HY, mb_yaw)

        # Side by side along Y: money box inside (near the centerline), coin tray outside.
        sep_y = cb_hy + mb_hy + PAIR_CLEARANCE
        wy = pair_side * random.uniform(*PAIR_CENTRE_Y)
        mb_y = wy - pair_side * sep_y / 2.0      # Inside → closer to the centerline.
        cb_y = wy + pair_side * sep_y / 2.0      # Outside.

        cb_x = random.uniform(*CB_REACH_X)
        # Offset the coin toward the centerline to reduce pickup reach.
        cb_side = -pair_side
        coin_y = cb_y + COIN_GROOVE_X * cb_side
        coin_x = cb_x                             # local_y = 0 ⇒ coin x equals tray-center x.
        mb_x = float(np.clip(coin_x + random.uniform(-CELL_DRIFT_X, CELL_DRIFT_X),
                             MB_REACH_X[0], MB_REACH_X[1]))

        # Reachability: both targets must lie in the measured envelope; check both lower and upper bounds.
        # Allow 2mm for rounding: the generator applies round(xyz,3), so the realized distance can exceed nominal
        # by ~1mm. Without this margin, boundary solutions fail regression (measured 1/20 at 785.0).
        pick_d = float(np.hypot(coin_x - bx, coin_y - by))
        insert_d = float(np.hypot(mb_x - bx, mb_y - by))
        if not (REACH_PICK[0] + 0.002 <= pick_d <= REACH_PICK[1] - 0.002):
            continue
        if not (REACH_INSERT[0] + 0.002 <= insert_d <= REACH_INSERT[1] - 0.002):
            continue

        on_table = (
            abs(cb_x) + cb_hx <= DESK_HX - EDGE_MARGIN
            and abs(cb_y) + cb_hy <= DESK_HY - EDGE_MARGIN
            and abs(mb_x) + mb_hx <= DESK_HX - EDGE_MARGIN
            and abs(mb_y) + mb_hy <= DESK_HY - EDGE_MARGIN
        )
        # Check that the coin-box placement remains on the selected arm side.
        same_side = (coin_y * pair_side > 0.0) and (mb_y * pair_side > -0.040)
        if on_table and same_side:
            return (tcx + cb_x, tcy + cb_y, cb_side,
                    tcx + mb_x, tcy + mb_y, mb_yaw, pair_side)

    # Fallback: window midpoint for the pair center, on the left; midpoint of each X band.
    # The fallback must also remain on the same side, or it silently restores a cross-centerline scene.
    sep_y = CB_WORLD_HX + MB_WORLD_HY + PAIR_CLEARANCE
    wy = sum(PAIR_CENTRE_Y) / 2.0
    cb_x0 = sum(CB_REACH_X) / 2.0
    return (tcx + cb_x0, tcy + wy + sep_y / 2.0, -1.0,
            tcx + sum(MB_REACH_X) / 2.0, tcy + wy - sep_y / 2.0, MB_YAW_BASE, 1.0)


@register()
def insert_coin_into_money_box_g2() -> Shape:
    # ── table ────────────────────────────────────────────────────────────────
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["table", "furniture", "workspace_table", "desktop_with_legs"],
    )
    table_info = get_object_info(table_shape)
    desktop_sub = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_top_z = float(table_info["center"][2] + desktop_sub["xyz_max"][2])
    desktop_center = table_info["center"] + desktop_sub["center"]
    tcx = float(desktop_center[0])
    tcy = float(desktop_center[1])

    # Solve both poses together: the money_box position depends on which way the
    # coin_box yaw throws the coin, so they cannot be sampled independently.
    (cb_cx, cb_cy, cb_side, mb_cx, mb_cy, mb_yaw,
     pair_side) = _solve_layout(tcx, tcy)
    # cb_yaw: ±π/2, sign follows cb_side (which half the cell column is thrown to)
    cb_yaw = (np.pi / 2) * cb_side

    shapes = [table_shape]

    # ── 1. coin_box (yaw ±90°) ─────────────────────────────────────────────────
    cb_shape = library_call(
        "usd",
        oid=COIN_BOX_OID,
        keywords=["coin_box", "coin_tray", "container", "POSITION_TAG_center"],
    )
    cb_bottom = -compute_shape_min(cb_shape)[2]
    cb_world_z = desktop_top_z + cb_bottom

    cb_placed = transform_shape(cb_shape, rotation_matrix(cb_yaw, (0, 0, 1), (0, 0, 0)))
    cb_placed = transform_shape(
        cb_placed, translation_matrix([cb_cx, cb_cy, cb_world_z])
    )
    shapes.append(cb_placed)

    # ── 2. coin (standing in the rib slot at the low end of the middle cell) ───
    # The coin's face normal goes along the CELL COLUMN (the box's local Y), so
    # the disc stands across the cell and its bottom edge drops into a rib slot.
    # coin_yaw = π/2 + cb_yaw makes the coin track the box, so this holds for
    # whichever ±90° variant is drawn; after the yaw the normal lands on world ±X
    # and the diameter spans world Y. No jitter on the coin's yaw — off-axis and
    # the rim no longer fits between the ribs.
    #
    # cb_yaw is exactly ±π/2, so use EXACT 0 / ±1 rather than np.cos/np.sin. Those
    # return 6.1e-17 for cos(π/2), which would leak a sub-femtometre term into the
    # world offsets — harmless in itself, but the whole-millimetre argument in the
    # docstring is cleaner if the offsets really are exactly k/1000.
    cos_y, sin_y = 0.0, cb_side

    def _box_local_world(payload_x, local_y):
        """(payload x, local_y) in the box's frame → world XY."""
        return (cb_cx + payload_x * cos_y - local_y * sin_y,
                cb_cy + payload_x * sin_y + local_y * cos_y)

    coin_cx, coin_cy = _box_local_world(COIN_GROOVE_X, COIN_SLOT_LOCAL_Y)
    # Height. cb_world_z is the box's UNDERSIDE (see CB_SHAPE_ORIGIN_Z), and the
    # translation below likewise sets the coin shape's underside, so both origin
    # offsets have to be paid explicitly. What this buys is that the two RECORDED
    # origins — which is what scene.usda gets and what physics sees — differ by
    # exactly COIN_SEAT_LOCAL_Z:
    #     coin_origin_z = coin_cz + COIN_SHAPE_ORIGIN_Z
    #     box_origin_z  = cb_world_z + CB_SHAPE_ORIGIN_Z
    # and round() of the two therefore differs by exactly a whole millimetre.
    coin_cz = (cb_world_z + CB_SHAPE_ORIGIN_Z
               - COIN_SHAPE_ORIGIN_Z + COIN_SEAT_LOCAL_Z)

    coin_shape_raw = library_call(
        "usd",
        oid=COIN_OID,
        keywords=["coin_target", "coin", "target", "currency", "in_cell"],
    )
    # ⚠ ROTATE ABOUT THE COIN'S OWN ORIGIN, NOT (0, 0, 0). The shape arrives
    # resting on z = 0, so its origin is 1 mm up; spinning it about the world
    # origin swings that 1 mm out sideways and the disc centre lands ~0.996 mm
    # off along X — which then rounds the local_y offset from 23 mm to 24 mm and
    # quietly spends the whole 0.29 mm clearance budget.
    # NO scale_matrix here — the pipeline drops it (see COIN_SCALE). The 1.03 is
    # baked into the asset benchmark_coin_w000 instead, so the shape that
    # library_call hands back is already the scaled one (item.py carries the scaled
    # bbox), and COIN_RADIUS / COIN_HALF_T / COIN_SHAPE_ORIGIN_Z below are the
    # scaled numbers that match it.
    coin_pivot = (0, 0, COIN_SHAPE_ORIGIN_Z)
    # Stand the coin up: baked normal +Z → +X.
    coin_shape = transform_shape(
        coin_shape_raw,
        rotation_matrix(np.pi / 2, (0, 1, 0), coin_pivot),
    )
    # Spawn tilt. Currently 0 (see COIN_LEAN_TILT for why, and for the clearance
    # sweep). Kept as a live call rather than deleted so that changing the constant
    # actually does something; positive tips the top toward the partition.
    coin_shape = transform_shape(
        coin_shape,
        rotation_matrix(COIN_LEAN_TILT, (0, 1, 0), coin_pivot),
    )
    # Yaw: normal +X → the cell-column axis (tracks cb_yaw, so the disc stays
    # square to the ribs whichever ±90° variant the box draws).
    coin_shape = transform_shape(
        coin_shape,
        rotation_matrix(np.pi / 2 + cb_yaw, (0, 0, 1), coin_pivot),
    )
    coin_placed = transform_shape(
        coin_shape,
        translation_matrix([coin_cx, coin_cy, coin_cz]),
    )
    shapes.append(coin_placed)

    # ── 3. money_box (beside the coin_box in Y, resting ON the desktop) ──────
    # Pose comes from _solve_layout. Its yaw is the task's generalisation axis —
    # data_gen.py solves the slot's world axis from the pose at runtime, nothing
    # is hard-coded there — and its position is already checked against the
    # coin_box footprint, the reach band and the table edge.
    #
    # Z sits MB_TABLE_GAP above the desktop, never below it. This box is a
    # KINEMATIC body and the table is a dynamic 200 kg one, so any overlap is
    # resolved by throwing the table — read MB_TABLE_GAP before touching this.
    mb_shape = library_call(
        "usd",
        oid=MONEY_BOX_OID,
        keywords=["money_box", "piggy_bank", "container", "POSITION_TAG_center"],
    )
    mb_bottom = -compute_shape_min(mb_shape)[2]
    mb_world_z = desktop_top_z + mb_bottom + MB_TABLE_GAP

    mb_shape = transform_shape(mb_shape, rotation_matrix(mb_yaw, (0, 0, 1), (0, 0, 0)))
    mb_placed = transform_shape(mb_shape, translation_matrix([mb_cx, mb_cy, mb_world_z]))
    shapes.append(mb_placed)

    return concat_shapes(*shapes)


@register()
def root_scene() -> Shape:
    return library_call("insert_coin_into_money_box_g2")
