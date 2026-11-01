# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
import numpy as np

"""
scene_name: insert_straw_into_milk_tea_cup_g2
description: A straw stands in a paper cup on the desktop, and a lidded milk tea cup with a
straw hole on its lid sits beside it. The two cups are arranged along the desktop Y axis with
a 100~150mm gap; the milk tea cup is the one farther from the working arm. The robot must pull
the straw out of the paper cup and insert it into the straw hole on the milk tea cup's lid.

Randomly mirror the layout, assigning about half the variants to each arm
- `straw_cup_on_left` flips one coin to choose a left-arm or right-arm variant. The left-arm
  layout places the paper cup (containing the straw) at +y and the milk-tea cup closer to the
  midline; the right-arm layout mirrors everything to -y.
- Arm selection **does not happen here**: data_gen.py deterministically selects the arm from
  the sign of the objects' actual robot-frame y positions. Never use `robot.judge_arm_to_use()`:
  `robot_api.py:941` sets `THRESHOLD = 0.15` and selects **probabilistically** for |y| <= 0.15
  (P(left) rises linearly from y=-0.15 (0) to +0.15 (1)). Both working points necessarily lie
  within this band (leaving it requires the paper cup at y≈+0.27, while the usable table range
  is only ±0.28 and describes the **base**; loading the overhang tips the table). Therefore,
  randomize the layout only and select the arm deterministically.

Why both cups share one x
- Place the cups along table Y, sharing a sampled X with independent +/-5 mm jitter.
  Their connecting line deviates from the Y axis by at most 10mm, retaining a row along Y.

Why the milk-tea cup has a base yaw
- The straw hole is offset 11.5mm in the milk-tea cup entity's **-y** direction (measured below).
  A -90° rotation about Z maps -y to world -x, facing the hole toward the robot and reducing
  arm extension by 11.5mm during insertion. This is a **small** advantage, not a requirement:
  data_gen always computes the hole from measured runtime poses, independent of this yaw.
  Thus the base angle also receives ±30° jitter. Mirroring does not change the base angle:
  the robot always lies in the -x direction.

The `Aligned.usda` assets' `entity` has a baked +90° X rotation (payload y-up → z-up).
The long axes of all three assets are **Z** in entity coordinates (measured), so the straw
returned by `library_call` is already upright: **do not add any X/Y tilt**, only yaw about Z.
"""

# ===== Assets =====
STRAW_OID = "benchmark_straw_001"
# ⚠ **_fixed is a copy of benchmark_cup_105**, differing only by the entity's added
# `physics:kinematicEnabled = 1` (kinematic fixed body: cannot be pushed or tipped, but still collides).
# Do not modify the original asset: `benchmark_cup_105` is shared by **42 tasks**; enabling this
# flag there would make all their paper cups immovable. Copy the entire directory and use
# `@Aligned.usd@` as a self-contained payload (cross-directory sibling references silently fail on cloud workers → free fall).
PAPER_CUP_OID = "benchmark_cup_105_fixed"  # Straw initially stands in this cup (kinematic copy)
MILK_TEA_OID = "benchmark_cup_132"  # Milk-tea cup with a straw hole in the lid
TABLE_OID = "table_000"

# Measured geometry from the straw and cup meshes. Keep these values
# consistent with the configured asset variants.
PAPER_CUP_INNER_BOTTOM_ABOVE_BASE = 0.00225
# Place the paper cup at its support height and the straw 5 mm above its
# settled position so the initial collision geometry is separated.
CUP_DROP_CLEARANCE = 0.000
STRAW_DROP_CLEARANCE = 0.005
# **The kinematic milk-tea cup needs its own clearance**: it experiences no forces and **will not settle**; flush placement leaves it embedded in the tabletop
# (measured cup12: deep penetration ejected the adjacent paper cup/straw during initialization). Dynamic objects do not have this issue.
MILK_KINEMATIC_CLEARANCE = 0.002
PAPER_CUP_RADIUS = 0.0317  # Outer radius (AABB half-width)
# Milk-tea cup cup_132: AABB 92.78 x 92.66 x 136.76mm
MILK_TEA_RADIUS = 0.0466
# Milk-tea-cup straw hole (entity frame, reference only; data_gen recomputes from measured runtime poses):
#   center (0.000, -0.0115), opening z=+0.06838, inscribed radius 0.011470, hole plate only 0.68mm thick
MILK_TEA_HOLE_LOCAL_XY = (0.000, -0.0115)
# Straw: hollow tube, circumscribed radius 6.000mm, length 150.54mm, uniform diameter, centered axis
STRAW_LEN = 0.15054

# ===== Placement ranges =====
# X: band relative to tabletop center on the robot-facing half (robot base x=-0.7, table bbox x∈[-0.30,+0.30])
X_BAND = (-0.15, -0.03)
X_JITTER = 0.005  # Independent cup x jitter, keeping their connecting line approximately along Y
# Y: milk-tea cup (insertion target) is closer to midline; paper cup (straw source) is closer to working arm.
# ⚠ Sample **milk-tea-cup distance from midline** first, then derive paper-cup position by adding separation. Reversing this
#   (sample paper cup then subtract separation) puts the worst case (paper cup 130mm + separation 150mm) 20mm **opposite** the arm,
#   crossing the midline; measured cross-arm success is much lower. This ordering keeps both cups on the same side.
MILK_CUP_Y_BAND = (0.02, 0.06)  # Milk-tea-cup distance from tabletop center
CUP_GAP_BAND = (0.10, 0.15)  # Cup separation: 10-15 cm.
# Usable table range (**base** dimensions, not tabletop; loading overhang tips table)
TABLE_HALF_EXTENT = 0.28
# Yaw
MILK_TEA_BASE_YAW = -np.pi / 2.0  # Rotate hole (entity -y) to world -x, toward robot
MILK_TEA_YAW_JITTER = np.deg2rad(30.0)
# Axis passed to rotation_matrix must be an ndarray; a tuple crashes the episode
Z_AXIS = np.array([0.0, 0.0, 1.0])

# The generator rounds scene translations to millimetres. Round all samples onto the millimetre grid first; otherwise
# different variants at the same level collapse to the same rounded value and their spread degenerates to 0.
MM = 3


def _bottom_lift_z(shape: Shape, top_z: float) -> float:
    """Place the shape's AABB bottom exactly at top_z.

    `library_call("usd", ...)` already aligns the returned shape's AABB bottom to z=0,
    so only apply the difference here. Do not add any extra "safety lift" margin:
    that would leave the object suspended in free fall.
    """
    info = get_object_info(shape)
    return float(top_z - info["min"][2])


@register()
def insert_straw_into_milk_tea_cup_g2() -> Shape:
    # ---------- Table and tabletop ----------
    table_shape = library_call(
        "usd",
        oid=TABLE_OID,
        keywords=["table", "furniture", "workspace", "desktop_with_legs"],
    )
    table_info = get_object_info(table_shape)
    # table_000's item.py has only bbox / origin shapes and **no** `desktop` subpart. Explicitly request "bbox":
    # get_subpart_info (helper.py:149) **silently** falls back to "bbox" for a missing subpart_id; writing
    # subpart_id="desktop" merely hides that fallback.
    desktop_sub = get_subpart_info(object_id=TABLE_OID, subpart_id="bbox")
    desktop_top_z = float(table_info["center"][2] + desktop_sub["xyz_max"][2])
    desktop_center = table_info["center"] + desktop_sub["center"]
    tcx = float(desktop_center[0])
    tcy = float(desktop_center[1])

    # ---------- Left/right mirroring ----------
    straw_cup_on_left = random.random() < 0.5
    side = 1.0 if straw_cup_on_left else -1.0

    cup_x = round(tcx + random.uniform(*X_BAND), MM)
    paper_x = round(cup_x + random.uniform(-X_JITTER, X_JITTER), MM)
    milk_x = round(cup_x + random.uniform(-X_JITTER, X_JITTER), MM)

    milk_offset = round(random.uniform(*MILK_CUP_Y_BAND), MM)
    gap = round(random.uniform(*CUP_GAP_BAND), MM)
    milk_y = round(tcy + side * milk_offset, MM)
    paper_y = round(milk_y + side * gap, MM)

    paper_tag = "POSITION_TAG_left" if straw_cup_on_left else "POSITION_TAG_right"
    milk_tag = "POSITION_TAG_right" if straw_cup_on_left else "POSITION_TAG_left"

    # ---------- Paper cup (initial straw container) ----------
    # Circular cup: yaw does not affect geometry, so sample its full range.
    paper_yaw = random.uniform(-np.pi, np.pi)
    paper_shape = library_call("usd", oid=PAPER_CUP_OID, keywords=[
        "source_paper_cup", "paper_cup", "cup", "container", "source", paper_tag])
    paper_shape = transform_shape(
        paper_shape, rotation_matrix(paper_yaw, direction=Z_AXIS, point=(0.0, 0.0, 0.0)))
    # ⚠ Paper cup is now **kinematic** (see PAPER_CUP_OID) and **will not settle**, like the milk-tea cup:
    # gravity does not move it, so it stays where placed. **Do not place it flush with the tabletop** (the old dynamic-body approach):
    # flush/penetrating placement stays embedded, and deep penetration can eject nearby dynamic objects (straw) during initialization.
    # Use the same MILK_KINEMATIC_CLEARANCE as the milk-tea cup.
    paper_z = _bottom_lift_z(paper_shape, desktop_top_z) + MILK_KINEMATIC_CLEARANCE
    paper_shape = transform_shape(paper_shape, translation_matrix([paper_x, paper_y, paper_z]))

    # ---------- Milk-tea cup (insertion target) ----------
    milk_yaw = MILK_TEA_BASE_YAW + random.uniform(-MILK_TEA_YAW_JITTER, MILK_TEA_YAW_JITTER)
    milk_shape = library_call("usd", oid=MILK_TEA_OID, keywords=[
        "target_milk_tea_cup", "milk_tea_cup", "lidded_cup", "cup", "container",
        "target_container", milk_tag])
    milk_shape = transform_shape(
        milk_shape, rotation_matrix(milk_yaw, direction=Z_AXIS, point=(0.0, 0.0, 0.0)))
    # Milk-tea-cup asset is **kinematic** (fixed): an sdf collider receives tiny spurious per-frame contact impulses at rest,
    # causing jitter (denser mesh/sdfResolution/damping cannot suppress it), but the hole requires sdf. Only kinematic mode removes it (no forces
    # => 0 jitter); see asset/data_gen comments. Kinematic bodies **will not settle**, so never press their bottom exactly onto
    # top_z: penetration persists (dynamic bodies can rebound). Same reasoning as CUP_DROP_CLEARANCE.
    milk_z = _bottom_lift_z(milk_shape, desktop_top_z) + MILK_KINEMATIC_CLEARANCE
    milk_shape = transform_shape(milk_shape, translation_matrix([milk_x, milk_y, milk_z]))

    # ---------- Straw (inside paper cup) ----------
    # Entity long axis is already Z => upright; add no rotation. Place on paper-cup axis, lower end at **inner cup bottom**.
    # Cup opening inner radius 30.6mm greatly exceeds straw radius 6mm => straw inevitably leans against a wall.
    # Calculated settled tilt is about 7.6° (lower-end offset 14.2mm, rim-contact offset 24.6mm, cup depth 78.3mm),
    # with upper end about 34mm from cup axis: more than 19mm clearance from milk-tea-cup edge, avoiding contact.
    # ⚠ This is only a calculated value. data_gen must **measure** the settled runtime pose before building grasp candidates; do not design motion from this number.
    straw_shape = library_call("usd", oid=STRAW_OID, keywords=[
        "target_straw", "straw", "drinking_straw", "target", "graspable", "rod"])
    # Place straw flush with inner cup bottom too (STRAW_DROP_CLEARANCE=0); see the ejection-rate comparison table there for rationale.
    straw_z = _bottom_lift_z(
        straw_shape,
        desktop_top_z + MILK_KINEMATIC_CLEARANCE   # Kinematic paper cup was raised by this amount; inner bottom rises with it
        + PAPER_CUP_INNER_BOTTOM_ABOVE_BASE + STRAW_DROP_CLEARANCE)
    straw_shape = transform_shape(straw_shape, translation_matrix([paper_x, paper_y, straw_z]))

    # ---------- Assertions ----------
    assert (paper_y > tcy) == straw_cup_on_left, (
        "Paper-cup actual position disagrees with side assignment: paper_y=%.3f tcy=%.3f straw_cup_on_left=%s"
        % (paper_y, tcy, straw_cup_on_left))
    assert abs(paper_y - tcy) > abs(milk_y - tcy), (
        "Milk-tea cup is not closer to midline than paper cup(=farther from working arm): paper_y=%.3f milk_y=%.3f tcy=%.3f"
        % (paper_y, milk_y, tcy))
    # Both cups must lie on the **same side** as the working arm: crossing the insertion point creates cross-arm work with much lower measured success.
    assert (paper_y - tcy) * (milk_y - tcy) > 0.0, (
        "Cups straddle midline: paper_y=%.3f milk_y=%.3f tcy=%.3f => insertion point lies opposite working arm"
        % (paper_y, milk_y, tcy))
    assert abs(milk_y - tcy) >= MILK_CUP_Y_BAND[0] - 1e-9, (
        "Milk-tea cup is only %.3f from midline, below minimum %.3f" % (abs(milk_y - tcy), MILK_CUP_Y_BAND[0]))
    actual_gap = abs(paper_y - milk_y)
    assert CUP_GAP_BAND[0] - 1e-9 <= actual_gap <= CUP_GAP_BAND[1] + 1e-9, (
        "Cup separation %.3f is outside required %.2f~%.2f m range(must also hold after millimetre rounding)"
        % (actual_gap, CUP_GAP_BAND[0], CUP_GAP_BAND[1]))
    assert actual_gap > PAPER_CUP_RADIUS + MILK_TEA_RADIUS, (
        "Cup separation %.3f is less than summed radii %.3f => initial interpenetration"
        % (actual_gap, PAPER_CUP_RADIUS + MILK_TEA_RADIUS))
    assert abs(paper_x - milk_x) <= 2 * X_JITTER + 1e-9, (
        "Cup x difference %.3f exceeds jitter limit => cups no longer form a row along Y" % abs(paper_x - milk_x))
    # Check ranges using distance **relative to tabletop center**, not absolute world coordinates: table center need not be the origin.
    for tag, cx, cy, r in (("paper cup", paper_x, paper_y, PAPER_CUP_RADIUS),
                           ("milk-tea cup", milk_x, milk_y, MILK_TEA_RADIUS)):
        assert abs(cx - tcx) + r < TABLE_HALF_EXTENT, (
            "%s x exceeds table-base usable range: |%.3f-%.3f|+%.3f >= %.3f => loading overhang will tip table"
            % (tag, cx, tcx, r, TABLE_HALF_EXTENT))
        assert abs(cy - tcy) + r < TABLE_HALF_EXTENT, (
            "%s y exceeds table-base usable range: |%.3f-%.3f|+%.3f >= %.3f => loading overhang will tip table"
            % (tag, cy, tcy, r, TABLE_HALF_EXTENT))

    print("[SCENE] straw_cup_on_left=%s | paper_cup=%s xy=(%+.3f,%+.3f) yaw=%+.1f° tag=%s | "
          "milk_tea=%s xy=(%+.3f,%+.3f) yaw=%+.1f° tag=%s | gap=%.3f milk_offset=%.3f | "
          "straw lower end z=%.4f(inner cup bottom) length%.3f | desktop_top_z=%.4f" % (
              straw_cup_on_left, PAPER_CUP_OID, paper_x, paper_y, np.rad2deg(paper_yaw), paper_tag,
              MILK_TEA_OID, milk_x, milk_y, np.rad2deg(milk_yaw), milk_tag,
              actual_gap, milk_offset,
              desktop_top_z + PAPER_CUP_INNER_BOTTOM_ABOVE_BASE, STRAW_LEN, desktop_top_z))

    return concat_shapes(table_shape, paper_shape, milk_shape, straw_shape)


@register()
def root_scene() -> Shape:
    return library_call("insert_straw_into_milk_tea_cup_g2")
