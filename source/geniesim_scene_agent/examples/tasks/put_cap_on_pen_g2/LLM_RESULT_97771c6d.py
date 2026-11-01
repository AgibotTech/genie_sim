# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
import numpy as np

"""
scene_name: put_cap_on_pen_g2
description: A table with exactly one marker pen and exactly one pen cap lying
flat on the desktop, one on each side of the workspace. The robot picks the pen
with one hand and the cap with the other, brings their long axes onto a common
line, and pushes the cap onto the pen tip.

New task design with no source JSON. Pool sizes and placement ranges follow the
geometry and reachability constraints below rather than another bundle.

Target uniqueness
- Exactly one pen and one cap, with no same-category distractors: data_gen uses
  `target_pen` / `target_cap` keywords, each of which must match exactly once.
- Deliberately omit pen holders (`benchmark_stationery_*`): a holder asset whose
  name starts with `benchmark_pen_` could be treated as a pen candidate by
  pen_cap_attractor's sibling search. This scene has only one pen, avoiding ambiguity.

Why yaw is not uniformly random over 0~2π
- After grasping, pen and cap must rotate into an assembly pose with collinear long
  axes. The framework reorients by rigid rotation through the difference between
  current and target orientations; large rotations can reject the entire IK batch.
  Initial yaw therefore jitters ±25° around the assembly direction (world Y).
  Both orientations must point the pen tip toward the cap's side, avoiding an
  approximately 180° reorientation. This is a reachability constraint.
- The asset `Aligned.usda` entity has a baked +90° X rotation (payload y-up → z-up).
  The X long axis is unaffected: apply Z yaw only, with no additional X/Y tilt.

Geometry (asset entity local frame, long axis X; checked for all 12 pens and 12 caps)
- Pen: total length 156mm. Tip endpoint at local −X; thin tip radius 4.51mm,
  length 23.4mm; barrel radius 8.77mm.
- Cap: total length 38.7mm. Mouth at local +X; cavity radius 8.35mm, depth 38.2mm.
- Barrel radius 8.77mm > cavity radius 8.35mm: only the thin tip enters the cap.
  This enables the task and explains data_gen's fixed 22mm insertion depth.

Assembly orientation ⇒ initial reference yaw
  When seated, cap mouth direction (+X_cap) and barrel direction (+X_pen) agree.
  With pen at −Y and cap at +Y, the tip points +Y: pen −X points +Y ⇒ Rz(−90°).
  Cap +X must point −Y ⇒ the same Rz(−90°). Both reference angles match.
  Swapping left and right gives +90°.
"""

# Sample pen and cap as a configured pair so their geometry remains compatible.
CAP_PEN_PAIRS = [
    ("benchmark_cap_w000", "benchmark_pen_010"),
    ("benchmark_cap_w001", "benchmark_pen_016"),
    ("benchmark_cap_w002", "benchmark_pen_021"),
    ("benchmark_cap_w003", "benchmark_pen_017"),
    ("benchmark_cap_w004", "benchmark_pen_018"),
    ("benchmark_cap_w005", "benchmark_pen_015"),
    ("benchmark_cap_w006", "benchmark_pen_014"),
    ("benchmark_cap_w007", "benchmark_pen_013"),
    ("benchmark_cap_w008", "benchmark_pen_012"),
    ("benchmark_cap_w009", "benchmark_pen_011"),
    ("benchmark_cap_w010", "benchmark_pen_019"),
    ("benchmark_cap_w011", "benchmark_pen_020"),
]

# Validate the pairing table: exactly 12 pairs, no repeated caps or pens, and full coverage of both asset pools.
# Missing or duplicate entries silently leave some assets unsampled; generation logs would not reveal this.
_caps = [c for c, _ in CAP_PEN_PAIRS]
_pens = [p for _, p in CAP_PEN_PAIRS]
assert len(CAP_PEN_PAIRS) == 12, "Pairing table must contain 12 pairs, got %d" % len(CAP_PEN_PAIRS)
assert len(set(_caps)) == 12, "Pairing table contains duplicate caps"
assert len(set(_pens)) == 12, "Pairing table contains duplicate pens"
assert set(_caps) == {"benchmark_cap_w%03d" % i for i in range(0, 12)},\
    "Pairing table does not cover cap_000~011"
assert set(_pens) == {"benchmark_pen_%03d" % i for i in range(10, 22)},\
    "Pairing table does not cover pen_010~021"

TABLE_OID = "table_000"

# Sample X from the absolute X_BAND relative to the table center.
# Sample Y from the normalized desktop-width band.
X_BAND = (-0.20, 0.00)

# Use normalized center ± half-width bands in Y to separate the left and right sides.
WS_SIZE_Y = 0.60

# Initial yaw jitter relative to the assembly reference angle.
YAW_JITTER = np.deg2rad(25.0)

# Pass the rotation_matrix axis as an ndarray, not a tuple; keep both files consistent.
Z_AXIS = np.array([0.0, 0.0, 1.0])


def _bottom_lift_z(shape: Shape, top_z: float) -> float:
    """Place the shape's AABB bottom exactly at top_z.

    Do not add a generic lift margin: `usd()` already aligns the AABB bottom to
    z=0. Move that current bottom to top_z; an arbitrary lift can leave the
    object floating or embedded in the table.
    """
    info = get_object_info(shape)
    return float(top_z - info["min"][2])


@register()
def put_cap_on_pen_g2() -> Shape:
    # ---------- Table and tabletop ----------
    table_shape = library_call(
        "usd",
        oid=TABLE_OID,
        keywords=["table", "furniture", "workspace", "desktop_with_legs"],
    )
    table_info = get_object_info(table_shape)
    # table_000 item.py has only bbox / origin shapes, with no `desktop` subpart.
    # Select bbox explicitly (its top is the tabletop), rather than requesting subpart_id="desktop" and relying on
    # get_subpart_info's silent fallback, which would misleadingly imply that the subpart exists.
    desktop_sub = get_subpart_info(object_id=TABLE_OID, subpart_id="bbox")
    desktop_top_z = float(table_info["center"][2] + desktop_sub["xyz_max"][2])
    desktop_center = table_info["center"] + desktop_sub["center"]
    tcx = float(desktop_center[0])
    tcy = float(desktop_center[1])

    # X uses the absolute X_BAND; Y uses the normalized desktop-width band.
    x_lo_abs = tcx + X_BAND[0]
    x_hi_abs = tcx + X_BAND[1]
    ws_y_min = tcy - WS_SIZE_Y / 2.0

    def to_world(ny_min, ny_max):
        """Convert normalized Y bands to world coordinates; use absolute X bands directly."""
        return (
            x_lo_abs,
            x_hi_abs,
            ws_y_min + ny_min * WS_SIZE_Y,
            ws_y_min + ny_max * WS_SIZE_Y,
        )

    placed = []  # [(cx, cy, hx, hy), ...] actual rotated AABB half-extents of already placed objects.

    def sample_xy(x_lo, x_hi, y_lo, y_hi, hx, hy, clearance, attempts=400):
        """Sample a center within the bounds while avoiding placed objects.

        hx/hy must come from the actual rotated AABB (get_object_info), not
        item.py's original box dimensions: a ~90° pen rotation swaps X/Y spans.
        On rejection, fall back to the known safe band center rather than
        retaining the last rejected sample.
        """
        for _ in range(attempts):
            cx = (random.uniform(x_lo + hx, x_hi - hx)
                  if x_hi - x_lo > 2 * hx else 0.5 * (x_lo + x_hi))
            cy = (random.uniform(y_lo + hy, y_hi - hy)
                  if y_hi - y_lo > 2 * hy else 0.5 * (y_lo + y_hi))
            ok = True
            for (pcx, pcy, phx, phy) in placed:
                if (abs(cx - pcx) < hx + phx + clearance
                        and abs(cy - pcy) < hy + phy + clearance):
                    ok = False
                    break
            if ok:
                return cx, cy
        return 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)

    # ---------- Left/right assignment ----------
    # Randomize the pen side and put the cap opposite it, so each arm holds one object and layouts vary.
    # Choose the side before sampling, so POSITION_TAG can go directly into keywords; labels must match the final
    # position, which the band guarantees and the final assertions verify.
    pen_on_left = random.random() < 0.5
    # Assembly reference yaw (derived in the docstring): pen on left → +90°, pen on right → −90°.
    base_yaw = (np.pi / 2.0) if pen_on_left else (-np.pi / 2.0)

    # Use X_BAND for X and the selected normalized band for Y. Keep pen and cap
    # near the table X axis with their configured separation.
    if pen_on_left:
        pen_band = (0.50, 0.82)
        cap_band = (0.18, 0.47)
        pen_tag, cap_tag = "POSITION_TAG_left", "POSITION_TAG_right"
    else:
        pen_band = (0.18, 0.50)
        cap_band = (0.53, 0.82)
        pen_tag, cap_tag = "POSITION_TAG_right", "POSITION_TAG_left"

    def place_rod(oid, keywords, band, clearance):
        """Place a slender object (pen or cap) at assembly reference yaw ± jitter.

        Rotate about its own origin around Z, measure the rotated AABB, then
        translate. Reversing the order rotates about the wrong center and
        measures the unrotated span.
        """
        shape = library_call("usd", oid=oid, keywords=keywords)
        yaw = base_yaw + random.uniform(-YAW_JITTER, YAW_JITTER)
        shape = transform_shape(
            shape, rotation_matrix(yaw, direction=Z_AXIS, point=(0.0, 0.0, 0.0))
        )
        info = get_object_info(shape)
        hx = float(info["size"][0]) / 2.0
        hy = float(info["size"][1]) / 2.0
        x_lo, x_hi, y_lo, y_hi = to_world(*band)
        # The X band is only 200mm wide; after ±25° yaw jitter, the pen can span ~82mm in X.
        # Verify that it fits: if 2*hx exceeds the band width, sample_xy silently falls back to
        # the band center, giving every variant the same x and silently disabling randomization.
        assert x_hi - x_lo > 2 * hx, (
            "X band width %.3f cannot fit rotated X span %.3f (%s); randomization would silently collapse to a fixed value"
            % (x_hi - x_lo, 2 * hx, oid))
        cx, cy = sample_xy(x_lo, x_hi, y_lo, y_hi, hx, hy, clearance)
        z = _bottom_lift_z(shape, desktop_top_z)
        shape = transform_shape(shape, translation_matrix([cx, cy, z]))
        placed.append((cx, cy, hx, hy))
        return shape, cx, cy, yaw

    shapes = [table_shape]

    # ---------- Pen (unique target) ----------
    # Place the larger footprint first: the pen extends 156mm along Y; choose its position before placing the cap around it.
    # Sample pen and cap as a pair to match their colors; draw one pair here,
    # rather than using two independent random.choice calls; see CAP_PEN_PAIRS.
    cap_oid, pen_oid = random.choice(CAP_PEN_PAIRS)
    pen_shape, pen_cx, pen_cy, pen_yaw = place_rod(
        pen_oid,
        [
            "target_pen",
            "pen",
            "marker_pen",
            "stationery",
            "writing_tool",
            "target",
            "graspable",
            "rod",
            pen_tag,
        ],
        pen_band,
        clearance=0.05,
    )
    shapes.append(pen_shape)

    # ---------- Cap (unique target) ----------
    # cap_oid was sampled together with pen_oid above to guarantee matching colors.
    cap_shape, cap_cx, cap_cy, cap_yaw = place_rod(
        cap_oid,
        [
            "target_cap",
            "cap",
            "pen_cap",
            "stationery",
            "target",
            "graspable",
            "rod",
            cap_tag,
        ],
        cap_band,
        clearance=0.05,
    )
    shapes.append(cap_shape)

    # ---------- Assertions ----------
    # Sampling one pair guarantees uniqueness; assertions prevent future pool/band changes from silently creating
    # multiple targets or mismatches between left/right labels and actual positions.
    # Verify pairing here too: the code above draws one pair, but if it is later split into
    # two independent samples, this assertion fails immediately; generation logs would not reveal mismatched colors.
    assert (cap_oid, pen_oid) in CAP_PEN_PAIRS, (
        "Pen and cap are not a matching pair: cap=%s pen=%s ⇒ colors will differ" % (cap_oid, pen_oid))
    assert (pen_cy > tcy) == pen_on_left, (
        "Pen position disagrees with side assignment: pen_cy=%.3f tcy=%.3f pen_on_left=%s"
        % (pen_cy, tcy, pen_on_left))
    assert (cap_cy > tcy) != pen_on_left, (
        "Cap position disagrees with side assignment: cap_cy=%.3f tcy=%.3f pen_on_left=%s"
        % (cap_cy, tcy, pen_on_left))

    print("[SCENE] pen=%s xy=(%+.3f,%+.3f) yaw=%+.1f° tag=%s | "
          "cap=%s xy=(%+.3f,%+.3f) yaw=%+.1f° tag=%s | "
          "pen_on_left=%s base_yaw=%+.1f° desktop_top_z=%.3f" % (
              pen_oid, pen_cx, pen_cy, np.rad2deg(pen_yaw), pen_tag,
              cap_oid, cap_cx, cap_cy, np.rad2deg(cap_yaw), cap_tag,
              pen_on_left, np.rad2deg(base_yaw), desktop_top_z))

    return concat_shapes(*shapes)


@register()
def root_scene() -> Shape:
    return library_call("put_cap_on_pen_g2")
