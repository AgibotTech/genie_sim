# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Extract a straw from the paper cup and insert it through the milk-tea lid.

get_object_pose returns the payload-to-world rotation, including the asset's
Rx(+90 degrees) transform. Payload +Y is up: world_up = R @ [0, 1, 0].
Convert Z-up mesh measurements once with payload = (x_zup, z_zup, -y_zup).

Custom grasp poses use world coordinates and the commanded gripper frame:
col0=closing axis, col1=upright-test axis, col2=approach. Do not apply the
annotation-PKL axis conversion to these candidates. Select the actual
lower straw endpoint from the live pose, including tilted or inverted poses.

The straw uses custom grasps without key-object interaction elements.
A nonzero pre_insert_offset selects an insertion API that requires those
elements; use the segmented object-trajectory and feedback moves below.
Keep the thin-walled straw's convexHull collision approximation.

Align in free space, establish hole-wall capture, then advance along the
captured axis before native gripper release. Clear open-finger contact
before resetting and checking final containment. Keep the source cup
upright and stationary; measure containment at the straw-axis/lid-plane
intersection after settling rather than at the lower endpoint.
"""

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import (
    RobotAPI,
    GraspConfig,
    MoveConfig,
    FollowTrajectoryConfig,
)
import numpy as np


# ---------------------------------------------------------------------------
# Asset geometry constants (**payload (y-up) frame**, in m).
# Values come from measurement scripts; the original measurements use a z-up frame.
# All imported values were converted once: payload = (x_zup, z_zup, -y_zup).
# measure_geometry.py: slicing (AABB / connected components / r_in and r_out of each ring).
# lid_through_hole.py: ray-based through-hole detection (slicing can miss the 0.65 mm plate).
# measure_fit.py: fit, plate thickness, passage depth, and paper-cup cavity profile.
# ---------------------------------------------------------------------------

# The vertical/long axis is **+Y** in the payload frame (object_parameters upAxis is "y" for all three assets).
# R returned by get_object_pose acts on this frame, so world up = R @ LOCAL_UP.
LOCAL_UP = np.array([0.0, 1.0, 0.0])

# Straw geometry is a hollow 12-sided tube with a uniform cross section.
# Its entity origin is offset from the axis and length midpoint; use
# measured endpoint coordinates rather than +/-STRAW_HALF_LEN * LOCAL_UP.
STRAW_LEN = 0.15054312
STRAW_HALF_LEN = STRAW_LEN / 2.0  # For grasp leverage/length bookkeeping only; do not construct endpoints from it.
STRAW_AXIS_LOCAL = np.array([0.0, 1.0, 0.0])
STRAW_END_NEG_LOCAL = np.array([0.0, -0.06395581, 0.0111821228])
STRAW_END_POS_LOCAL = np.array([0.0,  0.08658731, 0.0111821228])
STRAW_R = 0.004933  # Circumradius: use this to test whether the straw fits in the hole under worst-case yaw.
# Measured straw geometry: diameter approximately 9.87 mm, radial extent
# 4.596-4.933 mm and wall thickness 0.261 mm.
STRAW_R_INSCRIBED = 0.004596
STRAW_LOCAL_AXIS = STRAW_AXIS_LOCAL  # Long axis = payload +Y.
# Use convexHull for the thin straw collision geometry; its wall is thinner
# than the configured SDF voxel resolution.

# Milk-tea-cup hole coordinates use the measured payload frame. Fit the upper
# hole-wall ring directly from the mesh, keep the opening-plane location
# consistent with HOLE_LOCAL and use LOCAL_UP for the hole axis.
HOLE_LOCAL = np.array([0.0000305547, 0.0683797538, 0.0112533762])
HOLE_PLANE_H = float(HOLE_LOCAL[1])
# Align the measured hole center with the live straw lower endpoint.
# Use geometry-derived coordinates rather than a fixed correction offset.
FINAL_AIM_PRECOMP_LOCAL = np.zeros(3)
# The hole wall is a regular 25-gon. Use its measured center and aperture
# for fit clearances and alignment gates.
HOLE_R_IN = 0.008307  # Actual inradius of the current benchmark_cup_132.
HOLE_R_EQUIV = 0.008329  # Area-equivalent radius; differs from the inradius by 0.022 mm.
LID_PLATE_THICKNESS = 0.00065  # Hole wall extends only 0.65 mm below the top surface => plate thickness 0.65 mm (measured 68.38 -> 67.73).
CUP132_INNER_BOTTOM_H = -0.06610  # Interior floor height along LOCAL_UP (first ray hit directly below the hole center).
# Maximum insertion depth from the opening to the cup floor. After release the straw falls to this depth.
INSERTABLE_DEPTH_MAX = HOLE_PLANE_H - CUP132_INNER_BOTTOM_H  # 0.13448
MILK_TEA_RADIUS = 0.0466

# Use the task-specific benchmark_cup_105_fixed variant. The original cup asset
# is shared with other tasks and remains unchanged.
PAPER_CUP_DEPTH = 0.07830  # AABB height.
PAPER_CUP_INNER_BOTTOM_ABOVE_BASE = 0.00225  # Interior floor above the AABB bottom (base thickness 0.93 mm).
PAPER_CUP_MOUTH_R_IN = 0.0306
# Rim height relative to the lower straw endpoint when it rests on the floor = extraction travel needed to clear the cup.
STRAW_EMBED_DEPTH = PAPER_CUP_DEPTH - PAPER_CUP_INNER_BOTTOM_ABOVE_BASE  # 0.07605

# 90D finger-pad contact surfaces are measured relative to the commanded
# gripper center, not the support-link bounding box. Negative approach offsets
# lie behind the center; use the open and closed pad windows below.
PAD_SPAN_OPEN = (-0.0570, -0.0200)
PAD_SPAN_CLOSED = (-0.0400, -0.0030)
FINGERTIP_OFFSET = -PAD_SPAN_OPEN[1]  # 0.020: forward reach of the open pad's leading edge relative to gripper center.
# **The rod axis must lie inside the closed finger-pad surface; this is the only relevant window.**
#
# Using "open window intersect closed window" = (-0.040,-0.020) as the landing interval was **overconstrained**,
# pushing the landing point into the rear quarter of the closed pad, visibly away from its center.
# Intermediate-angle clearance profiles (q sweep in measure_ctek90d_pads.py) disprove the open-window constraint:
#
# position     q=0     -0.30   -0.50   -0.65   -0.75   -0.80    verdict
# -44 mm       89.7     64.0     71.6    39.3    28.4    22.9    not covered when closed: fail
# -41 mm       89.7     64.0     44.0    28.0    44.8    39.3    not covered when closed: fail
# -38..-17     89.7     64.0     44.0    28.0    17.1    11.6    monotonically tightens throughout: pass
# -14 mm         --     64.0     44.0    28.0    17.1    11.6    uncovered at q=0 but harmless (see below)
# -5 mm          --       --     44.0    28.0    17.1    11.6    leading edge sweeps past: caution
#
# The key row is -14 mm: at q=0 the pads do not reach it, **but clearance is still 64 mm when the leading edge sweeps across**.
# Each pad is 32 mm from the centerline, far from a 6 mm-radius rod, so **there is no contact**. Requiring the rod to already lie between
# the open pads before closure is incorrect: the leading edge moves forward while the pads are still far apart.
PAD_WINDOW = PAD_SPAN_CLOSED

# ---------------------------------------------------------------------------
# Grasp candidates.
# How far the gripper center passes the rod axis along approach determines **where the axis lands on the closed pad surface**.
#
# Three incorrect values had measurable consequences:
# 25.2 mm (reference-task lower bound for covering the full diameter, r+17.2, plus 2 mm): too far forward.
# 49.2 mm (chosen to stay farther from fingertips): the rod fell **behind** the closed pad interval [-40,-3],
# where finger clearance was 23.2 mm versus a 12 mm straw => **empty closure**. Measured pick step1 pos_diff
# was only 1.4 mm (arm reached its target but the rod did not move because pads never touched it); step2 lift pos_diff
# was **125 mm ~= the entire lift travel**, so the straw stayed in place; two of three attempts dropped it.
# 30.0 mm (center of the open/closed intersection): rod spans [-36.4,-23.6], leaving only 3.6 mm at the rear => too far back.
#
# **Offset the landing point forward by half the drift budget rather than centering it** (see GRASP_DRIFT_BUDGET).
# Frame-by-frame reconstruction shows **one-way inward slip**; centering wastes half the travel on a side the rod never visits.
GRASP_DEPTH = None   # Computed below from the window and budget after GRASP_DRIFT_BUDGET.


def _apparent_radius(elev_deg: float) -> float:
    """Apparent straw radius along approach. Oblique approach through a cylinder gives r / cos(elev)."""
    return STRAW_R / float(np.cos(np.radians(float(elev_deg))))


def _grasp_depth_for(elev_deg: float) -> float:
    """Return the approach offset from the rod axis to the commanded gripper center.

    Use the pad-window center at the selected elevation.
    """
    del elev_deg  # Keep the signature: callers supply elevation; this returns the depth for that setting.
    return GRASP_DEPTH


def _jaw_axial_position(L: float, elev_deg: float) -> float:
    """Actual axial gripper-center location relative to the lower straw endpoint.

    **Not equal to L**: approach includes vertical -sin(elev), and gripper center = rod anchor + approach*depth,
    putting it depth(elev)*sin(elev) below the anchor (8.75 mm at 20 degrees).
    This quantity determines lid collision after insertion; asserting on L tests a surrogate instead.
    """
    e = np.radians(float(elev_deg))
    return float(L) - _grasp_depth_for(elev_deg) * float(np.sin(e))


# Grasp-anchor distance to the lower endpoint controls both insertion clearance
# and gripper travel during straightening. Required travel is
# 2 * abs(L - half_length) * sin(rotation_angle / 2). Keep the configured
# anchor range within the pad window and insertion-clearance budget.
GRASP_L_LADDER = (0.105, 0.110, 0.100)
# Approach elevation is measured down from horizontal. Limit it to keep the
# straw inside the finger-pad window during straightening.
GRASP_ELEV_DEG = (0.0,)
# Horizontal approach azimuth relative to +x (reaching from the robot's side); sign depends on the arm:
# left arm -> negative angle (approach has a -y component, keeping the gripper body at +y, on the left arm's own side).
# right arm -> positive angle.
GRASP_AZIMUTH_DEG = (20.0, 35.0, 50.0, 65.0, 80.0)
# Required gripper-center clearance above the hole after insertion. 10 mm was too small, nearly pressing on the milk-tea cup; raised to 25 mm.
# Also determines the corrective push command limit; see _max_command_depth().
JAW_ABOVE_LID_MARGIN = 0.025
# Finger half-width along the rod axis: measured pad width is 23.0 mm both open and closed, hence half-width 11.5 mm.
JAW_HALF_WIDTH_ALONG_ROD = 0.0115

# Budget straw drift inside the gripper through closure, extraction and
# straightening, then check the resulting landing point.
GRASP_DRIFT_BUDGET = 0.010
# Boundary margin at each end of the landing interval. This is derived: the closed pad is only 37.0 mm long, and the rod occupies 21.2 mm
# (apparent diameter 12.8 + elevation spread 8.4); a 10.0 mm drift budget leaves **only 5.8 mm for both sides**.
# Thus this value cannot exceed (37.0 - 21.2 - budget)/2 = 2.9 mm. To increase boundary clearance,
# reduce the drift budget (= reduce |L - half-length|), rather than increasing this number.
PAD_EDGE_MARGIN = 0.0025
# Gripper-body extent along -approach, including wrist and 61 mm wrist-camera protrusion, used to check crossing the body centerline.
JAW_BODY_REACH = 0.15
JAW_MIDLINE_TOL = 0.05
# Log the cosine between the horizontal approach and base-to-object direction
# for diagnostics. It is not a grasp-candidate filter.
REACH_COS_MIN = None  # None = no gate.

# -- Landing point derived from pad window + drift budget; **do not enter manually** --
# Rod half-width along approach = apparent radius r/cos(elev) **plus** elevation-induced spread over the 23 mm pad width,
# +/-11.5*tan(elev) (4.2 mm at 20 degrees, easily overlooked). Use the worst elevation.
_ROD_HALF_SPAN = max(
    _apparent_radius(e) + JAW_HALF_WIDTH_ALONG_ROD * float(np.tan(np.radians(e)))
    for e in GRASP_ELEV_DEG)
# Valid rod-axis interval (entire rod segment inside the closed pad surface).
PAD_AXIS_LO = PAD_WINDOW[0] + _ROD_HALF_SPAN   # Innermost limit.
PAD_AXIS_HI = PAD_WINDOW[1] - _ROD_HALF_SPAN   # Forwardmost limit.
# Landing point equalizes margins at the start and end (start - drift budget):
# start s, end s - budget; require s <= HI and s - budget >= LO.
# => s in [LO + budget, HI]; take the midpoint.
GRASP_DEPTH = -0.5 * ((PAD_AXIS_LO + GRASP_DRIFT_BUDGET) + PAD_AXIS_HI)

# Closed-form interval for **perpendicular distance** from gripper center to rod axis: depth(elev)*cos(elev), excluding axial approach.
# Derive from GRASP_ELEV_DEG rather than manually entering a copy, which could silently diverge from _grasp_depth_for;
# otherwise the gate validates a different quantity from the one intended.
_PERP_DISTS = tuple(
    _grasp_depth_for(e) * float(np.cos(np.radians(e))) for e in GRASP_ELEV_DEG)
PERP_DIST_RANGE = (min(_PERP_DISTS), max(_PERP_DISTS))

# Lift must cover cup clearance, axial slip and endpoint drop during
# straightening. Keep the scalar lift compatible with the pre-insertion IK
# height. Framework lift filtering and execution use different frame
# compositions, so check actual clearance after the move.
PICK_UP_DISTANCE = 0.135

# Total extraction must clear the paper cup and reach the pre-insertion height.
# Use a small nonzero initial vertical lift, then extract along the straw axis
# away from the wall before further vertical motion.
PICK_UP_SEAT_BREAK = 0.003

# Step 3.4 axial extraction travel must **cover the full budget in one move**. Otherwise Step 3.5 adds lift, and every extra move
# adds gripper slip (measured: 95 mm axial extraction leaves +0.9 mm clearance => 36.2 mm supplemental lift => 10.2 mm cumulative slip after
# straightening, **exhausting the 10 mm drift budget**; rod rear edge only 2.7 mm from pad rear edge => geometry shifts => Step 5 IK rejects all).
# Accounting using measured 23.8-degree tilt and L~=103 mm:
# net extraction lift 77.8 mm / cos = 85.0 mm axially.
# + (straightening swing 10.5 + clearance 15 + slip allowance 12) / cos = 41.0 mm.
# = **126.0 mm** => choose 130 mm.
# A 130 mm command achieves ~119 mm vertical travel after slip, leaving ~+41 mm endpoint clearance and >15 mm after straightening.
SEAT_BREAK_TRAVEL = 0.130
JAW_SLIP_ALLOWANCE = 0.012          # (2) Measured 10.9 mm; allow 12 mm.
CUP_EXIT_MARGIN = 0.015             # Required endpoint clearance above paper-cup rim after straightening.
DESIGN_MAX_SETTLE_TILT_DEG = 28.0   # Design limit for settled tilt (measured 24.6 degrees; geometric limit ~26.3 degrees).

# Advance to the configured insertion depth after initial hole-wall capture,
# keeping the gripper clear of the lid before release.
INSERT_DEPTH_D = 0.060
PRE_INSERT_GAP = 0.018  # Lower endpoint clearance above hole during pre-insertion (reduced from 20 to 18 mm).
PRE_INSERT_EXTRA = (0.000, 0.004, 0.008, 0.012)  # Candidate retreat levels along hole axis only (4 instead of 6 to reduce planning cost).
# Try small yaw alternatives about the hole axis first. Cylindrical symmetry
# preserves insertion geometry while changing the arm configuration. Larger
# yaw alternatives are a separate fallback after the small-yaw batch fails IK,
# so reachable small-yaw motions retain their lower slip cost.
PRE_INSERT_YAWS = tuple(np.radians([-8.0, 0.0, 8.0]))  # Yaw about the hole axis: specify degrees, store radians.
# _rot_about_axis takes radians. Convert configured yaw degrees before
# constructing pre-insertion orientations.
PRE_INSERT_YAWS_FALLBACK = tuple(np.radians([-30.0, 30.0]))  # Fallback level after universal small-yaw failure; never combine with the batch above.
# These depth limits bound insertion by the gripper-to-lid clearance.
RESEAT_DEPTH_TARGET = INSERT_DEPTH_D - 0.005
MAX_COMMAND_DEPTH_CEILING = INSERTABLE_DEPTH_MAX - 0.005
# Two descent stages: first descend only along the hole axis to ~5 mm above it, then correct precisely while safely suspended.
# Do not translate laterally in contact/at the rim, or repeatedly at the 18 mm pre-insertion height: that triggers another
# complete robot motion and may slip the thin straw out of alignment. The first 5 mm target leaves physical clearance;
# stage two requires the lower endpoint at least 2 mm above the opening before straightening or radial movement.
NEAR_HOLE_AIR_GAP = 0.005
NEAR_HOLE_AIR_MIN_GAP = 0.002
# The one-shot target below is intentionally tighter than the acceptance gate:
# cuRobo + the compliant 90D grasp routinely leaves 1--2 mm of achieved
# lateral residual even when the commanded straw axis goes through the hole
# centre.  Requiring the commanded 0.5 mm target as an achieved-state gate
# prevented every recent episode from even attempting the 6 mm capture move.
# The physical single-side clearance is HOLE_R_IN - STRAW_R = 3.374 mm; keep a
# conservative 0.874 mm reserve here, then let Step 6.1 prove real capture.
NEAR_HOLE_ALIGN_RADIAL_TARGET = 0.0005
NEAR_HOLE_ALIGN_TILT_DEG_MAX = 1.0
NEAR_HOLE_ENTRY_RADIAL_MAX = 0.0025
NEAR_HOLE_ENTRY_TILT_DEG_MAX = 1.5
# Allow one bounded feedback retry from the achieved signed radial error
# while maintaining the minimum air gap.
NEAR_HOLE_FEEDBACK_MAX_RETRY = 1
NEAR_HOLE_FEEDBACK_MAX_CORRECTION = 0.006
# Insert in two segments: establish hole-wall capture at the initial depth,
# then keep the grip and advance along the captured straw axis to the
# configured final depth. Verify capture before each transition.
ENTRY_CAPTURE_INITIAL_DEPTH = 0.0150  # First segment: penetrate the plate and establish hole-wall guidance.
ENTRY_CAPTURE_DEPTH = 0.0600          # Second segment: release only after actual deep insertion to 60 mm below the opening.
ENTRY_MIN_DEPTH = 0.0020          # At least pass the hole plate and obtain ~1.35 mm of guidance beneath it.
# Release eligibility uses the faceted straw's measured inscribed radius and
# the bore clearance after the tip crosses the lid. Keep the tighter airborne
# alignment budget before contact, allow the configured readback tolerance
# after capture, and evaluate final containment after release.
ENTRY_BORE_RADIAL_MAX = HOLE_R_IN - STRAW_R_INSCRIBED + 0.0002
ENTRY_TILT_DEG_MAX = 1.5
INSERT_SETTLE_WAIT_S = 0.15
# After the first, purely axial descent, switch the 90D from its velocity
# close into a live-q position hold.  The next action is the sole final-air
# alignment, so this short settling window cannot leave an uncorrected offset
# at the hole mouth.
GRIPPER_HOLD_SETTLE_WAIT_S = 0.10
# The public RobotAPI has no partial-open/hold method.  All near-lid gripper
# position commands below therefore use its implemented ``joint_to`` API.
# Do not substitute ``open_gripper(opened_width=...)``: the collector's
# gripper RPC treats every "open" command as full-open and ignores that width.
GRIPPER_HOLD_MAX_ASYMMETRY_RAD = 0.002
# task_info uses physics_frequency=120 and the server doubles it automatically; set 1 to avoid another artificial 3x slowdown.
INSERT_SLOWDOWN_FACTOR = 1.0

# Release constants used by the gripper helpers. The active task opens the
# gripper natively after insertion and clears residual support contact radially.
GRIPPER_CLOSED_Q = -0.9100000112241267
RELEASE_Q_STEP_RAD = 0.001
# Release-related constants. The active release path uses native open,
# followed by bounded radial withdrawal of the empty gripper when needed.
RELEASE_Q_OPEN_TARGET = -0.600
RELEASE_POST_HOLD_WAIT_S = 0.05
RELEASE_FALL_DEPTH_MIN = 0.040
RELEASE_SETTLE_WAIT_S = 1.00
# After native open, withdraw the empty gripper radially from the live
# straw axis if a finger or support link still contacts the rod. Keep
# the motion perpendicular to the straw before resetting the arm.
RELEASE_RADIAL_WITHDRAW_STEPS = (0.004, 0.006)
RELEASE_RADIAL_WITHDRAW_WAIT_S = 0.20
# Entry requires <=1.5-degree tilt to leave enough margin for first contact above the hole. With
# ~11 mm guidance below the plate during slow stationary release, finger contact can cause small reversible orientation changes.
# Conservative extrapolation at q=-0.60 gives ~6 degrees; 7 degrees remains well below the final floor-contact limit of 20 degrees. The actual
# hard opening gate remains radial error of the axis/plane intersection; do not relax it beyond the hole.
RELEASE_CAPTURE_TILT_DEG_MAX = 7.0

# Use separate checks for initial capture and final containment. Before
# release, check measured entry depth, opening-plane radial error and tilt.
# After release and settling, apply the final depth and containment bounds.
SEAT_DEPTH_MIN = ENTRY_MIN_DEPTH  # Compatibility with geometry assertions below; actual release uses the three entry_ok criteria.
ACCEPT_DEPTH_MIN = 0.090
ACCEPT_DEPTH_MAX = INSERTABLE_DEPTH_MAX + 0.006
# Decisive metric: radial distance between hole center and intersection of straw axis with hole plane. Current one-sided physical clearance is
# HOLE_R_IN - STRAW_R = 3.374 mm; final check allows ~1 mm measurement/orientation tolerance after floor contact.
ACCEPT_BORE_RADIAL_MAX = 0.0044
ACCEPT_TILT_DEG_MAX = 20.0  # Tilt after release/floor contact may reach ~10 degrees (29 mm lower-end offset / 134 mm rod length).
# Check paper-cup tilt and displacement after extraction. Keep these checks
# with the kinematic variant and use the same tilt criterion as scene sanity checks.
ACCEPT_PAPER_TILT_DEG_MAX = 25.0
# Check cup displacement as well as tilt to detect carrying or dragging
# the source cup during extraction.
ACCEPT_PAPER_MOVED_MAX = 0.060
SETTLE_WAIT_S = 1.5  # Wait for settling after release; do not assess a falling straw as final state.
# Warning threshold for straw slip relative to the gripper. Axial retention relies only on friction, and slip invalidates gripper-arrival-based estimates.
SLIP_WARN = 0.005


def _assert_geometry_invariants():
    """Geometry invariants checked at import.

    These constants are coupled: grasp depth/pad window; insertion depth/grasp height/opening clearance;
    acceptance gates/physical clearance. A manual edit can silently invalidate another constraint.
    Check them together here rather than fail 1000 lines later.
    """
    # -- Grasp depth and pad window (check each elevation because apparent radius changes) --
    for elev in GRASP_ELEV_DEG:
        r_app = _apparent_radius(elev)
        depth = _grasp_depth_for(elev)
        assert r_app >= STRAW_R - 1e-12, (
            "Apparent radius %.5f is smaller than true radius %.5f => formula is reversed" % (r_app, STRAW_R))
        # Rod span along approach has two terms: apparent radius **plus** elevation-induced spread over pad width.
        # With the rod tilted relative to approach, its approach coordinate across the 23 mm pad width varies by
        # +/-JAW_HALF_WIDTH*tan(elevation) (+/-4.2 mm at 20 degrees). Omitting this underestimates occupied span.
        spread = JAW_HALF_WIDTH_ALONG_ROD * float(np.tan(np.radians(elev)))
        rod_lo = -depth - r_app - spread
        rod_hi = -depth + r_app + spread
        # The rod span must fit the closed pad surface at both its initial and
        # maximum-drift positions.
        for what, lo, hi in (("start", rod_lo, rod_hi),
                             ("drift endpoint", rod_lo - GRASP_DRIFT_BUDGET,
                              rod_hi - GRASP_DRIFT_BUDGET)):
            assert lo >= PAD_WINDOW[0] and hi <= PAD_WINDOW[1], (
                "Elevation %.0f°: at depth=%.1fmm, the **%s** rod occupies approach [%.1f,%.1f]mm, "
                "outside the closed finger-pad surface [%.1f,%.1f]mm. Beyond front edge = protrudes past pad; "
                "beyond rear edge = in linkage region with 23mm clearance, where closure cannot grip it. (drift budget %.0fmm)"
                % (elev, 1000 * depth, what, 1000 * lo, 1000 * hi,
                   1000 * PAD_WINDOW[0], 1000 * PAD_WINDOW[1],
                   1000 * GRASP_DRIFT_BUDGET))
        # Both margins should be **symmetric**: start-to-front ~= drift-end-to-rear. Asymmetry wastes the budget.
        front = PAD_WINDOW[1] - rod_hi                                  # Start-to-leading-edge margin.
        back = (rod_lo - GRASP_DRIFT_BUDGET) - PAD_WINDOW[0]            # Endpoint-to-rear-edge margin.
        assert min(front, back) >= PAD_EDGE_MARGIN, (
            "Elevation %.0f°: start is %.1fmm from front edge / drift endpoint is %.1fmm from rear edge, "
            "one end has less than %.1fmm clearance. Pad surface %.1fmm: rod occupies %.1f + drift budget %.1f "
            "=> only %.1f remains for both sides. More clearance requires reducing |L - half-length| to reduce drift budget."
            % (elev, 1000 * front, 1000 * back, 1000 * PAD_EDGE_MARGIN,
               1000 * (PAD_WINDOW[1] - PAD_WINDOW[0]), 1000 * (rod_hi - rod_lo),
               1000 * GRASP_DRIFT_BUDGET,
               1000 * (PAD_WINDOW[1] - PAD_WINDOW[0] - (rod_hi - rod_lo)
                       - GRASP_DRIFT_BUDGET)))
        assert abs(front - back) <= 0.004, (
            "Elevation %.0f°: depth=%.1fmm gives unbalanced end clearances: start %.1fmm from front edge, "
            "drift endpoint %.1fmm from rear edge, difference %.1fmm(required <=4.0mm)"
            % (elev, 1000 * depth, 1000 * front, 1000 * back,
               1000 * abs(front - back)))
    # 90D lateral grasp limit is set by support_link collision spheres (65~68 mm), not the 89.7 mm pad opening.
    assert 2 * STRAW_R <= 0.065, "Straw diameter %.1fmm exceeds the 90D side-grasp limit of 65mm" % (2000 * STRAW_R)

    # -- Insertion depth versus axial gripper position --
    # Use _jaw_axial_position, not L: gripper center is depth*sin(elev) below the anchor;
    # asserting on L would incorrectly accept insertion with only 6 mm gripper clearance above the opening.
    assert INSERT_DEPTH_D < INSERTABLE_DEPTH_MAX, (
        "Commanded depth %.4f exceeds maximum insertion %.4f from opening to inner cup bottom" % (INSERT_DEPTH_D, INSERTABLE_DEPTH_MAX))
    assert MAX_COMMAND_DEPTH_CEILING <= INSERTABLE_DEPTH_MAX, (
        "Extra-push ceiling %.4f exceeds insertion limit %.4f => would force the straw into the cup bottom"
        % (MAX_COMMAND_DEPTH_CEILING, INSERTABLE_DEPTH_MAX))
    assert MAX_COMMAND_DEPTH_CEILING > INSERT_DEPTH_D, (
        "Historical deep-push ceiling %.4f must exceed nominal depth %.4f"
        % (MAX_COMMAND_DEPTH_CEILING, INSERT_DEPTH_D))
    for L in GRASP_L_LADDER:
        for elev in GRASP_ELEV_DEG:
            axial = _jaw_axial_position(L, elev)
            above_lid = axial - INSERT_DEPTH_D
            assert above_lid >= JAW_ABOVE_LID_MARGIN, (
                "L=%.3f elevation=%.0f°: gripper center axial position only %.1fmm; at insertion %.1fmm it clears opening by only %.1fmm"
                "(required >=%.1fmm) => gripper will hit the lid top"
                % (L, elev, 1000 * axial, 1000 * INSERT_DEPTH_D,
                   1000 * above_lid, 1000 * JAW_ABOVE_LID_MARGIN))
            # Grasp anchor must not extend beyond the upper straw endpoint.
            assert L + JAW_HALF_WIDTH_ALONG_ROD < STRAW_LEN, (
                "L=%.3f plus finger half-width exceeds straw top %.4f => half the finger grips air"
                % (L, STRAW_LEN))
            # At grasp, gripper center must be above paper-cup rim or the gripper cannot enter.
            assert axial > STRAW_EMBED_DEPTH + JAW_HALF_WIDTH_ALONG_ROD, (
                "L=%.3f elevation=%.0f°: gripper center axial %.1fmm does not clear paper-cup rim %.1fmm(+finger half-width) => rim collision"
                % (L, elev, 1000 * axial, 1000 * STRAW_EMBED_DEPTH))

    # -- Fit clearance --
    clearance = HOLE_R_IN - STRAW_R
    assert clearance > 0.0018, (
        "One-sided clearance %.3fmm is below measured closed-loop alignment minimum 1.8mm => geometry issue, not control; modify asset"
        % (1000 * clearance))
    assert STRAW_R > STRAW_R_INSCRIBED, "Circumscribed radius must exceed inscribed radius; constants are reversed"

    # -- Acceptance gates must accommodate physically reachable values --
    # Pre-release gate concerns commanded depth; post-release gate concerns free slide to the floor. Their lower bounds differ.
    assert SEAT_DEPTH_MIN <= INSERT_DEPTH_D, (
        "Pre-release minimum depth %.4f exceeds commanded depth %.4f => correctly positioned episodes would also fail"
        % (SEAT_DEPTH_MIN, INSERT_DEPTH_D))
    assert ACCEPT_DEPTH_MIN <= INSERTABLE_DEPTH_MAX - 0.02, (
        "Final minimum depth %.4f approaches insertion limit %.4f => only exact bottom contact would pass"
        % (ACCEPT_DEPTH_MIN, INSERTABLE_DEPTH_MAX))
    # A final lower bound above commanded depth is **intentional**, relying on post-release free slide (measured 50 mm command ->
    # 130.6 mm after release). Document this dependency so the next maintainer does not assume a missing assertion.
    assert ACCEPT_DEPTH_MIN > INSERT_DEPTH_D, (
        "Final minimum depth %.4f does not exceed commanded depth %.4f => post-release sliding is no longer required; "
        "update this assertion together with the docstring explanation" % (ACCEPT_DEPTH_MIN, INSERT_DEPTH_D))
    assert ACCEPT_DEPTH_MAX >= INSERTABLE_DEPTH_MAX, (
        "Acceptance maximum depth %.4f excludes post-release bottom-contact depth %.4f => the deepest insertions would fail"
        % (ACCEPT_DEPTH_MAX, INSERTABLE_DEPTH_MAX))
    assert ACCEPT_BORE_RADIAL_MAX > clearance, (
        "Opening radial gate %.3fmm is tighter than physical clearance %.3fmm => threshold is smaller than the measurement's physical range"
        % (1000 * ACCEPT_BORE_RADIAL_MAX, 1000 * clearance))
    # Check this insertion-depth target against the physical gripper-clearance bound.
    assert 0.0 < RESEAT_DEPTH_TARGET <= INSERT_DEPTH_D, (
        "Historical deep-push target %.4f must be in (0, nominal depth %.4f]"
        % (RESEAT_DEPTH_TARGET, INSERT_DEPTH_D))

    # -- Candidate pool size --
    # For each (L, elev, az), `_straw_grasp_candidates` also expands two azimuth sides and two
    # grip_sign values. Verify the actual count; missing factors could wrongly reject the valid zero-elevation pool as empty.
    n_grasp = (len(GRASP_L_LADDER) * len(GRASP_ELEV_DEG)
               * len(GRASP_AZIMUTH_DEG) * 2 * 2)
    assert n_grasp >= 20, (
        "Only %d grasp candidates => no retry margin(candidate count is the framework's retry budget)" % n_grasp)
    n_insert = len(PRE_INSERT_EXTRA)
    assert n_insert <= 200 and n_grasp <= 300, (
        "Candidates exceeding framework random_downsample limits(place 200 / grasp 300) will be randomly discarded")

    # -- Lift height must satisfy two requirements; take the larger --
    # (1) Cup extraction + axial gripper slip + straightening swing (original design, ~115 mm).
    # (2) Above hole center (~58 mm above paper-cup rim, pre-insertion 18 mm below hole center, ~131 mm from cup floor).
    # Current 135 mm uses (2) + margin. Here check only (1) as the lower bound; (2) depends on scene geometry and must be checked at runtime.
    _tilt = float(np.radians(DESIGN_MAX_SETTLE_TILT_DEG))
    _swing = max(GRASP_L_LADDER) * (1.0 - float(np.cos(_tilt)))
    _need_exit_cup = PAPER_CUP_DEPTH + JAW_SLIP_ALLOWANCE + _swing + CUP_EXIT_MARGIN
    assert PICK_UP_DISTANCE >= _need_exit_cup, (
        "PICK_UP_DISTANCE %.1fmm is below required extraction %.1fmm = cup exit %.1f(cup depth) + in-gripper slip %.1f + "
        "straightening downward swing %.1f(L=%.0fmm, tilt %.0f°) + clearance %.1f"
        % (1000 * PICK_UP_DISTANCE, 1000 * _need_exit_cup, 1000 * PAPER_CUP_DEPTH,
           1000 * JAW_SLIP_ALLOWANCE, 1000 * _swing,
           1000 * max(GRASP_L_LADDER), DESIGN_MAX_SETTLE_TILT_DEG, 1000 * CUP_EXIT_MARGIN))
    # Must be scalar: framework interprets a list as an upward scan selecting the first IK solution, always taking the lower bound.
    assert isinstance(PICK_UP_DISTANCE, float), (
        "PICK_UP_DISTANCE must be a scalar float; passing a list uses only its lower bound(grasp.py:571-596)")
    # Post-lift gripper height must remain within measured reachability (similar configuration reached 1098.5 mm at 682 mm horizontal distance).
    _jaw_z = 0.7325 + min(GRASP_L_LADDER) * float(np.cos(_tilt)) + PICK_UP_DISTANCE
    assert _jaw_z < 1.08, (
        "Lifted gripper center reaches %.1fmm, near measured reach limit 1098.5mm => inspect other tasks' state.json before relaxing"
        % (1000 * _jaw_z))


_assert_geometry_invariants()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pose_from_xyz_wxyz(xyz, wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat2mat_wxyz(np.asarray(wxyz, dtype=float))
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def _rot_about_axis(axis, ang_rad) -> np.ndarray:
    """Rodrigues rotation matrix about an arbitrary unit axis by ang_rad.

    Adds hole-axis yaw to pre-insertion candidates. Cylindrical straw symmetry preserves insertion geometry
    while changing gripper orientation and expanding IK solutions; see PRE_INSERT_YAWS.
    """
    a = _unit(axis)
    c, s = float(np.cos(ang_rad)), float(np.sin(ang_rad))
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _find_unique_object(scene_info, keyword: str) -> str:
    """Find exactly one scene object by keyword.

    Raise on zero or multiple matches; silently taking the first turns nonunique-target scene bugs
    into unexplained downstream failures.
    """
    hits = [
        oid for oid, info in scene_info["layout"].items()
        if keyword in info.get("keywords", [])
    ]
    if len(hits) != 1:
        raise RuntimeError(
            "keyword %r matched %d objects(expected 1): %s" % (keyword, len(hits), hits))
    return hits[0]


def _collision_prim_paths(robot, obj_id):
    """Exact mesh paths for this object in the curobo obstacle world.

    Do not pass only `.../entity`: server remove_objects_from_world (`motion_gen_reacher.py:1877`)
    matches exact names, so entity paths never match yet silently report success. Attachment accepts entity
    because the server expands it into mesh paths; these operations are asymmetric. Extra names are skipped
    (`if not obs: continue`), so a broader list is harmless.
    """
    base = robot.get_object_prim_path(obj_id).rstrip("/")
    names = ["visual"] + ["collider_%02d" % i for i in range(1, 9)]
    return ["%s/body/%s" % (base, n) for n in names]


def _min_rotation_aligning(a, b) -> np.ndarray:
    """Minimum 3x3 rotation taking unit vector a to unit vector b.

    Avoid a fixed absolute orientation: when current and target orientations differ substantially,
    that requires large wrist rotation, measured to cause universal IK rejection.
    """
    a = _unit(a)
    b = _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, a)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, tmp))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def _straw_state(robot, straw_id):
    """Return (T, axis_up, tip_low, tip_high, local_low) from the live straw pose.

    Select the actual lower endpoint at runtime so inverted straw poses use
    the same geometry calculations.
    """
    pos, quat = robot.get_object_pose(straw_id)
    T = _pose_from_xyz_wxyz(pos, quat)
    axis = _unit(T[:3, :3] @ STRAW_LOCAL_AXIS)
    e_plus = T[:3, 3] + T[:3, :3] @ STRAW_END_POS_LOCAL
    e_minus = T[:3, 3] + T[:3, :3] @ STRAW_END_NEG_LOCAL
    # Do not simplify to +/-HALF_LEN: entity origin is offset laterally 11.182 mm from the true rod axis.
    if e_plus[2] <= e_minus[2]:
        tip_low, tip_high, local_low = e_plus, e_minus, STRAW_END_POS_LOCAL
    else:
        tip_low, tip_high, local_low = e_minus, e_plus, STRAW_END_NEG_LOCAL
    axis_up = axis if axis[2] >= 0.0 else -axis
    return T, axis_up, tip_low, tip_high, local_low


def _hole_frame(robot, milk_id):
    """Return (world hole center, unit hole axis). Hole axis = milk-tea cup entity +Z."""
    pos, quat = robot.get_object_pose(milk_id)
    T = _pose_from_xyz_wxyz(pos, quat)
    centre = T[:3, 3] + T[:3, :3] @ HOLE_LOCAL
    # Hole axis = cup up axis. Payload up is +Y; [0,0,1] would produce a **horizontal** vector.
    axis = _unit(T[:3, :3] @ LOCAL_UP)
    return centre, axis


def _final_air_aim_frame(robot, milk_id):
    """Return Step 5.6's (true hole center, compensated aim, hole axis, world compensation).

    Strictly separate true center from compensated target: true center alone defines measurements, gates,
    and success. Compensation is only for the final combined contact-free command. Defined in payload X/Z,
    it must rotate through the current cup entity pose into world coordinates, never be added directly to world X/Y/Z.
    """
    pos, quat = robot.get_object_pose(milk_id)
    T_cup = _pose_from_xyz_wxyz(pos, quat)
    R_cup = T_cup[:3, :3]
    true_centre = T_cup[:3, 3] + R_cup @ HOLE_LOCAL
    world_delta = R_cup @ FINAL_AIM_PRECOMP_LOCAL
    aim_centre = true_centre + world_delta
    hole_axis = _unit(R_cup @ LOCAL_UP)
    return true_centre, aim_centre, hole_axis, world_delta


def _rod_frame(rod_axis_up):
    """Orthonormal basis (h1, h2, n), using rod axis as up. h1 projects world +X onto the plane normal to rod axis.

    Azimuth az is measured from h1, so az~=0 means approach roughly world +x, reaching from
    the robot side (x=-0.7), the most natural approach direction.
    """
    n = _unit(rod_axis_up)
    ex = np.array([1.0, 0.0, 0.0])
    h1 = ex - float(np.dot(ex, n)) * n
    if np.linalg.norm(h1) < 1e-6:  # Rod exactly along world X: choose another reference.
        ey = np.array([0.0, 1.0, 0.0])
        h1 = ey - float(np.dot(ey, n)) * n
    h1 = _unit(h1)
    h2 = np.cross(n, h1)
    return h1, h2, n


def _straw_grasp_pose_world(rod_axis_up, tip_low, L, elev_deg, az_deg, grip_sign) -> np.ndarray:
    """Construct world grasp poses with the commanded gripper-axis convention.

    Use the current straw orientation to select the grip sign and approach
    direction, including tilted or inverted poses.
    """
    h1, h2, n = _rod_frame(rod_axis_up)
    e = np.radians(float(elev_deg))
    a = np.radians(float(az_deg))
    depth = _grasp_depth_for(elev_deg)

    approach = np.cos(a) * np.cos(e) * h1 + np.sin(a) * np.cos(e) * h2 - np.sin(e) * n
    grip = float(grip_sign) * (-np.sin(a) * h1 + np.cos(a) * h2)
    approach, grip = _unit(approach), _unit(grip)
    col1 = np.cross(approach, grip)
    R = np.column_stack([grip, col1, approach])

    assert np.allclose(np.cross(R[:, 0], R[:, 1]), R[:, 2], atol=1e-9), R
    assert abs(float(np.dot(grip, n))) < 1e-9, "Closing axis must be perpendicular to rod axis to grip it"
    assert abs(float(np.dot(grip, approach))) < 1e-9, "Closing axis must be perpendicular to approach"

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tip_low, dtype=float) + n * float(L) + approach * depth
    return T


def _straw_grasp_candidates(rod_axis_up, tip_low, tag):
    """World-frame candidate grid over L, elevation, azimuth, and both grip_sign values.

    Generate both signs rather than predict the correct one. disable_upside_down (`grasp.py:180-188`)
    is a retention mask: left keeps col1_z<0, right col1_z>0. Actual tilt determines sign; the closed form
    holds only for a vertical rod. Filter measured world col1_z instead, typically removing half the candidates.

    Never generate bad candidates merely to rank them later: `grasp.py:519` randomly downsamples and :645
    reorders by humanlike posture, discarding input order. Candidate count supplies framework retries.
    """
    poses = []
    for L in GRASP_L_LADDER:
        for elev in GRASP_ELEV_DEG:
            for az in GRASP_AZIMUTH_DEG:
                for sign in (+1.0, -1.0):
                    for s in (+1.0, -1.0):  # Generate both azimuth sides; the filter determines gripper-body side.
                        poses.append(_straw_grasp_pose_world(
                            rod_axis_up, tip_low, L, elev, s * az, sign))
    out = np.stack(poses, axis=0)
    print("[GRASP] %s: %d world-frame candidates (L=%s x elevation=%s x azimuth=±%s° x grip_sign=±1)"
          % (tag, len(out), list(GRASP_L_LADDER), list(GRASP_ELEV_DEG),
             list(GRASP_AZIMUTH_DEG)))
    print("[GRASP] %s: rod axis(up)=%s lower end=%s"
          % (tag, np.round(rod_axis_up, 4).tolist(), np.round(tip_low, 4).tolist()))
    return out


def _make_straw_grasp_gate(arm, cup_mouth_z, rod_axis_up, tip_low, base_xy, tag):
    """Hard grasp-filter callback with framework signature (poses, obj_pose, widths) -> (poses, widths).

    Receives the entire world/commanded-frame batch (`grasp.py:200-206`) after flip and disable_upside_down,
    before all IK. Slice widths synchronously, or later grasp_widths[random_indices] becomes misaligned.

    Caller supplies measured axis/endpoints rather than inferring them from obj_pose, whose entity axis can flip.
    base_xy is horizontal robot-base position (robot.init_position), used to assess natural arm approach.
    """
    n = _unit(rod_axis_up)
    tip = np.asarray(tip_low, dtype=float)
    base = np.asarray(base_xy, dtype=float)[:2]
    to_obj = tip[:2] - base
    to_obj = to_obj / np.linalg.norm(to_obj)
    mid_y = float(base[1])

    def gate(grasp_poses, obj_pose, grasp_widths):
        poses = np.asarray(grasp_poses, dtype=float)
        widths = np.asarray(grasp_widths, dtype=float)
        if len(poses) == 0:
            return poses, widths

        g = poses[:, :3, 0]   # Gripper closing axis.
        c1 = poses[:, :3, 1]  # Wrist-flip detection axis.
        a = poses[:, :3, 2]   # approach
        p = poses[:, :3, 3]   # Command point (gripper center).

        # (1) Wrist-flip direction: same inequality as framework disable_upside_down (preserve mask).
        # Both grip_sign values are present, so this should remove about half the candidates, not zero.
        ok_roll = (c1[:, 2] < 0.0) if arm == "left" else (c1[:, 2] > 0.0)
        # (2) Closing axis perpendicular to rod axis, allowing grasp rather than axial pushing.
        ok_perp = np.abs(g @ n) <= 0.10
        # (3) Cosine between approach and base-to-object direction: **compute/report only, no filtering**.
        # See comparisons at REACH_COS_MIN: successful candidate reach_cos was as low as -0.018,
        # while the launching candidate was +0.358; no verified correlation with success exists.
        reach_cos = a[:, :2] @ to_obj
        ok_reach = (np.ones(len(poses), dtype=bool) if REACH_COS_MIN is None
                    else reach_cos > REACH_COS_MIN)
        # (4) Farthest gripper-body extent along -approach must not cross the body centerline; this is the actual harmful quantity.
        body = p - JAW_BODY_REACH * a
        ok_body = ((body[:, 1] <= mid_y + JAW_MIDLINE_TOL) if arm == "right"
                   else (body[:, 1] >= mid_y - JAW_MIDLINE_TOL))
        # (5) Open fingertips must stay above paper-cup rim to avoid collision during entry.
        tip_open = p + PAD_SPAN_OPEN[1] * a
        ok_tip = tip_open[:, 2] >= cup_mouth_z + 0.005
        # (6) Perpendicular gripper-to-**rod-axis line** distance = depth(elev)*cos(elev); check an interval because elevation varies.
        w = p - tip[None, :]
        perp = w - np.outer(w @ n, n)
        perp_d = np.linalg.norm(perp, axis=1)
        ok_depth = (perp_d >= PERP_DIST_RANGE[0] - 0.004) & (perp_d <= PERP_DIST_RANGE[1] + 0.004)

        # Accumulate mask so each criterion counts only candidates first rejected at that stage.
        m1 = ok_roll
        m2 = m1 & ok_perp
        m3 = m2 & ok_reach
        m4 = m3 & ok_body
        m5 = m4 & ok_tip
        keep = m5 & ok_depth
        print("[GRASP][%s] candidates %d -> %d  (wrist flip %d[about half expected], closing axis not perpendicular to rod %d, "
              "approach deviates from extension %d[disabled], gripper crosses midline %d, fingertips below rim %d, gripper center off rod axis %d)"
              % (tag, len(poses), int(keep.sum()),
                 int((~ok_roll).sum()), int((m1 & ~ok_perp).sum()),
                 int((m2 & ~ok_reach).sum()), int((m3 & ~ok_body).sum()),
                 int((m4 & ~ok_tip).sum()), int((m5 & ~ok_depth).sum())))
        if keep.sum():
            print("[GRASP][%s] surviving: col1_z %+.3f~%+.3f(%s arm requires%s) reach_cos %+.3f~%+.3f(no gate) "
                  "lowest fingertip z=%.4f(cup rim %.4f) perpendicular distance %.4f~%.4f"
                  % (tag, c1[keep, 2].min(), c1[keep, 2].max(), arm,
                     "<0" if arm == "left" else ">0",
                     reach_cos[keep].min(), reach_cos[keep].max(),
                     tip_open[keep, 2].min(), cup_mouth_z,
                     perp_d[keep].min(), perp_d[keep].max()))
        return poses[keep], widths[keep]

    return gate


def _straw_pose_at_depth(depth, R_straw, local_low, hole_centre, hole_axis) -> np.ndarray:
    """Return a world straw pose whose lower endpoint is at the requested depth.

    Positive depth is below the opening plane. Retreat along the straw axis,
    accounting for its projection onto the hole axis; use the hole-axis fallback
    when that projection is near zero.
    """
    axis_up = R_straw @ LOCAL_UP
    cos_t = float(np.dot(axis_up, hole_axis))
    T = np.eye(4)
    T[:3, :3] = R_straw
    if abs(cos_t) < 1e-3:
        # For a nearly horizontal straw, use the hole-axis formula to avoid
        # dividing by a near-zero axial projection.
        T[:3, 3] = hole_centre - hole_axis * float(depth) - R_straw @ local_low
    else:
        T[:3, 3] = hole_centre - axis_up * (float(depth) / cos_t) - R_straw @ local_low
    return T


def _measure_engagement(robot, straw_id, milk_id):
    """Three decisive measurements.

    Returns (depth, bore_radial, tilt_deg, axis_up, tip_low, local_low, hole_centre, hole_axis).
    depth: lower endpoint distance below opening plane; positive means inserted.
    bore_radial: radial distance from hole center to straw-axis/opening-plane intersection, the decisive proof
      of real passage. Do not use endpoint radial distance: after falling to the floor it may be nearly 30 mm
      off center while correctly inserted at ~10-degree tilt. Nearly plane-parallel axes put intersection at
      infinity; return a large failing value rather than NaN.
    tilt_deg: angle between straw long axis and hole axis.
    """
    _, axis_up, tip_low, _, local_low = _straw_state(robot, straw_id)
    hole_centre, hole_axis = _hole_frame(robot, milk_id)

    depth = float(np.dot(hole_centre - tip_low, hole_axis))
    cos_t = float(np.dot(axis_up, hole_axis))
    if abs(cos_t) < 1e-3:
        bore_radial = 1.0  # Axis nearly parallel to opening plane: cannot claim penetration.
    else:
        t = depth / cos_t
        hit = tip_low + axis_up * t
        bore_radial = float(np.linalg.norm(hit - hole_centre))
    tilt = float(np.degrees(np.arccos(np.clip(abs(cos_t), -1.0, 1.0))))
    return depth, bore_radial, tilt, axis_up, tip_low, local_low, hole_centre, hole_axis


def _fractional_min_rotation(R_delta, fraction):
    """Return ``R_delta`` raised to ``fraction`` for a non-reflective rotation.

    The only caller feeds the minimum axis-alignment rotation, whose angle is
    at most 90 degrees.  Interpolating that rotation (instead of applying the
    whole correction at the last waypoint) is the important part of the coin
    insertion pattern: every ordered object waypoint has a continuous pose.
    """
    fraction = float(np.clip(fraction, 0.0, 1.0))
    trace = float(np.trace(R_delta))
    angle = float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1e-8 or fraction == 0.0:
        return np.eye(3)
    skew = np.array([
        R_delta[2, 1] - R_delta[1, 2],
        R_delta[0, 2] - R_delta[2, 0],
        R_delta[1, 0] - R_delta[0, 1],
    ])
    axis = _unit(skew / (2.0 * np.sin(angle)))
    return _rot_about_axis(axis, fraction * angle)


def _straw_pose_waypoint(T):
    """Serialize a straw entity pose as follow_trajectory xyz+wxyz."""
    q = mat2quat_wxyz(T[:3, :3])
    return np.concatenate((T[:3, 3], q)).astype(float)


def _follow_straw_lower_tip_path(robot, straw_id, milk_id, arm, target_depth,
                                  tag, action_description, *, align_axis,
                                  aim_centre=None, segment_count=3):
    """Follow an ordered *straw entity* path whose lower-end line hits the hole.

    ``follow_trajectory(actor=straw_id)`` accepts object poses, never gripper
    poses.  For each future waypoint we therefore solve the straw entity pose
    from its live lower endpoint and the live lid frame with
    ``_straw_pose_at_depth``.  The RobotAPI itself captures the current
    object-to-gripper transform and derives the gripper path.  Supplying a
    gripper waypoint here would silently track the wrong point whenever the
    straw slips in the compliant 90D jaws.

    The zeroth waypoint is the exact measured entity pose.  This preserves the
    captured object-to-EE relation at the start.  Later points linearly advance
    the *lower end* through the requested signed depths while the straw axis
    intersects the lid plane at ``aim_centre``.  Thus the path constraint is
    the lower straw end / hole centre geometry, rather than a guessed gripper
    centre trajectory.
    """
    if segment_count < 2:
        raise ValueError("segment_count must be at least 2")

    d_now, r_now, tilt_now, axis_up, _, local_low, hole_centre, hole_axis = (
        _measure_engagement(robot, straw_id, milk_id))
    T_now, _, _, _, _ = _straw_state(robot, straw_id)
    aim = (hole_centre if aim_centre is None
           else np.asarray(aim_centre, dtype=float))
    if aim.shape != (3,) or not np.isfinite(aim).all():
        raise ValueError("aim_centre must be a finite three-vector")

    if align_axis:
        axis_target = hole_axis if float(np.dot(axis_up, hole_axis)) >= 0.0 else -hole_axis
        R_delta = _min_rotation_aligning(axis_up, axis_target)
    else:
        # The pre-contact and contact legs intentionally preserve the measured
        # orientation.  Once close to the lid, only the lower-end line is
        # recentered; no late wrist correction is allowed to lever on the hole.
        R_delta = np.eye(3)

    cos_now = float(np.dot(axis_up, hole_axis))
    if abs(cos_now) < 1e-6:
        raise RuntimeError("current straw is nearly parallel to lid plane")
    # In the first measured pose, this is the physical straw-axis intersection
    # with the lid plane.  Interpolating it toward the hole centre is what
    # makes the air correction genuinely segmented rather than one large
    # lateral jump hidden behind several axial waypoints.
    _, _, tip_low_now, _, _ = _straw_state(robot, straw_id)
    current_plane_hit = tip_low_now + axis_up * (d_now / cos_now)

    waypoints = [_straw_pose_waypoint(T_now)]
    target_T = None
    for k in range(1, segment_count + 1):
        fraction = k / float(segment_count)
        R_k = _fractional_min_rotation(R_delta, fraction) @ T_now[:3, :3]
        d_k = (1.0 - fraction) * d_now + fraction * float(target_depth)
        plane_hit_k = (1.0 - fraction) * current_plane_hit + fraction * aim
        target_T = _straw_pose_at_depth(
            d_k, R_k, local_low, plane_hit_k, hole_axis)
        waypoints.append(_straw_pose_waypoint(target_T))

    # This is an algebraic check of the command before it reaches RobotAPI: the
    # final straw axis must cross the actual lid plane exactly at the requested
    # centre.  It also makes a payload/world-frame mix-up fail loudly.
    raw_axis = target_T[:3, :3] @ STRAW_LOCAL_AXIS
    final_tip = target_T[:3, 3] + target_T[:3, :3] @ local_low
    cos_final = float(np.dot(raw_axis, hole_axis))
    if abs(cos_final) < 1e-6:
        raise RuntimeError("straw trajectory target is nearly parallel to lid plane")
    hit = final_tip + raw_axis * (float(target_depth) / cos_final)
    final_error = float(np.linalg.norm(hit - aim))
    if final_error > 1e-7:
        raise RuntimeError("straw lower-tip trajectory misses requested hole centre: %.9fm"
                           % final_error)

    print("[FOLLOW-%s] actor=straw entity (not gripper): current %+.1fmm / %.2fmm / %.2f° -> "
          "target %+.1fmm; %d ordered waypoints, final lower-end axis-to-hole-center error %.4fmm%s"
          % (tag, 1000 * d_now, 1000 * r_now, tilt_now, 1000 * target_depth,
             len(waypoints), 1000 * final_error,
             ", straighten at each waypoint" if align_axis else ", preserve current orientation before and after contact"))
    ok = robot.follow_trajectory(
        actor=straw_id,
        waypoints=np.stack(waypoints, axis=0),
        arm=arm,
        trajectory_config=FollowTrajectoryConfig(
            # Precise insertion must not use the API's IK fallback noise: any
            # non-zero translation perturbs the lower-end/centre constraint.
            noise_range=[0.0] * 6,
            noise_frame="local",
            sample_count=1,
            unmovable_objects=[],
            action_description=action_description,
            extra_params={"skip_check": True},
        ),
    )
    robot.wait(INSERT_SETTLE_WAIT_S)
    return bool(ok)


def _near_hole_align_once(robot, straw_id, milk_id, arm, tag,
                          target_air_depth=-NEAR_HOLE_AIR_GAP,
                          aim_centre_override=None):
    """Straighten and center the straw above the opening in one motion.

    Compute the target from the measured endpoint, opening intersection and
    rod axis while preserving free-space clearance.
    """
    d, r, tilt, axis_up, _, local_low, hole_centre, hole_axis = _measure_engagement(
        robot, straw_id, milk_id)
    if d > -NEAR_HOLE_AIR_MIN_GAP:
        print("[ALIGN-%s] Current lower end is only %.1fmm above opening, below %.1fmm safe clearance; refusing correction at opening"
              % (tag, -1000 * d, 1000 * NEAR_HOLE_AIR_MIN_GAP))
        return False

    # The measured true hole above remains the safety/acceptance reference.  The returned aim
    # frame intentionally has zero precompensation: it is exactly the true hole frame.
    true_aim_centre, aim_centre, aim_hole_axis, aim_world_delta = _final_air_aim_frame(
        robot, milk_id)
    if aim_centre_override is not None:
        aim_centre = np.asarray(aim_centre_override, dtype=float)
        # This is an episode-local feedback correction, expressed in world
        # space from the just-measured real residual.  Do not call it a new
        # physical hole or feed it to any measurement/checker.
        aim_world_delta = aim_centre - true_aim_centre
    # Always choose a still-suspended target depth; do not reuse current d, which execution error may have made -2 mm.
    air_depth = min(float(target_air_depth), -NEAR_HOLE_AIR_GAP)
    print("[ALIGN-%s] Combined multipoint correction: depth %+.1fmm, radial %.2fmm, tilt %.2f° -> target %+.1fmm; "
          "true hole center=%s, payload precompensation(X/Z)=(%+.3f,%+.3f)mm, world offset=%smm, commanded aim=%s"
          % (tag, 1000 * d, 1000 * r, tilt, 1000 * air_depth,
             np.round(true_aim_centre, 6).tolist(),
             1000 * FINAL_AIM_PRECOMP_LOCAL[0],
             1000 * FINAL_AIM_PRECOMP_LOCAL[2],
             np.round(1000 * aim_world_delta, 3).tolist(),
             np.round(aim_centre, 6).tolist()))
    ok = _follow_straw_lower_tip_path(
        robot, straw_id, milk_id, arm, air_depth, "AIR-ALIGN-%s" % tag,
        {"zh": "沿吸管下端点轨迹在孔口上方逐点扶正并精确对心",
         "en": "follows the straw lower-tip path to upright and precisely center above the lid hole"},
        align_axis=True,
        aim_centre=aim_centre,
        segment_count=3)
    # Report the signed residual in the same payload X/Z coordinates used to derive the
    # precompensation.  This is diagnostic only: all gates below still call
    # `_measure_engagement()` against the unmodified real hole centre.
    if ok:
        d_after, _, _, axis_after, tip_after, _, hole_after, axis_hole_after = (
            _measure_engagement(robot, straw_id, milk_id))
        cos_after = float(np.dot(axis_after, axis_hole_after))
        if abs(cos_after) >= 1e-3:
            hit_after = tip_after + axis_after * (d_after / cos_after)
            cup_pos, cup_quat = robot.get_object_pose(milk_id)
            R_cup_after = _pose_from_xyz_wxyz(cup_pos, cup_quat)[:3, :3]
            local_residual = R_cup_after.T @ (hit_after - hole_after)
            print("[ALIGN-%s] Achieved opening intersection relative to true hole center in payload(X/Y/Z)=%smm"
                  % (tag, np.round(1000 * local_residual, 3).tolist()))
    return bool(ok)


def _near_hole_feedback_retry(robot, straw_id, milk_id, arm):
    """Apply one bounded feedback retry from the achieved signed radial residual.

    Keep the lower endpoint above the configured minimum air gap.
    """
    d, r, tilt, axis_up, tip_low, _, hole_centre, hole_axis = _measure_engagement(
        robot, straw_id, milk_id)
    if d > -NEAR_HOLE_AIR_MIN_GAP:
        print("[ALIGN-FEEDBACK] Current clearance %.1fmm is insufficient; skipping second airborne correction" % (-1000 * d))
        return False
    cos_t = float(np.dot(axis_up, hole_axis))
    if abs(cos_t) < 1e-3:
        print("[ALIGN-FEEDBACK] Straw axis nearly parallel to opening plane; cannot calculate valid intersection")
        return False
    hit = tip_low + axis_up * (d / cos_t)
    residual = hit - hole_centre
    residual -= float(np.dot(residual, hole_axis)) * hole_axis
    correction = float(np.linalg.norm(residual))
    if correction > NEAR_HOLE_FEEDBACK_MAX_CORRECTION:
        print("[ALIGN-FEEDBACK] Measured residual %.2fmm exceeds single-correction limit %.2fmm; stopping"
              % (1000 * correction, 1000 * NEAR_HOLE_FEEDBACK_MAX_CORRECTION))
        return False
    # If achieved hit = commanded aim + residual, use aim = true hole - residual.
    feedback_aim = hole_centre - residual
    print("[ALIGN-FEEDBACK] Compensate opposite to current measured residual: r=%.2fmm tilt=%.2f°, "
          "world correction=%smm; retry only once"
          % (1000 * r, tilt, np.round(-1000 * residual, 3).tolist()))
    return _near_hole_align_once(
        robot, straw_id, milk_id, arm, "NEAR-HOLE-FEEDBACK",
        aim_centre_override=feedback_aim)


def _withdraw_open_gripper_radially(robot, straw_id, milk_id, arm):
    """Withdraw an already-open, detached gripper radially from the live straw axis.

    Use bounded empty-hand moves and check that the straw remains captured.
    """
    for i, distance in enumerate(RELEASE_RADIAL_WITHDRAW_STEPS, start=1):
        d0, r0, tilt0, axis_up, tip_low, _, _, _ = _measure_engagement(
            robot, straw_id, milk_id)
        ee = np.asarray(robot.get_ee_pose(arm=arm), dtype=float)
        along = float(np.dot(ee[:3, 3] - tip_low, axis_up))
        axis_point = tip_low + along * axis_up
        away = ee[:3, 3] - axis_point
        away -= float(np.dot(away, axis_up)) * axis_up
        norm = float(np.linalg.norm(away))
        if norm < 1e-5:
            # Degenerate only if the gripper centre sits exactly on the rod;
            # use a deterministic horizontal vector perpendicular to the rod.
            away = np.cross(axis_up, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(away) < 1e-5:
                away = np.cross(axis_up, np.array([1.0, 0.0, 0.0]))
            norm = float(np.linalg.norm(away))
        away /= norm
        target = ee.copy()
        target[:3, 3] += float(distance) * away
        print("[RELEASE-RADIAL] %d/%d: withdraw empty gripper %.1fmm away from straw axis; "
              "before withdrawal +%.1fmm / %.2fmm / %.2f°"
              % (i, len(RELEASE_RADIAL_WITHDRAW_STEPS), 1000 * distance,
                 1000 * d0, 1000 * r0, tilt0))
        ok = robot.move_actor_to_target_pose(
            actor="gripper", target_pose=target[np.newaxis, ...], arm=arm,
            move_config=MoveConfig(
                unmovable_objects=[],
                action_description={"zh": "张开夹爪沿远离吸管轴方向退开",
                                    "en": "withdraws the open gripper radially away from the straw axis"},
                extra_params={"skip_check": True,
                              "slowdown_factor": INSERT_SLOWDOWN_FACTOR}))
        robot.wait(RELEASE_RADIAL_WITHDRAW_WAIT_S)
        d1, r1, tilt1 = _report(robot, straw_id, milk_id, "after native opening and radial withdrawal %d" % i)
        if not ok:
            print("[RELEASE-RADIAL] Planning failed; stopping further withdrawal")
            return False
        if r1 >= ACCEPT_BORE_RADIAL_MAX or tilt1 >= ACCEPT_TILT_DEG_MAX:
            print("[RELEASE-RADIAL] Opening state worsened after withdrawal; stopping further withdrawal")
            return False
        if d1 >= RELEASE_FALL_DEPTH_MIN:
            print("[RELEASE-RADIAL] Straw has fallen to +%.1fmm, confirming release from gripper support" % (1000 * d1))
            return True
    return True


def _move_axially_to_depth(robot, straw_id, milk_id, arm, target_depth, tag,
                            action_description):
    """Change endpoint depth only along hole axis, strictly preserving lateral position and orientation.

    First stage of two-stage descent. Do not use _straw_pose_at_depth(), which also centers the axis:
    that would combine descent to 5 mm with correction there, obscuring which motion introduced error.
    """
    d, r, tilt, _, _, _, _, hole_axis = _measure_engagement(robot, straw_id, milk_id)
    T_now, _, _, _, _ = _straw_state(robot, straw_id)
    target = T_now.copy()
    # depth = dot(hole_centre - tip_low, hole_axis). Translating the object by delta along -hole_axis
    # increases depth by exactly delta, preserving R and lateral components and thus opening radial error.
    target[:3, 3] -= hole_axis * (float(target_depth) - d)
    print("[AXIAL-%s] Along hole axis only: depth %+.1fmm -> %+.1fmm; preserving radial %.2fmm, tilt %.2f°"
          % (tag, 1000 * d, 1000 * target_depth, 1000 * r, tilt))
    ok = robot.move_actor_to_target_pose(
        actor=straw_id, target_pose=target[np.newaxis, ...], arm=arm,
        move_config=MoveConfig(
            # Target changes only world Z; lock orientation and world X/Y so cuRobo obstacle avoidance cannot turn
            # axial descent into lateral travel or wrist rotation. Matches planner/action/insert.py constraints.
            path_constraint=[0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
            offset_and_constraint_in_goal_frame=False,
            unmovable_objects=[],
            action_description=action_description,
            extra_params={"skip_check": True,
                          "slowdown_factor": INSERT_SLOWDOWN_FACTOR}))
    robot.wait(INSERT_SETTLE_WAIT_S)
    return ok


def _set_gripper_q(robot, arm, q_target, tag):
    """Use the implemented RobotAPI joint controller for a symmetric 90D q target."""
    if arm == "left":
        joint_names = [
            "idx32_gripper_l_outer_joint1",
            "idx31_gripper_l_inner_joint1",
        ]
    elif arm == "right":
        joint_names = [
            "idx72_gripper_r_outer_joint1",
            "idx71_gripper_r_inner_joint1",
        ]
    else:
        print("[GRIPPER-HOLD-%s] Unknown arm %r" % (tag, arm))
        return False

    q_target = float(np.clip(q_target, GRIPPER_CLOSED_Q, 0.0))
    if not np.isfinite(q_target):
        print("[GRIPPER-HOLD-%s] Invalid 90D target joint reading: %r" % (tag, q_target))
        return False

    # ``joint_to`` is the public API that reaches the collector's implemented
    # SET_JOINT_POSITION path.  One interpolation step keeps this a single
    # command; callers deliberately choose q from a live readback or a 1mrad
    # absolute ladder, so it never creates an implicit full-open move.
    print("[GRIPPER-HOLD-%s] Set 90D finger positions through joint_to q=%.5f" % (tag, q_target))
    ok = robot.joint_to(
        joint_names,
        {name: q_target for name in joint_names},
        steps=1,
        action_description={"zh": "保持吸管夹持时的夹爪开度",
                            "en": "holds the gripper opening while retaining the straw"},
    )
    if not ok:
        print("[GRIPPER-HOLD-%s] joint_to position-hold command failed" % tag)
    return bool(ok)


def _freeze_gripper_at_live_width(robot, arm, tag):
    """Switch the velocity-close command to an implemented live-q position hold."""
    if arm == "left":
        joint_names = ["idx32_gripper_l_outer_joint1", "idx31_gripper_l_inner_joint1"]
    elif arm == "right":
        joint_names = ["idx72_gripper_r_outer_joint1", "idx71_gripper_r_inner_joint1"]
    else:
        print("[GRIPPER-HOLD-%s] Unknown arm %r" % (tag, arm))
        return False
    live = robot.get_joint_positions()
    missing = [name for name in joint_names if name not in live]
    if missing:
        print("[GRIPPER-HOLD-%s] Cannot find 90D gripper joints %s" % (tag, missing))
        return False
    q_live = np.asarray([float(live[name]) for name in joint_names], dtype=float)
    if not np.all(np.isfinite(q_live)):
        print("[GRIPPER-HOLD-%s] Invalid 90D gripper joint readings: %s" % (tag, q_live.tolist()))
        return False
    asymmetry = float(np.ptp(q_live))
    if asymmetry > GRIPPER_HOLD_MAX_ASYMMETRY_RAD:
        print("[GRIPPER-HOLD-%s] Finger q difference %.3fmrad exceeds %.3fmrad; refusing forced symmetric hold"
              % (tag, 1000 * asymmetry, 1000 * GRIPPER_HOLD_MAX_ASYMMETRY_RAD))
        return False
    return _set_gripper_q(robot, arm, float(np.max(q_live)), tag)


def _slip_check(robot, straw_id, arm, grip_ref, tag):
    """Report straw slip relative to the gripper and its pad-window position.

    Measure the rod axis at the pad midline and compare live relative poses
    through closure, extraction, straightening and insertion.
    """
    ee = np.asarray(robot.get_ee_pose(arm=arm), dtype=float)
    T_s, axis_up, _, _, _ = _straw_state(robot, straw_id)
    rel = np.linalg.inv(ee) @ T_s
    d = rel[:3, 3] - grip_ref[:3, 3]
    total = float(np.linalg.norm(d))
    # Axial component (rod-axis direction in gripper frame).
    axis_in_ee = _unit(np.linalg.inv(ee)[:3, :3] @ axis_up)
    axial = float(np.dot(d, axis_in_ee))
    flag = "" if total < SLIP_WARN else "  ⚠ slip exceeds threshold"
    print("[SLIP-%s] In-gripper displacement %.1fmm(along rod axis %+.1fmm) reference %s -> current %s%s"
          % (tag, 1000 * total, 1000 * axial,
             np.round(grip_ref[:3, 3] * 1000, 1).tolist(),
             np.round(rel[:3, 3] * 1000, 1).tolist(), flag))
    # Locate the rod axis at col1=0 in the gripper frame, at the finger-pad
    # midline, rather than using a tilted endpoint or the straw center.
    c = rel[:3, 3]
    if abs(axis_in_ee[1]) > 1e-6:
        t0 = -c[1] / axis_in_ee[1]
        p_mid = c + t0 * axis_in_ee
        # Rod elevation relative to approach changes its approach coordinate across pad col1 width.
        p_a = p_mid + (JAW_HALF_WIDTH_ALONG_ROD / axis_in_ee[1]) * axis_in_ee
        p_b = p_mid - (JAW_HALF_WIDTH_ALONG_ROD / axis_in_ee[1]) * axis_in_ee
        span = (min(p_a[2], p_b[2]), max(p_a[2], p_b[2]))
    else:
        p_mid, span = c, (c[2], c[2])
    lo, hi = PAD_WINDOW
    frac = (p_mid[2] - lo) / (hi - lo) if hi > lo else float("nan")
    ok = (lo + STRAW_R) <= span[0] and span[1] <= (hi - STRAW_R)
    print("[PAD-%s] Rod axis(at pad midline) approach %+.1fmm closing axis %+.1fmm; span across pad width "
          "[%+.1f,%+.1f]mm => closed pad surface [%.0f,%.0f], position %.0f%%, "
          "rear-edge distance %+.1fmm front-edge distance %+.1fmm => %s"
          % (tag, 1000 * p_mid[2], 1000 * p_mid[0], 1000 * span[0], 1000 * span[1],
             1000 * lo, 1000 * hi, 100 * frac,
             1000 * (span[0] - STRAW_R - lo), 1000 * (hi - span[1] - STRAW_R),
             "on pad" if ok else "**rod beyond pad surface**"))
    return total, axial, float(p_mid[2])


def _jaw_axial_measured(robot, straw_id, arm):
    """Return measured distance L from the lower endpoint to the gripper center.

    Use the live relative pose to account for axial slip.
    """
    _, axis_up, tip_low, _, _ = _straw_state(robot, straw_id)
    ee = np.asarray(robot.get_ee_pose(arm=arm), dtype=float)
    return float(np.dot(ee[:3, 3] - tip_low, axis_up))


def _max_command_depth(robot, straw_id, arm):
    """Limit commanded insertion depth to preserve JAW_ABOVE_LID_MARGIN.

    The cap uses live gripper-to-straw geometry, so axial clearance is L-D.
    """
    L_act = _jaw_axial_measured(robot, straw_id, arm)
    cap = min(L_act - JAW_ABOVE_LID_MARGIN, MAX_COMMAND_DEPTH_CEILING)
    return max(0.0, cap), L_act


def _cup_exit_state(robot, straw_id, arm, cup_mouth_z, upright_axis):
    """Return lower-endpoint clearance above the source-cup rim and straightening drop.

    Predict endpoint drop from the live tilt and grasp-anchor distance.
    """
    _, axis_up, tip_low, _, _ = _straw_state(robot, straw_id)
    ee = np.asarray(robot.get_ee_pose(arm=arm), dtype=float)
    L_act = float(np.dot(ee[:3, 3] - tip_low, axis_up))
    tilt = float(np.arccos(np.clip(abs(float(np.dot(axis_up, _unit(upright_axis)))), -1.0, 1.0)))
    swing = max(0.0, L_act) * (1.0 - float(np.cos(tilt)))
    clear = float(tip_low[2]) - cup_mouth_z
    return float(tip_low[2]), clear, swing, clear - swing, L_act, float(np.degrees(tilt))


def _report(robot, straw_id, milk_id, tag):
    d, r, a = _measure_engagement(robot, straw_id, milk_id)[:3]
    print("[MEASURE-%s] Depth=%+.4f m  opening radial=%.4f m  tilt=%.2f°" % (tag, d, r, a))
    return d, r, a


def _paper_cup_state(robot, paper_id, paper_T0):
    """Return source-paper-cup tilt and displacement from its initial pose.

    Use the entity frame's local Z axis for the upright direction.
    """
    pos, quat = robot.get_object_pose(paper_id)
    T = _pose_from_xyz_wxyz(pos, quat)
    up = _unit(T[:3, :3] @ LOCAL_UP)
    tilt = float(np.degrees(np.arccos(np.clip(abs(float(up[2])), -1.0, 1.0))))
    moved = float(np.linalg.norm(np.asarray(pos) - np.asarray(paper_T0[:3, 3])))
    return tilt, moved


def _scene_sanity_check(robot, straw_id, paper_id, milk_id):
    """Check settled object poses and clearances before task execution.

    Report an invalid starting scene before grasping or transport.
    """
    problems = []
    for name, oid, upright_tol in (("paper cup", paper_id, 25.0), ("milk-tea cup", milk_id, 25.0)):
        pos, quat = robot.get_object_pose(oid)
        T = _pose_from_xyz_wxyz(pos, quat)
        up = _unit(T[:3, :3] @ LOCAL_UP)  # Payload up is +Y, not +Z.
        tilt = float(np.degrees(np.arccos(np.clip(abs(float(up[2])), -1.0, 1.0))))
        if tilt > upright_tol:
            problems.append("%s tipped %.1f°(allowed %.0f°)" % (name, tilt, upright_tol))
        if abs(float(pos[0])) > 0.45 or abs(float(pos[1])) > 0.45:
            problems.append("%s left tabletop xy=(%.3f,%.3f)" % (name, pos[0], pos[1]))
        print("[SANITY] %s: xy=(%+.3f,%+.3f) z=%.4f deviation from vertical=%.1f°"
              % (name, pos[0], pos[1], pos[2], tilt))

    paper_pos, _ = robot.get_object_pose(paper_id)
    _, axis_up, tip_low, tip_high, _ = _straw_state(robot, straw_id)
    horiz = float(np.linalg.norm(np.asarray(tip_low)[:2] - np.asarray(paper_pos)[:2]))
    tilt_straw = float(np.degrees(np.arccos(np.clip(abs(float(axis_up[2])), -1.0, 1.0))))
    print("[SANITY] Straw: lower end=%s upper end z=%.4f deviation from vertical=%.1f° lower-end horizontal distance to paper-cup axis=%.4f"
          % (np.round(tip_low, 4).tolist(), float(tip_high[2]), tilt_straw, horiz))
    # Straw should remain in paper cup: lower endpoint distance from cup axis must not exceed inner radius.
    if horiz > PAPER_CUP_MOUTH_R_IN:
        problems.append("Straw is no longer inside paper cup(lower-end distance to cup axis %.3f > cup inner radius %.3f)"
                        % (horiz, PAPER_CUP_MOUTH_R_IN))
    if tilt_straw > 45.0:
        problems.append("Straw tilt %.1f° is excessive(>45°)" % tilt_straw)

    if problems:
        return False, "；".join(problems)
    return True, "Initial state normal"


# ---------------------------------------------------------------------------


def execute_task(robot: RobotAPI):
    scene_info = robot.scene_info

    robot.client.set_task_task_basic_info({
        "task_name": 'Insert a straw into a milk tea cup',
        "english_task_name": (
            'Insert a straw into a milk tea cup'
        ),
        "init_scene_text": (
            "A straw stands in a paper cup, and a lidded milk tea cup with a straw hole "
            "sits beside it; the two cups are arranged along the desktop Y axis with the "
            "milk tea cup farther from the working arm."
        ),
    })

    # ------------------------------------------------------------------
    # Step 1: locate objects and choose arm.
    # ------------------------------------------------------------------
    straw_id = _find_unique_object(scene_info, "target_straw")
    paper_id = _find_unique_object(scene_info, "source_paper_cup")
    milk_id = _find_unique_object(scene_info, "target_milk_tea_cup")

    paper_pos, paper_quat = robot.get_object_pose(paper_id)
    paper_T = _pose_from_xyz_wxyz(paper_pos, paper_quat)
    milk_pos, milk_quat = robot.get_object_pose(milk_id)
    milk_T = _pose_from_xyz_wxyz(milk_pos, milk_quat)

    # Deterministically choose arm according to the side containing the straw's paper cup.
    # Do not replace with robot.judge_arm_to_use(): `robot_api.py:941` uses THRESHOLD=0.15,
    # selecting **probabilistically** for |y|<=0.15 (P(left) rises linearly from 0 at y=-0.15 to 1 at +0.15).
    # Both work points necessarily fall within that band, causing roughly one wrong-arm choice every nine episodes.
    # Layout is randomly mirrored, so arm/layout mismatch makes those episodes essentially unusable.
    paper_y = float(paper_T[1, 3])
    milk_y = float(milk_T[1, 3])
    arm = "left" if paper_y >= 0.0 else "right"

    print("[STEP1] straw=%s | paper_cup=%s y=%+.3f | milk_tea=%s y=%+.3f | arm=%s"
          % (straw_id, paper_id, paper_y, milk_id, milk_y, arm))
    if (milk_y - paper_y) * (1.0 if arm == "left" else -1.0) > 0.0:
        print("[WARN] Milk-tea cup is not closer to midline than paper cup: scene differs from expected layout; insertion point will be on opposite side")

    hole_centre0, hole_axis0 = _hole_frame(robot, milk_id)
    print("[STEP1] Hole center(world)=%s hole axis=%s hole inscribed radius=%.4f one-sided clearance=%.4f"
          % (np.round(hole_centre0, 4).tolist(), np.round(hole_axis0, 4).tolist(),
             HOLE_R_IN, HOLE_R_IN - STRAW_R))

    # ------------------------------------------------------------------
    # Step 2: initial-state sanity and measured settled straw pose.
    # ------------------------------------------------------------------
    # Paper-cup rim inner radius 30.6 mm greatly exceeds straw radius 6 mm, so the straw inevitably leans against one side.
    # **Measure** the settled pose; do not design later motions using the scene's upright initial pose or an inferred solution.
    sane, diag = _scene_sanity_check(robot, straw_id, paper_id, milk_id)
    if not sane:
        print("[ERROR] step 2 initial scene unusable: %s" % diag)
        print("[ERROR] This is a **scene/physics** issue, not a grasp issue. First inspect asset collision approximation"
              " (the 0.33mm straw wall was once ejected with sdf) and initial penetration; do not tune grasp candidates.")
        return False
    print("[STEP2] Sanity check: %s" % diag)

    straw_T0, straw_axis0, straw_tip_low0, straw_tip_high0, straw_local_low0 = _straw_state(
        robot, straw_id)
    tilt0 = float(np.degrees(np.arccos(np.clip(abs(straw_axis0[2]), -1.0, 1.0))))
    cup_mouth_z = float(paper_T[2, 3] + PAPER_CUP_DEPTH / 2.0)
    print("[STEP2] Straw settled: axis(up)=%s tilt=%.2f° lower end=%s upper end z=%.4f | paper-cup rim z=%.4f"
          % (np.round(straw_axis0, 4).tolist(), tilt0,
             np.round(straw_tip_low0, 4).tolist(), float(straw_tip_high0[2]), cup_mouth_z))

    # Before grasping any object, exclude the paper cup from obstacles: the gripper must reach directly above its opening and the rim lies in its travel.
    robot.remove_objs_from_obstacle(_collision_prim_paths(robot, paper_id))
    print("[STEP2] Removed paper-cup collider from obstacle world(exact mesh path)")

    # ------------------------------------------------------------------
    # Step 3: grasp and extract straw.
    # ------------------------------------------------------------------
    cands = _straw_grasp_candidates(straw_axis0, straw_tip_low0, "straw")
    base_xy = np.asarray(robot.init_position, dtype=float)
    gate = _make_straw_grasp_gate(
        arm, cup_mouth_z, straw_axis0, straw_tip_low0, base_xy, "dry run")
    kept, _ = gate(cands, straw_T0, np.full(len(cands), 0.08))
    if len(kept) == 0:
        print("[ERROR] step 3 hard filters removed all grasp candidates. Use the previous line's **per-filter counts** to find the rejecting filter; "
              "do not assume a scene issue.")
        return False

    # Use a small initial pick_up_distance, then extract along the straw axis
    # to reduce contact with the paper-cup wall.
    pick_cfg = GraspConfig(
        grasp_pose=cands,
        grasp_pose_frame="world",   # Candidates already constructed in world frame from measured axis and lower endpoint.
        grasp_offset=0.0,          # Explicit 0: PickStage defaults to 0.03, pushing gripper center another 30 mm forward.
        disable_upside_down=False,  # Implement the same gate locally to print per-criterion counts.
        flip_grasp=False,           # Both grip_sign values already generated; mirroring again would restore rejected candidates.
        pick_up_distance=PICK_UP_SEAT_BREAK,
        grasp_pose_filter=_make_straw_grasp_gate(
            arm, cup_mouth_z, straw_axis0, straw_tip_low0, base_xy, "stage"),
        # GraspConfig.pre_grasp_offset is ineffective (`grasp.py:50` value is overwritten by a same-name local at :548).
        # The effective parameter is extra_params pre_grasp_distance.
        extra_params={"pre_grasp_distance": 0.07},
        # Exclude the paper cup from the stationary-object gate during straw pickup;
        # use the explicit cup integrity checks after extraction.
        unmovable_objects=[milk_id],
        action_description={"zh": "从纸杯里抓起吸管并拔出来",
                            "en": "grasps the straw and pulls it out of the paper cup"},
    )
    paper_z_before = float(paper_T[2, 3])
    straw_z_before = float(straw_tip_low0[2])
    picked = robot.pick_actor(actor=straw_id, arm=arm, grasp_config=pick_cfg)

    # Log achieved straw and gripper poses after both successful and failed picks.
    _, straw_axis1, straw_tip_low1, _, _ = _straw_state(robot, straw_id)
    paper_pos1, _ = robot.get_object_pose(paper_id)
    milk_pos1, _ = robot.get_object_pose(milk_id)
    rise_abs = float(straw_tip_low1[2]) - straw_z_before
    paper_moved = float(np.linalg.norm(np.asarray(paper_pos1) - np.asarray(paper_T[:3, 3])))
    milk_moved1 = float(np.linalg.norm(np.asarray(milk_pos1) - np.asarray(milk_T[:3, 3])))
    tilt1 = float(np.degrees(np.arccos(np.clip(abs(straw_axis1[2]), -1.0, 1.0))))
    print("[STEP3-DIAGNOSTIC] pick_actor=%s | straw lower-end rise %+.1fmm tilt %.1f°->%.1f° | "
          "paper-cup displacement %.1fmm(no gate, contact is inevitable) | milk-tea-cup displacement %.1fmm"
          % (picked, 1000 * rise_abs, tilt0, tilt1,
             1000 * paper_moved, 1000 * milk_moved1))
    if not picked:
        # A failed pick rolls playback back before returning. Diagnose it from retained
        # move measurements and errors rather than the rolled-back final pose.
        print("[ERROR] step 3 pick_actor returned failure. **Do not use the previous line** (already rolled back)."
              "Inspect each step0/step1/step2 pos_diff/angle_diff in framework grasp.py: "
              "that target follows the straw: step0 passes but step1 has large angle error = closure ejected the straw.")
        robot.detach_obj()
        return False

    # Verify the straw is held using its pose relative to the gripper and cup
    # clearance, since pickup completion tolerances refer to the commanded motion.
    grip_moved_abs = abs(rise_abs)
    rise_rel = rise_abs - (float(paper_pos1[2]) - paper_z_before)
    print("[STEP3] Grasp complete: straw rose %+.1fmm(framework lifted only %.0fmm; relative to paper cup %+.1fmm)"
          ": Step 3.4's 95mm axial travel will determine whether grip holds"
          % (1000 * rise_abs, 1000 * PICK_UP_SEAT_BREAK, 1000 * rise_rel))

    # Record straw position inside the gripper and compare after every subsequent step to detect axial slip.
    # A thin rod has **only axial friction** in parallel jaws; move returning True confirms gripper arrival, not absence of rod slip.
    ee_now = np.asarray(robot.get_ee_pose(arm=arm), dtype=float)
    T_s_now, _, _, _, _ = _straw_state(robot, straw_id)
    grip_ref = np.linalg.inv(ee_now) @ T_s_now
    print("[STEP3] Recorded in-gripper reference position %s mm(compare at each subsequent step to detect axial straw slip)"
          % np.round(grip_ref[:3, 3] * 1000, 1).tolist())

    # attach reloads obstacles through set_obstacles(), so remove them again after grasp;
    # Step 2 removal does not persist.
    robot.remove_objs_from_obstacle(
        _collision_prim_paths(robot, paper_id) + _collision_prim_paths(robot, milk_id))
    print("[STEP3] attach reloaded obstacles; removed both cups again")

    # Track paper-cup motion together with straw motion to detect carrying the
    # cup during extraction.
    T_seat, axis_seat, tip_low_before_seat, _, _ = _straw_state(robot, straw_id)
    travel = SEAT_BREAK_TRAVEL
    # Axial upward translation points toward tip_high, extracting the lower endpoint from the tapered bottom.
    target_seat = T_seat.copy()
    target_seat[:3, 3] = T_seat[:3, 3] + axis_seat * travel
    print("[STEP3.4] Extract %.0fmm along straw axis(axis=%s) to free lower end from cup-bottom taper, "
          "avoiding a vertical pull that wedges it and lifts the paper cup"
          % (1000 * travel, np.round(axis_seat, 3).tolist()))
    ok_seat = robot.move_actor_to_target_pose(
        actor=straw_id, target_pose=target_seat[np.newaxis, ...], arm=arm,
        move_config=MoveConfig(
            # Exclude paper cup from unmovable, as in pick: rubbing-induced movement is inevitable here, not failure
            # (robot_api's 1e-2 threshold would reject the whole stage for this unavoidable displacement).
            unmovable_objects=[milk_id],
            action_description={"zh": "沿吸管方向把吸管从纸杯底拔松",
                               "en": "pulls the straw out along its own axis to free it "
                                     "from the paper cup bottom"},
            extra_params={"skip_check": True}))
    robot.wait(0.3)
    _slip_check(robot, straw_id, arm, grip_ref, "STEP3.4 after axial extraction")
    _, _, tip_low_seat, _, _ = _straw_state(robot, straw_id)
    # **Only reliable grip criterion**: does the straw follow this 95 mm axial travel?
    # The 3 mm post-pick lift cannot establish grip: cup support can limit rise to 0.1 mm despite real grasp.
    seat_moved = float(np.linalg.norm(np.asarray(tip_low_seat) - np.asarray(tip_low_before_seat)))
    print("[STEP3.4] Axial extraction %s; actual straw displacement %.1fmm(command %.0fmm) lower end z=%.4f(cup rim %.4f) clearance %+.1fmm"
          % ("pass" if ok_seat else "fail", 1000 * seat_moved, 1000 * travel,
             float(tip_low_seat[2]), cup_mouth_z,
             1000 * (float(tip_low_seat[2]) - cup_mouth_z)))
    if seat_moved < 0.5 * travel:
        print("[ERROR] step 3.4 straw followed only %.1fmm(less than half of commanded %.0fmm) => "
              "**no real grip**(gripper center arrived but rod did not follow). Inspect framework grasp.py:846 "
              "step1/step2: large step1 angle error = closure ejected straw; large step2 pos_diff = slipped during lift."
              % (1000 * seat_moved, 1000 * travel))
        robot.detach_obj()
        return False

    # Require the lower straw endpoint to clear the paper-cup rim before straightening.
    tip_z, clear, swing, clear_after, L_act, tilt_d = _cup_exit_state(
        robot, straw_id, arm, cup_mouth_z, hole_axis0)
    print("[STEP3.5] Lower end z=%.4f cup rim %.4f => clearance %+.1fmm; L=%.1fmm tilt %.1f° => "
          "straightening swings down %.1fmm => clearance after straightening %+.1fmm (required >%.0fmm)"
          % (tip_z, cup_mouth_z, 1000 * clear, 1000 * L_act, tilt_d,
             1000 * swing, 1000 * clear_after, 1000 * CUP_EXIT_MARGIN))

    # Supplemental lift must satisfy both cup clearance and the configured total lift.
    # Use the larger requirement.
    short_clear = max(0.0, CUP_EXIT_MARGIN - clear_after)
    risen = float(tip_z) - float(straw_tip_low0[2])          # Total achieved lift so far.
    short_height = max(0.0, PICK_UP_DISTANCE - risen)
    # Include JAW_SLIP_ALLOWANCE in the supplemental lift to preserve the
    # configured extraction margin before pre-insertion alignment.
    bump = max(short_clear + JAW_SLIP_ALLOWANCE, short_height)
    if bump > 1e-6:
        print("[STEP3.5] Add one **purely vertical** lift %.1fmm = max(clearance shortfall %.1f+slip margin %.0f, "
              "shortfall to target height %.0fmm is %.1f)(total lift so far %.1fmm)"
              % (1000 * bump, 1000 * short_clear, 1000 * JAW_SLIP_ALLOWANCE,
                 1000 * PICK_UP_DISTANCE, 1000 * short_height, 1000 * risen))
        T_now, _, _, _, _ = _straw_state(robot, straw_id)
        target = T_now.copy()
        target[2, 3] += bump
        ok_bump = robot.move_actor_to_target_pose(
            actor=straw_id, target_pose=target[np.newaxis, ...], arm=arm,
            move_config=MoveConfig(
                unmovable_objects=[milk_id],   # Paper-cup rubbing is not failure, as in pick.
                action_description={"zh": "把吸管再抬高一点，完全离开纸杯",
                                    "en": "lifts the straw clear of the paper cup"},
                extra_params={"skip_check": True}))
        robot.wait(0.3)
        _slip_check(robot, straw_id, arm, grip_ref, "STEP3.5 after extra lift")
        tip_z, clear, swing, clear_after, L_act, tilt_d = _cup_exit_state(
            robot, straw_id, arm, cup_mouth_z, hole_axis0)
        print("[STEP3.5] Extra lift -> %s; lower end z=%.4f clearance %+.1fmm after straightening %+.1fmm total lift %.1fmm"
              % ("pass" if ok_bump else "fail", tip_z, 1000 * clear, 1000 * clear_after,
                 1000 * (float(tip_z) - float(straw_tip_low0[2]))))

    # Hard gate uses the **physically necessary condition**, no endpoint below rim after straightening, rather than the margin-bearing target;
    # using the target would reject usable episodes with 10 mm clearance.
    if clear_after < 0.0:
        print("[ERROR] step 3.5 straw did not clear paper cup: lower end z=%.4f, cup rim %.4f, clearance %+.1fmm, "
              "straightening still swings down %.1fmm => afterward %+.1fmm."
              "PICK_UP_DISTANCE=%.0fmm is insufficient, or slip during lift exceeded %.0fmm margin(see [SLIP-*] above)."
              % (tip_z, cup_mouth_z, 1000 * clear, 1000 * swing, 1000 * clear_after,
                 1000 * PICK_UP_DISTANCE, 1000 * JAW_SLIP_ALLOWANCE))
        robot.detach_obj()
        return False

    # -- Paper-cup integrity: extraction is the most likely toppling stage; fail immediately if toppled --
    # Toppling is irreversible (after reaching 81.4 degrees, a measured episode never recovered). Continuing wastes
    # over ten seconds and creates bad data; cloud episode 14179967 completed and reported success despite toppling.
    paper_tilt_now, paper_moved_now = _paper_cup_state(robot, paper_id, paper_T)
    print("[STEP3.6] Paper-cup integrity: tilt %.1f°(gate %.0f°) displacement %.1fmm(gate %.0fmm)"
          % (paper_tilt_now, ACCEPT_PAPER_TILT_DEG_MAX,
             1000 * paper_moved_now, 1000 * ACCEPT_PAPER_MOVED_MAX))
    if paper_tilt_now >= ACCEPT_PAPER_TILT_DEG_MAX or paper_moved_now >= ACCEPT_PAPER_MOVED_MAX:
        print("[ERROR] step 3.6 **paper cup tipped/dragged during straw extraction**(tilt %.1f°, displacement %.1fmm)."
              "The straw was extracted from the paper cup; tipping the cup is an invalid operation, so fail immediately "
              "and stop: tipping is irreversible, and continuing would produce bad data with an inserted straw and fallen paper cup."
              "Use paper-cup displacement in [STEP3-DIAGNOSTIC]/[STEP3.4] above to locate the tipping stage."
              % (paper_tilt_now, 1000 * paper_moved_now))
        robot.detach_obj()
        return False

    # STEP 3.7: preserve physical finger contact through transport, alignment
    # and insertion. Pose pinning would override gravity and live contact response.
    straw_air_pin_group = None

    def _release_straw_air_pin(reason):
        """Release pin used only for contact-free transport; safe to call repeatedly."""
        nonlocal straw_air_pin_group
        if straw_air_pin_group is None:
            return True
        group = straw_air_pin_group
        try:
            released = robot.release_rigid_pin(group)
        except Exception as exc:
            print("[PIN] ERROR: Failed to release airborne rigid pin(%s): %s" % (reason, exc))
            return False
        if group not in released:
            print("[PIN] ERROR: Airborne rigid pin release unconfirmed(%s): group=%s released=%s"
                  % (reason, group, released))
            return False
        straw_air_pin_group = None
        print("[PIN] Released airborne rigid pin: %s (group=%s)" % (reason, group))
        return True

    print("[HOLD] Maintain real finger-friction grip throughout Step4--6.1")

    # ------------------------------------------------------------------
    # Step 4: straighten straw vertically about the gripper, rotating only the wrist.
    # ------------------------------------------------------------------
    # Separate stationary rotation and translation: combining them lets curobo trade orientation for position,
    # returning a more tilted solution when unreachable instead of failing.
    #
    # `move_actor_to_target_pose(actor=straw, ...)` preserves the object-to-EE transform
    # measured at call time. Changing only target rotation while preserving straw-center
    # position makes the target gripper trace an arc about **straw center**, rather than rotate only the wrist; measured
    # gripper translation of 17--22 mm pushes the thin straw beyond the pad's leading edge. The physical pivot must be
    # the current grasp point: move object center with rotation so the API's derived target EE still equals current EE.
    T_now, axis_now, _, _, _ = _straw_state(robot, straw_id)
    R_upright = _min_rotation_aligning(axis_now, hole_axis0) @ T_now[:3, :3]
    tilt_now = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(axis_now, hole_axis0))), -1.0, 1.0))))
    if tilt_now > 1.0:
        ee_now = np.asarray(robot.get_ee_pose(arm=arm), dtype=float)
        object_to_ee = np.linalg.inv(T_now) @ ee_now
        target = T_now.copy()
        target[:3, :3] = R_upright
        target[:3, 3] = ee_now[:3, 3] - R_upright @ object_to_ee[:3, 3]
        predicted_ee = target @ object_to_ee
        pivot_residual = float(np.linalg.norm(predicted_ee[:3, 3] - ee_now[:3, 3]))
        print("[STEP4] Straighten around current grip point: predicted gripper translation residual %.4fmm(should approach 0; "
              "straw center rotates by %s mm)"
              % (1000 * pivot_residual,
                 np.round((target[:3, 3] - T_now[:3, 3]) * 1000, 2).tolist()))
        ok = robot.move_actor_to_target_pose(
            actor=straw_id, target_pose=target[np.newaxis, ...], arm=arm,
            move_config=MoveConfig(
                unmovable_objects=[milk_id],  # As above: sweeping the paper cup is not failure.
                action_description={"zh": "把吸管扶正到竖直",
                                    "en": "straightens the straw to vertical"},
                extra_params={"skip_check": True}))
        robot.wait(0.3)
        _, axis_after, _, _, _ = _straw_state(robot, straw_id)
        tilt_after = float(np.degrees(np.arccos(
            np.clip(abs(float(np.dot(axis_after, hole_axis0))), -1.0, 1.0))))
        print("[STEP4] Straightening(minimal rotation %.2f°) -> %s; tilt %.2f° -> %.2f°"
              % (tilt_now, "pass" if ok else "fail", tilt_now, tilt_after))
        _, _, pad_z4 = _slip_check(robot, straw_id, arm, grip_ref, "STEP4 after straightening")
        # Account for measured inward drift through straightening and insertion so
        # the rod remains within the available pad length.
        drift4 = -GRASP_DEPTH - pad_z4          # Positive = inward drift amount.
        rod_lo4 = pad_z4 - _ROD_HALF_SPAN
        rod_hi4 = pad_z4 + _ROD_HALF_SPAN
        print("[STEP4] Rod axis %.1fmm(start %.1f, inward drift %+.1fmm / budget %.0fmm); "
              "rod span [%.1f, %.1f]mm, pads [%.1f, %.1f]mm"
              % (1000 * pad_z4, -1000 * GRASP_DEPTH, 1000 * drift4,
                 1000 * GRASP_DRIFT_BUDGET, 1000 * rod_lo4, 1000 * rod_hi4,
                 1000 * PAD_WINDOW[0], 1000 * PAD_WINDOW[1]))
        if rod_lo4 < PAD_WINDOW[0] or rod_hi4 > PAD_WINDOW[1]:
            edge = "rear edge" if rod_lo4 < PAD_WINDOW[0] else "front edge"
            overrun = (PAD_WINDOW[0] - rod_lo4
                       if rod_lo4 < PAD_WINDOW[0] else rod_hi4 - PAD_WINDOW[1])
            print("[ERROR] step 4 straw slipped beyond pad %s by %.1fmm after straightening(rod span [%.1f, %.1f], "
                  "pads [%.1f, %.1f])."
                  "This is not an insertion issue: continuing will only fail as 'insertion blocked/excessive tilt'."
                  "Lever arm is |L - half-length|(currently %.1fmm, straightening arcs gripper center %.1fmm, rod follows only halfway through friction): "
                  "move GRASP_L_LADDER closer to %.1fmm, or increase GRASP_DRIFT_BUDGET to place the grasp farther forward."
                  % (edge, 1000 * overrun, 1000 * rod_lo4, 1000 * rod_hi4,
                     1000 * PAD_WINDOW[0], 1000 * PAD_WINDOW[1],
                     1000 * abs(min(GRASP_L_LADDER) - STRAW_HALF_LEN),
                     2000 * abs(min(GRASP_L_LADDER) - STRAW_HALF_LEN)
                     * float(np.sin(np.radians(tilt_now) / 2.0)),
                     1000 * STRAW_HALF_LEN))
            _release_straw_air_pin("Step4 pad bounds exceeded")
            robot.detach_obj()
            return False
        # Straightening about gripper center swings the endpoint down. Step 3.5 budgeted predicted clearance, but prediction is not achievement
        # (the rod may also slip during rotation), so independently recheck actual clearance here.
        tip_z4, clear4, _, _, _, _ = _cup_exit_state(
            robot, straw_id, arm, cup_mouth_z, hole_axis0)
        print("[STEP4] After straightening lower end z=%.4f cup rim %.4f => clearance %+.1fmm" % (tip_z4, cup_mouth_z, 1000 * clear4))
        if clear4 < 0.0:
            print("[ERROR] step 4 straightening sent straw lower end back into paper cup(clearance %+.1fmm)."
                  "Step 3.5 provided insufficient clearance: increase PICK_UP_DISTANCE(currently %.0fmm), "
                  "do not work around this by reducing straightening angle(insertion requires an upright rod)."
                  % (1000 * clear4, 1000 * PICK_UP_DISTANCE))
            _release_straw_air_pin("Step4 fell back into paper cup")
            robot.detach_obj()
            return False
    else:
        print("[STEP4] Tilt only %.2f°; skipping straightening to avoid unnecessary wrist rotation" % tilt_now)

    # ------------------------------------------------------------------
    # Step 5: move above opening by translation, inheriting orientation from the **actual achieved pose**.
    # ------------------------------------------------------------------
    hole_centre, hole_axis = _hole_frame(robot, milk_id)
    T_now, _, tip_now, _, local_low = _straw_state(robot, straw_id)
    R_now = T_now[:3, :3]
    # Target endpoint is 18~30 mm above hole center. PICK_UP_DISTANCE=135 mm should already lift it above the center;
    # construct negative-depth candidates by retreat along hole axis. If the straw remains too low, lift was insufficient or
    # gripper slip excessive; report measured shortfall for diagnosis.
    # **Retreat alone is insufficient**: identical candidate orientations all fail if unreachable. Add **yaw about the hole axis**;
    # cylindrical symmetry preserves insertion geometry while changing gripper orientation, expanding IK. See PRE_INSERT_YAWS.
    pre_cands = np.stack([
        _straw_pose_at_depth(-(PRE_INSERT_GAP + extra),
                             _rot_about_axis(hole_axis, yaw) @ R_now,
                             local_low, hole_centre, hole_axis)
        for extra in PRE_INSERT_EXTRA
        for yaw in PRE_INSERT_YAWS
    ], axis=0)
    target_z_min = hole_centre[2] - PRE_INSERT_GAP - max(PRE_INSERT_EXTRA)
    tip_z_now = float(tip_now[2])
    if tip_z_now < target_z_min:
        print("[STEP5-WARNING] Current lower end z=%.4f is below minimum target z=%.4f (difference %.1fmm), "
              "which may cause move failure. Check whether PICK_UP_DISTANCE=%.0fmm is sufficient or in-gripper slip is excessive."
              % (tip_z_now, target_z_min, 1000 * (target_z_min - tip_z_now), 1000 * PICK_UP_DISTANCE))
    print("[STEP5] %d pre-insertion candidates(target: lower end above hole center z=%.4f by %.0f~%.0fmm × yaw %s° about hole axis)"
          % (len(pre_cands), hole_centre[2], 1000 * PRE_INSERT_GAP,
             1000 * (PRE_INSERT_GAP + PRE_INSERT_EXTRA[-1]),
             [round(float(np.degrees(y))) for y in PRE_INSERT_YAWS]))
    if not robot.move_actor_to_target_pose(
            actor=straw_id, target_pose=pre_cands, arm=arm,
            move_config=MoveConfig(
                unmovable_objects=[milk_id],
                action_description={"zh": "把吸管移到奶茶杯盖的孔口上方",
                                    "en": "moves the straw above the straw hole on the lid"},
                extra_params={"skip_check": True})):
        # Use the large-yaw IK fallback only after the complete small-yaw batch fails.
        pre_cands_fb = np.stack([
            _straw_pose_at_depth(-(PRE_INSERT_GAP + extra),
                                 _rot_about_axis(hole_axis, yaw) @ R_now,
                                 local_low, hole_centre, hole_axis)
            for extra in PRE_INSERT_EXTRA
            for yaw in PRE_INSERT_YAWS_FALLBACK
        ], axis=0)
        print("[STEP5] Small-yaw batch(±%s°) rejected entirely by IK; falling back to %d wide-yaw candidates(±%.0f°)"
              % ([round(float(np.degrees(y))) for y in PRE_INSERT_YAWS],
                 len(pre_cands_fb),
                 round(abs(float(np.degrees(PRE_INSERT_YAWS_FALLBACK[0]))))))
        if not robot.move_actor_to_target_pose(
                actor=straw_id, target_pose=pre_cands_fb, arm=arm,
                move_config=MoveConfig(
                    unmovable_objects=[milk_id],
                    action_description={"zh": "把吸管移到奶茶杯盖的孔口上方(大偏航兜底)",
                                        "en": "moves the straw above the straw hole on the lid (wide-yaw fallback)"},
                    extra_params={"skip_check": True})):
            print("[ERROR] step 5 move above opening failed(including wide-yaw fallback)")
            _release_straw_air_pin("Step5 pre-insertion failed")
            return False
        print("[STEP5] Wide-yaw fallback recovered: subsequent alignment is stricter and Step 6 blocking may increase; this is expected")
    robot.wait(0.3)
    _report(robot, straw_id, milk_id, "pre-insertion")
    _slip_check(robot, straw_id, arm, grip_ref, "STEP5 after pre-insertion")

    # Remove milk-tea cup from obstacles before insertion: curobo sphere approximation fills the entire hole.
    # After removal pass **one candidate at a time**; batches reload obstacles between candidates, making later ones rub past them.
    robot.remove_objs_from_obstacle(_collision_prim_paths(robot, milk_id))
    print("[STEP5] Removed milk-tea-cup collider from obstacle world(required before insertion)")

    # STEP 5.5: descend axially to NEAR_HOLE_AIR_GAP before alignment.
    # Measure the actual endpoint before the next correction.
    stage1_depth = -NEAR_HOLE_AIR_GAP
    if not _move_axially_to_depth(
            robot, straw_id, milk_id, arm, stage1_depth, "STEP5.5",
            {"zh": "仅沿孔轴把吸管降到孔口上方五毫米",
             "en": "lowers the straw axially to five millimetres above the lid hole"}):
        print("[ERROR] Step 5.5 axial-only descent failed")
        _release_straw_air_pin("Step5.5 axial descent failed")
        return False
    d_near, r_near, tilt_near = _report(robot, straw_id, milk_id, "after near-hole descent")
    _slip_check(robot, straw_id, arm, grip_ref, "STEP5.5 after near-hole descent")
    if d_near > -NEAR_HOLE_AIR_MIN_GAP:
        print("[ERROR] Step 5.5 achieved depth %+.1fmm, less than %.1fmm above opening; refusing lateral motion at opening."
              % (1000 * d_near, 1000 * NEAR_HOLE_AIR_MIN_GAP))
        _release_straw_air_pin("Step5.5 insufficient safe clearance")
        return False

    # Keep the gripper command unchanged during near-hole descent and alignment
    # to avoid disturbing the achieved straw pose.
    print("[HOLD] No more gripper commands after Step5.5; maintain current real grip and proceed directly to Step5.6 final airborne centering")

    # STEP 5.6: combine near-hole centering and straightening in one motion
    # while preserving the minimum air gap. Check the achieved endpoint
    # before the penetration path.
    print("[STEP5.6] One combined precise near-hole centering motion: initial radial %.2fmm, tilt %.2f°; "
          "target radial <=%.2fmm, tilt <=%.1f°"
          % (1000 * r_near, tilt_near, 1000 * NEAR_HOLE_ALIGN_RADIAL_TARGET,
             NEAR_HOLE_ALIGN_TILT_DEG_MAX))
    if not _near_hole_align_once(robot, straw_id, milk_id, arm, "NEAR-HOLE"):
        print("[ERROR] Step 5.6 combined near-hole correction planning failed")
        _release_straw_air_pin("Step5.6 combined centering failed")
        return False
    d_aligned, r_aligned, tilt_aligned = _report(robot, straw_id, milk_id, "after precise near-hole centering")
    _slip_check(robot, straw_id, arm, grip_ref, "STEP5.6 after precise near-hole centering")
    near_hole_entry_ok = (
        d_aligned <= -NEAR_HOLE_AIR_MIN_GAP and
        r_aligned <= NEAR_HOLE_ENTRY_RADIAL_MAX and
        tilt_aligned <= NEAR_HOLE_ENTRY_TILT_DEG_MAX
    )
    print("[STEP5.6] First combined achieved state: depth %+.1fmm(<=-%.1f), radial %.2fmm(<=%.2f), "
          "tilt %.2f°(<=%.1f) => %s"
          % (1000 * d_aligned, 1000 * NEAR_HOLE_AIR_MIN_GAP,
             1000 * r_aligned, 1000 * NEAR_HOLE_ENTRY_RADIAL_MAX,
             tilt_aligned, NEAR_HOLE_ENTRY_TILT_DEG_MAX,
             "proceed to Step6.1" if near_hole_entry_ok else "stop"))
    if not near_hole_entry_ok:
        # Use the bounded retry to correct the live signed residual while
        # preserving the minimum air gap.
        feedback_ok = False
        for retry in range(1, NEAR_HOLE_FEEDBACK_MAX_RETRY + 1):
            print("[STEP5.6] First achieved state failed gate; performing single-step measured feedback centering %d/%d"
                  % (retry, NEAR_HOLE_FEEDBACK_MAX_RETRY))
            if not _near_hole_feedback_retry(robot, straw_id, milk_id, arm):
                break
            d_aligned, r_aligned, tilt_aligned = _report(
                robot, straw_id, milk_id, "after near-hole feedback centering")
            _slip_check(robot, straw_id, arm, grip_ref, "STEP5.6 after feedback centering")
            feedback_ok = (
                d_aligned <= -NEAR_HOLE_AIR_MIN_GAP and
                r_aligned <= NEAR_HOLE_ENTRY_RADIAL_MAX and
                tilt_aligned <= NEAR_HOLE_ENTRY_TILT_DEG_MAX
            )
            print("[STEP5.6] Feedback achieved state: depth %+.1fmm, radial %.2fmm, tilt %.2f° => %s"
                  % (1000 * d_aligned, 1000 * r_aligned, tilt_aligned,
                     "proceed to Step6.1" if feedback_ok else "stop"))
            if feedback_ok:
                break
        if not feedback_ok:
            print("[ERROR] Step 5.6 still lacks conservative penetration margin after one measured feedback correction; stopping, never move laterally at opening.")
            _release_straw_air_pin("Step5.6 excessive centering residual")
            return False
    print("[STEP5.6] Near-hole result passes entry gate; proceeding to short real penetration segment")

    # STEP 6: establish hole-wall capture, advance along the captured straw
    # axis, then release after checking the achieved insertion state.
    robot.remove_objs_from_obstacle(_collision_prim_paths(robot, milk_id))

    # Rely on actual PhysX contact from here. There is no pin-release state transition;
    # reread actual straw state so short-penetration command always starts from the real grip relation.
    if not _release_straw_air_pin("Step6.1 before first real downward insertion"):
        robot.detach_obj()
        return False
    d_free, r_free, tilt_free = _report(robot, straw_id, milk_id, "before short penetration with real grip")
    # This gate checks safe entry into real contact, not the airborne commanded target. Retain Step5.6's
    # conservative entry gate; subsequent real penetration state is the sole verification.
    free_before_entry_ok = (
        d_free <= -NEAR_HOLE_AIR_MIN_GAP and
        r_free <= NEAR_HOLE_ENTRY_RADIAL_MAX and
        tilt_free <= NEAR_HOLE_ENTRY_TILT_DEG_MAX
    )
    print("[HOLD] Remeasure before short penetration: depth %+.1fmm(<=-%.1f), radial %.2fmm(<=%.2f), "
          "tilt %.2f°(<=%.1f) => %s"
          % (1000 * d_free, 1000 * NEAR_HOLE_AIR_MIN_GAP,
             1000 * r_free, 1000 * NEAR_HOLE_ENTRY_RADIAL_MAX,
             tilt_free, NEAR_HOLE_ENTRY_TILT_DEG_MAX,
             "allow short real penetration" if free_before_entry_ok else "stop"))
    if not free_before_entry_ok:
        print("[ERROR] Precise near-hole pose not maintained under real grip; "
              "no further lateral movement before contact, to avoid false success through interpenetration.")
        robot.detach_obj()
        return False

    # STEP 6.1: follow a straw-entity trajectory constrained by its lower
    # endpoint. Derive the gripper path through the live object-to-EE relation.
    # Start from the measured free-space pose and preserve its orientation
    # through initial capture; depths are absolute relative to the lid plane.
    entry_cap, jaw_L = _max_command_depth(robot, straw_id, arm)
    capture_target = min(ENTRY_CAPTURE_INITIAL_DEPTH, entry_cap)
    if capture_target < ENTRY_MIN_DEPTH:
        print("[ERROR] Step 6.1 gripper-center safe limit only %.1fmm(<penetration minimum %.1fmm; measured L=%.1fmm), "
              "refusing to bring gripper near lid."
              % (1000 * entry_cap, 1000 * ENTRY_MIN_DEPTH, 1000 * jaw_L))
        robot.detach_obj()
        return False
    _slip_check(robot, straw_id, arm, grip_ref, "STEP6.1 before short lower-tip trajectory penetration")
    print("[STEP6.1] Real penetration along lower-end object trajectory: achieved %+.1fmm / %.2fmm / %.2f° -> +%.1fmm; "
          "maintain current orientation to establish real capture inside hole first."
          % (1000 * d_free, 1000 * r_free, tilt_free,
             1000 * capture_target))
    if not _follow_straw_lower_tip_path(
            robot, straw_id, milk_id, arm, capture_target, "CAPTURE",
            {"zh": "沿吸管下端点轨迹短距离穿过杯盖孔板",
             "en": "follows the straw lower-tip path just through the lid hole"},
            align_axis=False,
            segment_count=4):
        print("[ERROR] Step 6.1 short lower-tip trajectory penetration planning/execution failed")
        robot.detach_obj()
        return False
    d_entry, r_entry, tilt_entry = _report(robot, straw_id, milk_id, "after short lower-tip trajectory penetration")
    _slip_check(robot, straw_id, arm, grip_ref, "STEP6.1 after short lower-tip trajectory penetration")
    entry_ok = (
        d_entry >= ENTRY_MIN_DEPTH and
        r_entry <= ENTRY_BORE_RADIAL_MAX and
        tilt_entry <= ENTRY_TILT_DEG_MAX
    )
    print("[ENTRY-CHECK] Short penetration: depth=%+.1fmm(>=%.1f), opening radial=%.2fmm(<=%.2f), "
          "tilt=%.2f°(<=%.1f) => %s"
          % (1000 * d_entry, 1000 * ENTRY_MIN_DEPTH,
             1000 * r_entry, 1000 * ENTRY_BORE_RADIAL_MAX,
             tilt_entry, ENTRY_TILT_DEG_MAX,
             "allow Step6.2 deep push" if entry_ok else "stop"))
    if not entry_ok:
        print("[ERROR] Step 6.1 failed to maintain real opening capture suitable for release after penetration; "
              "refusing deep push.")
        robot.detach_obj()
        return False

    # ------------------------------------------------------------------
    # Step 6.2: after capture, push deeper along the straw axis; do not rely on gravity after release.
    # ------------------------------------------------------------------
    # Do not pull the straw laterally toward hole center now: first stage already captured it. Use the measured opening
    # intersection as aim, advancing every waypoint only along the current straw axis, avoiding lever-induced
    # sweeps across the rim. Object follow_trajectory still tracks the actual straw endpoint rather than gripper.
    d_capture, r_capture, tilt_capture, axis_capture, tip_capture, _, hole_capture, hole_axis_capture = (
        _measure_engagement(robot, straw_id, milk_id))
    cos_capture = float(np.dot(axis_capture, hole_axis_capture))
    if abs(cos_capture) < 1e-3:
        print("[ERROR] Step 6.2 straw axis nearly parallel to lid; refusing deep push.")
        robot.detach_obj()
        return False
    capture_plane_hit = tip_capture + axis_capture * (d_capture / cos_capture)
    deep_cap, jaw_L_deep = _max_command_depth(robot, straw_id, arm)
    deep_target = min(ENTRY_CAPTURE_DEPTH, deep_cap)
    if deep_target <= d_capture + 0.002:
        print("[ERROR] Step 6.2 gripper-center safe limit %.1fmm prevents safe deep push after captured depth %+.1fmm."
              % (1000 * deep_cap, 1000 * d_capture))
        robot.detach_obj()
        return False
    _slip_check(robot, straw_id, arm, grip_ref, "STEP6.2 before in-hole deep push")
    print("[STEP6.2] Deep push along real straw axis after capture: %+.1fmm / %.2fmm / %.2f° -> +%.1fmm; "
          "no release, reliance on gravity, or lateral return toward hole."
          % (1000 * d_capture, 1000 * r_capture, tilt_capture, 1000 * deep_target))
    if not _follow_straw_lower_tip_path(
            robot, straw_id, milk_id, arm, deep_target, "DEEP-PUSH",
            {"zh": "在已插入孔内后沿吸管轴线深推",
             "en": "deep-pushes the captured straw along its own axis into the cup"},
            align_axis=False,
            aim_centre=capture_plane_hit,
            segment_count=4):
        print("[ERROR] Step 6.2 in-hole axial deep-push planning/execution failed")
        robot.detach_obj()
        return False
    d_entry, r_entry, tilt_entry = _report(robot, straw_id, milk_id, "after in-hole deep push")
    _slip_check(robot, straw_id, arm, grip_ref, "STEP6.2 after in-hole deep push")
    entry_ok = (
        d_entry >= ENTRY_MIN_DEPTH and
        r_entry <= ENTRY_BORE_RADIAL_MAX and
        tilt_entry <= ENTRY_TILT_DEG_MAX
    )
    print("[ENTRY-CHECK] After deep push: depth=%+.1fmm(>=%.1f), opening radial=%.2fmm(<=%.2f), "
          "tilt=%.2f°(<=%.1f) => %s"
          % (1000 * d_entry, 1000 * ENTRY_MIN_DEPTH,
             1000 * r_entry, 1000 * ENTRY_BORE_RADIAL_MAX,
             tilt_entry, ENTRY_TILT_DEG_MAX,
             "release immediately" if entry_ok else "stop"))
    if not entry_ok:
        print("[ERROR] Step 6.2 failed to maintain real opening capture suitable for release after deep push.")
        robot.detach_obj()
        return False

    # ------------------------------------------------------------------
    # Step 7: physically release at deep-insertion depth, retreat first, and allow full opening only far away.
    # ------------------------------------------------------------------
    # Step 6.2 already completed deep push along the first segment's measured axis; no further lateral motion or extra insertion here,
    # to avoid forming a lever inside the hole.
    # detach removes only cuRobo attachment and preserves real finger contact, exactly the state required here.
    if not robot.detach_obj():
        print("[ERROR] Step 7 cannot remove cuRobo attachment; refusing release.")
        return False
    robot.attached_obj_id = None

    # Use the native full-open command after capture, then withdraw the
    # empty hand radially if its linkage still supports the straw.
    print("[STEP7] In-hole capture confirmed; "
          "use native opening directly, then withdraw empty gripper away from straw axis")
    robot.open_gripper(arm=arm)
    robot.detach_obj()
    robot.attached_obj_id = None
    robot.wait(RELEASE_RADIAL_WITHDRAW_WAIT_S)
    d_partial, r_partial, tilt_partial = _report(robot, straw_id, milk_id, "after native opening")

    if r_partial >= ACCEPT_BORE_RADIAL_MAX or tilt_partial >= ACCEPT_TILT_DEG_MAX:
        print("[ERROR] Step 7 opening state invalid after native gripper opening.")
        return False

    # Clear residual open-finger contact before resetting the arm so the
    # reset motion cannot lever the straw out of the hole.
    if d_partial < RELEASE_FALL_DEPTH_MIN:
        print("[STEP7] Depth only +%.1fmm after native opening; first clear residual gripper contact radially"
              % (1000 * d_partial))
        if not _withdraw_open_gripper_radially(robot, straw_id, milk_id, arm):
            print("[ERROR] Step 7 empty-gripper radial withdrawal failed to preserve opening state.")
            return False

    # The gripper is now either clear already or has been moved clear radially;
    # give gravity time to finish the fall before the normal reset motion.
    robot.wait(RELEASE_SETTLE_WAIT_S)
    robot.wait(SETTLE_WAIT_S)
    d_released, r_released, tilt_released = _report(robot, straw_id, milk_id, "free settling after release in place")
    # Only nonblocking diagnosis here; remeasure true success after reset to retain one authoritative check.
    released_depth_ok = ACCEPT_DEPTH_MIN <= d_released <= ACCEPT_DEPTH_MAX
    released_bore_ok = r_released < ACCEPT_BORE_RADIAL_MAX
    released_tilt_ok = tilt_released < ACCEPT_TILT_DEG_MAX
    print("[RELEASE-CHECK] Free settling: depth=%+.1fmm(accepted %.1f~%.1f), opening radial=%.2fmm(<%.2f), "
          "tilt=%.2f°(<%.1f) => %s; final decision uses remeasurement after withdrawal"
          % (1000 * d_released, 1000 * ACCEPT_DEPTH_MIN, 1000 * ACCEPT_DEPTH_MAX,
             1000 * r_released, 1000 * ACCEPT_BORE_RADIAL_MAX,
             tilt_released, ACCEPT_TILT_DEG_MAX,
             "provisionally passed" if (released_depth_ok and released_bore_ok and released_tilt_ok)
             else "not yet passed"))

    # Reset the selected arm explicitly after the gripper clears the straw.
    robot.back_to_origin(arm=arm)
    robot.wait(0.5)

    # ------------------------------------------------------------------
    # Step 9: final check, independently remeasured after reset; do not reuse Step 7 readings.
    # ------------------------------------------------------------------
    depth, bore_r, tilt = _measure_engagement(robot, straw_id, milk_id)[:3]
    milk_pos_end, _ = robot.get_object_pose(milk_id)
    milk_moved = float(np.linalg.norm(np.asarray(milk_pos_end) - np.asarray(milk_pos)))
    paper_tilt, paper_moved = _paper_cup_state(robot, paper_id, paper_T)

    gate_depth = ACCEPT_DEPTH_MIN <= depth <= ACCEPT_DEPTH_MAX
    gate_bore = bore_r < ACCEPT_BORE_RADIAL_MAX
    gate_tilt = tilt < ACCEPT_TILT_DEG_MAX
    gate_cup = milk_moved < 0.02
    # Check the source paper cup remains upright and in place after extraction.
    gate_paper_tilt = paper_tilt < ACCEPT_PAPER_TILT_DEG_MAX
    gate_paper_moved = paper_moved < ACCEPT_PAPER_MOVED_MAX
    passed = bool(gate_depth and gate_bore and gate_tilt and gate_cup
                  and gate_paper_tilt and gate_paper_moved)

    print("[FINAL-CHECK] Depth=%+.4f m(accepted %.3f~%.3f) -> %s" % (
        depth, ACCEPT_DEPTH_MIN, ACCEPT_DEPTH_MAX, gate_depth))
    print("[FINAL-CHECK] Opening radial=%.4f m(accepted <%.4f, physical clearance %.4f) -> %s" % (
        bore_r, ACCEPT_BORE_RADIAL_MAX, HOLE_R_IN - STRAW_R, gate_bore))
    print("[FINAL-CHECK] Tilt=%.2f°(accepted <%.1f) -> %s" % (tilt, ACCEPT_TILT_DEG_MAX, gate_tilt))
    print("[FINAL-CHECK] Milk-tea-cup displacement=%.4f m(accepted <0.020) -> %s" % (milk_moved, gate_cup))
    print("[FINAL-CHECK] Paper-cup tilt=%.2f°(accepted <%.1f) -> %s" % (
        paper_tilt, ACCEPT_PAPER_TILT_DEG_MAX, gate_paper_tilt))
    print("[FINAL-CHECK] Paper-cup displacement=%.4f m(accepted <%.3f) -> %s" % (
        paper_moved, ACCEPT_PAPER_MOVED_MAX, gate_paper_moved))
    print("[FINAL-CHECK] Insertion ratio=%.0f%% (%.1fmm / total length %.1fmm)" % (
        100.0 * depth / STRAW_LEN, 1000 * depth, 1000 * STRAW_LEN))
    if not gate_paper_tilt:
        print("[FINAL-CHECK] ⚠ **Paper cup tipped**(%.1f° >= %.1f°). The straw was extracted from this cup; "
              "tipping it means failure: inspect paper-cup displacement in [STEP3.x] above; "
              "tipping usually occurs during extraction." % (paper_tilt, ACCEPT_PAPER_TILT_DEG_MAX))
    print("[FINAL-CHECK] => %s" % ("PASS" if passed else "FAIL"))
    return passed
