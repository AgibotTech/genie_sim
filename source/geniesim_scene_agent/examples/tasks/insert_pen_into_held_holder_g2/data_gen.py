# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np


# One hand holds the pen holder while the other picks and inserts a pen, then
# returns the holder to the table. Use asset geometry and live object poses for
# grasp filters, lift clearance and insertion targets. Expand obstacle exclusions
# to collider-mesh paths. Validate the grasp, achieved alignment and final
# containment at each stage.
PEN_LENGTHS = {
    "benchmark_pen_000": 0.145,
    "benchmark_pen_001": 0.145,
    "benchmark_pen_002": 0.145,
    "benchmark_pen_003": 0.145,
    "benchmark_pen_005": 0.145,
    "benchmark_pen_006": 0.143,
}

PEN_USDS = {
    "benchmark_pen_000",
    "benchmark_pen_001",
    "benchmark_pen_002",
    "benchmark_pen_003",
    "benchmark_pen_005",
    "benchmark_pen_006",
}

# How far (m) the holding arm lifts the cup off the desktop. Just enough to read as
# "lifted off the table" while keeping the held cup and the other gripper in the head
# cameras and within a comfortable reach for the inserting arm.
# This value now comes from CUP_GEOM[...]["lift"] per asset (tall holders need less lift or the rim exceeds reach).
# This is the fallback for a failed lookup; STEP 1 uses the local cup_lift variable.
CUP_LIFT = 0.05

# Measured geometry for the supported pen holder. Keep these values aligned
# with the asset and its scale; insertion uses the inner radius and rim height.
CUP_GEOM = {
    "benchmark_stationery_021": {
        "scale": 1.0,
        "h": 0.0799, "wall_r": 0.0302, "mouth_r": 0.029, "bore_depth": 0.0775,
        "lift": 0.05, "pre_insert": 0.06,
        # Retain the measured original ladder unchanged: it is this asset's only locally collected configuration (5/8).
        # Entries with tip_depth <=0 release above the rim and let the pen fall in. They do fall out more often,
        # but **the checker is now accurate** (see pen_in_cup): falling out is failure, not accepted data.
        # The framework settles and checks again after a failed first check; genuinely falling in still succeeds. Retain the candidates,
        # changing only the checker so any success-rate change can be attributed to one variable.
        "tip_depths": (0.04, 0.02, 0.005, -0.005, -0.02, 0.05),
    },
}

# Lookup fallback (normally unreachable: CUP_USDS is CUP_GEOM's key set). Use the pool's most conservative parameters;
# prefer strict checking and shallow insertion over silently giving an unregistered asset a large holder's deep-insertion parameters.
CUP_GEOM_FALLBACK = CUP_GEOM["benchmark_stationery_021"]

# The container pool. MUST stay in sync with CUP_CANDIDATES in LLM_RESULT.py. Matched by id
# rather than by a substring of the usd path, because these containers are "stationery" assets
# rather than "cup" ones and their keywords include "pen_holder" -- see the lookup below.
CUP_USDS = set(CUP_GEOM)

# Filter pen grasps to a middle band of the shaft. Keep all candidates within
# the selected band so IK can choose a reachable wrist configuration.
PEN_MID_BAND_LADDER = (0.025, 0.025, 0.030, 0.040)
# Each grasp band has a pick_up_distance range in metres. Select lift and
# approach together to clear the table without excessive wrist rotation.
PEN_PICKUP_LADDER = ([0.08, 0.12], [0.05, 0.09], [0.03, 0.07], [0.015, 0.06])
# Skip hard filtering this rung below this candidate count to avoid a needless round trip.
PEN_MID_MIN_KEEP = 20
# None keeps every in-band grasp candidate; a numeric value truncates candidates
# by distance from the pen center.
PEN_MID_KEEP_CAP = None
# Trust grasp diagnostics only when reconstructed wrist-to-TCP distance lies in this interval (m).
# Do not hardcode 0.156. Extrapolating the 156 mm inferred from state.json produced e8's
# reported axial -41.3 mm and perpendicular 136.4 mm, versus post-analysis +1.9 mm and 0.7 mm in the same episode.
# A 136 mm perpendicular error means the extrapolated TCP misses the pen axis: `robot.get_ee_pose()` already returns
# the **TCP**, whereas state.json's `ee` is the wrist. Applying 156 mm to both adds a spurious offset.
# Reconstruct analytically each time (before STEP 2b); use this interval only to judge reconstruction reliability.
EE_TCP_LEN_RANGE = (0.00, 0.25)


def _make_pen_mid_grasp_filter(band):
    """Create a grasp_pose_filter retaining only mid-shaft candidates; band=None disables filtering (framework default).

    End grasps leave a long overhang during STEP 2b reorientation and STEP 3 insertion. Contact with the rim
    rotates that end about the TCP, matching measured pick-lift failures of `pos_diff 0.142 / angle_diff 178.9 degrees`
    (half a turn inside the gripper). Middle grasps balance both ends and minimize torque, also satisfying the
    requirement that end grasps make holder insertion inconvenient.

    **Compute the long axis from candidate point clouds using PCA**, without reading obj_pose rotation or assuming
    an asset-local axis. Candidates are already world coordinates (obj_pose @ canonical); local long axes vary
    by asset (all six current pens happen to use +Y, but do not hardcode it). Offline checks of pen_000/001/006
    at three world yaws gave principal/secondary ratios 8.4~11.5, clean elongated-object directions, and projected
    extent +/-69 mm, matching half-length and the actual longitudinal span in pkl.

    Use obj_pose translation (pen origin), not point-cloud centroid: sampling at the two ends may be asymmetric
    (pen_006 median projection +20 mm), and centroid-based filtering would shift the middle.

    If fewer than PEN_MID_MIN_KEEP remain, **return candidates unchanged** and let the outer bandwidth ladder
    decide relaxation. This also avoids an empty set, which makes grasp.py:200 abort the whole pick stage.
    """

    def _filter(grasp_poses, obj_pose, grasp_widths):
        gp = np.asarray(grasp_poses, dtype=float)
        gw = np.asarray(grasp_widths, dtype=float)
        if band is None or gp.shape[0] < 3:
            return gp, gw
        pts = gp[:, :3, 3]
        centred = pts - pts.mean(axis=0)
        # Principal direction = covariance's largest eigenvector (equivalent to SVD's first right singular vector).
        _, sing, vt = np.linalg.svd(centred, full_matrices=False)
        long_axis = vt[0]
        ratio = float(sing[0] / max(sing[1], 1e-9))
        pen_centre = np.asarray(obj_pose, dtype=float)[:3, 3]
        proj = (pts - pen_centre[None, :]) @ long_axis
        mask = np.abs(proj) <= band
        keep = int(np.count_nonzero(mask))
        if keep == 0:
            # Only the **empty set** needs a fallback; grasp.py:200 aborts the whole pick stage on empty candidates.
            # Never relax to the pen ends: retain a small set with minimum absolute axial offset, even outside the band.
            idx = np.argsort(np.abs(proj))[:PEN_MID_MIN_KEEP]
            print(f"[DIAG_PEN_GRASP] No candidates within band ±{band * 1000:.0f}mm, "
                  f"falling back to the {len(idx)} closest to the center "
                  f"(|distance from pen center| max {np.abs(proj[idx]).max() * 1000:.0f}mm)")
            return gp[idx], gw[idx]
        print(f"[DIAG_PEN_GRASP] Pen-midsection filter: PCA major/minor={ratio:.1f} "
              f"projection range=[{proj.min() * 1000:+.0f},{proj.max() * 1000:+.0f}]mm "
              f"band ±{band * 1000:.0f}mm -> kept {keep}/{len(proj)} "
              f"(|distance from pen center| median {np.median(np.abs(proj[mask])) * 1000:.0f}mm)")
        # Return all in-band candidates so IK can use reachable candidates near the band edge.
        idx = np.where(mask)[0]
        if PEN_MID_KEEP_CAP is not None:
            idx = idx[np.argsort(np.abs(proj[idx]))][:PEN_MID_KEEP_CAP]
            print(f"[DIAG_PEN_GRASP] Truncated to {len(idx)} closest to center: |distance from pen center| "
                  f"median {np.median(np.abs(proj[idx])) * 1000:.0f}mm "
                  f"max {np.abs(proj[idx]).max() * 1000:.0f}mm")
        return gp[idx], gw[idx]

    return _filter


def execute_task(robot: RobotAPI):
    objects_list = {
        "table_000": {"has_interactions": False, "object_type": "table"},
    }
    # Register all three holder assets in the pool (any may be sampled), each with passive.grasp.default.
    for cid in CUP_USDS:
        objects_list[cid] = {"has_interactions": True, "object_type": "pen_holder", "interaction_type": "container"}
    for pid in PEN_USDS:
        objects_list[pid] = {"has_interactions": True, "object_type": "pen"}

    scene_info = robot.scene_info

    description_json = {
        "task_name": "hold the cup with one hand, insert a pen into it with the other, then set the cup back on the table",
        "english_task_name": "hold the cup with one hand, insert a pen into it with the other, then set the cup back on the table",
        "init_scene_text": "The robot is in front of a table with a cup near the middle and a single pen beside it",
    }
    robot.client.set_task_task_basic_info(description_json)

    # --- locate table ---
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        if obj_info.get("usd", "") == "table_000" or "table" in obj_info.get("keywords", []):
            table_id = obj_id
            break
    if not table_id:
        print("Table not found")
        return False

    table_pos, _ = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    if table_pos is None or table_size is None:
        print("Cannot get table pose/size")
        return False
    table_top_z = table_pos[2] + table_size[2] / 2.0

    # --- locate cup (the held container) and pen ---
    cup_id = None
    cup_usd = ""          # Record the holder USD id to retrieve this episode's geometry from CUP_GEOM below.
    pen_id = None
    pen_usd = ""          # Record the pen USD id so STEP 2b can retrieve its actual length from PEN_LENGTHS.
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        kw = obj_info.get("keywords", [])
        if obj_id == table_id:
            continue
        # The container is matched FIRST and by its own id, because every holder in the pool
        # carries the keyword "pen_holder" -- a substring test for "pen" would otherwise claim
        # it as the pen. The pen is matched only against PEN_USDS.
        if usd in CUP_USDS or "pen_holder" in kw or "target_container" in kw:
            cup_id = obj_id
            cup_usd = usd
        elif usd in PEN_USDS:
            pen_id = obj_id
            pen_usd = usd

    if cup_id is None or pen_id is None:
        print("Required objects not found (need 1 cup, 1 pen)")
        return False

    # This episode's holder determines its measured geometry. Inner radius, cavity depth, lift, insertion-depth ladder and pre_insert
    # all come from here; do not hardcode holder dimensions downstream.
    cup_geom = CUP_GEOM.get(cup_usd, CUP_GEOM_FALLBACK)
    cup_mouth_r = float(cup_geom["mouth_r"])
    cup_bore_depth = float(cup_geom["bore_depth"])
    cup_wall_r = float(cup_geom["wall_r"])
    cup_lift = float(cup_geom["lift"])
    cup_scale = float(cup_geom["scale"])
    if cup_usd not in CUP_GEOM:
        print(f"[WARN] Cup {cup_usd!r} is absent from CUP_GEOM; using the most conservative parameters — "
              f"check that LLM_RESULT.py CUP_CANDIDATES and CUP_GEOM are synchronized")

    # Compensate here because the framework does not scale client-side dimensions.
    # `get_object_size_in_world_frame` uses ASSETS_INDEX bbox **without the scene's
    # scale** (robot_api.py:3390 hardcodes object_info["scale"]=1.0). At 0.48/0.40 scale,
    # it still reports original dimensions, directly affecting rim_z, release height, and the containment
    # checker. Without compensation, 010 appears 18 cm tall (actually 8.6 cm), moving the rim by 4.7 cm and invalidating the checker.
    # Route every holder-size query through this function, never directly through get_object_size_in_world_frame(cup_id).
    #
    # Validate that scaled z size approximately equals CUP_GEOM h; WARN if the scale table and scene disagree.
    def _cup_world_size():
        raw = robot.get_object_size_in_world_frame(cup_id)
        if raw is None:
            return None
        return [float(v) * cup_scale for v in raw]

    _cs_chk = _cup_world_size()
    if _cs_chk is not None:
        _dz = abs(float(_cs_chk[2]) - float(cup_geom["h"]))
        if _dz > 0.008:
            print(f"[WARN][DIAG_CUP] Scale-compensated cup height {_cs_chk[2]:.4f} differs from CUP_GEOM['h']="
                  f"{cup_geom['h']:.4f} by {_dz * 1000:.1f}mm — scene asset scaling "
                  f"may differ from scale here; all subsequent height-related quantities will be wrong")
    print(f"[DIAG_CUP] Current cup={cup_usd} scale={cup_scale} mouth inner radius={cup_mouth_r:.4f} "
          f"bore depth={cup_bore_depth:.4f} wall outer radius={cup_wall_r:.4f} lift={cup_lift:.3f} "
          f"pre_insert={cup_geom['pre_insert']:.3f} insertion-depth candidates={list(cup_geom['tip_depths'])} "
          f"compensated world dimensions={None if _cs_chk is None else np.round(_cs_chk, 4).tolist()}")

    # --- resolve which arm HOLDS (lifts) the cup and which arm INSERTS the pen ---
    # robot at -x facing +x => +y is the robot's LEFT. The cup sits just to the HOLD
    # side of centre; the pen sits on the OPPOSITE (place) side. The holding arm is the
    # cup's side; the inserting arm is the pen's side (no cross-body).
    cup_pos0, cup_rot0 = robot.get_object_pose(cup_id)
    cup_pos0 = np.asarray(cup_pos0)
    hold_arm = "left" if cup_pos0[1] >= 0 else "right"
    place_arm = "right" if hold_arm == "left" else "left"
    hold_sign = 1.0 if hold_arm == "left" else -1.0
    print(f"Cup on {'left' if hold_arm == 'left' else 'right'}; hold arm = {hold_arm}, insert arm = {place_arm}")

    # ============================================================== STEP 1
    # Hold arm picks the cup UP off the table (CUP_LIFT) and holds it in the air.
    # passive_primitive="default" uses the cup's annotated grasp point.
    #
    # Grasp the cup on the HOLD-side edge (the far edge from the other hand) so the
    # opposite half of the cup mouth is left clear for the pen insert. The planner calls
    # grasp_pose_filter(grasp_poses, obj_pose, grasp_widths) with the candidate grasps
    # already in the WORLD frame, so we keep only grasps whose contact point sits on the
    # hold side of the cup centre (+y for the left hand, -y for the right). A fallback
    # returns all candidates if the side bias would prune everything, so the grasp never
    # fails purely from this preference.
    # In addition to the holder-side preference, require a **sufficiently horizontal approach axis**. G2's wrist camera
    # protrudes 6.1 cm perpendicular to the gripper axis from its base (robot_fix.usda:2885). The holder grasp persists
    # through the **entire episode**: an angled/downward grasp puts the camera in the other arm's workspace throughout.
    # Constraining the holding arm's pick is appropriate here because its persistent grasp
    # defines an obstacle for the other arm's manipulation.
    #
    # Measured successful holding elevations span 8.2-24.8 degrees; the matching eraser
    # task succeeded at 10.0-15.2 degrees. Higher approaches at 48.7-50.2 degrees
    # encountered reachability failures. Use the candidate-pool limits below.
    #
    # The 30-degree rung comes from the **authoritative candidate pool** (assets/interaction/<id>/grasp_pose/
    # grasp_pose.pkl). The holder upAxis is baked to Z, and placement varies only Z yaw, so local and world elevation agree.
    # 021 (251 candidates): elevation min 0.1 degrees, p10 3.5, median 23.4, p90 54.8, max 87.4;
    # <=20 degrees: 112 (45%); <=25: 127 (51%); <=30: 139~149 (~59%); <=45: 208 (83%).
    # The 30-degree rung has ample candidates before grasp.py:207's random downsampling to 300.
    #
    # Use a ladder instead of a hard limit: other tasks' hard horizontal constraints retained
    # 62~76 candidates but only 0~1 passed IK, reducing success to zero. Relax according to IK results:
    # try 30 degrees, then 45 if lifting fails, finally no limit with a WARN.
    HOLD_ELEV_CAPS = (30.0, 45.0, None)

    # Maximum candidate advance along approach axis to the holder wall (m). 021's annotated TCP median axial radius is 24.1 mm;
    # its outer wall radius is 30.2 mm (TCP already 6.1 mm inside), so this is an **identity transform** for it: only advance, clip to [0, limit].
    # Use the _deepen_grasp_filter pattern verified in the block task;
    # retain it so future assets with regenerated grasps automatically correct
    # annotations that stop outside the holder wall.
    HOLD_RADIAL_CORR_MAX = 0.03

    def make_hold_grasp_filter(elev_cap_deg):
        """Filter holder grasps by side, approach elevation and gripper posture.
        """
        def _f(grasp_poses, obj_pose, grasp_widths):
            grasp_poses = np.asarray(grasp_poses, dtype=float).copy()
            cup_cx, cup_cy = float(obj_pose[0, 3]), float(obj_pose[1, 3])

            # Scale candidate offsets about the holder center. Check the scaled grasp
            # against the holder radius, height and configured approach limits.
            if cup_scale != 1.0:
                _off = grasp_poses[:, :3, 3] - np.array([cup_cx, cup_cy, float(obj_pose[2, 3])])
                grasp_poses[:, :3, 3] = np.array([cup_cx, cup_cy, float(obj_pose[2, 3])]) + _off * cup_scale
                _r_before = np.hypot(_off[:, 0], _off[:, 1])
                _r_after = _r_before * cup_scale
                print(f"[DIAG_HOLD] Grasp-point scale compensation ×{cup_scale}: TCP distance from cup axis "
                      f"{np.median(_r_before) * 1000:.1f}mm -> {np.median(_r_after) * 1000:.1f}mm "
                      f"(target wall outer radius {cup_wall_r * 1000:.1f}mm); framework does not apply this; see notes above CUP_GEOM")

            # (c) Radial correction: solve |xy_tcp + s * xy_col2| = wall_r, choose the root with smaller |s|,
            # then clip to [0, HOLD_RADIAL_CORR_MAX]. Apply only to **near-horizontal** candidates: downward approaches
            # have degenerate horizontal projections and meaningless s. Leave unsolvable candidates unchanged.
            tcp = grasp_poses[:, :3, 3]
            app = grasp_poses[:, :3, 2]
            dx, dy = tcp[:, 0] - cup_cx, tcp[:, 1] - cup_cy
            ax, ay = app[:, 0], app[:, 1]
            a = ax * ax + ay * ay
            b = 2.0 * (dx * ax + dy * ay)
            c = dx * dx + dy * dy - cup_wall_r * cup_wall_r
            disc = b * b - 4.0 * a * c
            solvable = (disc >= 0.0) & (a > 1e-4)
            sq = np.sqrt(np.maximum(disc, 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                r1 = (-b + sq) / (2.0 * a)
                r2 = (-b - sq) / (2.0 * a)
            s = np.where(np.abs(r1) < np.abs(r2), r1, r2)
            step = np.where(solvable, np.clip(s, 0.0, HOLD_RADIAL_CORR_MAX), 0.0)
            if np.any(step > 1e-4):
                print(f"[DIAG_HOLD] Cup-hold radial correction: {int(np.count_nonzero(step > 1e-4))}/"
                      f"{len(step)} candidates advanced along approach axis, median {np.median(step[step > 1e-4]) * 1000:.1f}mm "
                      f"(cap {HOLD_RADIAL_CORR_MAX * 1000:.0f}mm, target wall outer radius {cup_wall_r:.4f})")
            grasp_poses[:, :3, 3] = tcp + app * step[:, None]

            side = hold_sign * (grasp_poses[:, 1, 3] - cup_cy)   # >0 => on the hold side
            mask = side > 0.0
            if elev_cap_deg is not None:
                elev = np.degrees(np.arcsin(np.clip(np.abs(grasp_poses[:, 2, 2]), 0.0, 1.0)))
                both = mask & (elev <= float(elev_cap_deg))
                # Do not discard constraints merely to increase count. Use candidates satisfying both conditions when enough exist;
                # otherwise **retain horizontality first** (camera penetration is a geometric defect) and drop only the side preference.
                # That preference merely clears half the opening; losing it may make insertion harder but does not cause camera collision.
                if np.count_nonzero(both) >= 3:
                    return grasp_poses[both], grasp_widths[both]
                only_elev = elev <= float(elev_cap_deg)
                if np.count_nonzero(only_elev) >= 3:
                    return grasp_poses[only_elev], grasp_widths[only_elev]
                # Too few horizontal candidates on this rung: return side-preferred candidates and let the outer ladder relax the elevation.
            if np.count_nonzero(mask) >= 3:
                return grasp_poses[mask], grasp_widths[mask]
            return grasp_poses, grasp_widths
        return _f

    lifted_ok = False
    for elev_cap in HOLD_ELEV_CAPS:
        lift_grasp_config = GraspConfig(
            pick_up_distance=cup_lift,
            passive_primitive="default",
            grasp_pose_filter=make_hold_grasp_filter(elev_cap),
            unmovable_objects=[table_id],
            action_description={"zh": "握起笔筒", "en": "lifts up the cup"},
        )
        cap_txt = "unlimited" if elev_cap is None else f"≤{elev_cap:.0f}°"
        if robot.pick_actor(cup_id, arm=hold_arm, grasp_config=lift_grasp_config):
            # Verify actual holder lift and orientation after pick_actor to confirm the grip.
            _really_lifted = True
            try:
                _cz = float(np.asarray(robot.get_object_pose(cup_id)[0])[2])
                _rise = _cz - float(cup_pos0[2])
                # Threshold = half pick_up_distance. A real grasp raises the holder cup_lift (0.05);
                # an empty grasp leaves it still (e2 maximum +3.0 mm). 0.025 clearly separates them.
                if _rise < 0.5 * cup_lift:
                    _really_lifted = False
                    print(f"[WARN][DIAG_HOLD] pick_actor returned True at elevation cap {cap_txt}, but "
                          f"cup rose only {_rise * 1000:+.1f}mm (expected ≈{cup_lift * 1000:.0f}mm)"
                          f" — classified as an empty grasp; relaxing and retrying")
            except Exception as _e:                                    # noqa: BLE001
                print(f"[WARN][DIAG_HOLD] Lift verification failed (assuming held): {_e}")
            if not _really_lifted:
                continue
            lifted_ok = True
            # Report achieved grasp elevation.
            try:
                _hg = np.asarray(robot.get_ee_pose(arm=hold_arm), dtype=float).reshape(4, 4)
                _elev = float(np.degrees(np.arcsin(np.clip(abs(float(_hg[2, 2])), 0.0, 1.0))))
                _kind = "horizontal grasp" if _elev < 35.0 else ("oblique grasp" if _elev < 60.0 else "top-down grasp")
                print(f"[DIAG_HOLD] Cup lifted successfully (elevation-cap rung {cap_txt}): actual approach-axis elevation "
                      f"{_elev:.1f}° -> {_kind}; actual cup rise {_rise * 1000:+.1f}mm")
                if _elev >= 35.0:
                    print(f"[WARN][DIAG_HOLD] Cup hold is not horizontal ({_elev:.1f}°); wrist camera may enter "
                          f"the other hand's workspace — pen pickup candidates will likely fall to single digits")
                # Report TCP height relative to the holder rim to diagnose a rim-only grasp.
                _cp, _cr = robot.get_object_pose(cup_id)
                _R = quat2mat_wxyz(_cr)
                _ax = int(np.argmax(np.abs(_R[2, :3])))          # The local column most aligned with world Z is the height axis.
                _sgn = float(np.sign(_R[2, _ax])) or 1.0
                _d = np.asarray(_hg[:3, 3], dtype=float) - np.asarray(_cp, dtype=float)
                _local = _R.T @ _d                                # TCP offset in holder-local coordinates.
                _along = float(_local[_ax]) * _sgn
                _rad = float(np.hypot(*[_local[k] for k in range(3) if k != _ax]))
                _half = 0.5 * cup_geom["h"]
                print(f"[DIAG_HOLD] Grasp height: gripper center along cup height {_along * 1000:+.1f}mm "
                      f"(half-height {_half * 1000:.1f}mm, fraction {_along / _half * 100:+.0f}%), "
                      f"distance from cup axis {_rad * 1000:.1f}mm (wall outer radius {cup_wall_r * 1000:.1f}mm)")
                if abs(_along) > 0.85 * _half:
                    print(f"[WARN][DIAG_HOLD] Grasp at extreme cup end ({_along / _half * 100:+.0f}% of half-height)"
                          f" — center of mass is far from grip, risking swing/rotation after lifting and reducing insertion success")
            except Exception as _e:                                    # noqa: BLE001
                print(f"[WARN][DIAG_HOLD] Cup-hold pose statistics failed (task unaffected): {_e}")
            break
        print(f"[DIAG_HOLD] Cannot lift cup at elevation cap {cap_txt}; relaxing and retrying")
    if not lifted_ok:
        print("Hold arm failed to lift the cup")
        return False

    # ============================================================== STEP 1b
    # Level the cup the hold arm just lifted -- get its MOUTH back to horizontal.
    #
    # Why it is needed: the cup is grasped on its rim/wall (the annotated cup grasp, biased
    # to the hold-side edge by make_hold_grasp_filter), so the contact sits well off the
    # cup's vertical axis. The moment the cup leaves the table it hangs tilted about that
    # contact -- a visibly slanted mouth in the recording, and a slanted mouth also makes
    # STEP 3's pen insert miss more often.
    #
    # Recipe (proven in place_fruit_into_held_bowl_g2 / straighten_and_place_beverage_g2):
    # ONE planned move, NOT an in-place ConstraintRotate. move_actor_to_target_pose keeps
    # the rigid grasp, maps every candidate OBJECT pose through it, batch-IK-filters them
    # (isaac + cuRobo AvoidObs) and plans a single collision-free arc to a reachable one --
    # so no wrist flailing. "Level" is exactly cup_rot0, the rotation the cup had resting
    # flat on the table (the same reference STEP 4 uses to set it back down). A cup is
    # symmetric about its vertical axis, so spinning that level rotation about WORLD Z
    # leaves the mouth just as level while handing the planner many reachable wrist angles.
    cup_level_rot = quat2mat_wxyz(cup_rot0)
    # The cup-mouth normal in the cup's OWN frame: whichever local direction cup_rot0
    # mapped to world +Z back when the cup sat flat. Derived, not guessed, so it is right
    # regardless of which local axis this cup asset happens to use as "up".
    mouth_normal_local = cup_level_rot.T @ np.array([0.0, 0.0, 1.0])

    def cup_tilt_deg():
        """Angle (deg) between the held cup's mouth normal and world +Z. 0 == level."""
        _, rot = robot.get_object_pose(cup_id)
        if rot is None:
            return None
        n = quat2mat_wxyz(rot) @ mouth_normal_local
        n = n / (np.linalg.norm(n) + 1e-12)
        return float(np.degrees(np.arccos(np.clip(float(n[2]), -1.0, 1.0))))

    # Skip the move when the cup came up level anyway: a move to where the cup already is
    # is pure redundant end-of-episode motion. Only level it when measurably tilted.
    LEVEL_TOL_DEG = 5.0
    # Small extra lift while levelling, so a rim that is currently dipping low cannot
    # scrape the desktop as it swings up to horizontal.
    LEVEL_CLEARANCE = 0.02

    tilt_before = cup_tilt_deg()
    if tilt_before is None:
        print("Cannot read held-cup rotation; skipping the levelling move")
    elif tilt_before <= LEVEL_TOL_DEG:
        print(f"Held cup already level (tilt {tilt_before:.1f} deg); no levelling move needed")
    else:
        cup_pos_lifted, _ = robot.get_object_pose(cup_id)
        if cup_pos_lifted is None:
            print("Cannot read held-cup position; skipping the levelling move")
        else:
            level_targets = []
            for dx in np.linspace(-0.01, 0.01, 3):
                for dy in np.linspace(-0.01, 0.01, 3):
                    for angle in np.linspace(0, 360, 12, endpoint=False):
                        pose = np.eye(4)
                        pose[:3, :3] = cup_level_rot
                        pose[:3, 3] = [
                            cup_pos_lifted[0] + dx,
                            cup_pos_lifted[1] + dy,
                            cup_pos_lifted[2] + LEVEL_CLEARANCE,
                        ]
                        level_targets.append(rotate_along_axis(pose, angle, "z", use_local=False))
            level_config = MoveConfig(
                unmovable_objects=[table_id],
                action_description={"zh": "把笔筒摆正", "en": "levels the cup"},
            )
            if robot.move_actor_to_target_pose(
                cup_id, np.array(level_targets), arm=hold_arm, move_config=level_config
            ):
                print(f"Levelled the held cup: tilt {tilt_before:.1f} -> {cup_tilt_deg():.1f} deg")
            else:
                # Non-fatal: the cup is still held and every later step re-reads its live
                # pose, so the episode degrades to the previous (un-levelled) behaviour
                # rather than being thrown away over a cosmetic step.
                print(f"[WARN] levelling move failed; cup stays tilted {tilt_before:.1f} deg")

    # STEP 2: pick the pen with middle-band and elevation filters. Try the configured
    # lift ladder, exclude holder collider meshes where appropriate, and verify
    # actual lift before continuing.
    pen_prim = robot.get_object_prim_path(pen_id)
    cup_prim = robot.get_object_prim_path(cup_id)

    # Expand obstacle exclusions to collider-mesh prim paths; entity paths do not
    # match the planner collision entries. Clear exclusions before returning.
    def obstacle_prims(base):
        subs = ["body/visual"] + [f"body/collider_body_{i:02d}" for i in range(4)]
        return [base] + [f"{base}/{s}" for s in subs]

    # Bandwidth ladder: narrow first, widen and retry if curobo cannot solve. Candidate-count relaxation
    # inside the filter is unreliable (see PEN_MID_BAND_LADDER: 115 candidates, zero IK solutions);
    # only asking curobo establishes reachability. Failure returns within two seconds, so another rung is cheap.
    pen_picked = False
    # Pen height before pickup, used after every rung to verify it actually left the table (see below).
    try:
        _pp0, _ = robot.get_object_pose(pen_id)
        pen_z_before = float(np.asarray(_pp0, float)[2])
    except Exception:                                                  # noqa: BLE001
        pen_z_before = None
    for rung, (band, pickup) in enumerate(zip(PEN_MID_BAND_LADDER, PEN_PICKUP_LADDER)):
        # Refresh collider exclusions before every retry because planner obstacle
        # updates may restore them after a failed motion. Each rung keeps the
        # configured grasp-band constraints.
        robot.remove_objs_from_obstacle(obstacle_prims(pen_prim) + obstacle_prims(cup_prim))
        pen_grasp_config = GraspConfig(
            grasp_offset=0.0,
            pick_up_distance=list(pickup),
            disable_upside_down=True,
            flip_grasp=True,
            grasp_pose_filter=_make_pen_mid_grasp_filter(band),
            unmovable_objects=[table_id, cup_id],
            action_description={"zh": "拿起笔", "en": "picks up the pen"},
        )
        label = ("unrestricted" if band is None else f"±{band * 1000:.0f}mm") +\
            f"/lift {pickup[0] * 1000:.0f}~{pickup[-1] * 1000:.0f}mm"
        if robot.pick_actor(pen_id, arm=place_arm, grasp_config=pen_grasp_config):
            # Physical recheck: pick_actor=True does not mean the pen is in the gripper.
            # e10 (scene 2155428703) reported a successful +/-25 mm, 50~90 mm lift, immediately followed by
            # TCP 64 mm above the table but pen center only 8 mm above it: the gripper rose while the ~10 mm-radius pen stayed flat.
            # curobo attach is **bookkeeping, not a measurement of physical attachment**;
            # after failed closure or slippage, the framework still assumes it holds the pen.
            # The consequence is a silently ineffective episode, not a slightly insufficient lift:
            # stand_dz_min correctly computed 84 mm lift, but all three reorientation rounds reported `89.8->89.8 degrees; wrist turn 0.0 degrees`.
            # move_actor_to_target_pose does nothing to an unheld object,
            # insertion inevitably fails, and success logs conceal ten wasted minutes.
            # More extreme e7 (holder19): every rung failed and the pen was never touched, but all 165 state.json frames
            # showed `if_attached=True`, median TCP-to-pen distance 291 mm. **Do not use this flag to judge grasp success.**
            #
            # Use **the pen's lift**, not TCP-to-pen distance. Distance separation is too narrow:
            # success medians 11.2~32.2 mm versus false-success e10 49.7 mm give only 8 mm margin. A valid
            # off-center 35~45 mm grasp could be falsely rejected, wasting a rung or an otherwise successful episode.
            # Lift separates much better: e10 measured 0 mm, smallest rung lifts 15 mm, measured actual lifts 28~82 mm.
            # An 8 mm threshold gives >=7 mm margin on both sides, independent of coordinate frame and asset size.
            # Frame pitfall: offline checks must not use state.json `ee`, the **flange** (156 mm from pen center).
            # TCP is `ee_center` (8.4 mm), matching live `robot.get_ee_pose()`. The wrong key produces
            # "successful runs also have 445 mm distance", falsely invalidating a correct checker.
            hold_ok, chk_txt = True, ""
            try:
                _pp, _ = robot.get_object_pose(pen_id)
                _pz = float(np.asarray(_pp, float)[2])
                _ee = np.asarray(robot.get_ee_pose(arm=place_arm), float).reshape(4, 4)
                _d = float(np.linalg.norm(np.asarray(_pp, float)[:3] - _ee[:3, 3]))
                if pen_z_before is None:
                    chk_txt = f", gripper-to-pen center {_d * 1000:.0f}mm (pre-grasp height unknown; skipping lift verification)"
                else:
                    _rise = _pz - pen_z_before
                    chk_txt = f", pen rise {_rise * 1000:+.0f}mm, gripper-to-pen center {_d * 1000:.0f}mm"
                    if _rise < 0.008:
                        hold_ok = False
            except Exception as _e:                                    # noqa: BLE001
                chk_txt = f", physical verification failed ({_e}); assuming held"
            if hold_ok:
                print(f"[DIAG_PEN_GRASP] Grasp succeeded, effective rung={label}{chk_txt}")
                if rung > 0:
                    print(f"[WARN][DIAG_PEN_GRASP] Used rung {rung + 1} — lift is lower than rung 1, "
                          f"so the arm is closer to the tabletop when standing the pen; monitor collision counts")
                pen_picked = True
                break
            print(f"[WARN][DIAG_PEN_GRASP] Rung {label}: framework reported success but pen stayed on table{chk_txt} "
                  f"— classified as failure; regrasp at next rung (continuing would silently accomplish nothing)")
        elif rung < len(PEN_PICKUP_LADDER) - 1:
            # Final-rung failure means no acceptable grasp (ladder has no None: reject rather than grasp an end).
            # Use the index rather than `band is not ...[-1]`: bandwidths repeat (first two +/-25 mm),
            # so float identity comparisons are unsafe.
            print(f"[WARN][DIAG_PEN_GRASP] Rung {label} grasp failed; removing obstacles again and retrying next rung")
    if not pen_picked:
        print("Insert arm failed to pick up the pen")
        robot.back_to_origin(arm=place_arm)
        robot.back_to_origin(arm=hold_arm)
        return False

    # STEP 2b: reorient the pen toward vertical after pickup. Keep the holder and
    # pen separated, derive targets from live poses and enforce cumulative wrist
    # rotation limits across retries.
    pen_pos_held, pen_rot_held = robot.get_object_pose(pen_id)
    # Report the axial grasp location from the pen and gripper poses.
    try:
        _ee = np.asarray(robot.get_ee_pose(arm=place_arm), dtype=float).reshape(4, 4)
        _ax = quat2mat_wxyz(pen_rot_held) @ np.array([0.0, 1.0, 0.0])
        _ax = _ax / max(float(np.linalg.norm(_ax)), 1e-9)
        # Analytically reconstruct how far along ee local +Z reaches the pen axis. Squared perpendicular TCP-to-axis distance
        # is quadratic in offset and has a closed-form minimum. Side grasps should lie on-axis, so this minimum also validates the result:
        # implausible perpendicular distance or inferred length invalidates the diagnosis (lesson from e8).
        _v0 = _ee[:3, 3] - np.asarray(pen_pos_held, dtype=float)
        _d = _ee[:3, :3] @ np.array([0.0, 0.0, 1.0])
        _a = _d - (_d @ _ax) * _ax
        _b = _v0 - (_v0 @ _ax) * _ax
        _len = -float(_a @ _b) / max(float(_a @ _a), 1e-12)
        _perp = float(np.linalg.norm(_b + _len * _a))
        _along = float((_v0 + _len * _d) @ _ax)
        _half = float(PEN_LENGTHS.get(pen_usd, 0.145)) / 2.0
        _ok = (EE_TCP_LEN_RANGE[0] <= _len <= EE_TCP_LEN_RANGE[1]) and _perp < 0.005
        print(f"[DIAG_PEN_GRASP] Grasp point: along pen axis from center {_along * 1000:+.1f}mm "
              f"(half-length {_half * 1000:.0f}mm, fraction {abs(_along) / max(_half, 1e-9) * 100:.0f}%, "
              f"distance from pen end {(_half - abs(_along)) * 1000:.0f}mm) | "
              f"inferred gripper length {_len * 1000:.0f}mm perpendicular distance {_perp * 1000:.1f}mm")
        if not _ok:
            print("[WARN][DIAG_PEN_GRASP] Inference unreliable (gripper length or perpendicular distance outside reasonable range); grasp point above is for reference only")
        elif abs(_along) > 0.040:
            print(f"[WARN][DIAG_PEN_GRASP] Grasp point offset {_along * 1000:+.1f}mm from pen center exceeds midsection constraint 40mm")
    except Exception as _e:                                        # noqa: BLE001
        print(f"[WARN][DIAG_PEN_GRASP] Grasp-point diagnostics failed (task unaffected): {_e}")
    # Save wrist orientation after pickup and measure cumulative rotation through
    # reorientation and insertion.
    ee_rot_at_pick = None
    # Pickup tilt is this episode's geometric lower bound for the end-of-STEP-3 net-turn warning. STEP 2b assigns it;
    # retain None if that block is skipped, and use a 90-degree warning fallback.
    tilt_at_pick_for_diag = None
    try:
        ee_rot_at_pick = np.asarray(
            robot.get_ee_pose(arm=place_arm), dtype=float
        ).reshape(4, 4)[:3, :3]
    except Exception as _e:                                        # noqa: BLE001
        print(f"[WARN][DIAG_WRIST] Cannot read wrist pose at grasp; net rotation cannot be measured: {_e}")
    pen_size_held = robot.get_object_size_in_world_frame(pen_id)
    # Measure **actual pen length** here, immediately after pickup while mostly flat; the longest world AABB side then equals length.
    # Once STEP 2b leaves it partly upright (often 31 degrees from vertical), the longest side underestimates length
    # (0.1348 vs 0.1388), biasing STEP 3's half-length tip calculation. Save it now and reuse it.
    # The six pooled pens measured 0.1375~0.1404; this fallback applies only when live size is unreadable.
    pen_len_live = PEN_LENGTHS.get(
        pen_usd, float(np.max(pen_size_held)) if pen_size_held is not None else 0.145
    )
    if pen_pos_held is None or pen_size_held is None:
        print("Cannot read held-pen pose/size; skipping the stand-up move")
    else:
        cup_pos_now, _ = robot.get_object_pose(cup_id)
        # Sideways nudge direction: toward the held holder, so the stand-up already makes
        # progress instead of moving the pen away from where STEP 3 needs it.
        toward_cup = np.zeros(2)
        if cup_pos_now is not None:
            d = np.asarray(cup_pos_now[:2]) - np.asarray(pen_pos_held[:2])
            n = np.linalg.norm(d)
            if n > 1e-6:
                toward_cup = d / n
        # rotation_matrix requires a NumPy axis array. Offer partially upright and
        # vertical goals derived from the live pen pose.
        STAND_FRAC_LADDERS = ((1.0,), (1.0, 0.78), (1.0, 0.78, 0.56, 0.39))
        # Offer small yaw alternatives about the pen axis, widening only through the
        # configured retry levels to limit accumulated wrist rotation.
        STAND_YAW_LADDERS = (
            (0.0, 20.0, -20.0),
            (0.0, 20.0, -20.0, 40.0, -40.0),
            (0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0),
        )

        # Offer either pen endpoint for insertion. Choose targets using the achieved
        # pen pose and cumulative wrist-rotation budget.
        FAR_END_MAX_EXTRA = 45.0

        def min_turn_upright_rots(pen_R_now, cur_tilt_deg, frac_ladder):
            """Return minimum-turn upright rotations as (fraction, rotation, endpoint).

            Derive rotations from the live pen axis and offer both endpoints within
            the cumulative wrist-rotation budget.
            """
            a = np.asarray(pen_R_now, float) @ np.array([0.0, 1.0, 0.0])
            a = a / (float(np.linalg.norm(a)) + 1e-12)
            # Geodesic angle for each endpoint, nearer first (only to establish allowance baseline; downstream order changes).
            ends = []
            for sgn in (1.0, -1.0):
                t = np.array([0.0, 0.0, sgn])
                ends.append((float(np.arccos(np.clip(float(a @ t), -1.0, 1.0))), t))
            ends.sort(key=lambda e: e[0])
            ang_min = ends[0][0]
            out = []
            for ang, t in ends:
                # Offer the far end only when its extra rotation is small (flat pens differ 0~5 degrees and pass).
                if np.degrees(ang - ang_min) > FAR_END_MAX_EXTRA:
                    continue
                axis = np.cross(a, t)
                if float(np.linalg.norm(axis)) < 1e-9:      # Already aligned to this endpoint; no rotation needed.
                    out.append((1.0, np.asarray(pen_R_now, float).copy(), float(t[2])))
                    continue
                for frac in frac_ladder:
                    # Retain only goals **more upright than now**. Without this v5 filter, round 1 reached
                    # 39.7 degrees but round 2 regressed to 55 because a shallow goal became reachable first.
                    # With fractions, this is a lower bound on frac, enforcing monotonic convergence.
                    if np.degrees(ang * (1.0 - frac)) < cur_tilt_deg - 5.0:
                        R = rotation_matrix(ang * frac, np.asarray(axis, float))[:3, :3] @ pen_R_now
                        out.append((frac, R, float(t[2])))
            return out

        def _turn_deg(r_from, r_to):
            """Rotation angle (degrees) required to turn the pen from r_from to r_to. Pen and gripper are rigidly connected,
            so this is also the required wrist rotation."""
            c = (np.trace(np.asarray(r_from).T @ np.asarray(r_to)) - 1.0) / 2.0
            return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
        # Minimum lift that keeps the stood-up pen's LOWER TIP clear of the desktop:
        # the tip ends up half a pen-length below the pen's centre, so the centre has to be at
        # least (tabletop + half length + margin). Derived from live geometry, never guessed;
        # clamped at 0 so a pen already high enough is not pushed down.
        TIP_TABLE_MARGIN = 0.02
        pen_len_held = pen_len_live
        stand_dz_min = max(
            0.0,
            (table_top_z + pen_len_held / 2.0 + TIP_TABLE_MARGIN) - float(pen_pos_held[2]),
        )
        print(f"[DIAG_STAND] pen_z={float(pen_pos_held[2]):.4f} pen_len={pen_len_held:.4f} "
              f"table_top_z={table_top_z:.4f} stand_dz_min={stand_dz_min:.4f}")
        # Arm-to-table clearance: 08-03 video feedback showed table contact during reorientation,
        # while stand_dz_min protects only the pen's lower end with a floor, leaving no arm margin. Log TCP
        # height above the table separately; it measures actual pickup lift (see PEN_PICKUP_LADDER),
        # the observable for this issue. Below 60 mm suggests a relaxed pick rung.
        try:
            _ee_z = float(np.asarray(robot.get_ee_pose(arm=place_arm),
                                     dtype=float).reshape(4, 4)[2, 3])
            _clear = _ee_z - table_top_z
            print(f"[DIAG_STAND] Gripper center above table {_clear * 1000:.0f}mm "
                  f"(pen center above table {(float(pen_pos_held[2]) - table_top_z) * 1000:.0f}mm)")
            if _clear < 0.06:
                print(f"[WARN][DIAG_STAND] Gripper center only {_clear * 1000:.0f}mm above table, "
                      f"so standing the pen may brush the table — check whether [DIAG_PEN_GRASP] used a fallback rung")
        except Exception as _e:                                        # noqa: BLE001
            print(f"[WARN][DIAG_STAND] Gripper-to-table height statistics failed (task unaffected): {_e}")
        def pen_tilt_deg():
            """Angle between pen long axis (local +Y) and world Z; 0 means fully vertical. Return None if unreadable."""
            _, r_now = robot.get_object_pose(pen_id)
            if r_now is None:
                return None
            axis_w = quat2mat_wxyz(r_now) @ np.array([0.0, 1.0, 0.0])
            axis_w = axis_w / (np.linalg.norm(axis_w) + 1e-12)
            return float(np.degrees(np.arccos(np.clip(abs(float(axis_w[2])), -1.0, 1.0))))

        # Check achieved pen verticality after reorientation and retry within the
        # remaining rotation budget when needed.
        STAND_TILT_TOL_DEG = 30.0
        STAND_MAX_TRIES = 3
        # Measure rotation cumulatively from the post-pick wrist pose across all retries.
        TURN_BUDGET_EXTRA = 45.0
        # Total candidate cap: move.py:105 random_downsample acts only **above** 200, so exactly 200 is safe.
        # Both ends in round 3 can yield 2 ends x 4 fractions x 7 yaws x 5 heights=280;
        # all candidates already satisfy turn_cap, so truncating by ascending rotation is harmless:
        # every remaining result is acceptable, and truncation removes
        # the largest rotations, preferable to the framework randomly dropping 80.
        MAX_STAND_TARGETS = 200
        stand_config = MoveConfig(
            unmovable_objects=[table_id, cup_id],
            action_description={"zh": "把笔竖起来", "en": "stands the pen upright"},
            # Framework's authoritative upright-gripper criterion is move.py:88-97: left retains
            # col1[2]<0, right >0, **hard-filtering** reversed wrists.
            # In three measured episodes, choosing the endpoint by a[2] selected the endpoint this mask rejected
            # (table above min_turn_upright_rots). Hard filtering enforces a constraint; sorting does not.
            # Enable it together with **both endpoints**; otherwise all round-1 candidates are removed,
            # reporting `No target_gripper_poses can pass upright filter`.
            extra_params={"disable_upside_down": True},
        )
        # Pickup tilt is the geometric lower bound for one-step reorientation; use it to set the budget.
        tilt_at_pick = pen_tilt_deg()
        if tilt_at_pick is None:
            tilt_at_pick = 90.0
        tilt_at_pick_for_diag = tilt_at_pick
        turn_budget = tilt_at_pick + TURN_BUDGET_EXTRA
        turn_spent = 0.0
        print(f"[DIAG_STAND] Tilt at pickup {tilt_at_pick:.1f}° -> total wrist-rotation budget "
              f"{turn_budget:.1f}° (= geometric minimum + {TURN_BUDGET_EXTRA:.0f}° margin)")
        for stand_try in range(1, STAND_MAX_TRIES + 1):
            pen_pos_now, _ = robot.get_object_pose(pen_id)
            if pen_pos_now is None:
                print("[WARN] Cannot read pen position; abandoning upright rotation")
                break
            turn_left = turn_budget - turn_spent
            if turn_left < 20.0:
                print(f"[DIAG_STAND] Rotation budget exhausted (used {turn_spent:.1f}°/budget "
                      f"{turn_budget:.1f}°); stopping retries to avoid wrist reversal even if the pen is not fully upright")
                break
            stand_dz_now = max(
                0.0,
                (table_top_z + pen_len_held / 2.0 + TIP_TABLE_MARGIN) - float(pen_pos_now[2]),
            )
            tilt_before_try = pen_tilt_deg()
            if tilt_before_try is None:
                tilt_before_try = 90.0
            # Live pen rotation is needed for both minimum rotation and the hard rotation cap.
            _, pen_rot_now = robot.get_object_pose(pen_id)
            pen_R_now = None if pen_rot_now is None else quat2mat_wxyz(pen_rot_now)
            if pen_R_now is None:
                print("[WARN] Cannot read pen orientation; abandoning upright rotation")
                break
            frac_ladder = STAND_FRAC_LADDERS[min(stand_try - 1, len(STAND_FRAC_LADDERS) - 1)]
            rots_now = min_turn_upright_rots(pen_R_now, tilt_before_try, frac_ladder)
            if not rots_now:
                print(f"[DIAG_STAND] Current tilt {tilt_before_try:.1f}° from vertical; no more upright candidates available; stopping retries")
                break
            turn_cap = turn_left
            yaw_ladder = STAND_YAW_LADDERS[min(stand_try - 1, len(STAND_YAW_LADDERS) - 1)]
            stand_targets = []
            turns_kept = []
            ends_kept = []
            # Smallest-viable-first ordering: the planner takes the first candidate its IK accepts,
            # so the extra lift / sideways nudge is only spent when reach genuinely requires it.
            for dz, ds in (
                (stand_dz_now, 0.0),
                (stand_dz_now + 0.03, 0.0),
                (stand_dz_now, 0.04),
                (stand_dz_now + 0.03, 0.04),
                (stand_dz_now + 0.06, 0.08),
            ):
                # Filter candidates by the cumulative wrist-rotation limit before passing
                # them to the planner, which may downsample and reorder them. Keep the
                # accepted candidate set within the planner's batch limit.
                for frac, base_rot, end_sgn in rots_now:
                    # Yaw about the pen's **long axis at the target pose** does not change verticality, merely
                    # moving within equivalent orientations. World-Z yaw did not preserve that equivalence:
                    # when partly upright it also moves the pen axis, so equal verticality is not guaranteed.
                    pen_axis_t = np.asarray(base_rot, float) @ np.array([0.0, 1.0, 0.0])
                    for yaw in yaw_ladder:
                        R = base_rot if yaw == 0.0 else (
                            rotation_matrix(np.radians(yaw), np.asarray(pen_axis_t, float))[:3, :3]
                            @ base_rot
                        )
                        turn = _turn_deg(pen_R_now, R)
                        if turn > turn_cap:
                            continue
                        pose = np.eye(4)
                        pose[:3, :3] = R
                        pose[:3, 3] = [
                            pen_pos_now[0] + toward_cup[0] * ds,
                            pen_pos_now[1] + toward_cup[1] * ds,
                            pen_pos_now[2] + dz,
                        ]
                        stand_targets.append(pose)
                        turns_kept.append(turn)
                        ends_kept.append(end_sgn)
            if not stand_targets:
                print(f"[WARN][DIAG_STAND] Rotation cap {turn_cap:.0f}° removed every candidate; skipping this round")
                continue
            # Precompute the framework disable_upside_down mask for two reasons:
            # 1) **Visibility**: otherwise only `No target_gripper_poses can pass upright filter` is logged,
            # obscuring which endpoint and how many rejected candidates (e3 wasted two rounds this way).
            # 2) **Filter before truncation**: MAX_STAND_TARGETS runs before framework filtering; otherwise all
            # 200 slots could be filled by candidates destined for rejection.
            # Reproduce move.py:26 and :80-97 exactly: gripper2obj=inv(obj_pose) @ ee_pose,
            # target_gripper=target_obj @ gripper2obj; inspect col1's z sign
            # (left <0, right >0 in the omnipicker branch).
            try:
                _ee_pen = np.asarray(robot.get_ee_pose(arm=place_arm), float).reshape(4, 4)
                _obj_pen = np.eye(4)
                _obj_pen[:3, :3] = pen_R_now
                _obj_pen[:3, 3] = np.asarray(pen_pos_now, float)[:3]
                _g2o = np.linalg.inv(_obj_pen) @ _ee_pen
                _col1z = np.array([float((p @ _g2o)[2, 1]) for p in stand_targets])
                _ok = _col1z < 0.0 if place_arm == "left" else _col1z > 0.0
                _n_ok = int(np.count_nonzero(_ok))
                if _n_ok >= 5:
                    idx_ok = np.flatnonzero(_ok)
                    stand_targets = [stand_targets[i] for i in idx_ok]
                    turns_kept = [turns_kept[i] for i in idx_ok]
                    ends_kept = [ends_kept[i] for i in idx_ok]
                    mask_txt = f", wrist-flip mask prefilter {len(_ok)}->{_n_ok}"
                else:
                    mask_txt = (f", [WARN] only {_n_ok}/{len(_ok)} pass wrist-flip mask "
                                f"(<5 is too few to prefilter; passing unchanged to framework, which will likely reject the whole round)")
            except Exception as _e:                                   # Diagnostics must not abort collection.
                mask_txt = f", wrist-flip mask budget failed ({_e})"
            if len(stand_targets) > MAX_STAND_TARGETS:
                keep = np.argsort(np.asarray(turns_kept, float))[:MAX_STAND_TARGETS]
                stand_targets = [stand_targets[i] for i in keep]
                turns_kept = [turns_kept[i] for i in keep]
                ends_kept = [ends_kept[i] for i in keep]
                print(f"[DIAG_STAND] Candidates exceed {MAX_STAND_TARGETS}; truncating by ascending rotation"
                      f"(avoiding random subsampling at move.py:105)")
            n_up = sum(1 for s in ends_kept if s > 0)
            print(f"[DIAG_STAND] Candidates this round: {len(stand_targets)} (rotation cap {turn_cap:.0f}°, "
                  f"actual {min(turns_kept):.1f}~{max(turns_kept):.1f}°, "
                  f"upright fractions {sorted({round(f, 2) for f, _, _ in rots_now})}, "
                  f"upward end +Z:{n_up} / -Z:{len(ends_kept) - n_up}"
                  f"{mask_txt}"
                  f" — framework disable_upside_down selects the end that avoids wrist reversal)")
            moved = robot.move_actor_to_target_pose(
                pen_id, np.array(stand_targets), arm=place_arm, move_config=stand_config
            )
            tilt_now = pen_tilt_deg()
            tilt_txt = "unavailable" if tilt_now is None else f"{tilt_now:.1f}°"
            # Log actual rotation: pen_tilt_deg uses abs(axis_w[2]) and cannot distinguish tip-up
            # from tip-down. v6's wrist unnecessarily turned 146.7 degrees while its logs showed the ideal
            # `89.2->1.1 degrees`; only later recording analysis found the defect. This is its sentinel.
            _, pen_rot_after = robot.get_object_pose(pen_id)
            turn_txt = "unavailable"
            end_txt = "unavailable"
            if pen_R_now is not None and pen_rot_after is not None:
                _turned = _turn_deg(pen_R_now, quat2mat_wxyz(pen_rot_after))
                turn_spent += _turned          # Add to total used budget, tightening the next round's cap.
                turn_txt = f"{_turned:.1f}°"
                # Log the selected endpoint to monitor the **both-endpoints plus framework hard-filter** fix.
                # Both are candidates; disable_upside_down chooses one, revealing which end the framework accepts.
                # Tilt alone cannot show this: pen_tilt_deg uses abs(axis_w[2]), ignoring direction,
                # which hid wrist reversal from logs and forced video-based discovery.
                _ax_after = quat2mat_wxyz(pen_rot_after) @ np.array([0.0, 1.0, 0.0])
                end_txt = "+Z" if float(_ax_after[2]) >= 0.0 else "-Z"
            # Gripper orientation corroboration (TCP frame, diagnostics only): framework masking uses gripper frame;
            # compare this sign with [DIAG_STAND] endpoint counts, not as an acceptance criterion.
            wrist_txt = ""
            try:
                _g = np.asarray(robot.get_ee_pose(arm=place_arm), dtype=float).reshape(4, 4)
                wrist_txt = f", gripper col1[2]={float(_g[2, 1]):+.2f}"
            except Exception:
                pass
            print(f"[DIAG_STAND] Upright attempt {stand_try}/{STAND_MAX_TRIES}: move={moved} "
                  f"tilt from vertical {tilt_before_try:.1f}° -> {tilt_txt}"
                  f"(acceptance threshold {STAND_TILT_TOL_DEG:.0f}°, orientations this round {len(rots_now)}, "
                  f"yaw rung ±{max(yaw_ladder):.0f}°, actual wrist turn {turn_txt}, "
                  f"selected {end_txt} end{wrist_txt}, "
                  f"total {turn_spent:.1f}°/budget {turn_budget:.1f}°)")
            if tilt_now is not None and tilt_now <= STAND_TILT_TOL_DEG:
                print(f"Stood the pen upright: long axis {tilt_now:.1f} deg off world Z")
                break
        else:
            print(f"[WARN] After {STAND_MAX_TRIES} upright attempts, pen failed to reach within {STAND_TILT_TOL_DEG:.0f}°; "
                  f"STEP 3 will insert along its actual tip direction, but excessive tilt will likely prevent insertion")

    # ============================================================== STEP 3
    # Insert arm drops the (now upright, see STEP 2b) pen into the held (airborne) cup.
    # Re-read the cup pose NOW -- it is wherever the hold arm lifted/levelled it to, not its
    # original table spot.
    cup_pos, _ = robot.get_object_pose(cup_id)
    cup_size = _cup_world_size()          # Already compensated by CUP_GEOM['scale']; see _cup_world_size.
    pen_pos, pen_rot = robot.get_object_pose(pen_id)
    pen_size = robot.get_object_size_in_world_frame(pen_id)
    if cup_pos is None or cup_size is None or pen_rot is None or pen_size is None:
        print("Cannot get held-cup / pen poses or sizes")
        robot.back_to_origin(arm=place_arm)
        robot.back_to_origin(arm=hold_arm)
        return False

    cup_top_z = cup_pos[2] + cup_size[2] / 2.0
    # Preserve the pen's achieved orientation while translating above the holder.
    # Derive the target from the live lower endpoint to account for residual tilt.
    pen_grasped_rot = quat2mat_wxyz(pen_rot)
    # Pen's CURRENT vertical extent. After STEP 2b this is the pen's LENGTH (~0.145) rather
    # than its ~0.02 thickness, because get_object_size_in_world_frame returns the WORLD-frame
    # AABB and the long axis is now vertical. The release height must be built from this live
    # value, never from a fixed axis of the asset AABB.
    pen_v_extent = float(pen_size[2])
    pen_length = float(np.max(pen_size))
    # One-line diagnostic: every quantity the release height is built from. Kept because four
    # separate fix attempts on this stage were each aimed at the wrong term.
    print(f"[DIAG_INSERT] cup_pos={np.round(cup_pos,4).tolist()} cup_size={np.round(cup_size,4).tolist()} "
          f"cup_top_z={cup_top_z:.4f} pen_pos={np.round(pen_pos,4).tolist()} "
          f"pen_size={np.round(pen_size,4).tolist()} pen_length={pen_length:.4f} "
          f"pen_v_extent={pen_v_extent:.4f}")
    # Parameterize height by the pen's lower endpoint relative to the rim.
    # Positive tip_depth is below the rim. Balance reachability, aperture depth
    # and finger clearance; every offered candidate must be acceptable because
    # the planner may reorder the candidate pool.
    tip_depths = tuple(cup_geom["tip_depths"])
    # Use five cross-shaped points instead of a 3x3=9 grid: total candidates must stay below 200 to avoid
    # move.py:106 random_downsample(...,200) discarding carefully chosen candidates.
    # Cross points are at most 8 mm away, safer within the 29.8 mm opening than grid corners at 11.3 mm.
    pen_offsets_xy = [(0.0, 0.0), (-0.008, 0.0), (0.008, 0.0), (0.0, -0.008), (0.0, 0.008)]
    # Use small yaw alternatives around the achieved orientation to limit wrist reversal.
    PEN_INSERT_YAWS = (0.0, 30.0, -30.0, 60.0, -60.0)
    # Target the pen's lower endpoint rather than its center, accounting for residual tilt.
    pen_axis_w = pen_grasped_rot @ np.array([0.0, 1.0, 0.0])   # Pen long axis = object-local +Y.
    pen_axis_w = pen_axis_w / (np.linalg.norm(pen_axis_w) + 1e-12)
    # Half-length must use **actual asset length**, not current world AABB's largest component: tilt shortens that bound
    # (e1 measured 0.1348 vs actual 0.1388). pen_len_live was measured before STEP 2b,
    # while flat, when the longest AABB side equals actual length.
    half_len = pen_len_live / 2.0
    tip_rel = pen_axis_w * half_len
    if tip_rel[2] > 0:                 # Choose the downward endpoint.
        tip_rel = -tip_rel
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(float(pen_axis_w[2])), -1.0, 1.0))))
    print(f"[DIAG_INSERT] Pen long-axis tilt from vertical {tilt_deg:.1f}°, half-length={half_len:.4f} "
          f"lower end relative to pen center={np.round(tip_rel, 4).tolist()}")
    # Tip targeting compensates horizontal tilt offset but cannot fix a pen that is too horizontal: after the tip enters,
    # the shaft interferes with the rim and usually lies across it. Above 60 degrees is almost hopeless; WARN for later diagnosis,
    # but still try (no return), allowing pen_in_cup to judge failed insertion.
    if tilt_deg > 60.0:
        print(f"[WARN][DIAG_INSERT] Pen is nearly horizontal ({tilt_deg:.1f}°); STEP 2b upright rotation had little effect; "
              f"insertion will likely fail")
    pen_targets = []
    for yaw in PEN_INSERT_YAWS:
        R = pen_grasped_rot if yaw == 0.0 else (
            rotation_matrix(np.radians(yaw), np.array([0.0, 0.0, 1.0]))[:3, :3] @ pen_grasped_rot
        )
        # Recompute the lower-end offset after applying yaw so the target position
        # uses the candidate's actual orientation.
        ax = np.asarray(R, float) @ np.array([0.0, 1.0, 0.0])
        ax = ax / (float(np.linalg.norm(ax)) + 1e-12)
        tr = ax * half_len
        if tr[2] > 0.0:                # Choose the downward endpoint.
            tr = -tr
        for tip_depth in tip_depths:
            # tip z = rim - tip_depth -> pen center z = rim - tip_depth - tr[2].
            place_z = cup_top_z - tip_depth - float(tr[2])
            for dx, dy in pen_offsets_xy:
                pose = np.eye(4)
                pose[:3, :3] = R
                # Pen center xy = opening axis - tip's horizontal center-relative offset, placing the tip exactly on-axis.
                pose[:3, 3] = [
                    cup_pos[0] - float(tr[0]) + dx,
                    cup_pos[1] - float(tr[1]) + dy,
                    place_z,
                ]
                pen_targets.append(pose)
    pen_targets = np.array(pen_targets)
    print(f"[DIAG_INSERT] Insertion candidates {len(pen_targets)} "
          f"(yaw {list(PEN_INSERT_YAWS)}°, depth {list(tip_depths)}, cross offsets at 5 points)")

    # Verify physical containment from the live pen and holder geometry after release.
    # Check endpoint height, radial position and settling rather than TCP arrival alone.
    PEN_IN_TILT_MAX = 55.0
    # Minimum insertion depth is 0.02: cavities are 74~78 mm deep, giving
    # >=54 mm margin at the bottom; a pen across the rim has tip depth only about +/-10 mm and is rejected.
    PEN_IN_MIN_DEPTH = 0.02

    def pen_in_cup(verbose=False):
        pp, pr = robot.get_object_pose(pen_id)
        cp, _ = robot.get_object_pose(cup_id)
        if pp is None or pr is None or cp is None:
            return False
        cs = _cup_world_size()            # Already scale-compensated; otherwise rim_z can be wrong by 4.7 cm.
        if cs is None:
            return False
        pp = np.asarray(pp, dtype=float)
        cp = np.asarray(cp, dtype=float)
        rim_z = float(cp[2]) + float(cs[2]) / 2.0
        floor_z = rim_z - cup_bore_depth

        # Long axis = object-local +Y (same convention as STEP 3 candidates); use its downward end as tip.
        # Use **actual asset half-length**, not world AABB: tilt shortens its longest side
        # (measured 0.1348 vs 0.1388), overestimating tip height by 9 mm.
        axis = quat2mat_wxyz(pr) @ np.array([0.0, 1.0, 0.0])
        axis = axis / (float(np.linalg.norm(axis)) + 1e-12)
        tip = pp + axis * (pen_len_live / 2.0) * (-1.0 if float(axis[2]) > 0.0 else 1.0)
        tilt = float(np.degrees(np.arccos(np.clip(abs(float(axis[2])), 0.0, 1.0))))
        tip_r = float(np.hypot(tip[0] - cp[0], tip[1] - cp[1]))
        tip_below = rim_z - float(tip[2])          # >0 means tip below the rim.
        centre_r = float(np.hypot(pp[0] - cp[0], pp[1] - cp[1]))

        ok_tip_xy = tip_r <= cup_mouth_r + 0.008
        ok_depth = tip_below >= PEN_IN_MIN_DEPTH
        ok_floor = float(tip[2]) >= floor_z - 0.015
        ok_tilt = tilt <= PEN_IN_TILT_MAX
        ok_centre = centre_r <= cup_mouth_r + 0.03
        if verbose:
            # Log all five measured components and thresholds. The prior only episode reaching settle logged merely
            # "FAILED", obscuring which condition failed; this false success left **no log trace** until
            # state.json reconstruction. Avoid `and` short-circuiting, which hides values after the first False.
            print(f"[DIAG_SETTLE] Tip radial distance={tip_r:.4f} (requires ≤{cup_mouth_r + 0.008:.4f}) {ok_tip_xy} | "
                  f"tip below cup mouth={tip_below:.4f} (requires ≥{PEN_IN_MIN_DEPTH:.3f}) {ok_depth} | "
                  f"tip z={float(tip[2]):.4f} (inner floor {floor_z:.4f}, requires ≥{floor_z - 0.015:.4f}) {ok_floor} | "
                  f"pen tilt={tilt:.1f}° (requires ≤{PEN_IN_TILT_MAX:.0f}°) {ok_tilt} | "
                  f"pen-center radial distance={centre_r:.4f} (requires ≤{cup_mouth_r + 0.03:.4f}) {ok_centre}")
        return bool(ok_tip_xy and ok_depth and ok_floor and ok_tilt and ok_centre)

    pen_place_config = PlaceConfig(
        # pre_insert_offset is added to the placement target. Account for the holder
        # rim, pen length and current tilt when selecting the offset.
        pre_insert_offset=float(cup_geom["pre_insert"]),
        # Apply aperture clearance and wrist-orientation constraints together. Use live
        # object geometry to select compatible placement candidates.
        disable_upside_down=False,
        unmovable_objects=[table_id, cup_id],
        extra_checkers=[pen_in_cup],
        action_description={"zh": "把笔插进笔筒里", "en": "inserts the pen into the cup"},
    )
    pen_success = robot.place_actor_to_target_pose(
        actor=pen_id,
        arm=place_arm,
        target_pose=pen_targets,
        place_config=pen_place_config,
    )
    # Measure net turn before back_to_origin, whose substantial wrist rotation would otherwise contaminate it.
    if ee_rot_at_pick is not None:
        try:
            _ee_now = np.asarray(
                robot.get_ee_pose(arm=place_arm), dtype=float
            ).reshape(4, 4)[:3, :3]
            _c = (float(np.trace(ee_rot_at_pick.T @ _ee_now)) - 1.0) / 2.0
            _net = float(np.degrees(np.arccos(np.clip(_c, -1.0, 1.0))))
            # Compare achieved wrist rotation with the geometric lower bound for this
            # pickup tilt rather than using one absolute warning threshold.
            _floor = float(tilt_at_pick_for_diag) if tilt_at_pick_for_diag else 90.0
            _limit = _floor + 70.0        # Combine 45-degree reorientation allowance and <=60-degree insertion yaw into a 70-degree net allowance.
            print(f"[DIAG_WRIST] Net wrist rotation from pen pickup through insertion {_net:.1f}° "
                  f"(geometric minimum for this run = pickup tilt {_floor:.1f}°, warning threshold {_limit:.1f}°)")
            if _net > _limit:
                print(f"[WARN][DIAG_WRIST] Net wrist rotation {_net:.1f}° exceeds warning threshold {_limit:.1f}°; "
                      f"check whether [DIAG_STAND] total/budget accumulated excessive rotation across rounds")
            # Direct reversal criterion uses the framework's same quantity: move.py:26 gripper2obj=inv(obj)@ee_pose,
            # so disable_upside_down checks this ee rotation's col1[2].
            # Left requires <0, right >0. For an upright pen side grasp, col1 approximately parallels the pen axis
            # because col0/col2 are perpendicular; its sign **encodes wrist reversal**.
            # Net rotation alone misses it: holder11 reoriented by the minimum 87.0 degrees yet remained reversed through insertion;
            # only this sign exposed it. Twice video review found the defect; this line makes it visible in logs.
            _s = float(_ee_now[2, 1])
            _want_neg = (place_arm == "left")
            _flipped = (_s > 0.0) if _want_neg else (_s < 0.0)
            print(f"[DIAG_WRIST_SIGN] Gripper col1[2] at insertion completion={_s:+.3f}"
                  f"({place_arm} arm criterion {'<0' if _want_neg else '>0'})"
                  f" -> {'[flipped]' if _flipped else 'gripper upright'}")
            if _flipped:
                print(f"[WARN][DIAG_WRIST_SIGN] Wrist reversed — check [DIAG_STAND] "
                      f"selected end X; framework disable_upside_down should have filtered this end during upright rotation")
        except Exception as _e:                                    # noqa: BLE001
            print(f"[WARN][DIAG_WRIST] Net-rotation statistics failed (task unaffected): {_e}")
    # get the inserting arm out of the way before the cup is set back down.
    robot.back_to_origin(arm=place_arm)

    # ============================================================== STEP 4
    # Hold arm sets the cup back down on the table (the pen rides inside). Requirements
    # that earlier attempts each broke one of: (1) curobo's collision-aware (AvoidObs) IK
    # must find a solution; (2) no wrist flailing / redundant moves; (3) the cup ends
    # LEVEL, released gently ON the table -- NOT pushed down into it (which ejects the pen).
    #
    # (1) IK: after STEP 3 the pen sits INSIDE the arm-attached cup and curobo reports a
    # permanent cup<->pen contact, so every place pose is in-collision. The ORIGINAL fix was to
    # ATTACH the pen TOGETHER with the cup to the holding arm (an attached group is not
    # collision-checked against itself).
    # STEP 2 now removes **both pen and holder** from obstacles (see item 3), eliminating
    # the deadlock at source: neither is an obstacle, so they cannot collide in planning. This attach_obj call therefore
    # finds both absent from runtime scene, logs `no attachable obstacles left after filter`, and skips.
    # Retain this deliberately: it is idempotent, with bookkeeping-only failure; if a later version drops STEP 2 removal,
    # this attach remains the correct fix for that deadlock and should not be casually deleted.
    #
    # (2)+(3) Use a PLAIN place (PlaceStage), NOT an insert. An insert (pre_insert_offset +
    # insert_depth) runs THREE separately-planned AvoidObs moves (pre-insert -> target ->
    # forced descent); the three arcs are the "back-and-forth swinging" and the final descent is a
    # skip_check CONSTRAINED push that drives the cup into the table, tilting it under load
    # so the trapped contents are flung out when the gripper opens. A plain place is a
    # SINGLE AvoidObs move to the target then a release, collision-checked the whole way so
    # it never rams the table. Target a single fixed LEVEL orientation (cup_rot0, the cup's
    # original flat-on-table rotation; NO yaw sweep -> no wrist twist) with the release
    # point a few mm ABOVE the flush rest height so the cup free-settles by gravity.
    # cup_prim / pen_prim were both resolved in STEP 2 (for the pre-pick obstacle removal).
    robot.attach_obj([cup_prim, pen_prim], arm=hold_arm)

    cup_pos_held, _ = robot.get_object_pose(cup_id)
    # cup_level_rot (the cup's original flat-on-table rotation) was computed in STEP 1b.
    # Release the cup a small clearance above its original flush resting centre
    # cup_pos0[2]. Keeping the target strictly above the table keeps the pose collision-free
    # so AvoidObs solves it in one shot (fewer failed retries = less wobble), and the cup
    # only free-falls this clearance -- gentle, no press, contents stay.
    SET_DOWN_CLEARANCE = 0.006
    cup_target_z = float(cup_pos0[2]) + SET_DOWN_CLEARANCE
    cup_targets = []
    for dx in np.linspace(-0.01, 0.01, 3):
        for dy in np.linspace(-0.01, 0.01, 3):
            pose = np.eye(4)
            pose[:3, :3] = cup_level_rot
            pose[:3, 3] = [cup_pos_held[0] + dx, cup_pos_held[1] + dy, cup_target_z]
            cup_targets.append(pose)
    cup_targets = np.array(cup_targets)

    # success = the cup is back resting on the desktop (near table-top height).
    def cup_back_on_table(verbose=False):
        cp, _ = robot.get_object_pose(cup_id)
        if cp is None:
            return False
        cs = _cup_world_size()            # Already scale-compensated.
        if cs is None:
            return False
        cup_bottom_z = float(cp[2]) - float(cs[2]) / 2.0
        if verbose:
            print(f"[DIAG_SETTLE] cup_bottom_z={cup_bottom_z:.4f} table_top_z={table_top_z:.4f} "
                  f"difference={abs(cup_bottom_z - table_top_z) * 1000:.1f}mm (requires < 30mm)")
        return abs(cup_bottom_z - table_top_z) < 0.03

    cup_place_config = PlaceConfig(
        disable_upside_down=False,
        passive_primitive="default",
        unmovable_objects=[table_id],
        extra_checkers=[cup_back_on_table],
        action_description={"zh": "把笔筒放回桌面", "en": "sets the cup back on the table"},
    )
    cup_success = robot.place_actor_to_target_pose(
        actor=cup_id,
        arm=hold_arm,
        target_pose=cup_targets,
        place_config=cup_place_config,
    )
    # release the attached cup+pen from curobo now that the cup is placed.
    robot.detach_obj()

    # Reset only the holding arm here; the inserting arm is already home.
    # Evaluate final containment after both arms return so the check covers
    # retreat and reset motion.
    robot.back_to_origin(arm=hold_arm)

    # FINAL settle gate. place extra_checkers only fire at the release instant; the pen can
    # still topple out just after (the framework only re-settles and re-checks when the
    # FIRST check fails, so a late fall-out would otherwise pass as SUCCESS). Let physics
    # settle, then independently re-confirm the cup is on the table AND the pen is still in
    # it.
    #
    # Use a settling window rather than one snapshot. In shelf tasks, disturbed objects **swing and rebound**,
    # which one frame can miss. A pen slipping from a holder likewise takes time and may still rest at the rim just after release.
    # Wait for physics to settle, then sample 12 frames (~1.2 s):
    # - **Decide from the final frame**: fallen-out pens cannot return by themselves, whereas testing intermediate downward settling
    # could reject a genuine success.
    # - WARN and record worst values for any failed intermediate frame. Final pass with transient disturbance is a marginal success;
    # these cases guide future tuning.
    robot.wait(0.6)
    win_fail = 0
    cup_ok = pen_ok = False
    for _i in range(12):
        robot.wait(0.1)
        # Verbose only on the final frame to avoid flooding logs; intermediate values matter only when trouble occurs.
        _last = (_i == 11)
        cup_ok = cup_back_on_table(verbose=_last)
        pen_ok = pen_in_cup(verbose=_last)
        if not (cup_ok and pen_ok):
            win_fail += 1
    settle_ok = cup_ok and pen_ok
    if not settle_ok:
        print(f"Set-down settle check FAILED (cup_on_table={cup_ok}, pen_in_cup={pen_ok}) "
              f"-- {win_fail} invalid frames in the 12-frame settling window")
    elif win_fail:
        print(f"[WARN][DIAG_SETTLE] Final state passes, but {win_fail} frames in the 12-frame settling window failed — "
              f"pen wobbled near the cup mouth; borderline success, prioritize this run when tuning")

    return pen_success and cup_success and settle_ok
