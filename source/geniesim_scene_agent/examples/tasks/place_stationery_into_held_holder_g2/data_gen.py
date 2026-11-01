# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np


# One hand holds the pen holder while the other inserts a stationery item; then return the holder to the table.
# This is group three ("one hand holds the container, the other loads it"). Per the task
# definition it is decomposed into FOUR explicit stages:
#   1. one arm picks the cup UP off the table and HOLDS it in the air;
#   1b. that arm levels the lifted cup so its MOUTH is horizontal (the rim grasp leaves it
#       hanging tilted); skipped when it came up level already;
#   2. the other arm picks the upright stationery item up;
#   3. that arm drops the item INTO the held (airborne) cup;
#   4. the holding arm sets the cup back down on the table (item rides inside).
# Preserve the verified held-container motion recipe: level the lifted holder, attach the
# holder and inserted item as a group for set-down, and return it with one fixed level
# orientation. All sampled items are upright before STEP 2. Their horizontal sections fit
# the holder's measured bore, and the final checker permits a slender item to lean while
# still requiring its lower end to be below the rim.
# Keep this pool in sync with STATIONERY_CANDIDATES in LLM_RESULT.py and object_urls.json.
# Real dimensions come from each asset's item.py shapes[bbox].size.
STATIONERY_SPECS = {
    # Use each asset's own grasp annotations and preserve its configured scale.
    # Check that the holder aperture accommodates the selected item.
    "benchmark_stationery_004": {
        "has_interactions": True,
        "object_type": "glue_bottle",
        "zh": "胶水瓶",
        "en": "glue bottle",
        # Measure cavity radius from non-bbox mesh vertices transformed into
        # the asset frame.
        "radius": 0.0257,
    },
    "benchmark_stationery_006": {
        "has_interactions": True,
        "object_type": "glue_stick",
        "zh": "固体胶",
        "en": "glue stick",
        "radius": 0.0140,
    },
    "benchmark_stationery_007": {
        "has_interactions": True,
        "object_type": "glue_stick",
        "zh": "固体胶",
        "en": "glue stick",
        "radius": 0.0110,
    },
    "benchmark_stationery_011": {
        "has_interactions": True,
        "object_type": "eraser",
        "zh": "橡皮擦",
        "en": "eraser",
        "radius": 0.0163,
    },
    "benchmark_stationery_012": {
        "has_interactions": True,
        "object_type": "eraser",
        "zh": "橡皮擦",
        "en": "eraser",
        "radius": 0.0174,
    },
    # Keep this item pool aligned with STATIONERY_CANDIDATES in the scene recipe.
    "benchmark_toy_024": {
        "has_interactions": True,
        "object_type": "toy",
        "zh": "玩偶",
        "en": "figurine",
        "radius": 0.0155,
    },
}
STATIONERY_USDS = set(STATIONERY_SPECS)

# Horizontal radius (m) that must clear the pen-holder mouth. Cuboid erasers use their
# upright half-diagonal; cylindrical glue sticks use their radius; the glue bottle uses the
# maximum XY vertex radius measured from Aligned.usda.
ITEM_R_CIRC = {oid: spec["radius"] for oid, spec in STATIONERY_SPECS.items()}

# How far (m) the holding arm lifts the cup off the desktop. Just enough to read as
# "lifted off the table" while keeping the held cup and the other gripper in the head
# cameras and within a comfortable reach for the placing arm.
CUP_LIFT = 0.05

# Scale compensation for the enlarged pen holder.
# Keep CUP_SCALE_XY synchronized with LLM_RESULT.py, which derives 1.45 and explains scaling XY only.
# Asset 021 uses [1.45, 1.45, 1.0]: horizontal dimensions increase 45%; height stays fixed.
# Client compensation is necessary because the framework has a known scale defect, confirmed by the
# 010*0.48 episode in insert_pen_into_held_holder_g2. Physics/collision follow scene.usda scaleOp,
# and curobo reads the USD stage, but two client operations read unscaled asset metadata:
# 1. robot_api.py:3390 hardcodes object_info["scale"]=1.0, skipping layout/object.py:252's
#    `if scale != 1.0:` branch. Grasp candidates retain their original dimensions. At scale 0.48,
#    010's TCP was 51.2mm from the axis versus a 32.6mm outer radius; the gripper closed in air,
#    and the holder rose <=3mm.
# 2. get_object_size_in_world_frame computes from the ASSETS_INDEX bbox and also returns original size.
# The upstream framework is read-only, so this task provides three local corrections:
# * Grasp points: make_hold_grasp_filter step (0) multiplies horizontal offset from the holder center
#   by CUP_SCALE_XY.
# * World size: use _cup_world_size() everywhere, rather than get_object_size_in_world_frame(cup_id).
# * Bore: CUP_MOUTH_R below is the scaled constant; height and CUP_DEPTH remain unchanged.
CUP_SCALE_XY = 1.45

# Inner radius (m) of benchmark_stationery_021's mouth, measured off the asset mesh
# (Aligned.usda, world-aligned): slicing the mesh horizontally gives an annulus whose inner
# radius is a CONSTANT 0.0298-0.0299 m from the rim down to 0.15H -- this holder is a
# straight-walled tube, not a tapering cup, and it is 0.080 m deep. A CONSTANT is used instead
# of the live world AABB because the AABB gives the OUTER extent (0.062 = 2 x 0.0313); using
# half of it as a mouth radius would overstate the hole by the wall thickness and admit items
# that wedge on the rim. Kept slightly conservative.
# The straight wall supports the upright stationery items over the inserted length instead
# of making them pivot on a tapered rim.
# After enlargement: inner mouth radius = original measurement * CUP_SCALE_XY; height is unchanged,
# so CUP_DEPTH stays 0.080. 0.029 * 1.45 = 0.04205m.
# Outer radius likewise: measured 0.0312 * 1.45 = 0.04524, needed for radial holding-grasp correction.
CUP_MOUTH_R_UNSCALED = 0.029
CUP_WALL_R_UNSCALED = 0.0312
CUP_MOUTH_R = CUP_MOUTH_R_UNSCALED * CUP_SCALE_XY
CUP_WALL_R = CUP_WALL_R_UNSCALED * CUP_SCALE_XY
CUP_DEPTH = 0.080

# Maximum gripper opening accepted by the framework (m). Of 344 annotated assets, 199 have a
# maximum width of exactly 0.0900 (grasp_gen's cap); the library maximum is 0.0999. Use 0.090 as a
# hard limit. make_hold_grasp_filter step (c) computes each candidate's required opening and rejects
# excessive values, testing actual grasp feasibility per candidate instead of imposing a diameter cap.
GRIPPER_MAX_WIDTH = 0.090

# For protruding items, derive the tilt limit from bore diameter and
# cavity depth. Check the actual endpoint and tilt together for containment.
TILT_MAX_DEG = float(np.degrees(np.arctan2(2.0 * CUP_MOUTH_R, CUP_DEPTH)))


# Apply the protruding-item tilt bound only while the item extends above the
# rim; verify complete containment from the actual upper endpoint.
def tilt_limit_deg(item_len, tip_depth=None):
    """Compute the posture-dependent tilt limit (degrees).

    Fully submerged (upper end also below rim): diameter-only asin(diameter/item length),
    about 56 degrees for a flat eraser. Otherwise (protruding or unknown), the rim/bottom
    bound is atan(diameter/cavity depth)=35.9 degrees.
    """
    L = float(item_len)
    if L <= 0.0:
        return TILT_MAX_DEG
    if L >= CUP_DEPTH:                       # Longer than the cavity depth; the item cannot fit entirely inside.
        return TILT_MAX_DEG
    if tip_depth is None:                    # Use the conservative bound when lower-end depth is unavailable.
        return TILT_MAX_DEG
    wide = float(np.degrees(np.arcsin(min(2.0 * CUP_MOUTH_R / L, 1.0))))
    # Upper-end depth >0 means full containment. Check consistently using the wider bound itself; take the stricter result.
    th = np.radians(min(wide, 89.9))
    if float(tip_depth) - L * float(np.cos(th)) > 0.0:
        return wide
    return TILT_MAX_DEG


# Hover-gate safety margin below the holder tilt limit (degrees). Module-level because grasping
# must use the same value; see the item_tilt_cap_deg derivation below.
# Two lateral limits (m), identical to STEP 3's _lateral_fit definitions, exposed here so
# item_tilt_cap_deg can use them during grasping before radial_slack has been calculated.
# lim_tip: maximum lower-end distance from holder axis = radial clearance - 3mm.
# lim_com: maximum object-center distance from holder axis = bore radius - 5mm.
HOVER_TILT_MARGIN = 4.0
def _lat_limits(item_r):
    return (max(CUP_MOUTH_R - float(item_r) - 0.003, 0.0),
            max(CUP_MOUTH_R - 0.005, 0.0))


# Derive the allowable in-gripper tilt from item geometry and aperture
# clearance rather than using one fixed angle for every item.
def item_tilt_cap_deg(item_r, half_len):
    """Maximum permitted in-gripper tilt after lift (degrees); exceed it to retry the next rung rather than continue with an infeasible pose."""
    hover_cap = TILT_MAX_DEG - HOVER_TILT_MARGIN - 2.0
    L = float(half_len)
    if L <= 1e-6:
        return hover_cap
    lim_tip, lim_com = _lat_limits(item_r)
    ratio = (lim_tip + lim_com) / L
    fit_cap = 90.0 if ratio >= 1.0 else float(np.degrees(np.arcsin(ratio)))
    return float(min(hover_cap, fit_cap))


# The container pool. MUST stay in sync with CUP_CANDIDATES in LLM_RESULT.py. Matched by
# EXACT id (not by a substring of the usd path) because the holder and the items are all
# "benchmark_stationery_*" -- a substring match would confuse the two. Same reason as
# insert_pen_into_held_holder_g2, which uses the same holder.
CUP_USDS = {
    "benchmark_stationery_021",
}

# Grasp-depth correction cap. It only deepens candidates and is bounded for thin stationery.
DEPTH_CORR_MAX = 0.010

# PCA threshold used by the existing depth-correction filter for elongated items.
ELONGATED_PCA_RATIO = 2.0


# Each pool asset supplies its own interaction.json and grasp_pose.pkl.
def _deepen_grasp_filter(grasp_poses, obj_pose, grasp_widths):
    """Deepen each candidate along its own approach direction to align the palm with the object.

    Paired with STEP 2 remove_objs_from_obstacle: removing the target from the obstacle world
    lets deep candidates pass AvoidObs IK instead of colliding with the inflated object shell.

    G2 gripper frame: matrix column 3 (col2) is the approach direction, confirmed by
    transport_vector=R@[0,0,1] in grasp.py:214-218. step=clip(depth deficit,0, DEPTH_CORR_MAX)
    only deepens, never retreats; candidates beyond the target stay unchanged. This filter
    translates without rotating, pruning candidates, or changing approach-direction distribution.

    Return only corrected candidates. The framework reorders using jacobian_score/joint distance
    and G2 sorted_by_position_humanlike, so mixing original shallow candidates back in would
    let it select the same under-deep grasp again.

    Unlike fruit/block tasks, elongated bottles must not always target the geometric center:
    * Elongated items (PCA primary/secondary >= ELONGATED_PCA_RATIO, including upright erasers)
      target the nearest point on the bottle axis to each candidate. Center targeting pushes
      upper-body grasps toward the center, causing 15-16mm p90 axial sliding in offline tests,
      from waist to shoulder or a changing-diameter region. Axis projection reduces p90 sliding
      to 1.2-1.7mm and radial distance from 11.8 to 11.6mm, instead of increasing it to 14.0mm.
    * Isotropic items (not expected in this pool, retained for generality) have no reliable
      long axis and target the geometric center, as in fruit/block tasks.
    Derive the long axis by PCA on candidate positions, not hardcoded asset-local axes:
    001/006 happen to use +Y while 003 uses +X.
    """
    gp = np.asarray(grasp_poses, dtype=float)
    gw = np.asarray(grasp_widths, dtype=float)
    if gp.shape[0] == 0:
        return gp, gw
    a = gp[:, :3, 2]
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n < 1e-9] = 1.0
    a = a / n
    obj_c = np.asarray(obj_pose, dtype=float)[:3, 3]
    pts = gp[:, :3, 3]
    shape = "isotropic"
    targets = np.tile(obj_c, (gp.shape[0], 1))
    if gp.shape[0] >= 3:
        centred = pts - pts.mean(axis=0)
        _, sing, vt = np.linalg.svd(centred, full_matrices=False)
        ratio = float(sing[0] / max(sing[1], 1e-9))
        if ratio >= ELONGATED_PCA_RATIO:
            long_axis = vt[0]
            # Nearest point to each candidate on the bottle axis: the line through the object origin along long_axis.
            rel = pts - obj_c[None, :]
            targets = obj_c[None, :] + (rel @ long_axis)[:, None] * long_axis[None, :]
            shape = f"elongated(PCA={ratio:.1f})"
        else:
            shape = f"isotropic(PCA={ratio:.1f})"
    along = np.einsum("ij,ij->i", targets - pts, a)
    step = np.clip(along, 0.0, DEPTH_CORR_MAX)
    gp[:, :3, 3] = pts + a * step[:, None]
    print(f"[DIAG_GRASP] depth centering for {len(step)} candidates ({shape}): added depth {step.min() * 1000:.1f}~"
          f"{step.max() * 1000:.1f}mm (clamped to {DEPTH_CORR_MAX * 1000:.0f}mm)")
    return gp, gw


def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_stationery_021": {"has_interactions": True, "object_type": "pen_holder", "interaction_type": "container"},
        "table_000": {"has_interactions": False, "object_type": "table"},
    }
    for oid, spec in STATIONERY_SPECS.items():
        objects_list[oid] = {
            "has_interactions": spec["has_interactions"],
            "object_type": spec["object_type"],
        }

    scene_info = robot.scene_info

    description_json = {
        "task_name": 'Place stationery into a holder held by the other hand',
        "english_task_name": 'Place stationery into a holder held by the other hand',
        "init_scene_text": "The robot is in front of a table with a pen holder near the middle and one upright stationery item beside it",
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

    # --- locate the pen holder (the held container) and the stationery item ---
    cup_id = None
    item_id = None
    item_usd = ""
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        kw = obj_info.get("keywords", [])
        if obj_id == table_id:
            continue
        # Match exact IDs, not USD substrings: both holder and stationery share the benchmark_stationery
        # prefix. A test like `"benchmark_cup" in usd` from related tasks would confuse them here; keywords
        # are also ambiguous because both include "stationery". Use the ID sets, with keywords only as a
        # redundant fallback on the container side.
        if usd in CUP_USDS or "pen_holder" in kw or "target_container" in kw:
            cup_id = obj_id
        elif usd in STATIONERY_USDS:
            item_id = obj_id
            item_usd = usd

    if cup_id is None or item_id is None:
        print("Required objects not found (need 1 pen holder and 1 stationery item)")
        return False

    item_spec = STATIONERY_SPECS[item_usd]
    item_zh = item_spec["zh"]
    item_en = item_spec["en"]

    # --- resolve which arm HOLDS (lifts) the cup and which arm PLACES, geometrically ---
    # robot at -x facing +x => +y is the robot's LEFT. The cup sits just to the HOLD side of
    # centre; the item sits on the OPPOSITE (place) side. The holding arm is the cup's side;
    # the placing arm is the item's side (no cross-body).
    cup_pos0, cup_rot0 = robot.get_object_pose(cup_id)
    cup_pos0 = np.asarray(cup_pos0)
    hold_arm = "left" if cup_pos0[1] >= 0 else "right"
    place_arm = "right" if hold_arm == "left" else "left"
    hold_sign = 1.0 if hold_arm == "left" else -1.0
    print(f"Cup on {'left' if hold_arm == 'left' else 'right'}; hold arm = {hold_arm}, place arm = {place_arm}")

    def _cup_world_size():
        """Holder world size compensated for scene scaling [CUP_SCALE_XY, CUP_SCALE_XY,1].

        get_object_size_in_world_frame reads the original ASSETS_INDEX bbox without scale;
        see CUP_SCALE_XY above. All holder-size consumers must use this helper. Height stays
        unchanged, so multiply only the first two dimensions.
        """
        raw = robot.get_object_size_in_world_frame(cup_id)
        if raw is None:
            return None
        return [float(raw[0]) * CUP_SCALE_XY, float(raw[1]) * CUP_SCALE_XY, float(raw[2])]

    # Sentinel: corrected outer diameter should be approximately 2*CUP_WALL_R. A mismatch means
    # CUP_SCALE_XY differs from LLM_RESULT.py, silently corrupting grasp compensation, bore, and tilt limits.
    _cs_chk = _cup_world_size()
    if _cs_chk is not None:
        _dr = abs(0.5 * max(_cs_chk[0], _cs_chk[1]) - CUP_WALL_R)
        print(f"[DIAG_CUP] CUP_SCALE_XY={CUP_SCALE_XY} compensated world dimensions="
              f"{np.round(_cs_chk, 4).tolist()} | inner mouth radius={CUP_MOUTH_R:.4f} "
              f"outer wall radius={CUP_WALL_R:.4f} holder height={CUP_DEPTH:.3f} tilt limit={TILT_MAX_DEG:.1f}°")
        if _dr > 0.004:
            print(f"[WARN][DIAG_CUP] compensated outer radius differs from CUP_WALL_R by {_dr * 1000:.1f}mm — "
                  f"CUP_SCALE_XY may differ between LLM_RESULT.py and data_gen.py;"
                  f"grasp compensation, mouth diameter, and tilt limit would all be wrong")

    # STEP 1: hold the container using a near-horizontal approach and validate
    # its actual lift and orientation before picking the item.
    HOLD_ELEV_CAPS = (30.0, 45.0, None)

    def make_hold_grasp_filter(elev_cap_deg):
        """(0) Compensate scale; (a) prefer holding-side edge; (b) require horizontal approach; (c) check opening.

        grasp_poses are already in the world frame, so col2=grasp_poses[:,:3,2] is the world
        approach axis, whose z component gives elevation. Measurements confirm this:
        a pen lying on the table had col2=[+0.38,-0.35,-0.85], a downward approach, while a
        horizontal holder grasp had z=0.18.
        """
        def _f(grasp_poses, obj_pose, grasp_widths):
            grasp_poses = np.asarray(grasp_poses, dtype=float).copy()
            grasp_widths = np.asarray(grasp_widths, dtype=float)
            cup_cx, cup_cy = float(obj_pose[0, 3]), float(obj_pose[1, 3])

            # (0) Scale compensation: multiply candidate horizontal offset from the holder center by CUP_SCALE_XY.
            # This must run first: steps (a)(b)(c) compare candidate positions against actual holder geometry.
            # Annotations are in the asset's local frame. Scaling [s, s,1] about the asset origin multiplies local
            # x, y by s and leaves z fixed. obj_pose translation is the holder center, which remains fixed;
            # in world coordinates this multiplies the horizontal offset from that center by s and leaves z fixed.
            # Rotation is unchanged: s_x=s_y preserves the circular cylinder and radial approach directions.
            # This is why LLM_RESULT.py uses [s, s,1] instead of unequal horizontal scaling.
            if abs(CUP_SCALE_XY - 1.0) > 1e-9 and grasp_poses.shape[0] > 0:
                grasp_poses[:, 0, 3] = cup_cx + (grasp_poses[:, 0, 3] - cup_cx) * CUP_SCALE_XY
                grasp_poses[:, 1, 3] = cup_cy + (grasp_poses[:, 1, 3] - cup_cy) * CUP_SCALE_XY

            side = hold_sign * (grasp_poses[:, 1, 3] - cup_cy)   # >0 => on the hold side
            mask = side > 0.0

            # (c) Can the gripper open far enough? Evaluate grasp feasibility from the actual wall contact.
            # Holding straddles the wall: TCP is the finger midpoint, wall radius is CUP_WALL_R, and fingers lie
            # at TCP radius +/- w/2. Contact requires w > 2*|CUP_WALL_R - r_TCP|.
            # * Candidates near the wall (r_TCP approximately wall radius) need only a few millimeters of opening,
            #   independently of holder diameter.
            # * Candidates across the entire mouth (r_TCP approximately 0) require the full outer diameter.
            # With outer diameter 90.5mm, the extreme case needs 2*45.2=90.5mm, at the 0.090 opening limit.
            # Compute and reject each excessive candidate rather than imposing a guessed diameter limit.
            # Leave 5mm margin for finger thickness and extra opening during approach.
            r_tcp = np.hypot(grasp_poses[:, 0, 3] - cup_cx, grasp_poses[:, 1, 3] - cup_cy)
            w_need = 2.0 * np.abs(CUP_WALL_R - r_tcp)
            span_ok = w_need <= (GRIPPER_MAX_WIDTH - 0.005)
            n_span = int(np.count_nonzero(span_ok))
            if n_span >= 3:
                if n_span < len(span_ok):
                    print(f"[DIAG_HOLD] gripper opening filter: {n_span}/{len(span_ok)} candidates pass "
                          f"(required min {w_need.min() * 1000:.0f} median "
                          f"{np.median(w_need) * 1000:.0f} max {w_need.max() * 1000:.0f}mm,"
                          f"limit {(GRIPPER_MAX_WIDTH - 0.005) * 1000:.0f}mm)")
                mask = mask & span_ok
            else:
                print(f"[WARN][DIAG_HOLD] gripper opening filter leaves only {n_span} candidates; skipping this filter for this tier — "
                      f"check whether CUP_SCALE_XY ({CUP_SCALE_XY}) makes the holder too large")
                span_ok = np.ones(len(span_ok), dtype=bool)

            if elev_cap_deg is not None:
                elev = np.degrees(np.arcsin(np.clip(np.abs(grasp_poses[:, 2, 2]), 0.0, 1.0)))
                both = mask & (elev <= float(elev_cap_deg))
                # Do not abandon constraints just to increase candidate count. Use candidates meeting both conditions
                # when sufficient. Otherwise retain horizontal approach (camera intersection is a geometric failure)
                # and relax only the holding-side preference, which merely clears the opposite mouth half.
                # Relaxing that preference may complicate insertion but does not create the camera collision.
                # Always retain span_ok at every rung: insufficient opening makes a grasp physically impossible.
                if np.count_nonzero(both) >= 3:
                    return grasp_poses[both], grasp_widths[both]
                only_elev = (elev <= float(elev_cap_deg)) & span_ok
                if np.count_nonzero(only_elev) >= 3:
                    return grasp_poses[only_elev], grasp_widths[only_elev]
                # Too few horizontal candidates at this rung; let the outer ladder relax it. Return side-preferred candidates here.
            if np.count_nonzero(mask) >= 3:
                return grasp_poses[mask], grasp_widths[mask]
            return grasp_poses, grasp_widths
        return _f

    lifted_ok = False
    for elev_cap in HOLD_ELEV_CAPS:
        lift_grasp_config = GraspConfig(
            pick_up_distance=CUP_LIFT,
            passive_primitive="default",
            grasp_pose_filter=make_hold_grasp_filter(elev_cap),
            unmovable_objects=[table_id],
            action_description={"zh": "端起笔筒", "en": "lifts up the pen holder"},
        )
        cap_txt = "unlimited" if elev_cap is None else f"≤{elev_cap:.0f}°"
        if robot.pick_actor(cup_id, arm=hold_arm, grasp_config=lift_grasp_config):
            lifted_ok = True
            # Report achieved grasp elevation.
            try:
                _hg = np.asarray(robot.get_ee_pose(arm=hold_arm), dtype=float).reshape(4, 4)
                _elev = float(np.degrees(np.arcsin(np.clip(abs(float(_hg[2, 2])), 0.0, 1.0))))
                _kind = "lateral grasp" if _elev < 35.0 else ("angled grasp" if _elev < 60.0 else "top-down grasp")
                print(f"[DIAG_HOLD] holder lifted (elevation cap {cap_txt}): actual approach-axis elevation "
                      f"{_elev:.1f}° -> {_kind}")
                if _elev >= 35.0:
                    print(f"[WARN][DIAG_HOLD] holder grasp is not lateral ({_elev:.1f}°); the wrist camera may enter "
                          f"the other hand's workspace — planning will likely fail during insertion")
            except Exception as _e:                                    # noqa: BLE001
                print(f"[WARN][DIAG_HOLD] elevation measurement failed (task unaffected): {_e}")
            break
        print(f"[DIAG_HOLD] holder lift failed at elevation cap {cap_txt}; relaxing and retrying")
    if not lifted_ok:
        print("Hold arm failed to lift the cup")
        return False

    # ============================================================== STEP 1b
    # Level the cup the hold arm just lifted -- get its MOUTH back to horizontal.
    # Why it is needed: the cup is grasped on its rim/wall (the annotated cup grasp, biased
    # to the hold-side edge by hold_side_grasp_filter), so the contact sits well off the
    # cup's vertical axis. The moment the cup leaves the table it hangs tilted about that
    # contact -- a visibly slanted mouth in the recording, and a slanted mouth also makes
    # STEP 3's drop miss more often.
    # Recipe (proven in place_fruit_into_held_bowl_g2 / straighten_and_place_beverage_g2): ONE
    # planned move, NOT an in-place ConstraintRotate. move_actor_to_target_pose keeps the
    # rigid grasp, maps every candidate OBJECT pose through it, batch-IK-filters them (isaac
    # + cuRobo AvoidObs) and plans a single collision-free arc to a reachable one -- so no
    # wrist flailing. "Level" is exactly cup_rot0, the rotation the cup had resting flat on
    # the table (the same reference STEP 4 uses to set it back down). A cup is symmetric
    # about its vertical axis, so spinning that level rotation about WORLD Z leaves the mouth
    # just as level while handing the planner many reachable wrist angles.
    cup_level_rot = quat2mat_wxyz(cup_rot0)
    # The cup-mouth normal in the cup's OWN frame: whichever local direction cup_rot0 mapped
    # to world +Z back when the cup sat flat. Derived, not guessed, so it is right regardless
    # of which local axis this cup asset happens to use as "up".
    mouth_normal_local = cup_level_rot.T @ np.array([0.0, 0.0, 1.0])

    def cup_tilt_deg():
        """Angle (deg) between the held cup's mouth normal and world +Z. 0 == level."""
        _, rot = robot.get_object_pose(cup_id)
        if rot is None:
            return None
        n = quat2mat_wxyz(rot) @ mouth_normal_local
        n = n / (np.linalg.norm(n) + 1e-12)
        return float(np.degrees(np.arccos(np.clip(float(n[2]), -1.0, 1.0))))

    # Skip the move when the cup came up level anyway: a move to where the cup already is is
    # pure redundant motion. Only level it when measurably tilted.
    LEVEL_TOL_DEG = 5.0
    # Small extra lift while levelling, so a rim that is currently dipping low cannot scrape
    # the desktop as it swings up to horizontal.
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
                action_description={"zh": "把笔筒摆正", "en": "levels the pen holder"},
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

    # ============================================================== STEP 2
    # Place arm picks the stationery item up. The held holder and the table are unmovable for
    # this arm so the collision-aware IK avoids the airborne cup.
    # Save the item's known-upright scene pose before grasping. A frictional grasp can leave a
    # slender item tilted after lift; STEP 3 deliberately restores this pose instead of preserving
    # that tilt over the narrow holder mouth.
    _, item_upright_rot_q = robot.get_object_pose(item_id)
    item_upright_size = robot.get_object_size_in_world_frame(item_id)
    if item_upright_rot_q is None or item_upright_size is None:
        print("Cannot read the stationery item's upright reference pose")
        robot.back_to_origin(arm=place_arm)
        robot.back_to_origin(arm=hold_arm)
        return False
    item_upright_rot = quat2mat_wxyz(item_upright_rot_q)
    item_half_height = float(item_upright_size[2]) / 2.0
    item_axis_local = item_upright_rot.T @ np.array([0.0, 0.0, 1.0])

    # This item's circumscribed radius and derived in-gripper tilt limit (see item_tilt_cap_deg).
    # Look up r_circ; use the live AABB half-diagonal only as fallback, since random yaw systematically inflates it.
    item_r_circ = ITEM_R_CIRC.get(
        item_usd, float(np.hypot(item_upright_size[0], item_upright_size[1])) / 2.0)
    ITEM_TILT_CAP_DEG = item_tilt_cap_deg(item_r_circ, item_half_height)
    _lt, _lc = _lat_limits(item_r_circ)
    print(f"[DIAG_GRASP] {item_en}: r_circ={item_r_circ:.4f} half-length={item_half_height:.4f} "
          f"radial clearance={(CUP_MOUTH_R - item_r_circ) * 1000:.1f}mm "
          f"(lim_tip={_lt * 1000:.1f} lim_com={_lc * 1000:.1f}) "
          f"→ in-gripper tilt limit {ITEM_TILT_CAP_DEG:.1f}°")
    # Try a pickup lift ladder and use collider-mesh obstacle exclusions. Select
    # clearance and stationary-object thresholds for an airborne held container.
    def obstacle_prims(base):
        subs = ["body/visual"] + [f"body/collider_body_{i:02d}" for i in range(4)]
        return [base] + [f"{base}/{s}" for s in subs]

    item_prim = robot.get_object_prim_path(item_id)
    cup_prim = robot.get_object_prim_path(cup_id)

    # Retry pickup through the configured lift ladder, preserving the intended
    # elevation filters and validating achieved lift.
    ITEM_PICKUP_LADDER = ([0.18, 0.22], [0.15, 0.19], [0.13, 0.17], [0.115, 0.16])
    # Original values, reserved as fallback. A separate constant preserves the first six item_rungs:
    # elevation relaxation immediately follows lift relaxation, without four fallback rungs inserted between.
    ITEM_PICKUP_FALLBACK = ([0.08, 0.12], [0.05, 0.09], [0.03, 0.07], [0.015, 0.06])

    # Filter placing-hand grasp elevation to keep the item upright enough for
    # transport and insertion.
    ITEM_ELEV_CAPS = (35.0, 50.0, None)

    def make_item_grasp_filter(elev_cap_deg):
        """Hard elevation filter followed by existing depth centering. Order matters: filter first; deepening changes translation only."""
        def _f(grasp_poses, obj_pose, grasp_widths):
            gp = np.asarray(grasp_poses, dtype=float)
            gw = np.asarray(grasp_widths, dtype=float)
            if elev_cap_deg is not None and gp.shape[0] > 0:
                elev = np.degrees(np.arcsin(np.clip(np.abs(gp[:, 2, 2]), 0.0, 1.0)))
                keep = elev <= float(elev_cap_deg)
                n = int(np.count_nonzero(keep))
                print(f"[DIAG_GRASP] elevation filter ≤{elev_cap_deg:.0f}°: {n}/{len(elev)} candidates pass "
                      f"(measured elevation min {elev.min():.1f}° median {np.median(elev):.1f}° max {elev.max():.1f}°)")
                # Skip filtering if fewer than three remain: eliminating the pool discards the episode, while a tilted grasp can be retried.
                if n >= 3:
                    gp, gw = gp[keep], gw[keep]
                else:
                    print(f"[WARN][DIAG_GRASP] elevation ≤{elev_cap_deg:.0f}° leaves only {n} candidates; skipping elevation filter for this tier")
            return _deepen_grasp_filter(gp, obj_pose, gw)
        return _f

    # Record pre-grasp item height to verify physical lift at each rung, and retain the complete position.
    # DIAG_SETTLE distinguishes slippage near the start from rim impacts landing near the holder mouth.
    item_pos_prepick = None
    try:
        _ip0, _ = robot.get_object_pose(item_id)
        item_pos_prepick = np.asarray(_ip0, float).copy()
        item_z_before = float(item_pos_prepick[2])
    except Exception:                                                  # noqa: BLE001
        item_z_before = None

    item_picked = False
    # The retry ladder pairs elevation relaxation with lift choices; avoid
    # repeating combinations that provide the same motion.
    item_rungs = ([(ITEM_ELEV_CAPS[0], p) for p in ITEM_PICKUP_LADDER]
                  + [(ITEM_ELEV_CAPS[1], ITEM_PICKUP_LADDER[0]),
                     (ITEM_ELEV_CAPS[2], ITEM_PICKUP_LADDER[0])]
                  + [(ITEM_ELEV_CAPS[0], p) for p in ITEM_PICKUP_FALLBACK])

    # Keep pickup recovery reentrant so a failed hover check can resume at the
    # next untried grasp rung.
    def try_pick_item(start_rung):
        """Try item-grasp rungs starting at start_rung. Return (grasp succeeded, starting rung for the next round)."""
        for rung in range(int(start_rung), len(item_rungs)):
            elev_cap, pickup = item_rungs[rung]
            # Repeat obstacle removal before every rung. Pen v7 e4 showed that a failed initial pick triggers
            # framework `Fast obstacle update`, reconstructing the world from the scene and restoring both held
            # holder and target. Every later retry then self-collides at START, preventing fallback from helping.
            robot.remove_objs_from_obstacle(obstacle_prims(item_prim) + obstacle_prims(cup_prim))
            item_grasp_config = GraspConfig(
                pick_up_distance=list(pickup),
                disable_upside_down=True,
                flip_grasp=True,
                grasp_pose_filter=make_item_grasp_filter(elev_cap),
                unmovable_objects=[table_id, cup_id],
                action_description={
                    "zh": f"拿起{item_zh}",
                    "en": f"picks up the {item_en}",
                },
            )
            label = (f"elevation≤{elev_cap:.0f}°" if elev_cap is not None else "no elevation filter") +\
                    f" / lift {pickup[0] * 1000:.0f}~{pickup[-1] * 1000:.0f}mm"
            if not robot.pick_actor(item_id, arm=place_arm, grasp_config=item_grasp_config):
                if rung < len(item_rungs) - 1:
                    print(f"[WARN][DIAG_GRASP] grasp failed at tier {label}; remove again and fall back to the next tier")
                continue

            # Verify physical lift: pick_actor=True does not prove the object is held. curobo attach is bookkeeping,
            # so the framework still believes it holds a missed or slipped object. The rest of the episode can
            # silently move an empty gripper, log success throughout, and ultimately fail placement.
            # Use the item's own rise, rather than TCP-to-center distance (only millimeters of margin with an
            # existing 8mm offset). Rise cleanly separates missed grips (0mm) from even the smallest 15mm lift.
            hold_ok, chk_txt = True, ""
            try:
                _ip, _ = robot.get_object_pose(item_id)
                _iz = float(np.asarray(_ip, float)[2])
                if item_z_before is None:
                    chk_txt = ",(pre-grasp height unknown; skipping lift verification)"
                else:
                    _rise = _iz - item_z_before
                    chk_txt = f",item rise {_rise * 1000:+.0f}mm"
                    if _rise < 0.008:
                        hold_ok = False
            except Exception as _e:                                        # noqa: BLE001
                chk_txt = f",physical verification failed ({_e}); treating as grasped"

            # After lift, check the item's achieved orientation as well as its height.
            if hold_ok:
                try:
                    _ip2, _iq2 = robot.get_object_pose(item_id)
                    _ax = quat2mat_wxyz(_iq2) @ item_axis_local
                    _ax = _ax / (float(np.linalg.norm(_ax)) + 1e-12)
                    _tilt = float(np.degrees(np.arccos(np.clip(abs(float(_ax[2])), 0.0, 1.0))))
                    chk_txt += f",in-gripper tilt {_tilt:.1f}° (limit {ITEM_TILT_CAP_DEG:.1f}°)"
                    if _tilt > ITEM_TILT_CAP_DEG and rung < len(item_rungs) - 1:
                        print(f"[WARN][DIAG_GRASP] tier {label} lifted the item but in-gripper tilt is {_tilt:.1f}° "
                              f"> {ITEM_TILT_CAP_DEG:.1f}° — downstream insertion will fail; retry grasp at the next tier")
                        hold_ok = False
                except Exception as _e:                                    # noqa: BLE001
                    chk_txt += f",in-gripper tilt verification failed ({_e})"

            if hold_ok:
                print(f"[DIAG_GRASP] grasp succeeded, effective tier={label}{chk_txt}")
                # Base the warning on achieved lift rather than the retry-rung number.
                if pickup[0] < ITEM_PICKUP_FALLBACK[0][0] + 1e-9:
                    print(f"[WARN][DIAG_GRASP] using tier {rung + 1} (fallback) — lift "
                          f"{pickup[0] * 1000:.0f}mm has reverted to the level before the +10cm increase;"
                          f"the item passes closer to the table when moving above the holder; watch the collision count")
                return True, rung + 1
            print(f"[WARN][DIAG_GRASP] tier {label} reported success but the item stayed on the table{chk_txt} "
                  f"— treating as failure and retrying at the next tier (continuing would silently execute with an empty gripper)")
        return False, len(item_rungs)

    item_picked, rung_cursor = try_pick_item(0)

    if not item_picked:
        print(f"Place arm failed to pick up the {item_en}")
        robot.back_to_origin(arm=place_arm)
        robot.back_to_origin(arm=hold_arm)
        return False

    # ============================================================== STEP 3
    # Place arm drops the item INTO the held (airborne) cup. Re-read the cup pose NOW -- it
    # is wherever the hold arm lifted/levelled it to, not its original table spot.
    cup_pos, cup_rot = robot.get_object_pose(cup_id)
    cup_size = _cup_world_size()
    item_pos, item_rot = robot.get_object_pose(item_id)
    item_size = robot.get_object_size_in_world_frame(item_id)
    if cup_pos is None or cup_rot is None or cup_size is None or item_rot is None or item_size is None:
        print("Cannot get held-cup / item poses or sizes")
        robot.back_to_origin(arm=place_arm)
        robot.back_to_origin(arm=hold_arm)
        return False

    # The AABB top is not the mouth centre when the held holder is slightly tilted. Track the
    # actual mouth along the holder's live local-up axis instead. mouth_normal_local was derived
    # from the holder's original level pose in STEP 1b.
    mouth_normal_world = quat2mat_wxyz(cup_rot) @ mouth_normal_local
    mouth_normal_world /= np.linalg.norm(mouth_normal_world) + 1e-12
    cup_mouth = np.asarray(cup_pos, dtype=float) + mouth_normal_world * (CUP_DEPTH / 2.0)
    cup_top_z = float(cup_mouth[2])

    # Choose insertion targets from the live item and holder geometry. Keep
    # approach orientation, aperture clearance and release depth consistent.
    print(f"[DIAG_PLACE] cup_pos={np.round(cup_pos,4).tolist()} cup_size={np.round(cup_size,4).tolist()} "
          f"mouth={np.round(cup_mouth,4).tolist()} mouth_tilt="
          f"{np.degrees(np.arccos(np.clip(mouth_normal_world[2], -1.0, 1.0))):.1f}deg "
          f"item_pos={np.round(item_pos,4).tolist()} item_size={np.round(item_size,4).tolist()} "
          f"upright_size={np.round(item_upright_size,4).tolist()}")

    # PlaceStage adds 10 mm in world Z before release. Subtract that offset
    # when targeting the item's lower endpoint below the rim.
    PLACE_STAGE_RELEASE_LIFT = 0.01
    tip_depths = (0.015, 0.005)
    # Verify hover height and lower-end clearance before opening the gripper.
    HOVER_CLEAR = 0.020
    item_r = ITEM_R_CIRC.get(item_usd, float(np.hypot(item_size[0], item_size[1])) / 2.0)
    radial_slack = CUP_MOUTH_R - item_r
    # Add cross offsets only with sufficient clearance and always retain 3mm margin; tight 004 automatically
    # uses only the center. Keep candidate count <200 to avoid place.py:131 random_downsample discarding
    # carefully chosen poses. Current count: one upright family * nine yaws * two depths * five offsets=90.
    # The <=2mm lateral jitter was inherited from center targeting and has modest individual value, but
    # all elements are acceptable. With random residuals, more candidates mean more retries, so restore five points.
    safe_off = min(0.002, max(0.0, radial_slack - 0.003))
    offsets_xy = [(0.0, 0.0)]
    if safe_off > 1e-4:
        offsets_xy += [(-safe_off, 0.0), (safe_off, 0.0),
                       (0.0, -safe_off), (0.0, safe_off)]

    # Use small yaw alternatives within the configured range for near-axially
    # symmetric objects.
    PLACE_YAWS = (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0)

    # Retain the live post-grasp pose; see rationale (1) above.
    item_live_rot = quat2mat_wxyz(item_rot)
    item_axis_w = item_live_rot @ item_axis_local
    item_axis_w /= np.linalg.norm(item_axis_w) + 1e-12
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(float(item_axis_w[2])), 0.0, 1.0))))
    # Tilt budget for the shallowest depth, diagnostic/warning only. Do not return here: item_in_cup
    # checks actual containment. This log helps attribute failures to tilt versus position.
    budget_deg = float(np.degrees(np.arctan2(max(radial_slack, 0.0), min(tip_depths))))
    print(f"[DIAG_PLACE] item_r={item_r:.4f} mouth_r={CUP_MOUTH_R:.4f} radial clearance={radial_slack*1000:.1f}mm "
          f"safe_off={safe_off:.4f} | item long-axis tilt from vertical {tilt_deg:.1f}°,"
          f"shallowest tier ({min(tip_depths)*1000:.0f}mm) tilt budget {budget_deg:.1f}° -> "
          f"{'sufficient' if tilt_deg <= budget_deg else 'insufficient'}")
    if tilt_deg > budget_deg:
        print(f"[WARN][DIAG_PLACE] in-gripper item tilt {tilt_deg:.1f}° exceeds the shallowest tier's {budget_deg:.1f}° budget "
              f"— the lower tip will scrape the rim and likely fail (tip targeting corrects horizontal offset, not insufficient uprightness)")

    item_targets = []
    hover_targets = []
    # Why include an upright reorientation family? With only the live pose, yaw/depth/cross-offset targets
    # all retain the same long-axis tilt and essentially the same wrist orientation. AvoidObs can reject
    # all at once, explaining the deterministic all-or-none pattern in collection (13/14 repeated scenes
    # had the same result). Apply the minimum rotation aligning the current long axis to world Z to the
    # live pose. An analytic minimal rotation preserves axial yaw, unlike a hardcoded absolute orientation,
    # and supplies a distinct wrist orientation to IK while improving insertion geometry.
    def _min_rot_to_up(axis_w):
        """Minimum Rodrigues rotation aligning axis_w to world +Z. Choose the upward end to avoid a 180-degree wrist flip."""
        a = np.asarray(axis_w, dtype=float)
        a = a / (float(np.linalg.norm(a)) + 1e-12)
        if a[2] < 0.0:
            a = -a
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(a, z)
        s = float(np.linalg.norm(v))
        c = float(a @ z)
        if s < 1e-9:                      # Already vertical (or exactly opposite, excluded above).
            return np.eye(3)
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

    # Offer upright and live-pose candidate families to accommodate residual
    # in-gripper tilt without unnecessary rotation.
    base_rots = [item_live_rot, _min_rot_to_up(item_axis_w) @ item_live_rot]
    def _lateral_fit(tr_xy):
        """Return horizontal axis-alignment offset fit from the object center (m); None means lateral fit is impossible.

        Convention: center xy=mouth axis+fit, so center-axis distance=|fit| and
        lower-end-axis distance=|fit+tr_xy|. Do not reverse these. Using -tr*(lim_tip/d)
        places the CENTER at lim_tip and the lower end at d-lim_tip. For 004's original
        0.3mm lim_tip, that put the lower end 4.7-19.4mm away while the formula falsely
        claimed feasibility: a silent filter applied to the wrong point.
        """
        tr_xy = np.asarray(tr_xy, dtype=float)
        d = float(np.linalg.norm(tr_xy))
        lim_tip = max(radial_slack - 0.003, 0.0)          # Maximum lower-end distance from axis; maximum center distance from axis.
                                                          # If lower-end alignment suffices, the center is naturally inside too.
                                                          # If no placement contains both points, discard it.
                                                          # Place the lower end exactly at its lim_tip limit; the center takes the remaining d-lim_tip,
                                                          # using its available lim_com room.
        lim_com = max(CUP_MOUTH_R - 0.005, 0.0)           #
        if d <= lim_tip:
            return -tr_xy                                  #
        if d > lim_tip + lim_com:
            return None                                    #
        return -tr_xy * (1.0 - lim_tip / d)

    rejected = 0
    for base_rot in base_rots:
        for angle in PLACE_YAWS:
            R = base_rot if angle == 0.0 else (
                rotation_matrix(np.radians(angle), np.array([0.0, 0.0, 1.0], dtype=float))[:3, :3]
                @ base_rot
            )
            # Recompute lower-end offset after yaw. An earlier pen implementation positioned before yaw, using
            # the pre-rotation long axis for compensation. Upright items are unaffected, but for tilted items
            # 30-degree world-Z yaw moves the lower end off-axis and cancels the alignment correction.
            ax = np.asarray(R, dtype=float) @ item_axis_local
            ax = ax / (float(np.linalg.norm(ax)) + 1e-12)
            tr = ax * item_half_height
            if tr[2] > 0.0:                    # Choose the downward end.
                tr = -tr
            fit = _lateral_fit(tr[:2])
            if fit is None:
                rejected += 1
                continue
            for tip_depth in tip_depths:
                # Lower-end z = rim - tip_depth; center z = rim - tip_depth - tr[2].
                target_z = cup_top_z - tip_depth - float(tr[2]) - PLACE_STAGE_RELEASE_LIFT
                for dx, dy in offsets_xy:
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    # Center xy = mouth axis + fit offset, keeping both center and lower end inside the cavity projection.
                    pose[:3, 3] = [
                        cup_mouth[0] + float(fit[0]) + dx,
                        cup_mouth[1] + float(fit[1]) + dy,
                        target_z,
                    ]
                    item_targets.append(pose)
            # STEP 3a hover family shares exactly the same R and xy compensation, lifting only the lower end
            # above the rim by HOVER_CLEAR. Share R/tr deliberately: descend with the verified hover wrist
            # orientation, preventing fresh residuals from changing orientation. MoveStage has no PlaceStage
            # +10mm place_transform_up, so do not subtract PLACE_STAGE_RELEASE_LIFT here.
            hover_z = cup_top_z + HOVER_CLEAR - float(tr[2])
            for dx, dy in offsets_xy:
                hpose = np.eye(4)
                hpose[:3, :3] = R
                hpose[:3, 3] = [
                    cup_mouth[0] + float(fit[0]) + dx,
                    cup_mouth[1] + float(fit[1]) + dy,
                    hover_z,
                ]
                hover_targets.append(hpose)
    item_targets = np.array(item_targets)
    hover_targets = np.array(hover_targets)
    # Use the calibrated residual-to-lower-end offset model for the configured
    # item geometry; verify the achieved endpoint before release.
    half_len = float(np.max(item_size)) / 2.0
    budgets = ", ".join(
        f"{d * 1000:.0f}mm tier" for d in tip_depths
    )
    print(f"[DIAG_PLACE] {len(item_targets)} insertion candidates (2 pose families (live+upright) × "
          f"{len(PLACE_YAWS)} yaws × {len(tip_depths)} depths × {len(offsets_xy)} offsets,"
          f"lateral pinch constraint rejected {rejected} poses)"
          f"(must be <200, otherwise randomly subsampled by place.py:131)")
    # Use this empirical scale reference for diagnostics only; the live hover
    # and containment checks determine acceptance.
    print(f"[DIAG_PLACE] lower-tip offset ≈ pos_diff + {half_len * 1000:.0f}mm·sin(angle_diff); "
          f"radial clearance={radial_slack * 1000:.1f}mm (run7 disproved the empirical 20~27mm boundary; dimensional reference only); "
          f"tiers={budgets}")
    print(f"[DIAG_PLACE] conversion reference: residual 10°→{half_len * 1000 * np.sin(np.radians(10)):.0f}mm, "
          f"20°→{half_len * 1000 * np.sin(np.radians(20)):.0f}mm, "
          f"30°→{half_len * 1000 * np.sin(np.radians(30)):.0f}mm (plus pos_diff)")

    # Use an airborne-holder stationary-object threshold during placement
    # rather than the framework's default table-object threshold.
    CUP_KNOCK_POS_M = 0.030   #
    CUP_KNOCK_ANG_DEG = 25.0  #
    cup_pose_before_place = robot.get_object_pose(cup_id)

    def cup_not_knocked(verbose=True):
        """Detect holder knockover with thresholds for airborne holding, rather than the framework 5-degree stationary limit."""
        if cup_pose_before_place[0] is None or cup_pose_before_place[1] is None:
            return True
        cp_now, cq_now = robot.get_object_pose(cup_id)
        if cp_now is None or cq_now is None:
            return True
        d_pos = float(np.linalg.norm(np.asarray(cp_now, dtype=float)
                                     - np.asarray(cup_pose_before_place[0], dtype=float)))
        # Compute axis-angle from the relative rotation matrices; subtracting quaternion components can turn sign flips into false 180-degree errors.
        r_rel = quat2mat_wxyz(cq_now).T @ quat2mat_wxyz(cup_pose_before_place[1])
        cos_a = (float(np.trace(r_rel)) - 1.0) / 2.0
        d_ang = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        ok = d_pos < CUP_KNOCK_POS_M and d_ang < CUP_KNOCK_ANG_DEG
        if verbose:
            print(f"[DIAG_SETTLE] holder displacement={d_pos * 1000:.1f}mm (required < {CUP_KNOCK_POS_M * 1000:.0f}mm) | "
                  f"holder rotation={d_ang:.1f}° (required < {CUP_KNOCK_ANG_DEG:.0f}°) not knocked out of alignment={ok}")
        return bool(ok)

    # Log item pose before hover, before release and after settling to
    # distinguish transport slip from rim impact.
    _settle_prev = {}

    def _drop_forensics(ip_now):
        """Add landing forensics to DIAG_SETTLE. Diagnostic only; exceptions must never affect the verdict."""
        try:
            ip_now = np.asarray(ip_now, dtype=float)
            g = np.asarray(robot.get_ee_pose(arm=place_arm), dtype=float).reshape(4, 4)
            d_ee = float(np.linalg.norm(ip_now - g[:3, 3]))
            d_start = float(np.linalg.norm(ip_now[:2] - np.asarray(item_pos, float)[:2]))
            if item_pos_prepick is None:
                d_home = float("nan")
            else:
                d_home = float(np.linalg.norm(ip_now[:2] - item_pos_prepick[:2]))
            prev = _settle_prev.get("p")
            d_move = float("nan") if prev is None else float(np.linalg.norm(ip_now - prev))
            _settle_prev["p"] = ip_now.copy()
            return (f"[DIAG_DROP] distance from gripper center={d_ee * 1000:.0f}mm (expected ~10mm when held) | "
                    f"distance from place start={d_start * 1000:.0f}mm distance from original grasp location={d_home * 1000:.0f}mm | "
                    f"displacement between checks={d_move * 1000:.1f}mm (NaN=first check) | "
                    f"gripper center z={float(g[2, 3]):.4f} item center z={float(ip_now[2]):.4f} tabletop z={table_top_z:.4f}")
        except Exception as _e:                                        # noqa: BLE001
            return f"[DIAG_DROP] evidence collection failed ({_e})"

    # Success means the item's lower end is really inside the live holder mouth. Checking the
    # lower tip rather than the AABB centre remains valid after a slender item leans in the bore.
    def item_in_cup(verbose=True):
        ip, iq = robot.get_object_pose(item_id)
        cp, cq = robot.get_object_pose(cup_id)
        if ip is None or iq is None or cp is None or cq is None:
            return False
        isz = robot.get_object_size_in_world_frame(item_id)
        if isz is None:
            return False
        live_mouth_normal = quat2mat_wxyz(cq) @ mouth_normal_local
        live_mouth_normal /= np.linalg.norm(live_mouth_normal) + 1e-12
        live_mouth = np.asarray(cp, dtype=float) + live_mouth_normal * (CUP_DEPTH / 2.0)
        item_axis = quat2mat_wxyz(iq) @ item_axis_local
        item_axis /= np.linalg.norm(item_axis) + 1e-12
        item_tip_rel = item_axis * item_half_height
        if item_tip_rel[2] > 0.0:
            item_tip_rel = -item_tip_rel
        item_tip = np.asarray(ip, dtype=float) + item_tip_rel
        rim_z = float(live_mouth[2])
        tip_depth = rim_z - float(item_tip[2])
        inside_depth = tip_depth > 0.005
        tip_dist_xy = np.linalg.norm(item_tip[:2] - live_mouth[:2])
        center_dist_xy = np.linalg.norm(np.asarray(ip[:2]) - live_mouth[:2])
        # Evaluate the live endpoint, radial clearance and tilt for the hover gate.
        inside_xy = tip_dist_xy < CUP_MOUTH_R
        # Tilt gate: a genuinely contained item must satisfy holder geometry. tilt_limit_deg classifies its
        # measured posture: entirely submerged, including the upper end, allows the diameter-only bound
        # (about 56 degrees for flat erasers); protruding items use atan(diameter/depth)=35.9 degrees.
        # Pass tip_depth so the function checks full containment directly.
        item_tilt_max = tilt_limit_deg(2.0 * item_half_height, tip_depth)
        item_tilt = float(np.degrees(np.arccos(np.clip(abs(float(item_axis[2])), 0.0, 1.0))))
        upright_enough = item_tilt < item_tilt_max
        if verbose:
            # Log each measured component and threshold so failures can be attributed to depth, lateral offset, or tilt.
            print(f"[DIAG_SETTLE] tip_depth={tip_depth:.4f}m (required > 0.005m) depth={inside_depth} | "
                  f"tip_dist_xy={tip_dist_xy:.4f}m (required < {CUP_MOUTH_R:.4f}m) horizontal={inside_xy} | "
                  f"tilt={item_tilt:.1f}° (required < {item_tilt_max:.1f}°) upright={upright_enough} | "
                  f"center_dist_xy={center_dist_xy:.4f}m item_size={np.round(isz, 4).tolist()}")
            print(_drop_forensics(ip))
        return bool(inside_depth and inside_xy and upright_enough)

    # Remove obstacles again before insertion. STEP 2 removes them before each grasp retry, but a
    # successful grasp also triggers Fast obstacle update, restoring both held holder and lifted item.
    # Placement then commands the lower end into a cavity registered as an obstacle; item/wall collision
    # makes AvoidObs reject every candidate. With use_near_point disabled, place.py continues without
    # release. Failed collection episodes stopped at substage 0 (movement), never substage 1 (opening);
    # repeat scenes failed consistently, indicating a persistent obstacle rather than random jitter.
    # Use obstacle_prims at collider-mesh level. Entity paths fail silently while the return value stays True;
    # see STEP 2's path-matching explanation.
    robot.remove_objs_from_obstacle(obstacle_prims(item_prim) + obstacle_prims(cup_prim))

    # STEP 3a: move to the hover target and verify the item's endpoint, radial
    # clearance and tilt before release. Recover according to the failed gate.
    HOVER_TIP_MARGIN = 0.0
    HOVER_EE_SLACK = 0.025          # Allowed increase over initial center-to-TCP distance (m); exceeding it means the item slipped out.

    def _ee_center():
        g = np.asarray(robot.get_ee_pose(arm=place_arm), dtype=float).reshape(4, 4)
        return g[:3, 3]

    # Use a mutable container: regrasp changes the center-to-TCP relationship, requiring a fresh reference.
    # item_ready_over_mouth is a closure; capturing an immutable float would retain only the first reference.
    _ref = {"d_ee": float("nan")}

    def _refresh_ee_ref():
        try:
            ip_now, _ = robot.get_object_pose(item_id)
            _ref["d_ee"] = float(np.linalg.norm(np.asarray(ip_now, dtype=float) - _ee_center()))
        except Exception as _e:                                        # noqa: BLE001
            _ref["d_ee"] = float("nan")
            print(f"[DIAG_HOVER] gripper center unavailable ({_e}); release criterion disabled for this trial (other two gates remain active)")

    _refresh_ee_ref()
    # During hover the item is above the mouth and cannot be entirely submerged, so use conservative
    # TILT_MAX_DEG. Omitting tip_depth deliberately makes tilt_limit_deg return that conservative value.
    HOVER_TILT_MAX = TILT_MAX_DEG - HOVER_TILT_MARGIN
    print(f"[DIAG_HOVER] {len(hover_targets)} hover candidates; lower tip stops {HOVER_CLEAR * 1000:.0f}mm above the mouth;"
          f"reference item center to gripper center={_ref['d_ee'] * 1000:.0f}mm;"
          f"thresholds: tilt<{HOVER_TILT_MAX:.1f}° "
          f"lower-tip distance from axis<{(max(radial_slack - HOVER_TIP_MARGIN, 0.001)) * 1000:.1f}mm")

    # Record which gate vetoed the last attempt. Fix 1 uses it to choose recovery: a dropped item needs
    # regrasp; an unsuitable orientation/landing point needs repositioning.
    _gate = {"tilt": True, "tip": True, "in_hand": True}

    def item_ready_over_mouth(verbose=True):
        """Gate after hover and before opening, using measured live poses; see the geometric derivation above."""
        ip, iq = robot.get_object_pose(item_id)
        cp, cq = robot.get_object_pose(cup_id)
        if ip is None or iq is None or cp is None or cq is None:
            return False
        live_mouth_normal = quat2mat_wxyz(cq) @ mouth_normal_local
        live_mouth_normal /= np.linalg.norm(live_mouth_normal) + 1e-12
        live_mouth = np.asarray(cp, dtype=float) + live_mouth_normal * (CUP_DEPTH / 2.0)
        axis = quat2mat_wxyz(iq) @ item_axis_local
        axis /= np.linalg.norm(axis) + 1e-12
        tip_rel = axis * item_half_height
        if tip_rel[2] > 0.0:
            tip_rel = -tip_rel
        tip = np.asarray(ip, dtype=float) + tip_rel
        tip_xy = float(np.linalg.norm(tip[:2] - live_mouth[:2]))
        com_xy = float(np.linalg.norm(np.asarray(ip[:2], dtype=float) - live_mouth[:2]))
        tilt = float(np.degrees(np.arccos(np.clip(abs(float(axis[2])), 0.0, 1.0))))
        d_ee_ref = float(_ref["d_ee"])
        try:
            d_ee = float(np.linalg.norm(np.asarray(ip, dtype=float) - _ee_center()))
        except Exception:                                              # noqa: BLE001
            d_ee = float("nan")
        # If distance is NaN (TCP or reference unavailable), pass this gate; diagnostic failure must not reject a valid grasp.
        in_hand = True if (np.isnan(d_ee) or np.isnan(d_ee_ref)) else (d_ee <= d_ee_ref + HOVER_EE_SLACK)
        tilt_ok = tilt < HOVER_TILT_MAX
        tip_ok = tip_xy < max(radial_slack - HOVER_TIP_MARGIN, 0.001)
        _gate.update(tilt=bool(tilt_ok), tip=bool(tip_ok), in_hand=bool(in_hand))
        if verbose:
            print(f"[DIAG_HOVER] tilt={tilt:.1f}° "
                  f"(required < {HOVER_TILT_MAX:.1f}) upright={tilt_ok} | "
                  f"lower-tip distance from axis={tip_xy * 1000:.1f}mm "
                  f"(required < {(max(radial_slack - HOVER_TIP_MARGIN, 0.001)) * 1000:.1f}) fits={tip_ok} | "
                  f"item center distance from axis={com_xy * 1000:.1f}mm | "
                  f"item center to gripper center={d_ee * 1000:.0f}mm (reference {d_ee_ref * 1000:.0f}+"
                  f"{HOVER_EE_SLACK * 1000:.0f}) still held={in_hand}")
        return bool(tilt_ok and tip_ok and in_hand)

    hover_ok = False
    # Choose recovery from the failed hover condition: retry placement for
    # alignment errors, or regrasp when the item is no longer held.
    HOVER_ROUNDS = 3

    def rebuild_hover_targets():
        """Rebuild hover candidates from the current live pose (orientation families * yaw * cross offsets), using the first-round geometry."""
        ip_r, iq_r = robot.get_object_pose(item_id)
        cp_r, cq_r = robot.get_object_pose(cup_id)
        if ip_r is None or iq_r is None or cp_r is None or cq_r is None:
            return np.zeros((0, 4, 4))
        n_r = quat2mat_wxyz(cq_r) @ mouth_normal_local
        n_r /= np.linalg.norm(n_r) + 1e-12
        mouth_r = np.asarray(cp_r, dtype=float) + n_r * (CUP_DEPTH / 2.0)
        live_rot = quat2mat_wxyz(iq_r)
        ax_r = live_rot @ item_axis_local
        ax_r /= np.linalg.norm(ax_r) + 1e-12
        out = []
        for base_rot in (live_rot, _min_rot_to_up(ax_r) @ live_rot):
            for angle in PLACE_YAWS:
                Rr = base_rot if angle == 0.0 else (
                    rotation_matrix(np.radians(angle), np.array([0.0, 0.0, 1.0], dtype=float))[:3, :3]
                    @ base_rot
                )
                axr = np.asarray(Rr, dtype=float) @ item_axis_local
                axr = axr / (float(np.linalg.norm(axr)) + 1e-12)
                trr = axr * item_half_height
                if trr[2] > 0.0:
                    trr = -trr
                fitr = _lateral_fit(trr[:2])
                if fitr is None:
                    continue
                hz = float(mouth_r[2]) + HOVER_CLEAR - float(trr[2])
                for dx, dy in offsets_xy:
                    hp = np.eye(4)
                    hp[:3, :3] = Rr
                    hp[:3, 3] = [float(mouth_r[0]) + float(fitr[0]) + dx,
                                 float(mouth_r[1]) + float(fitr[1]) + dy, hz]
                    out.append(hp)
        return np.array(out) if out else np.zeros((0, 4, 4))

    for _hround in range(HOVER_ROUNDS):
        if len(hover_targets) == 0:
            print("[WARN][DIAG_HOVER] no usable hover candidates this round (pinch constraint rejected all poses); stopping retries")
            break
        if _hround > 0:
            print(f"[DIAG_HOVER] hover round {_hround + 1}, {len(hover_targets)} candidates")
        # Reset gate state to all-pass each round. Movement can fail during planning before any checker call;
        # without reset it would reuse stale results and incorrectly regrasp. All-True state routes such failures
        # to rebuilding candidates and repositioning, the appropriate planning recovery.
        _gate.update(tilt=True, tip=True, in_hand=True)
        hover_ok = robot.move_actor_to_target_pose(
            actor=item_id,
            arm=place_arm,
            target_pose=hover_targets,
            move_config=MoveConfig(
                # Continue protecting the stationary table. Exclude the holder as in placement: normal reaction
                # causes 8-12-degree airborne jitter; see cup_not_knocked's derivation.
                unmovable_objects=[table_id],
                extra_checkers=[item_ready_over_mouth],
                action_description={
                    "zh": f"把{item_zh}移到笔筒上方",
                    "en": f"moves the {item_en} above the pen holder",
                },
            ),
        )
        if hover_ok:
            break
        if _hround == HOVER_ROUNDS - 1:
            break
        if not _gate["in_hand"]:
            # Regrasp an item that has dropped outside the gripper before retrying placement.
            if rung_cursor >= len(item_rungs):
                print("[WARN][DIAG_HOVER] hover check detected a dropped item, but all grasp tiers are exhausted; no further regrasp")
                break
            print(f"[WARN][DIAG_HOVER] hover check detected a dropped item — regrasp from tier {rung_cursor + 1} onward")
            _re_ok, rung_cursor = try_pick_item(rung_cursor)
            if not _re_ok:
                print("[WARN][DIAG_HOVER] regrasp failed; abandoning hover")
                break
            _refresh_ee_ref()
        else:
            print(f"[DIAG_HOVER] hover rejected (tilt_ok={_gate['tilt']} tip_ok={_gate['tip']}) "
                  f"— rebuilding candidates from the current live pose and trying again")
        hover_targets = rebuild_hover_targets()

    print(f"[DIAG_HOVER] hover verification {'passed' if hover_ok else 'failed (recovery exhausted; falling back to direct insertion with broad candidates)'}")

    # After hover passes, descend only from the verified live pose. The final motion before opening is
    # about 35mm of nearly vertical translation with negligible wrist change, limiting new residuals.
    # The old single move from lift to mouth changed both position and orientation, generating those residuals.
    descent_targets = []
    if hover_ok:
        ip_h, iq_h = robot.get_object_pose(item_id)
        cp_h, cq_h = robot.get_object_pose(cup_id)
        if ip_h is not None and iq_h is not None and cp_h is not None and cq_h is not None:
            n_h = quat2mat_wxyz(cq_h) @ mouth_normal_local
            n_h /= np.linalg.norm(n_h) + 1e-12
            mouth_h = np.asarray(cp_h, dtype=float) + n_h * (CUP_DEPTH / 2.0)
            R_h = quat2mat_wxyz(iq_h)
            ax_h = R_h @ item_axis_local
            ax_h /= np.linalg.norm(ax_h) + 1e-12
            tr_h = ax_h * item_half_height
            if tr_h[2] > 0.0:
                tr_h = -tr_h
            tip_h = np.asarray(ip_h, dtype=float) + tr_h
            for tip_depth in tip_depths:
                # Measure current lower-end height and derive descent needed to reach tip_depth below the mouth.
                drop = float(tip_h[2]) - (float(mouth_h[2]) - tip_depth)
                if drop <= 0.0:
                    continue
                base = np.eye(4)
                base[:3, :3] = R_h
                base[:3, 3] = np.asarray(ip_h, dtype=float)
                # (a) Pure vertical descent, with unchanged horizontal position.
                p = base.copy()
                p[2, 3] -= drop + PLACE_STAGE_RELEASE_LIFT
                descent_targets.append(p)
                # (b) Also realign horizontally with _lateral_fit, within tens of mm. Aligning only the lower end would
                # push a tilted item's center outside the cavity projection and make it fall outward after release.
                fit_h = _lateral_fit(np.asarray(tr_h[:2], dtype=float))
                if fit_h is not None:
                    q = p.copy()
                    q[0, 3] = float(mouth_h[0]) + float(fit_h[0])
                    q[1, 3] = float(mouth_h[1]) + float(fit_h[1])
                    descent_targets.append(q)
    print(f"[DIAG_HOVER] {len(descent_targets)} descent candidates"
          f"(pure vertical + lower-tip recentering) × {len(tip_depths)} depths")

    item_place_config = PlaceConfig(
        # Place from above with the task's orientation and containment checks;
        # disable the default filter that excludes these intended drop poses.
        disable_upside_down=False,
        # Exclude cup_id deliberately: the framework's stationary-object 5-degree limit is inappropriate
        # for an airborne holder and consumed one-third of measured retries, including the best 5.9-degree
        # residual. cup_not_knocked uses 30mm/25 degrees instead. Keep the genuinely stationary table_id.
        unmovable_objects=[table_id],
        extra_checkers=[item_in_cup, cup_not_knocked],
        action_description={
            "zh": f"把{item_zh}放进笔筒里",
            "en": f"places the {item_en} into the pen holder",
        },
    )
    # Hover completion triggers another Fast obstacle update, restoring held holder and item to the world.
    # Insertion must enter that cavity, so AvoidObs rejects every candidate if it remains an obstacle,
    # as in episodes stopping at substage 0 before release. Remove obstacles again before insertion.
    # Expand to collider-mesh paths; entity paths silently fail while the returned status remains True.
    robot.remove_objs_from_obstacle(obstacle_prims(item_prim) + obstacle_prims(cup_prim))
    item_success = False
    if len(descent_targets) > 0:
        item_success = robot.place_actor_to_target_pose(
            actor=item_id,
            arm=place_arm,
            target_pose=np.array(descent_targets),
            place_config=item_place_config,
        )
        print(f"[DIAG_HOVER] short descent insertion {'succeeded' if item_success else 'failed; falling back to broad candidates'}")
    if not item_success:
        # Fallback to the original broad candidate set after failed hover or descent. Its success rate is low,
        # but it remains the available escape route while the item is still physically held.
        robot.remove_objs_from_obstacle(obstacle_prims(item_prim) + obstacle_prims(cup_prim))
        item_success = robot.place_actor_to_target_pose(
            actor=item_id,
            arm=place_arm,
            target_pose=item_targets,
            place_config=item_place_config,
        )
    # get the placing arm out of the way before the cup is set back down.
    robot.back_to_origin(arm=place_arm)

    # STEP 4: return the held container and contents to the table with a
    # level plain-placement target slightly above the support surface. Keep
    # planner attachment/exclusion bookkeeping consistent for the held group.
    robot.attach_obj([cup_prim, item_prim], arm=hold_arm)

    cup_pos_held, _ = robot.get_object_pose(cup_id)
    # cup_level_rot (the cup's original flat-on-table rotation) was computed in STEP 1b.
    # Release the cup a small clearance above its original flush resting centre cup_pos0[2].
    # Keeping the target strictly above the table keeps the pose collision-free so AvoidObs
    # solves it in one shot (fewer failed retries = less wobble), and the cup only free-falls
    # this clearance -- gentle, no press, contents stay.
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
        cs = _cup_world_size()          # Height is unscaled, but always use this helper to keep a single holder-size calculation path.
        if cs is None:
            return False
        cup_bottom_z = float(cp[2]) - float(cs[2]) / 2.0
        if verbose:
            print(f"[DIAG_SETTLE] cup_bottom_z={cup_bottom_z:.4f} table_top_z={table_top_z:.4f} "
                  f"difference={abs(cup_bottom_z - table_top_z) * 1000:.1f}mm (required < 30mm)")
        return abs(cup_bottom_z - table_top_z) < 0.03

    cup_place_config = PlaceConfig(
        disable_upside_down=False,
        passive_primitive="default",
        unmovable_objects=[table_id],
        extra_checkers=[cup_back_on_table],
        action_description={"zh": "把笔筒放回桌面", "en": "sets the pen holder back on the table"},
    )
    cup_success = robot.place_actor_to_target_pose(
        actor=cup_id,
        arm=hold_arm,
        target_pose=cup_targets,
        place_config=cup_place_config,
    )
    # release the attached cup+item from curobo now that the cup is placed.
    robot.detach_obj()

    # FINAL settle gate. place extra_checkers only fire at the release instant; a tall
    # bottle can still topple out of a shallow cup just after (the framework only re-settles
    # and re-checks when the FIRST check fails, so a late fall-out would otherwise pass as
    # SUCCESS). Let physics settle, then independently re-confirm the cup is on the table AND
    # the item is still in it.
    robot.wait(0.6)
    # verbose=True logs both components' measurements and thresholds. Compute both checks before combining;
    # short-circuit `and` would hide the second measurement when the first fails.
    cup_ok = cup_back_on_table(verbose=True)
    item_ok = item_in_cup(verbose=True)
    settle_ok = cup_ok and item_ok
    if not settle_ok:
        print(f"Set-down settle check FAILED (cup_on_table={cup_ok}, item_in_cup={item_ok})")

    # Episode ends with both arms home. The PLACE arm was already sent home at the end of
    # STEP 3, so reset ONLY the HOLD arm here -- calling back_to_origin() with no arm resets
    # left THEN right, which re-commands the already-home place arm and makes it twitch a
    # few extra times at the very end.
    robot.back_to_origin(arm=hold_arm)

    return item_success and cup_success and settle_ok
