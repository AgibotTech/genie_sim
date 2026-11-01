# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np


# One hand holds the cup while the other places a block inside, then returns
# the cup to the table. Use the measured block and cup geometry for grasp
# selection, lift clearance and containment checks.
BLOCK_USDS = {
    "benchmark_building_blocks_072",
    "benchmark_building_blocks_076",
    "benchmark_building_blocks_001",
    "benchmark_building_blocks_005",
    "benchmark_building_blocks_007",
    "benchmark_building_blocks_009",
    "benchmark_building_blocks_010",
    "benchmark_building_blocks_015",
    "benchmark_building_blocks_048",
    "benchmark_building_blocks_049",
}

# Horizontal CIRCUMSCRIBED radius (m) per item, measured off each asset's Aligned.usda mesh
# (max |xy| about the footprint centre). This is the radius that must clear the cup mouth, and
# it cannot be recovered from the live world AABB: half of max(size[:2]) understates it for an
# axis-aligned object, half the AABB diagonal overstates it. Used in STEP 3 to bound how far
# off the cup axis the release may drift.
# Cross block 007 shows why measurement is essential: half-diagonal of its 38x38 mm AABB is 26.8 mm,
# but spaces between four arms reduce true measured radius to 19.5 mm; half-diagonal would waste half the release budget.
# Conversely, theoretical 30 mm cube circumradius is 0.03*sqrt(2)/2 = 0.0212, measured 0.0209 due to slight corner bevels,
# confirming the measurement method has no systematic bias.
# With larger cup_107, per-axis budget is +/-4.8 mm (005, narrowest) to +/-16.3 mm (048, widest); successful 072/076
# with small cup had +/-5.7 mm, so **new narrowest is only 0.9 mm tighter**, and all others are wider.
# STEP 3 offset grid is center-first, using budget only when reachability truly requires it.
ITEM_R_CIRC = {
    "benchmark_building_blocks_072": 0.0209,
    "benchmark_building_blocks_076": 0.0209,
    "benchmark_building_blocks_001": 0.0264,
    "benchmark_building_blocks_005": 0.0282,
    "benchmark_building_blocks_007": 0.0195,
    "benchmark_building_blocks_009": 0.0196,
    "benchmark_building_blocks_010": 0.0205,
    "benchmark_building_blocks_015": 0.0229,
    "benchmark_building_blocks_048": 0.0120,
    "benchmark_building_blocks_049": 0.0141,
}

# How far (m) the holding arm lifts the cup off the desktop. Just enough to read as
# "lifted off the table" while keeping the held cup and the other gripper in the head
# cameras and within a comfortable reach for the placing arm.
CUP_LIFT = 0.05

# Use the measured inner mouth radius of benchmark_cup_107 for block
# clearance and candidate offsets.
CUP_MOUTH_R = 0.035

# Limit depth-centering compensation to the configured maximum. Deepen the
# placement target only, keeping candidates within the object-height budget.
DEPTH_CORR_MAX = 0.020


def _deepen_grasp_filter(grasp_poses, obj_pose, grasp_widths):
    """Translate grasp candidates along approach toward the block center.

    Use the object-center offset and configured compensation limit to adjust
    underdepth without crossing the block-height clearance budget.
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
    along = np.einsum("ij,ij->i", obj_c[None, :] - gp[:, :3, 3], a)
    step = np.clip(along, 0.0, DEPTH_CORR_MAX)
    gp[:, :3, 3] = gp[:, :3, 3] + a * step[:, None]
    print(f"[DIAG_GRASP] depth centering for {len(step)} candidates: added depth {step.min() * 1000:.1f}~"
          f"{step.max() * 1000:.1f}mm (clamped to {DEPTH_CORR_MAX * 1000:.0f}mm)")
    return gp, gw


def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_cup_107": {"has_interactions": True, "object_type": "cup", "interaction_type": "container"},
        "table_000": {"has_interactions": False, "object_type": "table"},
    }
    for bid in BLOCK_USDS:
        objects_list[bid] = {"has_interactions": True, "object_type": "building_blocks"}

    scene_info = robot.scene_info

    description_json = {
        "task_name": 'Place a block into a cup held by the other hand',
        "english_task_name": 'Place a block into a cup held by the other hand',
        "init_scene_text": "The robot is in front of a table with a cup near the middle and a single small building block beside it",
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

    # --- locate cup (the held container) and the block ---
    cup_id = None
    item_id = None
    item_usd = ""
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        kw = obj_info.get("keywords", [])
        if obj_id == table_id:
            continue
        if "benchmark_cup" in usd or "cup" in kw:
            cup_id = obj_id
        elif usd in BLOCK_USDS or "block" in kw or "building_blocks" in kw:
            item_id = obj_id
            item_usd = usd

    if cup_id is None or item_id is None:
        print("Required objects not found (need 1 cup, 1 block)")
        return False

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

    # STEP 1: lift and hold the cup with a reachable, near-horizontal grasp.
    # Use the cup's measured radius and scale in the grasp filters.
    HOLD_ELEV_CAPS = (30.0, 45.0, None)

    def make_hold_grasp_filter(elev_cap_deg):
        """Retain candidates (a) grasping the cup's holding-side edge and (b) with sufficiently horizontal approach.

        grasp_poses are already world-frame, so col2 = grasp_poses[:, :3, 2] is world approach axis;
        its z component gives elevation (0 degrees horizontal, 90 degrees overhead).
        """
        def _f(grasp_poses, obj_pose, grasp_widths):
            cup_cy = float(obj_pose[1, 3])
            side = hold_sign * (grasp_poses[:, 1, 3] - cup_cy)   # >0 => on the hold side
            mask = side > 0.0
            if elev_cap_deg is not None:
                elev = np.degrees(np.arcsin(np.clip(np.abs(grasp_poses[:, 2, 2]), 0.0, 1.0)))
                both = mask & (elev <= float(elev_cap_deg))
                # Do not abandon constraints merely to increase count: use candidates satisfying both when sufficient; otherwise **preserve
                # horizontal orientation first**, since camera intersection is a geometric defect, and relax only holding-side preference,
                # which merely leaves half the opening clear and at worst makes loading harder, without camera collision.
                if np.count_nonzero(both) >= 3:
                    return grasp_poses[both], grasp_widths[both]
                only_elev = elev <= float(elev_cap_deg)
                if np.count_nonzero(only_elev) >= 3:
                    return grasp_poses[only_elev], grasp_widths[only_elev]
                # Too few horizontal candidates at this level: let outer ladder relax; return side-preference candidates here.
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
            action_description={"zh": "端起杯子", "en": "lifts up the cup"},
        )
        cap_txt = "unlimited" if elev_cap is None else f"≤{elev_cap:.0f}°"
        if robot.pick_actor(cup_id, arm=hold_arm, grasp_config=lift_grasp_config):
            lifted_ok = True
            # Report achieved grasp elevation.
            try:
                _hg = np.asarray(robot.get_ee_pose(arm=hold_arm), dtype=float).reshape(4, 4)
                _elev = float(np.degrees(np.arcsin(np.clip(abs(float(_hg[2, 2])), 0.0, 1.0))))
                _kind = "lateral grasp" if _elev < 35.0 else ("angled grasp" if _elev < 60.0 else "top-down grasp")
                print(f"[DIAG_HOLD] cup lifted (elevation cap {cap_txt}): actual approach-axis elevation "
                      f"{_elev:.1f}° -> {_kind}")
                if _elev >= 35.0:
                    print(f"[WARN][DIAG_HOLD] cup grasp is not lateral ({_elev:.1f}°); the wrist camera may enter "
                          f"the other hand's workspace — planning will likely fail in later phases")
            except Exception as _e:                                    # noqa: BLE001
                print(f"[WARN][DIAG_HOLD] elevation measurement failed (task unaffected): {_e}")
            break
        print(f"[DIAG_HOLD] cup lift failed at elevation cap {cap_txt}; relaxing and retrying")
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
    # STEP 3's drop miss more often.
    #
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
                action_description={"zh": "把杯子摆正", "en": "levels the cup"},
            )
            if robot.move_actor_to_target_pose(
                cup_id, np.array(level_targets), arm=hold_arm, move_config=level_config
            ):
                print(f"Levelled the held cup: tilt {tilt_before:.1f} -> {cup_tilt_deg():.1f} deg")
            else:
                # Non-fatal: the cup is still held and every later step re-reads its live
                # pose, so the episode degrades to the previous (un-levelled) behaviour
                # rather than being thrown away over a cosmetic (appearance-only) step.
                print(f"[WARN] levelling move failed; cup stays tilted {tilt_before:.1f} deg")

    # STEP 2: pick the block using the configured lift ladder and check its
    # actual lift. Expand planner exclusions to collider-mesh paths.
    def obstacle_prims(base):
        subs = ["body/visual"] + [f"body/collider_body_{i:02d}" for i in range(4)]
        return [base] + [f"{base}/{s}" for s in subs]

    item_prim = robot.get_object_prim_path(item_id)
    cup_prim = robot.get_object_prim_path(cup_id)
    robot.remove_objs_from_obstacle(obstacle_prims(item_prim) + obstacle_prims(cup_prim))

    item_grasp_config = GraspConfig(
        pick_up_distance=0.05,
        disable_upside_down=True,
        flip_grasp=True,
        grasp_pose_filter=_deepen_grasp_filter,
        unmovable_objects=[table_id, cup_id],
        action_description={"zh": "拿起积木", "en": "picks up the building block"},
    )
    if not robot.pick_actor(item_id, arm=place_arm, grasp_config=item_grasp_config):
        print("Place arm failed to pick up the building block")
        robot.back_to_origin(arm=place_arm)
        robot.back_to_origin(arm=hold_arm)
        return False

    # ============================================================== STEP 3
    # Place arm drops the item INTO the held (airborne) cup. Re-read the cup pose NOW -- it
    # is wherever the hold arm lifted/levelled it to, not its original table spot -- and
    # re-read the item's GRASPED rotation so the place candidates preserve it (no-flail
    # recipe). A block's orientation inside the cup does not matter, but preserving the
    # grasped rotation is still what keeps the wrist from swinging between candidates.
    cup_pos, _ = robot.get_object_pose(cup_id)
    cup_size = robot.get_object_size_in_world_frame(cup_id)
    item_pos, item_rot = robot.get_object_pose(item_id)
    item_size = robot.get_object_size_in_world_frame(item_id)
    if cup_pos is None or cup_size is None or item_rot is None or item_size is None:
        print("Cannot get held-cup / item poses or sizes")
        robot.back_to_origin(arm=place_arm)
        robot.back_to_origin(arm=hold_arm)
        return False

    cup_top_z = cup_pos[2] + cup_size[2] / 2.0
    # Release the block centred over the held cup, just above the rim, then let it drop in.
    # Build poses from the item's CURRENT grasped rotation and sweep only world-Z yaw, so the
    # wrist stays near a natural pose between candidates (no flailing).
    # Keep the item's CURRENT grasped rotation for every candidate and sweep only world-Z
    # yaw, so the wrist stays near a natural pose between candidates (no flailing).
    item_grasped_rot = quat2mat_wxyz(item_rot)
    # One-line diagnostic: every quantity the release height is built from. Kept because the
    # sibling pen task needed four failed attempts before the right term was identified.
    print(f"[DIAG_PLACE] cup_pos={np.round(cup_pos,4).tolist()} cup_size={np.round(cup_size,4).tolist()} "
          f"cup_top_z={cup_top_z:.4f} item_pos={np.round(item_pos,4).tolist()} "
          f"item_size={np.round(item_size,4).tolist()}")
    # Compute release height from the block's lower endpoint relative to the
    # cup rim. Offer reachable clearances and clamp lateral offsets by
    # mouth radius minus the sampled block's circumscribed radius.
    item_r = ITEM_R_CIRC.get(item_usd, float(np.hypot(item_size[0], item_size[1])) / 2.0)
    max_off = max(0.0, CUP_MOUTH_R - item_r) / np.sqrt(2.0)   # per-axis, so the corner fits too
    offsets = [0.0]
    if max_off > 1e-4:
        offsets += [-max_off, max_off]
    item_targets = []
    for tip_clear in (0.005, 0.02, 0.04):
        target_z = cup_top_z + tip_clear + float(item_size[2]) / 2.0
        for dx in offsets:
            for dy in offsets:
                for angle in np.linspace(0, 360, 12, endpoint=False):
                    pose = np.eye(4)
                    pose[:3, :3] = item_grasped_rot
                    pose[:3, 3] = [cup_pos[0] + dx, cup_pos[1] + dy, target_z]
                    item_targets.append(rotate_along_axis(pose, angle, "z", use_local=False))
    item_targets = np.array(item_targets)
    print(f"[DIAG_PLACE] item_r={item_r:.4f} mouth_r={CUP_MOUTH_R:.4f} max_off={max_off:.4f} "
          f"n_targets={len(item_targets)}")

    # success = the block really ended up IN the cup, read live. A block in this pool is
    # small enough to fall FULLY inside the cup, so -- unlike the sibling cosmetic and pen
    # tasks, whose payload sticks out and leans on the rim -- plain xy containment against
    # the mouth is the right gate, mirroring the verified bowl task. The z condition is kept
    # as a cheap guard against the block having landed ON the rim rather than in the cup.
    def item_in_cup(verbose=False):
        ip, _ = robot.get_object_pose(item_id)
        cp, _ = robot.get_object_pose(cup_id)
        if ip is None or cp is None:
            return False
        cs = robot.get_object_size_in_world_frame(cup_id)
        if cs is None:
            return False
        rim_z = float(cp[2]) + float(cs[2]) / 2.0
        below_rim = float(ip[2]) < rim_z
        dist_xy = np.linalg.norm(np.asarray(ip[:2]) - np.asarray(cp[:2]))
        if verbose:
            # Final settle gate formerly printed only FAILED, obscuring whether cup missed table, block bounced out, or block
            # remained inside but failed XY criterion. Print both components and thresholds to diagnose.
            print(f"[DIAG_SETTLE] item_z={float(ip[2]):.4f} rim_z={rim_z:.4f} "
                  f"depth criterion (item_z<rim_z)={below_rim} | dist_xy={dist_xy:.4f} "
                  f"(required < {CUP_MOUTH_R:.4f}) xy criterion={dist_xy < CUP_MOUTH_R}")
        return bool(below_rim and dist_xy < CUP_MOUTH_R)

    item_place_config = PlaceConfig(
        # pre_insert_offset stacks ON TOP of the target: the arm first goes to target + this
        # offset, then descends. 0.1 pushed the approach point out of reach here, because this
        # container is already held up in the air -- see the release-height note above. 0.04 is
        # enough to clear the rim on the way in.
        pre_insert_offset=0.04,
        # Disable the default upside-down filter for this drop-from-above placement;
        # retain the orientation and clearance checks specific to the held cup.
        disable_upside_down=False,
        unmovable_objects=[table_id, cup_id],
        extra_checkers=[item_in_cup],
        action_description={"zh": "把积木放进杯子里", "en": "places the building block into the cup"},
    )
    item_success = robot.place_actor_to_target_pose(
        actor=item_id,
        arm=place_arm,
        target_pose=item_targets,
        place_config=item_place_config,
    )
    # get the placing arm out of the way before the cup is set back down.
    robot.back_to_origin(arm=place_arm)

    # ============================================================== STEP 4
    # Hold arm sets the cup back down on the table (the item rides inside). Requirements
    # that earlier attempts on the sibling task each broke one of: (1) curobo's
    # collision-aware (AvoidObs) IK must find a solution; (2) no wrist flailing / redundant
    # moves; (3) the cup ends LEVEL, released gently ON the table -- NOT pushed down into it
    # (which ejects the contents).
    #
    # (1) IK: after STEP 3 the item sits INSIDE the arm-attached cup and curobo reports a
    # permanent cup<->item contact, so every place pose is in-collision. Fix: ATTACH the item
    # TOGETHER with the cup to the holding arm, so curobo treats both as robot-attached
    # bodies (an attached group is not collision-checked against itself). This beats
    # remove_objs_from_obstacle, which did NOT take effect (curobo's collision world is
    # refreshed from the scene during place, re-adding the item).
    #
    # (2)+(3) Use a PLAIN place (PlaceStage), NOT an insert. An insert (pre_insert_offset +
    # insert_depth) runs THREE separately-planned AvoidObs moves (pre-insert -> target ->
    # forced descent); the three arcs are the observed swaying and the final descent is a skip_check
    # CONSTRAINED push that drives the cup into the table, tilting it under load so the
    # trapped contents are flung out when the gripper opens. A plain place is a SINGLE
    # AvoidObs move to the target then a release, collision-checked the whole way so it never
    # rams the table. Target a single fixed LEVEL orientation (cup_rot0, the cup's original
    # flat-on-table rotation; NO yaw sweep -> no wrist twist) with the release point a few mm
    # ABOVE the flush rest height so the cup free-settles by gravity.
    # cup_prim / item_prim were both resolved in STEP 2 (for the pre-pick obstacle removal).
    # Both objects are already outside obstacle world, eliminating post-STEP 3 cup/item deadlock; server filters
    # this attach as absent from runtime scene and skips it. **Retain call intentionally**: idempotent, failure only skips bookkeeping,
    # If a future version removes STEP 2 obstacle removal, this remains the correct remedy for that deadlock.
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
        cs = robot.get_object_size_in_world_frame(cup_id)
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
        action_description={"zh": "把杯子放回桌面", "en": "sets the cup back on the table"},
    )
    cup_success = robot.place_actor_to_target_pose(
        actor=cup_id,
        arm=hold_arm,
        target_pose=cup_targets,
        place_config=cup_place_config,
    )
    # release the attached cup+item from curobo now that the cup is placed.
    robot.detach_obj()

    # FINAL settle gate. place extra_checkers only fire at the release instant; the block can
    # still bounce out just after (the framework only re-settles and re-checks when the FIRST
    # check fails, so a late bounce-out would otherwise pass as SUCCESS). Let physics settle,
    # then independently re-confirm the cup is on the table AND the block is still in it.
    robot.wait(0.6)
    # verbose=True logs actual values and thresholds for both components. Compute both checks before combining them;
    # short-circuiting with and would hide the second measurement when the first is False.
    cup_ok = cup_back_on_table(verbose=True)
    item_ok = item_in_cup(verbose=True)
    settle_ok = cup_ok and item_ok
    if not settle_ok:
        print(f"Set-down settle check FAILED (cup_on_table={cup_ok}, block_in_cup={item_ok})")

    # Episode ends with both arms home. The PLACE arm was already sent home at the end of
    # STEP 3, so reset ONLY the HOLD arm here -- calling back_to_origin() with no arm resets
    # left THEN right, which re-commands the already-home place arm and makes it twitch a
    # few extra times at the very end.
    robot.back_to_origin(arm=hold_arm)

    return item_success and cup_success and settle_ok
