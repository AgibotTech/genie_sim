# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np


# One hand lifts a bowl while the other picks and places a fruit inside it,
# then returns the bowl to the table. Use the asset's own grasp annotations
# and verify the grasp, clearance and final containment.
FRUIT_USDS = {
    "benchmark_peach_022",
    "benchmark_lemon_028",
    "benchmark_lemon_027",
    "benchmark_peach_000",
    "benchmark_apple_000",
    "benchmark_apple_003",
    "benchmark_green_apple_000",
    "benchmark_green_apple_001",
    "benchmark_orange_002",
    "benchmark_orange_004",
    "benchmark_pomegranate_001",
    "benchmark_pomegranate_047",
}

# Cap only movable task objects whose source Aligned.usda mass exceeds 0.2 kg.
# The 200 kg table is structural support and must retain its original mass.
MASS_OVERRIDES_KG = {
    "benchmark_bowl_025": 0.2,
    "benchmark_apple_000": 0.2,
    "benchmark_apple_003": 0.2,
    "benchmark_green_apple_000": 0.2,
    "benchmark_orange_002": 0.2,
    "benchmark_orange_004": 0.2,
    "benchmark_pomegranate_001": 0.2,
    "benchmark_pomegranate_047": 0.2,
}

# How far (m) the holding arm lifts the bowl off the desktop. Just enough to read as
# "lifted off the table" while keeping the held bowl and the other gripper in the
# head cameras and within a comfortable reach for the dropping arm.
BOWL_LIFT = 0.05


def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_bowl_025": {"has_interactions": True, "object_type": "bowl", "interaction_type": "container"},
        "table_000": {"has_interactions": False, "object_type": "table"},
    }
    for fid in FRUIT_USDS:
        objects_list[fid] = {"has_interactions": True, "object_type": "fruit"}

    scene_info = robot.scene_info

    description_json = {
        "task_name": 'Place a fruit into a bowl held by the other hand',
        "english_task_name": 'Place a fruit into a bowl held by the other hand',
        "init_scene_text": "The robot is in front of a table with a bowl near the middle and a single fruit beside it",
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

    # --- locate bowl and fruit ---
    bowl_id = None
    bowl_usd = None
    fruit_id = None
    fruit_usd = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        kw = obj_info.get("keywords", [])
        if obj_id == table_id:
            continue
        if "benchmark_bowl" in usd or "bowl" in kw:
            bowl_id = obj_id
            bowl_usd = usd
        elif usd in FRUIT_USDS or "fruit" in kw:
            fruit_id = obj_id
            fruit_usd = usd

    if bowl_id is None or fruit_id is None:
        print("Required objects not found (need 1 bowl, 1 fruit)")
        return False

    # Apply per-task mass overrides before any planning or manipulation. The USD
    # payloads already author physics:mass on each object's /entity prim, so this
    # creates a runtime-stage override without changing the shared asset library.
    for object_id, usd_id in ((bowl_id, bowl_usd), (fruit_id, fruit_usd)):
        target_mass = MASS_OVERRIDES_KG.get(usd_id)
        if target_mass is None:
            continue
        if not robot.set_object_parameter_action(
            prim_path=robot.get_object_prim_path(object_id),
            attribute_key="physics:mass",
            attribute_value=target_mass,
            reference="local",
        ):
            print(f"Failed to set {usd_id} mass to {target_mass} kg")
            return False
        print(f"Set {usd_id} mass to {target_mass} kg")

    # --- resolve which arm HOLDS (lifts) and which arm PLACES, geometrically ---
    # robot at -x facing +x => +y is the robot's LEFT. The bowl sits just to the
    # HOLD side of centre; the fruit sits on the OPPOSITE (place) side. The holding
    # arm is the bowl's side; the placing arm is the fruit's side (no cross-body).
    bowl_pos0, bowl_rot0 = robot.get_object_pose(bowl_id)
    bowl_pos0 = np.asarray(bowl_pos0)
    hold_arm = "left" if bowl_pos0[1] >= 0 else "right"
    place_arm = "right" if hold_arm == "left" else "left"
    hold_sign = 1.0 if hold_arm == "left" else -1.0
    print(f"Bowl on {'left' if hold_arm == 'left' else 'right'}; hold arm = {hold_arm}, place arm = {place_arm}")

    # ============================================================== STEP 1
    # Hold arm picks the bowl UP off the table (BOWL_LIFT) and holds it in the air.
    # passive_primitive="default" uses the bowl's annotated grasp point (the proven
    # stack_bowls_g2 bowl grasp).
    #
    # Grasp the bowl on the HOLD-side edge (the far edge from the other hand) so the
    # opposite half of the bowl mouth is left clear for the fruit drop. The planner
    # calls grasp_pose_filter(grasp_poses, obj_pose, grasp_widths) with the candidate
    # grasps already in the WORLD frame, so we keep only grasps whose contact point
    # sits on the hold side of the bowl centre (+y for the left hand, -y for the
    # right). A fallback returns all candidates if the side bias would prune
    # everything, so the grasp never fails purely from this preference.
    def hold_side_grasp_filter(grasp_poses, obj_pose, grasp_widths):
        bowl_cy = float(obj_pose[1, 3])
        side = hold_sign * (grasp_poses[:, 1, 3] - bowl_cy)   # >0 => on the hold side
        mask = side > 0.0
        if np.count_nonzero(mask) >= 3:
            return grasp_poses[mask], grasp_widths[mask]
        return grasp_poses, grasp_widths

    lift_grasp_config = GraspConfig(
        pick_up_distance=BOWL_LIFT,
        passive_primitive="default",
        grasp_pose_filter=hold_side_grasp_filter,
        unmovable_objects=[table_id],
        action_description={"zh": "端起碗", "en": "lifts up the bowl"},
    )
    if not robot.pick_actor(bowl_id, arm=hold_arm, grasp_config=lift_grasp_config):
        print("Hold arm failed to lift the bowl")
        return False

    # ============================================================== STEP 1b
    # Level the bowl the hold arm just lifted -- get its MOUTH back to horizontal.
    #
    # Why it is needed: the bowl is grasped by its RIM (the annotated bowl grasp, biased
    # to the hold-side edge by hold_side_grasp_filter), so the contact sits well off the
    # bowl's vertical axis. The moment the bowl leaves the table it hangs tilted about
    # that rim contact -- a visibly slanted mouth in the recording, and a slanted mouth
    # also makes STEP 3's fruit drop miss more often.
    #
    # Recipe (proven in straighten_and_place_beverage_g2): ONE planned move, NOT an
    # in-place ConstraintRotate. move_actor_to_target_pose keeps the rigid grasp, maps
    # every candidate OBJECT pose through it, batch-IK-filters them (isaac + cuRobo
    # AvoidObs) and plans a single collision-free arc to a reachable one -- so no wrist
    # flailing. "Level" is exactly bowl_rot0, the rotation the bowl had resting flat on
    # the table (the same reference STEP 4 uses to set it back down). A bowl is symmetric
    # about its vertical axis, so spinning that level rotation about WORLD Z leaves the
    # mouth just as level while handing the planner many reachable wrist angles.
    bowl_level_rot = quat2mat_wxyz(bowl_rot0)
    # The bowl-mouth normal in the bowl's OWN frame: whichever local direction bowl_rot0
    # mapped to world +Z back when the bowl sat flat. Derived, not guessed, so it is right
    # regardless of which local axis this bowl asset happens to use as "up".
    mouth_normal_local = bowl_level_rot.T @ np.array([0.0, 0.0, 1.0])

    def bowl_tilt_deg():
        """Angle (deg) between the held bowl's mouth normal and world +Z. 0 == level."""
        _, rot = robot.get_object_pose(bowl_id)
        if rot is None:
            return None
        n = quat2mat_wxyz(rot) @ mouth_normal_local
        n = n / (np.linalg.norm(n) + 1e-12)
        return float(np.degrees(np.arccos(np.clip(float(n[2]), -1.0, 1.0))))

    # Skip the move when the bowl came up level anyway: the user has repeatedly flagged
    # redundant end-of-episode motion, and a move to where the bowl already is is exactly
    # that. Only actually level it when it is measurably tilted.
    LEVEL_TOL_DEG = 5.0
    # Small extra lift while levelling, so a rim that is currently dipping low cannot
    # scrape the desktop as it swings up to horizontal.
    LEVEL_CLEARANCE = 0.02

    tilt_before = bowl_tilt_deg()
    if tilt_before is None:
        print("Cannot read held-bowl rotation; skipping the levelling move")
    elif tilt_before <= LEVEL_TOL_DEG:
        print(f"Held bowl already level (tilt {tilt_before:.1f} deg); no levelling move needed")
    else:
        bowl_pos_lifted, _ = robot.get_object_pose(bowl_id)
        if bowl_pos_lifted is None:
            print("Cannot read held-bowl position; skipping the levelling move")
        else:
            level_targets = []
            for dx in np.linspace(-0.01, 0.01, 3):
                for dy in np.linspace(-0.01, 0.01, 3):
                    for angle in np.linspace(0, 360, 12, endpoint=False):
                        pose = np.eye(4)
                        pose[:3, :3] = bowl_level_rot
                        pose[:3, 3] = [
                            bowl_pos_lifted[0] + dx,
                            bowl_pos_lifted[1] + dy,
                            bowl_pos_lifted[2] + LEVEL_CLEARANCE,
                        ]
                        level_targets.append(rotate_along_axis(pose, angle, "z", use_local=False))
            level_config = MoveConfig(
                unmovable_objects=[table_id],
                action_description={"zh": "把碗摆正", "en": "levels the bowl"},
            )
            if robot.move_actor_to_target_pose(
                bowl_id, np.array(level_targets), arm=hold_arm, move_config=level_config
            ):
                print(f"Levelled the held bowl: tilt {tilt_before:.1f} -> {bowl_tilt_deg():.1f} deg")
            else:
                # Non-fatal: the bowl is still held and every later step re-reads its live
                # pose, so the episode degrades to the previous (un-levelled) behaviour
                # rather than being thrown away over a cosmetic step.
                print(f"[WARN] levelling move failed; bowl stays tilted {tilt_before:.1f} deg")

    # ============================================================== STEP 2
    # Place arm picks the fruit up. The held bowl and the table are unmovable for
    # this arm so the collision-aware IK avoids the airborne bowl.
    #
    # pick_up_distance=0.05 (not the GraspConfig default 0.12): after grasping, the
    # planner lifts the fruit straight up by this distance and that lifted pose must
    # pass curobo AvoidObs IK. A 0.12 m straight-up lift makes that IK fail almost
    # always here (observed 0-1 of 7-8 poses passing) -- the fruit only needs to clear
    # the table before step 3 re-plans a path to the held bowl, so a small 0.05 m lift
    # (same as the bowl's BOWL_LIFT, which never fails this filter) is enough and far
    # more reliable. Does not affect step 3 (it targets an absolute pose over the bowl).
    fruit_grasp_config = GraspConfig(
        pick_up_distance=0.05,
        unmovable_objects=[table_id, bowl_id],
        action_description={"zh": "拿起水果", "en": "picks up the fruit"},
    )
    if not robot.pick_actor(fruit_id, arm=place_arm, grasp_config=fruit_grasp_config):
        print("Place arm failed to pick up the fruit")
        robot.back_to_origin(arm=place_arm)
        robot.back_to_origin(arm=hold_arm)
        return False

    # ============================================================== STEP 3
    # Place arm drops the fruit INTO the held (airborne) bowl. Re-read the bowl pose
    # NOW -- it is wherever the hold arm lifted it to, not its original table spot --
    # and re-read the fruit's GRASPED rotation so the place candidates preserve it
    # (no-flail recipe).
    bowl_pos, _ = robot.get_object_pose(bowl_id)
    bowl_size = robot.get_object_size_in_world_frame(bowl_id)
    fruit_pos, fruit_rot = robot.get_object_pose(fruit_id)
    fruit_size = robot.get_object_size_in_world_frame(fruit_id)
    if bowl_pos is None or bowl_size is None or fruit_rot is None or fruit_size is None:
        print("Cannot get held-bowl / fruit poses or sizes")
        robot.back_to_origin(arm=place_arm)
        robot.back_to_origin(arm=hold_arm)
        return False

    bowl_top_z = bowl_pos[2] + bowl_size[2] / 2.0
    # release the fruit centred over the held bowl, just above the rim. Build poses
    # from the fruit's CURRENT grasped rotation and sweep only world-Z yaw, so the
    # wrist stays near a natural pose between candidates (no flailing).
    target_z = bowl_top_z + 0.03 + float(fruit_size[2]) / 2.0
    fruit_grasped_rot = quat2mat_wxyz(fruit_rot)
    fruit_targets = []
    for dx in np.linspace(-0.01, 0.01, 3):
        for dy in np.linspace(-0.01, 0.01, 3):
            for angle in np.linspace(0, 360, 12, endpoint=False):
                pose = np.eye(4)
                pose[:3, :3] = fruit_grasped_rot
                pose[:3, 3] = [bowl_pos[0] + dx, bowl_pos[1] + dy, target_z]
                pose = rotate_along_axis(pose, angle, "z", use_local=False)
                fruit_targets.append(pose)
    fruit_targets = np.array(fruit_targets)

    # success = the fruit's centre lands within the bowl mouth (radial xy
    # containment), read live -- mirrors place_fruit_into_bowl.
    def fruit_in_bowl():
        fp, _ = robot.get_object_pose(fruit_id)
        bp, _ = robot.get_object_pose(bowl_id)
        if fp is None or bp is None:
            return False
        bs = robot.get_object_size_in_world_frame(bowl_id)
        if bs is None:
            return False
        dist_xy = np.linalg.norm(np.asarray(fp[:2]) - np.asarray(bp[:2]))
        return dist_xy < bs[0] / 2.0 + 0.02

    fruit_place_config = PlaceConfig(
        pre_insert_offset=0.1,
        unmovable_objects=[table_id, bowl_id],
        extra_checkers=[fruit_in_bowl],
        action_description={"zh": "把水果放进碗里", "en": "places the fruit into the bowl"},
    )
    fruit_success = robot.place_actor_to_target_pose(
        actor=fruit_id,
        arm=place_arm,
        target_pose=fruit_targets,
        place_config=fruit_place_config,
    )
    # get the dropping arm out of the way before the bowl is set back down.
    robot.back_to_origin(arm=place_arm)

    # ============================================================== STEP 4
    # Hold arm sets the bowl back down on the table. Requirements that earlier attempts
    # each broke one of: (1) curobo's collision-aware (AvoidObs) IK must find a solution;
    # (2) no wrist flailing / redundant moves; (3) the bowl ends LEVEL, released gently ON
    # the table -- NOT pushed down into it (which ejects the fruit).
    #
    # (1) IK: after step 3 the fruit sits INSIDE the arm-attached bowl, and curobo reports
    # a permanent bowl<->fruit contact, so every place pose is in-collision. Fix: ATTACH the
    # fruit TOGETHER with the bowl to the holding arm, so curobo treats both as robot-attached
    # bodies (an attached group is not collision-checked against itself). This beats
    # remove_objs_from_obstacle, which did NOT take effect (curobo's collision world is
    # refreshed from the scene during place, re-adding the fruit).
    #
    # (2)+(3) Use a PLAIN place (PlaceStage), NOT an insert. An insert (pre_insert_offset +
    # insert_depth) runs THREE separately-planned AvoidObs moves (pre-insert -> target ->
    # forced descent) -- the three arcs cause repeated swaying, and the final descent is a
    # skip_check CONSTRAINED push that drives the bowl straight into the table with excessive force:
    # the bowl bottoms out, tilts under load, and when the gripper opens the trapped
    # fruit is flung out. A plain place is a SINGLE AvoidObs move to the target then a
    # release, collision-checked the whole way so it never rams the table. Target a single
    # fixed LEVEL orientation (bowl_rot0, the bowl's original flat-on-table rotation; NO yaw
    # sweep -> no wrist twist) at the bowl's original xy-ish spot, with the release point a
    # few mm ABOVE the flush rest height so the bowl free-settles the last bit by gravity
    # instead of being pressed down. Detach, let it settle, then re-check (see below).
    bowl_prim = robot.get_object_prim_path(bowl_id)
    fruit_prim = robot.get_object_prim_path(fruit_id)
    robot.attach_obj([bowl_prim, fruit_prim], arm=hold_arm)

    bowl_pos_held, _ = robot.get_object_pose(bowl_id)
    # bowl_level_rot (the bowl's original flat-on-table rotation) was computed in STEP 1b.
    # Release the bowl a small clearance (SET_DOWN_CLEARANCE) above its original flush
    # resting centre bowl_pos0[2]. Keeping the target strictly above the table keeps the
    # pose collision-free so AvoidObs solves it in one shot (fewer failed retries = less
    # wobble), and the bowl only free-falls this clearance -- gentle, no press, fruit stays.
    SET_DOWN_CLEARANCE = 0.006
    bowl_target_z = float(bowl_pos0[2]) + SET_DOWN_CLEARANCE
    bowl_targets = []
    for dx in np.linspace(-0.01, 0.01, 3):
        for dy in np.linspace(-0.01, 0.01, 3):
            pose = np.eye(4)
            pose[:3, :3] = bowl_level_rot
            pose[:3, 3] = [bowl_pos_held[0] + dx, bowl_pos_held[1] + dy, bowl_target_z]
            bowl_targets.append(pose)
    bowl_targets = np.array(bowl_targets)

    # success = the bowl is back resting on the desktop (near table-top height).
    def bowl_back_on_table():
        bp, _ = robot.get_object_pose(bowl_id)
        if bp is None:
            return False
        bs = robot.get_object_size_in_world_frame(bowl_id)
        if bs is None:
            return False
        bowl_bottom_z = float(bp[2]) - float(bs[2]) / 2.0
        return abs(bowl_bottom_z - table_top_z) < 0.03

    # STRICT fruit-still-inside gate. A fruit resting in the bowl sits within ~0.02-0.06 m
    # of the bowl centre in xy; one that has popped out lands ~0.097 m away (one mouth
    # radius). The looser step-3 gate bs[0]/2 + 0.02 (== ~0.098 m for this bowl) would WRONGLY
    # accept that popped-out fruit -- which is exactly the false "success" the user caught.
    # Require the fruit centre inside the mouth OUTER radius bs[0]/2 (no +0.02 margin).
    def fruit_settled_in_bowl():
        fp, _ = robot.get_object_pose(fruit_id)
        bp, _ = robot.get_object_pose(bowl_id)
        if fp is None or bp is None:
            return False
        bs = robot.get_object_size_in_world_frame(bowl_id)
        if bs is None:
            return False
        dist_xy = np.linalg.norm(np.asarray(fp[:2]) - np.asarray(bp[:2]))
        return dist_xy < float(bs[0]) / 2.0

    # PLAIN place (no pre_insert_offset, no insert_depth) -> single AvoidObs move + release,
    # collision-checked the whole way (never rams the table). disable_upside_down=False keeps
    # the level bowl grasp poses.
    bowl_place_config = PlaceConfig(
        disable_upside_down=False,
        passive_primitive="default",
        unmovable_objects=[table_id],
        extra_checkers=[bowl_back_on_table],
        action_description={"zh": "把碗放回桌面", "en": "sets the bowl back on the table"},
    )
    bowl_success = robot.place_actor_to_target_pose(
        actor=bowl_id,
        arm=hold_arm,
        target_pose=bowl_targets,
        place_config=bowl_place_config,
    )
    # release the attached bowl+fruit from curobo now that the bowl is placed.
    robot.detach_obj()

    # FINAL settle gate. place extra_checkers only fire at the release instant; a fruit can
    # still bounce/roll out just after. Let physics settle, then independently re-confirm the
    # bowl is on the table AND the fruit is still inside it -- so a popped-out fruit is
    # correctly judged a FAILURE instead of the earlier false success.
    robot.wait(0.6)
    settle_ok = bowl_back_on_table() and fruit_settled_in_bowl()
    if not settle_ok:
        print("Set-down settle check FAILED (bowl off table or fruit popped out of bowl)")

    # Episode ends with both arms home. The PLACE arm was already sent home at the end of
    # step 3, so reset ONLY the HOLD arm here. Calling back_to_origin() with no arm resets
    # left THEN right, which re-commands the already-home place arm and makes it twitch a
    # few extra times at the very end (user-reported redundant motion) -- resetting just the
    # hold arm leaves the place arm untouched and already at home.
    robot.back_to_origin(arm=hold_arm)

    return fruit_success and bowl_success and settle_ok
