# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np


# Pick both fruits simultaneously with horizontal grasp candidates, then
# place them above their assigned bowls using reachable wrist orientations.
# Use fruit-in-EE offsets to maintain bowl-centered targets. Keep the
# table as a planner obstacle and exclude the target fruits and bowls
# only during the stages that require contact.


# -- Horizontal grasp filter and depth centering --
# Third homogeneous grasp_pose column (col2) is approach; absolute world z component measures departure from horizontal:
# |z|=0 is fully horizontal, gripper parallel to table; |z|=1 is vertical top-down, fingers toward table.
# Keep roughly horizontal candidates |z| < HORIZ_Z. If fewer than MIN_KEEP satisfy that criterion, retain the
# MIN_KEEP most horizontal (ascending |z|), preserving a nonempty pool with the closest-to-horizontal poses.
HORIZ_Z = 0.5     # Approach deviation from horizontal <=30 degrees (sin30=0.5).
MIN_KEEP = 12

# Shallow-grasp fix: maximum compensation in m moving gripper center toward fruit center along approach.
# Measured six recordings bowls1/2/9/10/11/12, frame_state+state frame by frame: project
# fruit-center minus gripper-center at closure onto approach. **Right arm was consistently underdepth +13.2~15.0 mm in 6/6**,
# fruit ~1.5 cm ahead and orange grasped only at outer edge; left arm -1.2~+2.6 mm, essentially centered. Systematic, not random.
# Root cause: attach has not yet happened during grasp, so **fruit remains a curobo world obstacle**. Deeper grasps move palm
# closer to fruit center and incur stronger collision rejection in move.py solve_ik, leaving only shallow candidates. Framework
# motion_gen_reacher.py:1948-1953 records the same apple-world-obstacle collision causing all IK failures.
# Both fixes are required: remove fruit from obstacle world before STEP 1 (below), and deepen candidates here.
DEPTH_CORR_MAX = 0.025


def _horizontal_grasp_filter(grasp_poses, obj_pose, grasp_widths):
    gp = np.asarray(grasp_poses, dtype=float)
    gw = np.asarray(grasp_widths, dtype=float)
    if gp.shape[0] == 0:
        return gp, gw
    approach_z = np.abs(gp[:, 2, 2])          # World z component of col2.
    mask = approach_z < HORIZ_Z
    if int(mask.sum()) >= MIN_KEEP:
        gp, gw = gp[mask], gw[mask]
    else:
        # Too few strictly horizontal candidates: retain the MIN_KEEP most horizontal; never empty the pool.
        order = np.argsort(approach_z)
        keep = order[: max(MIN_KEEP, int(mask.sum()))]
        gp, gw = gp[keep], gw[keep]

    # Depth centering: translate each candidate along its approach until gripper center aligns with fruit center, compensating exact underdepth.
    # step = clip(underdepth, 0, DEPTH_CORR_MAX): deepen only, never retreat; candidates already beyond center remain unchanged.
    # Cap prevents extreme outside candidates from being pushed through object. Orientation stays unchanged, preserving horizontal grasp
    # (parallel to tabletop, lift start above table) while improving contact depth.
    # move.py independently reorders G2 candidates via sorted_by_position_humanlike; task cannot control its selection.
    # Return **only corrected candidates**, never mix shallow originals back into the pool or framework may select them again.
    # Position changes <=2.5 cm with unchanged orientation keep originally reachable candidates in their nearby IK region.
    obj_c = np.asarray(obj_pose, dtype=float)[:3, 3]
    a = gp[:, :3, 2].copy()
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n < 1e-9] = 1.0
    a = a / n
    along = np.einsum("ij,ij->i", obj_c[None, :] - gp[:, :3, 3], a)
    step = np.clip(along, 0.0, DEPTH_CORR_MAX)
    gp[:, :3, 3] = gp[:, :3, 3] + a * step[:, None]
    return gp, gw


# Offer placement wrist orientations with limited rotation from the
# achieved pickup pose.
def _R_topdown(theta):
    """Family of top-down rotations (col2=[0,0,-1], downward approach), parameterized by yaw theta; verified orthogonal with det=+1."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def _Rx(phi):
    """Rotate by phi about gripper local x to tilt top-down approach sideways/downward and add reachable candidates."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# Use placement tilt candidates from vertical through the configured
# small-angle range.
PLACE_TILTS = (0.0, np.deg2rad(30.0), np.deg2rad(-30.0))

# Top-down yaw samples: 8 -> 12 (30-degree steps), improving resolution so the pick-pose-proximity filter
# can select closer wrist alignment (with eight steps, closest yaw may still differ by 22.5 degrees).
PLACE_YAW_STEPS = 12

# Limit overall placement rotation by geodesic angle from the achieved
# pickup wrist orientation.
MAX_PLACE_REORIENT_DEG = 110.0

# Fallback: if some R_pick leaves too few candidates (theoretically nonempty because R_pick itself has zero rotation),
# fill to this count in ascending geodesic angle. Slight relaxation is preferable to starving the pool and failing all placements,
# as with _horizontal_grasp_filter MIN_KEEP.
MIN_PLACE_ORIS = 6


def _geodesic_deg(A, B):
    """Overall rotation angle in degrees between two rotation matrices: minimum single-axis rotation from A to B.
    Objective whole-gripper rotation metric: 0 unchanged, 90 quarter-turn, 180 full flip."""
    c = (np.trace(np.asarray(A, dtype=float).T @ np.asarray(B, dtype=float)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _place_orientations(R_pick):
    """Placement EE orientation candidates (3x3): multiple top-down yaws times tilt levels, retaining only limited rotation
    from pick pose, plus pick pose itself (zero rotation, minimal translation). move.py filters all via IK,
    then ranks humanlike poses and chooses first. Script can therefore prevent detours only by excluding their orientations."""
    R_pick = np.asarray(R_pick, dtype=float)
    cands = []
    for theta in np.linspace(0.0, 2.0 * np.pi, PLACE_YAW_STEPS, endpoint=False):
        base = _R_topdown(theta)
        for phi in PLACE_TILTS:
            R = base @ _Rx(phi)
            cands.append((_geodesic_deg(R_pick, R), R))
    cands.sort(key=lambda kv: kv[0])                       # Ascending overall rotation angle.
    keep = [R for a, R in cands if a <= MAX_PLACE_REORIENT_DEG]
    if len(keep) < MIN_PLACE_ORIS:                         # Candidate-starvation fallback: fill by angle.
        keep = [R for _, R in cands[:MIN_PLACE_ORIS]]
    kept_angles = [round(_geodesic_deg(R_pick, R), 1) for R in keep]
    print(f"[DIAG] place wrist poses: {len(keep)}/{len(cands)} pass the ≤{MAX_PLACE_REORIENT_DEG:.0f}° rotation limit"
          f", rotation angles={kept_angles}")
    keep.append(R_pick)                                    # Fallback: preserve horizontal pick wrist pose (zero rotation).
    return keep


def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_apple_000": {"has_interactions": True, "object_type": "apple"},
        "benchmark_apple_001": {"has_interactions": True, "object_type": "apple"},
        "benchmark_apple_002": {"has_interactions": True, "object_type": "apple"},
        "benchmark_apple_003": {"has_interactions": True, "object_type": "apple"},
        "benchmark_orange_001": {"has_interactions": True, "object_type": "orange"},
        "benchmark_orange_002": {"has_interactions": True, "object_type": "orange"},
        "benchmark_orange_004": {"has_interactions": True, "object_type": "orange"},
        "benchmark_bowl_000": {"has_interactions": True, "object_type": "bowl"},
        "benchmark_bowl_001": {"has_interactions": True, "object_type": "bowl"},
        "benchmark_bowl_002": {"has_interactions": True, "object_type": "bowl"},
        "benchmark_bowl_003": {"has_interactions": True, "object_type": "bowl"},
        "benchmark_bowl_004": {"has_interactions": True, "object_type": "bowl"},
        "benchmark_bowl_025": {"has_interactions": True, "object_type": "bowl"},
        "table_000": {"has_interactions": False, "object_type": "table"},
    }

    scene_info = robot.scene_info

    description_json = {
        "task_name": "with both arms at the same time, put the apple into the left bowl and the orange into the right bowl",
        "english_task_name": "with both arms at the same time, put the apple into the left bowl and the orange into the right bowl",
        "init_scene_text": "The robot is in front of a table with an apple, an orange, and two bowls (one on the left and one on the right)",
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

    # --- locate bowls, apple, orange ---
    bowls = []
    apple_id = None
    orange_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        if obj_id == table_id:
            continue
        if "benchmark_bowl" in usd:
            pos, _ = robot.get_object_pose(obj_id)
            bowls.append({"id": obj_id, "position": pos})
        elif "benchmark_apple" in usd:
            apple_id = obj_id
        elif "benchmark_orange" in usd:
            orange_id = obj_id

    if apple_id is None or orange_id is None or len(bowls) < 2:
        print("Required objects not found (need apple, orange, 2 bowls)")
        return False

    # assign bowls to sides by world y (robot at -x facing +x => +y is left)
    bowls_sorted = sorted(bowls, key=lambda b: b["position"][1])
    right_bowl = bowls_sorted[0]
    left_bowl = bowls_sorted[-1]
    left_bowl_id = left_bowl["id"]
    right_bowl_id = right_bowl["id"]

    apple_size = robot.get_object_size_in_world_frame(apple_id)
    orange_size = robot.get_object_size_in_world_frame(orange_id)
    if apple_size is None or orange_size is None:
        print("Cannot get fruit sizes")
        return False

    protect = [table_id, left_bowl_id, right_bowl_id]

    # Set release height from the fruit's lower surface to the bowl rim,
    # accounting for the configured placement-height alternatives.
    RELEASE_GAP = 0.015   # fruit bottom above bowl rim at release

    def make_checker(fid, bid, fruit_size):
        def fruit_in_bowl():
            cur_pos, _ = robot.get_object_pose(fid)
            cur_bowl_pos, _ = robot.get_object_pose(bid)
            if cur_pos is None or cur_bowl_pos is None:
                return False
            cur_bowl_size = robot.get_object_size_in_world_frame(bid)
            if cur_bowl_size is None:
                return False
            dist_xy = np.linalg.norm(np.asarray(cur_pos[:2]) - np.asarray(cur_bowl_pos[:2]))
            bowl_radius = 0.5 * max(float(cur_bowl_size[0]), float(cur_bowl_size[1])) + 0.02
            if dist_xy > bowl_radius:
                return False
            cur_bowl_top_z = float(cur_bowl_pos[2]) + float(cur_bowl_size[2]) / 2.0
            return float(cur_pos[2]) <= cur_bowl_top_z + max(0.12, float(fruit_size[2]) * 1.5)
        return fruit_in_bowl

    def _fruit_in_ee(arm, fid):
        """Fruit-center position p_local in EE-local coordinates.
        Decouples desired fruit world position from chosen wrist orientation: given R and f_world,
        EE world position = f_world - R @ p_local, assuming constant p_local under rigid attachment."""
        ee = robot.get_ee_pose(arm=arm)
        fp, _ = robot.get_object_pose(fid)
        if ee is None or fp is None:
            return np.zeros(3)
        R = np.asarray(ee[:3, :3])
        t = np.asarray(ee[:3, 3])
        return R.T @ (np.asarray(fp) - t)

    def build_place_targets(fruit_id, bowl_id, arm, drop_gap, dz_list, p_local, R_pick):
        """Placement EE target pool: fruit above bowl center (bowl XY + drop_gap + dz height).

        For every wrist R and height fallback dz, set EE_pos = f_world - R @ p_local.
        All orientations then center fruit over the bowl; wrist orientation affects reachability, not landing.
        _place_orientations supplies top-down multiple yaws/small tilts filtered to rotation
        <=MAX_PLACE_REORIENT_DEG from pick, plus pick-pose fallback. Height dz lets higher targets
        substitute for unreachable lower ones. Framework solve_ik filters the entire pool to retain
        poses reaching side/rear bowls.
        """
        bowl_pos, _ = robot.get_object_pose(bowl_id)
        bowl_size = robot.get_object_size_in_world_frame(bowl_id)
        f_size = robot.get_object_size_in_world_frame(fruit_id)
        if bowl_pos is None or bowl_size is None or f_size is None:
            return None
        bowl_top_z = float(bowl_pos[2]) + float(bowl_size[2]) / 2.0
        base_fruit_z = bowl_top_z + float(f_size[2]) / 2.0 + drop_gap

        poses = []
        for R in _place_orientations(R_pick):
            R = np.asarray(R, dtype=float)
            for dz in dz_list:
                f_world = np.array([float(bowl_pos[0]), float(bowl_pos[1]), base_fruit_z + dz])
                ee_pos = f_world - R @ np.asarray(p_local, dtype=float)
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = ee_pos
                poses.append(pose)
        return np.array(poses)

    # Bowl prim paths (removed from the obstacle world just before placing — see STEP 2).
    left_bowl_prim = robot.get_object_prim_path(left_bowl_id)
    right_bowl_prim = robot.get_object_prim_path(right_bowl_id)
    # Exclude the held fruit prims during placement contact and restore
    # planner obstacle state afterward.
    apple_prim = robot.get_object_prim_path(apple_id)
    orange_prim = robot.get_object_prim_path(orange_id)

    # STEP 1: exclude the target fruits from planner obstacles before DualPick
    # so target contact remains plannable. Keep physical contact active,
    # retain the table obstacle and defer bowl exclusions until placement.
    robot.remove_objs_from_obstacle([apple_prim, orange_prim])

    # Table + bowls all stay as curobo obstacles. The bowls sit far to the side-back
    # (out of the arm's pick/lift workspace), so keeping them as obstacles doesn't
    # cramp the pick. The grasp_pose_filter keeps only near-horizontal (table-parallel)
    # grasps, so the post-grasp gripper sits at the fruit's equator height (fingers
    # sideways, not down): the lift step's start state clears the tabletop and no
    # longer trips "Start or End state in collision".
    left_cfg = GraspConfig(
        unmovable_objects=protect,
        grasp_pose_filter=_horizontal_grasp_filter,
        action_description={"zh": "横向抓起苹果", "en": "grasps the apple sideways"},
    )
    right_cfg = GraspConfig(
        unmovable_objects=protect,
        grasp_pose_filter=_horizontal_grasp_filter,
        action_description={"zh": "横向抓起橙子", "en": "grasps the orange sideways"},
    )
    pick_ok, _ = robot.dual_act(
        left=DualPick(actor=apple_id, grasp_config=left_cfg),
        right=DualPick(actor=orange_id, grasp_config=right_cfg),
        action_description={"zh": "双臂同时横向抓起苹果和橙子", "en": "both arms grasp the apple and the orange sideways at the same time"},
    )
    if not pick_ok:
        print("Dual pick failed")
        robot.back_to_origin("")
        return False

    robot.wait(0.2)

    # Measure fruit EE-local offset p_local after grasp to lock later landing point, plus current horizontal wrist pose
    # R_pick as placement fallback. Measure both after completed pick and before release to reflect actual grip relation.
    off_L = _fruit_in_ee("left", apple_id)
    off_R = _fruit_in_ee("right", orange_id)
    ee_L = robot.get_ee_pose(arm="left")
    ee_R = robot.get_ee_pose(arm="right")
    Rpick_L = np.asarray(ee_L[:3, :3]) if ee_L is not None else np.eye(3)
    Rpick_R = np.asarray(ee_R[:3, :3]) if ee_R is not None else np.eye(3)
    print(f"[DIAG] fruit_in_ee L={np.round(off_L,3)} R={np.round(off_R,3)}")

    # STEP 2: place using DualMoveToPose with live bowl and fruit geometry.
    # Choose candidates within the configured wrist-rotation and clearance limits.
    robot.remove_objs_from_obstacle([left_bowl_prim, right_bowl_prim, apple_prim, orange_prim])

    # Placement pool: reachable wrist orientations (top-down multiple yaws/small tilts, large rotations from pick filtered out,
    # plus horizontal fallback) times dz height fallbacks. Framework filters the whole pool through IK to reach side/rear bowls;
    # p_local locks fruit directly above bowl center.
    # DZ_LIST upper bound reduced 0.12 -> 0.07, and lowest level to 0 to prioritize near-rim release. Retain 0.07
    # as reachability fallback if low targets are unreachable, with maximum release only ~8.5 cm above rim.
    DZ_LIST = [0.0, 0.02, 0.04, 0.07]
    apple_place = build_place_targets(apple_id, left_bowl_id, "left", RELEASE_GAP, DZ_LIST, off_L, Rpick_L)
    orange_place = build_place_targets(orange_id, right_bowl_id, "right", RELEASE_GAP, DZ_LIST, off_R, Rpick_R)
    if apple_place is None or orange_place is None:
        print("Cannot build place targets")
        robot.back_to_origin("")
        return False

    left_release = MoveConfig(
        gripper_action="open",
        unmovable_objects=protect,
        action_description={"zh": "把苹果放进左边的碗", "en": "puts the apple into the left bowl"},
    )
    right_release = MoveConfig(
        gripper_action="open",
        unmovable_objects=protect,
        action_description={"zh": "把橙子放进右边的碗", "en": "puts the orange into the right bowl"},
    )
    place_ok, _ = robot.dual_act(
        left=DualMoveToPose(actor="gripper", target_pose=apple_place, move_config=left_release),
        right=DualMoveToPose(actor="gripper", target_pose=orange_place, move_config=right_release),
        action_description={
            "zh": "双臂同时把苹果和橙子放进各自的碗",
            "en": "both arms place the apple and the orange into their bowls at the same time",
        },
        extra_checkers=[
            make_checker(apple_id, left_bowl_id, apple_size),
            make_checker(orange_id, right_bowl_id, orange_size),
        ],
    )

    robot.detach_obj()

    robot.wait(1.0)
    apple_in_bowl = make_checker(apple_id, left_bowl_id, apple_size)()
    orange_in_bowl = make_checker(orange_id, right_bowl_id, orange_size)()

    try:
        _ap, _ = robot.get_object_pose(apple_id); _op, _ = robot.get_object_pose(orange_id)
        _lb, _ = robot.get_object_pose(left_bowl_id); _rb, _ = robot.get_object_pose(right_bowl_id)
        print(f"[DIAG] final apple={np.round(np.asarray(_ap),3)} Lbowl={np.round(np.asarray(_lb),3)} in={apple_in_bowl} | "
              f"orange={np.round(np.asarray(_op),3)} Rbowl={np.round(np.asarray(_rb),3)} in={orange_in_bowl}")
    except Exception as _e:
        print("[DIAG] final failed:", _e)

    robot.back_to_origin("")
    return bool(place_ok and apple_in_bowl and orange_in_bowl)
