# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""plug_charging_cable_into_phone_g2_crsb_90d: insert a charging cable into the phone charging port.

Seven steps plus FINAL-CHECK:

  Step 1: left phone pick.
    Left arm grasps benchmark_phone_001 in body-local frame:
      local xyz=(0, 0, 0.04495), wxyz=(0, -0.70711, -0.70711, 0).
    GraspConfig.grasp_pose_frame="relative" interprets the custom point in object-local coordinates.

  Step 2: move phone hole to static world target.
    Left hand moves body/hole to STATIC_HOLE_TARGET_WORLD_XYZ/WXYZ, facing forward at the chest.

  Step 3: right plug pick.
    Right arm grasps benchmark_plug_017 in body-local frame:
      local xyz=(0, 0, 0), wxyz=(0.70711, 0, 0, 0.70711).

  Step 4: move plug body to static world target.
    Move to STATIC_PLUG_TARGET_WORLD_XYZ/WXYZ, aligned with hole in XZ,
    separated only by 0.15m in Y before insertion.

  Step 4.5: right prealignment.
    Move plug 3cm along world -Y to hole_pos-[0, 0.03, 0] for a feasible insertion path.

  Step 5: right insertion along Y only.
    from_current_pose=True, goal_offset=[0, +0.02, 0, 1, 0, 0, 0],
    path_constraint=[1, 1, 1, 1, 0, 1]: constrain xyz and rx/rz as specified,
    allowing Y insertion; advance 2cm along world +Y.
    The plug's phone_plug_attractor.py BehaviorScript compares plug-local tip
    (-0.0213, 0, 0) with phone anchor, hole + phone-local (0, 0, 0.005).
    Per-axis distance below attractor.thresh (default 5mm) creates a permanent fixed joint.

  Step 6: right retreat.
    open_gripper(arm="right", opened_width=1.0), then back_to_origin(arm="right").
    The plug stays in the phone through the fixed joint.

  FINAL-CHECK:
    plug_tip_world=plug_pos+R(plug_quat)@(-0.0213, 0, 0).
    phone_anchor_world=hole_pos+R(hole_quat)@(0, 0, 0.005).
    Fixed attachment makes the points coincide; |diff|<5mm passes. The reference frame
    matches the attractor's pre-attachment threshold calculation.
    PLUG_LOCAL_TIP/PHONE_ANCHOR_OFFSET must match self.plug_local_point and
    self.phone_anchor_local_offset in phone_plug_attractor.py.

Recording deformable cable motion:
    The plug carries a volume soft-body cable at entity/part_01/visual with
    PhysxDeformableBodyAPI. Server _snapshot_cloth_states reads the rendered part from
    SoftBodyView.get_nodal_positions()[0] each frame:
    nodal[n_col:n_col+n_render], world coordinates after PhysX skinning, matching the
    asset points attribute size. It writes deformable/visual/*.npz.
    Replay cloth_replay.py removes deformable/collision schemas, retains asset render
    topology, converts world to local each frame, and writes mesh.GetPointsAttr().Set.
    Hydra vertex count matches the asset's 2414, avoiding invalid points buffer errors.

Nominal robot initialization is (-0.9, 0, 0); see task_info.json. Scene generation randomizes
translation within its ranges, so code must not hardcode the base position.
"""

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import (
    RobotAPI,
    GraspConfig,
    MoveConfig,
)
import numpy as np


# ---------------------------------------------------------------------------
# Specified constants.
# ---------------------------------------------------------------------------

# Step 1: grasp points in object body local frame.
LEFT_PICK_LOCAL_XYZ = np.array([0.0, 0.0, 0.04495])
LEFT_PICK_LOCAL_WXYZ = np.array([0.0, -0.70711, -0.70711, 0.0])

RIGHT_PICK_LOCAL_XYZ = np.array([0.0, 0.0, 0.0])
RIGHT_PICK_LOCAL_WXYZ = np.array([0.70711, 0.0, 0.0, 0.70711])

# Steps 2/3: nominal chest-front plug/body targets and phone-hole prealignment.
RIGHT_CHEST_XYZ = np.array(
    [-0.15, -0.05, 1.05]
)
RIGHT_CHEST_WXYZ_CANDIDATES = np.array(
    [
        [0.7071067811865476, 0.0, 0.0, -0.7071067811865475],
    ]
)
HOLE_IN_PLUG_LOCAL_XYZ = np.array([-0.1, 0.0, 0.0])
HOLE_TARGET_WORLD_WXYZ = np.array([0.5, -0.5, 0.5, 0.5])
PLUG_TARGET_LOCAL_X_OFFSETS = np.array([0.0])

# Step 2 position sweep: preserve specified orientation exactly to avoid reversing the plug;
# sweep position only. X stays in [-0.20, -0.05], 0.70-0.85m from the robot;
# sweep Z +/-10cm and let IK choose Y.
HOLE_Y_SWEEP = np.array([-0.05, 0.0, 0.05])
HOLE_X_SWEEP = np.linspace(-0.20, -0.05, 4)
HOLE_Z_SWEEP = np.linspace(0.95, 1.15, 5)

# Step 5: plug-tip position relative to plug body; match
# assets/objects/benchmark/plug/benchmark_plug_017/phone_plug_attractor.py
# self.plug_local_point = Gf.Vec3d(-0.0213, 0.0, 0.0) in that script.
# BehaviorScript tests distance from plug-local (-0.0213, 0, 0) to the world hole;
# move the body so the tip reaches the hole and triggers its attractor.
PLUG_TIP_IN_PLUG_LOCAL_XYZ = np.array([-0.0213, 0.0, 0.0])

# Static debug positions: edit here; draw them at startup.
# Step 2 phone-hole target world pose.
STATIC_HOLE_TARGET_WORLD_WXYZ = np.array([0.5, -0.5, 0.5, 0.5])
# Step 4 plug-body target world pose.
# STATIC_PLUG_TARGET_WORLD_XYZ = np.array([-0.15, -0.15, 0.9])
STATIC_PLUG_TARGET_WORLD_XYZ= np.array([-0.26, -0.15, 0.86])
STATIC_HOLE_TARGET_WORLD_XYZ = np.array([STATIC_PLUG_TARGET_WORLD_XYZ[0], STATIC_PLUG_TARGET_WORLD_XYZ[1] +0.15, STATIC_PLUG_TARGET_WORLD_XYZ[2]])

STATIC_PLUG_TARGET_WORLD_WXYZ = np.array(
    [6.123233995736766e-17, -0.7071067811865476, 0.7071067811865475, 4.329780281177466e-17]
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pose_from_xyz_wxyz(xyz, wxyz) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from world (xyz, wxyz)."""
    T = np.eye(4)
    T[:3, :3] = quat2mat_wxyz(np.asarray(wxyz, dtype=float))
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def _find_object_id(scene_info, keyword: str):
    """Return first layout id whose `keywords` contain `keyword`."""
    for oid, info in scene_info["layout"].items():
        if keyword in info.get("keywords", []):
            return oid
    return None


def _draw_grasp_debug_cube(robot, actor_id, local_pose, name, color):
    """Draw one local-frame grasp pose as a world-frame debug cube."""
    actor_pos, actor_quat = robot.get_object_pose(actor_id)
    actor_pose = _pose_from_xyz_wxyz(actor_pos, actor_quat)
    world_pose = actor_pose @ local_pose
    _draw_world_debug_cube(robot, world_pose, name, color, size=0.015)


def _draw_world_debug_cube(robot, world_pose, name, color, size=0.02):
    """Draw a world-frame target pose without affecting task execution."""
    try:
        robot.draw_debug_cube(
            name=name,
            pose_4x4=world_pose,
            size=size,
            color=color,
        )
        print(f"[DEBUG-CUBE] {name} world_pos={world_pose[:3, 3].tolist()}")
    except Exception as exc:
        # Debug visualization must not make the actual task fail.
        print(f"[DEBUG-CUBE] failed to draw {name}: {exc}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def execute_task(robot: RobotAPI):
    scene_info = robot.scene_info

    description_json = {
        "task_name": 'Plug a charging cable into a phone',
        "english_task_name": 'Plug a charging cable into a phone',
        "init_scene_text": "Two adjacent tables in front of the robot. The left table holds a phone in a phone holder. The right table holds a plug next to a power-strip storage box.",
    }
    robot.client.set_task_task_basic_info(description_json)

    # ------------------------------------------------------------------
    # Locate scene objects via keywords (matches LLM_RESULT.py keywords).
    # ------------------------------------------------------------------
    phone_id = _find_object_id(scene_info, "target_phone")
    plug_id = _find_object_id(scene_info, "target_plug")
    holder_id = _find_object_id(scene_info, "phone_holder")
    box_id = _find_object_id(scene_info, "power_strip_storage_box")
    left_table_id = _find_object_id(scene_info, "left_table")
    right_table_id = _find_object_id(scene_info, "right_table")

    if phone_id is None or plug_id is None:
        print(f"[ERROR] could not locate phone/plug: phone={phone_id} plug={plug_id}")
        return False

    print(f"phone={phone_id} plug={plug_id} holder={holder_id} box={box_id}")
    print(f"left_table={left_table_id} right_table={right_table_id}")

    static_objects = []

    # ==================================================================
    # Draw Step 2 hole and Step 4 plug static targets initially for visual debugging.
    # Positions are declared as STATIC_* constants at the top.
    # ==================================================================
    static_hole_target_pose = _pose_from_xyz_wxyz(
        STATIC_HOLE_TARGET_WORLD_XYZ, STATIC_HOLE_TARGET_WORLD_WXYZ
    )
    static_plug_target_pose = _pose_from_xyz_wxyz(
        STATIC_PLUG_TARGET_WORLD_XYZ, STATIC_PLUG_TARGET_WORLD_WXYZ
    )
    # _draw_world_debug_cube(
    #     robot,
    #     static_hole_target_pose,
    #     name="phone_move_target_hole",
    #     color=[1.0, 0.0, 1.0],
    #     size=0.015,
    # )
    # _draw_world_debug_cube(
    #     robot,
    #     static_plug_target_pose,
    #     name="plug_pre_align_target",
    #     color=[0.0, 1.0, 1.0],
    #     size=0.015,
    # )
    print(f"[STATIC] hole target xyz={STATIC_HOLE_TARGET_WORLD_XYZ.tolist()} wxyz={STATIC_HOLE_TARGET_WORLD_WXYZ.tolist()}")
    print(f"[STATIC] plug target xyz={STATIC_PLUG_TARGET_WORLD_XYZ.tolist()} wxyz={STATIC_PLUG_TARGET_WORLD_WXYZ.tolist()}")

    # ==================================================================
    # Step 1: left hand grasps phone separately.
    # ==================================================================
    # Custom grasp points use relative frame, in object body local coordinates.
    left_grasp_local = _pose_from_xyz_wxyz(
        LEFT_PICK_LOCAL_XYZ, LEFT_PICK_LOCAL_WXYZ
    )
    right_grasp_local = _pose_from_xyz_wxyz(RIGHT_PICK_LOCAL_XYZ, RIGHT_PICK_LOCAL_WXYZ)
    right_grasp_candidates = np.stack(
        [right_grasp_local], axis=0
    )

    # _draw_grasp_debug_cube(
    #     robot,
    #     phone_id,
    #     left_grasp_local,
    #     name="phone_grasp_point",
    #     color=[1.0, 0.0, 0.0],
    # )
    # _draw_grasp_debug_cube(
    #     robot,
    #     plug_id,
    #     right_grasp_candidates[0],
    #     name="plug_grasp_point",
    #     color=[0.0, 1.0, 0.0],
    # )
    # Keep only candidates rotated 180 degrees about the grasp point's local Z.

    left_grasp_cfg = GraspConfig(
        grasp_pose=left_grasp_local[np.newaxis, ...],           # (1, 4, 4)
        grasp_pose_frame="relative",                           # local to phone body
        unmovable_objects=static_objects + ([plug_id] if plug_id else []),
        action_description={"zh": "抓取手机", "en": "picks up the phone"},
        pick_up_distance=0.10,
    )
    right_grasp_cfg = GraspConfig(
        grasp_pose=right_grasp_candidates,                     # (1, 4, 4)
        grasp_pose_frame="relative",                           # local to plug body
        # The left hand intentionally moved the phone to the chest; exclude it from unmovable objects,
        # otherwise plug-pick movement checking would reject the deliberate phone motion.
        unmovable_objects=static_objects,
        action_description={"zh": "抓取插头", "en": "picks up the plug"},
        pick_up_distance=0.10,
        grasp_offset=0.005,
        disable_upside_down=False,
        flip_grasp=False
    )

    success = robot.pick_actor(
         actor=phone_id,
         arm="left",
         grasp_config=left_grasp_cfg,
     )
    if not success:
        print("[ERROR] step 1 left phone pick failed")
        return False

    # ==================================================================
    # Step 2: first move the phone hole to its nominal prealigned world pose.
    # ==================================================================
    # DEBUG SKIP: temporarily skip phone-hole movement, proceed to Step 3 plug pick and
    # Steps 4/5 near nominal_hole_pose to test right-arm target IK.
    nominal_plug_targets = np.stack(
        [
            _pose_from_xyz_wxyz(RIGHT_CHEST_XYZ, wxyz)
            for wxyz in RIGHT_CHEST_WXYZ_CANDIDATES
        ],
        axis=0,
    )

    hole_local_h = np.array([*HOLE_IN_PLUG_LOCAL_XYZ, 1.0])
    nominal_hole_positions = np.stack(
        [(target @ hole_local_h)[:3] for target in nominal_plug_targets],
        axis=0,
    )
    if not np.allclose(nominal_hole_positions, nominal_hole_positions[0]):
        print("[ERROR] plug target candidates do not share the same local-X hole target")
        return False

    nominal_hole_pose = _pose_from_xyz_wxyz(
        nominal_hole_positions[0], HOLE_TARGET_WORLD_WXYZ
    )

    # Generate hole target candidates by translating along world Y with fixed orientation.
    # Let stage.py's left-arm IK filter select a reachable Y.
    hole_target_candidates = []
    base_x = nominal_hole_positions[0][0]
    base_z = nominal_hole_positions[0][2]
    for y_hole in HOLE_Y_SWEEP:
        pos = np.array([base_x, y_hole, base_z])
        hole_target_candidates.append(
            _pose_from_xyz_wxyz(pos, HOLE_TARGET_WORLD_WXYZ)
        )
    hole_target_candidates = np.stack(hole_target_candidates, axis=0)  # (N, 4, 4)

    # Magenta: original hole world pose (Y=-0.037), a visual reference.
    # _draw_world_debug_cube(
    #     robot,
    #     nominal_hole_pose,
    #     name="phone_move_target_hole",
    #     color=[1.0, 0.0, 1.0],
    #     size=0.015,
    # )

    left_move_cfg = MoveConfig(
        unmovable_objects=static_objects + [plug_id],
        action_description={
            "zh": "把手机孔移动到插头预对准位置",
            "en": "moves the phone hole to the plug pre-alignment pose",
        },
        extra_params={
            "skip_check": True,
        },
    )

    # ==================================================================
    # Step 2: move phone hole to the static target drawn at startup.
    # ==================================================================
    print("[STEP2] moving hole to static target xyz :", static_hole_target_pose[:3, 3].tolist())

    left_move_cfg = MoveConfig(
        unmovable_objects=static_objects + [plug_id],
        action_description={
            "zh": "把手机孔移动到插头预对准位置",
            "en": "moves the phone hole to the plug pre-alignment pose",
        },
        extra_params={
            "skip_check": True,
        },
    )

    success = robot.move_actor_to_target_pose(
        actor=phone_id, target_pose=static_hole_target_pose[np.newaxis, ...],
        arm="left", move_config=left_move_cfg, part="body/hole",
    )
    if not success:
        print("[WARN] step 2 left phone move failed — continue Step 3 + sweep to collect pairs reachable by both arms")

    # After Step 2 IK succeeds, print achieved phone-body and hole positions.
    post_phone_pos, post_phone_quat = robot.get_object_pose(phone_id)
    post_hole_pos, post_hole_quat = robot.get_object_pose(
        phone_id, articulation_part="body/hole"
    )
    print("[STEP2-POST] phone body xyz :", np.asarray(post_phone_pos).tolist(),
          " wxyz:", np.asarray(post_phone_quat).tolist())
    print("[STEP2-POST] hole world xyz :", np.asarray(post_hole_pos).tolist(),
          " wxyz:", np.asarray(post_hole_quat).tolist())

    # ==================================================================
    # Step 3: once the phone settles, right hand grasps plug separately.
    # ==================================================================
    # Record pre-pick plug world position for a -Y 15cm retreat target.
    _preplug_pos, _ = robot.get_object_pose(plug_id)
    preplug_pos_for_retreat = np.asarray(_preplug_pos) + np.array([0.0, -0.15, 0.0])
    print(f"[STEP3-PRE] plug pos={np.asarray(_preplug_pos).tolist()}, "
          f"retreat target (plug -0.15 Y) = {preplug_pos_for_retreat.tolist()}")

    success = robot.pick_actor(
        actor=plug_id,
        arm="right",
        grasp_config=right_grasp_cfg,
    )
    if not success:
        print("[ERROR] step 3 right plug pick failed")
        return False

    # ==================================================================
    # Dual-arm X/Z alignment sweep: hole X=plug X, hole Z=plug Z, only Y differs
    # by 0.1m. Require both arms' IK; printed results can become STATIC_* targets.
    # ==================================================================
    # Get each arm's static T_actor_ee, fixed after pick.
    _phone_pos, _phone_quat = robot.get_object_pose(phone_id)
    _phone_mat = _pose_from_xyz_wxyz(_phone_pos, _phone_quat)
    _left_ee = robot.get_ee_pose(arm="left")
    T_phone_ee_static = np.linalg.inv(_phone_mat) @ _left_ee

    # Static hole pose in phone-body local frame.
    _hole_pos, _hole_quat = robot.get_object_pose(
        phone_id, articulation_part="body/hole"
    )
    _hole_mat = _pose_from_xyz_wxyz(_hole_pos, _hole_quat)
    T_phone_hole_static = np.linalg.inv(_phone_mat) @ _hole_mat

    _plug_pos, _plug_quat = robot.get_object_pose(plug_id)
    _plug_mat = _pose_from_xyz_wxyz(_plug_pos, _plug_quat)
    _right_ee = robot.get_ee_pose(arm="right")
    T_plug_ee_static_pair = np.linalg.inv(_plug_mat) @ _right_ee

    # Sweep X in [-0.30, -0.05] by 0.02 and Z in [0.80, 0.95] by 0.02.
    sweep_x = np.arange(-0.30, -0.049, 0.02)
    sweep_z = np.arange(0.80, 0.951, 0.02)
    pair_left_ee_targets = []
    pair_right_ee_targets = []
    pair_info = []
    for x in sweep_x:
        for z in sweep_z:
            hole_world = _pose_from_xyz_wxyz(
                [x, STATIC_HOLE_TARGET_WORLD_XYZ[1], z], STATIC_HOLE_TARGET_WORLD_WXYZ
            )
            plug_world = _pose_from_xyz_wxyz(
                [x, STATIC_PLUG_TARGET_WORLD_XYZ[1], z], STATIC_PLUG_TARGET_WORLD_WXYZ
            )
            # Infer phone-body target: phone_world = hole_world @ inv(T_phone_hole_static).
            phone_world = hole_world @ np.linalg.inv(T_phone_hole_static)
            left_ee_target = phone_world @ T_phone_ee_static
            right_ee_target = plug_world @ T_plug_ee_static_pair
            pair_left_ee_targets.append(left_ee_target)
            pair_right_ee_targets.append(right_ee_target)
            pair_info.append((float(x), float(z)))

    pair_left = np.stack(pair_left_ee_targets, axis=0)
    pair_right = np.stack(pair_right_ee_targets, axis=0)
    left_ok, _ = robot.solve_ik(pair_left, arm="left", ee_type="gripper", type="Simple")
    right_ok, _ = robot.solve_ik(pair_right, arm="right", ee_type="gripper", type="Simple")
    left_arr = np.asarray(left_ok)
    right_arr = np.asarray(right_ok)
    both = left_arr & right_arr
    print(f"[PAIR-SWEEP] {len(pair_info)} candidates: left pass={int(left_arr.sum())}, right pass={int(right_arr.sum())}, BOTH pass={int(both.sum())}")
    if int(both.sum()) > 0:
        print("[PAIR-SWEEP] BOTH-arms reachable (X, Z) — choose one as STATIC:")
        for i, (x, z) in enumerate(pair_info):
            if both[i]:
                hole_y = STATIC_HOLE_TARGET_WORLD_XYZ[1]
                plug_y = STATIC_PLUG_TARGET_WORLD_XYZ[1]
                print(f"  X={x:+.3f} Z={z:.3f}  hole=({x:+.3f}, {hole_y:+.3f}, {z:.3f})  plug=({x:+.3f}, {plug_y:+.3f}, {z:.3f})")
    else:
        # Print reachable positions for each arm to aid diagnosis.
        print("[PAIR-SWEEP] no BOTH-arm reachable pair. left-only sample:")
        left_only = [pair_info[i] for i in range(len(pair_info)) if left_arr[i] and not right_arr[i]]
        for x, z in left_only[:20]:
            print(f"  X={x:+.3f} Z={z:.3f}")
        print("[PAIR-SWEEP] right-only sample:")
        right_only = [pair_info[i] for i in range(len(pair_info)) if right_arr[i] and not left_arr[i]]
        for x, z in right_only[:20]:
            print(f"  X={x:+.3f} Z={z:.3f}")

    # ==================================================================
    # Step 4: vary plug static xyz by +/-0.1, in 0.01 increments, seeking IK solutions.
    # ==================================================================
    # step4_offsets_x = np.arange(-0.05, 0.0501, 0.01)
    # step4_offsets_y = np.arange(-0.05, 0.0501, 0.01)
    # step4_offsets_z = np.arange(-0.1, 0.101, 0.01)
    step4_offsets_x = [0]
    step4_offsets_y = [0]
    step4_offsets_z = [0]
    step4_batch = []
    step4_offset_info = []
    for dx in step4_offsets_x:
        for dy in step4_offsets_y:
            for dz in step4_offsets_z:
                cand = static_plug_target_pose.copy()
                cand[:3, 3] = STATIC_PLUG_TARGET_WORLD_XYZ + np.array([dx, dy, dz])
                step4_batch.append(cand)
                step4_offset_info.append((float(dx), float(dy), float(dz)))
    plug_target_batch = np.stack(step4_batch, axis=0)
    print(f"[STEP4] sweep batch size = {len(step4_batch)} "
          f"(offsets ±0.10, step 0.01) around static xyz "
          f"{STATIC_PLUG_TARGET_WORLD_XYZ.tolist()}")

    # Prefilter manually with client solve_ik(arm='right', type='Simple');
    # report passing offsets to choose a suitable static position.
    # T_plug_ee: static gripper-to-plug-body relative pose after pick.
    pre4_plug_pos, pre4_plug_quat = robot.get_object_pose(plug_id)
    pre4_plug_mat = _pose_from_xyz_wxyz(pre4_plug_pos, pre4_plug_quat)
    pre4_ee_pose = robot.get_ee_pose(arm="right")
    T_plug_ee_static = np.linalg.inv(pre4_plug_mat) @ pre4_ee_pose
    print("[STEP4-IK-DEBUG] pre4 plug world xyz:", np.asarray(pre4_plug_pos).tolist())
    print("[STEP4-IK-DEBUG] pre4 ee world xyz  :", pre4_ee_pose[:3, 3].tolist())
    print("[STEP4-IK-DEBUG] pre4 ee - plug (world diff):",
          (pre4_ee_pose[:3, 3] - np.asarray(pre4_plug_pos)).tolist())
    print("[STEP4-IK-DEBUG] T_plug_ee_static translation (plug-local):",
          T_plug_ee_static[:3, 3].tolist())

    ee_target_batch = plug_target_batch @ T_plug_ee_static[np.newaxis, ...]

    # Print center (dx=dy=dz=0) ee world target for a sanity check.
    center_idx = None
    for i, (dx, dy, dz) in enumerate(step4_offset_info):
        if abs(dx) < 1e-9 and abs(dy) < 1e-9 and abs(dz) < 1e-9:
            center_idx = i
            break
    if center_idx is not None:
        print(f"[STEP4-IK-DEBUG] center candidate (offset 0):")
        print(f"  plug target world xyz = {plug_target_batch[center_idx][:3, 3].tolist()}")
        print(f"  ee   target world xyz = {ee_target_batch[center_idx][:3, 3].tolist()}")

    right_ok, _ = robot.solve_ik(ee_target_batch, arm="right", ee_type="gripper", type="Simple")
    n_pass = int(np.asarray(right_ok).sum())
    print(f"[STEP4-IK-DEBUG] Simple IK pass: {n_pass}/{len(step4_batch)}")
    if n_pass > 0:
        # Print the first 30 passing offsets and world positions.
        pass_indices = [i for i, ok in enumerate(right_ok) if bool(ok)]
        print(f"[STEP4-IK-DEBUG] first {min(30, len(pass_indices))} passing offsets:")
        for i in pass_indices[:30]:
            dx, dy, dz = step4_offset_info[i]
            world_xyz = plug_target_batch[i][:3, 3]
            print(f"  offset=({dx:+.2f},{dy:+.2f},{dz:+.2f})  world_xyz={world_xyz.tolist()}")

    right_move_cfg = MoveConfig(
        unmovable_objects=static_objects,
        action_description={
            "zh": "移动插头对准手机孔",
            "en": "moves the plug into pre-alignment with the phone hole",
        },
        extra_params={
            "skip_check": True,
        },
    )

    success = robot.move_actor_to_target_pose(
        actor=plug_id,
        target_pose=plug_target_batch,
        arm="right",
        move_config=right_move_cfg,
    )
    if not success:
        print("[ERROR] step 4 right plug move failed (all sweep candidates rejected)")
        return False

    # Print achieved stage.py-selected position for updating STATIC_PLUG_TARGET_WORLD_XYZ.
    # Read through multiple paths to diagnose stale poses.
    post4_plug_pos, post4_plug_quat = robot.get_object_pose(plug_id)
    post4_hole_pos, post4_hole_quat = robot.get_object_pose(
        phone_id, articulation_part="body/hole"
    )
    post4_phone_pos, post4_phone_quat = robot.get_object_pose(phone_id)
    post4_ee_right = robot.get_ee_pose(arm="right")
    post4_ee_left = robot.get_ee_pose(arm="left")

    # Infer plug-body world pose from ee world pose and post-pick T_plug_ee_static.
    # With a stable attachment, this equals the physical plug-body world pose.
    predicted_plug_from_ee = post4_ee_right @ np.linalg.inv(T_plug_ee_static)
    reached_offset = np.asarray(post4_plug_pos) - STATIC_PLUG_TARGET_WORLD_XYZ

    print("=== [STEP4-POST] compare pose readings through multiple paths ===")
    print("  ee (right) world xyz            :", post4_ee_right[:3, 3].tolist())
    print("  ee (left)  world xyz            :", post4_ee_left[:3, 3].tolist())
    print("  plug via get_object_pose        :", np.asarray(post4_plug_pos).tolist(),
          "  quat:", np.asarray(post4_plug_quat).tolist())
    print("  plug via ee @ inv(T_plug_ee)    :", predicted_plug_from_ee[:3, 3].tolist(),
          " (large differences from get_object_pose indicate stale get_object_pose readings)")
    print("  phone (entity) via get_object   :", np.asarray(post4_phone_pos).tolist(),
          "  quat:", np.asarray(post4_phone_quat).tolist())
    print("  hole via get_object_pose        :", np.asarray(post4_hole_pos).tolist(),
          "  quat:", np.asarray(post4_hole_quat).tolist())
    print(f"  plug reached offset from static : {reached_offset.tolist()}"
          f" |norm|={float(np.linalg.norm(reached_offset)):.4f}m")
    print(f"  plug target static was          : {STATIC_PLUG_TARGET_WORLD_XYZ.tolist()}")
    print(f"  hole target static was          : {STATIC_HOLE_TARGET_WORLD_XYZ.tolist()}")

    # Wait and read again to check for lazy pose updates.
    robot.wait(1.0)
    post4_plug_pos_wait, _ = robot.get_object_pose(plug_id)
    post4_hole_pos_wait, _ = robot.get_object_pose(
        phone_id, articulation_part="body/hole"
    )
    print("=== [STEP4-POST-WAIT-1s] read again ===")
    print("  plug via get_object_pose (wait) :", np.asarray(post4_plug_pos_wait).tolist())
    print("  hole via get_object_pose (wait) :", np.asarray(post4_hole_pos_wait).tolist())

    # ==================================================================
    # Step 4.5: align plug 3cm behind the hole along world -Y.
    # ==================================================================
    align_hole_pos, _ = robot.get_object_pose(
        phone_id, articulation_part="body/hole"
    )
    align_hole_pos = np.asarray(align_hole_pos)
    # Alignment point: plug body = hole - [0, 0.03, 0].
    align_target_xyz = align_hole_pos - np.array([0.0, 0.03, 0.0])
    align_target_mat = np.eye(4)
    align_target_mat[:3, :3] = quat2mat_wxyz(np.asarray(post4_plug_quat))
    align_target_mat[:3, 3] = align_target_xyz
    print("[STEP4.5] align target plug xyz :", align_target_xyz.tolist(),
          " (hole - [0, 0.03, 0])")

    # Disable the phone in curobo collision checking.
    phone_prim_path = robot.get_object_prim_path(phone_id)
    robot.remove_objs_from_obstacle([phone_prim_path])

    align_move_cfg = MoveConfig(
        unmovable_objects=static_objects,
        action_description={
            "zh": "把插头移到手机充电孔后方两厘米对准",
            "en": "align: move plug 2cm behind the hole along -Y",
        },
        extra_params={
            "skip_check": True,
        },
    )
    success = robot.move_actor_to_target_pose(
        actor=plug_id,
        target_pose=align_target_mat[np.newaxis, ...],
        arm="right",
        move_config=align_move_cfg,
    )
    if not success:
        print("[WARN] step 4.5 align move failed")

    robot.wait(0.5)
    post_align_plug_pos, post_align_plug_quat = robot.get_object_pose(plug_id)
    print("[STEP4.5-POST] plug xyz :", np.asarray(post_align_plug_pos).tolist(),
          " wxyz:", np.asarray(post_align_plug_quat).tolist())

    # ==================================================================
    # Step 5: advance 2cm along world +Y so plug tip reaches the hole.
    # ==================================================================
    post_align_plug_mat = _pose_from_xyz_wxyz(post_align_plug_pos, post_align_plug_quat)

    insert_move_cfg = MoveConfig(
        unmovable_objects=static_objects,
        action_description={
            "zh": "推入插头到手机",
            "en": "inserts the plug into the phone hole",
        },
        path_constraint=[1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        offset_and_constraint_in_goal_frame=False,
        extra_params={
            "from_current_pose": True,
            "goal_offset": [0.0, 0.02, 0.0, 1.0, 0.0, 0.0, 0.0],
            "skip_check": True,
        },
    )
    print("[STEP5] insert plug via from_current_pose + goal_offset [0, +0.02, 0] (world Y)")

    success = robot.move_actor_to_target_pose(
        actor=plug_id,
        target_pose=post_align_plug_mat[np.newaxis, ...],
        arm="right",
        move_config=insert_move_cfg,
    )
    if not success:
        print("[WARN] step 5 plug insertion failed")

    # Allow attractor BehaviorScript time to detect the threshold and create FixedJoint.
    robot.wait(1.5)

    # Print achieved plug position after Step 5.
    post5_plug_pos, post5_plug_quat = robot.get_object_pose(plug_id)
    post5_hole_pos, _ = robot.get_object_pose(phone_id, articulation_part="body/hole")
    print("[STEP5-POST] plug xyz  :", np.asarray(post5_plug_pos).tolist(),
          " wxyz:", np.asarray(post5_plug_quat).tolist())
    print("[STEP5-POST] hole xyz  :", np.asarray(post5_hole_pos).tolist())
    diff = np.asarray(post5_plug_pos) - np.asarray(post5_hole_pos)
    print(f"[STEP5-POST] plug - hole :", diff.tolist(),
          f" |norm|={float(np.linalg.norm(diff)):.4f}m")

    # ==================================================================
    # Step 6: open right hand and return right arm to initial pose while left arm holds.
    # ==================================================================
    print("[STEP6] open right gripper + back_to_origin(right)")
    robot.open_gripper(arm="right", opened_width=1.0)
    robot.wait(0.5)
    robot.back_to_origin(arm="right")
    robot.wait(1.0)

    # ==================================================================
    # Final check: world distance between attractor plug tip and phone anchor.
    # ==================================================================
    # Keep constants synchronized with phone_plug_attractor.py:
    #   plug_local_point         = (-0.0213, 0, 0)   (plug entity local frame)
    # phone_anchor_local_offset=(0, 0, 0.005) in phone-body frame; static hole shares its orientation.
    # After attachment the fixed joint guarantees plug_tip_world == phone_anchor_world, distance zero.
    PLUG_LOCAL_TIP = np.array([-0.0213, 0.0, 0.0])
    PHONE_ANCHOR_OFFSET = np.array([0.0, 0.0, 0.005])
    ATTACH_ATTRACT_DIST_THRESH = 0.005  # Use a slightly relaxed 5mm finalCheck threshold after attachment.

    final_plug_pos, final_plug_quat = robot.get_object_pose(plug_id)
    final_hole_pos, final_hole_quat = robot.get_object_pose(
        phone_id, articulation_part="body/hole"
    )
    plug_rot = quat2mat_wxyz(final_plug_quat)
    hole_rot = quat2mat_wxyz(final_hole_quat)
    plug_tip_world = np.asarray(final_plug_pos) + plug_rot @ PLUG_LOCAL_TIP
    phone_anchor_world = np.asarray(final_hole_pos) + hole_rot @ PHONE_ANCHOR_OFFSET
    final_diff = plug_tip_world - phone_anchor_world
    final_dist = float(np.linalg.norm(final_diff))
    print(f"[FINAL-CHECK] plug tip world  : {plug_tip_world.tolist()}")
    print(f"[FINAL-CHECK] phone anchor    : {phone_anchor_world.tolist()}")
    print(f"[FINAL-CHECK] tip - anchor    : {final_diff.tolist()} |norm|={final_dist:.4f}m")
    if final_dist < ATTACH_ATTRACT_DIST_THRESH:
        print(f"[FINAL-CHECK] PASS: distance {final_dist:.4f}m < {ATTACH_ATTRACT_DIST_THRESH}m")
        return True
    else:
        print(f"[FINAL-CHECK] FAIL: distance {final_dist:.4f}m >= {ATTACH_ATTRACT_DIST_THRESH}m")
        return False
