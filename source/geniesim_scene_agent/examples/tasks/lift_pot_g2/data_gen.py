# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np


# Grasp point relative to pot local frame.
LEFT_GRASP_POS = np.array([0.17591, 0.02774, 0.00044])
LEFT_GRASP_QUAT = np.array([0.5, -0.5, -0.5, 0.5])  # wxyz
RIGHT_GRASP_POS = np.array([-0.17591, 0.02774, 0.00044])
RIGHT_GRASP_QUAT = np.array([0.5, 0.5, 0.5, 0.5])  # wxyz

LIFT_HEIGHT = 0.14


def _compute_world_grasp_pose(local_pos, local_quat_wxyz, pot_pos, pot_quat_wxyz):
    """Convert pot-local grasp point into world-frame 4x4 homogeneous matrix."""
    world_pos, world_quat = get_world_position_and_orientation_from_relative(
        local_pos, local_quat_wxyz, pot_pos, pot_quat_wxyz
    )
    return pose_from_position_quaternion(world_pos, world_quat)


def _gen_grasp_candidates(base_pose, pot_pose_mat, z_range=0.02, angle_range=5, n_z=5, n_angle=3):
    """Generate candidates near reference grasp: offsets along pot-local z and small rotations about local z/y."""
    pot_rot = pot_pose_mat[:3, :3]
    local_z = pot_rot[:, 2]
    local_y = pot_rot[:, 1]

    poses = []
    for dz in np.linspace(-z_range, z_range, n_z):
        shifted = base_pose.copy()
        shifted[:3, 3] += local_z * dz
        for az_deg in np.linspace(-angle_range, angle_range, n_angle):
            for ay_deg in np.linspace(-angle_range, angle_range, n_angle):
                p = shifted.copy()
                if az_deg != 0:
                    R_z = rotation_matrix(np.deg2rad(az_deg), local_z, point=p[:3, 3])
                    p = R_z @ p
                if ay_deg != 0:
                    R_y = rotation_matrix(np.deg2rad(ay_deg), local_y, point=p[:3, 3])
                    p = R_y @ p
                poses.append(p)
    return np.array(poses)


def _gen_move_targets(base_pose, offset):
    """Generate nearby candidates around base_pose + offset: z +/-0.02, x +/-0.01."""
    target = base_pose.copy()
    target[:3, 3] += offset
    targets = []
    for dz in np.linspace(-0.05, 0.05, 8):
        # for dx in np.linspace(-0.03, 0.03, 5):
        #     for dy in np.linspace(-0.03, 0.03, 5):
                t = target.copy()
                t[2, 3] += dz
                # t[0, 3] += dx
                # t[1, 3] += dy
                targets.append(t)
    targets = np.array(targets)
    np.random.shuffle(targets)
    return targets


def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_pot_001": {"has_interactions": False, "object_type": "pot"},
        "table_000": {"has_interactions": False, "object_type": "table"},
    }

    scene_info = robot.scene_info

    description_json = {
        "task_name": "bimanual grasp and lift the pot",
        "english_task_name": "bimanual grasp and lift the pot",
        "init_scene_text": "The robot is in front of the table, there is a pot on the table",
    }
    robot.client.set_task_task_basic_info(description_json)

    # -- 1. Find pot and table --
    pot_id = None
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        usd = obj_info.get("usd", "")
        if any("pot" in kw.lower() for kw in keywords):
            pot_id = obj_id
        if any("table" in kw.lower() for kw in keywords) or usd == "table_000":
            table_id = obj_id
    if pot_id is None:
        print("Pot not found")
        return False
    if table_id is None:
        print("Table not found")
        return False
    print(f"Found pot: {pot_id}, table: {table_id}")

    unmovable_objects = [pot_id]

    # -- 2. Move right arm to grasp point and close gripper --
    pot_pos, pot_quat = robot.get_object_pose(pot_id)
    pot_mat = pose_from_position_quaternion(pot_pos, pot_quat)
    initial_pot_z = pot_pos[2]
    print(f"Pot pose: pos={pot_pos}, quat={pot_quat}")

    right_base = _compute_world_grasp_pose(RIGHT_GRASP_POS, RIGHT_GRASP_QUAT, pot_pos, pot_quat)
    right_candidates = _gen_grasp_candidates(right_base, pot_mat)
    move_right_cfg = MoveConfig(
        gripper_action="close",
        unmovable_objects=unmovable_objects,
        action_description={"zh": "抓取锅右把手", "en": "grasps pot's right handle"},
    )
    if not robot.move_actor_to_target_pose("gripper", right_candidates, arm="right", move_config=move_right_cfg):
        print("Failed to move right arm to pot")
        return False
    print("Right arm grasped")

    # -- 3. Reread pot pose; right-arm contact may shift it --
    pot_pos, pot_quat = robot.get_object_pose(pot_id)
    pot_mat = pose_from_position_quaternion(pot_pos, pot_quat)
    print(f"Pot pose after right grasp: pos={pot_pos}, quat={pot_quat}")

    # -- 4. Move left arm to grasp point and close gripper --
    left_base = _compute_world_grasp_pose(LEFT_GRASP_POS, LEFT_GRASP_QUAT, pot_pos, pot_quat)
    left_candidates = _gen_grasp_candidates(left_base, pot_mat)
    move_left_cfg = MoveConfig(
        gripper_action="close",
        unmovable_objects=unmovable_objects,
        action_description={"zh": "抓取锅左把手", "en": "grasps pot's left handle"},
    )
    if not robot.move_actor_to_target_pose("gripper", left_candidates, arm="left", move_config=move_left_cfg):
        print("Failed to move left arm to pot")
        return False
    print("Left arm grasped")

    # -- 5. Attach pot to right arm --
    pot_prim = robot.get_object_prim_path(pot_id)
    robot.attach_obj([pot_prim], arm="right")
    robot.attach_obj([pot_prim], arm="left")
    print("Pot attached to both arms")

    # -- 6. Define successful-lift checker --
    def check_pot_lifted():
        current_pos, _ = robot.get_object_pose(pot_id)
        if current_pos is None:
            return False
        lift_delta = current_pos[2] - initial_pot_z
        min_lift = LIFT_HEIGHT * 0.5
        print(f"Pot lift delta: {lift_delta:.3f}m (min required: {min_lift:.3f}m)")
        if lift_delta < min_lift:
            print(f"Pot not lifted enough: {lift_delta:.3f}m < {min_lift:.3f}m")
            return False
        return True

    # -- 7. Lift with both arms in rigid coordination --
    ee_right = robot.get_ee_pose(arm="right")
    lift_targets = _gen_move_targets(ee_right, np.array([0, 0, LIFT_HEIGHT]))
    bimanual_lift_cfg = BimanualRigidMoveConfig(
        leader_arm="right",
        extra_checkers=[check_pot_lifted],
        action_description={"zh": "抬起锅", "en": "lift the pot"},
        max_consecutive_failures=3,
        offset_and_constraint_in_goal_frame=False,
    )
    if not robot.bimanual_rigid_move(target_pose=lift_targets, config=bimanual_lift_cfg):
        print("Failed to bimanual lift")
        return False

    print("Successfully lifted the pot with both arms")
    return True
