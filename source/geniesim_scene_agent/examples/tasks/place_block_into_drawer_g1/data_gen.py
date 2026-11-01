# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np

def execute_task(robot: RobotAPI):
    scene_info = robot.scene_info
    interactions = robot.interactions
    objects_list = {
        "benchmark_building_blocks_086": {
            "has_interactions": True,
            "object_type": "building_block",
            "active_place_primitives": ["侧放", "正放"],
            "passive_place_primitives": ["正放"],
            "passive_grasp_primitives": ["default"]
        },
        "iros_drawer_cabinet_000": {
            "has_interactions": False,  # No interaction data provided for drawer cabinet
            "object_type": "drawer_cabinet"
        }
    }
    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "one hand pulls open the drawer, the other hand places the block into the drawer, then closes the drawer",
        "english_task_name": "one hand pulls open the drawer, the other hand places the block into the drawer, then closes the drawer",
        "init_scene_text": "The robot is in front of a drawer cabinet, there is a building block on top of the cabinet"
    }
    robot.client.set_task_task_basic_info(description_json)

    # ============================================================
    # 1. Find cabinet and block objects from scene_info
    # ============================================================
    cabinet_id = None
    block_id = None

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        if "iros_drawer_cabinet" in usd:
            cabinet_id = obj_id
        elif "benchmark_building_blocks_086" in usd:
            block_id = obj_id

    if not cabinet_id or not block_id:
        print("Cabinet or block not found")
        return False

    print(f"Found cabinet: {cabinet_id}")
    print(f"Found block: {block_id}")

    # ============================================================
    # 2. Get block position and decide which arm to use for grasping
    #    grasp_arm: picks up the block
    #    pull_arm: opens and closes the drawer
    # ============================================================
    block_pos, _ = robot.get_object_pose(block_id)
    grasp_arm = robot.judge_arm_to_use(block_pos)

    # The other arm will pull the drawer
    pull_arm = "left" if grasp_arm == "right" else "right"

    print(f"Grasp arm: {grasp_arm}")
    print(f"Pull arm: {pull_arm}")

    # ============================================================
    # 3. Prepare drawer grasp points (world coordinates)
    #    Two possible grasp points with variations around Y and Z axes
    # ============================================================
    drawer_grasp_world_poses = [
        {
            "xyz": np.array([-0.15202, -0.06201, 0.79722]),
            "quat": np.array([0.707, 0.0, 0.707, 0.0])  # wxyz
        },
        {
            "xyz": np.array([-0.15202, -0.06201, 0.79722]),
            "quat": np.array([0.0, 0.707, 0.0, 0.707])  # wxyz
        }
    ]

    # ============================================================
    # 4. Generate multiple candidate poses for drawer grasping
    #    Add small variations in rotation around Y and Z axes
    # ============================================================
    cabinet_pos, cabinet_rot = robot.get_object_pose(cabinet_id)
    cabinet_rot_mat = quat2mat_wxyz(cabinet_rot)

    drawer_grasp_poses = []
    for grasp_point in drawer_grasp_world_poses:
        target_rotation = quat2mat_wxyz(grasp_point["quat"])
        target_pose = make_pose(grasp_point["xyz"], target_rotation)

        # Add Y-axis rotation variations (-90° to 90°, 10 samples)
        for y_rotate in np.linspace(-90, 90, 10):
            rotated_target_pose = rotate_along_axis(target_pose, y_rotate, "y", use_local=False)
            drawer_grasp_poses.append(rotated_target_pose)

        # Add Z-axis rotation variations (-90° to 90°, 10 samples)
        for z_rotate in np.linspace(-90, 90, 10):
            rotated_target_pose = rotate_along_axis(target_pose, z_rotate, "z", use_local=False)
            drawer_grasp_poses.append(rotated_target_pose)

    drawer_grasp_poses = np.array(drawer_grasp_poses)
    print(f"Generated {len(drawer_grasp_poses)} drawer grasp poses")

    # ============================================================
    # 5. Record initial drawer state for verification
    # ============================================================
    # Get initial position of drawer part_00 (the movable drawer part)
    initial_part00_pos, _ = robot.get_object_pose(cabinet_id, articulation_part="part_00")
    initial_part00_x = initial_part00_pos[0]
    print(f"Initial drawer part_00 X position: {initial_part00_x:.4f}")

    # ============================================================
    # 6. Prepare placement poses for block inside drawer
    #    Based on current block position, allow random Z rotation and position offsets
    # ============================================================
    building_block_position, building_block_quat = robot.get_object_pose(block_id)
    margin = 0.03
    building_block_place_pose = np.array([-0.15202- 0.02, -0.06201, building_block_position[2]])

    building_block_rot_mat = quat2mat_wxyz(building_block_quat)
    place_pose = make_pose(building_block_place_pose, building_block_rot_mat)
    building_block_place_poses = []

    # Generate poses with 12 different Z-axis rotations (0-360 degrees)
    for angle_deg in np.linspace(0, 360, 8, endpoint=False):
        target_pose = rotate_along_axis(place_pose, angle_deg, "z", use_local=False)
        for angle_x in np.linspace(-30, 30, 3):
            target_pose = rotate_along_axis(target_pose, angle_x, "x", use_local=False)
            for angle_y in np.linspace(-30, 30, 3):
                target_pose = rotate_along_axis(target_pose, angle_y, "y", use_local=False)
                # Add position variations (x: ±2cm, y: ±4cm, z: ±2cm)
                for dx in np.linspace(-0.02, 0.02, 3):
                    for dy in np.linspace(-0.04, 0.04, 5):
                        for dz in np.linspace(-0.02, 0.02, 3):
                            target_pose[0,3] += dx
                            target_pose[1,3] += dy
                            target_pose[2,3] += dz
                        building_block_place_poses.append(target_pose)

    building_block_place_poses = np.array(building_block_place_poses)
    print(f"Generated {len(building_block_place_poses)} building block place poses")

    # ============================================================
    # 7. Define checker functions for success verification
    # ============================================================

    # Checker: Verify drawer is opened (part_00 X position decreased by more than 0.05m)
    def check_drawer_opened():
        """Check if drawer is opened: part_00 x position < initial_x - 0.05"""
        current_part00_pos, _ = robot.get_object_pose(cabinet_id, articulation_part="part_00")
        displacement = initial_part00_x - current_part00_pos[0]
        print(f"Drawer displacement: {displacement:.4f}m (need > 0.05m)")
        return displacement > 0.05

    # Checker: Verify drawer is closed (part_00 X position difference < 0.03m)
    def check_drawer_closed():
        """Check if drawer is closed: |current_x - initial_x| < 0.03"""
        current_part00_pos, _ = robot.get_object_pose(cabinet_id, articulation_part="part_00")
        displacement = abs(current_part00_pos[0] - initial_part00_x)
        print(f"Drawer displacement after close: {displacement:.4f}m (need < 0.03m)")
        return displacement < 0.03

    # Checker: Verify block is inside drawer (block z in part_00 z range: -0.2 to 0.02)
    def check_block_in_drawer():
        """Check if block is inside drawer: block z in part_00 z range [-0.2, 0.02]"""
        block_pos_curr, _ = robot.get_object_pose(block_id)

        # Calculate relative z position of block with respect to part_00
        block_initial_z = building_block_position[2]
        z_diff = block_pos_curr[2] - block_initial_z
        print(f"Block relative Z: {z_diff:.4f}m (need in [-0.5, -0.05])")

        return -0.5 <= z_diff <= -0.05

    # ============================================================
    # 8. First step: Pull arm moves to drawer grasp point and opens drawer
    #    NOTE: Execution order is OPEN DRAWER -> PLACE BLOCK -> CLOSE DRAWER
    # ============================================================

    # MoveConfig for approaching drawer handle
    move_config = MoveConfig(
        action_description={"zh": "移动到抽屉抓取点", "en": "moves to drawer grasp point"},
        unmovable_objects=[cabinet_id, block_id]  # Protect both cabinet and block during approach
    )

    # Remove cabinet part from obstacle list (allow gripper to touch drawer)
    robot.remove_objs_from_obstacle([robot.get_object_prim_path(cabinet_id, "part_01")])

    # Move pull arm's gripper to drawer grasp point
    first_success = robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=drawer_grasp_poses,
        arm=pull_arm,
        move_config=move_config
    )
    if not first_success:
        print("Failed to move pull arm to drawer grasp point")
        return False

    # Close gripper before pulling drawer
    robot.close_gripper(arm=pull_arm)

    print("Pull arm moved to drawer grasp point")

    # ============================================================
    # 9. Second step: Pull drawer open (move along negative X direction by 0.1m)
    # ============================================================

    pull_move_config = MoveConfig(
        extra_params={
            "from_current_pose": True,
            "goal_offset": [-0.1, 0, 0, 1, 0, 0, 0],  # Move -0.1m in X direction
            "path_constraint": [0.1, 0.1, 0.1, 0.0, 0.1, 0.1],  # Only allow motion along X axis
            "offset_and_constraint_in_goal_frame": False,
        },
        action_description={"zh": "拉开抽屉", "en": "pulls drawer open"},
        unmovable_objects=[cabinet_id, block_id],  # Protect cabinet and block
        extra_checkers=[check_drawer_opened]  # Verify drawer opened successfully
    )

    # Modify poses: shift X by -0.1m (this is done in-place for each pose)
    for i in range(len(drawer_grasp_poses)):
        drawer_grasp_poses[i, 0, 3] -= 0.1

    pull_success = robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=drawer_grasp_poses,
        arm=pull_arm,
        move_config=pull_move_config
    )

    if not pull_success:
        print("Failed to pull drawer open")
        return False

    print("✅ Drawer pulled open successfully")

    # ============================================================
    # 10. Third step: Grasp arm picks up the block from cabinet
    # ============================================================

    grasp_config = GraspConfig(
        passive_primitive="default",
        unmovable_objects=[cabinet_id],  # Protect cabinet while grasping block
        pick_up_distance=0.03,
        action_description={"zh": "抓起柜子上的积木", "en": "picks up the block on the cabinet"}
    )

    current_success = robot.pick_actor(block_id, arm=grasp_arm, grasp_config=grasp_config)
    if not current_success:
        print("❌ Failed to grasp block")
        return False

    print("✅ Block grasped successfully")

    # ============================================================
    # 11. Fourth step: Place block into drawer
    # ============================================================

    place_config = PlaceConfig(
        unmovable_objects=[cabinet_id],  # Protect cabinet during placement
        extra_checkers=[check_block_in_drawer],  # Verify block is inside drawer
        action_description={"zh": "把积木放到抽屉中", "en": "places the block in the drawer"},
    )

    current_success = robot.place_actor_to_target_pose(
        actor=block_id,
        target_pose=building_block_place_poses,
        arm=grasp_arm,
        place_config=place_config
    )

    if not current_success:
        print("❌ Failed to move block to place pose")
        return False

    print("✅ Block placed in drawer successfully")

    # Return grasp arm to origin after placing block
    robot.back_to_origin(arm=grasp_arm)

    # ============================================================
    # 12. Fifth step: Pull arm closes the drawer
    # ============================================================

    push_config = MoveConfig(
        action_description={"zh": "合上抽屉", "en": "closes the drawer"},
        extra_params={
            "from_current_pose": True,
            "goal_offset": [0.101, 0, 0, 1, 0, 0, 0],  # Move +0.101m in X direction (close)
            "path_constraint": [0.1, 0.1, 0.1, 0.0, 0.1, 0.1],  # Only allow motion along X axis
            "offset_and_constraint_in_goal_frame": False,
        },
        unmovable_objects=[cabinet_id],  # Protect cabinet and block
        extra_checkers=[check_drawer_closed]  # Verify drawer closed successfully
    )

    # Remove cabinet drawer part from obstacle list (allow pushing)
    robot.remove_objs_from_obstacle([robot.get_object_prim_path(cabinet_id, "part_00")])
    robot.remove_objs_from_obstacle([robot.get_object_prim_path(block_id)])


    # Modify poses: shift X by +0.1m (reset to original then add 0.1m)
    for i in range(len(drawer_grasp_poses)):
        drawer_grasp_poses[i, 0, 3] += 0.1

    push_success = robot.move_actor_to_target_pose(
        actor="gripper",
        target_pose=drawer_grasp_poses,
        arm=pull_arm,
        move_config=push_config
    )

    if not push_success:
        print("❌ Failed to push drawer closed")
        return False

    print("✅ Drawer pushed closed successfully")

    # Open gripper and return pull arm to origin
    # robot.open_gripper(arm=pull_arm)
    # robot.back_to_origin(arm=pull_arm)

    # ============================================================
    # Final summary
    # ============================================================
    print("\n" + "="*60)
    print("TASK EXECUTION SUMMARY")
    print("="*60)
    print(f"Step 1: ✅ Drawer opened (pull arm)")
    print(f"Step 2: ✅ Block grasped (grasp arm)")
    print(f"Step 3: ✅ Block placed in drawer (grasp arm)")
    print(f"Step 4: ✅ Drawer closed (pull arm)")
    print("="*60)
    print("All steps completed successfully!")

    return success
