# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os


def build_part_grasp_poses(center, arm, n_azimuths=8):
    """Build generalized 45-degree oblique-down world-frame grasp poses."""
    center = np.asarray(center, dtype=float)
    center = center.copy()
    # center[2] += 0.02

    # Keep the arm argument for the existing call interface. IK filtering will
    # select the candidates reachable by the chosen arm.
    _ = arm
    azimuths = np.linspace(0.0, 2.0 * np.pi, n_azimuths, endpoint=False)
    tilts = np.deg2rad([30.0, 45.0, 60.0])

    poses = []
    for tilt in tilts:
        for azimuth in azimuths:
            # Gripper local +Z is the approach direction and points into the part.
            z_axis = np.array([
                np.cos(azimuth) * np.cos(tilt),
                np.sin(azimuth) * np.cos(tilt),
                -np.sin(tilt),
            ])
            z_axis /= np.linalg.norm(z_axis)

            tangent = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
            for x_sign in (1.0, -1.0):
                x_axis = x_sign * tangent
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)
                pose = np.eye(4)
                pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
                pose[:3, 3] = center
                poses.append(pose)
    return np.asarray(poses)


def execute_task(robot: RobotAPI):
    ## Initialize objects_list based on provided interaction data
    ## Note: benchmark_gearbox_000 interaction data is provided in the prompt text
    objects_list = {
        "benchmark_gearbox_000": {
            "has_interactions": True,
            "object_type": "gearbox",
            "interaction_type": "container"
        },
        "benchmark_bearing_001": {
            "has_interactions": True,
            "object_type": "bearing",
            "size": "small"
        },
        "benchmark_bearing_000": {
            "has_interactions": True,
            "object_type": "bearing",
            "size": "large"
        },
        "benchmark_gear_001": {
            "has_interactions": True,
            "object_type": "gear",
            "size": "small"
        }
    }

    ## Runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## MANDATORY: Set task description before first action
    description_json = {
        "task_name": "pick up the part and mount it on the corresponding column of the gearbox",
        "english_task_name": "pick up the part and mount it on the corresponding column of the gearbox",
        "init_scene_text": "The robot is in front of the table, there is a gearbox in the center and a gear or bearing placed nearby"
    }
    robot.client.set_task_task_basic_info(description_json)

    ## 1. Find the gearbox (benchmark_gearbox_000)
    gearbox_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        if "benchmark_gearbox_000" in usd:
            gearbox_id = obj_id
            break

    if not gearbox_id:
        print("Gearbox not found")
        return False

    print(f"Found gearbox: {gearbox_id}")

    ## 2. Find the gear or bearing (side item)
    part_candidates = [
        "benchmark_bearing_001", "benchmark_bearing_000",
        "benchmark_gear_001"
    ]

    part_id = None
    part_usd = None
    part_type = None
    part_size = None

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check if it's one of the candidate parts
        if any(candidate in usd for candidate in part_candidates):
            # Ensure it's not the gearbox (though keywords usually distinguish)
            if "gearbox" not in keywords:
                part_id = obj_id
                part_usd = usd
                # Determine type and size
                if "bearing" in usd:
                    part_type = "bearing"
                elif "gear" in usd:
                    part_type = "gear"

                # Determine size based on asset ID suffix
                if "gear_001" in usd or "bearing_000" in usd:
                    part_size = "small"
                elif "gear_000" in usd or "bearing_001" in usd:
                    part_size = "large"
                break

    if not part_id:
        print("Gear or bearing not found")
        return False

    print(f"Found part: {part_id} (type: {part_type}, size: {part_size})")

    ## 3. Determine target primitive based on size
    ## Small gear/bearing -> small_cylinder
    ## Large gear/bearing -> right_cylinder (or back_cylinder, choosing right_cylinder as primary)
    part_name_zh = "轴承" if part_type == "bearing" else "齿轮"
    part_name_en = "bearing" if part_type == "bearing" else "gear"
    if part_size == "small":
        target_primitive = "small_cylinder"
        place_desc_zh = f"将{part_name_zh}放置到齿轮盒的小柱子上"
        place_desc_en = f"places the {part_name_en} on the small column of the gearbox"
    else: # large
        target_primitive = "right_cylinder"
        place_desc_zh = f"将{part_name_zh}放置到齿轮盒的右侧柱子上"
        place_desc_en = f"places the {part_name_en} on the right column of the gearbox"
    ## 4. Get part position to determine arm
    part_position, _ = robot.get_object_pose(part_id)
    if part_position is None:
        print(f"Cannot get position for part {part_id}")
        return False

    arm = robot.judge_arm_to_use(part_position)
    print(f"Using arm: {arm}")

    ## 5. Define success checker
    ## Verify that the part is actually placed on the target cylinder
    def create_part_mounted_checker(part_id, gearbox_id, target_primitive):
        def checker():
            # Get current poses
            part_pos, _ = robot.get_object_pose(part_id)
            gear_pos, gear_rot = robot.get_object_pose(gearbox_id)

            if part_pos is None or gear_pos is None or gear_rot is None:
                return False

            # Get target cylinder coordinates from interaction data (in object local frame)
            gearbox_interaction = interactions.get("benchmark_gearbox_000", {})
            passive_place = gearbox_interaction.get("passive", {}).get("place", {})

            if target_primitive not in passive_place:
                print(f"Target primitive {target_primitive} not found in interaction data")
                return False

            # There should be at least one point for the primitive
            primitive_points = passive_place[target_primitive]
            if len(primitive_points) == 0:
                return False

            target_local_xyz = np.array(primitive_points[0]["xyz"])

            # Transform local target to world coordinates
            # World = Gear_Transform * Local
            gear_rot_matrix = quat2mat_wxyz(gear_rot)
            target_world_offset = np.dot(gear_rot_matrix, target_local_xyz)
            target_world_pos = gear_pos + target_world_offset

            # Check distance (tolerance: 2cm)
            distance = np.linalg.norm(part_pos[:2] - target_world_pos[:2])
            is_close = distance < 0.02

            if not is_close:
                print(f"Part distance to target {target_primitive}: {distance:.3f}m (threshold: 0.02m)")

            if part_pos[2] > target_world_pos[2] - 0.04:
                print(f"Part is not on target height: {part_pos[2]} > {target_world_pos[2]}")
                return False

            return is_close

        return checker

    ## 6. Execute Pick and Place
    grasp_config = GraspConfig(
        grasp_pose=build_part_grasp_poses(part_position, arm),
        grasp_pose_frame="world",
        flip_grasp=True,
        disable_upside_down=True,
        pick_up_distance=0.08,
        grasp_offset=0.005,
        unmovable_objects=[gearbox_id],
        action_description={"zh": f"拿起{part_name_zh}", "en": f"picks up the {part_name_en}"}
    )

    place_config = PlaceConfig(
        active_primitive="normal_place",  # Parts use normal_place for active placement
        passive_primitive=target_primitive,  # Target specific cylinder
        pre_insert_offset=0.1,
        extra_params={
            "insert_depth": 0.05
        },
        unmovable_objects=[gearbox_id],
        extra_checkers=[create_part_mounted_checker(part_id, gearbox_id, target_primitive)],
        action_description={"zh": place_desc_zh, "en": place_desc_en}
    )

    print(f"Attempting to place part on {target_primitive}")
    current_success = robot.pick_and_place_actor_to_target_actor(
        actor=part_id,
        target_actor=gearbox_id,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config
    )

    if not current_success:
        print(f"Failed to place part {part_id}")
        success = False
    else:
        print(f"Successfully placed part {part_id} on {target_primitive}")

    return success
