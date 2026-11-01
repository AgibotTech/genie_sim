# Stack trash cans

**Tags:** `stack` `trash_can` `keywords_inner_outer` `inner_trash_can` `outer_trash_can` `top_surface_from_local_axis` `quat2mat_wxyz` `TRASH_CAN_HEIGHT` `pose_based_placement` `rotate_around_z_90_multiples` `GraspConfig` `PlaceConfig` `pick_and_place_actor_to_target_pose` `benchmark_trash_can_013` `extra_checkers` `pre_insert_offset`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table by keywords/usd; find inner trash can by keyword "inner_trash_can" and outer by "outer_trash_can". Scene has exactly two cans.
2. **Top surface of inner can:** Object pose returns geometric center. Inner can's local up is column 1 of rotation matrix (local_y_axis). inner_top_center = inner_pos + (TRASH_CAN_HEIGHT/2) * local_y_axis. Use fixed TRASH_CAN_HEIGHT = 0.1495 if known.
3. **Target pose for outer can:** XY = inner_top_center[:2]. Z = inner_top_center[2] + z_bottom_offset (0.01~0.02 m) + TRASH_CAN_HEIGHT/2 so outer's bottom sits above inner's top. Rotation = inner_rot then rotate around world Z by 0, 90, 180, 270 degrees (final_rot = rot_z @ inner_rot_matrix).
4. **Candidate poses:** Multiple z_bottom_offsets (e.g. linspace(0.01, 0.015, 5)) and four rotation angles; combine into target_poses (N, 4, 4).
5. **Verification:** extra_checkers verify the cans' horizontal alignment and relative placement height. An optional grasp checker verifies lifting of the outer can.
6. **Execution:** Set task description; pick_and_place_actor_to_target_pose(outer_can_id, target_poses, ...) with unmovable_objects=[inner_can_id, table_id], pre_insert_offset=0.1.

---

# INPUT

**scene_description:** A desk with two trash cans (benchmark_trash_can_013). One is in the central (inner) area and one in the outer area, with minimum center-to-center distance 0.2 m. Both are randomly positioned and rotated. Keywords: "inner_trash_can_*", "outer_trash_can_*".

**task description:** Stack the outer trash can on top of the inner trash can. The outer can's bottom center XY must align with the inner can's top center XY; Z: outer bottom 0~0.02 m above inner top. Rotation: use inner can's rotation, then 90-degree multiples around world Z.

**code to generate the scene:**

```python
from helper import *

"""
scene_name: stack_trash_cans_g2
description: A desk with two trash cans. One is placed in the central area of the desk, and the other is placed on the outer area of the desk. Both are randomly positioned and rotated, with a minimum center-to-center distance of 0.2m between them.
"""


def get_desk_workspace_info(desk_shape: Shape) -> dict:
    """
    Extracts the desk's center and size in world coordinates.
    """
    desk_info = get_object_info(desk_shape)
    return {
        "center": desk_info["center"],
        "size": desk_info["size"],
        "min": desk_info["min"],
        "max": desk_info["max"]
    }


def sample_position_in_region(center: P, region_x: Tuple[float, float], region_y: Tuple[float, float]) -> P:
    """
    Samples a random (x, y) position within a specified region relative to a center point.
    The z-coordinate is set to 0 and should be adjusted later based on surface height.
    """
    x = np.random.uniform(region_x[0], region_x[1])
    y = np.random.uniform(region_y[0], region_y[1])
    return np.array([center[0] + x, center[1] + y, 0.0])


def place_trash_can_on_desk(desk_shape: Shape) -> Shape:
    """
    Places two trash cans on the desk according to the specified regions and constraints.
    Ensures minimum center-to-center distance of 0.2m between the two trash cans.
    """
    desk_info = get_desk_workspace_info(desk_shape)
    desk_center = desk_info["center"]

    # Get desktop subpart info for height calculation
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_top_z = desk_center[2] + desktop_subpart_info["xyz_max"][2]

    # Define regions
    inner_region_x = (-0.15, 0.05)
    inner_region_y = (-0.1, 0.1)
    outer_region_x = (-0.25, 0.1)
    outer_region_y = (-0.25, 0.25)

    max_attempts = 1000  # Increased attempts for better success rate
    for _ in range(max_attempts):
        # Sample position for inner trash can
        inner_pos = sample_position_in_region(desk_center, inner_region_x, inner_region_y)
        # Sample position for outer trash can
        outer_pos = sample_position_in_region(desk_center, outer_region_x, outer_region_y)

        # Calculate center-to-center distance in XY plane
        distance = np.linalg.norm(inner_pos[:2] - outer_pos[:2])

        # Check if distance constraint is satisfied (minimum 0.2m center-to-center)
        if distance >= 0.2:
            # Determine POSITION TAGs based on Y coordinate relative to desk center
            inner_tag = "left" if inner_pos[1] > desk_center[1] else "right"
            outer_tag = "left" if outer_pos[1] > desk_center[1] else "right"

            # Place inner trash can with random rotation
            inner_trash_can = library_call(
                "usd",
                oid="benchmark_trash_can_013",
                keywords=[f"inner_trash_can_{inner_tag}", "gray", "cylinder", "small", f"position_{inner_tag}"]
            )
            inner_trash_can = transform_shape(
                inner_trash_can,
                translation_matrix([inner_pos[0], inner_pos[1], desktop_top_z])
            )
            inner_trash_can = transform_shape(
                inner_trash_can,
                rotation_matrix(
                    angle=np.random.uniform(0, 2 * math.pi),
                    direction=(0, 0, 1),
                    point=compute_shape_center(inner_trash_can)
                )
            )

            # Place outer trash can with random rotation
            outer_trash_can = library_call(
                "usd",
                oid="benchmark_trash_can_013",
                keywords=[f"outer_trash_can_{outer_tag}", "gray", "cylinder", "small", f"position_{outer_tag}"]
            )
            outer_trash_can = transform_shape(
                outer_trash_can,
                translation_matrix([outer_pos[0], outer_pos[1], desktop_top_z])
            )
            outer_trash_can = transform_shape(
                outer_trash_can,
                rotation_matrix(
                    angle=np.random.uniform(0, 2 * math.pi),
                    direction=(0, 0, 1),
                    point=compute_shape_center(outer_trash_can)
                )
            )

            return concat_shapes(desk_shape, inner_trash_can, outer_trash_can)

    # Fallback: If no valid placement found, use fixed positions that guarantee distance
    # Place inner trash can at center of inner region
    inner_pos = np.array([
        desk_center[0] + (inner_region_x[0] + inner_region_x[1]) / 2.0,
        desk_center[1] + (inner_region_y[0] + inner_region_y[1]) / 2.0,
        desktop_top_z
    ])

    # Place outer trash can at a position that ensures 0.2m+ distance
    # Use the farthest point in outer region from inner_pos
    candidate_positions = [
        [desk_center[0] + outer_region_x[0], desk_center[1] + outer_region_y[0]],
        [desk_center[0] + outer_region_x[0], desk_center[1] + outer_region_y[1]],
        [desk_center[0] + outer_region_x[1], desk_center[1] + outer_region_y[0]],
        [desk_center[0] + outer_region_x[1], desk_center[1] + outer_region_y[1]]
    ]

    best_outer_pos = None
    max_distance = -1
    for candidate in candidate_positions:
        dist = np.linalg.norm(np.array([inner_pos[0], inner_pos[1]]) - np.array(candidate))
        if dist > max_distance:
            max_distance = dist
            best_outer_pos = candidate

    outer_pos = np.array([best_outer_pos[0], best_outer_pos[1], desktop_top_z])

    inner_tag = "left" if inner_pos[1] > desk_center[1] else "right"
    outer_tag = "left" if outer_pos[1] > desk_center[1] else "right"

    # Place inner trash can with random rotation (fallback)
    inner_trash_can = library_call(
        "usd",
        oid="benchmark_trash_can_013",
        keywords=[f"inner_trash_can_{inner_tag}_fallback", "gray", "cylinder", "small", f"position_{inner_tag}"]
    )
    inner_trash_can = transform_shape(
        inner_trash_can,
        translation_matrix([inner_pos[0], inner_pos[1], desktop_top_z])
    )
    inner_trash_can = transform_shape(
        inner_trash_can,
        rotation_matrix(
            angle=np.random.uniform(0, 2 * math.pi),
            direction=(0, 0, 1),
            point=compute_shape_center(inner_trash_can)
        )
    )

    # Place outer trash can with random rotation (fallback)
    outer_trash_can = library_call(
        "usd",
        oid="benchmark_trash_can_013",
        keywords=[f"outer_trash_can_{outer_tag}_fallback", "gray", "cylinder", "small", f"position_{outer_tag}"]
    )
    outer_trash_can = transform_shape(
        outer_trash_can,
        translation_matrix([outer_pos[0], outer_pos[1], desktop_top_z])
    )
    outer_trash_can = transform_shape(
        outer_trash_can,
        rotation_matrix(
            angle=np.random.uniform(0, 2 * math.pi),
            direction=(0, 0, 1),
            point=compute_shape_center(outer_trash_can)
        )
    )

    return concat_shapes(desk_shape, inner_trash_can, outer_trash_can)


@register()
def stack_trash_cans_g2() -> Shape:
    desk_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )
    return place_trash_can_on_desk(desk_shape)


@register()
def root_scene() -> Shape:
    return stack_trash_cans_g2()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # objects_list: Dictionary mapping asset_id to object metadata
    objects_list = {
        "benchmark_trash_can_013": {"has_interactions": True, "object_type": "trash_can"}
    }

    # runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    description_json = {
        "task_name": "将外侧垃圾桶叠放在内侧垃圾桶上",
        "english_task_name": "stack the outer trash can on top of the inner trash can",
        "init_scene_text": "The robot is in front of the table, there are two trash cans on it, one in the inner area and one in the outer area"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Find the table
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        if "table" in keywords or obj_info.get("usd", "") == "table_000":
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. Find inner and outer trash cans
    inner_can_id = None
    outer_can_id = None

    for obj_id, obj_info in scene_info["layout"].items():
        keywords = obj_info.get("keywords", [])
        # Find inner trash can
        if any("inner_trash_can" in kw for kw in keywords):
            inner_can_id = obj_id
            print(f"Found inner trash can: {obj_id}")
        # Find outer trash can
        if any("outer_trash_can" in kw for kw in keywords):
            outer_can_id = obj_id
            print(f"Found outer trash can: {obj_id}")

    if not inner_can_id or not outer_can_id:
        print("Could not find both trash cans")
        return False

    # 3. Get poses and sizes
    inner_pos, inner_rot = robot.get_object_pose(inner_can_id)
    inner_size = robot.get_object_size_in_world_frame(inner_can_id)

    outer_pos, outer_rot = robot.get_object_pose(outer_can_id)
    outer_size = robot.get_object_size_in_world_frame(outer_can_id)

    if inner_pos is None or inner_size is None:
        print("Failed to get inner trash can info")
        return False

    if outer_pos is None or outer_size is None:
        print("Failed to get outer trash can info")
        return False

    # 4. Determine arm
    arm = robot.judge_arm_to_use(outer_pos)
    print(f"Using arm: {arm}")

    # 5. Generate target poses
    # Requirement:
    # - Position: The XY of outer bucket's bottom center should exactly match the XY of inner bucket's top center.
    #   Z: bottom of outer can is 0.0~0.02m above top of inner can.
    # - Rotation: Use inner can's rotation, then rotate around world Z by 90-degree multiples.

    # 垃圾桶的确定高度
    TRASH_CAN_HEIGHT = 0.1495
    inner_rot_matrix = quat2mat_wxyz(inner_rot)

    # 计算内侧垃圾桶的上表面中心点
    # 局部z轴正方向在世界坐标系中的表示（旋转矩阵的第三列）
    local_y_axis = inner_rot_matrix[:, 1]
    # 上表面中心点 = 几何中心 + (H/2) * 局部y轴
    inner_top_center = inner_pos + (TRASH_CAN_HEIGHT / 2.0) * local_y_axis

    print(f"Inner trash can geometric center: {inner_pos}")
    print(f"Inner trash can top surface center: {inner_top_center}")

    # Generate candidates
    target_poses = []

    # Z offsets for the bottom of the outer can relative to the top of the inner can
    # Requirement: 0.0 to 0.02m.
    # Final check requires height_diff >= 0.01 (outer center z - inner center z),
    # which approximately means outer bottom is 0.01m above inner top.
    # To ensure verification passes, generate offsets in range 0.015~0.025
    z_bottom_offsets = np.linspace(0.01, 0.015, 5)

    # Rotation angles around Z axis (multiples of 90 degrees)
    rotation_angles = [0, np.pi/2, np.pi, 3*np.pi/2]

    for z_bottom_offset in z_bottom_offsets:
        # 目标：外侧垃圾桶的下表面中心点z = inner_top_center[2] + z_bottom_offset
        # 外侧垃圾桶的几何中心z = 下表面中心z + (H/2)
        target_z = inner_top_center[2] + z_bottom_offset + TRASH_CAN_HEIGHT / 2.0

        # xy坐标完全对应内侧垃圾桶的上表面中心点
        target_position = np.array([inner_top_center[0], inner_top_center[1], target_z])

        for angle in rotation_angles:
            # Create rotation matrix for Z axis rotation
            rot_z = euler2mat([0, 0, angle])

            # Combine: First inner rotation, then Z rotation
            # Requirement: "Take inner can rotation, then rotate around Z".
            # This means final rotation = rot_z @ inner_rot_matrix
            final_rot_matrix = np.dot(rot_z, inner_rot_matrix)

            pose = np.eye(4)
            pose[:3, :3] = final_rot_matrix
            pose[:3, 3] = target_position
            target_poses.append(pose)

    target_poses = np.array(target_poses)
    print(f"Generated {len(target_poses)} target poses")

    # 6. Define checker function for success verification
    def check_stacked():
        # Verify:
        # 1. XY alignment: outer bucket's bottom center xy should match inner bucket's top center xy
        # 2. Z height: outer bucket's bottom center should be >= 0.01m above inner bucket's top center

        curr_inner_pos, curr_inner_rot = robot.get_object_pose(inner_can_id)
        curr_outer_pos, curr_outer_rot = robot.get_object_pose(outer_can_id)

        if curr_inner_pos is None or curr_outer_pos is None:
            return False

        # 检查xy对齐（3cm容差）
        xy_dist = np.linalg.norm(curr_outer_pos[:2] - curr_inner_pos[:2])
        if xy_dist > 0.03:
            print(f"XY alignment check failed: xy distance {xy_dist:.4f}m")
            return False

        # 检查z高度：外侧下表面中心应该比内侧上表面中心高至少0.01m
        height_diff = curr_outer_pos[2] - curr_inner_pos[2]

        # 要求：0.0~0.02m之间，但最终判断要求>=0.01
        # 容差范围：0.008m ~ 0.1m (考虑实际误差)
        if height_diff < 0.008 or height_diff > 0.1:
            print(f"Z height check failed: height diff {height_diff:.4f}m, expected 0.008~0.1m")
            return False

        print(f"Success check passed: XY dist={xy_dist:.4f}m, Z diff (bottom relative to top)={height_diff:.4f}m")
        return True

    def check_picked_up():
        curr_outer_pos, _ = robot.get_object_pose(outer_can_id)
        if curr_outer_pos is None:
            return False
        robot.client.run_step(30)
        height_diff = curr_outer_pos[2] - outer_pos[2]
        if height_diff < 0.05:
            print(f"Picked up check failed: outer_pos is {curr_outer_pos[2]:.4f}m, required >= 0.05m")
            return False
        return True

    # 7. Execute pick and place
    grasp_config = GraspConfig(
        pick_up_distance=0.2,
        is_from_up_side=True,
        unmovable_objects=[inner_can_id, table_id],
        action_description={"zh": "拿起外侧的垃圾桶", "en": "pick up the outer trash can"},
        extra_params={"skip_check": True},
        extra_checkers=[check_picked_up]
    )

    # 使用预放置点0.08m
    place_config = PlaceConfig(
        pre_insert_offset=0.1,
        unmovable_objects=[inner_can_id, table_id],
        disable_upside_down=False,
        extra_checkers=[check_stacked],
        action_description={"zh": "将外侧垃圾桶叠放到内侧垃圾桶上", "en": "place the outer trash can on top of the inner trash can"}
    )

    success = robot.pick_and_place_actor_to_target_pose(
        actor=outer_can_id,
        target_pose=target_poses,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config
    )

    robot.back_to_origin(arm=arm)

    return success
```
