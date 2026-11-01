# Sort number blocks in descending order

**Tags:** `sort` `digit_blocks` `table_only` `pose_based_placement` `linear_arrangement` `ordering_by_value` `keywords` `usd_match` `top_surface_z_from_bbox` `safety_margin` `multiple_candidate_poses` `rotate_around_z` `align_local_axis` `extra_checkers` `unmovable_objects` `action_description` `GraspConfig` `PlaceConfig` `pick_and_place_actor_to_target_pose`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table by usd "table_000" or keywords; find all digit blocks by matching usd to objects_list (digit 0–9). Build list with id, usd, digit, position, rotation, size from scene_info["layout"] and get_object_pose/get_object_size_in_world_frame.
2. **Sorting:** Sort digit_blocks by digit value descending (large to small). Compute target positions: linear arrangement with spacing (e.g. 0.05 m), start_y and target_x; table_top_z = table_pos[2] + table_size[2]/2; each target_z = table_top_z + margin + block height/2.
3. **Target rotation:** Align block local x-axis with world -x (rotation around z). calculate_target_rotation(initial_rotation): project local x onto xy-plane, compute angle to (-1,0), apply rot_z; keep local y up.
4. **Candidate poses:** For each block generate multiple poses with small x/y/z offsets and the same target_rot_matrix. Use create_placement_checker(block_id, expected_position) for extra_checkers (xy tolerance, on table, orientation dot check).
5. **Execution:** Set task description; for each block in sorted order: unmovable_objects = [table_id] + placed_blocks; pick_and_place_actor_to_target_pose with grasp_config and place_config; back_to_origin when switching arms.

---

# INPUT

**scene_description:** A table with several digit blocks (building blocks with digits 0–9) placed on it. The number and identities of blocks may vary; positions are arbitrary. The robot is in front of the table.

**task description:** Arrange the digit blocks from large to small (by digit value). Place them in a line (e.g. left to right) with consistent spacing and orientation (e.g. local x-axis aligned with world -x, local y-axis up). All blocks must remain on the table.

**code to generate the scene:**

```python
from helper import *
import random
import numpy as np

"""
scene_name: table_with_digit_blocks
description: A white rectangular table with three different digit-shaped building blocks placed randomly on its surface within x ∈ (-0.1, 0.15), y ∈ (-0.3, 0.3), ensuring no collisions and at least 5 cm spacing between blocks. Each block is randomly rotated around the vertical (z) axis.
"""

# Mapping from digit to asset ID
DIGIT_TO_ASSET = {
    0: "benchmark_building_blocks_034",
    1: "benchmark_building_blocks_035",
    2: "benchmark_building_blocks_036",
    3: "benchmark_building_blocks_037",
    4: "benchmark_building_blocks_038",
    5: "benchmark_building_blocks_039",
    6: "benchmark_building_blocks_040",
    7: "benchmark_building_blocks_041",
    8: "benchmark_building_blocks_042",
    9: "benchmark_building_blocks_043",
}


@register()
def digit_block(digit: int) -> Shape:
    """Creates a single digit-shaped building block."""
    asset_id = DIGIT_TO_ASSET[digit]
    return library_call(
        "usd", oid=asset_id, keywords=[f"digit_{digit}", "plastic", "small", "building_block", "educational_toy"]
    )


@register()
def place_three_digit_blocks_on_table(table_shape: Shape) -> Shape:
    """
    Places three non-colliding digit blocks on the table within the specified region.
    Ensures at least 5 cm (0.05 m) spacing between any two blocks.
    Each block is randomly rotated around the z-axis after placement.
    """
    # Get table top surface info
    table_info = get_object_info(table_shape)
    desktop_subpart = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Compute world position of desktop top surface center
    desktop_center_world = table_info["center"] + desktop_subpart["center"]
    desktop_top_z = table_info["center"][2] + desktop_subpart["xyz_max"][2]
    desktop_pos = np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z])
    desktop_size = desktop_subpart["size"]

    # Define placement region relative to desktop center
    x_min, x_max = -0.2, -0.1
    y_min, y_max = -0.2, 0.2

    # Select three distinct digits
    selected_digits = random.sample(list(DIGIT_TO_ASSET.keys()), 3)

    placed_positions = []
    block_shapes = []

    for i, digit in enumerate(selected_digits):
        block_shape = library_call("digit_block", digit=digit)
        block_info = get_object_info(block_shape)
        block_size = block_info["size"]
        block_radius = max(block_size[0], block_size[1]) / 2.0  # conservative estimate for collision

        max_attempts = 100
        for attempt in range(max_attempts):
            # Random position within region
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            pos = np.array([x, y, desktop_top_z])

            # Check distance to already placed blocks
            valid = True
            for placed_pos in placed_positions:
                dist = np.linalg.norm(pos[:2] - placed_pos[:2])
                if dist < 0.05 + block_radius + 0.015:  # 5cm + both radii (approx 1.5cm each)
                    valid = False
                    break

            if valid:
                placed_positions.append(pos)

                # First translate to position
                block_translated = transform_shape(block_shape, translation_matrix(pos))
                # Then rotate around its own center (which is now at pos)
                rotation_angle = random.uniform(160.0 / 180 * np.pi, 200.0 / 180 * np.pi)  # Full random rotation
                block_rotated = transform_shape(
                    block_translated, rotation_matrix(angle=rotation_angle, direction=(0, 0, 1), point=pos)
                )
                block_shapes.append(block_rotated)
                break
        else:
            # Fallback: place at edge if failed (should not happen often)
            fallback_x = x_min + i * 0.1
            fallback_y = 0.0
            fallback_pos = np.array([fallback_x, fallback_y, desktop_top_z])
            placed_positions.append(fallback_pos)

            block_translated = transform_shape(block_shape, translation_matrix(fallback_pos))
            rotation_angle = random.uniform(0, 2 * np.pi)
            block_rotated = transform_shape(
                block_translated, rotation_matrix(angle=rotation_angle, direction=(0, 0, 1), point=fallback_pos)
            )
            block_shapes.append(block_rotated)

    return concat_shapes(table_shape, *block_shapes)


@register()
def table_with_digit_blocks() -> Shape:
    table_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "rectangular", "study_desk"]
    )
    return place_three_digit_blocks_on_table(table_shape)


@register()
def root_scene() -> Shape:
    return table_with_digit_blocks()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os


def execute_task(robot: RobotAPI):
    ## Initialize objects_list based on provided assets
    objects_list = {
        "benchmark_building_blocks_034": {
            "has_interactions": True,
            "object_type": "digit_block",
            "digit": 0,
        },
        "benchmark_building_blocks_035": {
            "has_interactions": True,
            "object_type": "digit_block",
            "digit": 1,
        },
        "benchmark_building_blocks_036": {
            "has_interactions": True,
            "object_type": "digit_block",
            "digit": 2,
        },
        "benchmark_building_blocks_037": {
            "has_interactions": True,
            "object_type": "digit_block",
            "digit": 3,
        },
        "benchmark_building_blocks_038": {
            "has_interactions": True,
            "object_type": "digit_block",
            "digit": 4,
        },
        "benchmark_building_blocks_039": {
            "has_interactions": True,
            "object_type": "digit_block",
            "digit": 5,
        },
        "benchmark_building_blocks_040": {
            "has_interactions": True,
            "object_type": "digit_block",
            "digit": 6,
        },
        "benchmark_building_blocks_041": {
            "has_interactions": True,
            "object_type": "digit_block",
            "digit": 7,
        },
        "benchmark_building_blocks_042": {
            "has_interactions": True,
            "object_type": "digit_block",
            "digit": 8,
        },
        "benchmark_building_blocks_043": {
            "has_interactions": True,
            "object_type": "digit_block",
            "digit": 9,
        },
        "table_000": {"has_interactions": False, "object_type": "table"},
    }

    ## Runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    ## MANDATORY: Set task description before first action
    task_description = {
        "task_name": "将数字积木按从大到小排列",
        "english_task_name": "arrange digit blocks from large to small",
        "init_scene_text": "The robot is in front of the table, there are digit blocks on the table"
    }
    robot.client.set_task_task_basic_info(task_description)

    ## 1. Find the table
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or "table" in keywords:
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    ## 2. Find all digit blocks
    digit_blocks = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        ## Check if it's a digit block
        is_digit_block = False
        digit_value = None
        for asset_id, asset_info in objects_list.items():
            if asset_id in usd and asset_info["object_type"] == "digit_block":
                is_digit_block = True
                digit_value = asset_info["digit"]
                break

        if is_digit_block:
            pos, rot = robot.get_object_pose(obj_id)
            size = robot.get_object_size_in_world_frame(obj_id)
            if pos is not None and size is not None:
                digit_blocks.append(
                    {
                        "id": obj_id,
                        "usd": usd,
                        "digit": digit_value,
                        "position": pos,
                        "rotation": rot,
                        "size": size,
                        "keywords": keywords,
                    }
                )
                print(f"Found digit block: {obj_id}, digit: {digit_value}, usd: {usd}")

    if len(digit_blocks) < 3:
        print(f"Found only {len(digit_blocks)} digit blocks, need at least 3")
        return False

    print(f"Total digit blocks found: {len(digit_blocks)}")

    ## 3. Sort digit blocks by their digit value (ascending order)
    digit_blocks.sort(key=lambda x: x["digit"], reverse=True)
    print("Digit blocks sorted by value:")
    for i, block in enumerate(digit_blocks):
        print(f"  {i+1}. {block['id']}: digit={block['digit']}")

    ## 4. Calculate target positions based on task requirements
    ## Target x range: (-0.2, -0.15) - select a random x within this range
    target_x = np.random.uniform(-0.05, 0.0)
    print(f"Randomly selected target_x: {target_x:.3f}")

    ## Starting y position: (0.075, 0.025) - select random within this range
    start_y = 0.05
    print(f"Randomly selected start_y: {start_y:.3f}")

    ## Spacing between blocks: 0.05m (5cm)
    spacing = 0.05

    ## Calculate target positions (from left to right, so y decreases as we go right)
    ## Leftmost block has highest y value, then each subsequent block has y decreased by spacing
    target_positions = []

    for i in range(len(digit_blocks)):
        if i == 0:
            target_y = start_y
        else:
            target_y = start_y - spacing * i

        ## Get table height for z-coordinate
        table_pos, table_rot = robot.get_object_pose(table_id)
        table_size = robot.get_object_size_in_world_frame(table_id)
        table_top_z = table_pos[2] + table_size[2] / 2.0

        ## Add safety margin (1cm) above table surface
        margin = 0.01
        target_z = table_top_z + margin + digit_blocks[i]["size"][2] / 2.0

        target_position = np.array([target_x, target_y, target_z])
        target_positions.append(target_position)
        print(
            f"Target position for digit {digit_blocks[i]['digit']}: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}]"
        )

    ## 5. Generate target rotation matrices that satisfy the orientation requirements
    ## Requirements:
    ## - Local y-axis parallel to world z-axis (already satisfied since blocks are on table with y-axis up)
    ## - Local x-axis parallel to world x-axis negative direction
    ## This means we need to rotate the block so its local x-axis points in world -x direction

    def calculate_target_rotation(initial_rotation):
        """
        Calculate the target rotation that aligns local x-axis with world -x direction
        while keeping local y-axis aligned with world z-axis.
        """
        ## Convert initial quaternion to rotation matrix
        initial_rot_matrix = quat2mat_wxyz(initial_rotation)

        ## Get the current local x-axis direction in world coordinates
        local_x_axis = initial_rot_matrix[:, 0]  # First column of rotation matrix

        ## We want local x-axis to point in world -x direction: [-1, 0, 0]
        target_x_direction = np.array([-1.0, 0.0, 0.0])

        ## Since local y-axis should remain aligned with world z-axis [0, 0, 1],
        ## we can compute the required rotation around the world z-axis

        ## Project local x-axis onto xy-plane (since we're rotating around z-axis)
        local_x_xy = local_x_axis[:2]
        if np.linalg.norm(local_x_xy) < 1e-6:
            ## If local x-axis is already aligned with z-axis, use identity rotation
            target_rot_matrix = np.eye(3)
        else:
            local_x_xy_normalized = local_x_xy / np.linalg.norm(local_x_xy)

            ## Calculate angle between current local x-axis projection and target (-1, 0)
            target_xy = np.array([-1.0, 0.0])
            dot_product = np.dot(local_x_xy_normalized, target_xy)
            dot_product = np.clip(
                dot_product, -1.0, 1.0
            )  # Clamp for numerical stability
            angle = np.arccos(dot_product)

            ## Determine rotation direction using cross product
            cross_product = np.cross(
                np.append(local_x_xy_normalized, 0), np.append(target_xy, 0)
            )
            if (
                cross_product[2] < 0
            ):  # Right-hand rule: negative z means clockwise rotation
                angle = -angle

            ## Create rotation matrix around z-axis by the calculated angle
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

            ## Apply this rotation to the initial rotation matrix
            target_rot_matrix = np.dot(rot_z, initial_rot_matrix)

        return target_rot_matrix

    ## 6. Execute placement for each digit block in sorted order
    placed_blocks = []
    last_arm = ""

    for i, block in enumerate(digit_blocks):
        block_id = block["id"]
        block_digit = block["digit"]
        current_pos = block["position"]
        current_rot = block["rotation"]
        block_size = block["size"]

        print(
            f"\n=== Processing digit block {i+1}/{len(digit_blocks)}: digit {block_digit} ({block_id}) ==="
        )

        ## Determine which arm to use
        arm = robot.judge_arm_to_use(current_pos)
        print(f"Using arm: {arm}")

        ## Return previous arm to origin if switching
        if last_arm != arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        ## Create grasp configuration
        unmovable_objects = [table_id]
        unmovable_objects.extend([b["id"] for b in placed_blocks])

        ## Create action description based on digit value
        grasp_config = GraspConfig(
            unmovable_objects=unmovable_objects,
            is_from_up_side=True,
            grasp_offset=-0.007,
            action_description={
                "zh": f"拿起数字{block_digit}",
                "en": f"picks up the digit {block_digit}",
            },
        )

        ## Calculate target rotation for this block
        target_rot_matrix = calculate_target_rotation(current_rot)

        ## Generate multiple candidate poses with small variations for robustness
        target_poses = []
        base_position = target_positions[i]

        ## Position variations (small offsets to increase success probability)
        x_offsets = np.linspace(-0.002, 0.002, 3)
        y_offsets = np.linspace(-0.002, 0.002, 3)
        z_offsets = np.linspace(0, 0.003, 3)

        for x_off in x_offsets:
            for y_off in y_offsets:
                for z_off in z_offsets:
                    position = base_position + np.array([x_off, y_off, z_off])
                    pose = np.eye(4)
                    pose[:3, :3] = target_rot_matrix
                    pose[:3, 3] = position
                    target_poses.append(pose)

        target_poses = np.array(target_poses)
        print(f"Generated {len(target_poses)} candidate poses for digit {block_digit}")

        ## Create placement verification function
        def create_placement_checker(block_id, expected_position):
            def check_placement():
                current_p, current_r = robot.get_object_pose(block_id)
                if current_p is None or current_r is None:
                    return False

                ## Check position (xy tolerance: 3cm, z tolerance: 1cm)
                xy_dist = np.linalg.norm(current_p[:2] - expected_position[:2])

                if xy_dist > 0.03:
                    print(f"Digit {block_digit} position error: xy_dist={xy_dist:.3f}m")
                    return False

                ## Check if block is on table
                table_p, _ = robot.get_object_pose(table_id)
                table_s = robot.get_object_size_in_world_frame(table_id)
                table_top = table_p[2] + table_s[2] / 2.0
                if current_p[2] < table_top:
                    print(f"Digit {block_digit} is below table surface")
                    return False

                # Check rotation
                curr_rot_mat = quat2mat_wxyz(current_r)
                curr_x_axis = curr_rot_mat[:, 0] # Local X in world
                target_x_axis = np.array([-1, 0, 0])
                dot = np.dot(curr_x_axis, target_x_axis)
                if dot < 0.95: # Allow small error
                    print(f"Orientation check failed: dot={dot:.3f}")
                    return False
                return True

            return check_placement

        ## Create placement description
        if i == 0:
            place_desc_zh = f"将数字{block_digit}放置到最左侧位置"
            place_desc_en = f"places the digit {block_digit} at the leftmost position"
        elif i == 1:
            place_desc_zh = f"将数字{block_digit}放置到中间位置"
            place_desc_en = f"places the digit {block_digit} at the middle position"
        else:
            place_desc_zh = f"将数字{block_digit}放置到最右侧位置"
            place_desc_en = f"places the digit {block_digit} at the rightmost position"

        place_config = PlaceConfig(
            unmovable_objects=unmovable_objects,
            extra_checkers=[create_placement_checker(block_id, base_position)],
            action_description={"zh": place_desc_zh, "en": place_desc_en},
            disable_upside_down=False,
        )

        ## Execute pick and place operation
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=block_id,
            target_pose=target_poses,
            arm=arm,
            grasp_config=grasp_config,
            place_config=place_config,
        )

        if not current_success:
            print(f"Failed to place digit {block_digit}")
            success = False
            break
        else:
            print(f"Successfully placed digit {block_digit}")
            placed_blocks.append(block)

    ## 7. Final verification
    print("\n=== Performing final verification ===")

    if len(placed_blocks) == len(digit_blocks):
        ## Verify the arrangement is correct (left to right: largest to smallest)
        final_positions = []
        for block in placed_blocks:
            pos, rot = robot.get_object_pose(block["id"])
            if pos is not None:
                final_positions.append(
                    (block["digit"], pos[1], block["id"])
                )  # (digit, y_position, id)

        ## Sort by y position (descending, since higher y = more left)
        final_positions.sort(key=lambda x: x[1], reverse=True)

        print("Final arrangement from left to right:")
        digits_in_order = []
        for digit, y_pos, block_id in final_positions:
            print(f"  Digit {digit} at y={y_pos:.3f}")
            digits_in_order.append(digit)

        ## Check if digits are in ascending order
        expected_digits = sorted(
            [block["digit"] for block in digit_blocks], reverse=True
        )
        if digits_in_order == expected_digits:
            print(
                "✅ Digits are correctly arranged in ascending order from left to right"
            )
        else:
            print(
                f"⚠️  Digits are not in correct order. Expected: {expected_digits}, Got: {digits_in_order}"
            )
            success = False

    ## Return arms to origin
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("\n=== Task completion summary ===")
    if success and len(placed_blocks) == len(digit_blocks):
        print("✅ SUCCESS: All digit blocks placed in correct order and orientation")
    else:
        print("⚠️ PARTIAL SUCCESS: Some operations failed or verification issues")

    return success and (len(placed_blocks) == len(digit_blocks))
```
