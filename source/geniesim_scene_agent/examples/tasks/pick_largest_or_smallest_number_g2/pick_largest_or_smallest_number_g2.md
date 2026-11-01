# Pick the largest or smallest number block

**Tags:** `pick_only` `digit_blocks` `random_largest_or_smallest` `dynamic_task_description` `numbered_blocks` `digit_to_usd` `sort_by_digit` `GraspConfig` `pick_actor` `extra_checkers` `initial_z` `lift_verification`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find digit blocks (benchmark_building_blocks_034–043). Map usd to digit (0–9). Sort by digit to get min/max.
2. **Choice:** choice = random.choice(["largest", "smallest"]). target_block = block with max digit if largest else min digit. Set dynamic task_description: "pick up the largest numbered block" or "pick up the smallest numbered block".
3. **Verification:** extra_checkers compare the block height before and after pickup to verify lifting.
4. **Execution:** pick_actor(target_block_id, arm, GraspConfig with action_description). No place step.

---

# INPUT

**scene_description:** A table with 3–5 numbered building blocks (digits 0–9, benchmark_building_blocks_034–043) randomly placed. The robot must pick either the block with the largest digit or the block with the smallest digit; the choice (largest vs smallest) is random.

**task description:** Pick up the largest numbered block, or pick up the smallest numbered block (task description set at runtime: random.choice([largest, smallest])). Verify that the selected block is lifted after pickup.

**code to generate the scene:**

```python
from helper import *
import random

"""
scene_name: pick_largest_or_smallest_number_g2
description: A table with 3-5 randomly placed non-repeating numbered building blocks (digits 0-9) in a specified region, with no collisions and minimum 5cm spacing.
"""


def get_tabletop_workspace(table_shape: Shape) -> Tuple[P, P]:
    """
    Get the tabletop surface position and size for placing objects.

    Args:
        table_shape (Shape): The table shape

    Returns:
        Tuple[P, P]: (workspace_center, workspace_size) in world coordinates
    """
    table_info = get_object_info(table_shape)
    desktop_subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")

    # Desktop center in world coordinates
    desktop_center = table_info["center"] + desktop_subpart_info["center"]

    # Top surface z-coordinate
    desktop_top_z = table_info["center"][2] + desktop_subpart_info["xyz_max"][2]

    workspace_center = np.array([desktop_center[0], desktop_center[1], desktop_top_z])
    workspace_size = desktop_subpart_info["size"]

    return workspace_center, workspace_size


def generate_non_colliding_positions(num_blocks: int, min_spacing: float = 0.05) -> List[Tuple[float, float]]:
    """
    Generate non-colliding positions within the specified region.

    Args:
        num_blocks (int): Number of blocks to place
        min_spacing (float): Minimum spacing between blocks in meters

    Returns:
        List[Tuple[float, float]]: List of (x, y) positions
    """
    positions = []
    max_attempts = 1000

    # Define the placement region relative to table center
    x_min, x_max = -0.2, 0.15
    y_min, y_max = -0.3, 0.3

    for i in range(num_blocks):
        for attempt in range(max_attempts):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)

            # Check if this position is far enough from all existing positions
            valid = True
            for existing_x, existing_y in positions:
                distance = ((x - existing_x) ** 2 + (y - existing_y) ** 2) ** 0.5
                if distance < min_spacing:
                    valid = False
                    break

            if valid:
                positions.append((x, y))
                break
        else:
            # If we can't find a valid position after max_attempts, just use the last generated one
            positions.append((x, y))

    return positions


@register()
def numbered_building_block(digit: int) -> Shape:
    """
    Create a numbered building block for a specific digit.

    Args:
        digit (int): The digit (0-9) to create

    Returns:
        Shape: The building block shape
    """
    # Map digits to object IDs
    digit_to_oid = {
        0: "benchmark_building_blocks_034",
        1: "benchmark_building_blocks_035",
        2: "benchmark_building_blocks_036",
        3: "benchmark_building_blocks_037",
        4: "benchmark_building_blocks_038",
        5: "benchmark_building_blocks_039",
        6: "benchmark_building_blocks_040",
        7: "benchmark_building_blocks_041",
        8: "benchmark_building_blocks_042",
        9: "benchmark_building_blocks_043"
    }

    oid = digit_to_oid[digit]
    block_shape = library_call(
        "usd",
        oid=oid,
        keywords=[f"digit_{digit}", "building_block", "colored", "numbered", f"digit_{digit}_block"]
    )

    # Random rotation around z-axis
    block_center = compute_shape_center(block_shape)
    rotation_angle = random.uniform(0, 2 * math.pi)

    rotated_block = transform_shape(
        block_shape,
        rotation_matrix(rotation_angle, direction=(0, 0, 1), point=block_center)
    )

    return rotated_block


@register()
def place_numbered_blocks_on_table(table_shape: Shape) -> Shape:
    """
    Place 3-5 randomly selected non-repeating numbered blocks on the table.

    Args:
        table_shape (Shape): The table shape

    Returns:
        Shape: Table with placed blocks
    """
    # Get table workspace
    workspace_center, workspace_size = get_tabletop_workspace(table_shape)

    # Select random number of blocks (3-5)
    num_blocks = random.randint(3, 5)

    # Select random non-repeating digits
    available_digits = list(range(10))
    selected_digits = random.sample(available_digits, num_blocks)

    # Generate non-colliding positions
    positions = generate_non_colliding_positions(num_blocks, min_spacing=0.05)

    # Place each block
    all_shapes = [table_shape]

    for i, digit in enumerate(selected_digits):
        block_shape = library_call("numbered_building_block", digit=digit)

        # Get block size to ensure proper placement
        block_info = get_object_info(block_shape)
        block_height = block_info["size"][2]

        # Position relative to table center (workspace_center is at table top surface)
        x_pos, y_pos = positions[i]
        world_x = workspace_center[0] + x_pos
        world_y = workspace_center[1] + y_pos
        world_z = workspace_center[2]  # Already at top surface

        positioned_block = transform_shape(
            block_shape,
            translation_matrix([world_x, world_y, world_z])
        )

        all_shapes.append(positioned_block)

    return concat_shapes(*all_shapes)


@register()
def pick_largest_or_smallest_number_g2() -> Shape:
    """
    Create the scene with numbered building blocks on a table for picking largest or smallest number task.
    """
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs", "for_number_blocks"]
    )

    return place_numbered_blocks_on_table(table_shape)


@register()
def root_scene() -> Shape:
    return pick_largest_or_smallest_number_g2()
```


---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random

def execute_task(robot: RobotAPI):
    # Initialize objects_list based on provided interaction data
    objects_list = {
        "benchmark_building_blocks_034": {"has_interactions": True, "object_type": "building_block", "digit": 0},
        "benchmark_building_blocks_035": {"has_interactions": True, "object_type": "building_block", "digit": 1},
        "benchmark_building_blocks_036": {"has_interactions": True, "object_type": "building_block", "digit": 2},
        "benchmark_building_blocks_037": {"has_interactions": True, "object_type": "building_block", "digit": 3},
        "benchmark_building_blocks_038": {"has_interactions": True, "object_type": "building_block", "digit": 4},
        "benchmark_building_blocks_039": {"has_interactions": True, "object_type": "building_block", "digit": 5},
        "benchmark_building_blocks_040": {"has_interactions": True, "object_type": "building_block", "digit": 6},
        "benchmark_building_blocks_041": {"has_interactions": True, "object_type": "building_block", "digit": 7},
        "benchmark_building_blocks_042": {"has_interactions": True, "object_type": "building_block", "digit": 8},
        "benchmark_building_blocks_043": {"has_interactions": True, "object_type": "building_block", "digit": 9},
        "table_000": {"has_interactions": False, "object_type": "table"}
    }

    # Runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions

    success = True

    # MANDATORY: Set task description before first action
    # Randomly choose whether to pick largest or smallest (according to task requirements)
    pick_largest = random.choice([True, False])
    task_target = "最大的数字积木" if pick_largest else "最小的数字积木"
    task_target_en = "the largest numbered block" if pick_largest else "the smallest numbered block"

    description_json = {
        "task_name": f"拿起{task_target}",
        "english_task_name": f"pick up {task_target_en}",
        "init_scene_text": "The robot is in front of the table, there are 3-5 numbered building blocks (digits 0-9) randomly placed on the table"
    }
    robot.client.set_task_task_basic_info(description_json)

    # 1. Find table (table_000)
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or any("table" in kw.lower() for kw in keywords) or any("workspace" in kw.lower() for kw in keywords):
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    print(f"Found table: {table_id}")

    # 2. Find all numbered building blocks (digits 0-9)
    numbered_blocks = []
    digit_to_usd = {
        0: "benchmark_building_blocks_034",
        1: "benchmark_building_blocks_035",
        2: "benchmark_building_blocks_036",
        3: "benchmark_building_blocks_037",
        4: "benchmark_building_blocks_038",
        5: "benchmark_building_blocks_039",
        6: "benchmark_building_blocks_040",
        7: "benchmark_building_blocks_041",
        8: "benchmark_building_blocks_042",
        9: "benchmark_building_blocks_043"
    }

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Check if this is a numbered block by USD
        block_digit = None
        for digit, block_usd in digit_to_usd.items():
            if block_usd in usd:
                block_digit = digit
                break

        # Alternatively, check by keywords
        if block_digit is None:
            for kw in keywords:
                if kw.startswith("digit_") and kw[6:].isdigit():
                    try:
                        block_digit = int(kw[6:])
                        break
                    except ValueError:
                        continue

        if block_digit is not None:
            pos, rot = robot.get_object_pose(obj_id)
            size = robot.get_object_size_in_world_frame(obj_id)

            if pos is not None and size is not None:
                numbered_blocks.append({
                    "id": obj_id,
                    "digit": block_digit,
                    "position": pos,
                    "rotation": rot,
                    "size": size,
                    "usd": usd,
                    "keywords": keywords,
                    "initial_z": pos[2]  # Store initial z position for verification
                })
                print(f"Found numbered block: {obj_id}, digit: {block_digit}, usd: {usd}")

    if len(numbered_blocks) < 3:
        print(f"Found only {len(numbered_blocks)} numbered blocks, need at least 3")
        return False

    print(f"\nTotal numbered blocks found: {len(numbered_blocks)}")
    for block in numbered_blocks:
        print(f"  Block {block['id']}: digit={block['digit']}, position={block['position']}")

    # 3. Determine which block to pick (largest or smallest digit)
    # Sort blocks by digit value
    numbered_blocks.sort(key=lambda x: x["digit"])

    if pick_largest:
        # Pick the largest digit (highest number)
        target_block = numbered_blocks[-1]
        target_desc_zh = f"数字{target_block['digit']}的积木"
        target_desc_en = f"digit {target_block['digit']} block"
        print(f"Randomly selected to pick LARGEST digit: {target_block['digit']} (block: {target_block['id']})")
    else:
        # Pick the smallest digit (lowest number)
        target_block = numbered_blocks[0]
        target_desc_zh = f"数字{target_block['digit']}的积木"
        target_desc_en = f"digit {target_block['digit']} block"
        print(f"Randomly selected to pick SMALLEST digit: {target_block['digit']} (block: {target_block['id']})")

    print(f"\n=== Processing target block ===")
    print(f"Block ID: {target_block['id']}")
    print(f"Digit: {target_block['digit']}")
    print(f"Initial Position: {target_block['position']}")
    print(f"Size: {target_block['size']}")

    # 4. Prepare for grasp operation
    target_id = target_block["id"]
    target_pos = target_block["position"]
    initial_z = target_block["initial_z"]

    # Determine which arm to use
    arm = robot.judge_arm_to_use(target_pos)
    print(f"Using arm: {arm}")

    # Create list of unmovable objects (all other blocks + table)
    unmovable_objects = [table_id]
    for block in numbered_blocks:
        if block["id"] != target_id:
            unmovable_objects.append(block["id"])

    # Determine action description based on whether picking largest or smallest
    if pick_largest:
        action_desc_zh = f"拿起数字{target_block['digit']}的积木"
        action_desc_en = f"pick up digit {target_block['digit']} block"
    else:
        action_desc_zh = f"拿起数字{target_block['digit']}的积木"
        action_desc_en = f"pick up digit {target_block['digit']} block"

    # Create grasp configuration
    grasp_config = GraspConfig(
        unmovable_objects=unmovable_objects,
        action_description={"zh": action_desc_zh, "en": action_desc_en}  # REQUIRED
    )

    # 5. Execute grasp operation (only pick up, don't place down)
    print(f"\nAttempting to grasp block {target_id} (digit: {target_block['digit']})")

    # Since all blocks have interactions (as specified), use pick_actor
    print(f"Using pick_actor with interaction primitives for block {target_id}")
    success = robot.pick_actor(
        actor=target_id,
        arm=arm,
        grasp_config=grasp_config
    )

    # 6. Final verification - check if final Z position is at least 3cm higher than initial
    if success:
        print(f"\nVerifying that block was lifted at least 3cm...")

        # Get current position after grasp
        final_pos, final_rot = robot.get_object_pose(target_id)

        if final_pos is not None:
            z_difference = final_pos[2] - initial_z
            print(f"Initial Z: {initial_z:.3f}m")
            print(f"Final Z: {final_pos[2]:.3f}m")
            print(f"Z difference: {z_difference:.3f}m")

            if z_difference >= 0.03:  # At least 3cm higher
                print("✅ SUCCESS: Block was lifted at least 3cm")
                success = True
            else:
                print(f"❌ FAILED: Block was only lifted {z_difference*100:.1f}cm, less than required 3cm")
                success = False
        else:
            print("❌ FAILED: Cannot get final position of block")
            success = False
    else:
        print(f"\n❌ FAILED: Failed to grasp block {target_id} (digit: {target_block['digit']})")


    # 7. Final summary
    print(f"\n=== Task summary ===")
    print(f"Task: Pick {'largest' if pick_largest else 'smallest'} numbered block")
    print(f"Target digit: {target_block['digit']}")
    print(f"Total blocks on table: {len(numbered_blocks)}")
    print(f"Success: {'YES' if success else 'NO'}")

    return success
```
