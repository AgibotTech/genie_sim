# Stack blocks by instruction

**Tags:** `stack` `pick_and_place` `building_blocks` `table_only` `pick_and_place_actor_to_target_pose` `extra_checkers` `euler2mat` `multiple_candidate_poses` `dynamic_task_description` `keywords` `usd_match` `random_order` `color_blocks`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table by usd "table_000" or keyword "table". Find 3 scattered blocks (red/yellow/purple) by matching usd ("benchmark_building_blocks_073/085/089") and keyword "scattered".
2. **Random stacking order:** Use `np.random.permutation` to generate a random bottom-to-top order of the 3 colors. Build dynamic task description with Chinese/English names based on this order.
3. **Target pose generation:** Stack at table center (`table_pos[:2]`). For each block, compute target Z from table top or previous block top + margin. Generate candidate poses with `euler2mat` upright rotations and small XY offsets.
4. **Execute stacking:** For each block in order, `pick_and_place_actor_to_target_pose` with `extra_checkers` verifying placement and alignment with the expected stack position. Add previously placed blocks to `unmovable_objects`.
5. **Arm management:** Track last arm used; call `back_to_origin` when switching arms.

---

# INPUT

**scene_description:** A table with three scattered building blocks (red, yellow, purple) placed in two regions.

**task description:** Randomly determine a bottom-to-top color order and stack the three blocks accordingly.

**code to generate the scene:**

```python
from helper import *
import numpy as np
import math

"""
scene_name: table_with_scattered_and_stacked_blocks
description: A table workspace (table_000) with exactly three scattered building blocks (red, yellow, purple)
placed in two specific regions with a minimum center distance of 0.1m.
A stack of three blocks is placed in a separate demonstration area.
"""


def get_table_top_z(table_shape: Shape) -> float:
    """
    Helper to get the Z height of the table top surface.
    """
    table_info = get_object_info(table_shape)
    try:
        subpart_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
        top_z = table_info["center"][2] + subpart_info["xyz_max"][2]
    except:
        top_z = table_info["max"][2]
    return top_z


@register()
def scattered_blocks(table_shape: Shape) -> Shape:
    """
    Places exactly 3 blocks (Red, Yellow, Purple) in two specific regions on the table.
    Regions:
    1. x(-0.05, 0.1), y(-0.2, -0.1)
    2. x(-0.05, 0.1), y(0.1, 0.2)
    Constraint: Center distance >= 0.1m.
    """
    top_z = get_table_top_z(table_shape)

    blocks = [
        {"oid": "benchmark_building_blocks_073", "color": "red", "name": "red_block"},
        {"oid": "benchmark_building_blocks_085", "color": "yellow", "name": "yellow_block"},
        {"oid": "benchmark_building_blocks_089", "color": "purple", "name": "purple_block"},
    ]

    positions = []
    shapes = []

    # Place 3 blocks
    for block in blocks:
        placed = False
        attempts = 0
        while not placed and attempts < 500:
            # Randomly choose a region to ensure distribution
            region = np.random.choice([0, 1])

            if region == 0:
                x = np.random.uniform(-0.05, 0.1)
                y = np.random.uniform(-0.2, -0.1)
            else:
                x = np.random.uniform(-0.05, 0.1)
                y = np.random.uniform(0.1, 0.2)

            # Check distance constraint (>= 0.1m)
            valid = True
            for px, py in positions:
                if np.hypot(x - px, y - py) < 0.1:
                    valid = False
                    break

            if valid:
                positions.append((x, y))

                # Create block
                block_shape = library_call(
                    "usd",
                    oid=block["oid"],
                    keywords=[block["name"], "building_block", block["color"], "cube", "scattered"],
                )

                # Translate to table surface
                block_shape = transform_shape(block_shape, translation_matrix([x, y, top_z]))

                # Random rotation +/- 10 degrees
                angle = np.random.uniform(-math.pi / 18, math.pi / 18)
                center = compute_shape_center(block_shape)
                block_shape = transform_shape(block_shape, rotation_matrix(angle, direction=(0, 0, 1), point=center))

                shapes.append(block_shape)
                placed = True
            attempts += 1

    return concat_shapes(*shapes)


@register()
def root_scene() -> Shape:
    """
    Main scene composition.
    """
    table_shape = library_call(
        "usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"]
    )

    scattered = library_call("scattered_blocks", table_shape=table_shape)

    return concat_shapes(table_shape, scattered)
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    # Initialize objects_list
    objects_list = {
        "benchmark_building_blocks_073": {"has_interactions": False},
        "benchmark_building_blocks_085": {"has_interactions": False},
        "benchmark_building_blocks_089": {"has_interactions": False},
        "table_000": {"has_interactions": False}
    }

    scene_info = robot.scene_info

    # 1. Find Table
    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        if obj_info.get("usd") == "table_000" or "table" in obj_info.get("keywords", []):
            table_id = obj_id
            break

    if not table_id:
        print("Table not found")
        return False

    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    # Calculate table top surface Z height
    table_top_z = table_pos[2] + table_size[2] / 2.0

    # 2. Find Scattered Blocks (No demo stack in scene anymore)
    scattered_blocks = {}
    color_to_usd = {
        "red": "benchmark_building_blocks_073",
        "yellow": "benchmark_building_blocks_085",
        "purple": "benchmark_building_blocks_089"
    }
    color_to_zh = {
        "red": "红",
        "yellow": "黄",
        "purple": "紫"
    }

    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        # Identify the block color based on USD and ensure it's a scattered block
        for color, target_usd in color_to_usd.items():
            if usd == target_usd and "scattered" in keywords:
                scattered_blocks[color] = {"id": obj_id, "usd": usd, "color": color}

    if len(scattered_blocks) != 3:
        print("Did not find exactly 3 scattered blocks")
        return False

    # 3. Determine Stacking Order (Randomly generate from bottom to top)
    colors = ["red", "yellow", "purple"]
    stack_order = list(np.random.permutation(colors))

    # 4. Set Task Description based on the random order
    bottom_zh = color_to_zh[stack_order[0]]
    middle_zh = color_to_zh[stack_order[1]]
    top_zh = color_to_zh[stack_order[2]]

    task_name = f"按照{bottom_zh}{middle_zh}{top_zh}从下到上的顺序堆叠积木"
    english_task_name = f"stack the blocks from bottom to top in the order of {stack_order[0]}, {stack_order[1]}, {stack_order[2]}"
    print(task_name)
    description_json = {
        "task_name": task_name,
        "english_task_name": english_task_name,
        "init_scene_text": "The robot is in front of a table with three scattered building blocks (red, yellow, purple)."
    }
    robot.client.set_task_task_basic_info(description_json)

    # 5. Map Scattered Blocks to Order
    ordered_blocks = []
    for color in stack_order:
        ordered_blocks.append(scattered_blocks[color])

    # 6. Define Target Stacking Position
    # Stack at the center of the table
    target_xy = table_pos[:2]

    # Block size and margin
    block_size = 0.04
    margin = 0.01

    # 7. Generate candidate upright rotations
    # Since there's no demo stack to copy orientation from, we generate upright Z-axis poses.
    # Assuming default block Y-axis is up, rotate 90 deg around X to make Z up.
    base_rotations = [
        euler2mat([np.pi/2, 0, 0]),   # Y -> Z
        euler2mat([-np.pi/2, 0, 0]),  # -Y -> Z
    ]
    z_rot_angles = np.linspace(0, 2*np.pi, 4, endpoint=False) # 8 orientations

    # 8. Execute Stacking
    placed_blocks = []
    last_arm = ""

    for i, block_info in enumerate(ordered_blocks):
        block_id = block_info["id"]
        color = block_info["color"]

        print(f"\n--- Stacking block {i+1}/3: {color} ({block_id}) ---")

        block_pos, _ = robot.get_object_pose(block_id)
        arm = robot.judge_arm_to_use(block_pos)

        # Calculate target Z height
        if i == 0:
            target_z = table_top_z + margin + block_size / 2.0
        else:
            prev_block_id = placed_blocks[-1]
            prev_pos, _ = robot.get_object_pose(prev_block_id)
            prev_top_z = prev_pos[2] + block_size / 2.0
            target_z = prev_top_z + margin + block_size / 2.0

        target_position = np.array([target_xy[0], target_xy[1], target_z])

        # Generate Candidate Poses (144 combinations)
        target_poses = []
        x_offsets = np.linspace(-0.002, 0.002, 3)
        y_offsets = np.linspace(-0.002, 0.002, 3)

        for dx in x_offsets:
            for dy in y_offsets:
                pos = target_position + np.array([dx, dy, 0])
                for base_rot in base_rotations:
                    for z_angle in z_rot_angles:
                        rot_z = euler2mat([0, 0, z_angle])
                        final_rot = np.dot(rot_z, base_rot)
                        pose = np.eye(4)
                        pose[:3, :3] = final_rot
                        pose[:3, 3] = pos
                        target_poses.append(pose)

        target_poses = np.array(target_poses)

        # Unmovable objects: Table + already placed blocks
        unmovable = [table_id] + placed_blocks

        # Checker function to verify successful placement
        def create_checker(b_id, exp_pos):
            def check():
                curr_pos, _ = robot.get_object_pose(b_id)
                if curr_pos is None: return False
                # Check XY distance from target center
                dist_xy = np.linalg.norm(curr_pos[:2] - exp_pos[:2])
                if dist_xy > 0.02: return False
                # Check Z height difference
                if abs(curr_pos[2] - exp_pos[2]) > 0.02: return False
                return True
            return check

        checker = create_checker(block_id, target_position)

        # Action Descriptions dynamically based on color and position
        position_zh = ["最底层", "中间层", "最顶层"][i]
        position_en = ["bottom", "middle", "top"][i]
        color_zh = color_to_zh[color]

        grasp_desc = {"zh": f"拿起{color_zh}积木", "en": f"picks up the {color} block"}
        place_desc = {"zh": f"将{color_zh}积木放置到塔的{position_zh}", "en": f"places the {color} block on the {position_en} of the tower"}

        grasp_config = GraspConfig(
            unmovable_objects=unmovable,
            action_description=grasp_desc
        )

        place_config = PlaceConfig(
            unmovable_objects=unmovable,
            extra_checkers=[checker],
            action_description=place_desc
        )

        # Return previous arm to origin if switching arms
        if last_arm != "" and last_arm != arm:
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        success = robot.pick_and_place_actor_to_target_pose(
            actor=block_id,
            target_pose=target_poses,
            arm=arm,
            grasp_config=grasp_config,
            place_config=place_config
        )

        if not success:
            print(f"Failed to stack block {block_id}")
            return False

        placed_blocks.append(block_id)
        print(f"Successfully stacked block {block_id}")

    # Return arm to origin at the end
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    print("Task completed successfully")
    return True
```
