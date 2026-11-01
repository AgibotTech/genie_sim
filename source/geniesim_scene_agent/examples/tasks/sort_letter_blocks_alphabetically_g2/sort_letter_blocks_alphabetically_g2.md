# Sort letter blocks alphabetically

**Tags:** `sort` `letter_blocks` `alphabetical_order` `table_only` `pick_and_place` `linear_arrangement` `spacing` `GraspConfig` `PlaceConfig` `extra_checkers`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table by usd "table_000" or keywords. Find all letter blocks by matching usd against benchmark_letter_XXX pattern.
2. **Letter identification:** Extract letter value from each block using objects_list mapping (benchmark_letter_026 -> 'A', etc.).
3. **Sorting:** Sort letter blocks alphabetically (A to Z) using letter_order mapping. This determines placement order.
4. **Target positions:** Calculate linear arrangement: target_x in (-0.05, 0), start_y in (0.05, 0.08), spacing 0.07m between letters. Letters placed left-to-right (y decreasing).
5. **Placement:** For each letter in alphabetical order, pick and place at calculated position. Use multiple candidate poses for robustness.
6. **Verification:** Check final positions form correct alphabetical sequence from left to right.

---

# INPUT

**scene_description:** A table with three randomly selected letter blocks placed within a specific region, avoiding a central sub-region. Each pair of letters is at least 0.05m apart.

**task description:** Arrange the letter blocks on the table in alphabetical order from A to Z, from left to right.

**code to generate the scene:**

```python
from helper import *
import random

"""
scene_name: table_with_random_letters
description: A table with three randomly selected letters placed within a specific region, avoiding a central sub-region.
"""

def get_tabletop_placement_area(table_shape: Shape) -> Tuple[P, P]:
    table_info = get_object_info(table_shape)
    desktop_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_center_world = table_info["center"] + desktop_info["center"]
    desktop_top_z = table_info["center"][2] + desktop_info["xyz_max"][2]
    return np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z]), desktop_info["size"]

def positions_too_close(pos1: P, pos2: P, min_dist: float = 0.05) -> bool:
    return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2])) < min_dist

@register()
def place_single_letter(letter_oid: str, letter_char: str, table_top_center: P, existing_positions: List[P]) -> Shape:
    letter_shape = library_call("usd", oid=letter_oid, keywords=[f"letter_{letter_char}", "alphabet"])
    x_min_allowed, x_max_allowed = -0.2, 0.1
    y_min_allowed, y_max_allowed = -0.2, 0.2
    forbidden_x_min, forbidden_x_max = 0.0, 0.07
    forbidden_y_min, forbidden_y_max = -0.07, 0.07
    min_letter_spacing = 0.05
    for _ in range(1000):
        x_offset = random.uniform(x_min_allowed, x_max_allowed)
        y_offset = random.uniform(y_min_allowed, y_max_allowed)
        if (forbidden_x_min < x_offset < forbidden_x_max) and (forbidden_y_min < y_offset < forbidden_y_max):
            continue
        candidate_pos = [table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]]
        too_close = any(positions_too_close(candidate_pos, ep, min_letter_spacing) for ep in existing_positions)
        if too_close:
            continue
        break
    return transform_shape(letter_shape, translation_matrix([table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]]))

@register()
def table_with_random_letters() -> Shape:
    table_shape = library_call("usd", oid="table_000", keywords=["table"])
    table_top_center, _ = get_tabletop_placement_area(table_shape)
    letter_map = {
        'A': 'benchmark_letter_026', 'B': 'benchmark_letter_027', 'C': 'benchmark_letter_028',
        'D': 'benchmark_letter_029', 'E': 'benchmark_letter_030', 'F': 'benchmark_letter_031',
        'G': 'benchmark_letter_032', 'H': 'benchmark_letter_033', 'I': 'benchmark_letter_034',
        'J': 'benchmark_letter_035', 'K': 'benchmark_letter_036', 'L': 'benchmark_letter_037',
        'M': 'benchmark_letter_038', 'N': 'benchmark_letter_039', 'O': 'benchmark_letter_040',
        'P': 'benchmark_letter_041', 'Q': 'benchmark_letter_042', 'R': 'benchmark_letter_043',
        'S': 'benchmark_letter_044', 'T': 'benchmark_letter_045', 'U': 'benchmark_letter_046',
        'V': 'benchmark_letter_047', 'W': 'benchmark_letter_048', 'X': 'benchmark_letter_049',
        'Y': 'benchmark_letter_050', 'Z': 'benchmark_letter_051',
    }
    selected_chars = random.sample(list(letter_map.keys()), 3)
    letter_shapes = []
    placed_positions = []
    for char in selected_chars:
        letter_shape = library_call("place_single_letter", letter_oid=letter_map[char], letter_char=char, table_top_center=table_top_center, existing_positions=placed_positions)
        letter_info = get_object_info(letter_shape)
        placed_positions.append(letter_info["center"])
        letter_shapes.append(letter_shape)
    return concat_shapes(table_shape, *letter_shapes)

@register()
def root_scene() -> Shape:
    return table_with_random_letters()
```

---

# OUTPUT

```python
from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    objects_list = {
        "benchmark_letter_026": {"has_interactions": True, "object_type": "letter_block", "letter": "A"},
        "benchmark_letter_027": {"has_interactions": True, "object_type": "letter_block", "letter": "B"},
        "benchmark_letter_028": {"has_interactions": True, "object_type": "letter_block", "letter": "C"},
        "benchmark_letter_029": {"has_interactions": True, "object_type": "letter_block", "letter": "D"},
        "benchmark_letter_030": {"has_interactions": True, "object_type": "letter_block", "letter": "E"},
        "benchmark_letter_031": {"has_interactions": True, "object_type": "letter_block", "letter": "F"},
        "benchmark_letter_032": {"has_interactions": True, "object_type": "letter_block", "letter": "G"},
        "benchmark_letter_033": {"has_interactions": True, "object_type": "letter_block", "letter": "H"},
        "benchmark_letter_034": {"has_interactions": True, "object_type": "letter_block", "letter": "I"},
        "benchmark_letter_035": {"has_interactions": True, "object_type": "letter_block", "letter": "J"},
        "benchmark_letter_036": {"has_interactions": True, "object_type": "letter_block", "letter": "K"},
        "benchmark_letter_037": {"has_interactions": True, "object_type": "letter_block", "letter": "L"},
        "benchmark_letter_038": {"has_interactions": True, "object_type": "letter_block", "letter": "M"},
        "benchmark_letter_039": {"has_interactions": True, "object_type": "letter_block", "letter": "N"},
        "benchmark_letter_040": {"has_interactions": True, "object_type": "letter_block", "letter": "O"},
        "benchmark_letter_041": {"has_interactions": True, "object_type": "letter_block", "letter": "P"},
        "benchmark_letter_042": {"has_interactions": True, "object_type": "letter_block", "letter": "Q"},
        "benchmark_letter_043": {"has_interactions": True, "object_type": "letter_block", "letter": "R"},
        "benchmark_letter_044": {"has_interactions": True, "object_type": "letter_block", "letter": "S"},
        "benchmark_letter_045": {"has_interactions": True, "object_type": "letter_block", "letter": "T"},
        "benchmark_letter_046": {"has_interactions": True, "object_type": "letter_block", "letter": "U"},
        "benchmark_letter_047": {"has_interactions": True, "object_type": "letter_block", "letter": "V"},
        "benchmark_letter_048": {"has_interactions": True, "object_type": "letter_block", "letter": "W"},
        "benchmark_letter_049": {"has_interactions": True, "object_type": "letter_block", "letter": "X"},
        "benchmark_letter_050": {"has_interactions": True, "object_type": "letter_block", "letter": "Y"},
        "benchmark_letter_051": {"has_interactions": True, "object_type": "letter_block", "letter": "Z"},
        "table_000": {"has_interactions": False, "object_type": "table"},
    }

    scene_info = robot.scene_info
    interactions = robot.interactions
    success = True

    task_description = {
        "task_name": "将字母积木按从A到Z排列",
        "english_task_name": "arrange letter blocks from A to Z",
        "init_scene_text": "The robot is in front of the table, there are letter blocks on the table"
    }
    robot.client.set_task_task_basic_info(task_description)

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

    letter_blocks = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        is_letter_block = False
        letter_value = None
        for asset_id, asset_info in objects_list.items():
            if asset_id in usd and asset_info["object_type"] == "letter_block":
                is_letter_block = True
                letter_value = asset_info["letter"]
                break
        if is_letter_block:
            pos, rot = robot.get_object_pose(obj_id)
            size = robot.get_object_size_in_world_frame(obj_id)
            if pos is not None and size is not None:
                letter_blocks.append({"id": obj_id, "usd": usd, "letter": letter_value, "position": pos, "rotation": rot, "size": size})

    if len(letter_blocks) < 3:
        print(f"Found only {len(letter_blocks)} letter blocks, need at least 3")
        return False

    letter_order = {chr(i): i - ord('A') for i in range(ord('A'), ord('Z') + 1)}
    letter_blocks.sort(key=lambda x: letter_order.get(x["letter"], 0))

    target_x = np.random.uniform(-0.05, 0.0)
    start_y = np.random.uniform(0.05, 0.08)
    spacing = 0.07

    target_positions = []
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0
    margin = 0.01

    for i in range(len(letter_blocks)):
        target_y = start_y - spacing * i
        target_z = table_top_z + margin + letter_blocks[i]["size"][2] * 3 / 2.0
        target_positions.append(np.array([target_x, target_y, target_z]))

    def calculate_target_rotation(initial_rotation):
        return quat2mat_wxyz(initial_rotation)

    placed_blocks = []
    last_arm = ""

    for i, block in enumerate(letter_blocks):
        block_id = block["id"]
        block_letter = block["letter"]
        current_pos = block["position"]
        current_rot = block["rotation"]

        arm = robot.judge_arm_to_use(current_pos)
        if last_arm != arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        unmovable_objects = [table_id] + [b["id"] for b in placed_blocks]

        grasp_config = GraspConfig(
            unmovable_objects=unmovable_objects,
            action_description={"zh": f"拿起字母{block_letter}", "en": f"picks up the letter {block_letter}"},
        )

        target_rot_matrix = calculate_target_rotation(current_rot)
        target_poses = []
        base_position = target_positions[i]
        for x_off in np.linspace(-0.002, 0.002, 3):
            for y_off in np.linspace(-0.002, 0.002, 3):
                for z_off in np.linspace(0, 0.003, 3):
                    position = base_position + np.array([x_off, y_off, z_off])
                    pose = np.eye(4)
                    pose[:3, :3] = target_rot_matrix
                    pose[:3, 3] = position
                    target_poses.append(pose)

        def create_placement_checker(block_id, expected_position, block_letter):
            def check_placement():
                current_p, current_r = robot.get_object_pose(block_id)
                if current_p is None or current_r is None:
                    return False
                xy_dist = np.linalg.norm(current_p[:2] - expected_position[:2])
                if xy_dist > 0.03:
                    return False
                return True
            return check_placement

        place_config = PlaceConfig(
            unmovable_objects=unmovable_objects,
            extra_checkers=[create_placement_checker(block_id, base_position, block_letter)],
            action_description={"zh": f"将字母{block_letter}放置到位置{i+1}", "en": f"places the letter {block_letter} at position {i+1}"},
            disable_upside_down=False,
        )

        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=block_id, target_pose=np.array(target_poses), arm=arm,
            grasp_config=grasp_config, place_config=place_config,
        )

        if not current_success:
            success = False
            return success
        placed_blocks.append(block)

    if len(placed_blocks) == len(letter_blocks):
        final_positions = []
        for block in placed_blocks:
            pos, rot = robot.get_object_pose(block["id"])
            if pos is not None:
                final_positions.append((block["letter"], pos[1]))
        final_positions.sort(key=lambda x: x[1], reverse=True)
        letters_in_order = [letter for letter, _ in final_positions]
        expected_letters = sorted([block["letter"] for block in letter_blocks], key=lambda x: letter_order.get(x, 0))
        if letters_in_order != expected_letters:
            success = False

    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    return success and (len(placed_blocks) == len(letter_blocks))
```
