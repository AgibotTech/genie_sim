# Complete a word with a letter block

**Tags:** `reasoning` `letter_blocks` `fill_in_blank` `table_only` `pick_and_place` `word_completion` `dynamic_task_description` `extra_checkers` `GraspConfig` `PlaceConfig`

**Task flow (how to implement and key points):**
1. **Scene analysis:** Find table and all letter blocks. Identify the word sequence (aligned letters in x: 0-0.05 region) and scattered letters.
2. **Word detection:** Sort word sequence blocks by y-position (descending: +Y to -Y, left to right). Analyze spacing to detect gaps indicating missing letter position.
3. **Word matching:** Use WORD_LIST to find target word whose letters match the displayed letters plus one missing. Determine which letter is missing and at what position.
4. **Correct letter identification:** Find the correct missing letter among scattered blocks. Calculate target position based on adjacent letters in the word.
5. **Placement:** Pick the correct letter and place it in the gap, aligned with other word letters (same X, appropriate Y based on position).
6. **Verification:** Check final position and verify complete word matches target.

---

# INPUT

**scene_description:** A table with a 4-letter word displayed as a fill-in-the-blank puzzle. The word letters (except one missing) are arranged neatly in a row from left to right (+Y to -Y direction) at the same X position (0-0.05), spaced 0.06m apart. The missing letter is at position 1 or 2 (middle two of four). Additionally, four letters are placed randomly in the wider area (avoiding central forbidden zone): one is the correct missing letter, and three are distractors.

**task description:** Select the appropriate letter from the scattered letters to fill the missing position in the word.

**code to generate the scene:**

```python
from helper import *
import random

"""
scene_name: table_with_fill_in_blank_letters
description: A table with a 4-letter word displayed as a fill-in-the-blank puzzle. The complete word letters (except one missing) are arranged neatly in a row from left to right (+Y to -Y direction) at the same X position (0-0.05), with the first letter's Y in (0.04, 0.08). Letters are spaced 0.06m apart along the Y-axis. The missing letter is always at position 1 or 2 (0-indexed middle two of four). Additionally, four letters are placed randomly in the wider area (avoiding central forbidden zone): one is the correct missing letter, and three are distractors. All letters are unique.
"""

def get_tabletop_placement_area(table_shape: Shape) -> Tuple[P, P]:
    table_info = get_object_info(table_shape)
    desktop_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_center_world = table_info["center"] + desktop_info["center"]
    desktop_top_z = table_info["center"][2] + desktop_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z])
    desktop_size = desktop_info["size"]
    return desktop_pos, desktop_size

def positions_too_close(pos1: P, pos2: P, min_dist: float = 0.07) -> bool:
    dist = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
    return dist < min_dist

@register()
def place_word_letters_aligned(word_chars: List[str], missing_index: int, table_top_center: P) -> Shape:
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
    assert len(word_chars) == 4, "Word must be exactly 4 letters long"
    x_offset = random.uniform(0.0, 0.05)
    first_y_offset = random.uniform(0.04, 0.08)
    letter_shapes = []
    for i, char in enumerate(word_chars):
        if i == missing_index:
            continue
        oid = letter_map[char]
        letter_shape = library_call("usd", oid=oid, keywords=[f"letter_{char}", "alphabet", "plastic", "educational", "word_sequence"])
        y_offset = first_y_offset - i * 0.06
        placed_letter = transform_shape(letter_shape, translation_matrix([table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]]))
        letter_shapes.append(placed_letter)
    return concat_shapes(*letter_shapes) if letter_shapes else []

@register()
def place_random_letter(letter_oid: str, letter_char: str, table_top_center: P, existing_positions: List[P], is_correct: bool = False) -> Shape:
    role_tag = "correct_answer" if is_correct else "distractor"
    letter_shape = library_call("usd", oid=letter_oid, keywords=[f"{role_tag}_{letter_char}", "alphabet", "plastic", "educational", role_tag])
    letter_min = compute_shape_min(letter_shape)
    letter_max = compute_shape_max(letter_shape)
    letter_size = letter_max - letter_min
    x_min_allowed, x_max_allowed = -0.2, 0.15
    y_min_allowed, y_max_allowed = -0.25, 0.25
    forbidden_x_min, forbidden_x_max = -0.05, 0.08
    forbidden_y_min, forbidden_y_max = -0.1, 0.1
    min_letter_spacing = 0.07
    for _ in range(2000):
        x_offset = random.uniform(x_min_allowed, x_max_allowed)
        y_offset = random.uniform(y_min_allowed, y_max_allowed)
        candidate_pos = [table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]]
        if (forbidden_x_min < x_offset < forbidden_x_max) and (forbidden_y_min < y_offset < forbidden_y_max):
            continue
        if (abs(x_offset) + letter_size[0] / 2 > 0.25) or (abs(y_offset) + letter_size[1] / 2 > 0.25):
            continue
        too_close = False
        for existing_pos in existing_positions:
            if positions_too_close(candidate_pos, existing_pos, min_dist=min_letter_spacing):
                too_close = True
                break
        if too_close:
            continue
        break
    else:
        fallback_positions = [[-0.18, 0.22], [-0.18, -0.22], [0.12, 0.22], [0.12, -0.22]]
        candidate_pos = [table_top_center[0] + fallback_positions[0][0], table_top_center[1] + fallback_positions[0][1], table_top_center[2]]
    return transform_shape(letter_shape, translation_matrix(candidate_pos))

EXPANDED_WORD_LIST = [
    "cat", "dog", "sun", "sky", "pig", "fox", "cow", "bus", "car", "pen",
    "cup", "box", "hat", "boy", "run", "fly", "eat", "old", "new", "hot",
    "bat", "man", "red", "big", "sad", "mad", "top", "map", "toy", "leg",
    "bird", "fish", "bear", "lion", "wolf", "frog", "duck", "hand", "time", "song",
    "play", "work", "read", "jump", "fast", "slow", "cold", "warm", "blue", "pink",
    # ... more words
]

def has_unique_letters(word: str) -> bool:
    return len(set(word)) == len(word)

VALID_WORDS = [word.upper() for word in EXPANDED_WORD_LIST if len(word) == 4 and has_unique_letters(word.upper())]
ALL_LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

@register()
def table_with_fill_in_blank_letters() -> Shape:
    table_shape = library_call("usd", oid="table_000", keywords=["workspace_table", "table", "white", "desktop_with_legs"])
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
    selected_word = random.choice(VALID_WORDS) if VALID_WORDS else "BEAD"
    word_chars = list(selected_word)
    missing_index = random.choice([1, 2])
    missing_char = word_chars[missing_index]
    word_letters_shape = library_call("place_word_letters_aligned", word_chars=word_chars, missing_index=missing_index, table_top_center=table_top_center)
    available_distractors = [letter for letter in ALL_LETTERS if letter not in word_chars and letter in letter_map]
    distractor_chars = random.sample(available_distractors, 3) if len(available_distractors) >= 3 else ['X', 'Y', 'Z']
    random_letters = [missing_char] + distractor_chars
    placed_positions = []
    random_letter_shapes = []
    for i, char in enumerate(random_letters):
        oid = letter_map[char]
        is_correct = (i == 0)
        random_letter_shape = library_call("place_random_letter", letter_oid=oid, letter_char=char, table_top_center=table_top_center, existing_positions=placed_positions, is_correct=is_correct)
        letter_info = get_object_info(random_letter_shape)
        placed_positions.append(letter_info["center"])
        random_letter_shapes.append(random_letter_shape)
    return concat_shapes(table_shape, word_letters_shape, *random_letter_shapes)

@register()
def root_scene() -> Shape:
    return table_with_fill_in_blank_letters()
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

    WORD_LIST = [
        "cat", "dog", "sun", "sky", "pig", "fox", "cow", "bus", "car", "pen",
        "cup", "box", "hat", "boy", "run", "fly", "eat", "old", "new", "hot",
        "bat", "man", "red", "big", "sad", "mad", "top", "map", "toy", "leg",
        "bird", "fish", "bear", "lion", "wolf", "frog", "duck", "hand", "time", "song",
        # ... more words
    ]
    WORD_LIST_UPPER = [word.upper() for word in WORD_LIST]

    scene_info = robot.scene_info
    interactions = robot.interactions
    success = True

    task_description = {
        "task_name": "选择合适的字母填入单词的空缺位置",
        "english_task_name": "select the appropriate letter to fill the missing position in the word",
        "init_scene_text": "The robot is in front of the table, there is a word with a missing letter displayed as a fill-in-the-blank puzzle, and candidate letters are scattered on the table"
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
                letter_blocks.append({
                    "id": obj_id, "usd": usd, "letter": letter_value,
                    "position": pos, "rotation": rot, "size": size, "keywords": obj_info.get("keywords", []),
                })

    word_sequence_blocks = []
    scattered_blocks = []
    for block in letter_blocks:
        if -0.02 <= block["position"][0] <= 0.05:
            word_sequence_blocks.append(block)
        else:
            scattered_blocks.append(block)

    word_sequence_blocks.sort(key=lambda x: x["position"][1], reverse=True)

    word_letters = [block["letter"] for block in word_sequence_blocks]
    word_positions = [block["position"][1] for block in word_sequence_blocks]

    target_word = None
    missing_letter = None
    missing_index_in_word = None

    if len(word_positions) >= 2:
        spacings = []
        for i in range(1, len(word_positions)):
            spacings.append(abs(word_positions[i-1] - word_positions[i]))
        avg_spacing = np.mean(spacings) if spacings else 0.06

        all_letters_on_table = [block["letter"] for block in letter_blocks]
        all_letters_set = set(all_letters_on_table)

        possible_words = []
        for word in WORD_LIST_UPPER:
            word_letters_set = set(word)
            if word_letters_set.issubset(all_letters_set):
                displayed_set = set(word_letters)
                if displayed_set.issubset(word_letters_set):
                    missing_letters = word_letters_set - displayed_set
                    if len(missing_letters) == 1:
                        possible_words.append((word, list(missing_letters)[0]))

        if possible_words:
            target_word, missing_letter = possible_words[0]
            target_word_list = list(target_word)
            missing_index_in_word = target_word_list.index(missing_letter)

    if target_word is None or missing_letter is None or missing_index_in_word is None:
        print("Could not determine target word and missing position")
        return False

    correct_answer_block = None
    for block in scattered_blocks:
        if block["letter"] == missing_letter:
            correct_answer_block = block
            break

    if correct_answer_block is None:
        print(f"Missing letter {missing_letter} not found among scattered blocks")
        return False

    letter_to_block = {block["letter"]: block for block in word_sequence_blocks}
    word_x = word_sequence_blocks[0]["position"][0]
    target_y = None

    target_word_list = list(target_word)
    for i in range(missing_index_in_word - 1, -1, -1):
        ref_letter = target_word_list[i]
        if ref_letter in letter_to_block:
            ref_y = letter_to_block[ref_letter]["position"][1]
            target_y = ref_y - (missing_index_in_word - i) * avg_spacing
            break

    if target_y is None:
        for i in range(missing_index_in_word + 1, len(target_word_list)):
            ref_letter = target_word_list[i]
            if ref_letter in letter_to_block:
                ref_y = letter_to_block[ref_letter]["position"][1]
                target_y = ref_y + (i - missing_index_in_word) * avg_spacing
                break

    if target_y is None:
        print("Could not calculate target position")
        return False

    target_x = word_x
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0
    margin = 0.01
    target_z = table_top_z + margin + correct_answer_block["size"][2] * 3 / 2.0
    target_position = np.array([target_x, target_y, target_z])

    arm = robot.judge_arm_to_use(correct_answer_block["position"])

    unmovable_objects = [table_id]
    for block in word_sequence_blocks:
        unmovable_objects.append(block["id"])
    for block in scattered_blocks:
        if block["id"] != correct_answer_block["id"]:
            unmovable_objects.append(block["id"])

    grasp_config = GraspConfig(
        unmovable_objects=unmovable_objects,
        action_description={"zh": f"拿起字母{missing_letter}", "en": f"picks up the letter {missing_letter}"},
    )

    target_rot_matrix = quat2mat_wxyz(correct_answer_block["rotation"])
    target_poses = []
    x_offsets = np.linspace(-0.002, 0.002, 3)
    y_offsets = np.linspace(-0.002, 0.002, 3)
    z_offsets = np.linspace(0, 0.003, 3)
    for x_off in x_offsets:
        for y_off in y_offsets:
            for z_off in z_offsets:
                position = target_position + np.array([x_off, y_off, z_off])
                pose = np.eye(4)
                pose[:3, :3] = target_rot_matrix
                pose[:3, 3] = position
                target_poses.append(pose)
    target_poses = np.array(target_poses)

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
        extra_checkers=[create_placement_checker(correct_answer_block["id"], target_position, missing_letter)],
        action_description={"zh": f"将字母{missing_letter}放置到单词'{target_word}'的正确位置", "en": f"places the letter {missing_letter} into the correct position in the word '{target_word}'"},
        disable_upside_down=False,
    )

    current_success = robot.pick_and_place_actor_to_target_pose(
        actor=correct_answer_block["id"],
        target_pose=target_poses,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config,
    )

    if not current_success:
        success = False
    else:
        robot.back_to_origin(arm=arm)
        final_pos, _ = robot.get_object_pose(correct_answer_block["id"])
        if final_pos is not None:
            final_xy_dist = np.linalg.norm(final_pos[:2] - target_position[:2])
            if final_xy_dist <= 0.03:
                all_blocks_now = word_sequence_blocks + [correct_answer_block]
                complete_word_letters = []
                for block in all_blocks_now:
                    pos, _ = robot.get_object_pose(block["id"])
                    if pos is not None:
                        complete_word_letters.append((block["letter"], pos[1]))
                complete_word_letters.sort(key=lambda x: x[1], reverse=True)
                complete_word = "".join([letter for letter, _ in complete_word_letters])
                if complete_word == target_word:
                    success = True
                else:
                    success = False

    return success
```
