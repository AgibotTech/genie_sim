# Spell a common-sense answer with letter blocks

**Tags:** `reasoning` `letter_blocks` `word_spelling` `common_sense` `table_only` `pick_and_place` `dynamic_task_description` `linear_arrangement` `spacing` `GraspConfig` `PlaceConfig`

**Task flow (how to implement and key points):**
1. **Object lookup:** Find table and all letter blocks. Map each block to its letter value using objects_list.
2. **Word search with questions:** Build available letter set. Search WORD_LIST for words whose letters are all present. Each word has an associated common-sense question.
3. **Word selection:** Randomly select one valid word from possible words. Get the associated question from QUESTION_DICT.
4. **Dynamic task description:** Set task description to include the question - the robot must answer by spelling the word.
5. **Target positions:** Linear arrangement: target_x in (-0.05, 0), start_y in (0.05, 0.08), spacing 0.07m.
6. **Placement and verification:** Place letters in word order, verify final arrangement matches target word.

---

# INPUT

**scene_description:** A table with letters that include all letters of a randomly selected word from a common-sense word list plus two distractor letters. Each word has an associated question that the robot must answer by spelling.

**task description:** Answer a common-sense question by spelling the answer word with letter blocks on the table.

**code to generate the scene:**

```python
from helper import *
import random

"""
scene_name: table_with_words_and_distractors
description: A table with letters that include all letters of a randomly selected word from the common-sense word list plus two distractor letters.
"""

EXPANDED_WORD_LIST = [
    "dog", "cat", "pig", "cow", "fox", "bird", "fish", "bear",
    "lion", "wolf", "frog", "duck", "bat", "owl", "crab", "bug",
    "sun", "sky", "sea", "ice", "rock", "sand", "wind", "rain",
    # ... more words
]

QUESTION_DICT = [
    {"word": "dog", "question": "什么动物被称为人类最好的朋友，并且会汪汪叫？"},
    {"word": "cat", "question": "什么宠物喜欢抓老鼠，并且发出「喵喵」的叫声？"},
    {"word": "pig", "question": "鼻子圆圆的，身体胖胖的，喜欢在泥巴里打滚的家畜是什么？"},
    {"word": "cow", "question": "我们平时喝的天然牛奶，主要是什么动物产的？"},
    # ... more question-word pairs
]

def get_tabletop_placement_area(table_shape: Shape) -> Tuple[P, P]:
    table_info = get_object_info(table_shape)
    desktop_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_center_world = table_info["center"] + desktop_info["center"]
    desktop_top_z = table_info["center"][2] + desktop_info["xyz_max"][2]
    return np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z]), desktop_info["size"]

@register()
def place_single_letter(letter_oid: str, letter_char: str, table_top_center: P, existing_positions: List[P]) -> Shape:
    letter_shape = library_call("usd", oid=letter_oid, keywords=[f"letter_{letter_char}", "alphabet"])
    x_min_allowed, x_max_allowed = -0.2, 0.15
    y_min_allowed, y_max_allowed = -0.25, 0.25
    forbidden_x_min, forbidden_x_max = -0.05, 0.08
    forbidden_y_min, forbidden_y_max = -0.1, 0.1
    for _ in range(2000):
        x_offset = random.uniform(x_min_allowed, x_max_allowed)
        y_offset = random.uniform(y_min_allowed, y_max_allowed)
        if (forbidden_x_min < x_offset < forbidden_x_max) and (forbidden_y_min < y_offset < forbidden_y_max):
            continue
        break
    return transform_shape(letter_shape, translation_matrix([table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]]))

def has_unique_letters(word: str) -> bool:
    return len(set(word)) == len(word)

VALID_WORDS = [word.upper() for word in EXPANDED_WORD_LIST if has_unique_letters(word.upper())]
WORD_TO_QUESTION = {item["word"].upper(): item["question"] for item in QUESTION_DICT}

@register()
def table_with_words_and_distractors() -> Shape:
    table_shape = library_call("usd", oid="table_000", keywords=["table"])
    table_top_center, _ = get_tabletop_placement_area(table_shape)
    letter_map = {
        'A': 'benchmark_letter_026', 'B': 'benchmark_letter_027', 'C': 'benchmark_letter_028',
        'D': 'benchmark_letter_029', 'E': 'benchmark_letter_030', 'F': 'benchmark_letter_031',
        # ... rest of alphabet mapping
    }
    selected_word = random.choice(VALID_WORDS)
    word_chars = list(selected_word)
    available_distractors = [l for l in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if l not in word_chars]
    distractor_chars = random.sample(available_distractors, 2)
    all_chars = word_chars + distractor_chars
    letter_shapes = []
    placed_positions = []
    for char in all_chars:
        letter_shape = library_call("place_single_letter", letter_oid=letter_map[char], letter_char=char, table_top_center=table_top_center, existing_positions=placed_positions)
        letter_info = get_object_info(letter_shape)
        placed_positions.append(letter_info["center"])
        letter_shapes.append(letter_shape)
    return concat_shapes(table_shape, *letter_shapes)

@register()
def root_scene() -> Shape:
    return table_with_words_and_distractors()
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
    WORD_LIST = [
        "dog", "cat", "pig", "cow", "fox", "bird", "fish", "bear",
        "lion", "wolf", "frog", "duck", "bat", "owl", "crab", "bug",
        "sun", "sky", "sea", "ice", "rock", "sand", "wind", "rain",
        # ... more words
    ]
    WORD_SET = set([word.upper() for word in WORD_LIST])

    QUESTION_DICT = [
        {"word": "dog", "question": "什么动物被称为人类最好的朋友，并且会汪汪叫？"},
        {"word": "cat", "question": "什么宠物喜欢抓老鼠，并且发出「喵喵」的叫声？"},
        # ... more pairs
    ]
    WORD_TO_QUESTION = {item["word"].upper(): item["question"] for item in QUESTION_DICT}

    objects_list = {
        "benchmark_letter_026": {"has_interactions": True, "object_type": "letter_block", "letter": "A"},
        # ... rest of letter mappings
        "table_000": {"has_interactions": False, "object_type": "table"},
    }

    scene_info = robot.scene_info
    interactions = robot.interactions
    success = True

    table_id = None
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])
        if usd == "table_000" or "table" in keywords:
            table_id = obj_id
            break

    all_letter_blocks = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        for asset_id, asset_info in objects_list.items():
            if asset_id in usd and asset_info.get("object_type") == "letter_block":
                pos, rot = robot.get_object_pose(obj_id)
                size = robot.get_object_size_in_world_frame(obj_id)
                if pos is not None:
                    all_letter_blocks.append({"id": obj_id, "letter": asset_info["letter"], "position": pos, "rotation": rot, "size": size})
                break

    available_letters = [block["letter"] for block in all_letter_blocks]
    available_letters_set = set(available_letters)

    possible_words = []
    word_to_blocks_mapping = {}
    word_to_question_mapping = {}

    for word in WORD_SET:
        word_letters_set = set(word)
        if word_letters_set.issubset(available_letters_set):
            letter_to_blocks = {}
            for block in all_letter_blocks:
                letter = block["letter"]
                if letter not in letter_to_blocks:
                    letter_to_blocks[letter] = []
                letter_to_blocks[letter].append(block)
            blocks_for_word = []
            can_spell = True
            for letter in word:
                if letter in letter_to_blocks and letter_to_blocks[letter]:
                    blocks_for_word.append(letter_to_blocks[letter].pop(0))
                else:
                    can_spell = False
                    break
            if can_spell:
                possible_words.append(word)
                word_to_blocks_mapping[word] = blocks_for_word
                word_to_question_mapping[word] = WORD_TO_QUESTION.get(word, f"拼写单词 {word}")

    if not possible_words:
        return False

    target_word = random.choice(possible_words)
    target_question = word_to_question_mapping[target_word]
    found_blocks_for_word = word_to_blocks_mapping[target_word]

    task_description = {
        "task_name": f"根据问题「{target_question}」，用字母积木拼写答案单词",
        "english_task_name": f"based on the question '{target_question}', spell the answer word '{target_word.lower()}' with letter blocks",
        "init_scene_text": "The robot is in front of the table with letter blocks and a question to answer"
    }
    robot.client.set_task_task_basic_info(task_description)

    target_x = np.random.uniform(-0.05, 0.0)
    start_y = np.random.uniform(0.05, 0.08)
    spacing = 0.07

    target_positions = []
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0
    margin = 0.01

    for i in range(len(found_blocks_for_word)):
        target_y = start_y - spacing * i
        block_size = found_blocks_for_word[i]["size"]
        target_z = table_top_z + margin + block_size[2] * 3 / 2.0
        target_positions.append(np.array([target_x, target_y, target_z]))

    placed_blocks = []
    last_arm = ""

    for i, block in enumerate(found_blocks_for_word):
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

        target_rot_matrix = quat2mat_wxyz(current_rot)
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

        place_config = PlaceConfig(
            unmovable_objects=unmovable_objects,
            extra_checkers=[lambda bid=block_id, exp=base_position, let=block_letter:
                           np.linalg.norm(robot.get_object_pose(bid)[0][:2] - exp[:2]) < 0.03],
            action_description={"zh": f"将字母{block_letter}放置到位置{i+1}", "en": f"places letter {block_letter}"},
        )

        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=block_id, target_pose=np.array(target_poses), arm=arm,
            grasp_config=grasp_config, place_config=place_config,
        )

        if not current_success:
            return False
        placed_blocks.append(block)

    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    return True
```
