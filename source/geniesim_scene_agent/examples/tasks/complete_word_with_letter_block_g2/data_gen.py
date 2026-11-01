# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os

def execute_task(robot: RobotAPI):
    ## Initialize objects_list based on provided interaction data
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

    ## Complete word list
    WORD_LIST = [
        "cat", "dog", "sun", "sky", "pig", "fox", "cow", "bus", "car", "pen",
        "cup", "box", "hat", "boy", "run", "fly", "eat", "old", "new", "hot",
        "bat", "man", "red", "big", "sad", "mad", "top", "map", "toy", "leg",
        "art", "bad", "bag", "bed", "bit", "bug", "cab", "cap", "cry", "cut",
        "day", "dry", "ear", "end", "fat", "fit", "fun", "gap", "gas", "god",
        "guy", "hit", "ice", "jam", "job", "joy", "key", "kid", "lap", "law",
        "let", "lip", "log", "lot", "low", "mix", "mug", "net", "nut", "oil",
        "out", "owl", "pad", "pan", "pat", "pay", "pet", "pie", "pot", "rag",
        "rat", "raw", "rib", "rub", "rug", "sea", "set", "sex", "sin", "sit",
        "six", "son", "tap", "tax", "tea", "ten", "tie", "tip", "toe", "try",
        "tub", "use", "van", "war", "way", "web", "wet", "win", "yes", "yet",
        "zip", "act", "age", "air", "ant", "any", "ape", "arm", "ash", "ask",
        "axe", "bar", "bet", "bid", "bow", "buy", "can", "dip", "dot", "due",
        "ego", "fan", "far", "fax", "few", "fix", "flu", "fog", "get", "gym",
        "hay", "hen", "hex", "hip", "hop", "how", "hut", "ink", "jar", "jaw",
        "bird", "fish", "bear", "lion", "wolf", "frog", "duck", "hand", "time", "song",
        "play", "work", "read", "jump", "fast", "slow", "cold", "warm", "blue", "pink",
        "girl", "town", "city", "path", "gold", "iron", "rock", "sand", "wind", "rain",
        "able", "ache", "acid", "aunt", "auto", "back", "bake", "bald", "band", "bang",
        "bank", "bare", "bark", "base", "bath", "beak", "bean", "beat", "belt", "bend",
        "best", "bind", "bite", "blow", "boat", "body", "boil", "bold", "bone", "born",
        "both", "bowl", "buck", "bulk", "bump", "burn", "bury", "bush", "busy", "cage",
        "cake", "calf", "calm", "camp", "cape", "care", "carp", "cart", "cash", "cast",
        "cave", "chef", "chew", "chin", "chip", "chop", "clam", "clap", "claw", "clay",
        "clip", "club", "clue", "coal", "coat", "code", "coin", "comb", "come", "cone",
        "cope", "copy", "cord", "core", "cork", "corn", "cost", "crab", "crop", "crow",
        "cure", "curl", "cute", "damp", "dare", "dark", "dart", "dash", "date", "dawn",
        "deaf", "deal", "dean", "debt", "deck", "defy", "desk", "dial", "dice", "diet",
        "dirt", "dish", "dive", "dock", "dome", "done", "dope", "dose", "dove", "down",
        "drag", "draw", "drop", "drug", "drum", "dual", "duet", "duke", "dumb", "dump",
        "dusk", "dust", "duty", "each", "earn", "easy", "echo", "edit", "epic", "evil",
        "exit", "face", "fact", "fade", "fail", "fair", "fake", "fame", "fang", "farm",
        "fate", "fawn", "fear", "feat", "find", "fine", "fire", "firm", "five", "flag",
        "flat", "flea", "fled", "flip", "flow", "foal", "foam", "foil", "fold", "folk",
        "fond", "fork", "form", "fort", "foul", "four", "foxy", "fray", "from", "fuel",
        "fund", "fury", "gain", "gait", "gale", "game", "gasp", "gate", "gave", "gear",
        "gift", "glad", "glow", "glue", "goal", "goat", "grab", "gram", "grew", "grid",
        "grim", "grip", "grow", "grub", "gulf", "gush", "hack", "hail", "hair", "half",
        "halo", "halt", "hard", "hare", "harp", "hate", "have", "hawk", "head", "heal",
        "hear", "heat", "help", "herd", "hero", "hide", "hike", "hind", "hint", "hire",
        "hold", "hole", "holy", "home", "hope", "horn", "host", "hour", "huge", "hunt",
        "hurt", "hype", "idea", "idle", "idol", "inch", "into", "item", "jack", "jail",
        "joke", "junk", "jury", "just", "kale", "kept", "kind", "king", "kite", "knit",
        "knot", "know", "lack", "lady", "lake", "lamb", "lamp", "land", "lane", "lark",
        "lash", "late", "lazy", "lead", "leaf", "leak", "lean", "leap", "left", "lend",
        "lent", "liar", "lice", "lick", "life", "lift", "like", "limb", "lime", "limp",
        "line", "link", "lisp", "live", "load", "loaf", "loan", "lock", "loft", "long",
        "lord", "lose", "loud", "love", "luck", "lump", "lung", "lure", "lurk", "lute",
        "lynx", "mace", "mach", "made", "maid", "mail", "main", "make", "male", "malt",
        "many", "mark", "mart", "mash", "mask", "mast", "mate", "math", "maze", "mead"
    ]

    WORD_LIST_UPPER = [word.upper() for word in WORD_LIST]

    ## Runtime-only data
    scene_info = robot.scene_info
    interactions = robot.interactions
    success = True

    ## MANDATORY: Set task description before first action
    task_description = {
        "task_name": 'Complete a word with a letter block',
        "english_task_name": 'Complete a word with a letter block',
        "init_scene_text": "The robot is in front of the table, there is a word with a missing letter displayed as a fill-in-the-blank puzzle, and candidate letters are scattered on the table"
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

    ## 2. Find all letter blocks
    letter_blocks = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")
        keywords = obj_info.get("keywords", [])

        ## Check if it's a letter block
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
                    "id": obj_id,
                    "usd": usd,
                    "letter": letter_value,
                    "position": pos,
                    "rotation": rot,
                    "size": size,
                    "keywords": keywords,
                })

    if len(letter_blocks) < 3:
        print(f"Found only {len(letter_blocks)} letter blocks, need at least 3")
        return False

    print(f"Total letter blocks found: {len(letter_blocks)}")
    for block in letter_blocks:
        print(f"  Letter {block['letter']} at position [{block['position'][0]:.3f}, {block['position'][1]:.3f}, {block['position'][2]:.3f}]")

    ## 3. Identify the word sequence (aligned letters in x(0,0.05) region)
    word_sequence_blocks = []
    scattered_blocks = []

    for block in letter_blocks:
        if -0.02 <= block["position"][0] <= 0.05:
            word_sequence_blocks.append(block)
        else:
            scattered_blocks.append(block)

    if len(word_sequence_blocks) < 2:
        print(f"Not enough letters in word sequence region: {len(word_sequence_blocks)}")
        return False

    ## Sort word sequence blocks by y-position (descending: +Y to -Y, left to right)
    word_sequence_blocks.sort(key=lambda x: x["position"][1], reverse=True)

    print(f"Word sequence blocks ({len(word_sequence_blocks)}):")
    for i, block in enumerate(word_sequence_blocks):
        print(f"  Position {i}: letter {block['letter']} at y={block['position'][1]:.3f}")

    print(f"Scattered blocks ({len(scattered_blocks)}):")
    for block in scattered_blocks:
        print(f"  Letter {block['letter']} at position [{block['position'][0]:.3f}, {block['position'][1]:.3f}]")

    ## 4. Analyze the word sequence to find missing position
    word_letters = [block["letter"] for block in word_sequence_blocks]
    word_positions = [block["position"][1] for block in word_sequence_blocks]

    ## Variables to store target word and missing letter info
    target_word = None
    missing_letter = None
    missing_index_in_word = None  # Position in the complete word

    ## Calculate expected spacing (should be around 0.06m)
    if len(word_positions) >= 2:
        spacings = []
        for i in range(1, len(word_positions)):
            spacing = abs(word_positions[i-1] - word_positions[i])
            spacings.append(spacing)

        avg_spacing = np.mean(spacings) if spacings else 0.06
        print(f"Average spacing in word sequence: {avg_spacing:.3f}m")

        ## Find which position has a larger gap (missing letter)
        gap_found = False
        for i in range(1, len(word_positions)):
            spacing = abs(word_positions[i-1] - word_positions[i])
            if spacing > avg_spacing * 1.5:  # Significant gap found
                gap_found = True
                print(f"Found gap at position {i}, suggesting missing letter between letters {word_letters[i-1]} and {word_letters[i]}")
                break

        ## Try to find the target word and missing letter from WORD_LIST
        all_letters_on_table = [block["letter"] for block in letter_blocks]
        all_letters_set = set(all_letters_on_table)

        print(f"All letters on table: {sorted(all_letters_on_table)}")

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
            print(f"Possible words found: {possible_words}")
            target_word, missing_letter = possible_words[0]
            print(f"Selected target word: {target_word}, missing letter: {missing_letter}")

            ## Find the exact position of missing letter in the target word
            target_word_list = list(target_word)
            missing_index_in_word = target_word_list.index(missing_letter)
            print(f"Missing letter '{missing_letter}' is at position {missing_index_in_word} in word '{target_word}'")
        else:
            print("Could not find a matching word from dictionary")
            return False
    else:
        print("Not enough word sequence blocks to analyze")
        return False

    if target_word is None or missing_letter is None or missing_index_in_word is None:
        print("Could not determine target word and missing position")
        return False

    ## 5. Find the correct letter among scattered blocks
    correct_answer_block = None
    for block in scattered_blocks:
        if block["letter"] == missing_letter:
            correct_answer_block = block
            break

    if correct_answer_block is None:
        print(f"Missing letter {missing_letter} not found among scattered blocks")
        return False

    print(f"Found correct answer: letter {missing_letter} at block {correct_answer_block['id']}")

    ## 6. Calculate target position for the missing letter
    ## Build a mapping of letter -> position for letters already placed
    letter_to_block = {block["letter"]: block for block in word_sequence_blocks}

    ## Calculate target position based on adjacent letters in the target word
    target_word_list = list(target_word)

    ## Find the nearest placed letter before or after the missing position
    ## to calculate the target Y position
    word_x = word_sequence_blocks[0]["position"][0]  # All letters share same X

    ## Find reference letter and calculate target position
    target_y = None

    ## Look for a letter before the missing position
    for i in range(missing_index_in_word - 1, -1, -1):
        ref_letter = target_word_list[i]
        if ref_letter in letter_to_block:
            ref_y = letter_to_block[ref_letter]["position"][1]
            # Missing letter should be (missing_index_in_word - i) positions after
            target_y = ref_y - (missing_index_in_word - i) * avg_spacing
            print(f"Using letter '{ref_letter}' at y={ref_y:.3f} as reference (before), target_y={target_y:.3f}")
            break

    ## If no letter before, look for a letter after the missing position
    if target_y is None:
        for i in range(missing_index_in_word + 1, len(target_word_list)):
            ref_letter = target_word_list[i]
            if ref_letter in letter_to_block:
                ref_y = letter_to_block[ref_letter]["position"][1]
                # Missing letter should be (i - missing_index_in_word) positions before
                target_y = ref_y + (i - missing_index_in_word) * avg_spacing
                print(f"Using letter '{ref_letter}' at y={ref_y:.3f} as reference (after), target_y={target_y:.3f}")
                break

    if target_y is None:
        print("Could not calculate target position")
        return False

    target_x = word_x

    ## Get table height
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0

    ## Add safety margin
    margin = 0.01
    target_z = table_top_z + margin + correct_answer_block["size"][2] * 3 / 2.0

    target_position = np.array([target_x, target_y, target_z])
    print(f"Target position for missing letter {missing_letter}: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}]")

    ## 7. Pick up the correct answer letter and place it
    arm = robot.judge_arm_to_use(correct_answer_block["position"])
    print(f"Using arm: {arm}")

    ## Create grasp configuration
    unmovable_objects = [table_id]
    for block in word_sequence_blocks:
        unmovable_objects.append(block["id"])
    for block in scattered_blocks:
        if block["id"] != correct_answer_block["id"]:  # Don't include the block we're moving
            unmovable_objects.append(block["id"])

    grasp_config = GraspConfig(
        unmovable_objects=unmovable_objects,
        action_description={
            "zh": f"拿起字母{missing_letter}",
            "en": f"picks up the letter {missing_letter}",
        },
    )

    ## Calculate target rotation (keep original rotation)
    target_rot_matrix = quat2mat_wxyz(correct_answer_block["rotation"])

    ## Generate multiple candidate poses
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
    print(f"Generated {len(target_poses)} candidate poses")

    ## Create placement verification function
    def create_placement_checker(block_id, expected_position, block_letter):
        def check_placement():
            current_p, current_r = robot.get_object_pose(block_id)
            if current_p is None or current_r is None:
                return False
            xy_dist = np.linalg.norm(current_p[:2] - expected_position[:2])
            if xy_dist > 0.03:
                print(f"Letter {block_letter} position error: xy_dist={xy_dist:.3f}m")
                return False
            return True
        return check_placement

    place_config = PlaceConfig(
        unmovable_objects=unmovable_objects,
        extra_checkers=[create_placement_checker(correct_answer_block["id"], target_position, missing_letter)],
        action_description={
            "zh": f"将字母{missing_letter}放置到单词'{target_word}'的正确位置",
            "en": f"places the letter {missing_letter} into the correct position in the word '{target_word}'",
        },
        disable_upside_down=False,
    )

    ## Execute pick and place operation
    current_success = robot.pick_and_place_actor_to_target_pose(
        actor=correct_answer_block["id"],
        target_pose=target_poses,
        arm=arm,
        grasp_config=grasp_config,
        place_config=place_config,
    )

    if not current_success:
        print(f"Failed to place missing letter {missing_letter}")
        success = False
    else:
        print(f"Successfully placed missing letter {missing_letter}")

        ## Return arm to origin
        robot.back_to_origin(arm=arm)

        ## Final verification
        print("\n=== Performing final verification ===")
        final_pos, final_rot = robot.get_object_pose(correct_answer_block["id"])
        if final_pos is not None:
            final_xy_dist = np.linalg.norm(final_pos[:2] - target_position[:2])
            if final_xy_dist <= 0.03:
                ## Verify the complete word
                complete_word_letters = []
                all_blocks_now = word_sequence_blocks + [correct_answer_block]
                for block in all_blocks_now:
                    pos, _ = robot.get_object_pose(block["id"])
                    if pos is not None:
                        complete_word_letters.append((block["letter"], pos[1]))

                ## Sort by y-position
                complete_word_letters.sort(key=lambda x: x[1], reverse=True)
                complete_word = "".join([letter for letter, _ in complete_word_letters])

                if complete_word == target_word:
                    print(f"SUCCESS: Letter {missing_letter} correctly placed!")
                    print(f"Complete word formed: {complete_word}")
                    print(f"Final position error: {final_xy_dist:.3f}m")
                    success = True
                else:
                    print(f"Word mismatch: Expected {target_word}, got {complete_word}")
                    success = False
            else:
                print(f"Position error too large: {final_xy_dist:.3f}m")
                success = False
        else:
            print("Could not verify final position")
            success = False

    print("\n=== Task completion summary ===")
    if success:
        print(f"SUCCESS: Missing letter {missing_letter} placed correctly in word {target_word}")
    else:
        print("FAILED: Could not complete the fill-in-the-blank task")

    return success
