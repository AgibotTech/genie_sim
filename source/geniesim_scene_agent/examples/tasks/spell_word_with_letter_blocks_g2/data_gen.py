# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.transform_utils import *
from client.llm_gen.robot_api import *
import numpy as np
import os
import random
import itertools

def execute_task(robot: RobotAPI):
    # Expanded vocabulary.
    WORD_LIST = [
        # Three-letter words.
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
        # Four-letter words.
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

    # Convert words to uppercase and deduplicate.
    WORD_SET = set([word.upper() for word in WORD_LIST])

    # Initialize letter-asset information.
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

    # Runtime data.
    scene_info = robot.scene_info
    interactions = robot.interactions
    success = True

    # 1. Locate the table.
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

    # 2. Locate all letter blocks.
    all_letter_blocks = []
    for obj_id, obj_info in scene_info["layout"].items():
        usd = obj_info.get("usd", "")

        # Check whether this is a letter block.
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
                all_letter_blocks.append({
                    "id": obj_id,
                    "letter": letter_value,
                    "position": pos,
                    "rotation": rot,
                    "size": size,
                })
                print(f"Found letter block: {obj_id}, letter: {letter_value}")

    if len(all_letter_blocks) < 3:
        print(f"Found only {len(all_letter_blocks)} letter blocks, need at least 3")
        return False

    print(f"Total letter blocks found: {len(all_letter_blocks)}")

    # 3. Analyze every possible word combination in the scene.
    # Find all word combinations from the scene's N letter blocks.

    available_letters = [block["letter"] for block in all_letter_blocks]
    print(f"Available letters in scene: {available_letters}")

    # Build a letter-to-block mapping for lookup.
    letter_to_blocks = {}
    for block in all_letter_blocks:
        letter = block["letter"]
        if letter not in letter_to_blocks:
            letter_to_blocks[letter] = []
        letter_to_blocks[letter].append(block)

    # Find all possible word combinations.
    # Match using letter sets; letters do not repeat.
    possible_words = []
    word_to_blocks_mapping = {}  # Store the block list for each word.

    print("\nSearching for all possible words from available letters...")

    # Set of letters present in the scene.
    available_letters_set = set(available_letters)

    # Iterate through the entire vocabulary.
    for word in WORD_SET:
        word_letters_set = set(word)

        # Check that scene letters include every letter in the word.
        # Without repeated letters, a subset check is sufficient.
        if word_letters_set.issubset(available_letters_set):
            # Check sufficient letter count, without duplicates.
            if len(word) <= len(available_letters):
                # Find a corresponding block for each word letter.
                letter_to_blocks_copy = {}
                for block in all_letter_blocks:
                    letter = block["letter"]
                    if letter not in letter_to_blocks_copy:
                        letter_to_blocks_copy[letter] = []
                    letter_to_blocks_copy[letter].append(block)

                blocks_for_word = []
                can_spell = True

                for letter in word:
                    if letter in letter_to_blocks_copy and letter_to_blocks_copy[letter]:
                        # Use the first available block.
                        block = letter_to_blocks_copy[letter].pop(0)
                        blocks_for_word.append(block)
                    else:
                        can_spell = False
                        break

                if can_spell and len(blocks_for_word) == len(word):
                    possible_words.append(word)
                    word_to_blocks_mapping[word] = blocks_for_word
                    print(f"  Found possible word: {word}")

    if not possible_words:
        print("ERROR: No possible words found from available letters")
        print(f"Available letters: {available_letters}")
        return False

    print(f"\nFound {len(possible_words)} possible words:")
    for i, word in enumerate(possible_words):
        print(f"  {i+1}. {word}")

    # 4. Randomly select a feasible word.
    target_word = random.choice(possible_words)
    found_blocks_for_word = word_to_blocks_mapping[target_word]

    print(f"\n=== Randomly selected word to spell: {target_word} ===")
    print(f"Selected blocks for spelling (in word order):")
    for i, block in enumerate(found_blocks_for_word):
        print(f"  {i+1}. Letter {block['letter']} (id: {block['id']})")

    # 5. MANDATORY: set task description.
    task_description = {
        "task_name": f"spell the word '{target_word.lower()}' with letter blocks",
        "english_task_name": f"spell the word '{target_word.lower()}' with letter blocks",
        "init_scene_text": f"The robot is in front of the table, there are letter blocks on the table"
    }
    robot.client.set_task_task_basic_info(task_description)

    # 6. Calculate targets using the reference script's placement logic.
    # Target x range (-0.05, 0.0): choose random x.
    target_x = np.random.uniform(-0.05, 0.0)
    print(f"Randomly selected target_x: {target_x:.3f}")

    # Starting y range (0.05, 0.08): choose random y.
    start_y = np.random.uniform(0.05, 0.08)
    print(f"Randomly selected start_y: {start_y:.3f}")

    # Block spacing: 0.07m (7cm).
    spacing = 0.07

    # Calculate targets left to right, with decreasing y.
    target_positions = []
    table_pos, table_rot = robot.get_object_pose(table_id)
    table_size = robot.get_object_size_in_world_frame(table_id)
    table_top_z = table_pos[2] + table_size[2] / 2.0
    margin = 0.01  # Safety margin.

    for i in range(len(found_blocks_for_word)):
        target_y = start_y - spacing * i
        block_size = found_blocks_for_word[i]["size"]
        target_z = table_top_z + margin + block_size[2] * 3 / 2.0

        target_position = np.array([target_x, target_y, target_z])
        target_positions.append(target_position)

        letter = found_blocks_for_word[i]["letter"]
        print(f"Target position for letter {letter}: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}]")

    # 7. Preserve initial orientation during placement.
    def calculate_target_rotation(initial_rotation):
        return quat2mat_wxyz(initial_rotation)

    # 8. Place letters in word order.
    placed_blocks = []
    last_arm = ""

    for i, block in enumerate(found_blocks_for_word):
        block_id = block["id"]
        block_letter = block["letter"]
        current_pos = block["position"]
        current_rot = block["rotation"]

        print(f"\n=== Processing letter {i+1}/{len(found_blocks_for_word)}: {block_letter} ===")

        arm = robot.judge_arm_to_use(current_pos)
        print(f"Using arm: {arm}")

        if last_arm != arm and last_arm != "":
            robot.back_to_origin(arm=last_arm)
        last_arm = arm

        # Unmovable object list.
        unmovable_objects = [table_id] + [b["id"] for b in placed_blocks]

        # Grasp configuration.
        grasp_config = GraspConfig(
            unmovable_objects=unmovable_objects,
            action_description={
                "zh": f"拿起字母{block_letter}",
                "en": f"picks up the letter {block_letter}",
            },
        )

        # Target rotation matrix.
        target_rot_matrix = calculate_target_rotation(current_rot)

        # Generate multiple candidate poses.
        target_poses = []
        base_position = target_positions[i]

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
        print(f"Generated {len(target_poses)} candidate poses for letter {block_letter}")

        # Create placement checker.
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

        # Placement description.
        if len(found_blocks_for_word) == 3:
            position_desc = "最左侧" if i == 0 else ("中间" if i == 1 else "最右侧")
        else:
            position_desc = "最左侧" if i == 0 else ("最右侧" if i == len(found_blocks_for_word)-1 else "中间")

        place_desc_zh = f"将字母{block_letter}放置到{position_desc}位置"
        place_desc_en = f"places the letter {block_letter} at the {position_desc} position"

        place_config = PlaceConfig(
            unmovable_objects=unmovable_objects,
            extra_checkers=[create_placement_checker(block_id, base_position, block_letter)],
            action_description={"zh": place_desc_zh, "en": place_desc_en},
            disable_upside_down=False,
        )

        # Execute grasp and placement.
        current_success = robot.pick_and_place_actor_to_target_pose(
            actor=block_id,
            target_pose=target_poses,
            arm=arm,
            grasp_config=grasp_config,
            place_config=place_config,
        )

        if not current_success:
            print(f"Failed to place letter {block_letter}")
            return False

        print(f"Successfully placed letter {block_letter}")
        placed_blocks.append(block)

    # 9. Final verification.
    print("\n=== Performing final verification ===")

    if len(placed_blocks) == len(found_blocks_for_word):
        # Collect final position information.
        final_positions = []
        for block in placed_blocks:
            pos, _ = robot.get_object_pose(block["id"])
            if pos is not None:
                final_positions.append((block["letter"], pos[1], block["id"]))

        # Sort by descending y, left to right.
        final_positions.sort(key=lambda x: x[1], reverse=True)

        print("Final arrangement from left to right:")
        letters_in_order = []
        for letter, y_pos, block_id in final_positions:
            print(f"  Letter {letter} at y={y_pos:.3f}")
            letters_in_order.append(letter)

        # Compose the actually spelled string.
        spelled_word = ''.join(letters_in_order)

        # Check vocabulary membership, the key success criterion.
        if spelled_word.upper() in WORD_SET:
            print(f"✅ SUCCESS: Successfully spelled word '{spelled_word}'!")
            print(f"  Word '{spelled_word}' is in the dictionary")
            print(f"  Original target word was: {target_word}")
            print(f"  Letters in order: {' '.join(letters_in_order)}")

            # Check spacing.
            print("\nChecking spacing between letters:")
            for j in range(len(final_positions) - 1):
                letter1, y1, _ = final_positions[j]
                letter2, y2, _ = final_positions[j + 1]
                actual_spacing = abs(y1 - y2)
                if abs(actual_spacing - spacing) > 0.02:
                    print(f"  Warning: Spacing between {letter1} and {letter2} is {actual_spacing:.3f}m")
                else:
                    print(f"  Good: Spacing between {letter1} and {letter2} is {actual_spacing:.3f}m")
        else:
            print(f"❌ ERROR: Spelled word '{spelled_word}' is not in dictionary")
            print(f"  Original target word was: {target_word}")
            print(f"  Letters in order: {' '.join(letters_in_order)}")
            success = False
    else:
        print(f"❌ ERROR: Only placed {len(placed_blocks)} out of {len(found_blocks_for_word)} letters")
        success = False

    # Return the arm to origin.
    if last_arm != "":
        robot.back_to_origin(arm=last_arm)

    return success and (len(placed_blocks) == len(found_blocks_for_word))
