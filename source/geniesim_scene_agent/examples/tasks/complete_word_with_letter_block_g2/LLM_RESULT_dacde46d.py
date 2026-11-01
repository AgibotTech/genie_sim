# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random

"""
scene_name: table_with_fill_in_blank_letters
description: A table with a 4-letter word displayed as a fill-in-the-blank puzzle. The complete word letters (except one missing) are arranged neatly in a row from left to right (+Y to -Y direction) at the same X position (0-0.05), with the first letter's Y in (0.04, 0.08). Letters are spaced 0.06m apart along the Y-axis. The missing letter is always at position 1 or 2 (0-indexed middle two of four). Additionally, four letters are placed randomly in the wider area (avoiding central forbidden zone): one is the correct missing letter, and three are distractors. All letters are unique.
"""

def get_tabletop_placement_area(table_shape: Shape) -> Tuple[P, P]:
    """
    Get the top surface center and size of the table's desktop in world coordinates.
    """
    table_info = get_object_info(table_shape)
    desktop_info = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_center_world = table_info["center"] + desktop_info["center"]
    desktop_top_z = table_info["center"][2] + desktop_info["xyz_max"][2]
    desktop_pos = np.array([desktop_center_world[0], desktop_center_world[1], desktop_top_z])
    desktop_size = desktop_info["size"]
    return desktop_pos, desktop_size

def positions_too_close(pos1: P, pos2: P, min_dist: float = 0.07) -> bool:
    """
    Check if two positions are closer than min_dist in XY plane.
    """
    dist = np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
    return dist < min_dist

@register()
def place_word_letters_aligned(word_chars: List[str], missing_index: int, table_top_center: P) -> Shape:
    """
    Place the word letters in an aligned row from left to right (+Y to -Y direction) at the same X position.
    All letters share the same X (0-0.05), first letter has Y in (0.04, 0.08), subsequent letters decrease in Y by 0.06m each.
    """
    letter_shapes = []
    letter_map = {
        'A': 'benchmark_letter_026',
        'B': 'benchmark_letter_027',
        'C': 'benchmark_letter_028',
        'D': 'benchmark_letter_029',
        'E': 'benchmark_letter_030',
        'F': 'benchmark_letter_031',
        'G': 'benchmark_letter_032',
        'H': 'benchmark_letter_033',
        'I': 'benchmark_letter_034',
        'J': 'benchmark_letter_035',
        'K': 'benchmark_letter_036',
        'L': 'benchmark_letter_037',
        'M': 'benchmark_letter_038',
        'N': 'benchmark_letter_039',
        'O': 'benchmark_letter_040',
        'P': 'benchmark_letter_041',
        'Q': 'benchmark_letter_042',
        'R': 'benchmark_letter_043',
        'S': 'benchmark_letter_044',
        'T': 'benchmark_letter_045',
        'U': 'benchmark_letter_046',
        'V': 'benchmark_letter_047',
        'W': 'benchmark_letter_048',
        'X': 'benchmark_letter_049',
        'Y': 'benchmark_letter_050',
        'Z': 'benchmark_letter_051',
    }
    # Only 4-letter words
    assert len(word_chars) == 4, "Word must be exactly 4 letters long"
    # Choose a single X position for all letters in the word sequence
    x_offset = random.uniform(0.0, 0.05)
    # First letter Y position in (0.04, 0.08) relative to table center
    first_y_offset = random.uniform(0.04, 0.08)
    for i, char in enumerate(word_chars):
        if i == missing_index:
            continue  # Skip the missing letter (create blank)
        oid = letter_map[char]
        letter_shape = library_call(
            "usd",
            oid=oid,
            keywords=[f"letter_{char}", "alphabet", "plastic", "educational", "word_sequence"],
        )
        # All letters use the same X offset
        # Y positions: first letter at first_y_offset, subsequent letters decrease by 0.06m (left to right: +Y to -Y)
        y_offset = first_y_offset - i * 0.06
        placed_letter = transform_shape(
            letter_shape,
            translation_matrix([table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]])
        )
        letter_shapes.append(placed_letter)
    return concat_shapes(*letter_shapes) if letter_shapes else []

@register()
def place_random_letter(letter_oid: str, letter_char: str, table_top_center: P, existing_positions: List[P], is_correct: bool = False) -> Shape:
    """
    Place a single letter (either correct answer or distractor) in the wider area, avoiding forbidden zone and existing letters.
    """
    role_tag = "correct_answer" if is_correct else "distractor"
    letter_shape = library_call(
        "usd",
        oid=letter_oid,
        keywords=[f"{role_tag}_{letter_char}", "alphabet", "plastic", "educational", role_tag],
    )
    # Compute letter dimensions for collision-aware placement
    letter_min = compute_shape_min(letter_shape)
    letter_max = compute_shape_max(letter_shape)
    letter_size = letter_max - letter_min
    # Define placement bounds
    x_min_allowed, x_max_allowed = -0.2, 0.15
    y_min_allowed, y_max_allowed = -0.25, 0.25
    # Forbidden zone: x in (-0.05, 0.08), y in (-0.1, 0.1)
    forbidden_x_min, forbidden_x_max = -0.05, 0.08
    forbidden_y_min, forbidden_y_max = -0.1, 0.1
    min_letter_spacing = 0.07
    max_attempts = 2000
    for _ in range(max_attempts):
        x_offset = random.uniform(x_min_allowed, x_max_allowed)
        y_offset = random.uniform(y_min_allowed, y_max_allowed)
        candidate_pos = [table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]]
        # Check if the sampled point is in the forbidden zone
        if (forbidden_x_min < x_offset < forbidden_x_max) and (forbidden_y_min < y_offset < forbidden_y_max):
            continue
        # Ensure the entire letter fits within the broader table area
        if (abs(x_offset) + letter_size[0] / 2 > 0.25) or (abs(y_offset) + letter_size[1] / 2 > 0.25):
            continue
        # Check distance to all existing positions
        too_close = False
        for existing_pos in existing_positions:
            if positions_too_close(candidate_pos, existing_pos, min_dist=min_letter_spacing):
                too_close = True
                break
        if too_close:
            continue
        # Valid position found
        break
    else:
        # Fallback positions outside forbidden zone
        fallback_positions = [
            [-0.18, 0.22],
            [-0.18, -0.22],
            [0.12, 0.22],
            [0.12, -0.22],
            [-0.15, 0.20],
            [-0.15, -0.20],
            [0.10, 0.20],
            [0.10, -0.20]
        ]
        candidate_pos = None
        for fx, fy in fallback_positions:
            fallback_pos = [table_top_center[0] + fx, table_top_center[1] + fy, table_top_center[2]]
            too_close = False
            for existing_pos in existing_positions:
                if positions_too_close(fallback_pos, existing_pos, min_dist=min_letter_spacing):
                    too_close = True
                    break
            if not too_close:
                candidate_pos = fallback_pos
                x_offset, y_offset = fx, fy
                break
        if candidate_pos is None:
            x_offset, y_offset = -0.15, 0.20
            candidate_pos = [table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]]
    # Translate to the final position on the table
    placed_letter = transform_shape(
        letter_shape,
        translation_matrix([table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]])
    )
    return placed_letter

# Expanded word list as provided by user
EXPANDED_WORD_LIST = [
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

# Filter words to ensure all have unique letters AND are exactly 4 letters long
def has_unique_letters(word: str) -> bool:
    return len(set(word)) == len(word)

# Create final word list with only 4-letter words having unique letters
VALID_WORDS = [word.upper() for word in EXPANDED_WORD_LIST if len(word) == 4 and has_unique_letters(word.upper())]

# All available letters for distractors
ALL_LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

@register()
def table_with_fill_in_blank_letters() -> Shape:
    # Load the table
    table_shape = library_call(
        "usd",
        oid="table_000",
        keywords=["workspace_table", "table", "white", "desktop_with_legs"],
    )
    # Get table top info for placement
    table_top_center, _ = get_tabletop_placement_area(table_shape)

    # Define available letters mapping
    letter_map = {
        'A': 'benchmark_letter_026',
        'B': 'benchmark_letter_027',
        'C': 'benchmark_letter_028',
        'D': 'benchmark_letter_029',
        'E': 'benchmark_letter_030',
        'F': 'benchmark_letter_031',
        'G': 'benchmark_letter_032',
        'H': 'benchmark_letter_033',
        'I': 'benchmark_letter_034',
        'J': 'benchmark_letter_035',
        'K': 'benchmark_letter_036',
        'L': 'benchmark_letter_037',
        'M': 'benchmark_letter_038',
        'N': 'benchmark_letter_039',
        'O': 'benchmark_letter_040',
        'P': 'benchmark_letter_041',
        'Q': 'benchmark_letter_042',
        'R': 'benchmark_letter_043',
        'S': 'benchmark_letter_044',
        'T': 'benchmark_letter_045',
        'U': 'benchmark_letter_046',
        'V': 'benchmark_letter_047',
        'W': 'benchmark_letter_048',
        'X': 'benchmark_letter_049',
        'Y': 'benchmark_letter_050',
        'Z': 'benchmark_letter_051',
    }

    # Select a random valid 4-letter word
    if not VALID_WORDS:
        selected_word = "BEAD"  # fallback
    else:
        selected_word = random.choice(VALID_WORDS)
    word_chars = list(selected_word)

    # Missing index must be 1 or 2 (middle two positions of 4-letter word)
    missing_index = random.choice([1, 2])
    missing_char = word_chars[missing_index]

    # Place the word letters (with one missing) in aligned row
    word_letters_shape = library_call(
        "place_word_letters_aligned",
        word_chars=word_chars,
        missing_index=missing_index,
        table_top_center=table_top_center
    )

    # Select 3 distractor letters that are not in the original word and are available
    available_distractors = [letter for letter in ALL_LETTERS if letter not in word_chars and letter in letter_map]
    if len(available_distractors) < 3:
        # Fallback: use a simple 4-letter word with more available distractors
        selected_word = "BEAD"
        word_chars = list(selected_word)
        missing_index = random.choice([1, 2])
        missing_char = word_chars[missing_index]
        available_distractors = [letter for letter in ALL_LETTERS if letter not in word_chars and letter in letter_map]

    # Randomly select 3 distractor letters
    distractor_chars = random.sample(available_distractors, 3)

    # The four random letters: [correct_answer, distractor1, distractor2, distractor3]
    random_letters = [missing_char] + distractor_chars

    # Place the four random letters
    random_letter_shapes = []
    placed_positions = []  # Track positions for spacing

    # Add positions of already placed word letters to avoid collisions
    if word_letters_shape:
        for letter_part in word_letters_shape:
            letter_info = get_object_info([letter_part])
            placed_positions.append(letter_info["center"])

    for i, char in enumerate(random_letters):
        oid = letter_map[char]
        is_correct = (i == 0)  # First one is the correct answer
        random_letter_shape = library_call(
            "place_random_letter",
            letter_oid=oid,
            letter_char=char,
            table_top_center=table_top_center,
            existing_positions=placed_positions,
            is_correct=is_correct
        )
        # Record the position of this letter for future spacing checks
        letter_info = get_object_info(random_letter_shape)
        placed_positions.append(letter_info["center"])
        random_letter_shapes.append(random_letter_shape)

    # Combine table, word letters, and random letters
    all_shapes = [table_shape]
    if word_letters_shape:
        all_shapes.append(word_letters_shape)
    all_shapes.extend(random_letter_shapes)
    return concat_shapes(*all_shapes)

@register()
def root_scene() -> Shape:
    return table_with_fill_in_blank_letters()
