# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
"""
scene_name: table_with_words_and_distractors
description: A table with letters that include all letters of a randomly selected word from the expanded word list plus two additional distractor letters, all unique and spaced at least 0.07m apart, placed within a specific region avoiding a central sub-region.
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
def place_single_letter(letter_oid: str, letter_char: str, table_top_center: P, existing_positions: List[P]) -> Shape:
    """
    Place a single letter on the table within the allowed region without additional rotation,
    ensuring it is at least 0.07m away from all existing letters.
    """
    letter_shape = library_call(
        "usd",
        oid=letter_oid,
        keywords=[f"letter_{letter_char}", "alphabet", "plastic", "educational"],
    )
    # Compute letter dimensions for collision-aware placement
    letter_min = compute_shape_min(letter_shape)
    letter_max = compute_shape_max(letter_shape)
    letter_size = letter_max - letter_min
    # Define placement bounds relative to table top center
    x_min_allowed, x_max_allowed = -0.2, 0.15
    y_min_allowed, y_max_allowed = -0.25, 0.25
    # Forbidden zone: x in (-0.05, 0.08), y in (-0.1, 0.1)
    forbidden_x_min, forbidden_x_max = -0.05, 0.08
    forbidden_y_min, forbidden_y_max = -0.1, 0.1
    min_letter_spacing = 0.07
    max_attempts = 2000  # Increased attempts due to stricter constraints
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
        # Fallback: use a safe default position if sampling fails
        fallback_positions = [
            [-0.18, 0.22],
            [-0.18, -0.22],
            [0.12, 0.22],
            [0.12, -0.22],
            [-0.1, 0.20],
            [-0.1, -0.20],
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
            # Last resort: use first fallback even if slightly close
            x_offset, y_offset = -0.15, 0.20
            candidate_pos = [table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]]
    # Translate to the final position on the table
    placed_letter = transform_shape(
        letter_shape,
        translation_matrix([table_top_center[0] + x_offset, table_top_center[1] + y_offset, table_top_center[2]])
    )
    return placed_letter
# Word list for common sense task
EXPANDED_WORD_LIST = [
    "dog", "cat", "pig", "cow", "fox", "bird", "fish", "bear",
    "lion", "wolf", "frog", "duck", "bat", "owl", "crab", "bug",
    "sun", "sky", "sea", "ice", "rock", "sand", "wind", "rain",
    "coal", "fire", "leaf", "cave", "bus", "car", "pen", "cup",
    "box", "hat", "map", "toy", "bed", "key", "nut", "oil",
    "tea", "tie", "gold", "boat", "bowl", "coat", "coin", "desk",
    "dish", "fork", "gift", "kite", "net", "leg", "ear", "toe",
    "hand", "bone", "hair", "run", "fly", "eat", "cry", "play",
    "read", "jump", "old", "hot", "cold", "fast", "red", "blue",
    "pink", "time", "song", "bank", "camp", "cash", "chef", "king"
]
# Filter words to ensure all have unique letters
def has_unique_letters(word: str) -> bool:
    return len(set(word)) == len(word)
# Create final word list with only words having unique letters
VALID_WORDS = [word.upper() for word in EXPANDED_WORD_LIST if has_unique_letters(word.upper())]
# All available letters for distractors
ALL_LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
@register()
def table_with_words_and_distractors() -> Shape:
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

    # Select a random valid word
    selected_word = random.choice(VALID_WORDS)
    word_chars = list(selected_word)

    # Select 2 distractor letters that are not in the word and are available
    available_distractors = [letter for letter in ALL_LETTERS if letter not in word_chars and letter in letter_map]
    if len(available_distractors) < 2:
        # Fallback: use a simple word with more available distractors
        selected_word = "CAT"
        word_chars = list(selected_word)
        available_distractors = [letter for letter in ALL_LETTERS if letter not in word_chars and letter in letter_map]

    # Randomly select 2 distractor letters
    distractor_chars = random.sample(available_distractors, 2)

    # Combine word letters and distractor letters
    all_chars = word_chars + distractor_chars

    # Verify all letters are available in our asset library
    unavailable_chars = [char for char in all_chars if char not in letter_map]
    if unavailable_chars:
        # Fallback to a simple known available setup
        selected_word = "CAT"
        word_chars = list(selected_word)
        available_distractors = [letter for letter in ALL_LETTERS if letter not in word_chars and letter in letter_map]
        distractor_chars = random.sample(available_distractors, 2)
        all_chars = word_chars + distractor_chars

    # Place each letter with spacing constraint
    letter_shapes = []
    placed_positions = []  # Track positions of already placed letters
    for char in all_chars:
        oid = letter_map[char]
        letter_shape = library_call(
            "place_single_letter",
            letter_oid=oid,
            letter_char=char,
            table_top_center=table_top_center,
            existing_positions=placed_positions,
        )
        # Record the position of this letter for future spacing checks
        letter_info = get_object_info(letter_shape)
        placed_positions.append(letter_info["center"])
        letter_shapes.append(letter_shape)

    # Combine table and letters
    return concat_shapes(table_shape, *letter_shapes)
@register()
def root_scene() -> Shape:
    return table_with_words_and_distractors()
