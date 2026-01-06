# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
TASK_MAPPING = {
    "pick_billards_color": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "pick", "cognition": "color"},
    },
    "pick_block_color": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "pick", "cognition": "color"},
    },
    "pick_block_shape": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "pick", "cognition": "shape"},
    },
    "pick_block_size": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "pick", "cognition": "size"},
    },
    "pick_block_number": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "pick", "cognition": "number"},
    },
    "pick_object_type": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "pick", "cognition": "category"},
    },
    "pick_pen_color": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "pick", "cognition": "color"},
    },
    "pick_specific_object": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    "pick_follow_logic_and": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "pick", "cognition": "logic"},
    },
    "pick_follow_logic_not": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "pick", "cognition": "logic"},
    },
    "pick_follow_logic_or": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "pick", "cognition": "logic"},
    },
    "place_block_into_box": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "planer_pick_place_", "cognition": "semantic"},
    },
    "place_object_into_box_position": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "planer_pick_place_", "cognition": "position"},
    },
    "place_object_into_box_color": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "planer_pick_place_", "cognition": "color"},
    },
    "place_object_into_box_size": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "planer_pick_place_", "cognition": "size"},
    },
    "put_pen_into_penholder": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "insert", "cognition": "semantic"},
    },
    "sort_accessory": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "planer_pick_place_", "cognition": "semantic"},
    },
    "straighten_object": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "straighten", "cognition": "semantic"},
    },
    "push_drawer_color": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "push", "cognition": "color"},
    },
    "pull_drawer_number": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "pull", "cognition": "number"},
    },
    "open_laptop_lid": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "open", "cognition": "semantic"},
    },
    "press_button_color": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "press", "cognition": "color"},
    },
    "turn_faucet": {
        "background": {"G2": "kitchen_04_g2"},
        "eval_dims": {"manip": "turn", "cognition": "semantic"},
    },
    "bimanual_hold_ball": {
        "background": {"G2": "study_room_05_g2"},
        "eval_dims": {"manip": "bimanual_hold", "cognition": "semantic"},
    },
    "place_object_on_convoy": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "planer_pick_place_", "cognition": "semantic"},
    },
    "clean_the_desktop": {
        "background": {"G2": "study_room_05_g2"},
        "eval_dims": "long-horizon",
    },
    "sort_fruit": {
        "background": {"G1": "table_task_g1", "G2": "table_task_g2"},
        "eval_dims": {"manip": "planer_pick_place_", "cognition": "semantic"},
    },
    "place_book": {
        "background": {"G2": "study_room_00_g2"},
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "place_book_hard": {
        "background": {"G2": "study_room_00_g2"},
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "open_door": {
        "background": {"G2": "home_g2"},
        "eval_dims": {"manip": "open", "cognition": "semantic"},
    },
    "take_book": {
        "background": {"G2": "study_room_00_g2"},
        "eval_dims": {"manip": "spatial_pick", "cognition": "semantic"},
    },
    "empty_desktop_bin": {
        "background": {"G2": "study_room_04_g2"},
        "eval_dims": {"manip": "dump", "cognition": "semantic"},
    },
    "dump_trash_kitchen": {
        "background": {"G2": "kitchen_00_g2"},
        "eval_dims": {"manip": "dump", "cognition": "semantic"},
    },
    "throw_away_garbage": {
        "background": {"G2": "kitchen_00_g2"},
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "heat_food": {"background": {"G2": "kitchen_01_g2"}, "eval_dims": "long-horizon"},
    "hold_pot": {
        "background": {"G2": "kitchen_02_g2"},
        "eval_dims": {"manip": "bimanual_hold", "cognition": "semantic"},
    },
    "hang_tableware": {
        "background": {"G2": "kitchen_03_g2"},
        "eval_dims": {"manip": "hang", "cognition": "semantic"},
    },
    "store_objects_in_drawer": {
        "background": {"G2": "home_00_g2"},
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "pick_common_sense": {
        "background": {"G2": "table_task_g2"},
        "eval_dims": {"manip": "pick", "cognition": "common_sense"},
    },
    "put_utensil_turn_faucet": {
        "background": {"G2": "kitchen_04_g2"},
        "eval_dims": {"manip": "turn", "cognition": "semantic"},
    },
}
