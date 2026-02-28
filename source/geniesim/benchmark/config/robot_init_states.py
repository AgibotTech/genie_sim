# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

G1_DEFAULT_STATES = {
    "body_state": [0.0, 0.4363, 0.8727, 0.4],
    "init_arm": [
        -1.0751,
        0.6109,
        0.2793,
        -1.2846,
        0.7295,
        1.4957,
        -0.1868,
        1.0734,
        -0.6109,
        -0.2793,
        1.2846,
        -0.7313,
        -1.4957,
        0.1868,
    ],
    "init_hand": [0.0, 0.0],
}

G2_DEFAULT_STATES = {
    "body_state": [0.0, 0.0, 0.10025, 1.2172, -0.83423],
    "head_state": [0.0, 0.0, 0.11464],
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.925845,
        0.839257,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_1 = {
    "body_state": [0.0, 0.0, -0.18151424220741028, 1.2172, -0.83423],
    "head_state": [0.0, 0.0, 0.11464],
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.925845,
        0.839257,
    ],
    "init_hand": [0.0, 0.0],
}

G2_DEFAULT_2 = {
    "body_state": [0.0, 0.0, 0.10025, 1.2172, -0.83423],
    "head_state": [0.0, 0.0, 0.0],
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.925845,
        0.839257,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_3 = {
    "body_state": [0.0, 0.0, -0.18151424220741028, 1.2172, -0.83423],
    "head_state": [0.0, 0.0, 0.0],
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.925845,
        0.839257,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_4 = {
    "body_state": [1.57, 0.0, -0.31939525311, 1.34390352404, -1.04545222194],
    "head_state": [0.0, 0.0, 0.11464],
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.925845,
        0.839257,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_5 = {
    "body_state": [0.0, 0.0, -0.3617, 1.583, -0.8342],
    "head_state": [0.0, 0.0, 0.1745],
    "init_arm": [
        0.7459,
        -0.7458,
        -1.5375,
        -1.5828,
        0.2801,
        -0.8858,
        -0.8803,
        -0.7279,
        -0.6707,
        1.5386,
        -1.54,
        -0.283,
        -0.911,
        0.8175,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_6 = {
    "body_state": [0.0, 0.0, -0.181514, 1.2172, -0.83423],
    "head_state": [0.0, 0.0, 0.0],
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.925845,
        0.839257,
    ],
    "init_hand": [0.0, 0.0],
}

TASK_INFO_DICT = {
    "pick_billards_color": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_block_color": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_block_shape": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_block_size": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_block_number": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_cup_size": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_fruit_size": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_object_type": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_pen_color": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_specific_object": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_accessory": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_object_position": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_stationery": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_follow_logic_and": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_follow_logic_not": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_follow_logic_or": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_block_into_box": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "stable_grasp": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_into_box_position": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_into_box_color": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_into_box_size": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "put_pen_into_penholder": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "sort_accessory": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "straighten_object": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "sort_fruit": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_book": {"G2_omnipicker": G2_STATES_1},
    "place_book_hard": {"G2_omnipicker": G2_STATES_1},
    "open_door": {"G2_omnipicker": G2_STATES_1},
    "bimanual_hold_ball": {"G2_omnipicker": G2_STATES_1},
    "take_book": {"G2_omnipicker": G2_STATES_1},
    "clean_the_desktop": {"G2_omnipicker": G2_DEFAULT_STATES},
    "empty_desktop_bin": {"G2_omnipicker": G2_DEFAULT_STATES},
    "dump_trash_kitchen": {"G2_omnipicker": G2_STATES_1},
    "throw_away_garbage": {"G2_omnipicker": G2_DEFAULT_STATES},
    "heat_food": {"G2_omnipicker": G2_STATES_1},
    "hold_pot": {"G2_omnipicker": G2_STATES_1},
    "hang_tableware": {"G2_omnipicker": G2_STATES_1},
    "put_utensil_turn_faucet": {"G2_omnipicker": G2_STATES_1},
    "put_pen_into_penholder": {"G2_omnipicker": G2_STATES_1},
    "store_objects_in_drawer": {"G2_omnipicker": G2_STATES_3},
    "pick_common_sense": {"G2_omnipicker": G2_DEFAULT_STATES},
    "pour_workpiece": {"G2_omnipicker": G2_STATES_1},
    "scoop_popcorn": {"G2_omnipicker": G2_STATES_6},
    "sorting_packages": {"G2_omnipicker": G2_STATES_4},
    "sorting_packages_continuous": {"G2_omnipicker": G2_STATES_4},
    "chassis_at_target": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "take_wrong_item_shelf": {
        "G2_omnipicker": G2_STATES_5,
    },
    "stock_and_straighten_shelf": {
        "G2_omnipicker": G2_STATES_1,
    },
}
