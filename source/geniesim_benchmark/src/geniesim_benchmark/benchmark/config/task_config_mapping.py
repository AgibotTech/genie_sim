# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .sandbox_tasks import SANDBOX_TASKS

TASK_MAPPING = {
    "bimanual_chip_handover": {
        "background": {
            "G1": "table_task_g1_op",
            "G2": "table_task_g2_op_s2r",
        },
        "eval_dims": {"manip": "bimanual_handover", "cognition": "semantic"},
    },
    "ci_pick_block_color": {
        "background": {
            "G2_crsB_omnipicker": "table_task_1_g2_crsb_op",
        },
        "eval_dims": {"manip": "pick", "cognition": "color"},
    },
    "arrange_block": {
        "background": {
            "G2": "table_task_g2_op_zeroshot_fixbase",
        },
        "eval_dims": {"manip": "planar_pick_place", "cognition": "position"},
    },
    "pick_largest_number": {
        "background": {
            "G2": "table_task_g2_op_zeroshot_fixbase",
        },
        "eval_dims": {"manip": "pick", "cognition": "number"},
    },
    "pick_object_color": {
        "background": {
            "G2": "table_task_g2_op_zeroshot_fixbase",
        },
        "eval_dims": {"manip": "pick", "cognition": "color"},
    },
    "pick_object_common_sense": {
        "background": {
            "G2": "table_task_g2_op_zeroshot_fixbase",
        },
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    "clean_the_desktop": {
        "background": {
            "G2": [
                "study_room_05_g2_op",
                "study_room_05_g2_op_backgen",
                "study_room_05_g2_op_camposgen",
                "study_room_05_g2_op_camqualgen",
                "study_room_05_g2_op_instructgen",
                "study_room_05_g2_op_posegen",
            ],
        },
        "eval_dims": "long-horizon",
    },
    "goal_image_clean_the_desktop": {
        "background": {
            "G2": [
                "study_room_05_g2_op",
                "study_room_05_g2_op_backgen",
                "study_room_05_g2_op_camposgen",
                "study_room_05_g2_op_camqualgen",
                "study_room_05_g2_op_instructgen",
                "study_room_05_g2_op_posegen",
            ],
        },
        "eval_dims": "long-horizon",
    },
    "clean_the_desktop_unseen": {
        "background": {
            "G2": "study_room_05_g2_op_fixbase",
        },
        "eval_dims": "long-horizon",
    },
    "pick_object_absolute_position_unseen": {
        "background": {
            "G2": "table_task_g2_op_zeroshot_fixbase",
        },
        "eval_dims": {"manip": "pick", "cognition": "position"},
    },
    "hold_pot": {
        "background": {
            "G2": "kitchen_02_g2_op",
        },
        "eval_dims": {"manip": "bimanual_hold", "cognition": "semantic"},
    },
    "open_door": {
        "background": {
            "G2": "home_g2_op",
        },
        "eval_dims": {"manip": "open", "cognition": "semantic"},
    },
    "pack_in_supermarket": {
        "background": {
            "G1": "table_task_g1_op",
            "G2": [
                "table_task_g2_op_pack",
                "table_task_g2_op_s2r",
            ],
        },
        "eval_dims": {"manip": "planar_pick_place_", "cognition": "semantic"},
    },
    "pick_billiards_color": {
        "background": {
            "G1": "table_task_g1_op_zeroshot",
            "G2": [
                "table_task_1_g2_op",
                "table_task_1_g2_op_backgen",
                "table_task_1_g2_op_camposgen",
                "table_task_1_g2_op_camqualgen",
                "table_task_1_g2_op_instructgen",
                "table_task_1_g2_op_posegen",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "color"},
    },
    "goal_image_pick_billiards_color": {
        "background": {
            "G1": "table_task_g1_op_zeroshot",
            "G2": [
                "table_task_1_g2_op",
                "table_task_1_g2_op_backgen",
                "table_task_1_g2_op_camposgen",
                "table_task_1_g2_op_camqualgen",
                "table_task_1_g2_op_instructgen",
                "table_task_1_g2_op_posegen",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "color"},
    },
    "wuji_pick_block": {
        "background": {"G2_crsB_wuji": "table_task_g2_crsB_wuji_old"},
        "eval_dims": {"manip": "pick", "cognition": "color"},
    },
    "pick_up_cola_to_coaster": {
        "background": {"G2_crsB_wuji": "table_task_g2_crsB_wuji_cola"},
        "eval_dims": {"manip": "g2op_spatial_pick_place", "cognition": "semantic"},
    },
    "pick_block_color": {
        "background": {
            "G1": [
                "table_task_g1_op",
                "table_task_g1_op_zeroshot",
            ],
            "G2": [
                "table_task_1_g2_op",
                "table_task_1_g2_op_backgen",
                "table_task_1_g2_op_camposgen",
                "table_task_1_g2_op_camqualgen",
                "table_task_1_g2_op_instructgen",
                "table_task_1_g2_op_posegen",
                "table_task_g2_op_zeroshot",
                "table_task_1_g2_crsB_wuji_old",
                "table_task_g2_op_s2r",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": [
                "table_task_arxone",
                "table_task_arxone_zeroshot",
            ],
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "color"},
    },
    "goal_image_pick_block_color": {
        "background": {
            "G1": [
                "table_task_g1_op",
                "table_task_g1_op_zeroshot",
            ],
            "G2": [
                "table_task_1_g2_op",
                "table_task_1_g2_op_backgen",
                "table_task_1_g2_op_camposgen",
                "table_task_1_g2_op_camqualgen",
                "table_task_1_g2_op_instructgen",
                "table_task_1_g2_op_posegen",
                "table_task_g2_op_zeroshot",
                "table_task_1_g2_crsB_wuji_old",
                "table_task_g2_op_s2r",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": [
                "table_task_arxone",
                "table_task_arxone_zeroshot",
            ],
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "color"},
    },
    "pick_block_color_instructgen": {
        "background": {
            "G2": "table_task_1_g2_op",
        },
        "eval_dims": {"manip": "pick", "cognition": "color"},
    },
    "pick_block_number": {
        "background": {
            "G1": "table_task_g1_op_zeroshot",
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
                "table_task_g2_op_zeroshot",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": [
                "table_task_arxone",
                "table_task_arxone_zeroshot",
            ],
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "number"},
    },
    "goal_image_pick_block_number": {
        "background": {
            "G1": "table_task_g1_op_zeroshot",
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
                "table_task_g2_op_zeroshot",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": [
                "table_task_arxone",
                "table_task_arxone_zeroshot",
            ],
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "number"},
    },
    "pick_block_shape": {
        "background": {
            "G1": "table_task_g1_op_zeroshot",
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
                "table_task_g2_op_zeroshot",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": [
                "table_task_arxone",
                "table_task_arxone_zeroshot",
            ],
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "shape"},
    },
    "goal_image_pick_block_shape": {
        "background": {
            "G1": "table_task_g1_op_zeroshot",
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
                "table_task_g2_op_zeroshot",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": [
                "table_task_arxone",
                "table_task_arxone_zeroshot",
            ],
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "shape"},
    },
    "pick_block_size": {
        "background": {
            "G1": "table_task_g1_op",
            "G2": [
                "table_task_1_g2_op",
                "table_task_1_g2_op_backgen",
                "table_task_1_g2_op_camposgen",
                "table_task_1_g2_op_camqualgen",
                "table_task_1_g2_op_instructgen",
                "table_task_1_g2_op_posegen",
                "table_task_g2_op_s2r",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "size"},
    },
    "goal_image_pick_block_size": {
        "background": {
            "G1": "table_task_g1_op",
            "G2": [
                "table_task_1_g2_op",
                "table_task_1_g2_op_backgen",
                "table_task_1_g2_op_camposgen",
                "table_task_1_g2_op_camqualgen",
                "table_task_1_g2_op_instructgen",
                "table_task_1_g2_op_posegen",
                "table_task_g2_op_s2r",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "size"},
    },
    "pick_common_sense": {
        "background": {
            "G1": "table_task_g1_op_zeroshot",
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
                "table_task_g2_op_zeroshot",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": [
                "table_task_arxone",
                "table_task_arxone_zeroshot",
            ],
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "common_sense"},
    },
    "goal_image_pick_common_sense": {
        "background": {
            "G1": "table_task_g1_op_zeroshot",
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
                "table_task_g2_op_zeroshot",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": [
                "table_task_arxone",
                "table_task_arxone_zeroshot",
            ],
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "common_sense"},
    },
    "pick_follow_logic_and": {
        "background": {
            "G2": "table_task_0_g2_op",
            "G2_90d": "table_task_g2_90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "logic"},
    },
    "pick_follow_logic_not": {
        "background": {
            "G2": "table_task_0_g2_op",
            "G2_90d": "table_task_g2_90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "logic"},
    },
    "pick_follow_logic_or": {
        "background": {
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "logic"},
    },
    "goal_image_pick_follow_logic_or": {
        "background": {
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "logic"},
    },
    "pick_object_absolute_position": {
        "background": {
            "G2": [
                "table_task_2_g2_op",
                "table_task_2_g2_op_backgen",
                "table_task_2_g2_op_camposgen",
                "table_task_2_g2_op_camqualgen",
                "table_task_2_g2_op_instructgen",
                "table_task_2_g2_op_posegen",
            ],
        },
        "eval_dims": {"manip": "pick", "cognition": "position"},
    },
    "pick_object_relative_position": {
        "background": {
            "G2": [
                "table_task_2_g2_op",
                "table_task_2_g2_op_backgen",
                "table_task_2_g2_op_camposgen",
                "table_task_2_g2_op_camqualgen",
                "table_task_2_g2_op_instructgen",
                "table_task_2_g2_op_posegen",
            ],
        },
        "eval_dims": {"manip": "pick", "cognition": "position"},
    },
    "pick_object_type": {
        "background": {
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "category"},
    },
    "goal_image_pick_object_type": {
        "background": {
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "category"},
    },
    "pick_specific_object": {
        "background": {
            "G1": [
                "table_task_g1_op",
                "table_task_g1_op_pick_specific_object",
            ],
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
                "table_task_g2_op_s2r",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
            "G2_omnipicker": "table_task_g2_omnipicker",
        },
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    "goal_image_pick_specific_object": {
        "background": {
            "G1": "table_task_g1_op",
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
                "table_task_g2_op_s2r",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    "pick_specific_object_unseen": {
        "background": {
            "G1": "table_task_g1_op_zeroshot",
            "G2": "table_task_g2_op_zeroshot_fixbase",
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": "table_task_arxone_zeroshot",
        },
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    "place_beverage_to_anothers_position": {
        "background": {
            "G2": [
                "table_task_2_g2_op",
                "table_task_2_g2_op_backgen",
                "table_task_2_g2_op_camposgen",
                "table_task_2_g2_op_camqualgen",
                "table_task_2_g2_op_instructgen",
                "table_task_2_g2_op_posegen",
            ],
        },
        "eval_dims": {"manip": "planar_pick_place", "cognition": "position"},
    },
    "goal_image_place_beverage_to_anothers_position": {
        "background": {
            "G2": [
                "table_task_2_g2_op",
                "table_task_2_g2_op_backgen",
                "table_task_2_g2_op_camposgen",
                "table_task_2_g2_op_camqualgen",
                "table_task_2_g2_op_instructgen",
                "table_task_2_g2_op_posegen",
            ],
        },
        "eval_dims": {"manip": "planar_pick_place", "cognition": "position"},
    },
    "place_block_into_box": {
        "background": {
            "G1": "table_task_g1_op",
            "G2": "table_task_2_g2_op",
        },
        "eval_dims": {"manip": "planar_pick_place_", "cognition": "semantic"},
    },
    "place_block_into_drawer": {
        "background": {
            "G1": "drawer_task_g1_op",
            "G2": [
                "drawer_task_g2_op",
                "drawer_task_g2_op_s2r",
            ],
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "place_object_into_box_color": {
        "background": {
            "G1": "table_task_g1_op",
            "G2": "table_task_g2_op_s2r",
        },
        "eval_dims": {"manip": "planar_pick_place_", "cognition": "color"},
    },
    "place_object_relative_position": {
        "background": {
            "G2": [
                "table_task_2_g2_op",
                "table_task_2_g2_op_backgen",
                "table_task_2_g2_op_camposgen",
                "table_task_2_g2_op_camqualgen",
                "table_task_2_g2_op_instructgen",
                "table_task_2_g2_op_posegen",
            ],
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "position"},
    },
    "goal_image_place_object_relative_position": {
        "background": {
            "G2": [
                "table_task_2_g2_op",
                "table_task_2_g2_op_backgen",
                "table_task_2_g2_op_camposgen",
                "table_task_2_g2_op_camqualgen",
                "table_task_2_g2_op_instructgen",
                "table_task_2_g2_op_posegen",
            ],
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "position"},
    },
    "pour_workpiece": {
        "background": {
            "G2": "laboratory_06_g2_op",
        },
        "eval_dims": {"manip": "pour", "cognition": "semantic"},
    },
    "scoop_popcorn": {
        "background": {
            "G2": "popcorn_g2_op",
        },
        "eval_dims": {"manip": "scoop", "cognition": "semantic"},
    },
    "sort_cubes_by_size": {
        "background": {
            "G2": "table_task_2_g2_op",
        },
        "eval_dims": {"manip": "planar_pick_place", "cognition": "size"},
    },
    "sort_fruit": {
        "background": {
            "G1": "table_task_g1_op",
            "G2": "table_task_g2_op_s2r",
        },
        "eval_dims": {"manip": "planar_pick_place_", "cognition": "semantic"},
    },
    "pack_groceries_into_carton": {
        "background": {
            "G2": "table_task_g2_op_zeroshot_fixbase",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "place_eggs_into_holder": {
        "background": {
            "G2": "table_task_g2_op_zeroshot_fixbase",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "set_the_table": {
        "background": {
            "G2": "table_task_g2_op_zeroshot_fixbase",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "sort_fruit_unseen": {
        "background": {
            "G2": "table_task_g2_op_zeroshot_fixbase",
        },
        "eval_dims": {"manip": "planar_pick_place_", "cognition": "semantic"},
    },
    "pick_object_relative_position_unseen": {
        "background": {
            "G2": "table_task_g2_op_zeroshot_fixbase",
        },
        "eval_dims": {"manip": "pick", "cognition": "position"},
    },
    "sort_number": {
        "background": {
            "G2": "table_task_2_g2_op",
        },
        "eval_dims": {"manip": "planar_pick_place", "cognition": "number"},
    },
    "sorting_packages": {
        "background": {
            "G2": [
                "warehouse_g2_op",
                "warehouse_g2_op_backgen",
                "warehouse_g2_op_camposgen",
                "warehouse_g2_op_camqualgen",
                "warehouse_g2_op_instructgen",
                "warehouse_g2_op_posegen",
            ],
        },
        "eval_dims": {"manip": "planar_pick_place_", "cognition": "semantic"},
    },
    "sorting_packages_continuous": {
        "background": {
            "G2": "warehouse_g2_op",
        },
        "eval_dims": {"manip": "planar_pick_place_", "cognition": "semantic"},
    },
    "stack_bowls": {
        "background": {
            "G2": "table_task_2_g2_op",
            "G2_crsB_wuji": "table_task_2_g2_crsB_wuji",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "find_block_in_three_storage_boxes": {
        "background": {
            "G2": "table_task_g2_op_zeroshot_fixbase",
        },
        "eval_dims": {"manip": "pull", "cognition": "logic"},
    },
    "stack_three_building_blocks": {
        "background": {
            "G2": "table_task_2_g2_op",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "stock_and_straighten_shelf": {
        "background": {
            "G2": [
                "market_00_g2_op",
                "market_00_g2_op_backgen",
                "market_00_g2_op_camposgen",
                "market_00_g2_op_camqualgen",
                "market_00_g2_op_instructgen",
                "market_00_g2_op_posegen",
            ],
        },
        "eval_dims": {"manip": "stock", "cognition": "semantic"},
    },
    "goal_image_stock_and_straighten_shelf": {
        "background": {
            "G2": [
                "market_00_g2_op",
                "market_00_g2_op_backgen",
                "market_00_g2_op_camposgen",
                "market_00_g2_op_camqualgen",
                "market_00_g2_op_instructgen",
                "market_00_g2_op_posegen",
            ],
        },
        "eval_dims": {"manip": "stock", "cognition": "semantic"},
    },
    "straighten_object": {
        "background": {
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "straighten", "cognition": "semantic"},
    },
    "goal_image_straighten_object": {
        "background": {
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
            ],
            "G2_90d": "table_task_g2_90d",
            "G2_crsB_90d": "table_task_g2_crsb90d",
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
        },
        "eval_dims": {"manip": "straighten", "cognition": "semantic"},
    },
    "take_wrong_item_shelf": {
        "background": {
            "G1": "market_01_g1_op",
            "G2": [
                "market_01_g2_op",
                "market_01_g2_op_backgen",
                "market_01_g2_op_camposgen",
                "market_01_g2_op_camqualgen",
                "market_01_g2_op_instructgen",
                "market_01_g2_op_posegen",
            ],
        },
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    "vln_test": {
        "background": {
            "G2": "vln_task_g2_op",
        },
        "eval_dims": {"manip": "navigation", "cognition": "vln"},
    },
    "wbc_pick_object": {
        "background": {
            "G2_90d": "wbc_task_1_g2_90d",
            "G2_crsB_90d": "wbc_task_1_g2_crsb90d",
        },
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "wbc_pick_trash": {
        "background": {"G2_crsB_90d": "wbc_task_1_g2_crsB_90d"},
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "wbc_pick_object_and_place": {
        "background": {
            "G2_90d": "wbc_task_3_g2_90d",
            "G2_crsB_90d": "wbc_task_3_g2_crsb90d",
        },
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "wbc_pick_object_from_shelf_high": {
        "background": {
            "G2_90d": "wbc_task_3_g2_90d",
            "G2_crsB_90d": "wbc_task_3_g2_crsb90d",
        },
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    "wbc_pick_object_from_shelf_low": {
        "background": {
            "G2_90d": "wbc_task_3_g2_90d",
            "G2_crsB_90d": "wbc_task_3_g2_crsb90d",
        },
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    "open_the_door": {
        "background": {
            "G2_90d": "open_the_door_g2_90d",
            "G2_crsB_90d": "open_the_door_g2_90d",
        },
        "eval_dims": {"manip": "push_pull", "cognition": "semantic"},
    },
    "sending_food": {
        "background": {
            "G2_90d": "sending_food_g2_90d",
            "G2_crsB_90d": "sending_food_g2_90d",
        },
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "pick_up_trash_wbc": {
        "background": {
            "G2_90d": "pick_up_trash_wbc_g2_90d",
            "G2_crsB_90d": "pick_up_trash_wbc_g2_90d",
        },
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "match_and_pick_from_conveyor": {
        "background": {
            "G2_90d": "match_and_pick_from_conveyor_g2_90d",
            "G2_crsB_90d": "match_and_pick_from_conveyor_g2_90d",
        },
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "put_the_plates_on_the_shelf": {
        "background": {
            "G2_90d": "put_the_plates_on_the_shelf_g2_90d",
            "G2_crsB_90d": "put_the_plates_on_the_shelf_g2_90d",
        },
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "open_the_roller_shutter": {
        "background": {
            "G2_90d": "open_the_roller_shutter_g2_90d",
            "G2_crsB_90d": "open_the_roller_shutter_g2_90d",
        },
        "eval_dims": {"manip": "push_pull", "cognition": "semantic"},
    },
    "clean_the_blackboard": {
        "background": {
            "G2_90d": "clean_the_blackboard_g2_90d",
            "G2_crsB_90d": "clean_the_blackboard_g2_90d",
        },
        "eval_dims": {"manip": "wipe", "cognition": "semantic"},
    },
    "transport_cartons_to_table_wbc": {
        "background": {
            "G2_90d": "transport_cartons_to_table_wbc_g2_90d",
            "G2_crsB_90d": "transport_cartons_to_table_wbc_g2_90d",
        },
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "find_target_in_drawer": {
        "background": {
            "G2": "table_task_g2_op_matgen",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "stack_multiple_blocks": {
        "background": {
            "G2": "table_task_g2_op_matgen",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "stack_blocks_based_on_pattern": {
        "background": {
            "G2": "table_task_g2_op_matgen",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    # InternDataEngine scene pack (intern_*)
    "intern_close_the_microwave": {
        "background": {
            "G2": "table_task_g2_op_intern_close_the_microwave",
            "lift2": "table_task_arxone_intern_y0800_z+1006",
            "split_aloha": "table_task_agilex_intern_y0860_z-0143",
        },
        "eval_dims": {"manip": "push_pull", "cognition": "category"},
    },
    "intern_heat_the_food_in_the_microwave": {
        "background": {
            "G2": "table_task_g2_op_intern_heat_the_food_in_the_microwave",
            "lift2": "table_task_arxone_intern_y0800_z+1006",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "logic"},
    },
    "intern_pull_the_storagefurniture": {
        "background": {
            "G2": "table_task_g2_op_intern_pull_the_storagefurniture",
            "lift2": "table_task_arxone_intern_y0800_z+1006",
            "split_aloha": "table_task_agilex_intern_y0860_z+0006",
        },
        "eval_dims": {"manip": "push_pull", "cognition": "category"},
    },
    "intern_hang_the_cup_on_rack": {
        "background": {
            "G2": "table_task_g2_op_intern_hang_the_cup_on_rack",
            "lift2": "table_task_arxone_intern_y0860_z+0256",
            "split_aloha": "table_task_agilex_intern_y0860_z-1543",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "category"},
    },
    "intern_insert_the_flower_in_vase": {
        "background": {
            "G2": "table_task_g2_op_intern_insert_the_flower_in_vase",
            "lift2": "table_task_arxone_intern_y0860_z+0256",
            "split_aloha": "table_task_agilex_intern_y0860_z-0143",
        },
        "eval_dims": {"manip": "insert", "cognition": "category"},
    },
    "intern_insert_the_markpen_in_penholder": {
        "background": {
            "G2": "table_task_g2_op_intern_insert_the_markpen_in_penholder",
            "lift2": "table_task_arxone_intern_y0860_z+0256",
            "split_aloha": "table_task_agilex_intern_y0860_z-1543",
        },
        "eval_dims": {"manip": "insert", "cognition": "category"},
    },
    "intern_pick_the_package_on_conveyor": {
        "background": {
            "G2": "table_task_g2_op_intern_pick_the_package_on_conveyor",
            "lift2": "table_task_arxone_intern_y1350_z+15506",
        },
        "eval_dims": {"manip": "pick", "cognition": "position"},
    },
    "intern_stack_bowls": {
        "background": {
            "G2": "table_task_g2_op_intern_stack_bowls",
            "lift2": "table_task_arxone_intern_y0800_z+0306",
            "split_aloha": "table_task_agilex_intern_y0860_z-0143",
        },
        "eval_dims": {"manip": "planar_pick_place", "cognition": "category"},
    },
    "intern_pack_in_objects": {
        "background": {
            "G2": "table_task_g2_op_intern_pack_in_objects",
            "lift2": "table_task_arxone_intern_y0650_z+0806",
        },
        "eval_dims": {"manip": "planar_pick_place", "cognition": "number"},
    },
    "intern_pack_out_objects": {
        "background": {
            "G2": "table_task_g2_op_intern_pack_out_objects",
            "lift2": "table_task_arxone_intern_y0700_z+0706",
            "split_aloha": "table_task_agilex_intern_y0800_z-0143",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "position"},
    },
    "intern_stack_a_beef_sandwich": {
        "background": {
            "G2": "table_task_g2_op_intern_stack_a_beef_sandwich",
            "lift2": "table_task_arxone_intern_y0750_z+0606",
        },
        "eval_dims": {"manip": "planar_pick_place", "cognition": "logic"},
    },
    "intern_store_objects_in_the_drawer": {
        "background": {
            "G2": "table_task_g2_op_intern_store_objects_in_the_drawer",
            "lift2": "table_task_arxone_intern_y0800_z+0506",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "logic"},
    },
    # ---- g290d room2 generalization tasks ----
    "pick_single_object": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "pick", "cognition": "position"},
    },
    "pick_specific_object_new": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    "pick_object_of_specific_category": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "pick", "cognition": "category"},
    },
    "pick_building_block_of_specific_color": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "pick", "cognition": "color"},
    },
    "pick_building_block_of_specific_size": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "pick", "cognition": "size"},
    },
    "pick_object_of_specific_size": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "pick", "cognition": "size"},
    },
    "pick_object_of_specific_shape": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "pick", "cognition": "shape"},
    },
    "pick_object_in_specific_container": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "pick", "cognition": "position"},
    },
    "pick_home_tableware": {
        "background": {
            "G2_90d": "home_00_g2_90d",
        },
        "eval_dims": {"manip": "pick", "cognition": "color"},
    },
    "pick_building_block_of_specific_position": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "pick", "cognition": "position"},
    },
    "pick_building_block_of_specific_corner_position": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "pick", "cognition": "position"},
    },
    "pick_building_block_of_relative_position": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "pick", "cognition": "position"},
    },
    "pick_combined_grasp_of_multiple_targets": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "pick", "cognition": "logic"},
    },
    "pick_combined_grasp_with_arm_and_position": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "pick", "cognition": "logic"},
    },
    "place_object_into_container": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "place_object_into_named_container": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "place_object_into_container_of_specific_color": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "color"},
    },
    "place_object_into_sized_container": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "size"},
    },
    "place_object_into_container_of_specific_shape": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "shape"},
    },
    "place_object_into_positional_container": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "position"},
    },
    "place_object_into_corner_container": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "position"},
    },
    "place_object_at_relative_position": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "planar_pick_place", "cognition": "position"},
    },
    "place_combined_object_into_container": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "logic"},
    },
    "pass_single_object": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "bimanual_handover", "cognition": "semantic"},
    },
    "pass_multiple_object": {
        "background": {
            "G2_90d": "table_task_g2_90d_room2",
            "G2_crsB_90d": "table_task_g2_crsb90d_room2",
        },
        "eval_dims": {"manip": "bimanual_handover", "cognition": "semantic"},
    },
    # sft single-task evals
    "geniesim_place_workpiece": {
        "background": {
            "G2": "place_workpiece_task_g2",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "phone_packaging": {
        "background": {
            "G2_90d": "phone_packaging_g2_90d",
            "G2_crsB_90d": "phone_packaging_g2_crsb90d",
        },
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "plug_charging_cable_into_phone": {
        "background": {
            "G2_crsB_90d": "table_task_g2_crsb_90d_room2",
        },
        "eval_dims": {"manip": "insert", "cognition": "semantic"},
    },
    "insert_building_block": {
        "background": {
            "G2_crsB_90d": "table_task_g2_crsb_90d_room2_building_block_main",
        },
        "eval_dims": {"manip": "insert", "cognition": "semantic"},
    },
    "insert_ethernet_cable": {
        "background": {
            "G2_crsB_90d": "table_task_g2_crsb_90d_room2_ethernet_main",
        },
        "eval_dims": {"manip": "insert", "cognition": "semantic"},
    },
    "install_bearing_shaft_seat5mm": {
        "background": {"G2_crsB_90d": "table_task_g2_crsb_90d_room2_bearing_main"},
        "eval_dims": {"manip": "insert", "cognition": "semantic"},
    },
    "put_the_cap_on_the_pen_new": {
        "background": {
            "G2_crsB_90d": "table_task_g2_crsb_90d_room2_pen_cap_main",
        },
        "eval_dims": {"manip": "insert", "cognition": "semantic"},
    },
    "insert_coin_into_money_box": {
        "background": {
            "G2_crsB_90d": "table_task_g2_crsb_90d_room2_coin_main",
        },
        "eval_dims": {"manip": "insert", "cognition": "semantic"},
    },
    "insert_straw_into_milk_tea_cup": {
        "background": {
            "G2_crsB_90d": "table_task_g2_crsb_90d_room2_straw_main",
        },
        "eval_dims": {"manip": "insert", "cognition": "semantic"},
    },
    # ---- uncovered arx/aloha eval tasks ----
    "chessboard_with_pieces_on_table_arx": {
        "background": {"arx": "table_task_arxone"},
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "math_blocks_equation_scene_aloha": {
        "background": {"split_aloha": "table_task_agilex"},
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "moving_chess_piece_to_execute_a_capture_arx": {
        "background": {"arx": "table_task_arxone"},
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "observe_block_sequence_then_press_buttons_aloha": {
        "background": {"split_aloha": "table_task_agilex"},
        "eval_dims": {"manip": "push_pull", "cognition": "semantic"},
    },
    "observe_block_sequence_then_press_buttons_arx": {
        "background": {"arx": "table_task_arxone"},
        "eval_dims": {"manip": "push_pull", "cognition": "semantic"},
    },
    "open_trash_can_aloha": {
        "background": {"split_aloha": "table_task_agilex"},
        "eval_dims": {"manip": "push_pull", "cognition": "semantic"},
    },
    "open_short_trash_can_and_throw_garbage_multi_robot": {
        "background": {
            "arx": "table_task_arxone",
            "agilex": "table_task_agilex",
            "G2_omnipicker": "table_task_g2_omnipicker",
            "G1": "table_task_g1_op",
        },
        "eval_dims": {"manip": "push_pull", "cognition": "semantic"},
    },
    "pick_largest_or_smallest_number_aloha": {
        "background": {"split_aloha": "table_task_agilex"},
        "eval_dims": {"manip": "pick", "cognition": "number"},
    },
    "pick_largest_or_smallest_number_arx": {
        "background": {"arx": "table_task_arxone"},
        "eval_dims": {"manip": "pick", "cognition": "number"},
    },
    "pick_object_relative_position_aloha": {
        "background": {"split_aloha": "table_task_agilex"},
        "eval_dims": {"manip": "pick", "cognition": "spatial"},
    },
    "pick_pen_from_holder_aloha": {
        "background": {"split_aloha": "table_task_agilex"},
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    "pick_pen_from_holder_arx": {
        "background": {"arx": "table_task_arxone"},
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    "place_object_relative_simple_aloha": {
        "background": {"split_aloha": "table_task_agilex"},
        "eval_dims": {"manip": "pick_place", "cognition": "spatial"},
    },
    "place_object_relative_simple_arx": {
        "background": {"arx": "table_task_arxone"},
        "eval_dims": {"manip": "pick_place", "cognition": "spatial"},
    },
    "press_button_aloha": {
        "background": {"split_aloha": "table_task_agilex"},
        "eval_dims": {"manip": "push_pull", "cognition": "semantic"},
    },
    "press_button_arx": {
        "background": {"arx": "table_task_arxone"},
        "eval_dims": {"manip": "push_pull", "cognition": "semantic"},
    },
    "stack_three_building_blocks_large_aloha": {
        "background": {"split_aloha": "table_task_agilex"},
        "eval_dims": {"manip": "stack", "cognition": "semantic"},
    },
    "store_chess_aloha": {
        "background": {"split_aloha": "table_task_agilex"},
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "pick_up_a_package": {
        "background": {"G2_crsB_wuji": "table_task_g2_crsB_wuji_warehouse_backgen"},
        "eval_dims": {"manip": "bimanual_hold", "cognition": "semantic"},
    },
    "cover_cup_lid": {
        "background": {"G2_crsB_wuji": "table_task_g2_crsB_wuji_cover_cup_lid"},
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "take_the_scissors": {
        "background": {"G2_crsB_wuji": "table_task_g2_crsB_wuji_take_the_scissors"},
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    "take_the_book": {
        "background": {"G2_crsB_wuji": "table_task_g2_crsB_wuji_take_the_book"},
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    # New instructions and geometry use separate foreground identities.
    "stack_bowls_wuji": {
        "background": {
            "G2": "table_task_2_g2_op",
            "G2_crsB_wuji": "table_task_2_g2_crsB_wuji",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
}


# Sandbox tasks use their complete authored scene, including interactive props.
TASK_MAPPING.update(
    {
        name: {
            "background": {"G2_crsB_90d": task["task_name"]},
            "eval_dims": {"manip": "long-horizon"},
        }
        for name, task in SANDBOX_TASKS.items()
    }
)
