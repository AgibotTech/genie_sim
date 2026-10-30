# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
TASK_MAPPING = {
    "bimanual_chip_handover": {
        "background": {
            "G1": "table_task_g1_op",
        },
        "eval_dims": {"manip": "bimanual_handover", "cognition": "semantic"},
    },
    "clean_the_desktop": {
        "background": {
            "G2": "study_room_05_g2_op",
        },
        "eval_dims": "long-horizon",
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
        },
        "eval_dims": {"manip": "planar_pick_place_", "cognition": "semantic"},
    },
    "pick_billiards_color": {
        "background": {
            "G2": [
                "table_task_1_g2_op",
                "table_task_1_g2_op_backgen",
                "table_task_1_g2_op_camposgen",
                "table_task_1_g2_op_camqualgen",
                "table_task_1_g2_op_instructgen",
                "table_task_1_g2_op_posegen",
            ],
            "G2_90d": "table_task_g2_90d",
        },
        "eval_dims": {"manip": "pick", "cognition": "color"},
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
            ],
            "G2_90d": "table_task_g2_90d",
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
            ],
            "G2_90d": "table_task_g2_90d",
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
        },
        "eval_dims": {"manip": "pick", "cognition": "common_sense"},
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
        },
        "eval_dims": {"manip": "pick", "cognition": "logic"},
    },
    "pick_object_absolute_position": {
        "background": {
            "G2": "table_task_2_g2_op",
        },
        "eval_dims": {"manip": "pick", "cognition": "position"},
    },
    "pick_object_relative_position": {
        "background": {
            "G2": "table_task_2_g2_op",
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
        },
        "eval_dims": {"manip": "pick", "cognition": "category"},
    },
    "pick_specific_object": {
        "background": {
            "G1": "table_task_g1_op",
            "G2": [
                "table_task_0_g2_op",
                "table_task_0_g2_op_backgen",
                "table_task_0_g2_op_camposgen",
                "table_task_0_g2_op_camqualgen",
                "table_task_0_g2_op_instructgen",
                "table_task_0_g2_op_posegen",
            ],
            "G2_90d": "table_task_g2_90d",
        },
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    "place_beverage_to_anothers_position": {
        "background": {
            "G2": "table_task_2_g2_op",
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
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "place_object_into_box_color": {
        "background": {
            "G1": "table_task_g1_op",
        },
        "eval_dims": {"manip": "planar_pick_place_", "cognition": "color"},
    },
    "place_object_relative_position": {
        "background": {
            "G2": "table_task_2_g2_op",
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
        },
        "eval_dims": {"manip": "planar_pick_place_", "cognition": "semantic"},
    },
    "sort_number": {
        "background": {
            "G2": "table_task_2_g2_op",
        },
        "eval_dims": {"manip": "planar_pick_place", "cognition": "number"},
    },
    "sorting_packages": {
        "background": {
            "G2": "warehouse_g2_op",
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
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "stack_three_building_blocks": {
        "background": {
            "G2": "table_task_2_g2_op",
        },
        "eval_dims": {"manip": "spatial_pick_place", "cognition": "semantic"},
    },
    "stock_and_straighten_shelf": {
        "background": {
            "G2": "market_00_g2_op",
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
        },
        "eval_dims": {"manip": "straighten", "cognition": "semantic"},
    },
    "take_wrong_item_shelf": {
        "background": {
            "G2": "market_01_g2_op",
        },
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
    # RLinf MuJoCo collect task. Unlike the benchmark entries above it carries an
    # ``mjcf`` (relative to the geniesim_assets root); rlinf_geniesim's
    # ProcessManager resolves it via system_utils.assets_path().
    "geniesim_place_workpiece": {
        "background": {"G2": "place_workpiece_task_g2"},
        "eval_dims": {"manip": "place", "cognition": "semantic"},
        "mjcf": "mujoco/place_workpiece.xml",
    },
    "phone_packaging": {
        "background": {"G2_crsB_90d": "phone_packaging_g2_crsb90d"},
        "eval_dims": {"manip": "pick_place", "cognition": "semantic"},
    },
    "take_the_scissors": {
        "background": {"G2_crsB_wuji": "table_task_g2_crsB_wuji_take_the_scissors"},
        "eval_dims": {"manip": "pick", "cognition": "semantic"},
    },
}
