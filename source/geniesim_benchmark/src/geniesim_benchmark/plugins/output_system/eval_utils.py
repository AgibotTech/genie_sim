# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from enum import Enum
import uuid
import re
import os, json
from geniesim_benchmark.benchmark.config.sandbox_tasks import SANDBOX_TASKS
from geniesim_benchmark.plugins.logger import Logger
from dataclasses import dataclass
from copy import deepcopy
from geniesim_benchmark.plugins.output_system.score_groups import (
    SCORE_SCHEMA_VERSION,
    score_group_ids,
    goal_tree_metadata,
    score_layout_steps,
    normalize_steps,
    episode_complete,
    episode_counts,
)
from geniesim_benchmark.plugins.output_system.weighted_scoring import (
    WeightProfile,
    add_weighted_score,
    canonical_score_layout,
    get_weight_profile,
    weighted_score_breakdown,
)
from geniesim_benchmark.plugins.output_system.policy_budget import (
    add_policy_budget_scores,
    policy_budget_breakdown,
)

logger = Logger()


class ErrorCode(Enum):
    INIT_VALUE = -1
    SUCCESS = 0
    ABNORMAL_INTERRUPTION = 1
    OUT_OF_MAX_STEP = 2

    UNKNOWN_ERROR = 500


EVAL_TEMPLATE = {
    "task_type": "benchmark",
    "task_uid": str(uuid.uuid4()),
    "task_name": "",
    "robot_type": "",
    "stage": "",
    "result": {"code": int(ErrorCode.INIT_VALUE.value), "step": 0, "msg": ""},
    "start_time": "",
    "end_time": "",
    "duration": "",
    "fps": "",
    "operator": "",
}
TASK_STEPS = {
    "cloth_hanger": [
        "WujiLiftCloth",
        "WujiLiftHanger",
        "WujiHangerNearCollar",
        "WujiHangerInserted",
        "WujiGarmentHung",
    ],
    "fold_cloth": ["WujiFoldLeftSide", "WujiFoldRightSide", "WujiFoldHemToCollar"],
    "drawstring_bag": [
        "WujiBagPlaced",
        "WujiBottle1InBin",
        "WujiBottle2InBin",
        "WujiBothBottlesInBag",
        "WujiBagCinched",
    ],
    "bimanual_chip_handover": ["Upright"],
    "clean_the_desktop": ["Inside", "PushPull", "Inside", "RelativePositionChecker", "Upright"],
    "hold_pot": ["LiftUp", "InBBox", "Upright"],
    "open_door": ["PushPull"],
    "pick_billiards_color": ["Follow", "PickUpOnGripper"],
    "pick_block_color": ["Follow", "PickUpOnGripper"],
    "pick_block_number": ["Follow", "PickUpOnGripper"],
    "pick_block_shape": ["Follow", "PickUpOnGripper"],
    "pick_block_size": ["Follow", "PickUpOnGripper"],
    "pick_common_sense": ["Follow", "PickUpOnGripper"],
    "pick_object_type": ["Follow", "PickUpOnGripper"],
    "pick_specific_object": ["Follow", "PickUpOnGripper"],
    "place_object_into_box_color": ["Follow", "PickUpOnGripper", "Inside"],
    "sort_fruit": ["Inside"],
    "straighten_object": ["VLM"],
    "sorting_packages": ["Follow", "PickUpOnGripper", "Inside", "Upright", "PickUpOnGripper", "Inside"],
    "sorting_packages_continuous": ["Upright", "Upright", "Upright", "Upright"],
    "pour_workpiece": ["Inside", "Inside", "Inside", "Inside"],
    "stock_and_straighten_shelf": ["Follow", "PickUpOnGripper", "InBBox", "Follow", "Upright"],
    "scoop_popcorn": ["VLM"],
    "place_block_into_box": ["Follow", "PickUpOnGripper", "Inside"],
    "take_wrong_item_shelf": ["Follow", "Inside"],
    "straighten_object": ["Follow", "Upright"],
    "pick_follow_logic_or": ["Follow", "PickUpOnGripper"],
    "pack_in_supermarket": ["Inside"],
    "place_block_into_drawer": ["Inside"],
    "sort_number": ["MixedRules"],
    "stack_bowls": ["Stack"],
    "stack_bowls_pickup": ["PickUpOnGripper", "Stack"],
    "stack_three_building_blocks": ["Stack"],
    "sort_cubes_by_size": ["MixedRules"],
    "pick_object_absolute_position": ["Follow", "PickUpOnGripper"],
    "pick_object_relative_position": ["Follow", "PickUpOnGripper"],
    "place_object_relative_position": ["Follow", "MixedRules"],
    "place_beverage_to_anothers_position": ["Follow", "MixedRules"],
    "open_short_trash_can_and_throw_garbage_multi_robot": ["PushPull", "PickUpOnGripper", "Inside"],
    # Two steps, each an ActionSetWaitAll pair: step 1 needs BOTH grippers on
    # the beam, step 2 needs the beam on BOTH supports. Caveat on the per-step
    # numbers below: score_layout_steps groups consecutive rows sharing a
    # class_name and takes max() over the run, so each pair is reported as
    # either-one-is-enough (one gripper, or one support, reads as a pass).
    # Runtime pass/fail and E2E are unaffected -- ActionSetWaitAll._is_done
    # requires all children -- so E2E stays strict while these two per-step
    # rates read optimistically. Registering them anyway beats an empty
    # layout, which would drop this task back to the "last leaf only"
    # fallback in checker_runner and report nothing per-step.
    "place_long_block_on_supports_multi_robot": ["PickUpOnGripper", "Ontop"],
    "pick_specific_object_multi_robot": ["Follow", "PickUpOnGripper"],
    "store_chess_multi_robot": ["PickUpOnGripper", "Ontop"],
    "hang_cups_on_cup_tree_arxone": ["PickUpOnGripper", "Hang"],
    "hang_cups_on_cup_tree_agilex": ["PickUpOnGripper", "Hang"],
    "hang_cups_on_cup_tree_g1": ["PickUpOnGripper", "Hang"],
    "hang_cups_on_cup_tree_g2_omnipicker": ["PickUpOnGripper", "Hang"],
    "scrape_chicken_bone_into_bin_arxone": ["PickUpOnGripper", "MixedRules"],
    "scrape_chicken_bone_into_bin_agilex": ["PickUpOnGripper", "MixedRules"],
    "scrape_chicken_bone_into_bin_g1": ["PickUpOnGripper", "MixedRules"],
    "scrape_chicken_bone_into_bin_g2_omnipicker": ["PickUpOnGripper", "MixedRules"],
    "keyboard_keycap_swap_arxone": ["PickUpOnGripper", "LiftUp", "PlaceOnRivet"],
    "keyboard_keycap_swap_agilex": ["PickUpOnGripper", "LiftUp", "PlaceOnRivet"],
    "keyboard_keycap_swap_g1": ["PickUpOnGripper", "LiftUp", "PlaceOnRivet"],
    "keyboard_keycap_swap_g2_omnipicker": ["PickUpOnGripper", "LiftUp", "PlaceOnRivet"],
    "write_uppercase_letter_with_pen_arxone": ["PickUpOnGripper", "InkOverlap"],
    "write_uppercase_letter_with_pen_agilex": ["PickUpOnGripper", "InkOverlap"],
    "write_uppercase_letter_with_pen_g1": ["PickUpOnGripper", "InkOverlap"],
    "write_uppercase_letter_with_pen_g2_omnipicker": ["PickUpOnGripper", "InkOverlap"],
    "pour_50ml_into_graduated_cylinder_arxone": ["PickUpOnGripper", "PourVolume"],
    "pour_50ml_into_graduated_cylinder_agilex": ["PickUpOnGripper", "PourVolume"],
    "pour_50ml_into_graduated_cylinder_g1": ["PickUpOnGripper", "PourVolume"],
    "pour_50ml_into_graduated_cylinder_g2_omnipicker": ["PickUpOnGripper", "PourVolume"],
    "stack_three_building_blocks_large_multi_robot": [
        "PickUpOnGripper",
        "PickUpOnGripper",
        "PickUpOnGripper",
        "Stack",
    ],
    "plug_charging_cable_into_phone": ["PickUpOnGripper", "PickUpOnGripper", "PlugInserted"],
    "insert_building_block": ["PickUpOnGripper", "InBBox"],
    "insert_ethernet_cable": ["PickUpOnGripper", "PlugInserted"],
    # Retired bearing names remain score-layout IDs for archived results only.
    "install_bearing": ["PickUpOnGripper", "PlaceOnRivet"],
    "install_bearing_shaft_progress": ["PickUpOnGripper", "ShaftInserted", "PlaceOnRivet"],
    "install_bearing_shaft_seat5mm": ["PickUpOnGripper", "ShaftInserted", "PlaceOnRivet"],
    "install_bearing_shaft_seat5mm_geometry10": ["PickUpOnGripper", "ShaftInserted", "PlaceOnRivet"],
    "install_bearing_shaft_seat5mm_geometry10_angle10": ["PickUpOnGripper", "ShaftInserted", "PlaceOnRivet"],
    "put_the_cap_on_the_pen_new": ["PickUpOnGripper", "PickUpOnGripper", "PlugInserted", "Ontop"],
    "insert_coin_into_money_box": ["PickUpOnGripper", "Inside"],
    "insert_straw_into_milk_tea_cup": ["PickUpOnGripper", "MixedRules"],
    "pick_up_trash": ["PickUpOnGripper"],
    "phone_packaging_precise": [
        "PickUpOnGripper",
        "MixedRules",
        "PickUpOnGripper",
        "MixedRules",
        "PickUpOnGripper",
        "MixedRules",
    ],
    "phone_packaging": [
        "PickUpOnGripper",
        "Inside",
        "PickUpOnGripper",
        "Ontop",
        "PickUpOnGripper",
        "Cover",
    ],
    "pick_up_a_package": ["LiftUp", "MixedRules"],
    "cover_cup_lid": ["PickUpOnGripper", "Ontop", "MixedRules"],
    "take_the_scissors": ["PickUpOnGripper", "ScissorsHandleInserted", "ScissorsHandleActuated"],
    "take_the_book": ["Rotated", "MixedRules"],
    # Whole-body-control: robot drives to the trash, picks it up, drives to
    # the bin, drops it in, in the home_b_aligned living-room background.
    # Whole-body layouts retain one slot per logical goal. The live ADER tree
    # distinguishes either-hand alternatives from independent repeated goals.
    "sending_food": ["LiftUp", "ChassisAtTarget", "Ontop", "Ontop"],
    "pick_up_trash_wbc": [
        "RobotNearObject",
        "PickUpOnGripper",
        "RobotNearObject",
        "PickUpOnGripper",
        "RobotNearObject",
        "PushPull",
        "Inside",
    ],
    "open_the_door": ["ChassisAtTarget", "PushPull", "ChassisAtTarget", "PushPull"],
    "match_and_pick_from_conveyor": ["PickUpOnGripper", "RobotNearObject", "InBBox"],
    "put_the_plates_on_the_shelf": ["PickUpOnGripper", "PickUpOnGripper", "RobotNearObject", "InBBox", "InBBox"],
    "open_the_roller_shutter": ["RobotNearObject", "PickUpOnGripper", "PushPull", "PushPull", "ChassisAtTarget"],
    # ONLY the wipe is scored. Picking the eraser up and walking to the board
    # are prerequisites, not deliverables: their checkers still run and still
    # gate the episode (and the Onfloor guard still aborts it), but they are
    # deliberately absent from this layout, so score_layout_steps skips their
    # rows and the episode score is exactly the erased fraction, 0..1.
    # E2E therefore requires a complete wipe (scores[-1] == 1.0).
    "clean_the_blackboard": ["Erased"],
    # Load all four parcels onto the cart (gate 1: any two on; gate 2: all four),
    # push the cart to the sorting tables, then unload the two yellow parcels onto
    # either left table and the two white ones onto either right table. The two
    # tables of each pair are edge-to-edge, so one bbox per pair covers "either".
    # One layout entry per leaf row: score_layout_steps splits a run of same-class
    # rows over however many consecutive entries share the name, capped by the run
    # length, so a step registered with fewer entries than it has rows steals the
    # next step's entries.
    "transport_cartons_to_table_wbc": [
        "Ontop",
        "Ontop",
        "Ontop",
        "Ontop",  # (1) any two parcels on the cart deck
        "Ontop",
        "Ontop",
        "Ontop",
        "Ontop",  # (2) all four on the cart deck
        "InBBox",  # (3) cart pushed to the tables
        "InBBox",
        "InBBox",  # (4) yellow parcels on a left table
        "InBBox",
        "InBBox",  #     white parcels on a right table
    ],
    # New instructions and geometry use separate foreground identities.
    "stack_bowls_wuji": ["Stack"],
}

# Sandbox episodes have only a StepOut exit; they declare no scored success checks.
TASK_STEPS.update({name: [] for name in SANDBOX_TASKS})

# Archived subtask-result layouts, retained only for reading/reweighting old
# results. Benchmark execution always uses the full task checker tree.
TASK_SUBTASK_STEPS = {
    "put_the_cap_on_the_pen_new": ["PickUpOnGripper", "PickUpOnGripper", "PlugInserted", "Ontop"],
    "insert_coin_into_money_box": ["PickUpOnGripper", "Inside"],
    "insert_straw_into_milk_tea_cup": ["PickUpOnGripper", "MixedRules"],
    "hold_pot": ["LiftUp", "InBBox", "Upright"],
    "clean_the_desktop": ["Inside", "PushPull", "Inside", "RelativePosition", "Upright"],
    "open_door": ["PushPull"],
    "place_block_into_box": ["PickUpOnGripper", "Inside"],
    "pour_workpiece": ["Inside", "Inside", "Inside", "Inside"],
    "scoop_popcorn": ["VLM"],
    "sorting_packages": ["PickUpOnGripper", "Inside", "Upright", "PickUpOnGripper", "Inside"],
    "sorting_packages_continuous": [
        "PickUpOnGripper",
        "Inside",
        "Upright",
        "PickUpOnGripper",
        "Inside",
        "PickUpOnGripper",
        "Inside",
        "Upright",
        "PickUpOnGripper",
        "Inside",
        "PickUpOnGripper",
        "Inside",
        "Upright",
        "PickUpOnGripper",
        "Inside",
        "PickUpOnGripper",
        "Inside",
        "Upright",
        "PickUpOnGripper",
        "Inside",
    ],
    "stock_and_straighten_shelf": ["PickUpOnGripper", "InBBox", "Upright"],
    "take_wrong_item_shelf": ["Follow", "Inside"],
    # sim2real (a2d_probe) tasks. Segmentation mirrors the training data's
    # meta/agibot/segments.parquet; see scripts/add_s2r_subtask_steps.py.
    "pick_block_color": ["PickUpOnGripper"],
    "pick_block_size": ["PickUpOnGripper"],
    "pick_specific_object": ["PickUpOnGripper"],
    "place_object_into_box_color": ["PickUpOnGripper", "Inside"],
    "pack_in_supermarket": ["PickUpOnGripper", "Inside"],
    "sort_fruit": ["PickUpOnGripper", "Inside"],
    "bimanual_chip_handover": ["PickUpOnGripper", "StableGrasp", "Upright"],
    "place_block_into_drawer": ["PushPull", "PickUpOnGripper", "Inside", "PushPull"],
    # New instructions and geometry use separate foreground identities.
}


def get_task_steps(sub_task_name, instruction_mode="full"):
    """Read full task layouts or explicitly selected archived result layouts."""
    key = canonical_score_layout(re.sub(r"_\d+$", "", sub_task_name))
    if instruction_mode == "subtask" and key in TASK_SUBTASK_STEPS:
        return TASK_SUBTASK_STEPS[key]
    return TASK_STEPS.get(key, [])


TASK_IDS = {}

SCORE_TEMPLATE = {"step": 0, "name": "", "score": 0.0}


def generate_eval_file_path(dir_path, prefix_name):
    count = 0
    while os.path.exists(os.path.join(dir_path, f"{prefix_name}_{count:02d}.json")):
        count += 1
    new_filename = f"{prefix_name}_{count:02d}.json"
    return os.path.join(dir_path, new_filename)


@dataclass
class TaskEvaluation:
    task_name: str
    task_type: str
    task_uid: str
    robot_type: str
    model_type: str
    result: dict
    start_time: str
    end_time: str
    duration: float
    task_instruction: str

    def __init__(
        self,
        task_name="",
        sub_task_name="",
        task_type="benchmark",
        instruction_mode="full",
        score_layout_name: str | None = None,
    ):
        self.task_name = task_name
        self.sub_task_name = sub_task_name
        self.task_type = task_type
        self.task_uid = str(uuid.uuid4())
        self.robot_type = ""
        self.result = {
            "code": int(ErrorCode.SUCCESS.value),
            "step": 0,
            "msg": "",
            "progress": [],
        }
        self.start_time = ""
        self.end_time = ""
        self.model_type = ""
        self.task_instruction = ""
        self.duration = 0.0
        self.instruction_mode = instruction_mode
        self.score_layout_name = (
            canonical_score_layout(score_layout_name) if isinstance(score_layout_name, str) else score_layout_name
        )
        if score_layout_name is not None and (not isinstance(score_layout_name, str) or not score_layout_name):
            raise ValueError("ScoreLayout must name a registered scoring layout")
        scoring_task = score_layout_name or self.sub_task_name
        self.sub_steps = get_task_steps(scoring_task, self.instruction_mode)
        if score_layout_name is not None and not self.sub_steps:
            raise ValueError(f"Unknown ScoreLayout: {score_layout_name}")
        self.weight_profile = get_weight_profile(scoring_task, self.sub_steps, self.instruction_mode)
        self.task_success: bool | None = None
        self.task_tree_finished: bool | None = None
        self.policy_dispatch_count: int | None = None
        self._policy_dispatch_start: int | None = None

    def start_policy_statistics(self, policy: object) -> None:
        """Snapshot an optional lifetime counter without changing policy state."""
        count = getattr(policy, "policy_dispatch_count", None)
        self._policy_dispatch_start = count if type(count) is int and count >= 0 else None
        self.policy_dispatch_count = 0 if self._policy_dispatch_start is not None else None

    def update_policy_statistics(self, policy: object) -> None:
        """Keep an episode delta; missing/reset counters are unavailable, not zero."""
        start = self._policy_dispatch_start
        count = getattr(policy, "policy_dispatch_count", None)
        self.policy_dispatch_count = (
            count - start if start is not None and type(count) is int and count >= start else None
        )
        if self.policy_dispatch_count is None:
            self._policy_dispatch_start = None

    def safe_deepcopy(self, task_progress):
        score_groups = score_group_ids(task_progress)
        goal_metadata = goal_tree_metadata(task_progress)
        all_progress = []
        for item in task_progress:
            progress = {}
            progress.setdefault("progress", {})
            progress["class_name"] = item["class_name"]
            progress["id"] = item["id"]
            if item["id"] in score_groups:
                progress["score_group"] = score_groups[item["id"]]
            elif "score_group" in item:
                progress["score_group"] = item["score_group"]
            if item.get("progress") and "SCORE" in item.get("progress", {}):
                progress["progress"]["SCORE"] = item["progress"]["SCORE"]
            if item.get("progress") and "STATUS" in item.get("progress", {}):
                progress["progress"]["STATUS"] = item["progress"]["STATUS"]
            progress.update(
                goal_metadata.get(
                    item["id"],
                    {
                        key: item[key]
                        for key in ("goal_operator", "goal_children", "goal_min_complete", "goal_finished")
                        if key in item
                    },
                )
            )
            for key in ("subtask_step_idx", "subtask_role"):
                if key in item:
                    progress[key] = item[key]
            for key in ("GOAL_COMPLETE", "CARRY_DISTANCE", "POLICY_BUDGET"):
                if key in (item.get("progress") or {}):
                    progress["progress"][key] = deepcopy(item["progress"][key])
            all_progress.append(progress)

        return all_progress

    def update_from_dict(self, data: dict):
        simple_fields = [
            "task_name",
            "task_type",
            "task_uid",
            "robot_type",
            "model_type",
            "start_time",
            "end_time",
            "task_instruction",
            "duration",
        ]
        for key in simple_fields:
            if key in data:
                setattr(self, key, data[key])

    def update_progress(self, task_progress):
        self.result["progress"] = self.safe_deepcopy(task_progress)

    def summarize_scores(self):
        self.result["score_schema_version"] = SCORE_SCHEMA_VERSION
        self.result["score_layout"] = list(self.sub_steps)
        if self.sub_steps:
            scores = score_layout_steps(self.result["progress"], self.sub_steps)
            complete = episode_complete(self.result, scores, self.sub_steps)
            if self.score_layout_name is not None:
                # Explicit task layouts also require every declared checkpoint.
                complete = complete and all(step["score"] == 1.0 for step in scores)
                if self.score_layout_name in {"fold_cloth", "cloth_hanger", "drawstring_bag"}:
                    complete = complete and self.task_success is True and self.task_tree_finished is True
            self.result["scores"] = {"STEPS": scores, "E2E": int(complete)}
        else:
            logger.warning(f"No substeps defined for task {self.task_name}")
        add_weighted_score(self.result, self.result.get("scores", {}).get("STEPS", []), self.weight_profile)
        add_policy_budget_scores(self.result, self.result.get("scores", {}).get("STEPS", []), self.weight_profile)

    def assemble_result(self, info):
        self.result["code"] = int(ErrorCode.SUCCESS.value)

    def assemble_expect_result(self, error=None):
        self.result["code"] = int(ErrorCode.ABNORMAL_INTERRUPTION.value)
        self.result["msg"] = f"{type(error).__name__}: {error}" if error is not None else "KeyboardInterrupt"

    def to_dir(self):
        dir = {
            "task_type": self.task_type,
            "task_uid": self.task_uid,
            "task_name": self.task_name,
            "robot_type": self.robot_type,
            "result": self.result,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "model_type": self.model_type,
            "task_instruction": self.task_instruction,
        }
        dir["sub_task_name"] = self.sub_task_name
        dir["instruction_mode"] = self.instruction_mode
        if self.score_layout_name is not None:
            dir["score_layout_name"] = self.score_layout_name
        if self.policy_dispatch_count is not None:
            dir["policy_dispatch_count"] = self.policy_dispatch_count
        return dir


def extract_scores(result, steps):
    recorded = (result.get("scores") or {}).get("STEPS")
    return (
        normalize_steps(recorded, steps)
        if recorded is not None
        else score_layout_steps(result.get("progress", []), steps)
    )


def get_statistics(evaluate_results, steps, weight_profile: WeightProfile | None = None):
    """Calculates statistics for evaluation results

    Args:
        evaluate_results: List of evaluation results
        steps: List of steps
        weight_profile: Explicit task/mode weights. None reports weighted as unavailable.

    Returns:
        statistics: {
            "scores": {
                "task_instruction": [step0_avg_score, step1_avg_score, ...],
                ...
            },
            "weighted_step_scores": [
                {"step": 0, "name": <step_name>, "score": <count-weighted avg across tasks>},
                ...
            ],
            "timecost": total_time,
            "task_counts": {
                "task_instruction": count,
                ...
            }
        }
    """
    statistics = {}
    time_cost = 0
    # Stores cumulative scores for each step of each task: {task_key: {step_idx: total_score}}
    task_step_scores = {}
    # Stores the number of executions for each task
    task_counts = {}
    weighted_totals = {}
    weighted_process_total = 0.0
    weighted_bonus_total = 0.0
    weighted_complete_count = 0
    policy_budget_results = []

    # Collect all scores
    for item in evaluate_results:
        time_cost += float(item["duration"])
        task_key = item["task_instruction"]

        # Initialize task
        if task_key not in task_step_scores:
            task_step_scores[task_key] = {}
            task_counts[task_key] = 0

        task_counts[task_key] += 1

        # Extract step scores for the current result
        scores = extract_scores(item["result"], steps)
        policy_budget = policy_budget_breakdown(item["result"], scores, weight_profile)
        if policy_budget is not None:
            policy_budget_results.append(policy_budget)
        if weight_profile is not None:
            breakdown = weighted_score_breakdown(
                scores,
                weight_profile,
                item["result"].get("scores", {}).get("E2E"),
            )
            weighted_totals[task_key] = weighted_totals.get(task_key, 0.0) + breakdown.total
            weighted_process_total += breakdown.process
            weighted_bonus_total += breakdown.completion_bonus
            weighted_complete_count += int(breakdown.complete)
        for score_item in scores:
            step_idx = score_item["step"]
            score_value = score_item["score"]

            if step_idx not in task_step_scores[task_key]:
                task_step_scores[task_key][step_idx] = 0.0

            task_step_scores[task_key][step_idx] += score_value

    # Calculate average scores and organize by index
    sub_task_scores = {}
    for task_key, step_scores in task_step_scores.items():
        count = task_counts[task_key]
        # Create a list of scores indexed by index
        num_steps = len(steps)
        avg_scores = [0.0] * num_steps

        for step_idx, total_score in step_scores.items():
            avg_scores[step_idx] = round(total_score / count, 4)

        sub_task_scores[task_key] = avg_scores

    statistics["scores"] = sub_task_scores
    all_scores = [s for scores in sub_task_scores.values() for s in scores]
    total_counts = sum(task_counts.values())
    num_steps = len(steps) if steps else 1
    weighted_sum = sum(s * task_counts[task_key] for task_key, scores in sub_task_scores.items() for s in scores)
    statistics["average"] = round(weighted_sum / (total_counts * num_steps), 4) if total_counts else 0.0

    weighted_step_scores = []
    for step_idx, step_name in enumerate(steps):
        if total_counts:
            score = round(
                sum(scores[step_idx] * task_counts[task_key] for task_key, scores in sub_task_scores.items())
                / total_counts,
                4,
            )
        else:
            score = 0.0
        weighted_step_scores.append({"step": step_idx, "name": step_name, "score": score})
    statistics["weighted_step_scores"] = weighted_step_scores

    # Average end-to-end success rate: mean of the per-episode 0/1 E2E scores.
    total_episodes = len(evaluate_results)
    e2e_sum = sum(item["result"].get("scores", {}).get("E2E", 0) for item in evaluate_results)
    statistics["average_e2e"] = round(e2e_sum / total_episodes, 4) if total_episodes else 0.0
    # Historical/unsupported-policy rows have no counter. Report the measured
    # denominator explicitly instead of silently treating absent values as zero.
    dispatch_counts = [
        item["policy_dispatch_count"]
        for item in evaluate_results
        if type(item.get("policy_dispatch_count")) is int and item["policy_dispatch_count"] >= 0
    ]
    if dispatch_counts:
        statistics["policy_dispatch_count"] = sum(dispatch_counts)
        statistics["average_policy_dispatch_count"] = round(sum(dispatch_counts) / len(dispatch_counts), 4)
        statistics["policy_dispatch_episodes"] = len(dispatch_counts)
    # Compute from unrounded episode scores, rather than rounded per-step means.
    if weight_profile is not None:
        statistics["weighted"] = round(sum(weighted_totals.values()) / total_episodes, 4) if total_episodes else 0.0
        statistics["weighted_by_instruction"] = {
            key: round(total / task_counts[key], 4) for key, total in weighted_totals.items()
        }
        statistics["weighted_scoring"] = weight_profile.to_dict()
        for key, total in (
            ("weighted_process", weighted_process_total),
            ("weighted_completion_bonus", weighted_bonus_total),
            ("weighted_completion_rate", weighted_complete_count),
        ):
            statistics[key] = round(total / total_episodes, 4) if total_episodes else 0.0

    statistics["episode_counts"] = episode_counts(evaluate_results)
    statistics["run_complete"] = statistics["episode_counts"]["invalid"] == 0
    statistics["timecost"] = time_cost
    statistics["task_counts"] = task_counts
    if policy_budget_results:
        if any(b["scoring"] != policy_budget_results[0]["scoring"] for b in policy_budget_results[1:]):
            raise ValueError("Cannot aggregate different policy budget configurations")
        count = len(policy_budget_results)
        statistics["policy_budget_episodes"] = count
        for key, source in (
            ("policy_budget_weighted", "total"),
            ("policy_budget_process", "process"),
            ("policy_budget_completion_bonus", "completion_bonus"),
        ):
            statistics[key] = round(sum(b[source] for b in policy_budget_results) / count, 4)
        statistics["policy_budget_scoring"] = policy_budget_results[0]["scoring"]

    return statistics


class EvaluationSummary:
    def __init__(self, dir, task_name, sub_task_name, instruction_mode="full"):
        out_dir = os.path.join(dir, task_name, sub_task_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.out_dir = out_dir
        self.file_path = generate_eval_file_path(self.out_dir, "evaluate_ret")
        self.results = []
        self.current_eval: TaskEvaluation = None
        self.task_name = task_name
        self.sub_task_name = sub_task_name
        self.instruction_mode = instruction_mode
        self.score_layout_name = None
        self.sub_steps = get_task_steps(sub_task_name, instruction_mode)
        self.weight_profile = get_weight_profile(sub_task_name, self.sub_steps, instruction_mode)

    def update_current(self, eval: TaskEvaluation):
        if self.results and (
            self.sub_steps != eval.sub_steps
            or self.instruction_mode != eval.instruction_mode
            or self.score_layout_name != eval.score_layout_name
            or self.weight_profile != eval.weight_profile
        ):
            raise ValueError("Cannot combine different scoring layouts, modes or weight profiles")
        self.sub_steps = list(eval.sub_steps)
        self.instruction_mode = eval.instruction_mode
        self.score_layout_name = eval.score_layout_name
        self.weight_profile = eval.weight_profile
        self.current_eval = eval

    def make_temp_statistic(self):
        temp_results = list(self.results) + [self.current_eval.to_dir()]
        return self.to_dynamic_msg_pub(temp_results)

    def make_cache(self):
        if self.current_eval:
            self.results.append(self.current_eval.to_dir())
            self.dump_eval_result()
            self.current_eval = None

    def to_static_msg_pub(self, episode, episodes_per_task, sim_time):
        msg = {
            "sim_time": sim_time,
            "model_type": self.current_eval.model_type,
            "progress": "{0}/{1}".format(episode + 1, episodes_per_task),
            "task_name": self.current_eval.task_name,
            "task_instruction": self.current_eval.task_instruction,
        }
        return json.dumps(msg)

    def to_dynamic_msg_pub(self, results=None):
        results = self.results if results is None else results
        task_steps = self.sub_steps
        temp_scores = get_statistics(results, task_steps, self.weight_profile)["scores"]
        if self.current_eval.task_instruction not in temp_scores.keys():
            temp_statistic = {"cnt": 1}
            for step in task_steps:
                temp_statistic[step] = 0.0
        else:
            temp_statistic = temp_scores[self.current_eval.task_instruction]

        return json.dumps(temp_statistic)

    def dump_eval_result(self):
        os.makedirs(self.out_dir, exist_ok=True)
        logger.info(f"Evaluation result file generated at {self.file_path}")
        with open(self.file_path, "w+") as f:
            general_results = {}
            general_results["details"] = self.results
            general_results["statistics"] = get_statistics(self.results, self.sub_steps, self.weight_profile)
            json.dump(general_results, f, indent=4)
