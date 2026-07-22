# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from enum import Enum
import uuid
import re
import os, json
from geniesim_benchmark.plugins.logger import Logger
from dataclasses import dataclass
from collections import deque
from copy import deepcopy

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
    "stack_three_building_blocks": ["Stack"],
    "sort_cubes_by_size": ["MixedRules"],
    "pick_object_absolute_position": ["Follow", "PickUpOnGripper"],
    "pick_object_relative_position": ["Follow", "PickUpOnGripper"],
    "place_object_relative_position": ["Follow", "MixedRules"],
    "place_beverage_to_anothers_position": ["Follow", "MixedRules"],
}

# Subtask-mode scoring layout. When the manip task's yaml sets
# `instruction_mode: subtask`, the runtime no longer drives the original
# problems.json checker tree — it builds one checker per sub-instruction.
# Class-name list here must match the LEAF checker class names registered
# into task_progress when each `subtask_steps[i].checker` entry in
# `source/geniesim_benchmark/src/geniesim_benchmark/benchmark/config/llm_task/<task>/<inst>/instructions.json`
# is parsed. Container nodes like ActionSetWaitAll/ActionSetWaitAny do not
# appear here — flatten them to their leaves in registration order.
TASK_SUBTASK_STEPS = {
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
}


def get_task_steps(sub_task_name, instruction_mode="full"):
    """Return the scoring layout for `sub_task_name`. In subtask mode, fall
    back to the legacy TASK_STEPS list if no subtask layout is registered."""
    key = re.sub(r"_\d+$", "", sub_task_name)
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

    def __init__(self, task_name="", sub_task_name="", task_type="benchmark", instruction_mode="full"):
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
        self.sub_steps = get_task_steps(self.sub_task_name, self.instruction_mode)

    def safe_deepcopy(self, task_progress):
        all_progress = []
        for item in task_progress:
            progress = {}
            progress.setdefault("progress", {})
            progress["class_name"] = item["class_name"]
            progress["id"] = item["id"]
            if item.get("progress") and "SCORE" in item.get("progress", {}):
                progress["progress"]["SCORE"] = item["progress"]["SCORE"]
            if item.get("progress") and "STATUS" in item.get("progress", {}):
                progress["progress"]["STATUS"] = item["progress"]["STATUS"]
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
        if len(self.sub_steps) != 0:
            self.result["scores"] = {"STEPS": [], "E2E": 0}
            rows = self.result["progress"]
            substeps = deque(self.sub_steps)
            step_idx = 0
            i = 0
            while substeps and i < len(rows):
                target = substeps[0]
                if rows[i].get("class_name") == target:
                    # Collect best score from consecutive rows with the same
                    # class_name (e.g. left/right gripper variants inside
                    # ActionSetWaitAny). Either hand succeeding counts.
                    best_score = None
                    while i < len(rows) and rows[i].get("class_name") == target:
                        prog = rows[i].get("progress", {})
                        if "SCORE" in prog:
                            s = float(prog["SCORE"])
                            if best_score is None or s > best_score:
                                best_score = s
                        i += 1
                    step_name = substeps.popleft()
                    score_template = deepcopy(SCORE_TEMPLATE)
                    score_template["step"] = step_idx
                    score_template["name"] = step_name
                    if best_score is not None:
                        score_template["score"] = best_score
                    self.result["scores"]["STEPS"].append(score_template)
                    step_idx += 1
                else:
                    i += 1
            # Set E2E only if there is at least one recorded step
            if len(self.result["scores"]["STEPS"]) > 0 and self.result["scores"]["STEPS"][-1]["score"] == 1.0:
                self.result["scores"]["E2E"] = 1
        else:
            logger.warning(f"No substeps defined for task {self.task_name}")

    def assemble_result(self, info):
        self.result["code"] = int(ErrorCode.SUCCESS.value)

    def assemble_expect_result(self):
        self.result["code"] = int(ErrorCode.ABNORMAL_INTERRUPTION.value)
        self.result["msg"] = "expect: KeyboardInterrupt"

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
        return dir


def extract_scores(result, steps):
    rows = list(result["progress"])
    sub_scores = []
    i = 0

    for idx, step in enumerate(steps):
        while i < len(rows):
            if rows[i]["class_name"] == step:
                best_score = 0
                while i < len(rows) and rows[i]["class_name"] == step:
                    s = rows[i]["progress"].get("SCORE", 0)
                    if s > best_score:
                        best_score = s
                    i += 1
                sub_scores.append({"step": idx, "name": step, "score": best_score})
                break
            i += 1
    return sub_scores


def get_statistics(evaluate_results, steps):
    """Calculates statistics for evaluation results

    Args:
        evaluate_results: List of evaluation results
        steps: List of steps

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

    statistics["timecost"] = time_cost
    statistics["task_counts"] = task_counts

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
        self.sub_steps = get_task_steps(sub_task_name, instruction_mode)

    def update_current(self, eval: TaskEvaluation):
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
        task_steps = get_task_steps(self.task_name, self.instruction_mode)
        temp_scores = get_statistics(results, task_steps)["scores"]
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
            general_results["statistics"] = get_statistics(self.results, self.sub_steps)
            json.dump(general_results, f, indent=4)
