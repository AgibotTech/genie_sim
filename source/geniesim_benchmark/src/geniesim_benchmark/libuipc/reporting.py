# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Map verified native Wuji scores to the common benchmark output contract."""

from __future__ import annotations
import copy


def common_result(task, execution, score, *, instruction="", endpoint=""):
    from geniesim_benchmark.plugins.output_system.eval_utils import TaskEvaluation

    evaluation = TaskEvaluation(task_name=task, sub_task_name=task, score_layout_name=task)
    evaluation.robot_type = "G2_crsB_wuji"
    evaluation.model_type = endpoint
    evaluation.task_instruction = instruction
    evaluation.policy_dispatch_count = execution.get("policy_dispatch_count", 0)
    clean = execution.get("execution_status") == "completed"
    evaluation.task_success = score.get("success") is True and clean
    evaluation.task_tree_finished = score.get("task_tree_finished") is True
    evaluation.result.update(step=execution.get("steps", 0), code=0 if clean else 1, msg=execution.get("error", ""))
    evaluation.update_progress(
        [
            {"class_name": stage["name"], "id": stage["step"], "progress": {"SCORE": stage["score"]}}
            for stage in score.get("stages", [])
        ]
    )
    evaluation.summarize_scores()
    result = evaluation.to_dir()
    result.update(
        execution_status=execution.get("execution_status"),
        inference_requests=execution.get("inference_requests", 0),
        task_success=evaluation.task_success,
        task_tree_finished=evaluation.task_tree_finished,
        native_score=copy.deepcopy(score),
        execution=copy.deepcopy(execution),
    )
    return result


def merge_score(execution, score):
    result = copy.deepcopy(execution)
    result["scoring_status"] = "scored"
    result["scoring"] = copy.deepcopy(score)
    for key in ("score", "success", "task_tree_finished", "events", "stages", "observed_frames", "profile_id"):
        if key in score:
            result[key] = copy.deepcopy(score[key])
    if result.get("execution_status") != "completed":
        result["success"] = False
    return result


def summarize(results):
    scored = [r for r in results if r.get("scoring_status") == "scored"]
    return {
        "episodes": len(results),
        "completed_episodes": sum(r.get("execution_status") == "completed" for r in results),
        "scored_episodes": len(scored),
        "successes": sum(r.get("success") is True for r in results),
        "mean_score": sum(r["score"] for r in scored) / len(scored) if scored else None,
        "results": results,
    }
