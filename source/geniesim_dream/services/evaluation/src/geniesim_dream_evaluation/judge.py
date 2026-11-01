"""Unified dispatcher: route instruction to bread / fold_towel / fold_cloth / doll / generic."""

from __future__ import annotations

from typing import Any, Dict

from geniesim_dream_evaluation.task_router import (
    TaskFamily,
    UnsupportedTaskError,
    classify_task_family,
)
from geniesim_dream_evaluation.flows.doll_inout.judge import judge_pair as judge_doll_inout
from geniesim_dream_evaluation.flows.bread_takeout.judge import judge_pair as judge_bread_takeout
from geniesim_dream_evaluation.flows.fold_towel.judge import judge_pair as judge_fold_towel
from geniesim_dream_evaluation.flows.fold_cloth.judge import judge_pair as judge_fold_cloth
from geniesim_dream_evaluation.flows.generic.judge import judge_pair as judge_generic


def judge_pair(
    model,
    processor,
    pair: Dict[str, Any],
    *,
    enable_task_route: bool = True,
    generic_prompt_mode: str = "one_shot",
) -> Dict[str, Any]:
    """Public Dream evaluation entry used by tests/smoke scripts. Dispatch by instruction."""
    instruction = pair.get("instruction", "") or ""
    base: Dict[str, Any] = {
        "episode": pair.get("episode"),
        "wm_path": pair.get("wm_path"),
        "instruction": instruction,
        "success": None,
        "error": None,
        "task_family": None,
        "judge_version": "dream_release",
        "enable_task_route": bool(enable_task_route),
        "generic_prompt_mode": generic_prompt_mode,
    }

    def _run_generic() -> Dict[str, Any]:
        result = judge_generic(
            model,
            processor,
            pair,
            generic_prompt_mode=generic_prompt_mode,
        )
        if isinstance(result, dict):
            result.setdefault("task_family", "generic")
            result["enable_task_route"] = bool(enable_task_route)
            result["generic_prompt_mode"] = result.get(
                "generic_prompt_mode", generic_prompt_mode
            )
            return result
        base["task_family"] = "generic"
        return base

    if not enable_task_route:
        return _run_generic()

    try:
        family = classify_task_family(instruction)
    except UnsupportedTaskError:
        return _run_generic()

    base["task_family"] = family.value

    if family == TaskFamily.BREAD_TAKEOUT:
        result = judge_bread_takeout(model, processor, pair)
    elif family == TaskFamily.DOLL_INOUT:
        result = judge_doll_inout(model, processor, pair)
    elif family == TaskFamily.FOLD_TOWEL:
        result = judge_fold_towel(model, processor, pair)
    elif family == TaskFamily.FOLD_CLOTH:
        result = judge_fold_cloth(model, processor, pair)
    else:
        base["error"] = f"unhandled_task_family: {family}"
        return base

    if isinstance(result, dict):
        result.setdefault("task_family", family.value)
        result["enable_task_route"] = True
        result["generic_prompt_mode"] = generic_prompt_mode
        return result
    return base
