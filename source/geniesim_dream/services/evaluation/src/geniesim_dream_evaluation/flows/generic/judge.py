"""Generic judge: one-shot (thinking/answer) or three-stage JSON."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from geniesim_dream_evaluation.frames import crop_start_end_top, sample_five_frames
from geniesim_dream_evaluation.parse import parse_json_object, parse_success
from geniesim_dream_evaluation.vlm_backend import run_qwen_vl_interleaved

from .prompt import (
    build_end_content,
    build_one_shot_content,
    build_start_content,
    build_verify_content,
)
from .resolver import resolve_task_completion
from .semantics import (
    normalize_end_state,
    normalize_start_state,
    normalize_trajectory_state,
)

STAGE1_MAX_NEW_TOKENS = 768
STAGE2_MAX_NEW_TOKENS = 1024
STAGE3_MAX_NEW_TOKENS = 1024
ONE_SHOT_MAX_NEW_TOKENS = 1024

GENERIC_PROMPT_MODES = frozenset({"one_shot", "three_stage"})


def _next_state(parsed: Optional[Dict[str, Any]], raw: str):
    if parsed is not None:
        return parsed
    return raw or "(unavailable)"


def _base_result(pair: Dict[str, Any], mode: str) -> Dict[str, Any]:
    return {
        "episode": pair.get("episode"),
        "wm_path": pair.get("wm_path"),
        "instruction": pair.get("instruction", ""),
        "frame_cache": None,
        "success": None,
        "thinking": "",
        "raw_response": "",
        "raw_stage1": "",
        "raw_stage2": "",
        "raw_stage3": "",
        "start_state": None,
        "end_state": None,
        "verify": None,
        "resolver_reason": None,
        "resolver_debug": None,
        "error": None,
        "prompt_family": "generic",
        "judge_version": "dream_generic_release",
        "task_family": "generic",
        "generic_prompt_mode": mode,
    }


def judge_pair(
    model,
    processor,
    pair: Dict[str, Any],
    *,
    generic_prompt_mode: str = "one_shot",
) -> Dict[str, Any]:
    mode = (generic_prompt_mode or "one_shot").strip().lower()
    if mode not in GENERIC_PROMPT_MODES:
        result = _base_result(pair, mode)
        result["error"] = f"invalid_generic_prompt_mode: {generic_prompt_mode!r}"
        return result

    if mode == "one_shot":
        return _judge_one_shot(model, processor, pair)
    return _judge_three_stage(model, processor, pair)


def _judge_one_shot(model, processor, pair: Dict[str, Any]) -> Dict[str, Any]:
    result = _base_result(pair, "one_shot")

    wm_path = pair.get("wm_path")
    if not wm_path or not os.path.isfile(wm_path):
        result["error"] = "wm_missing"
        return result

    wm_frames = sample_five_frames(wm_path)
    result["frame_cache"] = None
    if not wm_frames:
        result["error"] = "wm_no_frames"
        return result
    if len(wm_frames) < 5:
        result["error"] = "wm_too_few_frames"
        return result

    try:
        raw = run_qwen_vl_interleaved(
            model,
            processor,
            build_one_shot_content(pair.get("instruction", "") or "", wm_frames),
            max_new_tokens=ONE_SHOT_MAX_NEW_TOKENS,
        )
    except Exception as e:
        result["error"] = f"one_shot: {e}"
        return result

    result["raw_response"] = raw
    success, thinking, err = parse_success(raw)
    result["success"] = success
    result["thinking"] = thinking
    result["error"] = err
    result["resolver_reason"] = "one_shot_parse_success" if err is None else err
    result["resolver_debug"] = {"resolution_mode": "generic_release_one_shot"}
    return result


def _judge_three_stage(model, processor, pair: Dict[str, Any]) -> Dict[str, Any]:
    result = _base_result(pair, "three_stage")

    wm_path = pair.get("wm_path")
    if not wm_path or not os.path.isfile(wm_path):
        result["error"] = "wm_missing"
        return result

    wm_frames = sample_five_frames(wm_path)
    result["frame_cache"] = None
    if not wm_frames:
        result["error"] = "wm_no_frames"
        return result
    if len(wm_frames) < 5:
        result["error"] = "wm_too_few_frames"
        return result

    try:
        start_top, end_top = crop_start_end_top(wm_frames[0], wm_frames[-1])
    except Exception as e:
        result["error"] = f"top_crop_failed: {e}"
        return result

    mid_triplets = [wm_frames[1], wm_frames[2], wm_frames[3]]
    end_triplet = wm_frames[-1]
    instruction = pair.get("instruction", "") or ""

    try:
        raw1 = run_qwen_vl_interleaved(
            model,
            processor,
            build_start_content(instruction, start_top),
            max_new_tokens=STAGE1_MAX_NEW_TOKENS,
        )
    except Exception as e:
        result["error"] = f"stage1: {e}"
        return result
    result["raw_stage1"] = raw1
    start_state = normalize_start_state(parse_json_object(raw1))
    result["start_state"] = start_state

    try:
        raw2 = run_qwen_vl_interleaved(
            model,
            processor,
            build_end_content(
                instruction=instruction,
                start_frame=start_top,
                end_frame=end_top,
                end_triplet=end_triplet,
                start_state=_next_state(start_state, raw1),
            ),
            max_new_tokens=STAGE2_MAX_NEW_TOKENS,
        )
    except Exception as e:
        result["error"] = f"stage2: {e}"
        return result
    result["raw_stage2"] = raw2
    end_state = normalize_end_state(parse_json_object(raw2))
    result["end_state"] = end_state

    try:
        raw3 = run_qwen_vl_interleaved(
            model,
            processor,
            build_verify_content(
                instruction=instruction,
                mid_triplets=mid_triplets,
                end_triplet=end_triplet,
                start_state=_next_state(start_state, raw1),
                end_state=_next_state(end_state, raw2),
            ),
            max_new_tokens=STAGE3_MAX_NEW_TOKENS,
        )
    except Exception as e:
        result["error"] = f"stage3: {e}"
        return result
    result["raw_stage3"] = raw3
    result["raw_response"] = raw3
    verify = normalize_trajectory_state(parse_json_object(raw3))
    result["verify"] = verify

    if isinstance(verify, dict):
        summary = verify.get("trajectory_summary")
        if isinstance(summary, str) and summary.strip():
            result["thinking"] = summary.strip()
        else:
            reason = verify.get("completion_reason")
            if isinstance(reason, str) and reason.strip():
                result["thinking"] = reason.strip()
            else:
                result["thinking"] = json.dumps(verify, ensure_ascii=False)
    else:
        result["thinking"] = raw3 or ""

    success, reason, resolver_debug = resolve_task_completion(verify)
    result["success"] = success
    result["resolver_reason"] = reason
    result["resolver_debug"] = resolver_debug
    if success is None:
        result["error"] = reason
    return result
