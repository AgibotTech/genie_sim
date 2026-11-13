"""Judge one WM video with dream_bread_takeout_release.

Stage1 -> Stage2 -> Stage3 -> bread resolver. Stage4 is disabled.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from geniesim_dream_evaluation.frames import crop_start_end_top, sample_five_frames
from geniesim_dream_evaluation.parse import parse_json_object
from .prompt import (
    build_end_content,
    build_start_content,
    build_verify_content,
)
from .resolver import (
    derive_source_clear_candidate,
    resolve_task_completion,
)
from .semantics import (
    is_supported_bread_takeout_instruction,
    normalize_end_state,
    normalize_start_state,
    normalize_trajectory_state,
)
from geniesim_dream_evaluation.vlm_backend import run_qwen_vl_interleaved

STAGE1_MAX_NEW_TOKENS = 768
STAGE2_MAX_NEW_TOKENS = 1024
STAGE3_MAX_NEW_TOKENS = 1024


def _next_state(parsed: Optional[Dict[str, Any]], raw: str):
    if parsed is not None:
        return parsed
    return raw or "(unavailable)"


def judge_pair(model, processor, pair: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
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
        "raw_stage4": "",
        "raw_stage4_by_target": {},
        "start_state": None,
        "end_state": None,
        "verify": None,
        "end_completion_candidate": None,
        "end_completion_candidate_scope": "source_toaster_clear_only",
        "wrist_recheck_triggered": False,
        "wrist_recheck_target_ids": [],
        "wrist_recheck_completed_target_ids": [],
        "wrist_recheck_errors_by_target": {},
        "wrist_recheck": None,
        "wrist_recheck_error": None,
        "resolver_reason": None,
        "resolver_debug": None,
        "error": None,
        "prompt_family": "bread_takeout",
        "judge_version": "dream_bread_takeout_release",
        "task_family": "bread_takeout",
    }

    instruction = pair.get("instruction", "") or ""
    if not is_supported_bread_takeout_instruction(instruction):
        result["error"] = "unsupported_bread_takeout_instruction"
        return result

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
    start_state = normalize_start_state(parse_json_object(raw1), instruction)
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
    result["end_completion_candidate"] = derive_source_clear_candidate(
        end_state if isinstance(end_state, dict) else None
    )

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
            result["thinking"] = json.dumps(verify, ensure_ascii=False)
    else:
        result["thinking"] = raw3 or ""

    success, reason, resolver_debug = resolve_task_completion(
        instruction=instruction,
        start_state=start_state if isinstance(start_state, dict) else None,
        end_state=end_state if isinstance(end_state, dict) else None,
        verify_state=verify if isinstance(verify, dict) else None,
        wrist_state=None,
    )
    result["success"] = success
    result["resolver_reason"] = reason
    result["resolver_debug"] = resolver_debug
    if isinstance(resolver_debug, dict):
        result["required_target_source"] = resolver_debug.get(
            "required_target_source"
        )
        result["recovered_target_ids"] = resolver_debug.get(
            "recovered_target_ids"
        )
    if success is None:
        result["error"] = reason
    return result
